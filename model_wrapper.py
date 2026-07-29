"""Single interface for inference, loss, hidden-state capture, activation hooks,
adapter loading, and reliable state capture/restore."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


class LoRALinear(nn.Module):
    """Hand-rolled LoRA so that attach/detach is exact module swapping.

    Nothing is ever merged into base weights, so revert restores the original
    module by reference -- no numerical drift across apply/revert cycles.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        # Mirror the wrapped module's shape so a LoRALinear can itself be
        # wrapped. Stacking adapters is a real composition case -- a LoRA
        # candidate proposed for a state that already has one -- and it is where
        # cross-intervention interaction effects show up.
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        out = self.base(x)
        delta = (x.to(torch.float32) @ self.A.T) @ self.B.T
        return out + (delta * self.scale).to(out.dtype)


@dataclass(frozen=True)
class SystemState:
    instruction: str
    stack: Tuple  # tuple of applied Intervention objects, in application order


class SystemModel:
    def __init__(self, model_name: str = config.MODEL_NAME):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # eager attention: SDPA's cached-decode path miscompiles on MPS with
        # torch 2.5 (mps_matmul shape error). Measured identical prefill speed.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=config.DTYPE,
            attn_implementation=config.ATTN_IMPL).to(config.DEVICE)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.hidden_size = self.model.config.hidden_size

        self.base_instruction: Optional[str] = None
        self.instruction: Optional[str] = None
        self._applied: List = []
        self._steer_hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        self._steer_vecs: Dict[int, torch.Tensor] = {}
        self._steer_terms: Dict[int, List] = {}
        self._lora_modules: Dict[str, LoRALinear] = {}

    # ---------------- prompt state ----------------

    def set_base_instruction(self, instruction: str):
        self.base_instruction = instruction
        self.instruction = instruction

    # ---------------- tokenization / scoring ----------------

    def label_token_ids(self, labels: List[str]) -> List[int]:
        ids = [self.tok.encode(" " + l, add_special_tokens=False)[0] for l in labels]
        assert len(set(ids)) == len(ids), f"label first-tokens collide: {labels}"
        return ids

    @torch.no_grad()
    def label_logits(self, prompts: List[str], label_ids: List[int],
                     batch_size: int = config.FORWARD_BATCH) -> np.ndarray:
        """One forward pass per example; returns [N, n_labels] logits."""
        out = []
        for i in range(0, len(prompts), batch_size):
            b = self.tok(prompts[i:i + batch_size], return_tensors="pt",
                         padding=True).to(config.DEVICE)
            logits = self.model(**b).logits[:, -1, :].float()
            out.append(logits[:, label_ids].cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def hidden_states(self, prompts: List[str], layer: int,
                      batch_size: int = config.FORWARD_BATCH) -> torch.Tensor:
        """Residual-stream activations at the final token of `layer`."""
        grabbed = []

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            grabbed.append(h[:, -1, :].detach().float().cpu())

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            for i in range(0, len(prompts), batch_size):
                b = self.tok(prompts[i:i + batch_size], return_tensors="pt",
                             padding=True).to(config.DEVICE)
                self.model(**b)
        finally:
            handle.remove()
        return torch.cat(grabbed, dim=0)

    # ---------------- activation steering ----------------

    def add_steering(self, key: str, layer: int, vector: torch.Tensor,
                     strength: float):
        """Steering terms are tracked by identity, never by numeric cancellation.

        Two interventions can target the same layer; subtracting one back out in
        bfloat16 does not return exactly zero, so an "is the sum zero yet?" test
        silently leaves the hook attached and leaks steering into later trials.
        """
        vec = (vector.to(config.DEVICE, config.DTYPE) * strength)
        self._steer_terms.setdefault(layer, []).append((key, vec))
        self._recompute_steering(layer)

        if layer not in self._steer_hooks:
            def hook(_m, _i, out, _layer=layer):
                v = self._steer_vecs[_layer]
                if isinstance(out, tuple):
                    return (out[0] + v,) + out[1:]
                return out + v

            self._steer_hooks[layer] = self.layers[layer].register_forward_hook(hook)

    def _recompute_steering(self, layer: int):
        terms = self._steer_terms[layer]
        total = terms[0][1]
        for _, v in terms[1:]:
            total = total + v
        self._steer_vecs[layer] = total

    def remove_steering(self, key: str, layer: int):
        terms = self._steer_terms.get(layer, [])
        for i in range(len(terms) - 1, -1, -1):
            if terms[i][0] == key:
                terms.pop(i)
                break
        else:
            raise KeyError(f"steering term {key} not found at layer {layer}")
        if terms:
            self._recompute_steering(layer)
        else:
            self._steer_terms.pop(layer)
            self._steer_vecs.pop(layer)
            self._steer_hooks.pop(layer).remove()

    # ---------------- LoRA adapters ----------------

    def _target_modules(self, n_top_layers: int, projections: Tuple[str, ...]):
        targets = []
        for li in range(self.n_layers - n_top_layers, self.n_layers):
            for proj in projections:
                parent = self.layers[li].self_attn
                targets.append((f"layer{li}.{proj}", parent, proj))
        return targets

    def attach_lora(self, key: str, n_top_layers: int, projections: Tuple[str, ...],
                    r: int, alpha: float) -> List[LoRALinear]:
        mods = []
        for name, parent, attr in self._target_modules(n_top_layers, projections):
            base = getattr(parent, attr)
            lora = LoRALinear(base, r, alpha).to(config.DEVICE)
            setattr(parent, attr, lora)
            self._lora_modules[f"{key}::{name}"] = (parent, attr, base)
            mods.append(lora)
        return mods

    def detach_lora(self, key: str):
        for k in [k for k in self._lora_modules if k.startswith(f"{key}::")]:
            parent, attr, base = self._lora_modules.pop(k)
            setattr(parent, attr, base)

    def train_lora(self, mods: List[LoRALinear], prompts: List[str],
                   gold_token_ids: List[int], steps: int, lr: float,
                   batch_size: int = 4) -> float:
        params = [p for m in mods for p in (m.A, m.B)]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr)
        rng = np.random.RandomState(config.SEED)
        n = len(prompts)
        last = float("nan")
        for _ in range(steps):
            idx = rng.choice(n, size=min(batch_size, n), replace=False)
            b = self.tok([prompts[i] for i in idx], return_tensors="pt",
                         padding=True).to(config.DEVICE)
            logits = self.model(**b).logits[:, -1, :].float()
            gold = torch.tensor([gold_token_ids[i] for i in idx], device=logits.device)
            loss = F.cross_entropy(logits, gold)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = loss.item()
        for p in params:
            p.requires_grad_(False)
        return last

    # ---------------- generation (for prompt rewriting) ----------------

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 96,
                 temperature: float = 0.8, seed: int = 0) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        b = self.tok([text], return_tensors="pt").to(config.DEVICE)
        torch.manual_seed(seed)
        out = self.model.generate(**b, max_new_tokens=max_new_tokens, do_sample=True,
                                  temperature=temperature, top_p=0.95,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, b["input_ids"].shape[1]:], skip_special_tokens=True)

    # ---------------- state capture / restore ----------------

    def apply_intervention(self, intervention):
        intervention.apply(self)
        self._applied.append(intervention)

    def capture_state(self) -> SystemState:
        return SystemState(instruction=self.instruction, stack=tuple(self._applied))

    def revert_all(self):
        for iv in reversed(self._applied):
            iv.revert(self)
        self._applied = []
        self.instruction = self.base_instruction
        assert not self._steer_hooks, f"steering hooks leaked: {list(self._steer_hooks)}"
        assert not self._lora_modules, f"lora modules leaked: {list(self._lora_modules)}"

    def restore_state(self, state: SystemState):
        self.revert_all()
        for iv in state.stack:
            iv.apply(self)
            self._applied.append(iv)
        self.instruction = state.instruction
