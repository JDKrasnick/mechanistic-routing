"""Parameter-level intervention: a temporary LoRA update fitted on the failure batch.

Weights are never merged into the base model. apply() attaches wrapper modules,
revert() restores the original module references, so the base model is bitwise
unchanged across apply/revert cycles. Trained weights are cached on first apply
so restoring a state never retrains.
"""
from typing import Dict, List, Tuple

import torch

import config
from interventions.base import Intervention


class LoRAIntervention(Intervention):
    kind = "lora"

    def __init__(self, rank: int, lr: float, steps: int, n_top_layers: int,
                 projections: Tuple[str, ...], failure_examples: List, task):
        super().__init__()
        self.rank = rank
        self.lr = lr
        self.steps = steps
        self.n_top_layers = n_top_layers
        self.projections = projections
        self.failure_examples = failure_examples
        self.task = task
        self._weights = None
        self._final_loss = float("nan")
        self._mods = None

    @classmethod
    def propose(cls, rank: int, lr: float, steps: int, n_top_layers: int,
                projections: Tuple[str, ...], failure_examples: List, task):
        return cls(rank, lr, steps, n_top_layers, projections, failure_examples, task)

    def apply(self, system):
        mods = system.attach_lora(self.uid, self.n_top_layers, self.projections,
                                  r=self.rank, alpha=2.0 * self.rank)
        self._mods = mods
        if self._weights is None:
            prompts = [self.task.render(e.text, system.instruction)
                       for e in self.failure_examples]
            label_ids = system.label_token_ids(self.task.labels)
            gold = [label_ids[e.label] for e in self.failure_examples]
            self._final_loss = system.train_lora(mods, prompts, gold,
                                                 steps=self.steps, lr=self.lr)
            self._weights = [(m.A.detach().clone(), m.B.detach().clone()) for m in mods]
        else:
            with torch.no_grad():
                for m, (A, B) in zip(mods, self._weights):
                    m.A.copy_(A)
                    m.B.copy_(B)

    def revert(self, system):
        system.detach_lora(self.uid)
        self._mods = None

    def cost(self) -> float:
        # Scales with the dominant term: gradient steps over the failure batch.
        return config.COST_LORA * (self.steps / 12.0) * (1.0 + 0.25 * (self.rank / 8.0))

    def spec(self) -> Dict:
        return {"rank": self.rank, "lr": self.lr, "steps": self.steps,
                "n_top_layers": self.n_top_layers,
                "projections": list(self.projections),
                "n_failure": len(self.failure_examples),
                "final_train_loss": self._final_loss}
