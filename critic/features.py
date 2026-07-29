"""Feature construction for the counterfactual critic.

Hard constraint: every feature must be computable BEFORE the candidate is
executed. Quantities that only exist after apply() -- fitted steering-vector
norms, final LoRA training loss, any post_* measurement -- are recorded in the
dataset for analysis but are forbidden here, since using them would leak the
outcome the critic is supposed to predict.
"""
from typing import Dict, List

import numpy as np

FORBIDDEN = {"fit_norm", "act_norm", "rel_strength", "final_train_loss"}

PROMPT_SOURCES = ["llm_rewrite", "confusion_edit", "strategy_hint"]

STATE_KEYS = [
    "base_task_acc", "base_retention_acc", "base_gold_logprob", "base_margin",
    "base_entropy", "depth", "depth_prompt", "depth_activation", "depth_lora",
    "instruction_len", "failure_rate", "n_failures", "failure_entropy",
    "confusion_concentration",
]

CAND_KEYS = [
    "is_prompt", "is_activation", "is_lora", "cost",
    "p_chars_added", "p_instr_len", "p_src_llm", "p_src_confusion", "p_src_hint",
    "a_layer_frac", "a_strength", "a_log_strength",
    "l_rank", "l_log_lr", "l_steps", "l_depth", "l_n_proj", "l_n_failure",
]

FEATURE_NAMES = STATE_KEYS + CAND_KEYS + [
    "lora_x_lowacc", "act_x_depth", "prompt_x_confusion", "cost_x_depth"]


def _entropy(logits: np.ndarray) -> float:
    m = logits.max(1, keepdims=True)
    lp = logits - (m + np.log(np.exp(logits - m).sum(1, keepdims=True)))
    p = np.exp(lp)
    return float((-(p * lp).sum(1)).mean())


def compute_state_features(base_task, base_retention, system, failures,
                           pool_logits: np.ndarray, task) -> Dict:
    kinds = [iv.kind for iv in system._applied]
    gold_pred = [(e.label, p) for e, p in failures]
    if gold_pred:
        pairs = {}
        for g, p in gold_pred:
            pairs[(g, p)] = pairs.get((g, p), 0) + 1
        concentration = max(pairs.values()) / len(gold_pred)
    else:
        concentration = 0.0

    n_pool = pool_logits.shape[0]
    return {
        "base_task_acc": base_task.accuracy,
        "base_retention_acc": base_retention.accuracy,
        "base_gold_logprob": base_task.gold_logprob,
        "base_margin": base_task.margin,
        "base_entropy": base_task.entropy,
        "depth": len(kinds),
        "depth_prompt": kinds.count("prompt"),
        "depth_activation": kinds.count("activation"),
        "depth_lora": kinds.count("lora"),
        "instruction_len": len(system.instruction),
        "failure_rate": len(failures) / max(n_pool, 1),
        "n_failures": len(failures),
        "failure_entropy": _entropy(pool_logits),
        "confusion_concentration": concentration,
    }


def candidate_features(cand: Dict) -> Dict:
    kind = cand["kind"]
    src = cand.get("source", "")
    proj = cand.get("projections", []) or []
    f = {
        "is_prompt": float(kind == "prompt"),
        "is_activation": float(kind == "activation"),
        "is_lora": float(kind == "lora"),
        "cost": float(cand["cost"]),
        "p_chars_added": float(cand.get("n_chars_added", 0)),
        "p_instr_len": float(cand.get("instruction_len", 0)),
        "p_src_llm": float(src == "llm_rewrite"),
        "p_src_confusion": float(src == "confusion_edit"),
        "p_src_hint": float(src == "strategy_hint"),
        "a_layer_frac": float(cand.get("layer_frac", 0.0)),
        "a_strength": float(cand.get("strength", 0.0)),
        "a_log_strength": float(np.log1p(cand.get("strength", 0.0))),
        "l_rank": float(cand.get("rank", 0)),
        "l_log_lr": float(np.log10(cand["lr"])) if cand.get("lr") else 0.0,
        "l_steps": float(cand.get("steps", 0)),
        "l_depth": float(cand.get("n_top_layers", 0)),
        "l_n_proj": float(len(proj)),
        "l_n_failure": float(cand.get("n_failure", 0)),
    }
    return f


def build_row(state_feats: Dict, cand: Dict) -> np.ndarray:
    # candidate_features reads an explicit allowlist, so post-execution fields
    # present on the record can never reach the model.
    assert not (FORBIDDEN & set(CAND_KEYS)), "allowlist contains a leaked field"
    cf = candidate_features(cand)
    row = [state_feats[k] for k in STATE_KEYS] + [cf[k] for k in CAND_KEYS]
    # Explicit interactions: which type wins is state-dependent, and linear
    # models cannot express that on their own.
    row += [
        cf["is_lora"] * (1.0 - state_feats["base_task_acc"]),
        cf["is_activation"] * state_feats["depth"],
        cf["is_prompt"] * state_feats["confusion_concentration"],
        cf["cost"] * state_feats["depth"],
    ]
    return np.array(row, dtype=np.float64)


def build_matrix(records: List[Dict]) -> np.ndarray:
    return np.stack([build_row(r["state_features"], r["candidate"]) for r in records])
