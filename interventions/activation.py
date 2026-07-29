"""Activation-level intervention: contrastive steering vector via a forward hook.

The vector is fitted at apply() time as a difference-in-means between the
residual stream on gold-labelled and mispredicted-labelled continuations of the
failure batch (CAA-style), then cached so re-application is free.
"""
from typing import Dict, List

import torch

import config
from interventions.base import Intervention


class ActivationIntervention(Intervention):
    kind = "activation"

    def __init__(self, layer: int, strength: float, contrast_prompts_gold: List[str],
                 contrast_prompts_wrong: List[str]):
        super().__init__()
        self.layer = layer
        self.strength = strength
        self._gold = contrast_prompts_gold
        self._wrong = contrast_prompts_wrong
        self._vector = None

    @classmethod
    def propose(cls, layer: int, strength: float, contrast_gold: List[str],
                contrast_wrong: List[str]):
        return cls(layer, strength, contrast_gold, contrast_wrong)

    def _fit(self, system):
        h_gold = system.hidden_states(self._gold, self.layer)
        h_wrong = system.hidden_states(self._wrong, self.layer)
        # Keep the raw difference-in-means. Unit-normalizing destroys the scale
        # relative to the residual stream, which made steering a no-op.
        v = (h_gold.mean(0) - h_wrong.mean(0))
        self._vector = v
        self._fit_norm = float(v.norm())
        self._act_norm = float(h_gold.norm(dim=-1).mean())

    def apply(self, system):
        if self._vector is None:
            self._fit(system)
        system.add_steering(self.uid, self.layer, self._vector, self.strength)

    def revert(self, system):
        system.remove_steering(self.uid, self.layer)

    def cost(self) -> float:
        return config.COST_ACTIVATION

    def spec(self) -> Dict:
        return {"layer": self.layer, "strength": self.strength,
                "layer_frac": self.layer / 36.0,
                "fit_norm": getattr(self, "_fit_norm", 0.0),
                "act_norm": getattr(self, "_act_norm", 0.0),
                "rel_strength": self.strength * getattr(self, "_fit_norm", 0.0)
                                / max(getattr(self, "_act_norm", 1.0), 1e-6),
                "n_contrast": len(self._gold)}
