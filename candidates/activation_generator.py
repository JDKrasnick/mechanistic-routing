"""Activation candidate generation: contrastive steering vectors at varied
layers and strengths."""
from typing import List

import numpy as np

from interventions.activation import ActivationIntervention

# Calibrated on the raw difference-in-means scale (experiments/calibrate.py).
# Steering helps up to ~1.0 and degrades sharply beyond it, and lower layers are
# more effective than late ones. The grid deliberately spans both the helpful and
# the harmful range: avoiding a damaging candidate is part of the routing task.
LAYER_GRID = [10, 14, 18, 22, 26, 30]
STRENGTH_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


class ActivationGenerator:
    def __init__(self, system, task):
        self.system = system
        self.task = task

    def _contrast_sets(self, failures):
        gold, wrong = [], []
        for e, pred in failures:
            base = self.task.render(e.text, self.system.instruction)
            gold.append(f"{base} {self.task.labels[e.label]}")
            wrong.append(f"{base} {self.task.labels[pred]}")
        return gold, wrong

    def propose(self, failures, rng: np.random.RandomState, n: int = 3
                ) -> List[ActivationIntervention]:
        gold, wrong = self._contrast_sets(failures)
        layers = rng.choice(LAYER_GRID, size=n, replace=False)
        cands = []
        for layer in layers:
            strength = float(STRENGTH_GRID[int(rng.randint(len(STRENGTH_GRID)))])
            cands.append(ActivationIntervention.propose(int(layer), strength, gold, wrong))
        return cands
