"""LoRA candidate generation: small temporary adapters over varied
rank / lr / step / depth budgets."""
from typing import List

import numpy as np

from interventions.lora import LoRAIntervention

# Calibrated: lr 3e-4 / 16 steps is the sweet spot (+0.087); driving training
# loss to 0.000 (lr>=1e-3 with many steps) overfits the failure batch and costs
# accuracy (-0.112). The grid spans both regimes on purpose.
RANK_GRID = [4, 8]
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
STEPS_GRID = [6, 12, 20]
DEPTH_GRID = [4, 8, 12]
PROJ_GRID = [("q_proj", "v_proj"), ("q_proj", "k_proj", "v_proj")]


class LoRAGenerator:
    def __init__(self, system, task):
        self.system = system
        self.task = task

    def propose(self, failure_examples, rng: np.random.RandomState, n: int = 3
                ) -> List[LoRAIntervention]:
        cands = []
        for _ in range(n):
            cands.append(LoRAIntervention.propose(
                rank=int(RANK_GRID[rng.randint(len(RANK_GRID))]),
                lr=float(LR_GRID[rng.randint(len(LR_GRID))]),
                steps=int(STEPS_GRID[rng.randint(len(STEPS_GRID))]),
                n_top_layers=int(DEPTH_GRID[rng.randint(len(DEPTH_GRID))]),
                projections=PROJ_GRID[rng.randint(len(PROJ_GRID))],
                failure_examples=failure_examples,
                task=self.task,
            ))
        return cands
