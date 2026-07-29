"""Budgeted selection policies over proposed candidates.

Every policy sees the same cheap proposals and must commit to candidates without
measuring them -- except the oracle, which is allowed to execute all of them and
therefore defines both the target and the upper bound.
"""
from typing import Dict, List, Optional

import numpy as np

from critic.features import build_row


class SelectionPolicy:
    name = "base"

    def select(self, state_feats: Dict, candidates: List, budget: float, rng):
        raise NotImplementedError


class RandomPolicy(SelectionPolicy):
    name = "random"

    def select(self, state_feats, candidates, budget, rng):
        aff = [c for c in candidates if c.cost() <= budget]
        return aff[rng.randint(len(aff))] if aff else None


class SingleTypePolicy(SelectionPolicy):
    """prompt-only / activation-only / lora-only ablations."""

    def __init__(self, kind: str):
        self.kind = kind
        self.name = f"{kind}_only"

    def select(self, state_feats, candidates, budget, rng):
        aff = [c for c in candidates if c.kind == self.kind and c.cost() <= budget]
        return aff[rng.randint(len(aff))] if aff else None


class HeuristicPolicy(SelectionPolicy):
    """Static prior: always take the type with the best average training utility,
    breaking ties by lowest cost. Isolates the value of *state-dependent*
    routing, since this captures everything a fixed preference order can."""
    name = "heuristic"

    def __init__(self, kind_ranking: List[str]):
        self.kind_ranking = kind_ranking

    def select(self, state_feats, candidates, budget, rng):
        for kind in self.kind_ranking:
            aff = [c for c in candidates if c.kind == kind and c.cost() <= budget]
            if aff:
                return min(aff, key=lambda c: c.cost())
        aff = [c for c in candidates if c.cost() <= budget]
        return aff[rng.randint(len(aff))] if aff else None


class CriticPolicy(SelectionPolicy):
    """Rank by predicted utility minus normalized cost; apply the best affordable."""
    name = "critic"

    def __init__(self, critic, cost_weight: float = 0.002,
                 risk_aversion: float = 0.0, allow_decline: bool = True):
        self.critic = critic
        self.cost_weight = cost_weight
        self.risk_aversion = risk_aversion
        # A calibrated regression head can say "nothing here is worth applying".
        # Random/heuristic/single-type policies have no utility estimate and so
        # cannot decline; this asymmetry is a property of the critic, and is
        # reported as such.
        self.allow_decline = allow_decline

    def score(self, state_feats, candidates):
        """Budget arithmetic needs a calibrated scale, so selection uses the
        regression head; the pairwise head is scale-free and is used for the
        reported ranking metrics."""
        X = np.stack([build_row(state_feats, c.serialize()) for c in candidates])
        pred = self.critic.predict(X)
        score = pred.utility - self.cost_weight * np.array([c.cost() for c in candidates])
        if self.risk_aversion:
            score = score - self.risk_aversion * self.critic.predict_std(X)
        return score, pred.utility, pred.rank_score

    def select(self, state_feats, candidates, budget, rng):
        aff_idx = [i for i, c in enumerate(candidates) if c.cost() <= budget]
        if not aff_idx:
            return None
        score, util, _ = self.score(state_feats, candidates)
        best = max(aff_idx, key=lambda i: score[i])
        if self.allow_decline and util[best] <= 0:
            return None
        return candidates[best]


class OraclePolicy(SelectionPolicy):
    """Executes every candidate and keeps the measured best. Upper bound."""
    name = "oracle"
