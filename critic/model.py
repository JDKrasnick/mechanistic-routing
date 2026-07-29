"""Counterfactual critic: predicts held-out marginal utility of a candidate
from (state, candidate) features alone, without executing it.

Two heads, per the project plan:
  regression -- absolute predicted utility, needed for budget arithmetic and
                for deciding whether any candidate is worth applying at all.
  pairwise   -- ranking over candidates within the same state, which is what
                selection actually consumes and is invariant to per-state shifts.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class CriticPrediction:
    utility: np.ndarray
    rank_score: np.ndarray


class CandidateCritic:
    def __init__(self, mode: str = "gbm", seed: int = 0):
        self.mode = mode
        self.seed = seed
        self.scaler = StandardScaler()
        self.reg = None
        self.ranker: Optional[LogisticRegression] = None
        self.uncertainty_reg = None

    def _make_reg(self):
        if self.mode == "ridge":
            return Ridge(alpha=1.0)
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.9, random_state=self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.reg = self._make_reg()
        self.reg.fit(Xs, y)

        # Pairwise ranking on within-state candidate pairs.
        dX, dy = [], []
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            for i in idx:
                for j in idx:
                    if i >= j or y[i] == y[j]:
                        continue
                    dX.append(Xs[i] - Xs[j]); dy.append(int(y[i] > y[j]))
                    dX.append(Xs[j] - Xs[i]); dy.append(int(y[j] > y[i]))
        if dX and len(set(dy)) > 1:
            self.ranker = LogisticRegression(max_iter=2000, C=0.5)
            self.ranker.fit(np.array(dX), np.array(dy))

        # Uncertainty head: predict squared residual of the regression head.
        resid = (y - self.reg.predict(Xs)) ** 2
        self.uncertainty_reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=self.seed)
        self.uncertainty_reg.fit(Xs, resid)
        return self

    def predict(self, X: np.ndarray) -> CriticPrediction:
        Xs = self.scaler.transform(X)
        util = self.reg.predict(Xs)
        if self.ranker is not None:
            rank = Xs @ self.ranker.coef_.ravel()
        else:
            rank = util
        return CriticPrediction(utility=util, rank_score=rank)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.sqrt(np.clip(self.uncertainty_reg.predict(Xs), 1e-9, None))


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      groups: np.ndarray) -> float:
    correct = total = 0
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for i in idx:
            for j in idx:
                if i >= j or y_true[i] == y_true[j]:
                    continue
                total += 1
                correct += int((y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]))
    return correct / total if total else float("nan")


def top1_regret(y_true: np.ndarray, y_pred: np.ndarray,
                groups: np.ndarray) -> float:
    """Mean utility lost by trusting the critic's top pick over the oracle's."""
    regrets = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        regrets.append(y_true[idx].max() - y_true[idx[np.argmax(y_pred[idx])]])
    return float(np.mean(regrets))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0
