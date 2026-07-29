"""Train and cross-validate the counterfactual critic."""
import json
import pickle
from typing import Dict, List

import numpy as np
from sklearn.model_selection import GroupKFold

import config
from critic.features import FEATURE_NAMES, build_matrix
from critic.model import (CandidateCritic, pairwise_accuracy, spearman,
                          top1_regret)


def prepare(records: List[Dict], target: str = "utility"):
    X = build_matrix(records)
    y = np.array([r[target] for r in records], dtype=np.float64)
    groups = np.array([r["state_id"] for r in records])
    return X, y, groups


def cross_validate(records: List[Dict], mode: str = "gbm", n_splits: int = 5,
                   target: str = "utility") -> Dict:
    X, y, groups = prepare(records, target)
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        c = CandidateCritic(mode=mode).fit(X[tr], y[tr], groups[tr])
        oof[te] = c.predict(X[te]).utility

    # Baselines the critic must beat to be worth anything.
    kinds = np.array([r["kind"] for r in records])
    kind_mean = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        for k in np.unique(kinds):
            m = y[tr][kinds[tr] == k]
            kind_mean[te[kinds[te] == k]] = m.mean() if len(m) else 0.0
    rng = np.random.RandomState(0)
    rand = rng.randn(len(y))

    def block(pred):
        return {
            "spearman": spearman(y, pred),
            "pairwise_acc": pairwise_accuracy(y, pred, groups),
            "top1_regret": top1_regret(y, pred, groups),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        }

    return {
        "n_records": len(records), "n_states": int(n_groups), "target": target,
        "critic": block(oof),
        "type_prior": block(kind_mean),
        "random": block(rand),
        "oracle_top1_regret": 0.0,
        "oof": oof.tolist(),
        "y": y.tolist(),
        "groups": groups.tolist(),
        "kinds": kinds.tolist(),
    }


def fit_final(records: List[Dict], mode: str = "gbm", target: str = "utility"):
    X, y, groups = prepare(records, target)
    return CandidateCritic(mode=mode).fit(X, y, groups)


def feature_importance(critic: CandidateCritic) -> List:
    imp = getattr(critic.reg, "feature_importances_", None)
    if imp is None:
        imp = np.abs(critic.reg.coef_)
    order = np.argsort(imp)[::-1]
    return [(FEATURE_NAMES[i], float(imp[i])) for i in order]


def save(critic, path):
    with open(path, "wb") as f:
        pickle.dump(critic, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
