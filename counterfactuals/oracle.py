"""Exhaustive oracle: the best candidate under budget, given measured outcomes.

This is the critic's training target and its performance upper bound. It is only
computable by executing every candidate, which is exactly the cost the critic
is meant to avoid.
"""
from typing import Dict, List, Optional


def oracle_select(records: List[Dict], budget: float) -> Optional[Dict]:
    """Highest measured utility among candidates affordable within budget."""
    affordable = [r for r in records if r["cost"] <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda r: r["utility"])


def oracle_select_net(records: List[Dict], budget: float) -> Optional[Dict]:
    """Cost-adjusted variant: maximizes utility net of normalized spend."""
    affordable = [r for r in records if r["cost"] <= budget]
    if not affordable:
        return None
    return max(affordable, key=lambda r: r["net_value"])


def oracle_summary(records: List[Dict], budget: float) -> Dict:
    best = oracle_select(records, budget)
    by_kind = {}
    for r in records:
        by_kind.setdefault(r["kind"], []).append(r["utility"])
    return {
        "best_kind": best["kind"] if best else None,
        "best_utility": best["utility"] if best else 0.0,
        "best_cost": best["cost"] if best else 0.0,
        "mean_utility_by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
        "max_utility_by_kind": {k: max(v) for k, v in by_kind.items()},
        "exhaustive_cost": sum(r["cost"] for r in records),
    }
