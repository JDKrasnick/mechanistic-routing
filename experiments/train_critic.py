"""Train the counterfactual critic on collected data and report held-out skill."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config
from counterfactuals.dataset import InterventionDataset
from critic import train as ct

OUT_METRICS = config.RESULTS / "critic_metrics.json"
OUT_MODEL = config.DATA / "critic.pkl"


def main():
    ds = InterventionDataset(config.DATA / "intervention_dataset.jsonl")
    records = list(ds)
    print(f"{len(records)} records / {len({r['state_id'] for r in records})} states")

    kinds = {}
    for r in records:
        kinds.setdefault(r["kind"], []).append(r["utility"])
    print("\nmeasured utility by intervention type:")
    for k, v in kinds.items():
        v = np.array(v)
        print(f"  {k:11s} n={len(v):4d} mean={v.mean():+.4f} sd={v.std():.4f} "
              f"max={v.max():+.3f} P(util>0)={float((v > 0).mean()):.2f}")

    # How often does each type win its state? If one type always wins, routing
    # is unnecessary and the whole premise collapses.
    wins = {}
    for sid in {r["state_id"] for r in records}:
        rs = [r for r in records if r["state_id"] == sid]
        best = max(rs, key=lambda r: r["utility"])
        wins[best["kind"]] = wins.get(best["kind"], 0) + 1
    print("\noracle winner counts by type:", wins)

    all_metrics = {"winner_counts": wins,
                   "utility_by_kind": {k: {"mean": float(np.mean(v)),
                                           "sd": float(np.std(v)),
                                           "n": len(v)} for k, v in kinds.items()}}

    for mode in ("ridge", "gbm"):
        m = ct.cross_validate(records, mode=mode)
        all_metrics[mode] = m
        print(f"\n=== critic ({mode}) grouped 5-fold CV ===")
        for name in ("critic", "type_prior", "random"):
            b = m[name]
            print(f"  {name:11s} spearman={b['spearman']:+.3f} "
                  f"pairwise={b['pairwise_acc']:.3f} "
                  f"top1_regret={b['top1_regret']:.4f} rmse={b['rmse']:.4f}")

    critic = ct.fit_final(records, mode="gbm")
    ct.save(critic, OUT_MODEL)
    imp = ct.feature_importance(critic)
    all_metrics["feature_importance"] = imp
    print("\ntop features:")
    for name, v in imp[:12]:
        print(f"  {name:26s} {v:.4f}")

    with open(OUT_METRICS, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nsaved critic -> {OUT_MODEL}\nmetrics -> {OUT_METRICS}")


if __name__ == "__main__":
    main()
