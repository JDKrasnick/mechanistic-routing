"""Final result figures."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from counterfactuals.dataset import InterventionDataset

KIND_COLOR = {"prompt": "#4C72B0", "activation": "#DD8452", "lora": "#55A868"}
POLICY_ORDER = ["prompt_only", "activation_only", "lora_only", "random",
                "heuristic", "critic", "oracle"]
POLICY_COLOR = {"prompt_only": "#4C72B0", "activation_only": "#DD8452",
                "lora_only": "#55A868", "random": "#8C8C8C",
                "heuristic": "#C44E52", "critic": "#8172B3", "oracle": "#000000"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def load_all():
    ds = InterventionDataset(config.DATA / "intervention_dataset.jsonl")
    records = list(ds)
    metrics = json.load(open(config.RESULTS / "critic_metrics.json"))
    rpath = config.RESULTS / "routing_results.json"
    routing = json.load(open(rpath)) if rpath.exists() else []
    return records, metrics, routing


def panel_utility_by_kind(ax, records):
    kinds = ["prompt", "activation", "lora"]
    data = [[r["utility"] for r in records if r["kind"] == k] for k in kinds]
    bp = ax.boxplot(data, tick_labels=kinds, patch_artist=True, widths=0.55,
                    showfliers=False, medianprops=dict(color="black"))
    for patch, k in zip(bp["boxes"], kinds):
        patch.set_facecolor(KIND_COLOR[k]); patch.set_alpha(0.65)
    rng = np.random.RandomState(0)
    for i, d in enumerate(data):
        ax.scatter(rng.normal(i + 1, 0.07, len(d)), d, s=5, alpha=0.35,
                   color=KIND_COLOR[kinds[i]], zorder=3, linewidths=0)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("measured held-out utility")
    ax.set_title("A  Counterfactual utility by intervention type", loc="left",
                 fontweight="bold")


def panel_winners(ax, records):
    states = sorted({r["state_id"] for r in records})
    wins = {}
    for sid in states:
        rs = [r for r in records if r["state_id"] == sid]
        best = max(rs, key=lambda r: r["utility"])
        wins[best["kind"]] = wins.get(best["kind"], 0) + 1
    kinds = ["prompt", "activation", "lora"]
    vals = [100 * wins.get(k, 0) / len(states) for k in kinds]
    ax.bar(kinds, vals, color=[KIND_COLOR[k] for k in kinds], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("% of states where type wins")
    ax.set_ylim(0, max(vals) * 1.25 + 5)
    ax.set_title("B  No single type dominates", loc="left", fontweight="bold")


def panel_critic_scatter(ax, metrics):
    m = metrics["gbm"]
    y = np.array(m["y"]); oof = np.array(m["oof"]); kinds = np.array(m["kinds"])
    for k in ["prompt", "activation", "lora"]:
        s = kinds == k
        ax.scatter(oof[s], y[s], s=10, alpha=0.55, color=KIND_COLOR[k], label=k,
                   linewidths=0)
    lo = min(y.min(), oof.min()); hi = max(y.max(), oof.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("critic predicted utility (out-of-fold)")
    ax.set_ylabel("measured utility")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"C  Critic prediction  (Spearman {m['critic']['spearman']:+.2f})",
                 loc="left", fontweight="bold")


def panel_critic_skill(ax, metrics):
    m = metrics["gbm"]
    names = ["random", "type_prior", "critic"]
    labels = ["random", "type prior", "critic"]
    pw = [m[n]["pairwise_acc"] for n in names]
    x = np.arange(len(names))
    ax.bar(x, pw, color=["#8C8C8C", "#C44E52", "#8172B3"], alpha=0.85, width=0.6)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    for i, v in enumerate(pw):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("within-state pairwise ranking acc.")
    ax.set_ylim(0.35, max(pw) * 1.12)
    ax.set_title("D  Ranking skill vs baselines", loc="left", fontweight="bold")


def _agg(routing, key):
    out = {}
    for p in POLICY_ORDER:
        v = [r[key] for r in routing if r["policy"] == p]
        if v:
            out[p] = (float(np.mean(v)), float(np.std(v) / max(np.sqrt(len(v)), 1)))
    return out


def panel_routing(ax, routing):
    agg = _agg(routing, "task_gain")
    ps = [p for p in POLICY_ORDER if p in agg]
    means = [agg[p][0] for p in ps]; errs = [agg[p][1] for p in ps]
    ax.bar(range(len(ps)), means, yerr=errs, capsize=3,
           color=[POLICY_COLOR[p] for p in ps], alpha=0.85, width=0.65)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([p.replace("_", "\n") for p in ps], fontsize=7.5)
    ax.set_ylabel("held-out task accuracy gain")
    ax.set_title("E  Routing policies under matched budget", loc="left",
                 fontweight="bold")


def panel_efficiency(ax, routing):
    gain = _agg(routing, "task_gain")
    cost = _agg(routing, "exploration_cost")
    for p in POLICY_ORDER:
        if p not in gain:
            continue
        ax.errorbar(cost[p][0], gain[p][0], yerr=gain[p][1], fmt="o", ms=7,
                    color=POLICY_COLOR[p], capsize=2)
        ax.annotate(p, (cost[p][0], gain[p][0]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("exploration cost (candidates executed, normalized units)")
    ax.set_ylabel("held-out task accuracy gain")
    ax.set_title("F  Utility vs execution cost", loc="left", fontweight="bold")


def panel_retention(ax, routing):
    gain = _agg(routing, "task_gain")
    ret = _agg(routing, "final_retention")
    for p in POLICY_ORDER:
        if p not in gain:
            continue
        ax.errorbar(gain[p][0], ret[p][0], xerr=gain[p][1], yerr=ret[p][1],
                    fmt="o", ms=7, color=POLICY_COLOR[p], capsize=2)
        ax.annotate(p, (gain[p][0], ret[p][0]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("task accuracy gain")
    ax.set_ylabel("final retention accuracy")
    ax.set_title("G  Task/retention trade-off", loc="left", fontweight="bold")


def panel_type_usage(ax, routing):
    ps = [p for p in ["random", "heuristic", "critic", "oracle"]
          if any(r["policy"] == p for r in routing)]
    kinds = ["prompt", "activation", "lora"]
    bottom = np.zeros(len(ps))
    for k in kinds:
        vals = []
        for p in ps:
            used = [x for r in routing if r["policy"] == p for x in r["kinds_used"]]
            vals.append(100 * used.count(k) / max(len(used), 1))
        ax.bar(ps, vals, bottom=bottom, label=k, color=KIND_COLOR[k], alpha=0.85)
        bottom += np.array(vals)
    ax.set_ylabel("% of applied interventions")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title("H  Which types each policy chooses", loc="left", fontweight="bold")


def panel_regret(ax, metrics):
    m = metrics["gbm"]
    names = ["random", "type_prior", "critic"]
    labels = ["random", "type prior", "critic", "oracle"]
    vals = [m[n]["top1_regret"] for n in names] + [0.0]
    ax.bar(range(4), vals, color=["#8C8C8C", "#C44E52", "#8172B3", "#000000"],
           alpha=0.85, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("top-1 utility regret vs oracle")
    ax.set_title("I  Selection regret", loc="left", fontweight="bold")


def main():
    records, metrics, routing = load_all()
    has_routing = bool(routing)
    n = 9 if has_routing else 5
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    ax = axes.ravel()

    panel_utility_by_kind(ax[0], records)
    panel_winners(ax[1], records)
    panel_critic_scatter(ax[2], metrics)
    panel_critic_skill(ax[3], metrics)
    panel_regret(ax[4], metrics)
    if has_routing:
        panel_routing(ax[5], routing)
        panel_efficiency(ax[6], routing)
        panel_retention(ax[7], routing)
        panel_type_usage(ax[8], routing)
    else:
        for a in ax[5:]:
            a.axis("off")

    fig.suptitle("Heterogeneous credit assignment: candidate-level counterfactual "
                 "critic  (Qwen2.5-3B-Instruct)", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = config.RESULTS / "results.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")

    # Individual panels for reuse in the writeup.
    for name, fn, arg in [("utility_by_kind", panel_utility_by_kind, records),
                          ("winners", panel_winners, records),
                          ("critic_scatter", panel_critic_scatter, metrics),
                          ("critic_skill", panel_critic_skill, metrics),
                          ("regret", panel_regret, metrics)] + (
            [("routing", panel_routing, routing),
             ("efficiency", panel_efficiency, routing),
             ("retention", panel_retention, routing),
             ("type_usage", panel_type_usage, routing)] if has_routing else []):
        f, a = plt.subplots(figsize=(5.2, 4))
        fn(a, arg)
        f.tight_layout()
        f.savefig(config.RESULTS / f"fig_{name}.png", bbox_inches="tight")
        plt.close(f)
    print(f"saved individual panels -> {config.RESULTS}")


if __name__ == "__main__":
    main()
