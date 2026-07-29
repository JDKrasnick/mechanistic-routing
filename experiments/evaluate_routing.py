"""Routing experiment harness under matched budgets.

Policies compared: prompt-only, activation-only, lora-only, random, heuristic,
critic, oracle. Every policy runs the same held-out episodes from the same
initial states with the same evaluation batches, so differences are attributable
to the routing decision rather than to the draw.

Cost accounting separates two quantities:
  selection cost   -- cost of the candidate actually applied (what the budget caps)
  exploration cost -- cost of every candidate executed to make the decision
Only the oracle pays a large exploration cost; that gap is the efficiency claim.
"""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config
from model_wrapper import SystemModel
from evaluation.tasks import load_tasks, sample_batch
from evaluation.evaluator import Evaluator, utility
from candidates.prompt_generator import PromptGenerator
from candidates.activation_generator import ActivationGenerator
from candidates.lora_generator import LoRAGenerator
from counterfactuals.runner import CounterfactualRunner
from counterfactuals.dataset import InterventionDataset
from critic.features import compute_state_features
from critic.selector import (CriticPolicy, HeuristicPolicy, RandomPolicy,
                             SingleTypePolicy)
from critic import train as critic_train
from states import make_context, current_failures, propose_all, seed_state

CRITIC_PATH = config.DATA / "critic.pkl"
OUT = config.RESULTS / "routing_results.json"
FINAL_EVAL_N = 80


def episode_seed(ep): return 5000 + ep


def build_policies(critic, kind_ranking):
    return [
        SingleTypePolicy("prompt"),
        SingleTypePolicy("activation"),
        SingleTypePolicy("lora"),
        RandomPolicy(),
        HeuristicPolicy(kind_ranking),
        CriticPolicy(critic),
    ]


def run_episode(policy_name, policy, system, ev, runner, task, retention, gens,
                ep, final_batch, final_ret_batch, budget):
    """One routing episode. policy=None means the exhaustive oracle."""
    rng_state = np.random.RandomState(episode_seed(ep))
    ctx = make_context(task, retention, rng_state)

    system.revert_all()
    state = seed_state(system, ev, task, gens, ctx, rng_state, depth=1)

    start = runner.baseline(ctx.task_batch, ctx.retention_batch)
    start_final = ev.evaluate(task, final_batch)
    hist = {"selection_cost": 0.0, "exploration_cost": 0.0, "applied": [],
            "round_task": [], "round_utility": []}
    # Each round's baseline is the previous round's post-measurement, so it is
    # carried forward rather than re-evaluated.
    cur = start

    for rd in range(config.N_ROUTING_ROUNDS):
        rng = np.random.RandomState(episode_seed(ep) * 100 + rd)
        base = cur
        failures, pool_logits = current_failures(ev, task, ctx, rng, return_logits=True)
        sfeat = compute_state_features(base["task"], base["retention"], system,
                                       failures, pool_logits, task)
        candidates = propose_all(gens, failures, rng)
        snapshot = system.capture_state()

        if policy is None:  # oracle: execute everything, keep measured best
            recs = runner.run_all(candidates, snapshot, base, ctx.task_batch,
                                  ctx.retention_batch, verbose=False)
            hist["exploration_cost"] += sum(r["cost"] for r in recs)
            affordable = [(r, c) for r, c in zip(recs, candidates) if r["cost"] <= budget]
            if not affordable:
                break
            rec, chosen = max(affordable, key=lambda rc: rc[0]["utility"])
            system.restore_state(snapshot)
            if rec["utility"] <= 0:  # oracle declines to make things worse
                hist["round_task"].append(base["task"].accuracy)
                hist["round_utility"].append(0.0)
                continue
            system.apply_intervention(chosen)
        else:
            chosen = policy.select(sfeat, candidates, budget, rng)
            if chosen is None:  # critic may decline when nothing looks worthwhile
                hist["round_task"].append(base["task"].accuracy)
                hist["round_utility"].append(0.0)
                continue
            system.apply_intervention(chosen)
            hist["exploration_cost"] += chosen.cost()

        t = ev.evaluate(task, ctx.task_batch)
        r = ev.evaluate(retention, ctx.retention_batch)
        cur = {"task": t, "retention": r, "runtime": 0.0}
        post_task, post_ret = t.accuracy, r.accuracy
        gain = utility(post_task - base["task"].accuracy,
                       post_ret - base["retention"].accuracy)

        hist["selection_cost"] += chosen.cost()
        hist["applied"].append({"kind": chosen.kind, "cost": chosen.cost(),
                                "utility": gain})
        hist["round_task"].append(post_task)
        hist["round_utility"].append(gain)

    final = ev.evaluate(task, final_batch)
    final_ret = ev.evaluate(retention, final_ret_batch)
    system.revert_all()

    return {
        "policy": policy_name, "episode": ep,
        "start_task_small": start["task"].accuracy,
        "start_task_final": start_final.accuracy,
        "final_task": final.accuracy,
        "final_retention": final_ret.accuracy,
        "task_gain": final.accuracy - start_final.accuracy,
        "selection_cost": hist["selection_cost"],
        "exploration_cost": hist["exploration_cost"],
        "applied": hist["applied"],
        "round_task": hist["round_task"],
        "kinds_used": [a["kind"] for a in hist["applied"]],
    }


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else config.N_ROUTING_EPISODES
    ds = InterventionDataset(config.DATA / "intervention_dataset.jsonl")
    records = list(ds)
    assert records, "run collect_oracle_data.py first"

    critic = critic_train.load(CRITIC_PATH)
    by_kind = {}
    for r in records:
        by_kind.setdefault(r["kind"], []).append(r["utility"])
    kind_ranking = sorted(by_kind, key=lambda k: -np.mean(by_kind[k]))
    print("heuristic type ranking (by train mean utility):",
          [(k, round(float(np.mean(by_kind[k])), 4)) for k in kind_ranking], flush=True)

    task, retention = load_tasks()
    system = SystemModel()
    system.set_base_instruction(task.instruction)
    ev = Evaluator(system)
    runner = CounterfactualRunner(system, ev, task, retention)
    gens = (PromptGenerator(system, task), ActivationGenerator(system, task),
            LoRAGenerator(system, task))

    # Held-out final-evaluation batches, fixed across all policies and episodes.
    fr = np.random.RandomState(999)
    final_batch = sample_batch(task.eval, FINAL_EVAL_N, fr)
    final_ret_batch = sample_batch(retention.eval, FINAL_EVAL_N, fr)

    policies = [(p.name, p) for p in build_policies(critic, kind_ranking)]
    policies.append(("oracle", None))

    results, t0 = [], time.time()
    for ep in range(n_eps):
        for name, pol in policies:
            t = time.time()
            r = run_episode(name, pol, system, ev, runner, task, retention, gens,
                            ep, final_batch, final_ret_batch, config.ROUTING_BUDGET)
            r["seconds"] = time.time() - t
            results.append(r)
            print(f"ep{ep} {name:16s} gain={r['task_gain']:+.3f} "
                  f"final={r['final_task']:.3f} ret={r['final_retention']:.3f} "
                  f"sel_cost={r['selection_cost']:.1f} "
                  f"explore={r['exploration_cost']:.1f} "
                  f"kinds={r['kinds_used']} [{r['seconds']:.0f}s]", flush=True)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
        print(f"--- episode {ep} done, elapsed {(time.time()-t0)/60:.1f}m ---",
              flush=True)

    print(f"\nrouting complete in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
