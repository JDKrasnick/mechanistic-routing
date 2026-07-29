"""Exhaustive counterfactual data collection.

For each state: propose one candidate set per intervention type, execute EVERY
candidate in isolation, and record its measured held-out utility. This is the
expensive procedure the critic is meant to replace, and it produces both the
critic's training targets and the oracle upper bound.

Resumable: re-running appends only the states still missing.
"""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config
from model_wrapper import SystemModel
from evaluation.tasks import load_tasks
from evaluation.evaluator import Evaluator
from candidates.prompt_generator import PromptGenerator
from candidates.activation_generator import ActivationGenerator
from candidates.lora_generator import LoRAGenerator
from counterfactuals.runner import CounterfactualRunner
from counterfactuals.dataset import InterventionDataset
from counterfactuals.oracle import oracle_summary
from critic.features import compute_state_features
from states import make_context, current_failures, propose_all, seed_state

OUT = config.DATA / "intervention_dataset.jsonl"
SUMMARY = config.DATA / "collection_summary.json"
DEPTH_CYCLE = [0, 0, 1, 1, 2]


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else config.N_COLLECT_STATES
    ds = InterventionDataset(OUT)
    done = {r["state_id"] for r in ds}
    print(f"target {n_target} states, {len(done)} already collected", flush=True)

    task, retention = load_tasks()
    system = SystemModel()
    system.set_base_instruction(task.instruction)
    ev = Evaluator(system)
    runner = CounterfactualRunner(system, ev, task, retention)
    gens = (PromptGenerator(system, task), ActivationGenerator(system, task),
            LoRAGenerator(system, task))

    t_start = time.time()
    summaries = []
    for si in range(n_target):
        sid = f"train-{si:03d}"
        if sid in done:
            continue
        t0 = time.time()
        rng = np.random.RandomState(1000 + si)
        ctx = make_context(task, retention, rng)
        depth = DEPTH_CYCLE[si % len(DEPTH_CYCLE)]

        system.revert_all()
        state = seed_state(system, ev, task, gens, ctx, rng, depth)

        base = runner.baseline(ctx.task_batch, ctx.retention_batch)
        failures, pool_logits = current_failures(ev, task, ctx, rng, return_logits=True)
        sfeat = compute_state_features(base["task"], base["retention"], system,
                                       failures, pool_logits, task)
        candidates = propose_all(gens, failures, rng)

        print(f"[{si+1}/{n_target}] {sid} depth={depth} "
              f"base_task={base['task'].accuracy:.3f} "
              f"base_ret={base['retention'].accuracy:.3f} "
              f"failures={len(failures)} cands={len(candidates)}", flush=True)

        records = runner.run_all(candidates, state, base, ctx.task_batch,
                                 ctx.retention_batch)
        for r in records:
            r.update({"state_id": sid, "depth": depth, "state_features": sfeat,
                      "seed": 1000 + si})
            ds.append(r)

        summ = oracle_summary(records, config.ROUTING_BUDGET)
        summ.update({"state_id": sid, "depth": depth,
                     "base_task": base["task"].accuracy,
                     "state_seconds": time.time() - t0})
        summaries.append(summ)
        elapsed = time.time() - t_start
        remaining = n_target - si - 1
        print(f"    oracle: {summ['best_kind']} util={summ['best_utility']:+.3f} "
              f"| state {time.time()-t0:.0f}s | elapsed {elapsed/60:.1f}m "
              f"| eta {remaining*(time.time()-t0)/60:.0f}m", flush=True)

        system.revert_all()

    with open(SUMMARY, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\ncollection complete: {sum(1 for _ in ds)} records "
          f"in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
