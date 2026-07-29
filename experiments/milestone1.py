"""First functional milestone: generate and evaluate one candidate of each
intervention type from a single state, with full counterfactual isolation.

Also calibrates per-state wall-clock cost for the full collection run.
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
from counterfactuals.oracle import oracle_summary
from states import make_context, current_failures, propose_all

N_PER_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def main():
    t_start = time.time()
    rng = np.random.RandomState(config.SEED)
    task, retention = load_tasks()

    system = SystemModel()
    system.set_base_instruction(task.instruction)
    ev = Evaluator(system)
    runner = CounterfactualRunner(system, ev, task, retention)
    gens = (PromptGenerator(system, task), ActivationGenerator(system, task),
            LoRAGenerator(system, task))

    ctx = make_context(task, retention, rng)

    state = system.capture_state()
    t = time.time()
    base = runner.baseline(ctx.task_batch, ctx.retention_batch)
    t_base = time.time() - t
    print(f"baseline: task={base['task'].accuracy:.3f} "
          f"retention={base['retention'].accuracy:.3f} "
          f"gold_logprob={base['task'].gold_logprob:.3f}  ({t_base:.0f}s)\n")

    t = time.time()
    failures = current_failures(ev, task, ctx, rng)
    t_fail = time.time() - t
    t = time.time()
    candidates = propose_all(gens, failures, rng, n_per_type=N_PER_TYPE)
    t_prop = time.time() - t
    print(f"{len(failures)} failures ({t_fail:.0f}s), "
          f"{len(candidates)} candidates proposed ({t_prop:.0f}s)\n")

    results = runner.run_all(candidates, state, base, ctx.task_batch, ctx.retention_batch)

    print("\n--- results ---")
    for r in results:
        print(f"{r['kind']:10s} util={r['utility']:+.3f}  task {r['baseline_task']:.3f}"
              f"->{r['post_task']:.3f}  ret {r['baseline_retention']:.3f}"
              f"->{r['post_retention']:.3f}  cost={r['cost']:.1f}  {r['runtime']:.0f}s")

    print("\noracle:", json.dumps(oracle_summary(results, config.ROUTING_BUDGET), indent=2))

    # Confirm the system really is back at baseline after the sweep.
    system.restore_state(state)
    after = runner.baseline(ctx.task_batch, ctx.retention_batch)
    ok = (after["task"].accuracy == base["task"].accuracy and
          after["retention"].accuracy == base["retention"].accuracy)
    print(f"\npost-sweep restore exact: {ok} "
          f"(task {after['task'].accuracy:.3f}, ret {after['retention'].accuracy:.3f})")

    total = time.time() - t_start
    per_state = t_base + t_fail + t_prop + sum(r["runtime"] for r in results)
    print(f"\ntotal {total/60:.1f} min | per-state cost {per_state:.0f}s "
          f"({len(candidates)} candidates)")
    print(f"projected {config.N_COLLECT_STATES} states with "
          f"{3*config.CANDIDATES_PER_TYPE} candidates: "
          f"{config.N_COLLECT_STATES * (t_base + t_fail + t_prop + sum(r['runtime'] for r in results) * config.CANDIDATES_PER_TYPE / N_PER_TYPE) / 3600:.2f} h")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
