"""Base-accuracy probe: score each registry task's eval batch with the pristine
model (no interventions) to bucket tasks as ceiling / floor-noise / healthy
BEFORE spending collection budget.

Reports accuracy plus the low-variance margin/gold_logprob signals, the binomial
noise band at this N, and a verdict. Cheap: one forward-eval pass per task.

    python3 experiments/probe_base_accuracy.py [n_eval]
"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model_wrapper import SystemModel
from evaluation.evaluator import Evaluator
from evaluation.tasks import load_task_registry


def verdict(acc: float, chance: float, sd: float) -> str:
    if acc > 0.90:
        return "CEILING (little headroom)"
    if acc <= chance + 2 * sd:
        return "FLOOR/NOISE (near chance)"
    if acc < chance:
        return "BELOW CHANCE (broken)"
    return "HEALTHY"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    reg = load_task_registry()
    tasks = list(reg.targets.values()) + [reg.retention]

    print("loading model...", flush=True)
    system = SystemModel()
    ev = Evaluator(system)

    print(f"{'task':14} {'fam':9} {'acc':>6} {'chance':>7} {'±noise':>7} "
          f"{'margin':>7} {'gold_lp':>8}  verdict", flush=True)
    print("-" * 92, flush=True)
    for t in tasks:
        batch = t.eval[:n]
        chance = 1.0 / len(t.labels)
        sd = math.sqrt(chance * (1 - chance) / len(batch))
        system.set_base_instruction(t.instruction)
        r = ev.evaluate(t, batch)
        print(f"{t.name:14} {t.family:9} {r.accuracy:6.3f} {chance:7.3f} "
              f"{2*sd:7.3f} {r.margin:7.2f} {r.gold_logprob:8.2f}  "
              f"{verdict(r.accuracy, chance, sd)}", flush=True)


if __name__ == "__main__":
    main()
