"""Calibrate intervention magnitudes so each type has a real effect range.

Without this, the counterfactual dataset has no signal and the critic learns
nothing. Uses a larger eval batch than collection to keep noise down.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config
from model_wrapper import SystemModel
from evaluation.tasks import load_tasks, sample_batch
from evaluation.evaluator import Evaluator
from interventions.activation import ActivationIntervention
from interventions.lora import LoRAIntervention
from interventions.prompt import PromptIntervention
from candidates.prompt_generator import STRATEGY_HINTS
from states import make_context, current_failures

N_EVAL = 80


def main():
    rng = np.random.RandomState(config.SEED)
    task, retention = load_tasks()
    system = SystemModel()
    system.set_base_instruction(task.instruction)
    ev = Evaluator(system)
    ctx = make_context(task, retention, rng)
    big = sample_batch(task.eval, N_EVAL, np.random.RandomState(123))

    base = ev.evaluate(task, big)
    base_ret = ev.evaluate(retention, ctx.retention_batch)
    print(f"BASE task={base.accuracy:.3f} logp={base.gold_logprob:.3f} "
          f"ret={base_ret.accuracy:.3f}  (n={N_EVAL})\n")

    failures = current_failures(ev, task, ctx, rng)
    fex = [e for e, _ in failures]
    gold = [task.render(e.text, system.instruction) + f" {task.labels[e.label]}"
            for e, _ in failures]
    wrong = [task.render(e.text, system.instruction) + f" {task.labels[p]}"
             for e, p in failures]
    state = system.capture_state()

    print("=== ACTIVATION: layer x strength ===")
    for layer in (14, 22, 30):
        iv0 = ActivationIntervention(layer, 1.0, gold, wrong)
        iv0._fit(system)
        v, vn, an = iv0._vector, iv0._fit_norm, iv0._act_norm
        print(f"layer {layer}: ||v||={vn:.2f} mean||h||={an:.2f} ratio={vn/an:.3f}")
        for s in (1.0, 2.0, 4.0, 8.0, 16.0):
            iv = ActivationIntervention(layer, s, gold, wrong)
            iv._vector, iv._fit_norm, iv._act_norm = v, vn, an
            system.apply_intervention(iv)
            r = ev.evaluate(task, big)
            rr = ev.evaluate(retention, ctx.retention_batch)
            system.restore_state(state)
            print(f"   s={s:5.1f} rel={s*vn/an:5.2f}  task={r.accuracy:.3f} "
                  f"({r.accuracy-base.accuracy:+.3f}) logp={r.gold_logprob:+.3f} "
                  f"ret={rr.accuracy-base_ret.accuracy:+.3f}")

    print("\n=== LORA: lr x steps (rank 8, top 8 layers) ===")
    for lr in (3e-4, 1e-3, 3e-3):
        for steps in (16, 32):
            t = time.time()
            iv = LoRAIntervention(8, lr, steps, 8, ("q_proj", "v_proj"), fex, task)
            system.apply_intervention(iv)
            r = ev.evaluate(task, big)
            rr = ev.evaluate(retention, ctx.retention_batch)
            system.restore_state(state)
            print(f"   lr={lr:.0e} steps={steps:2d} loss={iv._final_loss:.3f} "
                  f"task={r.accuracy:.3f} ({r.accuracy-base.accuracy:+.3f}) "
                  f"logp={r.gold_logprob:+.3f} ret={rr.accuracy-base_ret.accuracy:+.3f} "
                  f"[{time.time()-t:.0f}s]")

    print("\n=== PROMPT: strategy hints ===")
    for i, hint in enumerate(STRATEGY_HINTS):
        iv = PromptIntervention(task.instruction + " " + hint, "hint", task.instruction)
        system.apply_intervention(iv)
        r = ev.evaluate(task, big)
        system.restore_state(state)
        print(f"   hint{i}: task={r.accuracy:.3f} ({r.accuracy-base.accuracy:+.3f}) "
              f"logp={r.gold_logprob:+.3f}  \"{hint[:60]}\"")


if __name__ == "__main__":
    main()
