"""Composition gate: stacked and same-target interventions must revert exactly.

Both production crashes so far were composition bugs that only appear on states
that already carry an intervention:
  1. a LoRA candidate proposed for a state that already had a LoRA (wrapper
     could not wrap a wrapper);
  2. a steering candidate proposed at a layer that already had steering (removal
     relied on bfloat16 cancellation returning exactly zero, which it does not).
Depth-0 states exercise neither path, so these cases are tested explicitly.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import config
from model_wrapper import SystemModel
from evaluation.tasks import load_tasks
from interventions.activation import ActivationIntervention
from interventions.lora import LoRAIntervention
from interventions.prompt import PromptIntervention

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}", flush=True)
    if not cond:
        FAILED.append(name)


def main():
    task, _ = load_tasks()
    sm = SystemModel()
    sm.set_base_instruction(task.instruction)
    ids = sm.label_token_ids(task.labels)
    probe = [task.render(e.text, task.instruction) for e in task.eval[:8]]
    base = sm.label_logits(probe, ids)
    fex = task.train[:12]

    def logits():
        return sm.label_logits(probe, ids)

    def exact(a, b):
        return float(np.abs(a - b).max()) == 0.0

    gold = [task.render(e.text, task.instruction) + f" {task.labels[e.label]}" for e in fex]
    wrong = [task.render(e.text, task.instruction) + f" {task.labels[(e.label+1) % 6]}" for e in fex]

    scenarios = {
        "two steering, SAME layer": lambda: [
            ActivationIntervention(14, 0.5, gold, wrong),
            ActivationIntervention(14, 0.75, gold, wrong)],
        "two steering, different layers": lambda: [
            ActivationIntervention(14, 0.5, gold, wrong),
            ActivationIntervention(22, 0.5, gold, wrong)],
        "three steering, same layer": lambda: [
            ActivationIntervention(18, 0.25, gold, wrong),
            ActivationIntervention(18, 0.5, gold, wrong),
            ActivationIntervention(18, 0.75, gold, wrong)],
        "two LoRA, same targets": lambda: [
            LoRAIntervention(8, 3e-4, 4, 8, ("q_proj", "v_proj"), fex, task),
            LoRAIntervention(4, 3e-4, 4, 8, ("q_proj", "v_proj"), fex, task)],
        "LoRA + steering + prompt": lambda: [
            LoRAIntervention(8, 3e-4, 4, 8, ("q_proj", "v_proj"), fex, task),
            ActivationIntervention(14, 0.5, gold, wrong),
            PromptIntervention("Classify the emotion. Be precise.", "test", task.instruction)],
        "steering then LoRA at same depth": lambda: [
            ActivationIntervention(26, 0.5, gold, wrong),
            LoRAIntervention(4, 3e-4, 4, 4, ("q_proj",), fex, task)],
    }

    for name, make in scenarios.items():
        print(f"\n{name}")
        ivs = make()
        for iv in ivs:
            sm.apply_intervention(iv)
        stacked = logits()
        state = sm.capture_state()
        check("applied has effect", not np.allclose(stacked, base, atol=1e-5))

        sm.revert_all()
        check("revert_all exact", exact(logits(), base))
        check("no hook leak", not sm._steer_hooks and not sm._steer_terms,
              f"(hooks={list(sm._steer_hooks)})")
        check("no lora leak", not sm._lora_modules)

        sm.restore_state(state)
        check("restore reproduces stack", exact(logits(), stacked))
        sm.revert_all()
        check("final back to base", exact(logits(), base))

    # Reverting out of order must be rejected rather than silently corrupting.
    a = ActivationIntervention(14, 0.5, gold, wrong)
    sm.apply_intervention(a)
    a.revert(sm)
    try:
        a.revert(sm)
        check("double-revert rejected", False)
    except KeyError:
        check("double-revert rejected", True)
    sm._applied = []
    sm.revert_all()

    print("\n" + ("ALL COMPOSITION CHECKS PASSED" if not FAILED else f"FAILURES: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
