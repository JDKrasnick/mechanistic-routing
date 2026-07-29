"""Verify that apply/revert is exact for all three intervention types.

If reverting leaks state, every counterfactual measurement downstream is
contaminated, so this gate runs before any data collection.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import config
from model_wrapper import SystemModel
from evaluation.tasks import load_tasks, sample_batch
from evaluation.evaluator import Evaluator
from candidates.prompt_generator import PromptGenerator
from candidates.activation_generator import ActivationGenerator
from candidates.lora_generator import LoRAGenerator
from states import make_context, current_failures, propose_all

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}", flush=True)
    if not cond:
        FAILED.append(name)


def main():
    rng = np.random.RandomState(0)
    task, retention = load_tasks()
    system = SystemModel()
    system.set_base_instruction(task.instruction)
    ev = Evaluator(system)
    ctx = make_context(task, retention, rng)

    label_ids = system.label_token_ids(task.labels)
    probe = [task.render(e.text, system.instruction) for e in ctx.task_batch[:8]]
    base_logits = system.label_logits(probe, label_ids)
    base_acc = ev.evaluate(task, ctx.task_batch).accuracy
    print(f"baseline acc={base_acc:.3f}\n")

    gens = (PromptGenerator(system, task), ActivationGenerator(system, task),
            LoRAGenerator(system, task))
    failures = current_failures(ev, task, ctx, rng)
    print(f"failure batch size = {len(failures)}\n")
    cands = propose_all(gens, failures, rng, n_per_type=1)

    state = system.capture_state()
    for c in cands:
        print(f"{c.kind}: {c.spec()}")
        c.apply(system)
        post = system.label_logits(probe, label_ids)
        changed = not np.allclose(post, base_logits, atol=1e-4)
        # A prompt edit changes the rendered text, not the logits of a fixed probe.
        if c.kind == "prompt":
            changed = system.instruction != task.instruction
        check(f"{c.kind} apply has effect", changed)

        c.revert(system)
        restored = system.label_logits(probe, label_ids)
        maxdiff = float(np.abs(restored - base_logits).max())
        check(f"{c.kind} revert is exact", maxdiff == 0.0, f"(max|delta|={maxdiff:.2e})")
        check(f"{c.kind} no steering leak", not system._steer_hooks)
        check(f"{c.kind} no lora leak", not system._lora_modules)
        check(f"{c.kind} instruction restored", system.instruction == task.instruction)
        print()

    # Stacked state: apply all three, capture, perturb, restore.
    print("stacked state capture/restore")
    for c in cands:
        system.apply_intervention(c)
    stacked = system.capture_state()
    stacked_logits = system.label_logits(probe, label_ids)
    stacked_instr = system.instruction

    system.revert_all()
    back_to_base = system.label_logits(probe, label_ids)
    check("revert_all returns to base", float(np.abs(back_to_base - base_logits).max()) == 0.0)

    system.restore_state(stacked)
    re_stacked = system.label_logits(probe, label_ids)
    d = float(np.abs(re_stacked - stacked_logits).max())
    check("restore_state reproduces stacked logits", d == 0.0, f"(max|delta|={d:.2e})")
    check("restore_state reproduces instruction", system.instruction == stacked_instr)

    system.restore_state(state)
    final = system.label_logits(probe, label_ids)
    check("restore to captured base state", float(np.abs(final - base_logits).max()) == 0.0)

    print("\n" + ("ALL CHECKS PASSED" if not FAILED else f"FAILURES: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
