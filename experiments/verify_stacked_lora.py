"""Regression test: stacked LoRA adapters must compose and revert in order."""
import sys
sys.path.insert(0, "/Users/fastcheetah/mechanistic-routing")
import numpy as np, torch

import config
from model_wrapper import SystemModel
from evaluation.tasks import load_tasks
from interventions.lora import LoRAIntervention

task, _ = load_tasks()
sm = SystemModel()
sm.set_base_instruction(task.instruction)
ids = sm.label_token_ids(task.labels)
probe = [task.render(e.text, task.instruction) for e in task.eval[:8]]
base = sm.label_logits(probe, ids)
fex = task.train[:12]

a = LoRAIntervention(8, 3e-4, 4, 8, ("q_proj", "v_proj"), fex, task)
b = LoRAIntervention(4, 3e-4, 4, 8, ("q_proj", "v_proj"), fex, task)

sm.apply_intervention(a)
after_a = sm.label_logits(probe, ids)
print("A applied, changed:", not np.allclose(after_a, base, atol=1e-4))

sm.apply_intervention(b)   # wraps A -- this is what crashed before
after_ab = sm.label_logits(probe, ids)
print("B stacked on A, changed:", not np.allclose(after_ab, after_a, atol=1e-4))
print("nesting:", type(sm.layers[-1].self_attn.q_proj).__name__,
      "->", type(sm.layers[-1].self_attn.q_proj.base).__name__,
      "->", type(sm.layers[-1].self_attn.q_proj.base.base).__name__)

state = sm.capture_state()
sm.revert_all()
back = sm.label_logits(probe, ids)
print("revert_all exact:", float(np.abs(back - base).max()) == 0.0)

sm.restore_state(state)
re = sm.label_logits(probe, ids)
print("restore stacked exact:", float(np.abs(re - after_ab).max()) == 0.0)

sm.revert_all()
print("final exact:", float(np.abs(sm.label_logits(probe, ids) - base).max()) == 0.0)
print("no leaks:", not sm._lora_modules and not sm._steer_hooks)
print("STACKED LORA OK")
