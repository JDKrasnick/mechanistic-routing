"""Counterfactual runner: clone state, apply each candidate independently,
evaluate, record, restore."""
import time
from typing import Dict, List

from evaluation.evaluator import EvalResult, Evaluator, utility


class CounterfactualRunner:
    def __init__(self, system, evaluator: Evaluator, task, retention_task):
        self.system = system
        self.ev = evaluator
        self.task = task
        self.retention_task = retention_task

    def baseline(self, task_batch, retention_batch) -> Dict:
        t = time.time()
        task_res = self.ev.evaluate(self.task, task_batch)
        ret_res = self.ev.evaluate(self.retention_task, retention_batch)
        return {"task": task_res, "retention": ret_res, "runtime": time.time() - t}

    def run_candidate(self, candidate, state, base: Dict,
                      task_batch, retention_batch) -> Dict:
        """Apply one candidate in isolation from `state`, measure, then restore."""
        self.system.restore_state(state)

        t0 = time.time()
        # Must go through the tracked stack, otherwise restore_state cannot
        # unwind the candidate and hooks/adapters leak into the next trial.
        self.system.apply_intervention(candidate)
        apply_time = time.time() - t0

        t1 = time.time()
        task_res: EvalResult = self.ev.evaluate(self.task, task_batch)
        ret_res: EvalResult = self.ev.evaluate(self.retention_task, retention_batch)
        eval_time = time.time() - t1

        rec = {
            "candidate": candidate.serialize(),
            "kind": candidate.kind,
            "baseline_task": base["task"].accuracy,
            "baseline_retention": base["retention"].accuracy,
            "baseline_gold_logprob": base["task"].gold_logprob,
            "baseline_margin": base["task"].margin,
            "post_task": task_res.accuracy,
            "post_retention": ret_res.accuracy,
            "post_gold_logprob": task_res.gold_logprob,
            "post_margin": task_res.margin,
            "improvement": task_res.accuracy - base["task"].accuracy,
            "retention_change": ret_res.accuracy - base["retention"].accuracy,
            "logprob_improvement": task_res.gold_logprob - base["task"].gold_logprob,
            "margin_improvement": task_res.margin - base["task"].margin,
            "cost": candidate.cost(),
            "apply_seconds": apply_time,
            "eval_seconds": eval_time,
            "runtime": apply_time + eval_time,
        }
        rec["utility"] = utility(rec["improvement"], rec["retention_change"])
        rec["net_value"] = rec["utility"] - 0.002 * rec["cost"]

        self.system.restore_state(state)
        return rec

    def run_all(self, candidates: List, state, base: Dict,
                task_batch, retention_batch, verbose: bool = True) -> List[Dict]:
        out = []
        for i, c in enumerate(candidates):
            rec = self.run_candidate(c, state, base, task_batch, retention_batch)
            out.append(rec)
            if verbose:
                print(f"    [{i+1}/{len(candidates)}] {c.kind:10s} "
                      f"util={rec['utility']:+.3f} task={rec['improvement']:+.3f} "
                      f"ret={rec['retention_change']:+.3f} cost={rec['cost']:.1f} "
                      f"{rec['runtime']:.0f}s", flush=True)
        return out
