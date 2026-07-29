"""Standardized evaluation: task score, retention score, and failure feedback."""
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

import config
from evaluation.tasks import Example, Task


@dataclass
class EvalResult:
    accuracy: float
    gold_logprob: float      # low-variance calibration signal
    margin: float            # mean(gold logit - best competitor logit)
    entropy: float
    n: int

    def to_dict(self) -> Dict:
        return asdict(self)


class Evaluator:
    def __init__(self, system):
        self.system = system
        self._label_ids: Dict[str, List[int]] = {}

    def _ids(self, task: Task) -> List[int]:
        if task.name not in self._label_ids:
            self._label_ids[task.name] = self.system.label_token_ids(task.labels)
        return self._label_ids[task.name]

    def evaluate(self, task: Task, batch: List[Example],
                 use_current_instruction: bool = True) -> EvalResult:
        instr = self.system.instruction if use_current_instruction else task.instruction
        prompts = [task.render(e.text, instr) for e in batch]
        logits = self.system.label_logits(prompts, self._ids(task))
        gold = np.array([e.label for e in batch])

        pred = logits.argmax(1)
        acc = float((pred == gold).mean())

        lp = logits - _logsumexp(logits, axis=1, keepdims=True)
        gold_lp = float(lp[np.arange(len(gold)), gold].mean())

        gold_logit = logits[np.arange(len(gold)), gold]
        masked = logits.copy()
        masked[np.arange(len(gold)), gold] = -np.inf
        margin = float((gold_logit - masked.max(1)).mean())

        p = np.exp(lp)
        entropy = float((-(p * lp).sum(1)).mean())
        return EvalResult(acc, gold_lp, margin, entropy, len(batch))

    def failures(self, task: Task, pool: List[Example]):
        """Return mispredicted examples plus the model's wrong prediction."""
        prompts = [task.render(e.text, self.system.instruction) for e in pool]
        logits = self.system.label_logits(prompts, self._ids(task))
        pred = logits.argmax(1)
        out = []
        for e, p in zip(pool, pred):
            if p != e.label:
                out.append((e, int(p)))
        return out, logits


def _logsumexp(x, axis, keepdims=False):
    m = x.max(axis=axis, keepdims=True)
    s = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
    return s if keepdims else np.squeeze(s, axis=axis)


def utility(task_delta: float, retention_delta: float) -> float:
    """Scalar utility the oracle maximizes and the critic predicts.

    Retention loss is penalized; retention gains are not rewarded (they are
    noise around an already-satisfactory capability level).
    """
    return task_delta + config.RETENTION_PENALTY * min(0.0, retention_delta)
