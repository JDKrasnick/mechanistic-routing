"""Prompt candidate generation: an LLM textual-gradient rewrite plus grounded
edits derived from the observed confusion structure of the failure batch."""
from collections import Counter
from typing import List

import numpy as np

from interventions.prompt import PromptIntervention

STRATEGY_HINTS = [
    "Focus on the feeling the writer reports about themselves, not the topic they mention.",
    "Choose the single dominant emotion; ignore secondary feelings mentioned in passing.",
    "Words like 'feel' or 'feeling' are often followed by the true label word.",
    "Distinguish affection toward a person (love) from general happiness (joy).",
    "Distinguish anxiety about the future (fear) from present irritation (anger).",
    "Do not default to the most common emotion; weigh the specific wording.",
]

REWRITE_TEMPLATE = """You are improving the instruction line of a text classifier.

Current instruction: "{instr}"

The classifier made these mistakes (text -> correct label, but predicted label):
{failures}

Write ONE improved instruction line (a single sentence, under 30 words) that would
help avoid these specific mistakes. Output only the instruction, nothing else."""


class PromptGenerator:
    def __init__(self, system, task):
        self.system = system
        self.task = task

    def _llm_rewrite(self, failures, seed: int) -> str:
        lines = []
        for e, pred in failures[:6]:
            txt = " ".join(e.text.split()[:18])
            lines.append(f'- "{txt}" -> {self.task.labels[e.label]}, but predicted {self.task.labels[pred]}')
        prompt = REWRITE_TEMPLATE.format(instr=self.system.instruction,
                                         failures="\n".join(lines))
        out = self.system.generate(prompt, max_new_tokens=64, temperature=0.8, seed=seed)
        cand = out.strip().strip('"').split("\n")[0].strip()
        if len(cand) < 15 or len(cand) > 220:
            return self.system.instruction + " " + STRATEGY_HINTS[seed % len(STRATEGY_HINTS)]
        return cand

    def _confusion_edit(self, failures) -> str:
        pairs = Counter((self.task.labels[e.label], self.task.labels[p]) for e, p in failures)
        if not pairs:
            return self.system.instruction + " Read the text carefully before choosing."
        (gold, pred), _ = pairs.most_common(1)[0]
        return (f"{self.system.instruction} Be careful not to label {gold} text as "
                f"{pred}; check for explicit {gold} cues first.")

    def propose(self, failures, rng: np.random.RandomState, n: int = 3
                ) -> List[PromptIntervention]:
        base = self.system.instruction
        cands = [
            PromptIntervention.propose(self._llm_rewrite(failures, int(rng.randint(1 << 30))),
                                       "llm_rewrite", base),
            PromptIntervention.propose(self._confusion_edit(failures), "confusion_edit", base),
            PromptIntervention.propose(
                base + " " + STRATEGY_HINTS[int(rng.randint(len(STRATEGY_HINTS)))],
                "strategy_hint", base),
        ]
        return cands[:n]
