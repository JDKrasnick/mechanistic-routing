"""Textual intervention: rewrite the task instruction."""
from typing import Dict

import config
from interventions.base import Intervention


class PromptIntervention(Intervention):
    kind = "prompt"

    def __init__(self, new_instruction: str, source: str, n_chars_added: int = 0):
        super().__init__()
        self.new_instruction = new_instruction
        self.source = source
        self.n_chars_added = n_chars_added
        self._prev = None

    @classmethod
    def propose(cls, new_instruction: str, source: str, base_instruction: str = ""):
        return cls(new_instruction, source,
                   n_chars_added=len(new_instruction) - len(base_instruction))

    def apply(self, system):
        self._prev = system.instruction
        system.instruction = self.new_instruction

    def revert(self, system):
        system.instruction = self._prev
        self._prev = None

    def cost(self) -> float:
        return config.COST_PROMPT

    def spec(self) -> Dict:
        return {"instruction": self.new_instruction, "source": self.source,
                "n_chars_added": self.n_chars_added,
                "instruction_len": len(self.new_instruction)}
