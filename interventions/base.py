"""Unified intervention interface.

Contract that makes budgeted routing possible:
  propose()   -- cheap. Produces a *specification* only; no model execution.
  apply()     -- expensive. Realizes the spec (may train weights / fit vectors)
                 and mutates the system. Caches its realized payload so that a
                 later re-apply during state restoration is free.
  revert()    -- exactly undoes apply(); leaves no residue.
  cost()      -- normalized expense of realizing + evaluating this candidate.
  serialize() -- JSON-safe record for the intervention dataset.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import itertools

_ids = itertools.count()


class Intervention(ABC):
    kind: str = "abstract"

    def __init__(self):
        self.uid = f"{self.kind}-{next(_ids)}"
        self._realized: Any = None

    @classmethod
    @abstractmethod
    def propose(cls, *args, **kwargs) -> "Intervention":
        """Cheap: return a candidate specification without touching the model."""

    @abstractmethod
    def apply(self, system) -> None:
        ...

    @abstractmethod
    def revert(self, system) -> None:
        ...

    @abstractmethod
    def cost(self) -> float:
        ...

    @abstractmethod
    def spec(self) -> Dict:
        """Type-specific hyperparameters, JSON-safe."""

    def serialize(self) -> Dict:
        return {"uid": self.uid, "kind": self.kind, "cost": self.cost(), **self.spec()}

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.uid} {self.spec()}>"
