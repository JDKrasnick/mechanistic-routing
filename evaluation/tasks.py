"""Task definitions and the multi-task registry.

Core types (`Task`, `Example`) plus the two task families used for collection:

- synthetic : autocreated, exact ground truth        (evaluation/synthetic.py)
- benchmark : vetted external datasets               (evaluation/benchmarks.py)

Every task is a small fixed-label-set classification so it scores through the
single-forward-pass evaluator. `load_task_registry()` returns the target tasks
plus the retention probe; `load_tasks()` is the legacy single-pair accessor kept
for the baseline scripts.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

import config


@dataclass
class Example:
    text: str
    label: int


@dataclass
class Task:
    name: str
    instruction: str
    labels: List[str]
    train: List[Example] = field(default_factory=list)
    eval: List[Example] = field(default_factory=list)
    family: str = "benchmark"

    def render(self, text: str, instruction: str = None) -> str:
        """Raw completion format. Measured to beat chat-template formats on both
        accuracy and sequence length for this model."""
        instr = instruction if instruction is not None else self.instruction
        return f"{instr} ({', '.join(self.labels)})\n\nText: {text}\nLabel:"


def _truncate(t: str, n: int) -> str:
    return " ".join(t.split()[:n])


# Target tasks span both families; ag_news is held out as the retention probe.
# code_defect (CodeXGLUE) measured at chance (0.47, margin ~0) on the 3B base;
# replaced by the autocreated code_output task, which has tunable, real signal.
REGISTRY_TARGETS = ["arithmetic_tf", "grammar", "code_output", "sst2", "snli", "emotion"]
RETENTION_TASK = "ag_news"


@dataclass
class TaskRegistry:
    targets: Dict[str, Task]
    retention: Task

    def by_family(self, family: str) -> Dict[str, Task]:
        return {n: t for n, t in self.targets.items() if t.family == family}


def load_task_registry(target_names: List[str] = None,
                       seed: int = config.SEED) -> TaskRegistry:
    """Build the family-tagged registry: target tasks + retention probe."""
    from evaluation.benchmarks import BENCHMARK_BUILDERS, ag_news_task
    from evaluation.synthetic import SYNTHETIC_BUILDERS

    names = target_names if target_names is not None else REGISTRY_TARGETS
    builders = {**SYNTHETIC_BUILDERS, **BENCHMARK_BUILDERS}
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; known: {sorted(builders)}")

    targets = {n: builders[n](seed=seed) for n in names}
    return TaskRegistry(targets=targets, retention=ag_news_task(seed=seed))


def load_tasks(seed: int = config.SEED):
    """Legacy accessor: the baseline's (emotion target, ag_news retention) pair."""
    from evaluation.benchmarks import emotion_task, ag_news_task
    return emotion_task(seed=seed), ag_news_task(seed=seed)


def sample_batch(examples: List[Example], n: int, rng: np.random.RandomState) -> List[Example]:
    idx = rng.choice(len(examples), size=min(n, len(examples)), replace=False)
    return [examples[i] for i in idx]
