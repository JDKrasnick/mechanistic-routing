"""Construction of diverse system states.

A state is (current prompt, stack of already-applied interventions). States are
built by a short random walk from the base system so that the collected dataset
covers already-adapted systems, not just the pristine one. This is what makes
the routing problem state-dependent: the best next intervention differs
depending on what has already been applied, which also exposes interaction
effects between intervention types.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

import config
from evaluation.tasks import Example, Task, sample_batch


@dataclass
class EpisodeContext:
    task_batch: List[Example]
    retention_batch: List[Example]
    failure_pool: List[Example]


def make_context(task: Task, retention_task: Task, rng: np.random.RandomState
                 ) -> EpisodeContext:
    return EpisodeContext(
        task_batch=sample_batch(task.eval, config.TASK_EVAL_N, rng),
        retention_batch=sample_batch(retention_task.eval, config.RETENTION_EVAL_N, rng),
        failure_pool=sample_batch(task.train, config.FAILURE_POOL_N, rng),
    )


def current_failures(evaluator, task: Task, ctx: EpisodeContext,
                     rng: np.random.RandomState, return_logits: bool = False):
    """Failure batch driving candidate generation, from the TRAIN split only."""
    failures, pool_logits = evaluator.failures(task, ctx.failure_pool)
    if len(failures) < 4:  # degenerate state: pad with random train examples
        extra = sample_batch(task.train, 8, rng)
        failures = failures + [(e, (e.label + 1) % len(task.labels)) for e in extra]
    if len(failures) > config.FAILURE_BATCH_N:
        idx = rng.choice(len(failures), config.FAILURE_BATCH_N, replace=False)
        failures = [failures[i] for i in idx]
    return (failures, pool_logits) if return_logits else failures


def propose_all(generators, failures, rng, n_per_type: int = config.CANDIDATES_PER_TYPE):
    """Cheap proposal of one candidate set per intervention type."""
    prompt_gen, act_gen, lora_gen = generators
    failure_examples = [e for e, _ in failures]
    return (prompt_gen.propose(failures, rng, n_per_type)
            + act_gen.propose(failures, rng, n_per_type)
            + lora_gen.propose(failure_examples, rng, n_per_type))


def seed_state(system, evaluator, task, generators, ctx, rng, depth: int):
    """Random-walk `depth` interventions from base to create a starting state."""
    system.revert_all()
    for _ in range(depth):
        failures = current_failures(evaluator, task, ctx, rng)
        cands = propose_all(generators, failures, rng, n_per_type=1)
        choice = cands[rng.randint(len(cands))]
        system.apply_intervention(choice)
    return system.capture_state()
