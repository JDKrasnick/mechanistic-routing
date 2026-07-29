"""Autocreated ('basic') task family.

Each task is generated programmatically with exact ground truth and framed as a
small fixed-label classification so it scores through the single-forward-pass
evaluator. Ground truth is known by construction, so no dataset vetting is
needed and difficulty is a tunable knob (`level`) for later calibration.

Three tasks:
- arithmetic_tf : verify an equation            -> (yes, no)   [math]
- grammar       : judge grammatical correctness -> (yes, no)   [writing]
- code_output   : verify a snippet's output     -> (yes, no)   [code]

Both are class-balanced. Train and eval draws are disjoint (deduplicated on the
rendered text) so a corrupted/true item cannot leak across the split.
"""
from typing import Callable, List, Tuple

import numpy as np

from evaluation.tasks import Example, Task

YESNO = ["yes", "no"]  # index 0 = yes (correct/grammatical), 1 = no


# ---------------------------------------------------------------- arithmetic

_OPS: List[Tuple[str, Callable[[int, int], int]]] = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
]


def _arith_item(rng: np.random.RandomState, level: int) -> Tuple[str, int]:
    """One equation string and its label (0=correct, 1=incorrect)."""
    sym, fn = _OPS[rng.randint(len(_OPS))]
    hi = 10 ** level
    lo = 10 ** (level - 1) if level > 1 else 2
    if sym == "*":  # keep products readable: shrink the operand range
        a, b = rng.randint(2, 13), rng.randint(2, 13)
    else:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    truth = fn(a, b)

    if rng.rand() < 0.5:
        shown, label = truth, 0
    else:
        shown, label = truth + _distractor_delta(rng, truth), 1
    return f"{a} {sym} {b} = {shown}", label


def _distractor_delta(rng: np.random.RandomState, truth: int) -> int:
    """A wrong-but-plausible offset: small, non-zero, common-error shaped."""
    while True:
        d = rng.choice([-10, -2, -1, 1, 2, 9, 10, 11])
        if d != 0 and truth + d != truth:
            return int(d)


# ------------------------------------------------------------------- grammar

_SUBJECTS = [
    ("The dog", "The dogs"), ("The child", "The children"),
    ("The teacher", "The teachers"), ("My friend", "My friends"),
    ("The car", "The cars"), ("The bird", "The birds"),
    ("The student", "The students"), ("The engineer", "The engineers"),
]
_VERBS = [  # (third-person singular, base/plural)
    ("runs", "run"), ("eats", "eat"), ("writes", "write"),
    ("plays", "play"), ("reads", "read"), ("builds", "build"),
    ("watches", "watch"), ("drives", "drive"),
]
_TAILS = [
    "every morning", "in the park", "after school", "on weekends",
    "quite often", "near the river", "before lunch", "with great care",
]
_AN_NOUNS = ["apple", "hour", "orange", "umbrella", "idea", "engine"]
_A_NOUNS = ["banana", "house", "dog", "car", "book", "table"]


def _grammar_item(rng: np.random.RandomState, level: int) -> Tuple[str, int]:
    """One sentence and its label (0=grammatical, 1=ungrammatical)."""
    kind = rng.randint(2)
    if kind == 0:
        (sing_s, plur_s) = _SUBJECTS[rng.randint(len(_SUBJECTS))]
        (v_sing, v_plur) = _VERBS[rng.randint(len(_VERBS))]
        tail = _TAILS[rng.randint(len(_TAILS))]
        singular = rng.rand() < 0.5
        subj = sing_s if singular else plur_s
        good_v = v_sing if singular else v_plur
        bad_v = v_plur if singular else v_sing
        if rng.rand() < 0.5:
            return f"{subj} {good_v} {tail}.", 0
        return f"{subj} {bad_v} {tail}.", 1
    else:
        if rng.rand() < 0.5:
            noun, art = _AN_NOUNS[rng.randint(len(_AN_NOUNS))], "an"
            bad = "a"
        else:
            noun, art = _A_NOUNS[rng.randint(len(_A_NOUNS))], "a"
            bad = "an"
        tail = _TAILS[rng.randint(len(_TAILS))]
        if rng.rand() < 0.5:
            return f"I saw {art} {noun} {tail}.", 0
        return f"I saw {bad} {noun} {tail}.", 1


# ---------------------------------------------------------------- code output

_CODE_STRINGS = ["hello", "world", "python", "apple", "table", "river", "cloud"]


def _code_templates(rng: np.random.RandomState, level: int):
    """One snippet and its exact printed output, computed analytically (never
    executed). Returns (code, output_str, kind) with kind in {int,bool,str}."""
    lo, hi = (1, 10) if level < 2 else (10, 100)
    xs = [int(rng.randint(lo, hi)) for _ in range(4)]
    a, b = int(rng.randint(lo, hi)), int(rng.randint(lo, hi))
    s = _CODE_STRINGS[rng.randint(len(_CODE_STRINGS))]
    pick = rng.randint(9 if level < 2 else 10)
    if pick == 0:
        return f"a = {a}\nb = {b}\nprint(a * b)", str(a * b), "int"
    if pick == 1:
        i = rng.randint(4)
        return f"xs = {xs}\nprint(xs[{i}])", str(xs[i]), "int"
    if pick == 2:
        return f"xs = {xs}\nprint(len(xs))", "4", "int"
    if pick == 3:
        return f"xs = {xs}\nprint(sum(xs))", str(sum(xs)), "int"
    if pick == 4:
        return f"xs = {xs}\nprint(max(xs))", str(max(xs)), "int"
    if pick == 5:
        return f's = "{s}"\nprint(s.upper())', s.upper(), "str"
    if pick == 6:
        return f's = "{s}"\nprint(len(s))', str(len(s)), "int"
    if pick == 7:
        return f"print({a} > {b})", str(a > b), "bool"
    if pick == 8:
        return f'print("{s[:2]}" * 3)', s[:2] * 3, "str"
    return f"a = {a}\nb = {b}\nprint(a - b)", str(a - b), "int"


def _code_distractor(truth: str, kind: str, rng: np.random.RandomState) -> str:
    if kind == "bool":
        return "False" if truth == "True" else "True"
    if kind == "int":
        return str(int(truth) + _distractor_delta(rng, int(truth)))
    variants = [truth.swapcase(), truth[:-1], truth + truth[:1],
                _CODE_STRINGS[rng.randint(len(_CODE_STRINGS))]]
    for v in rng.permutation(len(variants)):
        if variants[v] != truth:
            return variants[v]
    return truth + "x"


def _code_item(rng: np.random.RandomState, level: int):
    """One snippet-with-stated-output and its label (0=correct, 1=incorrect)."""
    code, truth, kind = _code_templates(rng, level)
    if rng.rand() < 0.5:
        shown, label = truth, 0
    else:
        shown, label = _code_distractor(truth, kind, rng), 1
    return f"{code}\nOutput: {shown}", label


# --------------------------------------------------------------- generation

def _generate(item_fn, n: int, rng: np.random.RandomState, level: int,
              seen: set) -> List[Example]:
    """Draw `n` deduplicated, class-balanced examples."""
    per_class = n // 2
    buckets = {0: [], 1: []}
    tries = 0
    while (len(buckets[0]) < per_class or len(buckets[1]) < per_class):
        text, label = item_fn(rng, level)
        tries += 1
        if text in seen or len(buckets[label]) >= per_class:
            if tries > n * 200:
                break  # exhausted template space at this level; take what we have
            continue
        seen.add(text)
        buckets[label].append(Example(text, label))
    out = buckets[0] + buckets[1]
    rng.shuffle(out)
    return out


def _synthetic_task(name: str, instruction: str, item_fn,
                    n_train: int, n_eval: int, level: int, seed: int) -> Task:
    rng = np.random.RandomState(seed)
    seen: set = set()
    eval_ex = _generate(item_fn, n_eval, rng, level, seen)
    train_ex = _generate(item_fn, n_train, rng, level, seen)
    return Task(name=name, instruction=instruction, labels=list(YESNO),
                train=train_ex, eval=eval_ex, family="synthetic")


def arithmetic_task(n_train: int = 1200, n_eval: int = 600,
                    level: int = 2, seed: int = 0) -> Task:
    return _synthetic_task(
        "arithmetic_tf",
        "Decide whether the arithmetic statement is correct.",
        _arith_item, n_train, n_eval, level, seed)


def grammar_task(n_train: int = 1200, n_eval: int = 600,
                 level: int = 1, seed: int = 0) -> Task:
    return _synthetic_task(
        "grammar",
        "Decide whether the sentence is grammatically correct.",
        _grammar_item, n_train, n_eval, level, seed)


def code_output_task(n_train: int = 1200, n_eval: int = 600,
                     level: int = 1, seed: int = 0) -> Task:
    return _synthetic_task(
        "code_output",
        "Decide whether the stated output of the Python snippet is correct.",
        _code_item, n_train, n_eval, level, seed)


SYNTHETIC_BUILDERS = {"arithmetic_tf": arithmetic_task, "grammar": grammar_task,
                      "code_output": code_output_task}
