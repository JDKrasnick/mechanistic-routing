"""Vetted-benchmark ('existing') task family.

Loaders for standard, externally validated classification datasets. Each returns
a Task whose `labels` order matches the dataset's integer label indices, so the
evaluator's argmax lines up with the gold label without remapping.

- sst2    : sentiment            (negative, positive)          GLUE/SST-2
- snli    : natural language inf (entailment, neutral, contradiction)
- code_defect : defect detection (safe, vulnerable)            CodeXGLUE/Devign
- emotion : emotion              (6-way)                       dair-ai/emotion
- ag_news : news topic           (4-way)                       fancyzhx/ag_news
"""
from datasets import load_dataset

import config
from evaluation.tasks import Example, Task, _truncate

SST2_LABELS = ["negative", "positive"]
SNLI_LABELS = ["entailment", "neutral", "contradiction"]
CODE_DEFECT_LABELS = ["safe", "vulnerable"]  # index 0 = target False, 1 = True
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
AG_LABELS = ["World", "Sports", "Business", "Technology"]


def _truncate_code(code: str, max_chars: int = 600) -> str:
    """Cap a code snippet by character budget while preserving its line
    structure, so indentation/newlines the model relies on survive truncation."""
    return code[:max_chars].rstrip()


def sst2_task(n_train: int = 1200, n_eval: int = 600, seed: int = config.SEED) -> Task:
    ds = load_dataset("stanfordnlp/sst2")
    tr = ds["train"].shuffle(seed=seed)
    va = ds["validation"].shuffle(seed=seed)  # test labels are hidden (-1)
    return Task(
        name="sst2",
        instruction="Classify the sentiment of the movie review sentence.",
        labels=list(SST2_LABELS),
        train=[Example(_truncate(t, 40), l)
               for t, l in zip(tr["sentence"][:n_train], tr["label"][:n_train])],
        eval=[Example(_truncate(t, 40), l)
              for t, l in zip(va["sentence"][:n_eval], va["label"][:n_eval])],
        family="benchmark",
    )


def snli_task(n_train: int = 1200, n_eval: int = 600, seed: int = config.SEED) -> Task:
    ds = load_dataset("stanfordnlp/snli")
    tr = ds["train"].filter(lambda r: r["label"] != -1).shuffle(seed=seed)
    va = ds["validation"].filter(lambda r: r["label"] != -1).shuffle(seed=seed)

    def render_pair(p, h):
        return f"Premise: {_truncate(p, 25)}\nHypothesis: {_truncate(h, 25)}"

    return Task(
        name="snli",
        instruction="Decide the relation of the hypothesis to the premise.",
        labels=list(SNLI_LABELS),
        train=[Example(render_pair(p, h), l) for p, h, l in
               zip(tr["premise"][:n_train], tr["hypothesis"][:n_train], tr["label"][:n_train])],
        eval=[Example(render_pair(p, h), l) for p, h, l in
              zip(va["premise"][:n_eval], va["hypothesis"][:n_eval], va["label"][:n_eval])],
        family="benchmark",
    )


def code_defect_task(n_train: int = 1200, n_eval: int = 600, seed: int = config.SEED) -> Task:
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    tr = ds["train"].shuffle(seed=seed)
    va = ds["validation"].shuffle(seed=seed)
    return Task(
        name="code_defect",
        instruction="Decide whether the C function contains a security vulnerability.",
        labels=list(CODE_DEFECT_LABELS),
        train=[Example(_truncate_code(f), int(t))
               for f, t in zip(tr["func"][:n_train], tr["target"][:n_train])],
        eval=[Example(_truncate_code(f), int(t))
              for f, t in zip(va["func"][:n_eval], va["target"][:n_eval])],
        family="benchmark",
    )


def emotion_task(n_train: int = 1200, n_eval: int = 600, seed: int = config.SEED) -> Task:
    ds = load_dataset("dair-ai/emotion", "split")
    tr = ds["train"].shuffle(seed=seed)
    va = ds["validation"].shuffle(seed=seed)
    return Task(
        name="emotion",
        instruction="Classify the emotion expressed in the text.",
        labels=list(EMOTION_LABELS),
        train=[Example(_truncate(t, 40), l)
               for t, l in zip(tr["text"][:n_train], tr["label"][:n_train])],
        eval=[Example(_truncate(t, 40), l)
              for t, l in zip(va["text"][:n_eval], va["label"][:n_eval])],
        family="benchmark",
    )


def ag_news_task(n_eval: int = 600, seed: int = config.SEED) -> Task:
    ag = load_dataset("fancyzhx/ag_news")["test"].shuffle(seed=seed)
    return Task(
        name="ag_news",
        instruction="Classify the topic of the news snippet.",
        labels=list(AG_LABELS),
        eval=[Example(_truncate(t, 25), l)
              for t, l in zip(ag["text"][:n_eval], ag["label"][:n_eval])],
        family="benchmark",
    )


BENCHMARK_BUILDERS = {
    "sst2": sst2_task, "snli": snli_task, "code_defect": code_defect_task,
    "emotion": emotion_task, "ag_news": ag_news_task,
}
