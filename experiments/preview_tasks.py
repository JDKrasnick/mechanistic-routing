"""Inspect the task registry without running the model.

Prints per-task family, sizes, class balance, train/eval disjointness, and a
sample rendered prompt. Validates label first-token distinctness against the
model tokenizer when it is available (the same check `label_token_ids` enforces).

    python3 experiments/preview_tasks.py              # synthetic only (offline)
    python3 experiments/preview_tasks.py --benchmarks # also load HF benchmarks
"""
import sys, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluation.tasks import Task, REGISTRY_TARGETS, RETENTION_TASK
from evaluation.synthetic import SYNTHETIC_BUILDERS
from evaluation.benchmarks import BENCHMARK_BUILDERS


def _balance(examples, n_labels):
    c = collections.Counter(e.label for e in examples)
    return [c.get(i, 0) for i in range(n_labels)]


def _first_token_ids(labels):
    try:
        from transformers import AutoTokenizer
        import config
        tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        ids = [tok.encode(" " + l, add_special_tokens=False)[0] for l in labels]
        return ids, len(set(ids)) == len(ids)
    except Exception as e:
        return None, f"skipped ({type(e).__name__})"


def describe(task: Task):
    n = len(task.labels)
    tr_texts = {e.text for e in task.train}
    overlap = len(tr_texts & {e.text for e in task.eval})
    ids, distinct = _first_token_ids(task.labels)
    print(f"\n== {task.name}  [{task.family}]")
    print(f"   instruction : {task.instruction}")
    print(f"   labels      : {task.labels}  distinct_first_token={distinct}")
    print(f"   train/eval  : {len(task.train)} / {len(task.eval)}  "
          f"(train∩eval texts = {overlap})")
    print(f"   balance tr  : {_balance(task.train, n)}")
    print(f"   balance ev  : {_balance(task.eval, n)}")
    if task.eval:
        e = task.eval[0]
        print(f"   sample      : {task.render(e.text)!r} -> {task.labels[e.label]}")


def main():
    include_bench = "--benchmarks" in sys.argv
    print("SYNTHETIC FAMILY")
    for name, build in SYNTHETIC_BUILDERS.items():
        describe(build())

    if include_bench:
        print("\nBENCHMARK FAMILY")
        for name, build in BENCHMARK_BUILDERS.items():
            try:
                describe(build())
            except Exception as e:
                print(f"\n== {name}  [benchmark]  LOAD FAILED: {type(e).__name__}: {e}")

    print(f"\nregistry targets: {REGISTRY_TARGETS}")
    print(f"retention probe : {RETENTION_TASK}")


if __name__ == "__main__":
    main()
