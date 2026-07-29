# Heterogeneous Credit Assignment in Variable LLM Systems
## Research Plan & Requirements

Status: **baseline complete and verified**; this document defines the path from
baseline to a defensible final result.

---

## 1. Big-picture research

### 1.1 The problem

When a deployed LLM system underperforms, it can be repaired at three different
levels of the stack:

| Level | Intervention | Cost | Reversible |
|-------|--------------|------|-----------|
| Textual | rewrite the prompt / instruction | cheap | trivially |
| Activation | steering vector via forward hook | medium | trivially |
| Parameter | LoRA / adapter update | expensive | with bookkeeping |

These are *heterogeneous* interventions: different mechanisms, different costs,
different failure modes, different blast radius on unrelated capabilities. The
core difficulty is **credit assignment** — given a failing state, which
intervention (and at which level) deserves the "credit" for fixing it, and by
how much, *before* paying to execute it?

Exhaustively running every candidate and keeping the best is the obvious
solution and the wrong one: it scales linearly in candidates × cost, and most
candidates are useless or harmful. The research bet is that the outcome of an
intervention is **predictable from cheap features of the state and the candidate
specification**, so the expensive execution can be routed rather than exhausted.

### 1.2 The claim under test

> A candidate-level counterfactual critic predicts the held-out marginal utility,
> cost, and interaction effects of proposed textual, activation-level, and
> parameter-level interventions — enabling budget-constrained heterogeneous
> adaptation without exhaustively executing every candidate update.

### 1.3 Decomposition into falsifiable predictions

| # | Prediction | Fails if | Metric |
|---|-----------|----------|--------|
| P1 | **Heterogeneity** — best type is state-dependent | one type always wins | oracle winner share by type |
| P2 | **Learnability** — utility predictable pre-execution | critic ≤ static type prior | within-state pairwise acc, top-1 regret |
| P3 | **Efficiency** — near-oracle utility at low cost | critic ≈ random, or ≈ oracle cost | task gain vs exploration cost |
| P4 | **Retention** — avoids capability damage | best-utility picks still forget | Δ retention accuracy |
| P5 | **Interaction** — composed utility predictable | stack utility ≠ f(marginals) | composed vs sum-of-marginals error |
| P6 | **Transfer** — policy generalizes across tasks | critic collapses on unseen task | train-A / test-B pairwise acc |

P1–P4 are addressed by the baseline. **P5 and P6 are the substance of final
testing** and are currently unmeasured.

### 1.4 Baseline result (one task pair, 30 states)

- P1 supported: winners split prompt 40 / activation 30 / lora 30.
- P2 supported: critic pairwise 0.68 vs type-prior 0.48 (below chance), Spearman +0.51.
- P3 supported: 78% of oracle task-gain at 4% of exploration cost (23× cheaper).
- P4 supported (with caveat): critic retention 0.669 vs random 0.538.
- Key nuance: ~75% of candidates are net-harmful, so much of the critic's value
  is *declining damage*, not spotting the single best win.

Full numbers in `RESULTS.md`.

---

## 2. System architecture

The baseline already implements all twelve planned components. Final testing
extends three of them and leaves the rest intact.

```
project/
├── interventions/      # [FINAL-READY] unified API; bit-exact apply/revert; composition-tested
│   ├── base.py         #   propose (cheap spec) / apply (expensive realize) / revert / cost / serialize
│   ├── prompt.py       #   instruction rewrite
│   ├── activation.py   #   contrastive steering vector, forward hook
│   └── lora.py         #   temporary adapter, never merged
├── model_wrapper.py    # [FINAL-READY] inference, loss, hidden states, hooks, adapters, state capture/restore
├── evaluation/         # [EXTEND] add task suite + split retention metric
│   ├── tasks.py        #   -> multi-task registry
│   ├── evaluator.py    #   single-forward-pass label scoring
│   ├── retention.py    #   -> separate specialization vs capability loss
│   └── costs.py        #   normalized cost model
├── candidates/         # [FINAL-READY] one generator per type, calibrated grids
├── counterfactuals/    # [EXTEND] scale states, deepen stacks, add interaction records
│   ├── runner.py       #   clone / apply / eval / restore
│   ├── oracle.py       #   exhaustive best under budget = upper bound
│   └── dataset.py      #   append-only JSONL, resumable
├── critic/             # [EXTEND] interaction targets, uncertainty selection, transfer splits
│   ├── features.py     #   pre-execution allowlist (leak-guarded)
│   ├── model.py        #   regression + pairwise ranking + uncertainty heads
│   ├── train.py        #   grouped CV
│   └── selector.py     #   budgeted policies
└── experiments/        # [EXTEND] budget sweep, seed replication, transfer eval
```

### 2.1 Load-bearing invariants (do not break)

- **propose() is cheap, apply() is expensive.** The entire efficiency argument
  depends on the critic ranking specifications without realizing them.
- **apply/revert is bit-exact** including stacked and same-target composition.
  Enforced by `verify_apply_revert.py` and `verify_composition.py`; both must
  stay green after any change.
- **The critic never sees post-execution quantities.** Features come from an
  explicit allowlist in `critic/features.py`; fitted norms, final training loss,
  and every `post_*` field are forbidden.
- **Split hygiene.** Candidate generation uses the train split; all reported
  scores use validation; routing uses seeds disjoint from critic training.

---

## 3. The components for final testing

### 3.1 Data — the intervention corpus

| | Baseline | Final requirement |
|---|---|---|
| task pairs | 1 | 3–5 across families |
| states / task | 30 | 150–300 |
| stack depth | 0–2 | 0–4 (interaction coverage) |
| records | 270 | ~5,000–15,000 |
| purpose | proof of life | statistical power + interaction estimability |

This is the dominant cost and the dominant driver of result quality. Everything
downstream is bottlenecked by it.

### 3.2 Critic

- **Interaction head (P5):** predict the utility of an already-stacked state's
  next candidate, and separately predict composed-vs-marginal gap. Requires the
  deeper-stack data from 3.1.
- **Uncertainty-aware selection:** the residual-variance head already exists and
  is unused. Wire `predicted utility − λ·std` into the selector and measure
  whether it reduces harmful picks.
- **Transfer protocol (P6):** train on tasks {A,B,C}, evaluate on held-out D. The
  feature set is already task-agnostic (no task identity leaks in), so this is a
  harness change, not a model change.

### 3.3 Tasks & model

- Multi-task registry with per-task instruction, labels, retention probe.
- Retention split into two numbers: *task-specialization loss* (retention scored
  with a fixed neutral instruction) vs *capability loss* (weight/activation
  damage). Prevents P4 from measuring two things at once.
- Optional second base model (e.g. a 7–8B) as one extra point to show results
  aren't 3B-specific.

### 3.4 Oracle & baselines

- Oracle recomputed per task and per budget (it is the upper bound and the cost
  denominator).
- Baseline policy set is already correct: prompt-only, activation-only,
  lora-only, random, heuristic (static type prior), critic, oracle.

### 3.5 Experiment harness

- **Budget sweep:** replace the single budget (12.0) with a curve. The efficiency
  claim is far stronger as task-gain-vs-cost across budgets than as one point.
- **Seed replication:** ≥10 seeds (baseline has 4) to put confidence intervals on
  every headline comparison.
- **Transfer runs:** the held-out-task evaluation from 3.2.

### 3.6 Metrics & statistics

- Confidence intervals / significance across seeds on: pairwise acc, top-1
  regret, task gain, retention.
- Report the noise floor explicitly (at N-example eval batches, one flipped
  label = 1/N utility) so Spearman ceilings are interpretable.
- Interaction: mean |composed − predicted| and whether the interaction head beats
  a sum-of-marginals baseline.

---

## 4. Requirements

### 4.1 Functional requirements

- [ ] Multi-task registry; ≥3 target tasks + retention probes.
- [ ] Collection scales to 150–300 states/task and stack depth up to 4.
- [ ] Interaction records: utility of candidate *given* an existing stack.
- [ ] Critic transfer split (train-A / test-B) with task-agnostic features.
- [ ] Uncertainty-aware selector path (λ·std) exercised and reported.
- [ ] Retention decomposed into specialization vs capability loss.
- [ ] Budget-sweep routing harness (curve, not point).
- [ ] ≥10-seed replication with CIs on all headline metrics.

### 4.2 Invariants that must remain true

- [ ] `verify_apply_revert.py` and `verify_composition.py` pass bit-exact.
- [ ] No post-execution feature reaches the critic (allowlist enforced).
- [ ] Candidate/gen on train split, scoring on validation, routing seeds disjoint.
- [ ] propose() performs no model execution.

### 4.3 Compute / infra

- Collection is the bottleneck: ~4 min/state on M4 MPS × (150–300 states × 3–5
  tasks) ≈ 30–120 GPU-hours. Feasible on the laptop only overnight-serially;
  realistically wants a CUDA box or parallel collection.
- LoRA training is ~4–5 s/step and dominates per-candidate cost; batch and cache
  aggressively (already cached across state restoration).
- MPS-specific: `attn_implementation="eager"` is required (SDPA miscompiles on
  cached decode in torch 2.5.1). Verify this is unnecessary on the target box.
- Storage trivial (JSONL, ~15 KB/state).

### 4.4 Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Eval noise caps Spearman | P2 looks weak | larger eval batches; regress on low-variance margin/logprob |
| Too few seeds | claims not significant | ≥10 seeds, report CIs |
| Interaction data too sparse | P5 untestable | prioritize depth-3/4 states in collection schedule |
| Retention metric conflates two effects | P4 ambiguous | split specialization vs capability loss |
| Single model/task family | no generalization claim | multi-task + second model |
| Collection cost | timeline slips | parallelize / move off laptop |

---

## 5. Milestones

1. **M1 — Baseline** ✅ done. One task pair, P1–P4 supported, pipeline verified.
2. **M2 — Multi-task collection.** Task registry + scaled collection (3 tasks ×
   ~150 states). Unlocks statistical power.
3. **M3 — Transfer (P6).** Train-A/test-B critic; the central generalization test.
4. **M4 — Interaction (P5).** Deep-stack data + interaction head + composed-utility
   metric.
5. **M5 — Efficiency curve & stats.** Budget sweep, ≥10 seeds, CIs on all claims.
6. **M6 — Writeup.** Consolidated results across P1–P6 with honest scope.

**The one component that converts "promising baseline" into "final result": a
held-out task the critic never trained on (M3).** Everything else is scale;
transfer is the only test of whether the central claim is real or fit to one
distribution.

---

## 6. Defensible claim ladder

- **Now (baseline):** *On one task pair, a candidate-level critic predicts held-out
  intervention utility well enough to route heterogeneous updates at ~1/23 the
  execution cost of exhaustive search, recovering 78% of oracle gain — largely by
  declining harmful candidates.*
- **After M3–M5:** add *…and this transfers to unseen tasks, holds across budgets
  with statistical significance, and predicts interaction effects between stacked
  interventions.* — which is the full project claim.
