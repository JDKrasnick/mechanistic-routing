# Heterogeneous Credit Assignment via a Candidate-Level Counterfactual Critic

Baseline implementation for the claim:

> A candidate-level counterfactual critic predicts the held-out marginal utility,
> cost, and interaction effects of proposed textual, activation-level, and
> parameter-level interventions, enabling budget-constrained heterogeneous
> adaptation without exhaustively executing every candidate update.

Model: `Qwen/Qwen2.5-3B-Instruct`, bf16 on Apple Silicon MPS.

## Setup

```bash
python3.11 -m venv ~/hca-venv
~/hca-venv/bin/pip install torch==2.5.1 transformers==4.57.1 peft accelerate \
    datasets scikit-learn matplotlib pandas
```

## Pipeline

```bash
python experiments/verify_apply_revert.py     # gate: apply/revert must be exact
python experiments/calibrate.py               # find effective intervention magnitudes
python experiments/milestone1.py              # single-state loop over all 3 types
python experiments/collect_oracle_data.py     # exhaustive counterfactual dataset
python experiments/train_critic.py            # regression + pairwise ranking critic
python experiments/evaluate_routing.py        # 7 routing policies, matched budgets
python experiments/plot_results.py            # figures
```

## Design decisions that matter

**Evaluation is a single forward pass per example.** Scores come from the logits
of the label words' first tokens at the final position, not from generation.
Label first-tokens are verified collision-free. This is what makes several
hundred counterfactual evaluations tractable on a laptop GPU.

**Raw completion prompts, not chat templates.** Measured on 200 validation
examples: raw completion reached 0.605 accuracy at 68 tokens versus 0.595 at 94
tokens for a chat template with a system message, and was better calibrated
(gold log-prob -2.79 vs -3.66). It wins on accuracy, cost, and calibration.

**`propose()` is cheap, `apply()` is expensive.** A candidate is a
*specification* until applied; realizing it may train a LoRA or fit a steering
vector. This split is what allows the critic to rank candidates without paying
to execute them, and it is the source of the efficiency claim.

**Apply/revert is exact by construction.** LoRA is a hand-rolled module swap
holding a reference to the original `nn.Linear`; nothing is ever merged into
base weights. Steering is a forward hook over an accumulated vector. Reverting
restores the original module reference, so repeated apply/revert cycles cannot
drift. `verify_apply_revert.py` asserts bitwise-identical logits after revert
and after stacked state restoration.

**Interventions are tracked on a stack.** Candidates must be applied through
`system.apply_intervention()`, never `candidate.apply()` directly, or
`restore_state` cannot unwind them. `revert_all` asserts that no hooks or
adapter modules leaked.

**The critic never sees post-execution quantities.** `critic/features.py` reads
an explicit allowlist. Fitted steering-vector norms, final LoRA training loss,
and every `post_*` measurement are recorded in the dataset for analysis but are
forbidden as features, since they would leak the outcome being predicted.

**States are built by a random walk.** Collection covers systems that have
already been adapted 0-2 times, not just the pristine model. This is what makes
routing state-dependent and exposes interaction effects between types.

**Evaluation split hygiene.** Failure batches driving candidate generation come
from the *train* split; all reported scores come from *validation*. Routing
episodes use held-out seeds disjoint from the critic's training states, and a
larger fixed final-evaluation batch that no policy optimizes against.

## Environment notes

- SDPA attention miscompiles on MPS in torch 2.5.1 during cached decoding
  (`mps_matmul` shape error). The wrapper forces `attn_implementation="eager"`,
  which measured identical prefill throughput at these sequence lengths.
- Steering vectors must keep the raw difference-in-means scale. Unit-normalizing
  them made steering a no-op, since the residual stream norm is ~60-155 while a
  unit vector contributes ~1.
