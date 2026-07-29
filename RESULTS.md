# Results — Heterogeneous Credit Assignment Baseline

Qwen2.5-3B-Instruct · emotion (target) → ag_news (retention) · 30 collection
states, 270 counterfactual records, 4 routing episodes × 3 rounds × 7 policies.

The claim decomposes into four predictions. Three are supported; one is
supported with an important caveat about *what* the critic is doing.

## 1. Heterogeneity — SUPPORTED

Oracle winner share across 30 states: **prompt 40% · activation 30% · lora 30%**.
No type dominates, so routing is a real decision. (Panel B.)

Every type is net-negative on average and mostly harmful per candidate:

| type | mean utility | P(utility>0) | best |
|------|-------------:|-------------:|-----:|
| prompt | −0.080 | 0.23 | +0.175 |
| activation | −0.075 | 0.27 | +0.350 |
| lora | −0.121 | 0.22 | +0.275 |

**~75% of all candidates hurt.** This task is primarily about *avoiding damage*,
not finding wins — which shapes how the critic result should be read (see §2).

## 2. Learnability — SUPPORTED

Grouped 5-fold CV (GroupKFold by state), GBM head:

| predictor | pairwise acc | Spearman | top-1 regret |
|-----------|-------------:|---------:|-------------:|
| random | 0.505 | +0.03 | 0.149 |
| static type prior | **0.478** | −0.05 | 0.137 |
| **critic** | **0.681** | **+0.51** | **0.081** |

The static type prior scores **below chance** — the sharpest confirmation of §1.
Since no type is reliably better, "always pick the historically-best type" is
actively misleading, so the critic's skill cannot be coming from type identity;
it is reading state × candidate structure. Top features are mechanistically
sensible: LoRA learning rate (the dominant knob in calibration), cost×depth,
baseline accuracy, activation strength. (Panels C, D, I.)

Caveat: at 40-example eval batches, one flipped label = 0.025 utility. Paired
evaluation cuts variance on the difference, but this noise floor caps achievable
Spearman. 0.51 is good for a laptop-scale baseline, not a ceiling.

## 3. Efficiency — SUPPORTED

Routing under matched per-round budget (12.0), 4 episodes × 3 rounds:

| policy | task gain | final retention | exploration cost |
|--------|----------:|----------------:|-----------------:|
| prompt_only | +0.031 | 0.478 | 3.0 |
| activation_only | +0.091 | 0.494 | 6.0 |
| lora_only | +0.069 | 0.553 | 20.4 |
| random | +0.103 | 0.538 | 12.1 |
| heuristic | +0.050 | 0.556 | 6.0 |
| **critic** | **+0.112** | **0.669** | **5.2** |
| oracle | +0.144 | 0.688 | 121.2 |

**The critic captures 78% of oracle task-gain at 4% of its exploration cost —
23× cheaper.** It also beats every single-type ablation, random, and the
heuristic on task gain, and is second only to the oracle on retention.
(Panels E, F.)

## 4. Retention — SUPPORTED (with a measurement caveat)

Critic final retention 0.669 vs random 0.538 and single-type 0.48–0.55; only the
oracle is higher (0.688). The utility penalty on retention loss steers selection
away from catastrophic candidates. (Panel G.)

**Caveat — retention loss is not homogeneous across types.** The retention task
is rendered with the system's *current* instruction, so a prompt edit tuned for
emotion is also applied to news classification. Thus prompt "retention loss" is
partly task-specialization, activation/LoRA loss is genuine capability damage.
This is a deliberate single-shared-prompt design (prompt specialization is a
real cost when one prompt serves the whole system), but the number should not be
read as one clean quantity.

## Honest scope

- **One task pair, one model.** Nothing here shows the routing policy transfers
  across task families or scales beyond 3B.
- **Interaction effects are only weakly probed.** States carry 0–2 prior
  interventions — enough to make routing state-dependent and to surface two
  composition bugs (stacked LoRA, same-layer steering), but a real interaction
  study needs far more states than 30. Step 8 of the plan (explicit interaction
  + uncertainty modeling) is scaffolded (uncertainty head exists) but not
  evaluated at this scale.
- **Defensible one-line claim:** *on one task pair, a candidate-level critic
  predicts held-out intervention utility well enough to route heterogeneous
  updates at ~1/23 the execution cost of exhaustive search, recovering 78% of
  oracle gain and most of its retention protection — largely by declining
  harmful candidates that dominate the pool.*

## Figures

`results/results.png` (9-panel) and `results/fig_*.png` (individual).
Metrics: `results/critic_metrics.json`, `results/routing_results.json`.
