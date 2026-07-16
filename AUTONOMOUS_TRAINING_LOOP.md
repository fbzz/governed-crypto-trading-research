# Autonomous Research Loop — V47 to V49

## Objective

Run the next research sequence without further design choices until one frozen
joint absolute/relative model family has been fully trained and audited on
non-target assets.

```text
V47 metadata-only specification
  -> V48 deterministic synthetic harness
  -> V49 preflight
  -> V49 one-job MPS smoke
  -> V49 complete MPS training
  -> V49 read-only verification
  -> STOP
```

The loop succeeds only when all 36 registered V49 checkpoints exist, reopen
semantically, pass their hashes and data-access audits, and no resume artifact
remains. V50 evaluation is outside this loop.

## Scientific boundary

- No BTC, ETH, or SOL observation may be loaded.
- No held-out-asset outcome may be loaded.
- No 2025 or 2026-H1 outcome may be used by V47-V49.
- Ephemeral train/validation forward outputs and the registered objective losses
  are allowed. Held-out/deployment prediction artifacts, prediction-quality
  evaluation, portfolios, PnL, Sharpe, drawdown, bootstrap, and model-comparison
  metrics are forbidden.
- V45 remains retired. Its weights, scalers, policy, thresholds, and consumed
  outcome packet are not reusable.
- V35, V36, V43, and V44 checkpoints are forbidden parents. Source-code
  patterns may be reused, but V49 weights start fresh.
- A training loss is an engineering diagnostic, not evidence of alpha.
- A failed gate stops the loop. It does not authorize parameter, threshold,
  batch-size, split, or policy changes.

## Frozen V47 design

| Item | Contract |
|---|---|
| Family | `tlm_joint_absolute_relative_triplet_medium_v1` |
| Input | `float32 [batch, 256 days, 3 assets, 9 features]` |
| Decision unit | One exact triplet; no cross-context asset averaging |
| Backbone | Patch 16/stride 8, width 128, four temporal layers, two cross-asset layers, four attention heads, FFN 512, dropout 0.20 |
| Symmetry | Shared asset encoder, no asset-slot embedding, permutation equivariant |
| Outputs | Per-asset `excess_score_z` and `market_component_z` |
| Removed | Quantiles, volatility head, mask token, reconstruction head, PnL loss |
| Parameter ceiling | 1,212,930 expected; strictly below V41's 1,231,634 |
| Seeds | `[42, 7, 123]`, all retained |
| Initialization | Fresh registered seed; no checkpoint reuse |

For every eligible training triplet and signal date:

```text
r_i = log(open[t+2] / open[t+1])
m   = mean_i(r_i)
e_i = r_i - m
sigma = sqrt(mean(r_i^2)) over the complete train-only enumeration
z_r = r / sigma
z_m = m / sigma
z_e = e / sigma
```

`sigma` is fold/origin/geometry-local, fitted only on its 20 train assets and
train dates, and floored at `1e-6`.

Model reconstruction is deterministic:

```text
e_hat_z_i  = excess_score_z_i - mean_i(excess_score_z_i)
m_hat_z    = mean_i(market_component_z_i)
mu_hat_z_i = m_hat_z + e_hat_z_i
mu_hat_i   = sigma * mu_hat_z_i
```

The single registered objective is:

```text
L_rank   = RankNet over pairs (0,1), (0,2), (1,2)
L_excess = SmoothL1(e_hat_z, z_e, beta=1)
L_level  = 0.5 * SmoothL1(m_hat_z, z_m, beta=1)
         + 0.5 * mean_i SmoothL1(mu_hat_z_i, z_r_i, beta=1)
L_total  = L_rank + L_excess + L_level
```

Return ties within `1e-12` are excluded from RankNet. There is no clipping,
outcome weighting, calibration, architecture tournament, or hyperparameter
search. Early stopping monitors exactly `L_total`.

Within each job, the first completed epoch initializes the best state. A later
epoch replaces it only when finite validation `L_total` is strictly lower
(`min_delta=0`). Equal loss is a non-improvement. Five consecutive
non-improvements stop training; the maximum is 30 epochs. The persisted final
checkpoint restores that job's best validation state. Selecting between seeds,
folds, origins, or geometries is forbidden; restoring the preregistered best
epoch inside every job is not cross-job model selection.

## Frozen policy for later evaluation

V47 freezes the policy so training cannot be followed by an outcome-driven
policy choice. V47-V49 must not execute it on real data.

- Actions: cash or one eligible long asset.
- Risky gross weight: exactly `1/3`; the remaining `2/3` stays in cash.
- Desired risky asset: the largest predicted excess, with lexical final ties.
- Base decision cost: 10 bps per unit turnover.
- Candidate actions at a decision are cash, the eligible incumbent, and the
  desired challenger.
- Action value is predicted one-day portfolio return minus the exact base-cost
  turnover from the current weights.
- Exact utility ties prefer the current action, then cash, then lexical symbol.
- Ineligible or non-finite assets cannot receive weight.
- Entry, switch, exit, forced ineligibility, and final liquidation are charged.
- Positions produced at 10 bps remain unchanged when V50 later reports
  10/20/30/50 bps accounting.
- Cash, Ridge, momentum, and equal-weight controls must later use the same
  `1/3` gross budget.

The `1/3` sleeve is the ex-ante concentration rule. No momentum gate or V46-
derived filter is allowed.

For current risky-weight vector `w_current` and action vector `w_a`, the exact
utility is:

```text
U(a) = dot(w_a, mu_hat) - 0.001 * sum_i(abs(w_a_i - w_current_i))
```

Cash is the all-zero vector. A risky action has exactly one `1/3` component.
Forced incumbent ineligibility removes the hold action but does not waive exit
or replacement turnover.

## Frozen V49 training grid

V49 contains two chronological origins and two geometries. None may be chosen
or discarded after training.

| Origin | Geometry | Supervised train signals | Train-asset early stop |
|---|---|---|---|
| `origin_2024` | expanding | 2021-03-01 through 2022-12-23 | 2023-01-01 through 2023-12-23 |
| `origin_2024` | rolling | 2022-01-01 through 2022-12-23 | 2023-01-01 through 2023-12-23 |
| `origin_2025` | expanding | 2021-03-01 through 2023-12-23 | 2024-01-01 through 2024-12-23 |
| `origin_2025` | rolling | 2023-01-01 through 2023-12-23 | 2024-01-01 through 2024-12-23 |

The conservative eight-day year-end purge is retained. Feature scalers and
return scales are unique to each origin, geometry, and asset fold. Each cell
trains all three asset folds and all three seeds:

```text
2 origins x 2 geometries x 3 folds x 3 seeds = 36 checkpoints
```

The later primary ensemble is predesignated as the nine
`origin_2025/expanding` checkpoints. The other 27 checkpoints are mandatory
stability evidence for V50 and are never a model-selection pool.

### Exact V49 data and sampling contract

- Inputs are the hash-exact V32 panel, sequence index, feature schema, asset
  folds, and lexical triplet catalog. V47 freezes all five hashes.
- Project only `date`, `symbol`, the eight frozen base features, and, for label
  reads, `target_window_end_date` plus
  `target_next_open_to_next_open_log_return`.
- Do not project or read `target_realized_volatility_7d` or any other target.
- Reuse the V32 `supervised_sequence_ready` rule. A sample exists only when all
  three distinct train assets have a finite contiguous 256-day sequence and a
  mature permitted return label. Missing values are never imputed.
- Materialize only the feature rows referenced by permitted train or
  train-asset validation sequences, including their causal lookback prefixes.
  A lookback may precede the signal interval; it may never extend past its
  signal date. Held-out-asset features and labels remain forbidden.
- Fit the eight-feature scaler on all finite rows of the cell's 20 train assets
  whose dates fall inside that cell's supervised-train interval. Validation
  rows and held-out assets never fit it.
- Compute the ninth feature from raw same-date
  `log_close_to_close_return - triplet_mean`, then divide it by the fitted scale
  of `log_close_to_close_return`, matching the frozen V32 tensor semantics.
- Fit `sigma` by enumerating every eligible train date/triplet and all three raw
  action returns. It is shared across the three seeds in that
  origin/geometry/fold cell.
- Training draws 8,192 date/triplet pairs uniformly with replacement per epoch.
  Validation uses 2,048 fixed draws with sampling epoch zero.
- The RNG seed for each sampler is the unsigned integer encoded by the first
  eight bytes of canonical SHA-256 over
  `(20260713, version, origin, geometry, fold, seed, epoch, role)`.
- Sampler order, lexical triplet order, batch order, and validation samples are
  persisted by hash. No seed may share another seed's sampled validation set
  unless the canonical key produces it exactly.

## Frozen V50/V51 contract — not executed by this loop

Freezing evaluation before training prevents a new gate or refit from being
chosen after seeing V49 losses. The autonomous loop still stops before V50.

- `origin_2024` is evaluated only on its fold's held-out assets from
  2024-01-01 through 2024-12-23. `origin_2025` is evaluated only from
  2025-01-01 through 2025-12-23. Both are adaptive historical development,
  never clean confirmation.
- Each eligible lexical held-out triplet is an independent pseudo-deployment.
  Average the three seed predictions only within the same
  origin/geometry/fold/triplet. Never average one asset across different
  triplet contexts.
- Apply the frozen policy independently to each triplet. Equal-weight triplet
  portfolio returns within fold/date, then equal-weight the three folds. Keep
  every origin and geometry separate; no aggregate may rescue a failed cell.
- Cash is the all-zero control.
- Shared Ridge uses alpha 10 with no tuning. It fits one asset-shared linear
  mapping from flattened `256 x 9` inputs to normalized absolute return on the
  identical train samples; triplet market/excess components are derived by
  mean/centering its three outputs.
- Dual momentum selects the largest positive causal 30-day close-to-close log
  return, otherwise cash. Equal weight holds every eligible triplet member.
  Both use total gross `1/3`, no leverage or shorting, and the same accounting.
- Candidate positions are fixed at the 10 bps policy cost. Report candidate
  and controls at 10/20/30 bps; 50 bps, one-day extra signal delay, and missing-
  asset removal are diagnostic stresses only.
- Mandatory metrics per origin/geometry/fold are Spearman, pairwise accuracy,
  top-1 excess, total net return, daily Sharpe, max drawdown, turnover, and
  cost. Candidate Spearman, top-1 excess, and 10 bps net return must be strictly
  positive and pairwise accuracy must exceed 0.50 in every cell. Within each
  origin/geometry aggregate, candidate Spearman and top-1 excess must also
  exceed Ridge.
- At aggregate level the candidate must beat Ridge, dual momentum, and equal
  weight in total return at 10/20/30 bps; beat dual-momentum Sharpe at all three
  costs; keep absolute drawdown at or below 35% and within five percentage
  points of dual momentum; and keep 10 bps turnover at or below dual momentum.
- Run 10,000 paired circular-block paths at 7/21/63 days with identical indices
  for candidate and controls. The fifth percentile of absolute candidate net
  return and every candidate-minus-control return must be strictly above zero.
- Daily gross simple return is `sum_i(w_i * (exp(r_i) - 1))`; daily net return
  subtracts `turnover * bps / 10000`. Total return is compounded, Sharpe is
  `sqrt(365) * mean(net) / sample_std(net)`, and drawdown is computed from the
  compounded net-equity curve. Final liquidation is included.
- Any mandatory failure retires the family without tuning.
- V51 performs no refit. If and only if V50 passes unchanged, V51 packages and
  registers the already-predesignated nine `origin_2025/expanding` checkpoints,
  their scalers, return scales, code, policy, controls, and evaluator hashes.

## V47 — Metadata-only specification

### Required implementation

```text
configs/v47_joint_absolute_relative_triplet_spec.yaml
src/tlm/joint_absolute_relative_spec.py
tests/test_joint_absolute_relative_spec.py
artifacts/v47_joint_absolute_relative_triplet_spec/
```

Add `make run-v47` and the matching CLI command. The implementation may hash
and read only the frozen V32 metadata, V45 retirement metadata, and V46
diagnostic metadata explicitly allowlisted by the config. It may calculate the
parameter count analytically, but may not instantiate a model or open a panel,
label, checkpoint, prediction, or outcome table.

### V47 gate

- Exact input hashes and retired lineage pass.
- Family identity, architecture, equations, loss, data roles, 36-job grid,
  policy, later controls, later gates, and lifecycle are hash-locked.
- All forbidden operation counters are zero.
- Two runs reproduce byte-identical artifacts.
- Final decision is exactly
  `authorize_v48_joint_absolute_relative_synthetic_harness_only`.

After a pass, update `AGENTS.md`, `TASKS.md`, and `README.md`, commit V47, then
continue to V48. A V47 failure ends the autonomous loop.

## V48 — Deterministic synthetic harness

### Required implementation

```text
configs/v48_joint_absolute_relative_harness.yaml
src/tlm/joint_absolute_relative_model.py
src/tlm/joint_absolute_relative_harness.py
tests/test_joint_absolute_relative_harness.py
artifacts/v48_joint_absolute_relative_harness/
```

Add `make run-v48` and the matching CLI command. V48 may read only the exact
V47 specification packet and synthetic fixtures.

### Mandatory checks

- Exact `[B,256,3,9]` dtype/shape and exact parameter count.
- Causal-prefix invariance.
- Permutation equivariance of per-asset heads and invariance of `m_hat_z`.
- `sum(e_hat_z)=0`, `mean(mu_hat_z)=m_hat_z`, and exact `mu=m+e` replay.
- Train-only scale fit, zero-variance floor, and validation-outlier isolation.
- RankNet ordering/tie behavior and finite loss/gradients.
- Correct trainable parameter scope for both heads and the entire backbone.
- Cash, entry, hold, rejected switch, accepted switch, exit, forced exit,
  missing asset, non-finite input, and deterministic tie fixtures.
- Turnover in `1/3` units: entry/exit `1/3`, switch `2/3`, hold `0`.
- Gross cap, cash remainder, risk-matched controls, monotonic reporting costs,
  and final liquidation.
- Checkpoint/model/optimizer/early-stop/CPU-RNG roundtrip and the registered
  MPS-RNG payload schema; the real MPS resume is proven by the V49 smoke.
- Interrupted synthetic epoch plus resume equals uninterrupted training.
- Two independent executions produce byte-identical artifacts.

### V48 gate

Every check passes, every forbidden operation counter is zero, no resume file
remains, and the final decision is exactly
`authorize_v49_purged_non_target_training_only`.

After a pass, update the contracts, commit V48, then continue to V49. A V48
failure ends the autonomous loop.

## V49 — Real non-target training

### Required implementation

```text
configs/v49_joint_absolute_relative_training.yaml
src/tlm/joint_absolute_relative_training.py
tests/test_joint_absolute_relative_training.py
artifacts/v49_joint_absolute_relative_training_preflight/
artifacts/v49_joint_absolute_relative_training_smoke/
artifacts/v49_joint_absolute_relative_training/
data/checkpoints/v49_joint_absolute_relative_training/
```

Add these commands:

```text
make preflight-v49
make smoke-v49
make run-v49
make verify-v49
```

Commit the V49 source, config, tests, and clean Git receipt before the first
real-label smoke. Do not modify source or config while the full run is active.

### Runtime contract

```yaml
device: mps
dtype: float32
amp: false
deterministic_algorithms: true
cpu_fallback_allowed: false
torch_threads: 10
train_samples_per_epoch: 8192
validation_samples: 2048
batch_size: 128
maximum_epochs: 30
early_stopping_patience: 5
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.0001
gradient_clip_norm: 1.0
```

Jobs run serially in lexical grid order. The smoke runs only
`origin_2024/expanding/fold_1/seed_42` with 128 train samples, 128 validation
samples, batch 32, two epochs, and patience two. Smoke outputs are isolated and
can never become full-run parents.

MPS execution must set `PYTORCH_ENABLE_MPS_FALLBACK=0`. If the sandbox hides
MPS, request host-execution approval if required rather than silently using
CPU. `caffeinate` may wrap the foreground full run to prevent macOS sleep.

### Resume and concurrency

- Use one process-level lock for the full V49 tree.
- Atomically replace `progress.json`, `complete.json`, and resume/checkpoint
  files.
- Persist only at completed epoch boundaries: current/best model, AdamW,
  exact step count, CPU/MPS RNG, early-stop state, history, scales, and all
  source/config/job hashes.
- Re-running `make run-v49` verifies and skips completed jobs, then resumes the
  first incomplete job from its last complete epoch.
- A partial epoch repeats from the previous epoch boundary.
- If `complete.json` exists with a stale resume, verify completion first and
  remove only the matching stale resume.
- Never run two seeds concurrently on the same MPS device.

### V49 hard stops

Stop immediately on hash drift, forbidden asset/date materialization, maturity
failure, leakage, non-finite values, parameter/head/scope mismatch, corrupt
resume, checkpoint roundtrip failure, missing grid member, or any attempt to
persist held-out/deployment predictions or compute predictive/economic
evaluation. Ephemeral train/validation outputs required for `L_total` remain
allowed.

A process interruption, MPS out-of-memory, or 20-minute no-progress stall may
receive one exact resume after competing GPU work is closed. Do not change the
batch size, enable AMP/CPU fallback, delete state, or drop a job. The same
environmental failure twice is a hard stop.

### V49 completion gate

- Metadata-only preflight passes with zero Parquet deserializations and zero
  optimizer steps.
- One-job MPS smoke and interrupted-resume fixture pass.
- All 36 full jobs complete through frozen early stopping or epoch ceiling.
- All checkpoints reopen without label access and match the complete grid.
- All scaler, target-scale, model-state, optimizer-step, history, and data-
  access hashes pass.
- There are no resume files or partial jobs.
- `make verify-v49` passes.
- An idempotent second `make run-v49` performs zero optimizer steps and
  reproduces the result hash.
- Final decision is exactly
  `v49_training_complete_economic_evaluation_still_forbidden`.

The observed V44 rate suggests roughly 30–160 minutes for 36 fresh supervised
jobs, depending on early stopping. The epoch ceiling, not a wall-clock promise,
is authoritative.

## Operator command sequence

Each command becomes available only after its implementation has passed the
preceding source/test gate.

```bash
make test
make run-v47
make run-v47
make run-v48
make run-v48
make preflight-v49
PYTORCH_ENABLE_MPS_FALLBACK=0 make smoke-v49
PYTORCH_ENABLE_MPS_FALLBACK=0 make run-v49
make verify-v49
PYTORCH_ENABLE_MPS_FALLBACK=0 make run-v49
```

The second V47/V48 calls verify byte-identical replay. The second full V49 call
must perform zero optimizer steps and verify the completed tree.

## Terminal condition

On V49 success:

1. Update `AGENTS.md`, `TASKS.md`, and `README.md` with exact hashes, epochs,
   optimizer steps, and checkpoint counts.
2. Commit the V49 completion without adding ignored datasets or checkpoints.
3. Leave the working tree clean.
4. Report the commands, elapsed training summary, hashes, and limitations.
5. Stop before V50. Do not produce held-out/deployment predictions or open any
   held-out outcome.

Use `prompts/run_v47_to_v49.md` as the autonomous-agent task prompt.
