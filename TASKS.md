# TLM MVP engineering loops

- [x] Loop 0 - repository contract, package scaffold, config, and CLI.
- [x] Loop 1 - public OHLCV adapter, deterministic fixture, cache, and data tests.
- [x] Loop 2 - causal features, next-day labels, sequences, and leakage tests.
- [x] Loop 3 - Ridge baseline, expanding walk-forward, costs, and metrics.
- [x] Loop 4 - compact Transformer, early stopping, and checkpoints.
- [x] Loop 5 - one-command smoke/full pipelines and experiment artifacts.
- [x] Loop 6 - leakage, accounting, reproducibility, and result review.

## Definition of done

`make test` and `make smoke` pass. A full run can obtain public Binance daily
candles without an API key and writes `predictions.parquet`, `metrics.json`,
`equity_curve.png`, and `report.md` under an isolated artifact directory.

## Current research decision

V37, V45, V50, V59, and V64 remain retired. V64 preserved positive relative
ranking information but failed economic conversion after its point state gate.
Its decision and consumed 2025 adaptive evidence are immutable.

V65 passed and froze the exact V64 ranker architecture, objective, and nine
ranker-state identities, while changing only the independent state gate to a
fixed Student-t location/scale head and abstention to a fixed 60% probability
of clearing the exact transition cost.

V66 and V67 passed the synthetic and bounded dataset gates. V68 then passed the
exact nine-job gate-only MPS training, semantic checkpoint verification, and
zero-step replay. Every frozen V63 ranker identity was preserved with zero
ranker optimizer steps; 3,776 optimizer steps trained only the fresh Student-t
gates. BTC/ETH/SOL remained sealed and no prediction, metric, PnL, or outcome
was opened.

The owner paused the V70 prospective wait as the primary path without deleting
or reclassifying its immutable artifacts. V71 completed the outcome-blind,
post-hoc replay preparation over the consumed 2025 non-target window and froze
all V64-R2 predictions, candidate/control positions, costs, accounting,
bootstrap, and gates. Its prepare packet passed and replayed with zero new
inference or outcome access.

### V72 — Hash-bound post-hoc outcome unseal

- [x] Verify the exact user authorization against evaluation spec
      `6812586005...`, prepare receipt `d6d29ff80...`, and registered contract
      `b1825271ba...` before outcome access.
- [x] Atomically write the V72 authorization receipt, then deserialize the
      immutable 2025 non-target packet exactly once with zero underlying source
      rereads.
- [x] Evaluate candidate, frozen V64 control, equal-weight, and cash at
      10/20/30 bps using unchanged turnover and final-liquidation accounting.
- [x] Run 10,000-path circular-block Monte Carlo at 7/21/63 days and preserve
      all 24 mandatory candidate gate cells, including failures.
- [x] Freeze completion receipts and prove source-free replay with matching
      result hashes.
- [x] Keep BTC/ETH/SOL sealed and label all evidence post-hoc consumed-2025,
      never clean confirmation or deployable evidence.

**V72 gate:** exactly one packet unseal, zero underlying source rereads, frozen
economic evidence, all registered cells preserved, and source-free replay.
Passing or failing the diagnostic does not change family status and authorizes
only V73 metadata recording.

V72 completed with exactly one packet unseal, zero underlying source rereads,
and matching source-free replay hashes. The candidate passed 13/24 gates and
failed the diagnostic. At 10 bps it produced `+3.18%` gross, `-2.57%` net,
Sharpe `-0.20`, `-12.44%` maximum drawdown, and `57.33` turnover. Family status
did not change; target assets remained sealed.

### V73 — Metadata-only V72 diagnostic record

- [x] Read only the four exact hash-bound V72 JSON result/audit/completion/replay
      files.
- [x] Record the failed 13/24-gate diagnostic without recomputing any metric,
      bootstrap, gate, policy, prediction, or position.
- [x] Preserve one packet unseal, zero underlying rereads, unchanged family
      status, source-free replay, and the post-hoc evidence label.
- [x] Emit a deterministic metadata packet with zero Parquet, outcome-packet,
      checkpoint, model, training, inference, or target-asset access.
- [x] Replay byte-identically and authorize at most a separate V74 new-family
      specification.

**V73 gate:** deterministic metadata-only record with all access counters at
zero except four JSON metadata reads. Stop before V74 specification.

V73 passed. Two complete invocations produced byte-identical hashes and record
SHA-256 `22e48c9534a88d6870f9b8d640b42f023d03641978dfa39638c3abc596b295a2`.
The ledger records four JSON metadata reads and zero Parquet, outcome packet,
checkpoint, model, optimizer, inference, prediction, position, or target-asset
operations. The receipt authorizes only the separate V74 specification.

### V74 — Persistent multi-horizon duration family specification

- [x] Verify the exact V73 receipt and candidate source hashes without opening
      Parquet, market panels, checkpoints, models, outcomes, or target assets.
- [x] Freeze one 1,083,155-parameter causal multi-asset Transformer with
      Student-t 1/3/7-day return heads and an explicit 1..7-day duration hazard.
- [x] Freeze cumulative open-to-open labels and the earliest-argmax duration
      target with day-7 right censoring and eight-day maturity/purge/embargo.
- [x] Freeze return NLL + `0.25` pairwise ranking + `0.50` duration NLL, with no
      PnL loss, outcome weighting, capacity sweep, or hyperparameter search.
- [x] Freeze the long-one-or-cash stateful policy whose survival-weighted
      multi-horizon utility subtracts exact L1 transition cost before switching.
- [x] Freeze the three-fold x three-seed MPS training grid and adaptive 2025
      evaluation with costs 10/20/30 bps and strict no-rescue financial gates.
- [x] Emit and replay one deterministic metadata-only packet, authorizing only
      the V75 synthetic harness when every V74 check passes.

**V74 gate:** one frozen architecture, objective, policy, training grid, and
financial decision contract; zero scientific data/model/outcome/target access.
Stop before V75 implementation or execution.

V74 passed 16/16 checks. Two exact executions produced identical hashes for all
nine packet files. The access summary records four JSON reads, 1,083,155 frozen
parameters, nine future training jobs, and zero Parquet, checkpoint, model,
optimizer, prediction, position, metric/PnL, outcome, or target operations.
Only V75 is authorized.

### V75 — Synthetic persistent-duration harness

- [x] Verify the six exact V74 specification packet inputs and canonical hashes.
- [x] Instantiate the exact 1,083,155-parameter model only on deterministic
      synthetic tensors and verify shapes, causality, and asset permutation.
- [x] Verify centered excess, positive Student-t scales, monotone survival,
      event/right-censor likelihood, joint loss, and finite CPU/MPS backward.
- [x] Verify exact stateful entry/hold/exit/switch costs and incumbent/cash/
      lexical tie priority over a deterministic synthetic policy path.
- [x] Write one synthetic-only checkpoint and prove roundtrip plus interrupted
      resume equivalence without touching any previous or real checkpoint.
- [x] Replay the entire packet byte-identically with zero real data, outcome,
      metric/PnL, or target-asset access.

**V75 gate:** all synthetic scientific, accounting, CPU/MPS, checkpoint, and
replay checks pass. Stop before V76 implementation or dataset access.

V75 passed 17/17 checks. Two independent executions produced identical hashes
for all ten packet files. The exact 1,083,155-parameter model completed finite
CPU and Mac MPS joint backward; the ledger records six metadata reads, six
synthetic optimizer steps, one synthetic checkpoint write/read, and zero real
data, prior checkpoint, performance/PnL, outcome, or target operations.

During the post-gate V76 handoff, the V74 source sequence-index receipt was
found to contain only 61 characters. The authoritative 64-character V32
manifest value is explicitly registered before V76 dataset deserialization.
Two hash-only reads and zero Parquet deserializations are disclosed; the V75
packet and scientific contract are unchanged.

### V76 — Non-target persistent-duration dataset

- [x] Verify the exact V75 harness packet and authoritative V32 metadata hashes.
- [x] Deserialize only the registered V32 non-target panel and sequence index.
- [x] Build cumulative open-to-open return labels for 1, 3, and 7 days.
- [x] Build earliest-argmax 1..7-day duration labels with day-7 right censoring.
- [x] Apply the frozen eight-day maturity, purge, embargo, and chronological
      train/internal-validation/adaptive-evaluation roles.
- [x] Preserve missing rows and masks without imputation, reselection, scaling,
      model/checkpoint work, prediction, metric/PnL, or target access.
- [x] Emit and replay a byte-identical dataset packet authorizing only V77
      frozen non-target training when every check passes.

**V76 gate:** exact hash-audited non-target labels and roles with no scientific
model or financial evaluation. Stop before any V77 implementation or training.

V76 passed 16/16 checks. It produced 43,830 label rows, 43,478 complete
persistent rows, 24,060 train-eligible rows, and 10,628 internal-validation
rows. The 9,798 adaptive-evaluation role rows were derived from dates only and
no 2025 label values were loaded. All 12 packet files and both Parquets replayed
byte-identically, with zero scaler, model, optimizer, checkpoint, prediction,
metric/PnL, outcome, or target operations.

### V77 — Frozen non-target persistent-duration training

- [x] Verify all V74/V75/V76/V32 metadata and registered data receipts before
      any training data deserialization.
- [x] Run the clean-tree and storage preflight with MPS fallback disabled.
- [x] Fit exactly one train-only feature scaler per fold, shared across its
      three seeds, without heldout-fold or 2025 value access.
- [x] Pass the fold-1/seed-42 interrupted/resume smoke equivalence gate.
- [x] Train exactly nine fresh 1,083,155-parameter jobs on MPS/float32 using
      only the frozen train and internal-validation roles.
- [x] Retain and semantically verify every checkpoint without seed/fold/epoch
      selection for economic use.
- [x] Prove zero-step replay and emit the full deterministic training packet.

**V77 gate:** all nine checkpoints, scalers, histories, resume checks, and
receipts must pass with no predictions, positions, performance/PnL, outcomes,
or target access. Stop before V78 evaluation preparation.

V77 passed. Nine fresh checkpoints completed 6,976 optimizer steps; every
fold/seed checkpoint and all three train-only scalers were retained. The MPS
interrupted/resume smoke, semantic roundtrip, complete-grid verification, and
zero-step replay all passed with no prediction, PnL, outcome, or target access.

### V78 — Outcome-blind persistent-duration evaluation prepare

- [x] Verify the exact V74/V76/V77/V32 receipts and all nine checkpoint hashes.
- [x] Read only 2025 feature/readiness projections through signal date
      2025-12-23, with no raw-open, label, or outcome projection.
- [x] Infer every exact lexical heldout triplet with every registered seed and
      freeze the ordered three-seed ensemble without selection.
- [x] Freeze candidate and registered control positions before any outcome
      access.
- [x] Run all 12 outcome-blind behavior gates, including exact stateful
      turnover and the preregistered turnover ceiling.
- [x] Preserve the failed turnover gate, emit a failure receipt, and prove
      hash-only replay without creating or authorizing an outcome packet.

**V78 gate:** predictions, positions, controls, and behavior gates are frozen;
outcome rows read remain zero; BTC/ETH/SOL remain sealed. Stop and require a new
exact hash-bound user authorization before the single financial outcome unseal.

V78 failed only `aggregate_turnover_within_registered_ceiling`: aggregate
candidate turnover was `59.55` against the frozen ceiling of `45.0`. Independent
accounting errors were exactly zero. No outcome, performance metric, PnL, or
target asset was opened, no one-shot packet was created, and the frozen result
requires a pivot without retuning or target evaluation.

### V79 — Metadata-only V78 terminal record

- [x] Read only the four exact hash-bound V78 JSON result, audit, failure, and
      replay receipts.
- [x] Register the V78 behavior-gate failure without recomputing any scientific
      metric, gate, prediction, position, policy, or PnL.
- [x] Retire `tlm_persistent_multi_horizon_duration_v1` at V78 without outcome
      unseal, target evaluation, retuning, or regeneration.
- [x] Emit a deterministic metadata-only packet with zero Parquet, market-panel,
      checkpoint, model, training, inference, outcome-packet, or target access.
- [x] Replay byte-identically and authorize only the separate V80 outcome-blind
      specification of `tlm_low_turnover_cross_sectional_rank_v1`.

**V79 gate:** one deterministic terminal record, four JSON metadata reads, all
scientific-access counters at zero, retired V78 family, and an explicit V80
specification-only receipt. Stop before implementing or executing V80.

V79 passed all seven metadata checks. Two executions produced byte-identical
hashes for all eight files, terminal record SHA-256
`0c96dcb97a4a0178435e43960ae6d5db79623d1c2a011b070974acd2117b297d`,
and result SHA-256
`945fdf7307df510c29eb887ae57476d8de728462ce5498e3969f3daa2422588d`.
The access ledger records four JSON metadata reads and zero scientific, model,
outcome, or target operations. Only V80 specification is authorized.

### V80 — Final low-turnover cross-sectional rank specification

- [x] Verify the four exact V79 terminal metadata receipts without opening data,
      checkpoints, models, outcomes, or targets.
- [x] Freeze one genuinely new, compact cross-sectional relative-rank family;
      do not increase parameters by default or reuse any retired state.
- [x] Freeze one structural turnover budget and low-turnover long-one-or-cash
      policy before any scientific data access.
- [x] Freeze exactly one architecture, objective, parameter count, training grid,
      evaluation protocol, controls, costs, gates, and terminal kill criteria.
- [x] Emit and replay one deterministic metadata-only specification packet,
      authorizing only V81 synthetic harness if every specification gate passes.

**V80 gate:** one fully frozen final-family blueprint with structural turnover
control, no variants or sweep, zero scientific data/model/outcome/target access,
and an explicit V81 synthetic-only receipt. Stop before V81.

V80 passed 14/14 checks. Two executions produced identical hashes for all nine
files, specification SHA-256
`d7fd9306ede1afc0fc193e705c2d1d539e5fde60c023b587012ecfa8812f9cfd`
and blueprint SHA-256
`3b080b6cfcea2be6ef2a3347397e7f669573870abba0f6966bc3eb76eeb1d649`.
It froze 10,993 parameters, nine future training jobs, and structural turnover
at most `16.0`, with four JSON reads and zero scientific/model/target access.

### V81 — Synthetic low-turnover rank harness

- [x] Verify the six exact V80 metadata receipts and immutable blueprint hashes.
- [x] Instantiate exactly 10,993 parameters on synthetic `[4,128,3,8]` tensors.
- [x] Verify causal-prefix invariance, asset permutation equivariance, centered
      scores, finite point/pairwise loss, and finite CPU/MPS backward.
- [x] Verify one synthetic checkpoint roundtrip, interrupted/resume equivalence,
      and adversarial policy turnover never above `16.0`.
- [x] Emit and replay a deterministic synthetic-only packet with zero real data,
      prior checkpoint, outcome, metric/PnL, or target access.

**V81 gate:** exact synthetic model and policy behavior, finite CPU/MPS training,
resume equivalence, structural turnover proof, byte-identical packet replay, and
zero prohibited access. Stop before V82 dataset construction.

V81 passed 15/15 checks on CPU and Apple MPS. The exact 10,993-parameter model,
interrupted resume, and adversarial structural-turnover ceiling of `16.0` all
passed. Both executions reproduced all nine packet files byte-identically; the
ledger records six JSON reads, ten synthetic optimizer steps, one synthetic
checkpoint write/read, and zero real data, prior checkpoint, outcome, metric,
PnL, or target access.

### V82-R0 — Metadata-only chronology erratum

- [x] Verify the exact user authorization and eight hash-bound V80/V81 JSON
      metadata receipts.
- [x] Register only `final_evaluation_signal_end` from `2026-06-09` to
      `2026-06-08` and `final_evaluation_signal_dates` from `160` to `159`.
- [x] Prove that the corrected signal plus 22 days matures on `2026-06-30`,
      while eight decisions and structural turnover `16.0` remain unchanged.
- [x] Prove target, architecture, objective, policy, costs, gates, universe,
      folds, seeds, and hyperparameters remain frozen.
- [x] Emit and replay a deterministic metadata-only packet with zero Parquet,
      data, outcome, checkpoint, model, training, inference, prediction,
      position, metric/PnL, bootstrap, or BTC/ETH/SOL access.

**V82-R0 gate:** exactly two chronology fields corrected by immutable overlay,
all scientific invariants unchanged, byte-identical packet replay, and an
explicit V82 dataset-only receipt. Stop before implementing or executing V82.

V82-R0 passed 14/14 checks. Two executions reproduced all eight packet files
byte-identically. Erratum SHA-256 is
`721f1ca0d4770288a9cd6a03dfdfa1c2920f1f0ecb32f7b0716d5169fa5267ea`
and result SHA-256 is
`56ef31fcb37a5566e3b0badbf1e2681862e9bc86a1affa898950c823dc490b96`.
The ledger records eight JSON metadata reads and zeros for every scientific,
model, outcome, and target operation. V82 dataset is authorized by receipt but
was not implemented, registered as current, or executed in this loop.

### V82 — Non-target low-turnover rank dataset

- [x] Bind the exact user authorization to V82-R0 result SHA-256
      `56ef31fcb37a5566e3b0badbf1e2681862e9bc86a1affa898950c823dc490b96`.
- [x] Freeze the exact V32 30-symbol non-target universe, folds, triplets,
      official Binance monthly source scope, eight features, 128-day lookback,
      21-interval target, chronology, and output schemas before source access.
- [x] Verify and hash-register every admitted official ZIP and published
      checksum; reject invalid archives and preserve gaps without repair.
- [x] Build causal development features and `open[t+1] -> open[t+22]` labels
      for only the frozen train/internal-validation roles and fold assets.
- [x] Build outcome-free 2026 evaluation features and a physically separate,
      hash-sealed daily evaluation-outcome packet through maturity 2026-06-30.
- [x] Prove zero scaler, model, checkpoint, training, inference, prediction,
      position, metric/PnL, bootstrap, outcome-unseal, or BTC/ETH/SOL access.
- [x] Execute the exact command twice and reproduce all 15 metadata files plus
      four Parquets byte-identically.

**V82 gate:** exact causal non-target dataset, explicit source gaps, 159 frozen
evaluation signal dates, sealed 2026 outcomes with unseal count zero, all
forbidden counters zero, and deterministic replay. Stop before V83.

V82 passed 20/20 checks. It verified 2,393 official archives and explicitly
rejected `AXSUSDT 2026-02` because the checksum-valid source contained duplicate
UTC dates; no deduplication, repair, replacement, or imputation was performed.
It produced 75,420 development-feature rows, 70,290 development-label rows,
8,610 outcome-free evaluation-feature rows, and 5,370 sealed daily outcome
rows. Both complete executions reproduced all 19 files byte-identically. The
sealed outcome packet SHA-256 is
`9cc5be0e9dfdc40b4fe8d6433602769d67bfa6b269b5f02fa2d241e6eca0024a`,
its unseal count is zero, and result SHA-256 is
`87d0b0dcd5e6b6b07be6572e73eb53eb2dffbddfc2b8c254d2144eb2f67cc0e6`.
Only a separately registered V83 frozen non-target training phase is
authorized; V83 was not implemented or executed in this loop.

### V83 — Frozen non-target low-turnover rank training

- [x] Register the exact V82 result receipt and exclude evaluation features,
      sealed outcomes, heldout-fold values, and BTC/ETH/SOL from the allowlist.
- [x] Freeze one deterministic balanced schedule: 12,120 train contexts per
      epoch and 19,380 fixed validation contexts, independent of model seed.
- [x] Freeze one train-only median/IQR feature scaler and full lexical
      train-triplet excess-RMS target scale per fold.
- [x] Implement the exact fresh 10,993-parameter model, AdamW objective,
      interrupted resume, semantic checkpoint verification, and zero-step replay.
- [x] Commit one clean source snapshot and pass research doctor plus MPS smoke.
- [x] Train all nine registered fold/seed jobs and retain every checkpoint.
- [x] Verify the complete grid, zero-step replay, access ledger, and terminal
      V83 artifact packet.

**V83 gate:** all nine fresh checkpoints, three train-only scalers and target
scales, interrupted/resume equivalence, semantic verification, complete-grid
verification, and zero-step replay must pass with every prohibited-access
counter zero. Stop before any V84 evaluation implementation or outcome access.

**V83 result:** passed all 12 terminal checks with 9/9 checkpoints and 5,040
optimizer steps. The replay created zero jobs, optimizer steps, or checkpoint
rewrites. Predictions, positions, performance metrics, PnL, outcomes, prior
checkpoints, and target-asset loads remained zero. The terminal receipt
authorizes only a separately registered V84 outcome-blind evaluation prepare;
V84 was not implemented or executed in this loop.

### V84 — Outcome-blind low-turnover rank evaluation prepare

- [x] Register the exact V83 terminal result, completion receipt, nine
      checkpoints, three fold scalers, V82 feature packet, V32 folds/triplets,
      frozen policy, controls, costs, accounting, bootstrap, and gates.
- [x] Load all nine best checkpoint states without selection and infer every
      exact lexical heldout triplet over the 159 registered signal dates.
- [x] Freeze seed and ensemble scores, candidate positions, three control
      position sets, transition turnover, final liquidation, and outcome request.
- [x] Pass every outcome-blind behavior gate, validate the one-shot prepare
      packet, and replay by hashes without model/checkpoint/feature reads.

**V84 gate:** outcomes, economic metrics, PnL, Sharpe, drawdown, bootstrap, and
target assets remain sealed. A passing prepare may emit only the exact V85
hash-bound unseal request. Generic continuation is not unseal authorization.

V84 passed all 12 behavior gates using all nine checkpoints without selection.
It froze 440,784 predictions, 193,320 candidate-position rows, and 579,960
control-position rows. Aggregate structural turnover was `1.527778`, exposure
was `0.260863`, and the hash-only replay matched with zero outcome, target,
performance, PnL, or bootstrap access.

### V85 — Exactly-once low-turnover rank economic evaluation

- [x] Register the exact user authorization against evaluation spec
      `70191a2e...`, prepare receipt `e0d65959...`, registered contract
      `82ee01cb...`, and one-shot packet `a5a21254...`.
- [ ] Atomically write the authorization receipt before the only registered
      5,370-row non-target outcome-source deserialization.
- [ ] Freeze the immutable V85 outcome packet and receipt, then compute only
      the registered 10/20/30 bps accounting, controls, 10,000-path circular
      block bootstrap, and 19 mandatory gate cells.
- [ ] Preserve every failed cell without aggregate rescue, keep BTC/ETH/SOL
      sealed, and replay from the V85 packet with zero source outcome reads.

**V85 gate:** exactly one source opening, immutable receipts, all registered
economic cells preserved, source-free deterministic replay, and one terminal
pass-or-retire decision. Stop before any target-transfer specification.

## Next research objectives — V47 onward

These loops follow `specification -> synthetic harness -> training -> frozen
development evaluation -> immutable registration -> clean confirmation`.
Historical development results cannot substitute for new confirmation data.
The unattended execution contract through the first trained family is frozen
in `AUTONOMOUS_TRAINING_LOOP.md`; its agent prompt is
`prompts/run_v47_to_v49.md`.

### V47 — Joint absolute/relative triplet specification

- [x] Create a new family identity without reviving or modifying V45.
- [x] Keep one exact deployment unit: `float32 [batch, 256, 3, 9]`, with no
      multi-context averaging at inference.
- [x] Freeze one architecture no larger than the V41 Medium model, with no
      parameter-size tournament or hyperparameter sweep.
- [x] Define a coherent return decomposition: triplet market return `m`,
      centered per-asset excess `e_i`, and absolute forecast `mu_i = m + e_i`.
- [x] Freeze one supervised objective, train-only scales, chronology, folds,
      seeds, and no-selection rules before any real label is opened.
- [x] Freeze one long-at-most-one/cash policy that ranks by relative excess but
      enters or switches only when absolute edge covers the exact transition
      cost.
- [x] Freeze one ex-ante concentration rule and risk-matched controls without
      deriving a threshold from V45 diagnostic slices.
- [x] Assign every historical interval exactly one role. Consumed 2025 and
      2026-H1 outcomes may never become clean evaluation evidence; any
      training-only reclassification must be explicit before data access.
- [x] Keep BTC/ETH/SOL sealed and require a future non-target confirmation of
      at least 180 mature signal dates after immutable registration.
- [x] Pass a metadata-only specification audit with zero panel, label,
      checkpoint, prediction, performance, or PnL reads.

**V47 gate:** one hash-locked blueprint, policy, data-role map, gate matrix, and
lifecycle contract. Passing V47 authorizes only V48.

V47 froze `tlm_joint_absolute_relative_triplet_medium_v1` with 1,212,930
analytic parameters and a 36-job V49 grid. Blueprint SHA-256 is
`cb4e068a42ea122db7196ebf118f3e0dc50839e7acb08e9874fa103134508a74`.
Two executions reproduced every artifact byte. V48 may now instantiate and
exercise only the frozen model and scientific harness on synthetic data.

### V48 — Synthetic scientific harness

- [x] Implement only the frozen V47 model, losses, policy, controls, and
      checkpoint schema on deterministic synthetic data.
- [x] Verify causal-prefix behavior, asset permutation equivariance, centered
      excess, and exact `mu = m + e` reconstruction.
- [x] Verify entry, hold, switch, exit, ineligibility, turnover, costs,
      concentration, cash, and final liquidation accounting.
- [x] Verify missing-asset behavior, finite gradients, checkpoint roundtrip,
      hashes, and deterministic resume.
- [x] Produce two byte-identical runs without opening real market data.

**V48 gate:** every scientific and accounting test passes. Passing V48
authorizes only the frozen V49 training workflow.

V48 passed under harness SHA-256
`67906d659fe3780cf08a89a1c7dd153b7887a2f3c42bb1c68b6d70329912c86d`.
The 1,212,930-parameter model completed two deterministic synthetic optimizer
steps, and interrupted checkpoint resume matched uninterrupted training
exactly. A second run reproduced every JSON, report, config, and checkpoint
byte. V49 implementation and metadata-only preflight are authorized next.

### V49 — Purged non-target walk-forward training

- [x] Use the two origins and expanding/rolling geometries frozen in V47, with
      chronological purge and train-only scaling in every cell.
- [x] Train all `2 origins x 2 geometries x 3 asset folds x 3 seeds` on Apple
      MPS and retain all 36 checkpoints without selection.
- [x] Initialize every job from its fresh registered seed; do not reuse any
      V35/V36/V43/V44/V45 checkpoint or run representation pretraining.
- [x] Restore the strictly best validation-`L_total` state inside every job;
      forbid selection between seeds, folds, origins, or geometries.
- [x] Record optimizer, RNG, scaler, data-access, checkpoint, and resume hashes.
- [x] Keep held-out fold assets, forbidden windows, and BTC/ETH/SOL outside
      every training tensor.

**V49 gate:** all 36 jobs, checkpoint roundtrips, idempotent replay, and access
audits pass with no resume artifact remaining. Training loss alone does not
authorize an economic claim. Passing V49 authorizes only V50.

V49 passed under contract SHA-256
`2aa4984082bc402fb8b259ab17c821fca6853c37ade6c5e83187157781a0a49a`.
The 36 checkpoints completed 235 epochs and 15,040 optimizer steps; every job
restored its local best state, all checkpoint roundtrips passed, and no resume
artifact remained. The idempotent full replay reported zero new jobs and zero
new optimizer steps while preserving result SHA-256
`55aa56e309482da96c7985fb47367df5073576c1e05d05c2dd3509f33a2ca256`.
The verified training-source Git receipt is
`78c74938db4443aa0c3b437671d8433014c7301d`. V50 subsequently evaluated the
complete frozen checkpoint set without selection or refit.

### V50 — Frozen historical development evaluation

- [x] Evaluate one exact-triplet policy, cash, shared Ridge decomposition,
      dual momentum, and risk-matched controls with no candidate tournament.
- [x] Require 10/20/30 bps cost cells; retain 50 bps, one-day signal delay, and
      missing-asset scenarios as preregistered stress tests.
- [x] Run 10,000-path paired circular block bootstrap at 7/21/63-day blocks
      using identical resample indices for candidate and controls.
- [x] Report every time window, asset fold, asset, regime, turnover,
      concentration, drawdown, and cost cell without aggregate rescue.
- [x] Require positive candidate net return in every mandatory fold/window,
      drawdown at or below 35%, and positive absolute and paired-control
      bootstrap fifth percentiles as frozen by V47.
- [x] Preserve the result as adaptive historical development evidence only;
      do not inspect BTC/ETH/SOL.

**V50 gate:** any mandatory failure retires the family without tuning on the
same windows. A complete pass authorizes only immutable registration V51.

V50 failed and retired the family. The audit passed, but only 67/180 gate cells
passed: Spearman 12/12, pairwise accuracy 12/12, top-1 excess 8/12, per-fold
10 bps return 3/12, turnover versus dual momentum 0/4, return versus dual
momentum 0/12, return versus equal weight 0/12, and bootstrap 0/48. The
idempotent rerun preserved result SHA-256
`3ba18a664e909ebebb86b90438323ff6098d339614c134dc8ecc0b644d3a2afa`.
V51 is not authorized.

### V51 — Immutable registration without refit

**Status:** not authorized because V50 failed.

- [ ] Perform no retraining or recalibration; package only the nine
      predesignated V49 `origin_2025/expanding` checkpoints.
- [ ] Hash-lock source, feature schema, data roles, scalers, all fold/seed
      checkpoints, policy, controls, gates, evaluator, and environment.
- [ ] Register the first eligible prospective signal strictly after the final
      registration receipt; never backfill from 2026-07-14 or any earlier date.
- [ ] Start no daily inference, simulated order stream, or interim PnL view.

**V51 gate:** one immutable candidate packet and prospective clock. Any change
creates a new family and a new clock.

### V52 — Deferred future non-target confirmation

- [ ] Accumulate at least 180 eligible signal dates after V51 and wait until
      every registered target horizon has matured.
- [ ] Expose no interim candidate actions, returns, equity, regime metrics, or
      performance-derived data repair decisions.
- [ ] Freeze predictions and positions before the single outcome unseal.
- [ ] Run the registered fold, cost, drawdown, turnover, concentration, and
      paired-bootstrap gates exactly once.

**V52 gate:** any failure retires the family. A pass authorizes only a separate
BTC/ETH/SOL protocol; it does not authorize deployment.

### V53 — Separate target-domain protocol

- [ ] Preregister an unopened future BTC/ETH/SOL window, target inference,
      costs, controls, gates, and one-shot lifecycle before reading outcomes.
- [ ] Reuse the exact V51 candidate without calibration, tuning, or selection.
- [ ] Perform one target evaluation only after V52 passes and all target labels
      mature.

**V53 gate:** any failure retires the candidate. A pass triggers a separate
deployment review; it never places orders automatically.

### V54 — V50 economic failure autopsy

- [x] Freeze all inputs, hypotheses, diagnostic slices, and forbidden actions.
- [x] Pass a metadata/hash preflight with zero Parquet deserializations.
- [x] Diagnose absolute calibration in all 12 origin/geometry/fold cells.
- [x] Decompose registered turnover, transitions, triplet consensus, and costs.
- [x] Preserve the V50 retirement and prohibit counterfactual policy tuning.

V54 passed under autopsy-spec SHA-256
`4ebeaba6de4794263a9d0cbcca14a0d8547350edde80140d9328052128f8f6a4`.
Absolute Pearson correlation ranged from `-0.0472` to `0.0225`, sign accuracy
from `47.08%` to `50.44%`, and turnover from `3.27x` to `4.46x` dual momentum.
The decision is `v50_retirement_confirmed_diagnostic_only`.

### V55 — State-conditioned multi-horizon family specification

- [x] Create a new family identity with no V49 checkpoint reuse.
- [x] Freeze one 465,513-parameter shared-asset architecture.
- [x] Freeze 1/3/7-day q20/q50/q80 targets and losses.
- [x] Freeze h7-q20 state-conditioned utility and seven-date decision clock.
- [x] Freeze the 36-job training grid, controls, evaluation gates, and lifecycle.
- [x] Pass metadata-only with zero Parquet, model, optimizer, prediction, PnL,
      or target-asset access.

V55 passed under blueprint SHA-256
`0c91c65ed422d081ba1ce59544c3911cbd4624a0f4c184cc24d6c02dfc41d435`.
Its only authorized next action is V56 synthetic state/policy harness.

### V56 — Synthetic state/policy harness

- [x] Instantiate exactly the V55 model on deterministic synthetic tensors.
- [x] Verify causal prefix behavior and joint asset permutation equivariance.
- [x] Verify quantile ordering, pinball/ranking/crossing losses, and finite gradients.
- [x] Verify the seven-date clock, state-conditioned costs, forced cash, and liquidation.
- [x] Verify checkpoint roundtrip and interrupted deterministic resume.
- [x] Prove zero real panel, checkpoint, performance, PnL, or target-asset access.

**V56 gate:** every synthetic scientific and accounting check passes twice
byte-identically. Passing authorizes only V57 non-target multi-horizon dataset
construction; it does not authorize training.

V56 passed all 30 registered checks under harness SHA-256
`df1370ecb4f00c97222fc56ae18645362d5695535fcf32b48f978d0889137847`.
The exact 465,513-parameter model completed two deterministic synthetic CPU
optimizer steps; interrupted resume matched uninterrupted execution, and two
full invocations reproduced all packet files byte-identically. Parquet, real
labels, prior checkpoints, real predictions/performance/PnL, and BTC/ETH/SOL
access all remained zero. Only V57 dataset construction is now authorized.

### V57 — Non-target multi-horizon dataset

- [x] Verify the exact V56 completion packet and every registered V32 input hash.
- [x] Load only the frozen 30-asset non-target panel and sequence index.
- [x] Build `open[t+1] -> open[t+2]`, `open[t+1] -> open[t+4]`, and
      `open[t+1] -> open[t+8]` log-return labels for horizons 1/3/7.
- [x] Enforce the eight-day maximum maturity purge at every chronological role
      boundary without moving, imputing, or repairing missing rows.
- [x] Preserve the exact nine-feature order, three asset folds, triplet catalog,
      keys, and target-asset exclusion.
- [x] Prove zero scaler fit, model instantiation, optimizer step, prediction,
      performance/PnL, or BTC/ETH/SOL access.
- [x] Reproduce the full dataset and artifact packet byte-identically.

**V57 gate:** every causal data, maturity, access, hash, and replay check must
pass. Passing authorizes only V58 frozen non-target training; it does not
authorize evaluation, target assets, or any later phase.

V57 passed all 14 checks. It preserved 60,210 label keys and 49,919 sequence
keys, with 58,344 rows complete across h1/h3/h7. Both Parquets and all packet
files replayed byte-identically. Labels SHA-256 is
`6d12e9d49f1be807a1eba5596295fa40f43c3d89745a9a39f3e3f42d76544f50`;
sequence-role SHA-256 is
`1f12eff301984943d0a907e55b33aa65618cf2977f88bb318610f3cfebc52860`.

### V58 — Frozen non-target MPS training

The V58 runner, train-only data loader/scalers, exact MPS engine, checkpoint
resume/verification, policy-conditional storage enforcement, and zero-step
replay completed under the hash-bound V58r1 owner storage waiver. The waiver
changed no data, architecture, objective, optimizer, grid, seed, or scientific
gate.

- [x] Register the hash-bound V58r1 owner waiver and prove it is storage-only.
- [x] Freeze and validate the exact training/operator packet and clean source receipt.
- [x] Pass the full runtime doctor: clean Git, bound V58r1 owner waiver, at least
      50 GiB free, float32 MPS available, deterministic algorithms, and fallback disabled.
- [x] Prove one-job interrupted/resumed smoke equals uninterrupted training.
- [x] Train all `2 origins x 2 geometries x 3 folds x 3 seeds` jobs.
- [x] Fit only per-cell train-only feature scalers and retain all 36 local checkpoints.
- [x] Keep held-out folds, evaluation roles, BTC/ETH/SOL, predictions, policy,
      performance metrics, and PnL outside every training access.
- [x] Verify checkpoint roundtrips, complete grid, no orphan resume, and source hashes.
- [x] Replay the full command with zero new jobs and zero optimizer steps.

**V58 gate:** every runtime, grid, checkpoint, access, resume, verification, and
zero-step replay check must pass. Passing authorizes only V59 frozen adaptive
development evaluation; it does not open outcomes in the training loop.

V58 passed with 36/36 jobs, 36 retained and verified checkpoints, 12 scalers,
and 22,976 optimizer steps. No job resumed. Verification and replay passed;
replay created zero jobs, executed zero optimizer steps, and rewrote zero
checkpoints. Result file SHA-256 is
`3d3a78eb8ccc62593a9bfebc2e1ab03e16da452e1e858e1ad71452405d6f9440`;
registered result SHA-256 is
`2b76a66015fda7b5899ed242cbb8bfb909535f828ffaeaaa81f982258a421273`.

### V59 — Frozen adaptive development evaluation

- [x] Register V58 as the completed parent experiment with exact result,
      completion, checkpoint, scaler, source, verification, and replay hashes.
- [x] Freeze all 12 origin/geometry/fold cells and all 36 checkpoints without
      checkpoint selection, weighting, or geometry blending.
- [x] Freeze three-seed arithmetic averaging before policy construction, 120
      lexical triplets per fold, independent triplet state, controls, costs,
      accounting, metrics, 10,000-path paired block bootstrap, and terminal gates.
- [x] Freeze prepare -> explicit new user authorization -> exactly one atomic
      outcome unseal -> immutable outcome packet -> source-free replay.
- [x] Implement the registered V59 config and prepare-only CLI.
- [x] Verify every input, checkpoint, and scaler binding; fit only the frozen
      shared linear control on train-role values.
- [x] Run inference for all registered cells and freeze candidate/control
      predictions, positions, and exact outcome keys with zero outcome reads.
- [x] Pass every pre-outcome behavior, accounting, access, and prepare-replay gate.
- [x] Stop and obtain a new explicit user authorization bound to the evaluation
      spec and prepare-receipt hashes. Generic continuation is insufficient.
- [x] Write one atomic authorization receipt, read the exact registered
      non-target development projection once, and freeze the outcome packet.
- [x] Compute only the frozen metrics, costs, controls, bootstrap, and gate matrix.
- [x] Retire on any mandatory failure or authorize only V60 immutable registration
      on a complete pass; then prove replay performs zero source reads and inference.

Observed V59 result: **retired without tuning** after 603/700 mandatory gates
failed. The one-shot unseal count is 1, replay source reads are 0, and
BTC/ETH/SOL remained sealed.

### V59 immutable retirement autopsy

- [x] Freeze and lint a nine-file, hash-exact JSON/Markdown input allowlist.
- [x] Verify all frozen inputs before and after analysis without opening Parquet.
- [x] Preserve all 700 gate cells, 12 predictive cells, 80 aggregate economic
      cells, 240 fold economic cells, and all 108 registered bootstrap cells.
- [x] Attribute signal, calibration, activation/churn, and registered-cost failure.
- [x] Prove an idempotent replay with zero source reads, checkpoint loads,
      inference, new bootstrap paths, target access, or counterfactual PnL.

The autopsy confirmed `weak_ordinal_signal_failed_policy_and_absolute_return_conversion`
as the primary attribution. All 24 ordinal gates passed, but only 1/12 fold
return gates passed, all 540 bootstrap gates failed, five policy cells were
cash-only, and overall risky exposure was 0.3984%. Costs amplified but did not
cause the failure: every aggregate candidate cell was already negative before
registered cost drag. The V59 retirement remains immutable. Autopsy result
SHA-256 is `7a84b8e2f00ce17a79dcb33ad0306037a97773f7dd6ea6a09f6a9d8b2c1b96cf`;
artifact-manifest SHA-256 is
`34458a0178f2986c7330537ffc545086b51a56381854e72c276a22c32f32ac87`.

**V59 gate:** preparation must finish with zero development-outcome reads and
then stop. A separate explicit user authorization permits at most one outcome
unseal. Any failed mandatory cell retires the family without tuning. BTC/ETH/SOL
remain sealed throughout V59.

The frozen V59 phase-contract SHA-256 is
`321c6a805b94f73d441def62af7478337b5e33f5f23631808dba032a376df6a2`.
Its V59-specific owner storage waiver SHA-256 is
`c31f88585e248eac17f6c4f1d0c03df9029d4712f9c5c2180bbf677b68c73ae0`;
external redundancy is waived, but every required local artifact, atomic write,
and content hash remains mandatory.

## Historical v5 provisional candidate (superseded)

- [x] Diagnose intraday confidence, fold drift, and turnover.
- [x] Replace intraday target with causal next-open-to-next-open evaluation.
- [x] Add a deterministic 30-day dual-momentum baseline and risk gate.
- [x] Train three Transformer seeds and reject single-seed/mean-ensemble claims.
- [x] Use unanimous Transformer ranking only as an override to dual momentum.
- [x] Pass buy-and-hold and dual-momentum gates on return, Sharpe, and drawdown.
- [x] Pass 5-30 bps/side cost sensitivity and fold return tolerance.
- [x] Audit chronology, labels, predictions, turnover, costs, and finite metrics.

The v5 consensus passed its original three-fold test, but its acceptance is
**suspended** after the v6 extended validation. It materially failed the
expanding-six-fold scenario and did not beat dual momentum reliably in the
rolling-six-fold Monte Carlo. Pure 30-day dual momentum remains a research
control only. See `artifacts/v6_validation_suite/validation_suite.md`.

## Extended validation

- [x] Add expanding and rolling walk-forward geometries.
- [x] Run six-fold expanding and six-fold rolling/730-day training scenarios.
- [x] Add paired circular block bootstrap with 7/21/63-day blocks.
- [x] Run 3,000 Monte Carlo paths per block and scenario.
- [x] Add one-day signal-delay stress.
- [x] Preserve and report failed scenarios rather than selecting winners.

## OverrideNet v1

- [x] Freeze dual momentum 30d as the default action and research control.
- [x] Build causal regime/ranking features and gross residual action targets.
- [x] Add deterministic Huber gradient boosting for BTC/ETH/SOL/CASH actions.
- [x] Subtract exact path-dependent incremental turnover before overriding.
- [x] Calibrate abstention only on a purged inner chronological window.
- [x] Preserve the three v6 expanding/rolling outer scenarios.
- [x] Pass all unit tests and execute the offline v7 validation suite.
- [x] Record the reject decision without promoting the failed candidate.

OverrideNet v1 was rejected in all three frozen scenarios. The calibrated
overrides underperformed dual momentum on return, Sharpe, drawdown and paired
block-bootstrap probability. See `artifacts/v7_override_net/report.md`.

## Risk-Off Meta-Labeler v1

- [x] Freeze the action space to `{dual momentum, cash}`.
- [x] Add q10/q50 Gradient Boosting models for active control returns.
- [x] Add hysteresis and purged nested threshold calibration.
- [x] Register return-retention, Sharpe, drawdown, fold, cost and MC gates.
- [x] Pass unit tests and execute the offline v8 validation suite.
- [x] Record the result as adaptive research, not a clean holdout decision.

Risk-Off Meta-Labeler v1 was rejected. Q10 empirical coverage remained near
10%, but risk-off loss precision stayed near 50%; missed gains exceeded avoided
losses. The rolling scenario improved Sharpe and drawdown but retained only
88.4% of control return and failed fold/cost gates. See
`artifacts/v8_risk_off/report.md`.

## Signal Existence Study v9

- [x] Register nine causal OHLCV regime signals before evaluation.
- [x] Fit quintile boundaries and downside orientation on purged train data only.
- [x] Add OOS downside/tail lift and fold monotonicity metrics.
- [x] Add circular block bootstrap gates for 7/21/63-day dependence.
- [x] Pass leakage/unit tests and execute all three frozen scenarios.
- [x] Decide whether to close OHLCV-only research or permit one new policy test.

No registered OHLCV signal passed all scenarios. Momentum, trend and dispersion
had strong downside/tail lift but only 1.6-3.6% risk coverage in expanding
windows, versus roughly 11.6-18.6% rolling coverage. This non-stationarity fails
the frozen existence rule. Close the OHLCV-only policy branch and add a new,
correctly timestamped derivatives data family. See
`artifacts/v9_signal_existence/report.md`.

## Derivatives Data Layer v10

- [x] Freeze a common BTC/ETH/SOL archive window before model research.
- [x] Download funding, premium-index basis, and futures metrics archives.
- [x] Verify every archive against its published SHA-256 checksum.
- [x] Normalize legacy headerless premium-index CSVs with an explicit schema.
- [x] Aggregate raw 8-hour, daily, and 5-minute observations by UTC day.
- [x] Enforce `source_max_timestamp < open(t+1)` with a leakage test.
- [x] Record source gaps without forward-filling and audit rolling coverage.
- [x] Execute the full official backfill and preserve the policy-free report.

V10 passed. All 5,349 archives verified. Base coverage is 99.82% for BTC/ETH
and 99.88% for SOL; rolling-derived coverage is 95.86%, 96.04%, and 97.87%,
respectively. The 2-3 missing days per asset are absent premium-index candles
and remain explicit. See `artifacts/v10_derivatives_data/report.md`.

## Derivatives Signal Existence Study v11

- [x] Join OHLCV control outcomes to observed complete derivatives rows.
- [x] Register funding, basis, OI, and positioning diagnostics before evaluation.
- [x] Reuse expanding/rolling walk-forward and block-bootstrap gates.
- [x] Record insufficient-quantile registrations as rejected, not retuned.
- [x] Persist per-scenario gate failures and decide signal existence.

No derivatives signal passed all scenarios. Taker long/short flow was the
closest: it retained one orientation and passed expanding-6 plus rolling-6,
but expanding-3 produced only 8.7% risk coverage, 1.22x tail lift, and failed
bootstrap. Daily funding was non-stationary across folds and the raw daily sum
could not form five unique quintiles in two scenarios. No policy is authorized.
See `artifacts/v11_derivatives_signal_existence/report.md`.

## Intraday Derivatives Path Study v12

- [x] Pre-register path summaries from raw 5-minute metrics before evaluation.
- [x] Reverify all source checksums and audit per-feature availability.
- [x] Cover taker-flow persistence/reversal and open-interest path shape.
- [x] Preserve UTC causality, complete cases, and no forward-fill.
- [x] Reuse the v11 signal-existence and Monte Carlo gates without tuning.
- [x] Decide whether any path signal justifies one separate policy experiment.

No v12 signal passed any complete scenario. Taker-ratio autocorrelation was the
strongest rejected pattern, with 1.46-1.73x downside lift and 1.45-2.10x tail
lift, but its risk buckets covered only 7.5-9.5% versus the frozen 12% minimum;
fold monotonicity also failed expanding-3/6. This branch is closed and no
policy is authorized. See
`artifacts/v12_intraday_path_signal_existence/report.md`.

## Independent Data Family Feasibility v13

- [x] Audit historical availability and timestamp semantics before modeling.
- [x] Compare options volatility, liquidations/order book, macro, and clean holdout.
- [x] Record licensing, cost, coverage, and reproducibility constraints.
- [x] Select at most one genuinely independent family or stop research.
- [x] Complete the audit without training a model or designing a policy.

V13 selected Deribit DVOL as the only feasible independent family. Its public
daily endpoint returned 1,673/1,673 observations for both BTC and ETH from
2021-12-01 through 2026-06-30, with locally persisted response hashes. Daily
candles finalize at the next UTC boundary, so a candle stamped `t` may first
affect execution at `t+2` under the strict chronology rule. Vintage macro was
deferred because revision-safe access requires a key and release-time handling;
the official Binance archive does not provide the required historical order-
book/liquidation reconstruction. See
`artifacts/v13_data_family_feasibility/report.md`.

## Options Volatility Data Layer v14

- [x] Build a cache-first DVOL downloader with deterministic pagination.
- [x] Preserve raw response hashes and source metadata for BTC and ETH.
- [x] Normalize daily OHLC without filling gaps or fabricating SOL DVOL.
- [x] Materialize market-level causal features with the frozen `t -> t+2` lag.
- [x] Audit full-window coverage, finiteness, chronology, and reproducibility.
- [x] Keep v14 data-only; do not evaluate signals, models, policies, or PnL.

V14 passed. BTC and ETH each have 1,673 complete daily observations; the
cache-only replay reproduced the frozen observation hashes. The 30-day warmup
leaves 1,644 rows across 16 named market-level features, every source candle is
shifted to execution at `t+2`, and no SOL-specific series is fabricated. See
`artifacts/v14_dvol_data/report.md`.

## Options Volatility Signal Existence v15

- [x] Register a small DVOL signal set and train-derived orientation gates before evaluation.
- [x] Join only the audited `t+2` feature table to the dual-momentum control outcome.
- [x] Reuse expanding-3/6, rolling-6, purge, coverage, and block-bootstrap gates.
- [x] Evaluate observed complete rows without imputation or policy construction.
- [x] Freeze all outcomes and authorize at most one bounded policy experiment.

No v15 signal survived all three scenarios. Mean one-day DVOL change was the
only near signal: it passed expanding-3 and rolling-6 with one orientation, but
failed expanding-6 fold monotonicity and produced only 1.17x downside lift
against the frozen 1.25x gate. Absolute DVOL level produced high lifts only in
1.5%-2.1% expanding-window buckets. The DVOL policy branch is closed. See
`artifacts/v15_dvol_signal_existence/report.md`.

## CFTC Positioning Feasibility v16

- [x] Audit official CFTC COT access, schema, BTC/ETH coverage, and update cadence.
- [x] Establish conservative as-of-to-availability timing and revision limitations.
- [x] Record access terms, cost, local hashing, and solo-project reproducibility.
- [x] Select or reject the family before building a dataset or evaluating returns.

V16 rejected CFTC positioning despite complete 239/239-week BTC and ETH
coverage and a usable schema. The official report date is an as-of date, CFTC
states that a complete historical publication-date list does not exist, and
the window includes the 2025 appropriations interruption with releases weeks
late. The public dataset also lacks a point-in-time revision archive. See
`artifacts/v16_cftc_positioning_feasibility/report.md`.

## Treasury Curve Feasibility v17

- [x] Probe official annual daily par-yield CSVs without credentials.
- [x] Verify 2-year/10-year coverage and methodology consistency after 2021-12-06.
- [x] Register a conservative `source date -> t+3 execution` availability rule.
- [x] Define bounded causal carry-forward for non-trading days and record revisions.
- [x] Select or reject the family before producing features or evaluating returns.

V17 selected the U.S. Treasury daily par-yield curve. The official annual CSVs
provided 1,141 finite 2y/10y observations from 2021-12-06 through 2026-06-30,
95.72% of weekdays including federal holidays, with a maximum four-day source
gap. Annual payload hashes are frozen. See
`artifacts/v17_treasury_curve_feasibility/report.md`.

## Treasury Curve Data Layer v18

- [x] Build a cache-first downloader with annual raw-response hashes.
- [x] Normalize 2y, 10y, curve slope, changes, and frozen rolling diagnostics.
- [x] Materialize daily known-state rows only after `source date + 3 days`.
- [x] Retain source date, eligibility time, state age, and seven-day carry bound.
- [x] Audit coverage, finiteness, chronology, schema, hashes, and cache replay.
- [x] Keep v18 data-only; do not evaluate returns, signals, models, or policies.

V18 passed with 1,141 raw observations, 1,639 causal daily state rows, ten
registered features, stable cache replay hashes, and a maximum realized state
age of six days. Every source finalization strictly precedes execution. See
`artifacts/v18_treasury_curve_data/report.md`.

## Treasury Curve Signal Existence v19

- [x] Register eight economically distinct rate-level/change/curve diagnostics.
- [x] Join only audited daily states with the frozen `t+3` eligibility contract.
- [x] Reuse expanding-3/6, rolling-6, purge, coverage, and block-bootstrap gates.
- [x] Evaluate complete observed rows without feature, threshold, or policy tuning.
- [x] Freeze all outcomes and authorize at most one bounded policy experiment.

No v19 signal passed any complete scenario. Ten-year rate changes retained one
orientation and showed tail lift, but downside lift stayed at 1.04x-1.16x and
risk coverage was 11.0%-13.0%. Rate levels, curve shape, and change-volatility
were weak or produced non-quintile coverage. The Treasury policy branch is
closed. See `artifacts/v19_treasury_signal_existence/report.md`.

## Evidence and Multiple-Testing Ledger v20

- [x] Register every v1-v19 decision, artifact, audit, and supersession link.
- [x] Count policy trials, data audits, signal families, and signal-scenario tests.
- [x] Verify all evidence files and source audits before synthesis.
- [x] Quantify historical-window reuse and clean-holdout contamination.
- [x] Decide whether any further historical model search is authorized.

V20 audited 19 decisions, 61 registered signals, and 183 signal-scenario
evaluations. It found zero robust signals, zero clean-holdout decisions, and no
active candidate. Further model or feature search on the exposed historical
window is halted. See `artifacts/v20_evidence_ledger/report.md`.

## Deterministic Research-Control Certificate v21

- [x] Verify the frozen v6 validation and v20 evidence-ledger inputs.
- [x] Freeze the 30-day control construction without tuning.
- [x] Extract all historical and 3,000-path bootstrap risk diagnostics.
- [x] Separate benchmark certification from deployment authorization.
- [x] Preserve source hashes and a machine-readable certificate.

V21 certifies dual momentum 30d only as a deterministic research control. It
has no learned parameters and beat equal-weight buy-and-hold in the frozen v6
scenarios, but all nine bootstrap cells have negative fifth-percentile total
returns and observed drawdown is about 50%. Live, shadow, paper, and real-money
execution remain unauthorized. See `artifacts/v21_control_certificate/report.md`.

## Prospective Holdout Protocol v22

- [x] Freeze the first untouched UTC decision date and quarantine boundary.
- [x] Define deferred batch evaluation with no interim performance inspection.
- [x] Register minimum duration, observations, regimes, and promotion gates.
- [x] Require candidate registration before the holdout clock can start.
- [x] Keep the protocol evaluation-only with no execution or daily simulation.

V22 freezes a one-shot prospective protocol but leaves it dormant because no
candidate exists. A future candidate must be fully hashed before its future
window. It is evaluated only once, in batch, after at least 365 days/observations
and the registered activity/regime quotas; there is no live inference, shadow
trading, or interim score access. See
`artifacts/v22_prospective_holdout/report.md`.

## Reproducibility Bundle v23

- [x] Hash the research source, configs, and decision artifacts.
- [x] Capture runtime, dependency, and platform metadata.
- [x] Run the complete offline test suite and persist its result.
- [x] Add a standalone verifier for the frozen manifest.
- [x] Produce a compact reproducibility report without rerunning research.

V23 content-addresses the current source, tests, configs, project contracts,
and the chained v20-v22 decisions. It persists the full pytest result plus
runtime/package metadata and verifies every registered file with
`make verify-v23`. See `artifacts/v23_reproducibility_bundle/report.md`.

## Independent Research Review v24

- [x] Recheck v20-v23 decisions from their frozen machine artifacts.
- [x] Audit whether any model, control, or holdout was accidentally promoted.
- [x] Grade evidence quality, reproducibility, and deployment readiness.
- [x] Record prioritized findings and explicit authorized actions.
- [x] Produce a reviewer decision without modifying historical results.

V24 passes as a read-only artifact review and blocks promotion. Research
controls and reproducibility grade strongly, but deployment readiness is 0/4:
there is no active candidate, no completed prospective holdout, no prospective
superiority result, and no risk authorization. See
`artifacts/v24_research_review/report.md`.

## Final Completion Audit v25

- [x] Verify the complete v1-v24 decision chain and every required audit.
- [x] Re-run the full suite after all final code and documentation changes.
- [x] Hash the final v24 review and current project contract.
- [x] Publish the final research status, limitations, and next legal transition.
- [x] Close the engineering goal only if every v25 gate passes.

V25 closes the engineering program as a complete, reproducible research
framework with a negative trading result. No learned candidate, deterministic
control, or execution mode is approved for deployment. See
`artifacts/v25_final_audit/report.md`.

## Zero-Shot Candidate-Family Specification v26

- [x] Anchor all new work to the immutable annotated `tlm-v25` release.
- [x] Freeze one compact cross-asset architecture without a parameter sweep.
- [x] Exclude BTC, ETH, SOL, and registered target proxies from development.
- [x] Register chronological non-target splits and PnL/drawdown gates.
- [x] Keep training, target inference, target PnL, and v22 registration disabled.

V26 freezes the `tlm_zero_shot_cross_asset_v1` blueprint and its contamination
boundary. No model or performance result exists. The only authorized next step
is a policy-free v27 inventory of non-target official archives; training or
target evaluation remains forbidden. See
`artifacts/v26_zero_shot_candidate_spec/report.md`.

## Non-Target Universe Audit v27

- [x] Enumerate official Binance monthly spot 1d archive prefixes without using return or liquidity rankings.
- [x] Apply the frozen v26 target, proxy, stablecoin, and leveraged-token exclusions.
- [x] Verify every accepted ZIP against its published SHA-256 and validate its daily schema.
- [x] Enforce the 2021-01-01 through 2026-06-30 listing and 98% coverage gates.
- [x] Freeze the first 48 eligible symbols in lexical order and preserve source-index hashes.
- [x] Keep target observations, training, returns, PnL, and portfolio construction at zero.

V27 selected 48 symbols from 58 eligible candidates among the first 60 audited,
with 3,167 accepted monthly archives and 98.60%-100% coverage. The malformed
AXS February 2026 archive is recorded as rejected. EURUSDT remains in the exact
v26 lexical result and is flagged as a crypto-thesis scope mismatch that must be
resolved before training. V28 is authorized only for non-target dataset
materialization. See `artifacts/v27_non_target_universe_audit/report.md`.

## Non-Target Causal Dataset v28

- [x] Reverify all 3,167 cached ZIPs and checksum sidecars against the frozen v27 manifest.
- [x] Materialize the 48-symbol daily calendar panel without filling 39 missing raw rows.
- [x] Freeze eight per-asset features and the causal within-triplet relative-strength transform.
- [x] Materialize `open[t+1] -> open[t+2]` return and seven-day forward-volatility labels.
- [x] Preserve the v26 chronological windows and three performance-blind asset-disjoint folds.
- [x] Hash the Parquet panel and reproduce it byte-for-byte from the cache.
- [x] Keep target assets, scaling, training, portfolios, performance metrics, and PnL at zero.

V28 produced 96,336 panel rows from 96,297 observations. The 12 MB Parquet
stays outside Git; its SHA-256 is
`0fc76703c14071cca5b11c77bae91737a5863b75141de296ab0555bb910e49b4`.
All 106 tests pass. Because EURUSDT remains a v26 scope mismatch, v29 is limited
to a performance-blind universe amendment and the corresponding manifest/data
refresh. Training is still forbidden. See
`artifacts/v28_non_target_dataset/report.md`.

## Multi-Asset Training Scope Amendment v29

- [x] Keep representation learning multiasset while restricting inference/trading to BTC/ETH/SOL.
- [x] Supersede the alphabetical 48-asset cap before observing any model performance.
- [x] Freeze a 30-asset universe with three equal asset-disjoint folds.
- [x] Restrict selection inputs to coverage and quote volume from 2021-2023.
- [x] Exclude targets/proxies, fiat, stablecoins, fan tokens, and leveraged tokens.
- [x] Prohibit labels, returns, validation/calibration/confirmation data, training, and PnL.

V29 freezes `tlm_multi_asset_target_transfer_v2`. V27/v28 remain immutable
historical artifacts but are superseded for training. V30 may inventory all
eligible Binance spot archives and select the top 30 by median training-window
USDT quote volume, with no access to target assets or performance. See
`artifacts/v29_multi_asset_scope_amendment/report.md`.

## Training-Universe Liquidity Inventory v30

- [x] Enumerate every Binance spot USDT archive prefix without using future availability for selection.
- [x] Apply the frozen target/proxy, fiat, stablecoin, fan-token, and leveraged-token exclusions.
- [x] Read only symbol, UTC date, and quote volume from 2021-01-01 through 2023-12-31.
- [x] Verify 5,616 candidate monthly archives against published checksums.
- [x] Enforce 99.5% calendar coverage and 99% nonzero quote-volume coverage.
- [x] Rank eligible assets by median daily USDT quote volume with lexical tie-breaking.
- [x] Freeze exactly 30 symbols and three asset-disjoint folds of ten symbols.
- [x] Preserve the candidate and selected manifests without labels, returns, model training, or PnL.

V30 selected 30 assets from 145 eligible candidates. All selected assets have
100% coverage and nonzero daily quote volume in the selection window. The
candidate manifest contains 5,616 checksum-verified records; the selected
manifest contains 1,080. Training remains forbidden. V31 may only freeze and
audit the full development-window source manifest for these exact symbols. See
`artifacts/v30_training_universe_inventory/report.md`.

## Selected-Universe Source Manifest v31

- [x] Preserve the exact v30 symbol list, ranking, and three asset-disjoint folds.
- [x] Inventory all 1,980 expected symbol-months through 2026-06-30.
- [x] Verify 1,928 accepted archives against their published checksums and daily schema.
- [x] Preserve 52 rejected/missing archives without repair, proxy substitution, or reselection.
- [x] Record full-window coverage as a diagnostic rather than a future-availability filter.
- [x] Keep features, labels, returns, models, portfolios, performance, PnL, and targets at zero.

V31 materialized 58,591 observed source rows and preserved 1,619 missing panel
rows. MATIC, FTM, and EOS retain their post-migration/delisting gaps; the
malformed AXS February 2026 archive remains rejected. This is intentional
survivorship-bias control, not a reason to replace those assets. V32 may only
build and audit the causal 30-asset dataset. See
`artifacts/v31_selected_source_manifest/report.md`.

## Selected-Universe Causal Dataset v32

- [x] Reverify all 1,928 accepted v31 cache files and source hashes.
- [x] Materialize the 30-asset, 2,007-day calendar panel with 1,619 gaps preserved.
- [x] Freeze eight per-asset features and two causal forward labels.
- [x] Index every valid 256-day sequence without imputing missing observations.
- [x] Freeze all train/test-role triplet combinations inside each asset fold.
- [x] Implement an on-demand `[256, 3, 9]` triplet tensor and `[3, 2]` label loader.
- [x] Verify within-triplet relative strength is same-date and zero-sum.
- [x] Reproduce panel and sequence-index Parquet files byte-for-byte.
- [x] Keep scaling, model training, targets, portfolio, performance, and PnL disabled.

V32 produced 60,210 panel rows and 49,919 sequence-index rows. Each fold has
1,140 train-role and 120 test-role lexical triplets, with eligible samples in
every frozen chronological window. Panel SHA-256 is
`dc8d50af79a9272a25f952cfd266e461ee938d60d8a19654b9eedd93a4ac5f3a`.
V33 may implement only the frozen model and checkpoint contract. See
`artifacts/v32_selected_universe_dataset/report.md`.

## Patch Transformer Implementation v33

- [x] Implement 16-day patches with stride 8 over the frozen 256-day input.
- [x] Implement a shared three-layer causal temporal encoder.
- [x] Implement one asset-order-equivariant cross-asset attention layer.
- [x] Implement q10/q50/q90 and seven-day log-volatility heads.
- [x] Implement the masked-patch reconstruction interface.
- [x] Verify shapes, causal-prefix invariance, permutation equivariance, and gradients.
- [x] Freeze and exactly replay the synthetic checkpoint metadata contract.
- [x] Keep real data, scaling, optimizer steps, training, targets, performance, and PnL disabled.

V33 implements 380,276 parameters and 31 temporal patches. The synthetic
checkpoint and every smoke artifact replay identically; model-spec SHA-256 is
`15cfe72e754ea83e198ba9d9f7176a353303309f41754b08ddd5d4ee43315594`.
V34 is now the final engineering gate before v35 pretraining. See
`artifacts/v33_patch_transformer/report.md`.

## Scientific Training Harness v34

- [x] Freeze train-only scaling for eight base features and relative-strength scaling.
- [x] Freeze uniform deterministic sampling over eligible date-triplet pairs.
- [x] Freeze exactly 14 masked patches per sample at the registered 15% rate.
- [x] Implement reconstruction, quantile, and log-volatility losses.
- [x] Smoke AdamW, gradient clipping, and five-epoch early stopping.
- [x] Validate q-policy, dual-momentum, and equal-weight accounting after costs.
- [x] Run 10,000-path paired circular block bootstrap at 7/21/63-day blocks.
- [x] Replay every harness artifact without real market data or target assets.

V34 passed 28 audit checks and authorizes full non-target pretraining. Harness
spec SHA-256 is
`fcebe639a6c1ec89c539e88104b1fd04fa64dbaa295e951399215a0fc3847066`.
The system is now ready to train. V35 must pretrain all three seeds within all
three frozen asset folds without selecting winners. See
`artifacts/v34_scientific_harness/report.md`.

## Non-Target Masked Pretraining v35

- [x] Fit one 2021-2023 train-asset-only scaler per frozen asset fold.
- [x] Train all three registered seeds in all three folds without selection.
- [x] Use fixed 2024 feature-only validation and restore each best state.
- [x] Persist epoch-granular resume checkpoints and immutable final hashes.
- [x] Reopen all nine final checkpoints and verify finite model state/metadata.
- [x] Keep labels and BTC/ETH/SOL unloaded; exclude held-out assets from each job.

V35 completed all nine fold-seed jobs for 450 epochs and 28,800 optimizer
steps. The best feature-only validation losses range from 0.12126 to 0.13101.
Checkpoint/spec hashes and all histories are frozen under
`artifacts/v35_non_target_pretraining/`; model binaries remain generated data
under `data/checkpoints/`. Pretraining-spec SHA-256 is
`b5229279d62d36502a14106b329f7fbbb30cb0de7c0b00ef2e4cf8bd1c424e56`.
V36 may run only the frozen supervised non-target phase across the same nine
jobs, without using held-out asset folds or loading target assets.

## Supervised Non-Target Training v36

- [x] Initialize every fold-seed job from its exact v35 checkpoint.
- [x] Train all nine jobs on 2021-2023 train-asset labels with an eight-day maturity purge.
- [x] Early-stop independently on fixed 2024 train-asset validation samples.
- [x] Retain every best checkpoint without selecting a seed or fold.
- [x] Calibrate each three-seed ensemble on fixed 2025 train-asset samples only.
- [x] Hash checkpoint files, deterministic tensor states, calibration states, and calibration semantics.
- [x] Verify byte-identical replay of every consolidated v36 artifact.
- [x] Keep held-out assets, BTC/ETH/SOL, portfolio construction, and performance metrics sealed.

V36 completed 114 total epochs and 7,296 optimizer steps on Apple MPS. All nine
checkpoints and three 8,192-sample calibration states passed 21 audit checks.
Calibrated quantile coverage is approximately 10%/50%/90% in every fold and
the calibrated crossing rate is zero. Supervised-spec SHA-256 is
`8df3aebf5e8bd2e38ce79f64e5d8af163a48a5f463453eaf30dab6ad1542c660`.
This is a training/calibration result, not evidence of trading performance.
V37 may perform the single pre-registered 2026 held-out source-domain test. See
`artifacts/v36_supervised_non_target/report.md`.

## One-Shot Source-Domain Evaluation v37

- [x] Commit the complete inference and gate protocol before loading held-out label values.
- [x] Commit the evaluator and synthetic tests before consuming the holdout.
- [x] Verify nine checkpoints, three calibrations, three folds, and 173 dates in a label-free preflight.
- [x] Average all three seeds and every eligible lexical triplet context per held-out asset.
- [x] Evaluate candidate, dual momentum 30, and equal weight at 10/20/30 bps.
- [x] Run 10,000-path paired circular block bootstrap at 7/21/63 days against both controls.
- [x] Preserve all failed cells and cache the result after exactly one execution.
- [x] Keep BTC/ETH/SOL sealed and perform no training, recalibration, or model selection.

V37 evaluated 4,521 held-out asset-date predictions from 39,580 triplet
contexts across 173 signal dates. At 10 bps the candidate returned -16.44%
with Sharpe -2.943 and max drawdown -16.54%. It lost less than both controls
and passed the point drawdown gates at every cost, but failed the Sharpe gate
at 10/20/30 bps and all six paired-bootstrap p05 gates. The audit passed; the
candidate did not. Decision: `retire_candidate_family_without_target_evaluation`.
No v38-v40 work is authorized for this family. See
`artifacts/v37_source_domain_one_shot/report.md`.

## Frozen Failure Autopsy v37

- [x] Commit the allowed inputs, exact hashes, slices, bins, and limitations before analysis.
- [x] Read only frozen v37 results, predictions, and daily-return artifacts.
- [x] Decompose gross/net PnL, costs, episodes, folds, assets, months, and 1/3/5-day loss concentration.
- [x] Diagnose q10/q50/q90 ranking, top-1/top-3 selection, direction, calibration, and confidence relationships.
- [x] Verify context-count completeness and mark seed/context disagreement unavailable.
- [x] Preserve the v37 retirement, keep BTC/ETH/SOL sealed, and test no alternative policy or threshold.
- [x] Reconcile selected returns, gross/net/cost accounting, and equal-fold aggregation.
- [x] Pass the focused autopsy tests, the full suite, and the immutable-input audit.

The candidate was negative before costs (-14.11% gross, -16.44% net at 10
bps), active on 41/519 fold-days, and profitable in only 12/41 one-day
episodes. Q50 top-1 hit rate was 10.02% versus 11.52% random expectation; mean
daily q50 rank IC was 0.0179, and selected assets lost 1.086% on active dates.
The q10 IC of 0.1100 and volatility diagnostics are risk-structure hypotheses
only, not permission to retest the consumed window. Decision:
`v37_retirement_confirmed_new_ex_ante_family_required`. The next allowed work
is specification of a new ranking/excess-return family; clean confirmation
requires future unseen data. See `artifacts/v37_failure_autopsy/report.md`.

## Ranking/Excess Family Specification v41

- [x] Start a new family without reviving v38-v40 or the retired v37 model.
- [x] Reuse the exact v32 non-target universe, features, folds, and triplet catalog by hash.
- [x] Freeze one 1,231,634-parameter Medium architecture with no size sweep.
- [x] Replace winner/q50 training with pairwise ranking plus continuous triplet excess.
- [x] Freeze train-only excess scaling, three seeds, chronology, maturity purge, and no-selection rules.
- [x] Freeze a momentum cash gate and a 20 bps cost-derived switch hurdle.
- [x] Reserve 2025 held-out assets for one development screen and forbid the consumed 2026 window.
- [x] Keep BTC/ETH/SOL sealed pending a future 180-date non-target confirmation.
- [x] Pass all 28 specification checks without reading the panel, labels, or performance.

V41 freezes `tlm_cross_sectional_rank_excess_medium_v1` under blueprint SHA-256
`dc28004a9419424f6d9e437b9ac8a7bf42f73ec9ceb1892494e280d9240fdf5e`.
Improvement, PnL, and risk remain unknown because no model was instantiated or
trained. V42 may implement and smoke-test only the frozen Medium model, loss,
policy, controls, and checkpoint contract on synthetic data. See
`artifacts/v41_ranking_excess_spec/report.md`.

## Synthetic Ranking/Excess Harness v42

- [x] Instantiate the single frozen 1,231,634-parameter Medium architecture.
- [x] Implement the exact centered pairwise ranking, excess, and volatility losses.
- [x] Verify causal-prefix behavior and permutation equivariance across asset slots.
- [x] Exercise deterministic masked reconstruction and finite optimizer gradients.
- [x] Implement the shared-asset Ridge baseline without asset identity leakage.
- [x] Verify the momentum cash gate, strict switch hurdle, turnover, and 10/20/30 bps costs.
- [x] Save and reload a hash-audited synthetic-only checkpoint.
- [x] Pass all 33 harness checks without opening a real panel, label, target asset, or PnL.
- [x] Reproduce every output byte across two independent runs.

V42 passed under harness SHA-256
`36551aaa94b516dac08dd27ed08216ab96c7deda79926423086606cd5f9ba83d`.
The harness executed exactly two synthetic optimizer steps and verified the
frozen model, objective, baseline, policy, accounting, and checkpoint contracts.
It makes no performance claim. The only authorized next action is V43
masked-patch pretraining on non-target representation data, with all supervised
labels, held-out assets, BTC/ETH/SOL, predictions, and PnL still forbidden. See
`artifacts/v42_ranking_excess_harness/report.md`.

## Ranking/Excess Medium Pretraining v43

- [x] Commit the V43 data-access, initialization, training, checkpoint, and MPS contract before reading the panel.
- [x] Pass a metadata-only preflight without opening either Parquet file.
- [x] Read each fold through projected columns and pushed-down train-symbol/date filters only.
- [x] Prove loaded symbols equal the 20 train assets and exclude held-out and target assets.
- [x] Fit one immutable 2021-2023 feature scaler per fold and use 2024 only for feature reconstruction validation.
- [x] Initialize all nine Medium models from fresh registered seeds; reject V35/V36/V42 parents.
- [x] Train only temporal reconstruction parameters with no cross-asset/head forward, gradient, or update.
- [x] Preserve deterministic CPU/MPS RNG state in exact-job resume checkpoints.
- [x] Pass the one-job MPS smoke without labels, predictions, selection, performance, or PnL.
- [x] Complete and audit all 3 folds by 3 seeds, retaining all nine checkpoints.

V43 passed all 25 full-run checks under pretraining-spec SHA-256
`752a1f351232160256ad0416fa51ad0fa5eb9c5b00d0d6e64f92c5ece6a0f774`.
The nine retained checkpoints completed 394 epochs and 25,216 optimizer steps;
an idempotent rerun reproduced result SHA-256
`a30c7aa5cbf692ff552a7c839508b50bf6ecc35899f4ce17a3b682906a1e7958`
without training again. V43 is representation learning only and cannot establish
ranking quality or economic value. The only authorized next action is V44
supervised training on the frozen non-target train folds.

## Ranking/Excess Supervised Non-Target Training v44

- [x] Commit the V44 parent, label-access, objective, parameter-scope, checkpoint, and MPS contract before reading label values.
- [x] Pass a metadata/checkpoint-only preflight without opening either Parquet file.
- [x] Read features, train labels, validation labels, and sequence keys through separate fold-local projections and filters.
- [x] Prove every materialized row belongs to the fold's 20 train assets and no signal or target maturity exceeds 2024.
- [x] Reuse the immutable V43 feature scaler and fit one train-only excess RMS scale by full eligible-triplet enumeration per fold.
- [x] Initialize every job from its exact matching V43 fold/seed checkpoint.
- [x] Train all model parameters except `mask_token` and `reconstruction_head`; prove both frozen states remain parent-exact.
- [x] Optimize the exact ranking + excess + 0.1 log-volatility objective and early-stop on ranking + excess only.
- [x] Preserve exact current/best model, AdamW state, CPU/MPS RNG, patience, history, and optimizer-step coherence in resumes.
- [x] Pass the one-job MPS smoke without held-out assets, target assets, 2025, predictions, performance metrics, or PnL.
- [x] Complete all 3 folds by 3 seeds and retain all nine checkpoints without seed/fold selection.

V44 may inspect only supervised-train and 2024 validation labels for each fold's
20 train assets. It cannot establish held-out ranking quality or economic value.
V44 passed all 35 full-run checks under supervised-spec SHA-256
`a0feb135a76bd7c4f8fa0162acf9d7ecaf821863f7a329bc2e0f7bc7b98e7e26`.
The nine retained checkpoints completed 55 epochs and 3,520 optimizer steps.
Every fold scale used all 914,280 eligible train triplets; all checkpoints
reopened semantically with zero resume artifacts. An idempotent rerun reproduced
result SHA-256
`4f905d00224b5a511ff811d930b7add5822bf1e26cb870f7609b09dba653f5f8`.
This is a supervised-training result, not held-out ranking or economic evidence.
The only authorized next action is the frozen one-shot V45 development screen
on the ten held-out 2025 assets per fold; BTC/ETH/SOL remain sealed.

## Ranking/Excess Asset-Disjoint Development Screen v45

- [x] Freeze the V45 predictive, policy, accounting, bootstrap, Ridge, data-access, and lifecycle contract before opening held-out 2025 outcomes.
- [x] Pass metadata/checkpoint preflight with zero Parquet deserialization.
- [x] Fit three fixed shared-asset Ridge controls from 8,192 train-only triplets per fold.
- [x] Freeze all Transformer and Ridge context/asset predictions before reading held-out outcomes.
- [x] Prove prepare reads no held-out label column, train asset in screen inference, target asset, or 2026 row.
- [x] Execute the one-shot outcome unseal for exactly 2025-01-01 through 2025-12-23.
- [x] Evaluate triplet-level ranking/excess diagnostics after daily/fold aggregation.
- [x] Evaluate long/cash policy and both controls at 10/20/30 bps with final liquidation.
- [x] Run 10,000-path circular block bootstraps at 7/21/63 days.
- [x] Preserve every predictive, economic, drawdown, turnover, and bootstrap gate cell.
- [x] Cache the immutable result and retire or freeze the family exactly as preregistered.

V45 has no real-outcome smoke. Its `prepare` phase may read train labels for the
fixed Ridge and held-out features/sequence readiness, but it must freeze every
prediction before the `evaluate` phase opens any held-out 2025 return. Any
failed gate retires the family without BTC/ETH/SOL evaluation or parameter
tuning. A pass still requires a later prospective non-target confirmation.

V45 completed under evaluation-spec SHA-256
`f7c6cd57555feb9496e77dc74321320dfaf6290fd5480fe3405f0fa8859a6888`.
Preflight reopened all nine checkpoints with zero Parquet deserializations.
Prepare fit three train-only Ridge models from 24,576 sampled triplets and
froze 96,864 context predictions, 9,782 eligible asset-dates, and 10,710
position rows with zero held-out outcomes. The one-shot evaluation opened
exactly 9,782 registered 2025 outcomes and passed 36/39 gates. Predictive gates
all passed: Transformer aggregate Spearman was `0.0546`, pairwise accuracy
`52.48%`, and top-1 excess `0.0986%`, above Ridge on the registered aggregate
comparisons. The candidate returned `15.34%` at 10 bps, but fold 3 returned
`-10.55%`; aggregate drawdown was `-36.20%` at 20 bps and `-37.67%` at 30 bps.
Those three failures trigger the frozen decision
`retire_family_without_target_evaluation_or_parameter_tuning`. Result SHA-256
is `b2d76e82770fefbce86e1f7a51cf5dd33b3e6c1e6e73db314cb9fe048bbe8f4a`;
an idempotent rerun reproduced the complete artifact-directory digest
`604fbebb5bd45189596d3e1c84fb29e50452143ce79a5e68b7c6ef33891b3c0e`.
BTC/ETH/SOL and all 2026 outcomes remain sealed. This family may be analyzed
diagnostically, but it may not be retuned or evaluated on target assets.

## Ranking/Excess Failure Autopsy v46

- [x] Freeze the 20-file V45 hash allowlist and immutable retirement lineage.
- [x] Pass metadata/hash preflight with zero Parquet deserializations.
- [x] Reconcile all frozen context, daily predictive, fold, aggregate, cost, and final-liquidation ledger cells.
- [x] Decompose the 10 bps result by fold, asset, month, holding episode, position state, and signal-time regime.
- [x] Diagnose triplet ranking, context-averaged asset ranking, held-asset outcomes, and context stability without re-inference.
- [x] Preserve every registered 10/20/30 bps cost and drawdown cell and every empty registered group.
- [x] Prove all 30 non-target assets are represented while BTC/ETH/SOL and post-2025 observations remain sealed.
- [x] Record unavailable seed disagreement and predicted volatility without reconstructing either quantity.
- [x] Seal and cache a reproducible diagnostic packet with the V45 retirement unchanged.

V46 is retrospective diagnosis, not a candidate experiment. It may use only the
hash-exact frozen V45 packet. It cannot train, infer, recalibrate, alter a model,
exclude an asset/date/fold, test a threshold or policy, compute counterfactual
PnL, rerun a bootstrap, or convert a diagnostic slice into promotion evidence.

V46 passed under autopsy-spec SHA-256
`d96a9cf45495d991f56596e665ae0d6d61d1be595830f552327dd4c7781695ea`.
The preflight hash-verified 20 frozen inputs and deserialized zero Parquets. The
run reconciled 96,864 contexts, 9,782 asset-dates, 10,710 positions, 1,071
fold-dates, 357 dates, all 10/20/30 bps cells, and 97 holding episodes. At 10
bps the candidate returned `23.06%` gross and `15.34%` net. Fold 3 was already
negative before costs (`-3.68%` gross, `-10.55%` net): its held assets retained
positive relative excess but negative absolute return. Exposure was dominated
by BCH, BNB, and TRX in folds 1, 2, and 3; momentum-regime associations reversed
sign across folds; the worst day explained only `4.36%` of losing-day magnitude.
Result SHA-256 is
`9948487a8a62a3b862e995602b71d57796581961a2b3cf366c37e719082d47ce`;
an idempotent rerun reused the sealed packet. V45 remains retired, BTC/ETH/SOL
remain sealed, and any continuation requires a new ex-ante family plus genuinely
new non-target observations.

### V60 — Decoupled V45 rank/state family specification

- [x] Verify the explicit user authorization and all immutable JSON input hashes.
- [x] Preserve V45 as retired scientific ancestry with zero checkpoint reuse.
- [x] Freeze the exact 1,231,634-parameter V45 ranker without a size sweep.
- [x] Freeze one independent 27,489-parameter absolute market-state gate.
- [x] Prove no shared parameters, combined loss, or cross-module gradient path.
- [x] Freeze the nine-job fresh training grid and cost-aware long-one/cash policy.
- [x] Mark all historical windows as adaptive development evidence only.
- [x] Keep BTC/ETH/SOL sealed and require future prospective non-target evidence.
- [x] Reproduce the metadata packet byte-identically with zero prohibited access.

**V60 gate:** every metadata, lineage, capacity, isolation, chronology, hash, and
replay check must pass. Passing authorizes only the V61 synthetic harness; it
does not authorize real data access, training, evaluation, or target assets.

V60 passed under blueprint SHA-256
`3cf97271c977633b1ee8ab4641a8a9fc95a1a9d79b72c80c063d21f553498232`.
The ranker has 1,231,634 parameters, the independent state gate has 27,489,
and the frozen total is 1,259,123 under the 1.3M ceiling. Two executions
reproduced all nine packet files byte-identically. Every prohibited access
counter remained zero. Only V61 synthetic harness work is authorized next.

### V61 — Synthetic decoupled rank/state harness

- [x] Instantiate the exact V60 components only on deterministic synthetic data.
- [x] Prove causal prefix and asset-permutation behavior for both modules.
- [x] Prove parameter identity and gradient isolation in both directions.
- [x] Verify independent optimizers, checkpoint roundtrip, and exact resume.
- [x] Verify absolute-return decomposition and cost-aware policy accounting.
- [x] Reproduce the full synthetic packet byte-identically with zero real-data,
      prior-checkpoint, outcome, performance/PnL, or target-asset access.

**V61 gate:** all synthetic scientific and accounting checks pass. Passing can
authorize only a separate V62 non-target dataset phase; it does not authorize
real training, evaluation, or target assets.

V61 passed 20/20 audit checks with harness SHA-256
`6961ed2b0d5479c9fab8ffa9dc4d27ee18cda96103151ada72090719634275a0`.
It instantiated the exact 1,231,634-parameter ranker and independent
27,489-parameter gate, proved two-way gradient isolation, resumed both
optimizers exactly, and reproduced the complete packet byte-identically across
two executions. All real-data, prior-checkpoint, outcome, performance/PnL, and
target-access counters remained zero.

### V62 — Non-target decoupled rank/state dataset

- [x] Verify all 14 registered V61, V60, and V32 input hashes.
- [x] Deserialize only the exact V32 non-target panel and sequence index.
- [x] Materialize the frozen `open[t+1] -> open[t+2]` log-return labels.
- [x] Bind train and consumed-development-validation roles without boundary
      crossing.
- [x] Preserve missing observations without imputation or repair.
- [x] Verify centered excess, triplet market component, and all 18 independent
      state features.
- [x] Prove BTC/ETH/SOL are absent and no scaler, model, optimizer, checkpoint,
      prediction, metric, outcome, or PnL operation occurred.
- [x] Reproduce the dataset packet byte-identically.

**V62 gate:** a hash-exact, causal non-target dataset packet passes every check.
Passing may authorize only a separately frozen V63 training phase. V62 itself
cannot instantiate or train a model and cannot access BTC/ETH/SOL.

V62 passed 15/15 audit checks under dataset-spec SHA-256
`82d6f649cde0eb64a752e18c239c27488ecc0530d383e2fc8743256b73cd5794`.
It preserved all 60,210 label rows and 49,919 sequence-role rows, including
1,619 missing panel rows without imputation. There are 34,934 eligible train
rows and 9,794 consumed-development-validation rows. Six deterministic triplet
contexts covered all three folds and both roles. Two complete executions
reproduced all packet and Parquet hashes byte-identically. The access ledger
recorded two authorized Parquet deserializations and zero scaler, model,
optimizer, checkpoint, prediction, performance/PnL, or target operations.

### V63 — Frozen non-target decoupled rank/state training

- [x] Require a clean committed tree and phase-specific storage-policy receipt.
- [x] Preflight MPS float32 determinism with fallback disabled.
- [x] Fit exactly three fold/train-only feature and target-scaling packets.
- [x] Pretrain the exact ranker encoder from fresh registered seeds on masked
      patches; reuse no prior-family representation or checkpoint.
- [x] Train exactly nine `fold x seed` jobs with separate ranker and state-gate
      optimizers, losses, gradients, and early stopping.
- [x] Exclude every held-out-fold asset from scaler, pretraining, training, and
      validation materialization.
- [x] Pass interrupted/resume smoke equivalence before the full grid.
- [x] Retain all nine final checkpoints without seed/fold selection.
- [x] Verify checkpoint roundtrips, complete-grid manifest, and zero-step replay.
- [x] Prove zero prediction, portfolio, performance/PnL, target, or V64 access.

**V63 gate:** all nine fresh non-target jobs and checkpoint receipts are complete
and replayable under the frozen contract. Passing may authorize only a separate
V64 adaptive development evaluation; it does not authorize target assets or a
deployable claim.

V63 passed its terminal audit under result SHA-256
`99d367e761028fa570c060e0b1d0c6164bac9d6158d1cb122a9706d22e29f5d8`.
All nine checkpoints completed and were retained after 34,496 total optimizer
steps. The MPS smoke proved exact interruption/resume in both pretraining and
supervised stages. All nine checkpoints reopened semantically; replay created
zero jobs, optimizer steps, or checkpoint rewrites. No prediction,
performance/PnL, heldout-fold training, target asset, or V64 operation occurred.

### V64 — Adaptive-development decoupled rank/state evaluation

- [x] Implement only the hash-bound outcome-blind prepare contract.
- [x] Verify all nine V63 checkpoints and infer only each fold's ten held-out
      non-target assets.
- [x] Freeze all context/seed/asset forecasts and policy outputs before outcomes.
- [x] Open the exact registered 2025 adaptive outcome packet exactly once.
- [x] Preserve every fold, seed, asset, date, context, and 10/20/30 bps cell.
- [x] Keep BTC/ETH/SOL sealed and label all results consumed adaptive evidence.

**V64 gate:** V64 can report predictive and economic diagnostics, but cannot
establish clean holdout evidence, prospective confirmation, deployability, or
target authorization.

V64 completed the exact user-authorized one-shot unseal under evaluation-spec
SHA-256 `f6fbf371b5e33efdaaf0b0d1622acefce4938efe1e22d63ffa6e086e0a45d134`.
It opened 9,794 registered non-target outcomes once, then replayed from the
immutable outcome packet with zero source reads, inference, or position
generation. Relative prediction was positive in all folds: aggregate Spearman
was `0.0652`, pairwise accuracy `52.96%`, and top-1 centered excess `0.1208%`.
The frozen policy nevertheless lost `15.07%` at 10 bps, `20.78%` at 20 bps,
and `30.21%` at 30 bps; all three folds lost money at 10 bps and all three
economic bootstrap lower bounds were negative. The registered matrix passed
19/36 gates, so the immutable decision is
`retire_family_without_target_evaluation_or_retuning`. BTC/ETH/SOL remain
sealed and the family is not deployable.

### V65 / V64-R2 — Probabilistic state-gate specification

- [x] Register the exact owner authorization and direct-successor boundary.
- [x] Preserve the V64 retirement decision as immutable.
- [x] Freeze the exact V64 ranker architecture, objective, and nine ranker-state
      identities without opening checkpoint files.
- [x] Replace only the point state gate with one fixed Student-t location/scale
      gate.
- [x] Freeze one 60% probability-of-clearing-cost abstention threshold with no
      distribution or threshold sweep.
- [x] Preserve the long-one/cash action space, momentum gate, asset ranking,
      switch hurdle, cost grid, leverage, and shorting rules.
- [x] Prove zero data, Parquet, checkpoint, model, prediction, metric, PnL,
      outcome-source, or BTC/ETH/SOL access.
- [x] Reproduce the full metadata packet byte-identically.

**V65 gate:** every authorization, lineage, frozen-ranker, probabilistic-gate,
abstention, capacity, sealed-target, access, and replay check must pass. Passing
may authorize only a V66 synthetic harness; it does not authorize data access,
checkpoint deserialization, training, inference, outcomes, or target assets.

V65 passed every audit check under blueprint SHA-256
`00952d655a0c78ff633bd14b0264d77a17a392fc7c7e3633b77fde3dcb390f33`.
The exact 1,231,634-parameter ranker is frozen by nine fold/seed state hashes;
the new gate has 27,522 parameters and the total is 1,259,156. The complete
nine-file packet replayed byte-identically. Every prohibited-access counter was
zero. Only the V66 synthetic harness is authorized next.

### V66 — Synthetic V64-R2 probabilistic state-gate harness

- [x] Instantiate the exact frozen ranker architecture on synthetic data only.
- [x] Instantiate the exact two-output Student-t state gate and prove positive
      scale, fixed degrees of freedom, and correct negative log-likelihood.
- [x] Prove the ranker remains gradient-free and has no optimizer while the new
      gate trains independently in the synthetic harness.
- [x] Verify exact 60% abstention boundaries for entry, hold, switch, and exit at
      each transition-specific cost.
- [x] Preserve momentum gating, relative asset ordering, the 0.002 switch
      hurdle, long-one/cash action space, costs, and final liquidation.
- [x] Prove causal-prefix, asset-permutation, checkpoint roundtrip, interrupted
      resume, and byte-identical replay behavior.
- [x] Prove zero real-data, prior-checkpoint, V64 gate-state, real prediction,
      metric/PnL, outcome-source, or BTC/ETH/SOL access.

**V66 gate:** every synthetic architecture, distribution, gradient-isolation,
policy-boundary, accounting, resume, access, and replay check must pass. Passing
may authorize only a separate V67 non-target dataset phase; it does not
authorize real data access, training, evaluation, outcomes, or target assets.

V66 passed 22/22 audit checks under harness SHA-256
`7f5e4f4787c38567d85870ce7722a302a42e19238a60ce4c90ce0fcc75740e25`.
The ranker remained frozen with no optimizer, the 27,522-parameter gate trained
for only four bounded synthetic optimizer steps across uninterrupted and resume
paths, and policy actions were exactly `entry, hold, switch, hold,
probability_exit, cash, entry`. The complete 12-file packet, including the
synthetic checkpoint, replayed byte-identically. All prohibited-access counters
were zero. Only V67 dataset construction is authorized next.

### V67 — Non-target V64-R2 gate dataset

- [x] Verify the exact V66, V65, and V62 metadata and data hashes.
- [x] Deserialize only the exact V62 labels and sequence-role Parquets with
      projection and predicate pushdown.
- [x] Admit only eligible V62 training rows through `2024-12-23` and load no
      consumed-development or 2025 value.
- [x] Freeze disjoint chronological `gate_train` (`2021-03-01` through
      `2024-06-30`) and internal validation (`2024-07-01` through `2024-12-23`).
- [x] Preserve the H1 open-to-open market-component target, 18 state-feature
      derivation, missing rows, universe, folds, and triplets without reselection.
- [x] Prove BTC/ETH/SOL are absent and no scaler, model, optimizer, checkpoint,
      prediction, metric/PnL, or outcome operation occurs.
- [x] Reproduce the dataset packet and both derived Parquets byte-identically.

**V67 gate:** every hash, predicate, chronology, role, label, missingness,
sealed-target, access, and replay check must pass. Passing may authorize only a
separate V68 frozen non-target gate-training phase; it does not authorize
training, evaluation, outcomes, or target assets.

V67 passed 15/15 checks under result SHA-256
`cca1e5d978038bd2f8eafff606fe3d492a9491578c86aad93d1898064ec115ae`.
Predicate pushdown loaded 41,820 label rows and 34,934 eligible V62 training
rows, admitted 29,700 gate-train and 5,174 internal-validation rows, and
embargoed 60 split-boundary rows whose H1 maturity crossed into validation.
The complete packet and both derived Parquets replayed byte-identically. The
ledger recorded exactly two Parquet deserializations and zero 2025, scaler,
model, optimizer, checkpoint, prediction, metric/PnL, outcome, or target access.

### V68 — Frozen non-target V64-R2 gate training

- [x] Verify the exact V65/V66/V67, V63 ranker-state, scaler, fold, triplet,
      feature-panel, dataset, and checkpoint hashes.
- [x] Preflight deterministic Apple MPS float32 with fallback disabled.
- [x] Reopen each exact V63 checkpoint only to extract its registered frozen
      ranker substate and fold feature scaler; never load or reuse the old gate.
- [x] Fit only the three gate-train market-target RMS scalers.
- [x] Train exactly nine fresh Student-t probabilistic gate jobs for the frozen
      `3 folds x 3 seeds` grid, with no ranker gradient or optimizer.
- [x] Early-stop only on the frozen internal-validation Student-t NLL and retain
      every job without seed, fold, checkpoint, or epoch selection.
- [x] Pass interrupted/resume smoke, semantic checkpoint roundtrip, complete
      grid verification, and zero-step replay.
- [x] Prove no held-out-fold asset, 2025 value, prediction, position, policy,
      metric/PnL, outcome, or BTC/ETH/SOL access.

**V68 gate:** all nine gate-only jobs and exact frozen-ranker receipts must be
complete and replayable. Passing may authorize only a separate outcome-blind
prospective non-target confirmation preparation phase; it does not authorize
evaluation, outcomes, target assets, paper/shadow/live trading, or a deployable
claim.

V68 passed under result SHA-256
`a77788d338db995c44ce6d3d765261b1c46383895c3f56bcb31016fb2d84d8e1`.
All 9/9 checkpoints completed with 3,776 gate optimizer steps and zero ranker
optimizer steps. Interrupted/resume smoke, semantic roundtrip, exact ranker
identity checks, and zero-job/zero-step/zero-rewrite replay all passed. The
ledger recorded nine authorized V63 container reads, no old gate applied to a
model, and zero predictions, metrics, PnL, outcomes, or target loads.

### V69 — Outcome-blind prospective confirmation preparation

- [x] Verify only the 13 registered V65/V68/V32 metadata receipts.
- [x] Freeze one future non-target confirmation protocol before any data access.
- [x] Require the first admissible signal strictly after the V69 completion
      receipt commit; never relabel 2025 or pre-registration 2026 as prospective.
- [x] Freeze capture timestamps, minimum window, mandatory cells, costs,
      accounting, prediction-freeze receipt, outcome packet, and one-shot rules.
- [x] Prove zero Parquet, raw data, checkpoint, model, inference, prediction,
      position, metric/PnL, outcome, or BTC/ETH/SOL access.
- [x] Replay the complete metadata packet byte-identically.

**V69 gate:** a complete ex-ante prospective protocol may authorize only a
separate V70 non-target capture and prediction-freeze phase. It cannot authorize
outcomes, financial claims, target assets, or deployment.

V69 passed 24/24 audit checks. It froze a 120-day minimum clock, at least 90
fully matured dates and 20 active-position dates per fold, a 365-day maximum,
the exact three folds and three-seed aggregation, costs 10/20/30 bps, and all
36 mandatory gates. All eight packet files replayed byte-identically with zero
data, checkpoint, model, prediction, position, outcome, metric/PnL, or target
access.

### V70 — Prospective non-target capture and prediction freeze

V70-R1 metadata-only repair authorized on 2026-07-15. The original allowlist
omitted the frozen `ranker_excess_rms` values required to denormalize ranker
outputs into the unchanged policy/cost units. Before any market data,
prediction, position, or outcome access, the omission was registered as an
incident, the exact three V63 values were projected into a hash-bound receipt,
and prospective admissibility was re-anchored to the first commit containing
the exact V70-R1 amendment. No model, scaler fit, policy, threshold, cost,
accounting, universe, checkpoint, target, or outcome changed.

- [ ] Resolve the first Git commit registering the exact V70-R1 amendment and
      admit only feature-candle closes strictly after that timestamp.
- [ ] Fetch only public unauthenticated daily UTC klines for the exact V32
      non-target fold symbols; preserve prior history only as causal lookback.
- [ ] Verify all nine V68 checkpoints and three scalers without training,
      refit, selection, or mutation.
- [ ] Generate the exact context/seed-averaged predictions and long-one/cash
      positions, freezing each immutable packet before its h1 maturity.
- [ ] Append only integrity, source, prediction, and position receipts; never
      rewrite a frozen signal date.
- [ ] During accumulation expose only counts, hashes, continuity, activity and
      maturity timestamps—never outcomes, returns, metrics, PnL, equity,
      drawdown, bootstrap cells, or gate results.
- [ ] Continue until every fold has at least 120 calendar days, 90 fully
      matured signal dates, and 20 active-position dates, or retire as
      inconclusive at 365 days.

**V70 gate:** a complete immutable prospective prediction/position packet may
authorize only a separate V71 outcome-blind one-shot preparation phase. V70
cannot open outcomes, compute financial performance, access target assets, or
perform paper, shadow, live, or real-money trading.

Owner direction on 2026-07-15 paused the 120-day wait as the primary path. The
V70 implementation and immutable zero-packet initialization evidence remain
preserved; they are not reclassified as a completed prospective confirmation.

### V71 — Outcome-blind post-hoc 2025 diagnostic preparation

V71-R1 records a pre-prepare schema-probe mistake: two rows dated 2021-01-01
were displayed with two training-era target columns while inspecting the
general panel. No 2025 evaluation outcome, sealed outcome packet, target asset,
prediction, position, metric, or PnL was accessed. The actual prepare is
reanchored after this incident and all panel reads must explicitly project only
the registered feature columns.

- [ ] Verify the exact owner authorization, V68 checkpoints/scalers, V32 folds
      and triplets, V64 feature/readiness inputs, and frozen V64 control
      positions.
- [ ] Reproduce the exact 2025 non-target signal scope from `2025-01-01`
      through `2025-12-23` without opening the registered outcome packet.
- [ ] Run all nine V68 checkpoints without training, selection, optimizer,
      scaler fit, or parameter mutation.
- [ ] Freeze V64-R2 candidate positions, the exact frozen V64 control, cash,
      and equal-weight controls at 10/20/30 bps.
- [ ] Freeze the economic accounting, 10,000-path circular block bootstrap,
      7/21/63-day blocks, and mandatory diagnostic cells before outcomes.
- [ ] Produce and validate one outcome-blind one-shot prepare packet with zero
      outcome rows, metrics/PnL, or BTC/ETH/SOL access.

**V71 gate:** a complete prepare packet may authorize only a separate V72
hash-bound opening of the already immutable V64 outcome packet. The user must
authorize that exact spec and prepare receipt after V71 passes. V71 results are
post-hoc consumed-development diagnostics, never clean prospective, target,
deployable, paper, shadow, live, or real-money evidence.
