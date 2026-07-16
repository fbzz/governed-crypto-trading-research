# TLM research status

This file is a human-readable projection of `research/current.yaml`.

- Active family: `tlm_low_turnover_cross_sectional_rank_v1`
- State: V84 outcome-blind prepare passed; exact hash-bound V85 one-shot
  economic evaluation is authorized but has not opened outcomes yet
- Governed families: 8 total; 8 trained, 6 retired, with no deployable family
- Scientific predecessor: V64-R2; its failed post-hoc diagnostic, unchanged
  family status, and consumed evidence are immutable
- Architecture: one 10,993-parameter causal depthwise-TCN ranker with shared
  asset encoding, DeepSets-style cross-asset context, and centered scores
- Objective: SmoothL1 centered 21-day excess return plus `0.50` pairwise
  logistic ranking; no PnL or turnover loss
- Policy: long one asset or cash, with a fixed 21-day decision clock, 63-day
  market gate, `0.25` switch margin, final liquidation, and structural turnover
  at most `16.0`
- Training: one frozen 3-fold x 3-seed MPS grid; no architecture, size,
  threshold, seed, fold, or hyperparameter selection
- V83 sampling: 12,120 deterministic balanced train contexts per epoch and
  19,380 fixed validation contexts; target RMS uses full lexical train-triplet
  enumeration, while every fold/seed checkpoint is retained
- V83 result: 9/9 fresh checkpoints retained, 5,040 optimizer steps, best epochs
  `1/1/1`, `2/5/2`, and `1/1/1` across folds 1-3 and seeds 42/7/123;
  interrupted/resume equivalence, semantic verification, complete-grid verify,
  and zero-step replay all passed
- V83 access audit: zero predictions, positions, policy actions, performance
  metrics, PnL, outcomes, prior checkpoints, or target-asset loads; evaluation
  features and the sealed 2026 outcome packet were not opened
- Target assets: BTCUSDT, ETHUSDT, SOLUSDT remain sealed
- Deployable, paper, shadow, or live strategy: none
- V64 result: positive relative ranking diagnostics but 19/36 gates passed and
  negative net return at 10/20/30 bps; terminal decision remains retirement
- V64 postmortem: state calibration was the primary conversion failure; short
  episodes and turnover amplified it
- V65 result: 23/23 specification audit checks passed; nine ranker-state
  identities frozen; all nine output files replayed byte-identically
- V65 access audit: zero Parquet, checkpoint, model, optimizer, prediction,
  metric/PnL, outcome-source, or target-asset access
- V66 result: 22/22 checks passed; exact causal/permutation behavior, Student-t
  NLL/CDF, gate-only optimizer, interrupted resume, 60% abstention, transition
  costs, and full packet replay were verified
- V66 access audit: six metadata reads, one synthetic checkpoint write/read,
  four synthetic gate steps, and zero prohibited real or target operations
- V67 result: 15/15 checks passed; 41,820 projected labels, 29,700 gate-train
  rows, 5,174 internal-validation rows, and a two-day split embargo over 60 rows
- V67 access audit: exactly two predicate-pushed Parquet reads; zero 2025,
  scaler, model, optimizer, checkpoint, prediction, metric/PnL, outcome, or
  target operations; full packet and both Parquets replayed byte-identically
- V68 result: 9/9 gate-only checkpoints, 3,776 gate optimizer steps, zero ranker
  steps; interrupted/resume, semantic roundtrip, exact-ranker verification, and
  zero-step replay all passed
- V68 access audit: nine authorized V63 container reads; no old gate was applied,
  inspected, selected, or reused; zero predictions, metrics/PnL, outcomes, or
  target operations
- V69 result: 24/24 checks passed; the 120-day/90-matured-date/20-active-day
  prospective clock, 365-day maximum, three folds, fixed seed/context
  aggregation, costs 10/20/30 bps, and exact 36-gate matrix are frozen
- V69 access audit: exactly 13 metadata reads and zero data, checkpoint, model,
  prediction, position, outcome, metric/PnL, or target operations; all eight
  packet files replayed byte-identically
- V70 runner and its zero-packet initialization evidence remain immutable, but
  the owner paused the 120-day wait as the primary path; V70 is not completed
  and no prospective claim is made
- V71 froze candidate/control positions, costs 10/20/30 bps, accounting,
  bootstrap, and gates using all nine frozen V68 checkpoints and the exact V65
  policy; all outcome-blind gates and the zero-step replay passed
- V71-R1 registered a pre-prepare projection incident involving two 2021
  training rows; the 2025 outcome window and sealed packet were not opened, and
  the real prepare is reanchored with feature-only column projection required
- V72 completed with exactly one immutable packet unseal, zero underlying source
  rereads, and matching source-free replay hashes
- V72 passed 13/24 gates. At 10 bps the candidate returned `+3.18%` gross and
  `-2.57%` net with Sharpe `-0.20`, maximum drawdown `-12.44%`, and turnover
  `57.33`; the diagnostic failed and family status did not change
- V73 passed with byte-identical replay. Its access ledger contains four JSON
  metadata reads and zero Parquet, outcome packet, checkpoint, model, optimizer,
  inference, prediction, position, or target operations. It authorizes only a
  separate V74 persistent-duration family specification
- V74 passed 16/16 checks and replayed all nine packet files byte-identically.
  It froze 1,083,155 parameters and nine future MPS jobs after four JSON reads,
  with zero Parquet, checkpoint, model, optimizer, prediction, position,
  metric/PnL, outcome, or target access
- V75 passed all 17 checks with 1,083,155 parameters, finite CPU and Mac MPS
  joint backward, exact persistent-policy transition costs, interrupted resume,
  and byte-identical internal plus full-packet replay. Its ledger records six
  metadata reads, six synthetic optimizer steps, one synthetic checkpoint
  write/read, and zero real data, prior checkpoint, prediction, metric/PnL,
  outcome, or target-asset operations
- The V76 registration identified and disclosed a malformed 61-character V74
  sequence-index hash. The authoritative 64-character V32 manifest receipt is
  frozen before dataset deserialization; two post-gate hash-only reads and zero
  Parquet deserializations are recorded. Scientific semantics did not change
- V76 passed 16/16 checks with 43,830 label rows, 43,478 complete persistent
  rows, 24,060 train-eligible rows, 10,628 internal-validation rows, and 9,798
  date-only adaptive-evaluation role rows. The full 12-file packet and both
  Parquets replayed byte-identically; there were zero scaler fits, models,
  optimizer steps, checkpoint reads, predictions, metrics/PnL, or target loads
- V77 passed with nine fresh 1,083,155-parameter checkpoints and 6,976 optimizer
  steps. All three train-only scalers, interrupted/resume equivalence, semantic
  checkpoint verification, complete-grid verification, and zero-step replay
  passed; no predictions, PnL, outcomes, or target assets were opened
- V78 used all nine checkpoints and every exact heldout triplet. Eleven of 12
  behavior gates passed; aggregate turnover was `59.55` against the frozen
  `45.0` ceiling. Independent accounting errors were zero, and no outcome,
  performance metric, PnL, one-shot unseal, or target asset was opened
- V79 passed its seven checks and replayed all eight files byte-identically. It
  read four exact JSON metadata receipts and recorded zero Parquet, checkpoint,
  model, training, inference, prediction, position, metric/PnL, outcome-packet,
  or target-asset operations; the V78 family is retired
- V80 passed 14/14 checks and replayed all nine files byte-identically. It froze
  one 10,993-parameter causal depthwise-TCN/DeepSets ranker, 21-day excess-rank
  target, nine future MPS jobs, 2026 non-target evaluation, nine terminal gates,
  and a 21-day decision clock limiting turnover to `16.0` by construction
- V81 passed 15/15 checks on CPU and Apple MPS. All nine files replayed
  byte-identically; exact 10,993 parameters, interrupted resume, centered and
  permutation-equivariant scores, and adversarial turnover `16.0` passed
- V82-R0 passed 14/14 checks and replayed all eight files byte-identically. The
  overlay changes only `2026-06-09` to `2026-06-08` and 160 to 159 signals;
  final maturity is `2026-06-30`, while eight decisions and turnover `16.0`
  remain unchanged
- Its ledger records eight JSON reads and zero data, Parquet, checkpoint, model,
  outcome, target, training, inference, prediction, position, metric/PnL, or
  bootstrap access
- V82 passed 20/20 checks over 2,393 checksum-verified official non-target
  archives. `AXSUSDT 2026-02` contained duplicate UTC dates and was explicitly
  rejected without deduplication, repair, replacement, or imputation
- V82 produced 75,420 development-feature rows, 70,290 train/validation label
  rows, 8,610 outcome-free evaluation-feature rows, and 5,370 daily rows in the
  physically separate 2026 evaluation-outcome packet
- The V82 metadata packet and all four Parquets replayed byte-identically. The
  outcome packet SHA-256 is
  `9cc5be0e9dfdc40b4fe8d6433602769d67bfa6b269b5f02fa2d241e6eca0024a`;
  its unseal count is zero
- V82 result SHA-256 is
  `87d0b0dcd5e6b6b07be6572e73eb53eb2dffbddfc2b8c254d2144eb2f67cc0e6`.
  All scaler, model, checkpoint, training, inference, prediction, position,
  metric/PnL, bootstrap, and target counters are zero
- V83 passed all 12 terminal audit checks. It retained nine fresh checkpoints
  after 5,040 optimizer steps and replayed with zero new jobs, optimizer steps,
  or checkpoint rewrites. Result SHA-256 is
  `d94bd923444301ead42a82eb5b4f5e8fdb3dd7456ff11db718a47bbdf3278856`
- V83-R2 records the deterministic-MPS LayerNorm incident before official
  training: two development-Parquet deserializations, one fold feature-scaler
  fit, one fold target-scale fit, one model instantiation, and zero optimizer
  steps, checkpoint writes, predictions, outcomes, or target loads. The
  algebraically equivalent deterministic normalization repair preserved the
  frozen architecture, parameter count, objective, grid, and sampling contract
- V84 passed all 12 behavior gates using every registered checkpoint without
  selection. It froze 440,784 predictions, 193,320 candidate-position rows,
  and 579,960 control-position rows; aggregate structural turnover was
  `1.527778` and exposure was `0.260863`.
- V84 replayed by hashes with zero outcome, performance, PnL, bootstrap, or
  target access. V85 is now authorized for exactly one 5,370-row non-target
  outcome-source opening bound to the registered spec, receipt, and contract.
- No economic performance claim exists yet. BTC/ETH/SOL remain sealed, and
  V85 cannot authorize direct target evaluation or deployment.

Run `PYTHONPATH=src python3 -m tlm research-status` for the validated current
state.
