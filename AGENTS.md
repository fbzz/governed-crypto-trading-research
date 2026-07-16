# TLM repository contract

## Mission

Build a reproducible daily multi-asset trading-research system. BTCUSDT,
ETHUSDT, and SOLUSDT are target assets and remain sealed unless the current
machine-readable contract explicitly authorizes access.

## Sources of truth

1. Read `research/current.yaml` before every research action.
2. Read the immutable experiment contract referenced by `current_experiment`.
3. Treat `authorized_next_action` as an exact capability boundary.
4. Use `docs/research/lineage-v1-v55.md` only as historical evidence.
5. If prose conflicts with the machine-readable contract, stop and report
   contract drift.

Never infer authorization from a promising metric, a prior command, or a user
request to continue. A research phase may authorize only its registered next
phase after every gate passes.

## Scientific invariants

- Python only; no live orders, API keys, leverage, shorting, or real-money use.
- Never use random time-series splits or fit scalers outside training data.
- Features at `t` may use only information available by the close of `t`.
- Signals after close `t` may act no earlier than the registered next execution.
- Charge entry, switch, exit, forced exit, and final liquidation turnover.
- Preserve missing observations; never silently impute or repair source gaps.
- Freeze splits, targets, seeds, controls, costs, gates, and evidence roles before
  opening the corresponding data.
- Keep target assets, held-out outcomes, and forbidden columns outside every
  tensor unless the current phase explicitly permits them.
- Persist every registered seed, fold, origin, and geometry without selection.
- Treat training loss and predictive metrics as diagnostics, not economic alpha.
- Never rescue a failed cell with an aggregate result.
- A retired family is immutable and cannot be tuned on its consumed window.

## Experiment lifecycle

`specification -> synthetic harness -> data contract -> training -> outcome-blind prepare -> one-shot evaluation -> registration or retirement`

Every phase must emit a deterministic packet containing the applicable
contract/config, audit, manifest, result, report, input hashes, source receipt,
and explicit next authorization. Replays must be idempotent.

## Current family boundary

The active technical family is
`tlm_low_turnover_cross_sectional_rank_v1`, a newly trained final family whose
V80 specification passed 14/14 checks and froze one 10,993-parameter causal
depthwise-TCN ranker with a structural evaluation-turnover ceiling of `16.0`.
V81 passed all 15 synthetic checks on CPU and Apple MPS and replayed all nine
packet files byte-identically. V82-R0 then passed 14/14 metadata checks and
replayed all eight files byte-identically, correcting only the final evaluation
signal end from `2026-06-09` to `2026-06-08` and its inclusive count from 160
to 159. V82 dataset then passed 20/20 checks over 2,393 checksum-verified
non-target archives, with one corrupt official AXSUSDT month rejected and
preserved as a gap. Its metadata packet and four Parquets replayed
byte-identically; the separate 2026 evaluation-outcome packet remains sealed
with unseal count zero. V83 then passed all 12 terminal checks with nine fresh
10,993-parameter checkpoints, three train-only median/IQR scalers and excess-RMS
target scales, 5,040 optimizer steps, exact interrupted/resume equivalence, and
a zero-step replay. Evaluation features, the sealed outcome packet, predictions,
positions, metrics, PnL, bootstrap, prior checkpoints, and BTC/ETH/SOL remained
untouched. Its terminal receipt now anchors the separately registered V84
outcome-blind evaluation-prepare phase. V84 must use all nine checkpoints over
the exact 2026 non-target feature window, freeze predictions, candidate/control
positions, and behavior gates, and stop before the sealed outcome packet. No
performance, PnL, bootstrap, outcome, or BTC/ETH/SOL access is allowed until a
new exact hash-bound unseal authorization is supplied after prepare passes.
Consult `research/current.yaml` for the exact command and boundary.

## Skill routing

- State, recap, family counting, or next action: `tlm-research-governor`.
- Implement one authorized spec/harness/dataset loop: `tlm-loop-engineer`.
- Operate a frozen MPS training grid: `tlm-training-operator`.
- Prepare or open a registered evaluation: `tlm-one-shot-evaluator`.
- Diagnose a retired family: `tlm-retirement-autopsy`.

Skills orchestrate repository code and contracts. They do not replace tests or
scientific enforcement in source.

## Engineering rules

- Inspect Git state and affected contracts before editing.
- Preserve frozen V1-V55 implementations; add shared infrastructure only for
  future experiments unless a historical verifier requires a compatibility fix.
- Use one writer for overlapping files. Parallel agents may perform read-only
  review or edit disjoint paths.
- Use `rg` for search and `apply_patch` for edits.
- Do not read or upload `.env*`, keys, wallet files, tokens, or credentials.
- Keep generated panels, checkpoints, predictions, and large artifacts out of
  ordinary Git; track content-addressed receipts and follow the exact frozen
  storage policy. A hash-bound owner waiver may remove external redundancy, but
  it never permits deleting required local artifacts or weakening verification.
- Do not add dashboards, RL, hyperparameter sweeps, or unrelated refactors.

## Quality gate

Before completing a loop:

1. Verify the current authorization and input hashes.
2. Run focused tests, then the full test suite.
3. Run the registered smoke/preflight/verify command.
4. Verify deterministic replay and forbidden-access counters.
5. Update `research/current.yaml`, `STATUS.md`, and `TASKS.md` only after success.
6. Report changed files, commands, checks, artifacts, and remaining limitations.

Do not advance to a later loop in the same task unless the current contract
explicitly authorizes it and the user requested that additional scope.
