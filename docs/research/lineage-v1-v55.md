# Archived research lineage V1-V55

This file preserves the historical research-policy chain formerly loaded as the
root `AGENTS.md`. It is evidence, not the current authorization source. Current
authorization lives in `research/current.yaml`.

# Mission

Build a reproducible daily trading-research MVP for BTC/USDT, ETH/USDT,
and SOL/USDT.

# Required pipeline

data -> features -> dataset -> model -> signal -> backtest -> report

# Hard constraints

- Python only.
- No real order execution, API keys, shorting, or leverage.
- Never use random time-series splits.
- Never fit scalers on validation or test data.
- Features at day `t` may only use information available by the close of `t`.
- A signal produced after the close of `t` may only trade the `t+1` open-to-close return.
- Apply costs to portfolio turnover.
- Every behavior change needs a test.
- Keep generated datasets and experiment outputs out of source control.

# Models

- Mandatory baseline: multi-output Ridge regression.
- Main model: a small Transformer encoder.
- V1 target: `log(close[t+1] / open[t+1])` for each asset.
- V2 target may use `log(open[t+2] / open[t+1])` to evaluate a persistent
  daily position entered after the close-of-`t` signal.

# Strategy

- Hold at most one asset each day.
- Go long only when the largest prediction exceeds the configured threshold.
- Otherwise hold cash.
- Maximum gross exposure is 100%.
- Intraday mode exits at that day's close and charges both legs.
- Persistent mode rebalances at daily opens and charges every risky-asset buy
  and sell, including final liquidation.

# Quality gates

Before concluding a loop:

1. Run the related tests.
2. Run the relevant smoke command.
3. Fix materially important warnings.
4. Update `TASKS.md`.
5. Report changed files, commands, passed tests, and remaining limitations.

# Forbidden in the MVP

- Dashboard or live trading.
- Hyperparameter sweeps.
- On-chain data or unversioned/un-audited derivatives inputs.
- Reinforcement learning.
- Unrelated refactors.

# Current research policy

The `dual_momentum_with_unanimous_tlm_override` candidate is suspended after
failing extended walk-forward validation. For new experiments:

- use 30-day relative momentum as the default asset rank and absolute risk gate;
- use three fixed Transformer seeds;
- override the momentum asset only when every Transformer member agrees;
- remain in cash when every 30-day asset momentum is non-positive;
- never promote a single-seed result;
- use pure 30-day dual momentum as the deterministic research control;
- require expanding and rolling walk-forward plus block-bootstrap evidence
  before reconsidering any learned override.

The next bounded research candidate is `OverrideNet v1`: deterministic
gradient boosting predicts only gross residual edge over dual momentum. Exact
incremental turnover cost is applied by the policy, and a small pre-registered
abstention grid may be calibrated only inside each outer training fold with a
purged chronological calibration window. This nested calibration is permitted;
broad hyperparameter search remains forbidden.

OverrideNet v1 is rejected under the frozen v7 suite unless a later experiment
is explicitly versioned. Do not tune v7 thresholds or tree parameters against
its observed outer results.

The frozen v8 candidate is a risk-off meta-labeler. It may only replace an
active dual-momentum position with cash; it cannot change the ranked asset.
It predicts q10/q50 of the control return, uses a pre-registered q10 threshold
grid with hysteresis, and must retain 90% of control return while improving
Sharpe and max drawdown. Because v6/v7 exposed the same historical outer
windows, even a v8 offline pass is research-only until a clean external or
future holdout exists.

Risk-Off Meta-Labeler v1 is rejected under the frozen v8 suite. Its q10
coverage was close to nominal, but the risk-off subset did not separate losses
from gains and return retention failed materially in expanding windows. Do not
tune v8 thresholds, hysteresis, quantiles, or tree parameters against these
observed results. Before another learned policy, require a versioned signal-
existence study or a genuinely new timestamped data family.

The frozen v9 study is diagnostic only. It tests nine registered OHLCV regime
signals using train-derived quintiles and orientation, then evaluates downside
and tail lift out of sample. It must not create positions or tune a policy.
An OHLCV signal exists only if the same orientation clears monotonicity, lift,
coverage, and circular block-bootstrap gates in all expanding and rolling
scenarios. Otherwise the OHLCV-only research branch closes.

V9 found no robust OHLCV-only signal. Momentum, long trend, and cross-asset
dispersion showed large downside lift only in rare expanding-window extremes;
their registered risk-bucket coverage collapsed to roughly 2-4%, while rolling
coverage was materially higher. Treat this as non-stationary regime evidence,
not permission to relax the frozen coverage gate. No new OHLCV-only policy or
threshold experiment is allowed. The next branch must first add and audit a
new timestamped data family, starting with funding, open interest, and basis.

V10 freezes the first derivatives data contract. Use only Binance public USD-M
futures archives with published SHA-256 verification. Funding events, daily
premium-index basis, and 5-minute metrics must be aggregated by UTC day, retain
their maximum raw source timestamp, and satisfy `source_max_timestamp <
open(t+1)`. Never forward-fill source gaps. The registered common window is
2021-12-01 through 2026-06-30, with base coverage >=98% and rolling-derived
coverage >=95% for every asset. V10 is data infrastructure only; it does not
authorize a policy or model experiment.

V11 tested 18 pre-registered derivatives diagnostics on observed complete
cases using train-only quintiles, the expanding 3/6-fold geometries, a rolling
6-fold/730-day geometry, and 7/21/63-day circular block bootstrap. No signal
passed every scenario. Taker long/short flow was closest and kept the same
orientation, but failed the expanding-3 coverage, tail-lift, and bootstrap
gates. Do not tune v11 bins, thresholds, or gates and do not design a policy
from its near signals.

V12 tested 18 pre-registered intraday path diagnostics from the audited
5-minute metrics archives. It reverified 5,019 checksums and preserved
complete-case/no-forward-fill handling. No signal passed even one complete
scenario. Taker-ratio autocorrelation produced large aggregate downside/tail
lift, but only in 7.5-9.5% risk buckets (below the frozen 12% minimum) and with
unstable fold monotonicity. Treat this as an episodic extreme, not permission
to relax coverage. Do not tune v12 bins, coverage, or path definitions and do
not design a policy from this branch.

The OHLCV, daily derivatives, and intraday derivatives-path branches are now
closed for policy design. Before v13 implementation, require a data-family
feasibility audit covering historical availability, timestamp semantics,
licensing/cost, and reproducibility for a genuinely independent source such as
options volatility/skew, liquidation/order-book data, or a clean future
holdout. No model experiment is allowed during that audit.

V13 selected Deribit DVOL as the sole feasible independent family. The live
audit found complete BTC and ETH daily coverage over 2021-12-01 through
2026-06-30 through the public unauthenticated endpoint. V14 may implement only
a cache-first, hash-audited DVOL data layer. Do not invent a SOL-specific DVOL
series; BTC/ETH DVOL may only enter as named market-level options-volatility
context. A DVOL candle stamped `t` covers `[t, t+1)` and becomes final at
`t+1`; because source availability must be strictly earlier than execution, it
may first affect a portfolio open at `t+2`. Preserve gaps, usage-policy source
metadata, and local hashes. No signal study, model, policy, PnL, or threshold
selection is authorized until the v14 data audit passes.

V14 passed the data audit with complete BTC/ETH coverage, stable cached
observation hashes, 16 registered market-level features, and a strict `t+2`
execution index. V15 may run one policy-free signal-existence study over those
frozen columns. It must reuse the v11/v12 train-oriented quintile, expanding
3/6-fold, rolling 6-fold, purge, coverage, downside/tail lift, and circular
block-bootstrap gates without retuning. Complete observed rows only; no
imputation. The study may authorize at most one separately versioned policy
experiment if the same signal survives every scenario. It may not optimize a
threshold, train a trading model, or evaluate a candidate portfolio itself.

V15 found no robust DVOL signal. Mean one-day DVOL change passed two scenarios
but failed expanding-6 fold monotonicity and downside lift; DVOL level only
looked strong in rare 1.5%-2.1% expanding buckets. Do not tune the eight v15
signals, quintiles, gates, lag, or scenario geometry and do not design a DVOL
policy. The next branch must again begin with a source-feasibility audit. V16
may audit official CFTC Commitment of Traders positioning, but it must resolve
as-of versus publication timing, corrections/vintages, historical BTC/ETH
coverage, and unauthenticated reproducibility before any data or signal layer.

V16 rejected CFTC positioning. Weekly BTC/ETH coverage and schema passed, but
report dates are position as-of dates rather than publication timestamps; CFTC
does not provide a complete historical release-date list, and the 2025 window
contains multi-week delayed releases. No COT dataset, signal, model, or policy
is authorized. V17 may audit the official U.S. Treasury daily par-yield curve.
It must begin after the 2021-12-06 methodology change, verify 2y/10y coverage,
use no credentials, and pre-register a conservative source-date-to-execution
lag plus bounded known-state carry across weekends before dataset construction.

V17 selected the official U.S. Treasury daily par-yield curve. V18 may build
only a cache-first data layer for the frozen 2y/10y family. The source date may
first affect crypto execution at `source date + 3 calendar days`. A daily row
may carry the latest already-eligible public state for at most seven calendar
days, but it must retain source date, eligibility timestamp, and state age.
Use only observations on or after the 2021-12-06 methodology change and hash
every annual response. No return join, signal study, model, policy, or PnL is
authorized until the v18 data audit passes.

V18 passed with stable annual hashes, 1,639 causal daily state rows, ten frozen
features, and a maximum state age of six days. V19 may run one policy-free
signal-existence study over a pre-registered subset of rate level, curve,
change, and change-volatility diagnostics. Reuse the v11/v12/v15 train-derived
quintile, expanding 3/6, rolling 6, purge, coverage, lift, and circular block-
bootstrap gates without tuning. No feature search, portfolio, cost threshold,
model, or PnL is allowed during v19.

V19 found no robust Treasury signal. No registered rate diagnostic passed even
one complete scenario. Do not tune rates features or build a Treasury policy.
V20 must be an evidence/multiple-testing ledger over v1-v19, not another model
or feature experiment. It must enumerate every decision and audit, count all
registered signal-scenario evaluations, mark reused historical windows as
adaptive research, and halt further historical model search unless an already-
registered robust signal exists. Dual momentum remains a research control, not
a promoted trading system.

V20 halted further outcome-driven historical model search after auditing 61
registered signals, 183 signal-scenario evaluations, zero robust survivors,
and zero clean-holdout decisions. V21 certifies dual momentum 30d only as a
deterministic research comparator; its certificate never authorizes execution.

V22 is a dormant, deferred-batch prospective protocol. It must not run daily
candidate inference, simulate orders, or expose interim performance. Its clock
starts only after one immutable candidate is registered. A failed candidate is
retired and cannot be tuned against the same future window.

V23 content-addresses source, tests, configs, contracts, and the v20-v22
decision chain. V24 is a read-only cross-artifact review and may not retrain,
select, or promote anything. V25 completes the engineering framework only if
the full v1-v24 chain and final test suite pass.

The final scientific status is negative: there is no deployable TLM. Do not
resume historical architecture, feature, lookback, threshold, or policy search;
do not start shadow, paper, live, or real-money trading. A future transition
requires one ex-ante candidate, v22 registration before its future window, and
one deferred evaluation after every maturity gate.

V26 is the ex-ante zero-shot candidate-family specification anchored to the
annotated `tlm-v25` release. It freezes one three-seed compact cross-asset
Transformer blueprint, non-target chronological splits, a long/cash policy,
and source-domain PnL/drawdown/bootstrap gates. BTC, ETH, SOL, and registered
target proxies are forbidden from development, calibration, model selection,
and performance reporting.

V26 does not register a candidate because no exact universe, source manifest,
feature schema, checkpoint, or source-domain gate result exists yet. V27 may
only inventory official Binance non-target daily archives, apply the frozen
lexical eligibility rules, verify checksums/coverage, and publish the exact
development universe. V27 must not train a model, compute PnL, or load target
asset observations.

V27 passed the frozen mechanical inventory gate with 48 lexically selected
non-target symbols, 3,167 accepted monthly archives, published-checksum
verification, and 98.60%-100% daily coverage. The malformed AXS February 2026
archive is recorded and treated as missing, never repaired. EURUSDT is retained
because v26 did not exclude fiat bases; resolve that scope mismatch through an
explicit performance-blind amendment before any training. V28 may build only
the causal feature/label dataset from the frozen manifest. It may not train a
model, construct a portfolio, load target assets, or compute return/PnL metrics.

V28 passed the cache-only dataset gate with 96,336 calendar-panel rows, 96,297
observed raw rows, 39 preserved missing rows, and a byte-identical replay hash.
It materializes eight per-asset causal features plus the two registered forward
labels. `within_triplet_relative_strength` remains a deterministic same-date
triplet transform and is not precomputed cross-sectionally. No imputation,
scaling, model fitting, target-asset loading, portfolio, or performance metric
is allowed in v28. Training remains blocked. V29 may only resolve EURUSDT via
a versioned, performance-blind universe amendment and refresh the affected
source manifest/dataset before any learned experiment.

V29 supersedes the lexical 48-asset training rule before any model or
performance observation. Future training remains multiasset, but inference and
trading are restricted to BTCUSDT, ETHUSDT, and SOLUSDT. V30 must select exactly
30 non-target cryptoassets using only 2021-2023 coverage and median daily USDT
quote volume, with lexical tie-breaking and three equal 10-asset holdout folds.
Exclude target/proxy, fiat, stablecoin, fan-token, and leveraged-token bases.
Do not use 2024-2026 availability, labels, returns, or PnL for universe
selection. V30 is inventory-only; training remains forbidden.

V30 selected the exact 30-asset training universe from 145 eligible assets
using only symbol, UTC date, and daily USDT quote volume over 2021-01-01
through 2023-12-31. It verified 5,616 candidate archives and froze 1,080
selected-window archive records plus three lexical round-robin holdout folds
of ten assets. Every selected asset has 100% daily coverage and nonzero quote
volume in the selection window. No 2024-2026 availability, target asset,
label, return, model, performance metric, or PnL affected selection.

V31 may only freeze the full 2021-01-01 through 2026-06-30 source manifest for
the exact v30 symbols, reverify every published checksum, audit daily gaps and
timestamp semantics, and preserve rejected archives as missing. It may not
change the universe, build features or labels, train a model, construct a
portfolio, load BTC/ETH/SOL, or compute returns/PnL.
Full-window coverage after 2023 is diagnostic only: delistings, migrations,
and missing archives must remain visible and may not trigger replacement or
reselection using future availability.

V32 passed the fixed-universe causal dataset gate with 60,210 calendar-panel
rows, 58,591 observed raw rows, 1,619 preserved missing rows, and 49,919
indexed 256-day sequences. It freezes eight per-asset features, one
within-triplet relative-strength feature, two forward labels, the v30 folds,
and all lexical triplet combinations within each fold role. The on-demand
loader contract is `[256, 3, 9]` float32 input and `[3, 2]` float32 labels.
Panel and sequence-index Parquet files replay byte-identically from cache.

V33 may only implement and unit/smoke-test the frozen Patch Transformer,
masked-patch reconstruction interface, four probabilistic heads, and
checkpoint metadata contract. It may not fit a scaler on the real dataset,
run pretraining or supervised training, inspect performance, construct a
portfolio, or load BTC/ETH/SOL. Full training remains blocked through v34.

V33 passed on synthetic data with a 380,276-parameter shared temporal Patch
Transformer, 31 causal patches, one permutation-equivariant cross-asset layer,
four per-asset heads, and masked-patch reconstruction. It verified causal
prefix invariance, asset-permutation equivariance, finite gradients, exact
checkpoint roundtrip, and a byte-identical smoke checkpoint. No real panel,
sequence, label, scaler, optimizer step, training epoch, target, or PnL was
loaded or executed.

V34 may only freeze and smoke-test the scientific harness: train-only feature
scaling, deterministic eligible-triplet sampling, deterministic patch masks,
pretraining/supervised losses, optimizer/early-stopping state, source-domain
controls, turnover costs, and paired block bootstrap. It must use synthetic or
bounded fixture data and may not start the full v35 training run or inspect
BTC/ETH/SOL.

V34 passed its synthetic end-to-end smoke with train-only scaling, uniform
sampling over eligible date-triplet pairs, exactly 14 masked patches per
sample, Smooth L1 reconstruction, mean q10/q50/q90 pinball plus 0.1-weighted
log-volatility loss, AdamW, gradient clipping, five-epoch early stopping,
long/cash accounting with final liquidation, 10/20/30 bps costs, both frozen
controls, and 10,000-path paired bootstrap at 7/21/63-day blocks. The harness
is now frozen and full non-target pretraining is authorized.

V35 may fit one scaler per asset fold using only that fold's 2021-2023
representation-train assets, then run masked-patch pretraining for every
registered seed `[42, 7, 123]`. This produces three checkpoints per fold and
nine total; no seed or fold checkpoint may be selected or discarded. V35 may
use 2024 feature-only validation for early stopping, but may not read forward
labels, start supervised training, compute portfolio performance, or load
BTC/ETH/SOL.

V35 passed with nine hash-audited non-target checkpoints, one train-only scaler
per asset fold, 450 total epochs, and 28,800 optimizer steps. Every fold-seed
job reached the frozen 50-epoch ceiling; the best feature-only validation state
was restored independently for each checkpoint. No seed or fold was selected.
Forward labels and BTC/ETH/SOL were not loaded. Held-out asset-fold rows never
entered that fold's scaler, sampler, tensors, objective, or validation; no
portfolio or performance metric was computed.

V36 may initialize all nine fold-seed jobs from their exact v35 checkpoints
and run the frozen supervised objective. Each job may use labels only from its
20 training assets in the 2021-03-01 through 2023-12-31 supervised-train
window, with 2024 labels from those same training assets for early stopping.
It must reuse the frozen fold scaler, sampler, optimizer, batch size, maximum
30 epochs, and five-epoch patience. It must retain all nine checkpoints with no
seed/fold selection and may not use held-out fold assets, BTC/ETH/SOL,
portfolios, PnL, or source-domain metrics.

Before any v36 supervised label was read, the previously unspecified 2025
calibration step was frozen. Every train, validation, and calibration window
must purge its final eight signal dates so the seven-day volatility label's
`t+8` maturity cannot cross a chronological boundary. Training samples 8,192
eligible date-triplet pairs per epoch through 2023-12-23; fixed validation uses
2,048 samples through 2024-12-23. After all three seed checkpoints in a fold
are frozen, their heads are averaged arithmetically on 8,192 deterministic
2025 train-asset samples through 2025-12-23. Calibration adds the empirical
10th/50th/90th percentiles of `observed_return - predicted_quantile` to q10,
q50, and q90, adds the median `observed_log_vol - predicted_log_vol` residual,
then applies a stable ascending projection to the three return quantiles. The
calibration seed is 20260713. It cannot update weights, choose checkpoints,
change policy thresholds, or inspect held-out/target assets. These fold-level
offsets and all nine weights form the final frozen v36 source-domain ensemble.

V36 passed on Apple MPS with all nine fold-seed checkpoints and three frozen
fold calibrations. It completed 114 epochs and 7,296 optimizer steps; every
job restored its own best 2024 validation state without selecting a seed or
fold. The 2025 calibration used exactly 8,192 deterministic samples per fold,
reached approximately 10%/50%/90% empirical quantile coverage, and eliminated
quantile crossings through the registered stable projection. Checkpoint file,
model-tensor, calibration-state, and calibration-semantic hashes all pass and
the consolidated artifacts replay byte-identically. No held-out fold asset,
BTC/ETH/SOL observation, portfolio, return metric, PnL, Sharpe, or drawdown was
inspected.

V37 may run exactly one 2026 asset-disjoint source-domain evaluation. For each
fold, use only that fold's ten held-out assets, its frozen three-seed mean
ensemble, frozen scaler, and frozen calibration state. V37 may calculate only
the already-registered source-domain predictive and policy metrics/gates. It
may not retrain, recalibrate, select a seed/fold/checkpoint, change a threshold,
or load BTC/ETH/SOL. Preserve failures; do not tune after observing the result.

Before any 2026 held-out label value or model prediction was loaded, the v37
inference and gate interpretation were frozen. Evaluate the 173 signal dates
from 2026-01-01 through 2026-06-22; the final eight dates are purged so every
registered volatility label matures by 2026-06-30. Eligibility is exactly the
v32 `supervised_sequence_ready` rule. On each date and fold, enumerate every
eligible lexical test triplet from the frozen catalog. Average each head across
all three seeds inside a triplet, then average equally across every eligible
triplet context containing the asset. Apply that fold's v36 residual offsets
to the context-averaged heads and stably sort q10/q50/q90 per asset. Rank every
currently eligible held-out asset in the fold by calibrated q50; hold the top
asset only when q50 > 0.002 and q10 > -0.03, otherwise cash.

The dual-momentum control uses the causal 30-day sum of close-to-close log
returns, ranks currently eligible assets, and holds the strongest only when it
is positive. The equal-weight control holds the currently eligible fold assets
without ranking; source-availability changes are the only rebalance trigger.
All policies execute the registered next-open-to-next-open return, include
entry/switch/exit turnover and final liquidation, and are evaluated separately
at 10, 20, and 30 bps per unit turnover. Each fold receives one-third of source
capital; aggregate daily returns are the equal-weight mean of the three fold
portfolios. Report fold metrics, but do not select or weight folds by outcome.

The aggregate candidate must beat both controls in total compounded return at
all three costs, beat dual momentum in annualized daily Sharpe at all three
costs, remain within 0.05 absolute max-drawdown of dual momentum at all three
costs, and keep absolute max drawdown at or below 0.35. At the base 10 bps,
run the frozen 10,000-path paired circular block bootstrap at 7, 21, and 63
days. For both controls in every block cell, the fifth percentile of paired
`candidate total return - control total return` must exceed zero. Predictive
loss and coverage metrics are diagnostics only and cannot alter the gate.

V37 consumed the source-domain holdout exactly once under evaluation-spec hash
`55ba101498b5d665d9fff7c742e14a0a7abe66aab349962ab1e69798c70f8955`.
The engineering audit passed, but the candidate gate failed. At 10 bps the
equal-fold aggregate candidate returned -16.44%, with Sharpe -2.943 and max
drawdown -16.54%; dual momentum returned -37.82%, Sharpe -1.488, and max
drawdown -51.61%; equal weight returned -38.18%, Sharpe -1.384, and max
drawdown -47.19%. The candidate beat both controls in point total return and
passed both drawdown checks at 10/20/30 bps, but its Sharpe was below dual
momentum at every cost. All six paired bootstrap fifth-percentile return deltas
were negative across both controls and 7/21/63-day blocks. Every candidate fold
also had negative total return.

The registered failure action is final: retire
`tlm_multi_asset_target_transfer_v2` without target evaluation. Do not run v38
robustness consolidation, v39 candidate registration, v40 target preparation,
or any BTC/ETH/SOL inference for this family. Do not tune thresholds, context
aggregation, architecture, costs, folds, seeds, or calibration using the v37
result. A future family requires a new ex-ante specification and genuinely new
unseen confirmation data; the consumed 2026 window is permanently development
evidence, never a clean holdout again.

The v37 failure autopsy is retrospective diagnosis only, not v38 and not a
candidate experiment. It may read only the frozen v37 result, metrics, gate,
receipt, prediction Parquet, and daily-return Parquet. It may not open the raw
panel, checkpoints, scalers, model code for inference, BTC/ETH/SOL, or any new
market observation. It may not rerun predictions, train, recalibrate, alter a
threshold, simulate an alternative policy, select assets/folds/dates, or claim
promotion evidence. The v37 retirement decision is immutable.

Before running the autopsy, freeze all slices and definitions. Use 10 bps as
the descriptive accounting view because it was the registered base cost.
Define an active day from the frozen candidate weight, and define a holding
episode as consecutive calendar dates in the same active asset; cash or an
asset change ends the episode. Report fold, asset, calendar-month, and episode
PnL; 1/3/5-day loss concentration; candidate-active versus candidate-cash
control returns; and the causal signal-time regime `cross-sectional median
momentum_30 > 0` versus `<= 0`. Ranking diagnostics must cover every fold-date:
Spearman correlation, predicted-top1 actual rank, top1/top3 hit rate, predicted
top1 excess return versus the same-date fold mean, and active-day versions of
those metrics. Report direction accuracy, quantile coverage, q50 bias, and
continuous relationships of q50, q10, top1 margin, and predicted volatility
with realized returns. Fixed confidence bins are descriptive only: q50
`[0.002,0.005)`, `[0.005,0.01)`, and `>=0.01`; q10 `[-0.03,-0.02)`,
`[-0.02,-0.01)`, and `>=-0.01`; top1 margin `[0,0.001)`,
`[0.001,0.003)`, and `>=0.003`. Do not hide sparse or empty bins.

The v37 artifacts contain context counts but not per-seed or per-triplet
predictions. The autopsy must explicitly mark seed/context disagreement as
unrecoverable without forbidden re-inference. Its output may recommend only a
new ex-ante family or termination; it cannot recommend salvaging v37.

The frozen v37 failure autopsy passed all input-hash, scope, accounting, and
target-absence checks. At 10 bps the candidate was already negative before
costs: -14.11% gross and -16.44% net, with 2.73% of aggregate turnover charges.
It was active on only 41 of 519 fold-days; every holding episode lasted one day
and only 12 of 41 episodes were profitable after round-trip cost. Q50 top-1 hit
rate was 10.02% versus an 11.52% random expectation, mean daily q50 rank IC was
0.0179, and selected assets lost 1.086% on average on active dates. The three
worst aggregate calendar days produced 44.79% of losing-day magnitude, but 29
of 41 episodes lost, so no single outlier explains the failure.

Q10 rank IC of 0.1100, marginal tail coverage, and volatility diagnostics are
retrospective evidence of limited risk structure only. They do not authorize a
q10 policy, new confidence rule, or v37 threshold test. Per-seed and per-context
disagreement remain unavailable without prohibited re-inference. The family
stays retired, BTC/ETH/SOL stay sealed, and the only permitted next research
action is to pre-register a genuinely new family centered on cross-sectional
ranking or excess-return learning. Any future clean confirmation requires new
unseen data; the 2026 v37 window is development evidence forever.

V41 starts a new family, `tlm_cross_sectional_rank_excess_medium_v1`; it is
not v38-v40 and cannot revive or modify the retired v37 family. V41 may only
freeze and audit the ex-ante blueprint. It may read the hash-locked v32
metadata and the v37 retirement/autopsy records, but it may not open the v32
panel or labels, instantiate a model, fit a scaler, execute an optimizer,
compute a prediction or performance metric, or load BTC/ETH/SOL. The 2026
v37 interval is forbidden for training, early stopping, model selection,
policy calibration, or evaluation of the new family.

The V41 candidate has exactly one architecture: 256-day causal input, 16-day
patches with stride 8, width 128, four temporal layers, two cross-asset layers,
four attention heads, feed-forward width 512, dropout 0.20, and no asset-slot
embedding. Its only supervised heads are normalized excess return and log
seven-day volatility. There is no compact/medium tournament and no
hyperparameter sweep. All three seeds `[42, 7, 123]` in all three asset folds
must be retained and averaged without selection.

For every eligible training triplet/date, define the one-day next-open log
return `r_i`, triplet excess `e_i = r_i - mean(r)`, and the train-only fold
scale `sigma = sqrt(mean(e_i^2))`, floored at `1e-6`. Normalize with
`z_i = e_i / sigma` and center model scores across the triplet. The frozen
supervised objective is pairwise logistic RankNet loss plus Smooth L1 excess
loss with beta 1, plus 0.1 times Smooth L1 log-volatility loss. Early stopping
monitors only rank plus excess loss. Exact return ties within `1e-12` are
excluded from pairwise loss; no outcome weighting or clipping is allowed.

The V41 policy ranks the context/seed-averaged raw excess estimates. It stays
in cash only when every currently eligible asset has causal 30-day momentum
at or below zero. A held position changes only when the challenger's predicted
raw excess exceeds the incumbent's by more than 0.002, exactly two legs at the
registered 10 bps base cost. This margin is fixed for all 10/20/30 bps
reporting cells and may not be calibrated. Entry, switch, exit, and final
liquidation turnover are charged. The controls are shared-asset Ridge with
alpha 10, dual momentum 30, and momentum-gated equal weight.

Training uses only each fold's 20 non-target train assets through 2023-12-23;
2024 train-asset labels are used only for early stopping. A single 2025
asset-disjoint screen uses each fold's ten held-out assets and is explicitly a
development gate, not prospective confirmation. It enumerates every eligible
triplet and permits no tuning after observation. Any failed registered rank,
excess, PnL, cost, drawdown, turnover, or block-bootstrap cell retires the
family. Passing only freezes the family for a future non-target confirmation
of at least 180 mature signal dates beginning no earlier than 2026-07-14;
BTC/ETH/SOL remain sealed until that future source-domain confirmation passes.

After a passing V41 specification audit, V42 may implement and smoke-test only
the Medium model, train-only normalization, exact losses, turnover-aware
policy, Ridge/control accounting, and checkpoint contract on synthetic data.
V42 may not open the real panel, reuse a v35/v36 checkpoint, pretrain, run
supervised training, inspect 2025 outcomes, or compute real PnL.

V41 passed 28 specification checks under blueprint SHA-256
`dc28004a9419424f6d9e437b9ac8a7bf42f73ec9ceb1892494e280d9240fdf5e`.
The exact v32 30-asset universe, nine-feature tensor, three 20/10 asset folds,
and train/test triplet counts were preserved. The Medium parameter count was
derived analytically as 1,231,634 without model instantiation. No panel, label,
checkpoint, target asset, prediction, performance metric, or PnL was loaded.
The only authorized next action is the synthetic v42 harness described above.

V42 passed all 33 synthetic harness checks under harness SHA-256
`36551aaa94b516dac08dd27ed08216ab96c7deda79926423086606cd5f9ba83d`.
The 1,231,634-parameter Medium model passed exact-head, causal-prefix,
asset-permutation, masked-reconstruction, ranking/excess-loss, finite-gradient,
shared-Ridge, turnover/cash-gate, cost-accounting, and checkpoint-roundtrip
checks. Two independent executions produced byte-identical JSON and checkpoint
artifacts. No real panel, label, target asset, pretrained checkpoint, prediction,
performance metric, or PnL was read or produced; improvement remains unknown.

The only authorized next action is V43 masked-patch pretraining for the frozen
Medium architecture on each fold's non-target representation-train assets and
dates. V43 must start from the registered seeds rather than the synthetic V42
checkpoint. It may not read forward-return or volatility labels, train or score
the supervised heads, open held-out assets, load BTC/ETH/SOL, compute a real
prediction, inspect 2024/2025 outcomes, or evaluate performance or PnL.

V43 has three execution modes: metadata-only preflight, one-job smoke, and the
full nine-job run. The full contract is exactly folds `[1,2,3]` by seeds
`[42,7,123]`, 8,192 sampled train triplets per epoch, 2,048 fixed feature-only
validation triplets, batch size 128, at most 50 epochs, and patience 5. AdamW,
learning rate `3e-4`, weight decay `1e-4`, gradient clipping 1.0, mask fraction
0.15, Smooth L1 masked-patch loss, float32, deterministic algorithms, and MPS
are frozen. There is no CPU fallback for smoke or full execution.

Unlike V35, V43 may never materialize the global 30-asset panel. For each fold,
it must project only `date`, `symbol`, and the eight registered base features;
push down filters for exactly that fold's 20 train assets and dates no later
than 2024-12-23; and prove that loaded symbols equal the train set and are
disjoint from held-out symbols and BTC/ETH/SOL. The scaler fits only 2021-2023
train-asset rows. Feature-only validation is 2024-01-01 through 2024-12-23.
Target columns, 2025/2026 rows, held-out assets, and target assets are forbidden.
The current Parquet has one row group, so pushed-down filters guarantee semantic
materialization isolation but not physical byte-level pruning. V43 must record
the requested columns and filters and must not claim stronger isolation.

Every V43 model starts fresh from its registered seed. The synthetic V42 and
all V35/V36 checkpoints are forbidden parents. The run records initialization
state hashes, keeps every fold/seed checkpoint without selection, and permits
resume only from the matching V43 job format and metadata. Cross-asset layers
and supervised heads must receive no forward call, gradient, or update during
pretraining. Only a passing full nine-job audit may authorize V44 supervised
non-target training.

V43 passed all 25 full-run checks under pretraining-spec SHA-256
`752a1f351232160256ad0416fa51ad0fa5eb9c5b00d0d6e64f92c5ece6a0f774`.
The metadata-only preflight opened zero Parquets and executed zero optimizer
steps. The MPS smoke passed one fold/seed, and the full run retained all nine
fresh fold/seed checkpoints after 394 epochs and 25,216 optimizer steps. Every
checkpoint reopened semantically; all three fold-local data/scaler audits
passed; the cross-asset layers and supervised heads remained unchanged; and no
label, held-out asset, BTC/ETH/SOL row, prediction, seed/fold selection,
performance metric, or PnL was loaded or computed. An idempotent rerun reused
all checkpoints and reproduced result SHA-256
`a30c7aa5cbf692ff552a7c839508b50bf6ecc35899f4ce17a3b682906a1e7958`.

The only authorized next action is V44 supervised ranking/excess training on
the frozen non-target train folds. V43 establishes no ranking quality or
economic value, and target assets remain sealed.

V44 is the only authorized next action. It initializes each fold/seed job from
the exact matching retained V43 checkpoint and reuses that fold's immutable V43
feature scaler. No V42 synthetic, V35/V36, mismatched fold/seed, or fresh parent
is permitted. The nine-job grid remains folds `[1,2,3]` by seeds `[42,7,123]`;
training samples 8,192 triplets per epoch, fixed validation samples 2,048,
batch size 128, at most 30 epochs, patience 5, AdamW at learning rate `3e-4`
and weight decay `1e-4`, gradient clipping 1.0, float32 deterministic MPS, and
no CPU fallback are frozen.

For each fold, V44 must separately project and filter the feature panel,
supervised-train labels, 2024 validation labels, supervised-train sequence keys,
and validation sequence keys. Every materialized row must belong to exactly the
fold's 20 train assets. Train signal dates end 2023-12-23 with target maturity no
later than 2023-12-31; validation signal dates end 2024-12-23 with maturity no
later than 2024-12-31. Held-out features/labels, 2025 signal dates or outcomes,
BTC/ETH/SOL, raw market fields, and any unregistered target are forbidden. The
single-row-group Parquets provide semantic materialization isolation only, not
physical byte-level pruning, and V44 must preserve that claim.

The fold target scale is the RMS of centered triplet excess returns over the
full lexical enumeration of every eligible supervised-train date/triplet, with
floor `1e-6`; validation labels never fit it. The exact objective is unit-weight
pairwise ranking plus unit-weight centered normalized excess Smooth L1 plus
`0.1`-weight log-volatility Smooth L1. Early stopping monitors ranking plus
excess only. All parameters train except `mask_token` and
`reconstruction_head`, whose states must remain byte-semantically identical to
the parent. Every current/best model, optimizer, CPU/MPS RNG, patience, history,
and step count must be resume-audited.

V44 is training only. It may record objective losses but may not compute held-out
predictions, rank metrics, policy decisions, PnL, costs, bootstraps, seed/fold
selection, or comparisons. Only a passing full nine-job audit may authorize the
single frozen V45 2025 asset-disjoint development screen. Target assets remain
sealed.

V45 is the single asset-disjoint 2025 development screen authorized by V44.
It has three phases: metadata/checkpoint-only preflight, prediction preparation
without held-out outcomes, and one-shot evaluation. There is no real-outcome
smoke. The prepare phase fits exactly one shared-asset Ridge per fold using
8,192 deterministic train-only triplets sampled with seed 20260713 at epoch 0,
alpha 10, the immutable V43 scaler, and V44 fold excess scale. It then freezes
all three-seed Transformer and Ridge predictions over every eligible lexical
held-out triplet from 2025-01-01 through 2025-12-23 before any held-out label is
read.
Preflight may hash the raw panel and sequence Parquet bytes and may semantically
reopen registered checkpoints, but it must call no Parquet deserializer and may
inspect no table value.

Predictive metrics are computed per triplet context. Each seed score is
triplet-centered and multiplied by its fold train-only scale before the three
seeds are averaged. Spearman uses average ranks; pairwise accuracy excludes
only actual-return ties within 1e-12; top-1 uses lexical tie-breaking and excess
against the same triplet mean. Context metrics are averaged within fold-date,
then equally across 357 dates and folds. The top-1 bootstrap resamples the
equal-fold daily context-mean excess, never raw context rows.
Pairwise accuracy pools correct and active pairs within each fold-date before
the same equal-date/equal-fold aggregation.

Economic policy uses context-averaged asset scores. It holds cash only if every
eligible 30-day momentum is nonpositive and switches only when the challenger
strictly exceeds the incumbent by 0.002. Controls are eligible dual momentum 30
and momentum-gated equal weight. Entry, switch, exit, and final liquidation are
charged at 10/20/30 bps; aggregate capital is one third per fold. All registered
predictive, Ridge, return, Sharpe, drawdown, turnover, and 7/21/63-day circular
block bootstrap cells must pass. Any failure retires the family without tuning
or target evaluation; a pass only freezes it for later prospective non-target
confirmation. BTC/ETH/SOL and all 2026 outcomes remain sealed.
Momentum 30 is the rolling sum of the 30 close-to-close log returns ending on
the signal date, with 30 observations required.
Entry from cash has no hurdle. An ineligible incumbent exits immediately and is
replaced by the lexical top-score eligible asset when the absolute gate remains
open. Dual momentum holds only the lexical highest positive momentum; the equal
weight control holds all eligible assets whenever any eligible momentum is
positive. Prepare must hash-freeze predictions and all three policies before
the unseal receipt. Evaluation may consume only the hash-verified prepare
packet and must bind it to the atomic outcome packet and final result receipt.

V45 completed under evaluation-spec SHA-256
`f7c6cd57555feb9496e77dc74321320dfaf6290fd5480fe3405f0fa8859a6888`.
The preflight deserialized zero Parquets; prepare froze 96,864 contexts, 9,782
eligible asset-dates, and 10,710 position rows before the one-shot read of
exactly 9,782 registered 2025 outcomes. All predictive gates passed, including
aggregate Spearman `0.0546`, pairwise accuracy `52.48%`, top-1 excess `0.0986%`,
and both registered Transformer-over-Ridge comparisons. The candidate returned
`15.34%` at 10 bps, but failed the positive-return gate in fold 3 (`-10.55%`)
and the aggregate 35% drawdown cap at 20/30 bps (`-36.20%`/`-37.67%`). V45
therefore passed 36/39 cells and the immutable decision is
`retire_family_without_target_evaluation_or_parameter_tuning`. Do not tune this
family on the consumed 2025 window and do not evaluate it on BTC/ETH/SOL.
Target assets and all 2026 outcomes remain sealed.

V46 is a frozen failure autopsy over the retired V45 family. Its only legal
inputs are the 20 hash-exact V45 files registered in
`configs/v46_ranking_excess_failure_autopsy.yaml`. The metadata preflight must
deserialize zero Parquets; the run must validate that preflight before its first
table read and must rehash every input after analysis.

V46 may not open a raw panel, sequence index, checkpoint, scaler, coefficient,
or any unregistered file. It may not instantiate a model, infer, train,
recalibrate, finetune, bootstrap, change an asset/fold/date, test an alternative
threshold, hurdle, filter, policy, or counterfactual PnL, or inspect BTC/ETH/SOL
or post-2025 observations. The V45 decision is immutable.

The primary descriptive cost is 10 bps, while every registered 10/20/30 bps
cell must remain visible and reconciled. Fold, asset, month, episode, position
state, regime, context-stability, concentration, cost, and drawdown groups must
be preserved, including empty registered groups and assets never held. Regimes
may use only frozen signal-time momentum and are descriptive associations, not
causal claims or candidate filters.

Per-seed disagreement and the predicted-volatility head are unavailable because
V45 did not persist them at the required granularity; V46 must state that fact
instead of reconstructing either value. The only recommendation enums are
`new_ex_ante_family` and `terminate_research_line`, both requiring genuinely
new non-target evidence before evaluation. V46 can never revive V45.

V46 completed under autopsy-spec SHA-256
`d96a9cf45495d991f56596e665ae0d6d61d1be595830f552327dd4c7781695ea`
with zero preflight Parquet deserializations and a passing full ledger/accounting
audit. Its diagnostic result SHA-256 is
`9948487a8a62a3b862e995602b71d57796581961a2b3cf366c37e719082d47ce`.
The frozen evidence shows a relative-ranking versus absolute-long-return gap in
fold 3, concentrated held-asset exposure, and signal-time regime associations
whose sign changes across folds. These observations do not authorize a filter
or V45 retune. V45 remains retired and target assets remain sealed.

V44 passed all 35 full-run checks under supervised-spec SHA-256
`a0feb135a76bd7c4f8fa0162acf9d7ecaf821863f7a329bc2e0f7bc7b98e7e26`.
The nine exact-parent jobs completed 55 epochs and 3,520 optimizer steps; each
fold target scale enumerated all 914,280 eligible train triplets. All nine
checkpoints reopened semantically, no resume artifact remained, and an
idempotent rerun reproduced result SHA-256
`4f905d00224b5a511ff811d930b7add5822bf1e26cb870f7609b09dba653f5f8`.
No held-out asset, target asset, 2025 row, prediction metric, performance
statistic, or PnL was loaded or computed. The only authorized next action is
the frozen V45 2025 asset-disjoint development screen; BTC/ETH/SOL remain
sealed.

# Current authorization after V55

V49 passed under contract SHA-256
`2aa4984082bc402fb8b259ab17c821fca6853c37ade6c5e83187157781a0a49a`
and result SHA-256
`55aa56e309482da96c7985fb47367df5073576c1e05d05c2dd3509f33a2ca256`.
The clean training-source receipt is Git commit
`78c74938db4443aa0c3b437671d8433014c7301d`. All 36 fresh-weight jobs trained
serially on MPS, completed 235 epochs and 15,040 optimizer steps, restored their
job-local best state, and reopened semantically. The verifier found zero resume
artifacts. A second full invocation performed zero new jobs and zero optimizer
steps while preserving the registered result.

V50 evaluated that complete checkpoint tree under spec SHA-256
`df31cca33b1b248324f7e6e9bf4ccabd3c9dea81454c4a4b95e1f6bae836be7e`.
Predictions and positions were frozen before the registered 2024/2025
non-target outcomes were atomically opened. The audit passed and BTC/ETH/SOL
remained sealed, but only 67/180 mandatory cells passed. The candidate failed
9/12 fold return gates, every return comparison with dual momentum and equal
weight, every turnover gate, and all 48 bootstrap gates. Result SHA-256 is
`3ba18a664e909ebebb86b90438323ff6098d339614c134dc8ecc0b644d3a2afa`.

The binding decision is `retire_family_without_tuning`. V51, V52, and V53 are
not authorized, and there is still no deployable, shadow, paper, live, or
real-money strategy. Do not change this family's architecture, objective,
seeds, folds, checkpoints, policy, costs, controls, scenarios, or gates in
response to V50. The consumed 2024/2025 windows are diagnostic only.

Any continuation must begin with a genuinely new ex-ante family identity and
frozen hypothesis before using real outcomes. It may cite the V50 failure mode
as motivation, but it may not present a retune as clean evidence. No historical
result can start a prospective clock, and BTC/ETH/SOL remain sealed.

V54 completed that permitted diagnosis under autopsy-spec SHA-256
`4ebeaba6de4794263a9d0cbcca14a0d8547350edde80140d9328052128f8f6a4`.
It found absolute-return Pearson correlation between `-0.0472` and `0.0225`,
sign accuracy between `47.08%` and `50.44%`, and candidate turnover between
`3.27x` and `4.46x` dual momentum. This confirms a ranking-without-calibrated-
direction failure plus structural churn; it does not authorize V50 tuning.

V55 created the new family
`tlm_state_conditioned_multi_horizon_quantile_small_v1`. Blueprint SHA-256 is
`0c91c65ed422d081ba1ce59544c3911cbd4624a0f4c184cc24d6c02dfc41d435`.
The frozen design has 465,513 analytic parameters, predicts 1/3/7-day q20/q50/
q80 returns, uses h7 q20 with exact current-state transition costs, and may
change action only every seventh eligible signal date. It registers 36 future
jobs without selection and forbids every V49 checkpoint.

The only authorized next action is V56 synthetic state/policy harness. V56 may
instantiate the exact V55 model only on deterministic synthetic tensors. It
must not deserialize the real panel or labels, train a real checkpoint, create
a market prediction, compute performance/PnL, or access BTC/ETH/SOL. V57 and
all later real-data work remain unauthorized until V56 passes exactly.
