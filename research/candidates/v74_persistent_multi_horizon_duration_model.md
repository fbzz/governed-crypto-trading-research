# V74 candidate — Persistent multi-horizon duration model

Status: **implementation precursor to the registered V74 specification**. The
formal contract in `configs/v74_persistent_duration_spec.yaml` is authoritative;
this note and model code do not authorize data access, training, checkpoint
reuse, predictions, policy selection, or outcome evaluation.

## Research decision

Build one successor that addresses the V72 failure mechanism directly. Do not
increase capacity as the primary intervention and do not add another daily
abstention threshold.

The V64-R2/V72 post-mortem showed three coupled problems:

1. weak and fold-unstable gross edge;
2. approximately one-day holding episodes caused by a daily probability exit;
3. a small gross edge consumed by repeated round-trip costs.

V59 already tested a multi-horizon quantile model, but its conservative weekly
`h7 q20` policy was effectively inactive. Multi-horizon prediction alone is
therefore not the new hypothesis. The new hypothesis is to learn **return,
relative rank, and the economically viable duration of the state jointly**.

## Architecture

Working family id: `tlm_persistent_multi_horizon_duration_v1`.

```text
[B, 256 days, 3 assets, 9 features]
                 |
       shared causal patch encoder
        16-day patches / 8-day stride
                 |
       cross-asset self-attention
       (no asset-slot embeddings)
                 |
     gated asset + market-state fusion
          /          |           \
 relative return   market       discrete
 location/scale  location/scale  duration hazard
      1/3/7d        1/3/7d       days 1..7
          \          |           /
        cost-conditioned persistent score
```

Frozen implementation defaults for the candidate code:

| Component | Value |
|---|---:|
| Lookback | 256 days |
| Patch / stride | 16 / 8 days |
| Model width | 128 |
| Temporal layers | 4 |
| Cross-asset layers | 1 |
| Attention heads | 8 |
| Feed-forward width | 512 |
| Dropout | 0.15 |
| Parameters | 1,083,155 |
| Return distribution | Student-t, df 5 |
| Return horizons | 1, 3, 7 days |
| Duration support | 1 through 7 days |

This capacity is close to the strongest prior ranker and intentionally below
two million parameters. The change is structural, not a parameter sweep.

## Output contract

- `excess_location[B,3,3]`: centered cross-sectional return component;
- `market_location[B,3]`: triplet-wide return component;
- `gross_location/gross_scale[B,3,3]`: combined Student-t parameters;
- `net_location[B,3,3]`: gross location less an explicit broadcastable
  round-trip cost;
- `hazard_probability[B,3,7]`: conditional probability that the state ends on
  each holding day;
- `survival_probability[B,3,7]`: monotone probability that the state remains
  economically viable;
- `persistent_net_score[B,3,3]`: net forecast weighted by survival at the
  corresponding horizon.

Asset outputs are permutation equivariant and market outputs are permutation
invariant. Temporal patches use a causal attention mask.

## Training labels and objective

The future dataset phase should create, from non-target assets only:

1. cumulative open-to-open returns for 1, 3 and 7 days;
2. duration per asset: earliest argmax day of cumulative gross open-to-open log
   return over days 1 through 7;
3. a right-censor flag when that earliest argmax occurs on day 7;
4. purge/embargo based on the maximum eight-day label maturity already used by
   the earlier multi-horizon family.

The implemented objective is:

```text
Student-t return NLL
+ 0.25 * cross-asset pairwise ranking loss
+ 0.50 * explicit-duration negative log likelihood
```

Turnover regularization belongs in the future training/policy integration,
where previous positions exist. It is deliberately not faked inside this
stateless model forward pass.

## Why these components

- PatchTST motivates patch tokens and shared channel encoders for longer
  histories at manageable attention cost:
  https://arxiv.org/abs/2211.14730
- TFT motivates one representation serving multiple horizons with gating for
  local and long-range interactions:
  https://arxiv.org/abs/1912.09363
- Deep Momentum Networks show that transaction costs/turnover should affect the
  learned objective rather than only the final backtest:
  https://arxiv.org/abs/1904.04912
- Deep Explicit Duration Switching Models motivate an explicit duration state
  instead of assuming memoryless one-day switching:
  https://papers.neurips.cc/paper_files/paper/2021/file/fb4c835feb0a65cc39739320d7a51c02-Paper.pdf

## Minimal next gate

Before any full grid, register one synthetic harness that proves:

1. causal-prefix invariance;
2. asset permutation equivariance and market invariance;
3. centered excess outputs and positive scales;
4. monotone survival curves;
5. finite joint-loss backward pass on CPU and MPS;
6. interrupted checkpoint resume equivalence.

Only after that gate should a non-target duration-label dataset and a small
training grid be authorized. BTC, ETH, SOL and evaluation outcomes remain
sealed during those phases.
