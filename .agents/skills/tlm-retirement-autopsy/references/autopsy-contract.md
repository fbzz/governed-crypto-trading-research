# Autopsy contract

Freeze a JSON contract before reading detailed outcomes. The contract is an authorization boundary, not a list to expand after seeing results.

## Required shape

```json
{
  "schema_version": "1",
  "family_id": "family_name",
  "retirement": {
    "input": "evaluation_result",
    "decision": "retire_family_without_tuning",
    "immutable": true
  },
  "inputs": {
    "evaluation_result": {
      "path": "artifacts/example/result.json",
      "sha256": "64 lowercase hexadecimal characters"
    }
  },
  "diagnostics": {
    "signal": ["registered ranking and predictive diagnostics"],
    "calibration": ["registered bias, coverage, and drift diagnostics"],
    "churn": ["registered transition and episode diagnostics"],
    "cost": ["registered gross, turnover-cost, and net decomposition"]
  },
  "forbidden": {
    "counterfactual_pnl": true,
    "parameter_or_threshold_tuning": true,
    "model_training_or_inference": true,
    "new_bootstrap_or_cost_grid": true,
    "post_hoc_selection": true
  },
  "outputs": ["artifacts/example_autopsy/result.json"]
}
```

Use repository-relative paths only. The `inputs` object is the complete allowlist. Include the artifact carrying the retirement decision in that map and refer to its logical name from `retirement.input`.

If an axis cannot be measured from frozen inputs, register an explicit limitation such as `unavailable_seed_outputs_were_not_persisted`; do not recreate it. Each diagnostic entry must name a previously registered computation or a deterministic decomposition of already-persisted values.

## Required interpretation

| Axis | Question | Invalid inference |
|---|---|---|
| Signal | Was there predictive or ordinal information? | Positive rank correlation implies profitable trading |
| Calibration | Were sign, magnitude, quantiles, or origins calibrated? | Nominal aggregate coverage proves useful selection |
| Churn | Did state transitions convert forecasts into stable holdings? | Lower drawdown proves better risk prediction |
| Cost | Was failure gross, cost-driven, or both? | Beating a losing control proves positive edge |

Report the frozen decision verbatim. Diagnostic findings cannot reopen the family or authorize target evaluation.
