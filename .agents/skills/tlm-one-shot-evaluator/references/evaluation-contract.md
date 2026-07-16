# One-shot evaluation packet

The validator accepts one JSON object. Paths are relative to `--repo-root` and must remain inside it.

```json
{
  "schema_version": "tlm-one-shot-evaluator/v1",
  "phase": "prepare|unseal|complete|replay",
  "research_state": {
    "path": "research/current.yaml",
    "sha256": "<sha256>",
    "authorized_phase": "v59",
    "authorized_next_action": "<exact action>",
    "authorized_command": "<exact command>"
  },
  "evaluation_spec": {
    "path": "artifacts/.../evaluation_spec.json",
    "sha256": "<sha256>",
    "frozen": true
  },
  "source_receipt": {
    "git_clean": true,
    "git_head": "<40+ lowercase hex>",
    "files": {"relative/source.py": "<sha256>"},
    "bundle_sha256": "<sha256 of canonical JSON files map>"
  },
  "registered": {
    "cost_bps": [10, 20, 30],
    "accounting": {"net_return": "gross_minus_turnover_cost"},
    "controls": {"cash": "all_zero"},
    "gates": {"absolute_return_positive": true},
    "outcome_blind_gate_names": ["turnover", "concentration"],
    "sha256": "<sha256 of canonical JSON of cost_bps/accounting/controls/gates/outcome_blind_gate_names>"
  },
  "prepare": {
    "receipt": {"path": "artifacts/.../prepare_receipt.json", "sha256": "<sha256>"},
    "artifacts": [
      {"kind": "predictions", "path": "artifacts/.../predictions.parquet", "sha256": "<sha256>"},
      {"kind": "positions", "path": "artifacts/.../positions.parquet", "sha256": "<sha256>"}
    ],
    "outcome_rows_read": 0,
    "outcome_artifacts_present": false,
    "outcome_blind_gates": {"turnover": true, "concentration": true},
    "predictions_frozen": true,
    "positions_frozen": true,
    "all_checkpoints_used_without_selection": true,
    "authorizes_unseal": true
  },
  "authorization": {
    "explicit_user_authorization": false,
    "exact_registered_unseal": false
  },
  "unseal": null,
  "safety": {
    "target_assets_loaded": [],
    "retuning_performed": false,
    "thresholds_changed": false,
    "costs_or_accounting_changed": false,
    "second_unseal_attempted": false
  },
  "completion": null,
  "replay": null
}
```

## Receipt schemas

Every phase reruns `research-status`, requires its exact live phase/action/command, and verifies the clean Git/source receipt. The keys of `prepare.outcome_blind_gates` must exactly equal the frozen `registered.outcome_blind_gate_names` and every value must be `true`.

The external prepare receipt must contain:

```json
{
  "schema_version": "tlm-one-shot-prepare/v1",
  "evaluation_spec_sha256": "<sha256>",
  "registered_sha256": "<sha256>",
  "artifact_hashes": {"relative/path": "<sha256>"},
  "outcome_rows_read": 0,
  "outcome_blind_gates_passed": true,
  "authorizes_unseal": true
}
```

The unseal authorization receipt additionally contains `explicit_user_authorization: true` and `authorized_command` equal to the command in the bound research state.

For `unseal`, `complete`, and `replay`, replace `unseal: null` with paths and hashes for:

- `authorization_receipt`: schema `tlm-one-shot-unseal-authorization/v1`, `unseal_count: 1`, explicit authorization, exact authorized command, and bindings to evaluation, prepare, and registered hashes;
- `outcome_packet`: immutable artifact path and SHA-256;
- `outcome_receipt`: schema `tlm-one-shot-outcome/v1`, bindings to all prior hashes, `unseal_count: 1`, and `written_atomically: true`, `immutable: true`.

For `complete` and `replay`, `completion` binds the evaluation, prepare, outcome receipt, registered contract, result artifacts, and final decision. Schema is `tlm-one-shot-completion/v1`; decision is `pass` or `retire`.

For `replay`, require `reused_existing_outcome_packet: true`, `new_unseal_receipts: 0`, `source_outcome_rows_read: 0`, and `result_hashes_match: true`.

Canonical hashes use sorted JSON keys and separators `(',', ':')`.
