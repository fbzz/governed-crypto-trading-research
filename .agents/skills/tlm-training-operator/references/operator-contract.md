# Training operator packet

The validator accepts one JSON object. Paths are relative to `--repo-root` and must remain inside it.

```json
{
  "schema_version": "tlm-training-operator/v1",
  "operation": "doctor|smoke|full|verify|replay",
  "research_state": {
    "path": "research/current.yaml",
    "sha256": "<sha256>",
    "authorized_phase": "v58",
    "authorized_next_action": "<exact action>",
    "authorized_command": "<exact command>"
  },
  "contract": {
    "path": "artifacts/.../training_spec.json",
    "sha256": "<64 lowercase hex>",
    "frozen": true,
    "authorized_operations": ["doctor", "smoke", "full", "verify", "replay"]
  },
  "source_receipt": {
    "git_clean": true,
    "git_head": "<40+ lowercase hex>",
    "files": {"relative/source.py": "<sha256>"},
    "bundle_sha256": "<sha256 of canonical JSON files map>"
  },
  "doctor": {
    "passed": true,
    "python_ok": true,
    "torch_ok": true,
    "mps_available": true,
    "mps_operational": true,
    "device": "mps",
    "dtype": "float32",
    "deterministic_algorithms": true,
    "fallback_enabled": false,
    "disk_free_bytes": 20000000000,
    "required_free_bytes": 10000000000,
    "active_job_count": 0,
    "process_lock_path": "data/checkpoints/.v58_state_conditioned_multi_horizon_training.lock",
    "backup_mode": "external|owner_waiver",
    "backup_required": "true|false according to mode",
    "backup_passed": true,
    "backup_receipt_sha256": "<sha256 or null>",
    "backup_waiver_path": "<registered path or null>",
    "backup_waiver_sha256": "<registered sha256 or null>",
    "backup_waiver_verified": "false|true according to mode",
    "backup_objects_verified": "19 in external mode; 0 in owner_waiver mode",
    "code_backup_verified": "true in external mode; false in owner_waiver mode",
    "full_training_ready": true
  },
  "data_access": {
    "outcome_rows_read": 0,
    "target_assets_loaded": [],
    "forbidden_columns_loaded": [],
    "predictions_written": false,
    "policy_actions_emitted": false,
    "performance_metrics_computed": false,
    "pnl_computed": false,
    "hyperparameters_changed": false
  },
  "grid": {
    "expected_jobs": ["origin|geometry|fold|seed"],
    "completed_jobs": [],
    "active_jobs": [],
    "selected_jobs": []
  },
  "resume": {
    "granularity": "epoch_boundary",
    "cross_job_resume_allowed": false,
    "active_resume_artifacts": [],
    "pending_resume_artifacts": [],
    "pending_resume_job": null,
    "orphan_resume_artifacts": [],
    "interrupted_resume_matched": false
  },
  "verification": {
    "checkpoint_jobs_verified": [],
    "all_checkpoints_retained": false,
    "checkpoint_roundtrip_passed": false
  },
  "replay": {
    "new_jobs": 0,
    "new_optimizer_steps": 0,
    "artifact_hashes_match": false
  },
  "evidence": {
    "data_access": {"path": "artifacts/.../data_access.json", "sha256": "<sha256>"},
    "checkpoint_manifest": {"path": "artifacts/.../checkpoint_manifest.json", "sha256": "<sha256>"},
    "backup_policy": {"path": "artifacts/.../backup_policy_receipt.json", "sha256": "<owner_waiver mode only>"}
  }
}
```

## Phase requirements

- The frozen `training_spec.json` must contain the live phase-contract path/hash, a deep-exact projection of that YAML contract, and the exact source-receipt file list.
- Every phase reruns `research-status` and `research-doctor`, binds the exact live state/action/command, requires `full_training_ready`, and checks the live Git head/status, source hashes, MPS operation, global lock, the exact frozen storage policy, access prohibitions, the exact derived 36-job grid, and at most one active or pending same-job resume.
- Evidence references are path-and-SHA bound. `smoke` binds both its interrupted/resumed result and data-access receipt. `full` binds data access plus the current checkpoint manifest. `verify` and `replay` additionally bind verification and policy-conditional storage evidence: `checkpoint_backup` in external mode or `backup_policy` in owner-waiver mode. Local retention and roundtrip verification of all 36 checkpoints plus the exact `checkpoint_manifest.json` are mandatory in both modes. Replay also binds its replay receipt.
- `smoke` additionally requires an interrupted/resumed execution to match its uninterrupted control.
- `verify` requires every expected job to be completed and checkpoint-verified, all checkpoints retained, roundtrip success, and no resume artifacts.
- `replay` requires all `verify` invariants plus zero new jobs, zero new optimizer steps, and matching artifact hashes.
- `full` may represent a ready, running, or completed grid; use `verify` to claim completion.

Compute `bundle_sha256` as SHA-256 of UTF-8 canonical JSON for the `files` map: sorted keys and separators `(',', ':')`.
