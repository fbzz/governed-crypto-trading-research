---
name: tlm-training-operator
description: Operate a frozen TLM training grid safely on Apple MPS. Use when Codex is asked to preflight, smoke-test, start, monitor, resume, verify, or replay a registered non-target training run; when a long MPS job failed or was interrupted; or when checkpoint-grid completeness and zero-step idempotence must be proven without changing hyperparameters or reading evaluation outcomes.
---

# TLM Training Operator

Run only the training operation authorized by the frozen experiment contract. Treat this as a low-freedom operational workflow, not a model-design task.

## Required workflow

1. Run `PYTHONPATH=src python3 -m tlm research-status` and read the active experiment contract. Require the exact phase, action, and command from `research/current.yaml`; do not infer authorization from `TASKS.md` prose alone.
2. Read [references/operator-contract.md](references/operator-contract.md).
3. Enforce the storage mode in the exact frozen phase contract. In `external` mode, copy every frozen input and the clean Git HEAD to a real external filesystem with `python3 .agents/skills/tlm-training-operator/scripts/create_v58_backup.py inputs --repo-root <repo> --backup-root <external-root>`; never substitute another directory on the source device. In `owner_waiver` mode, require the exact hash-bound waiver and later `backup_policy_receipt.json`; do not create or claim external copies.
4. Run `PYTHONPATH=src python3 -m tlm research-doctor`. Training is blocked unless `full_training_ready` is true.
5. Materialize each operator packet from existing receipts with `python3 .agents/skills/tlm-training-operator/scripts/build_training_packet.py ...`. The builder must self-validate; do not invent passing values.
6. Run `python3 .agents/skills/tlm-training-operator/scripts/validate_training_packet.py --repo-root <repo> --packet <packet>` before every irreversible phase. The validator independently rechecks status and doctor. Stop on any failure.
7. Execute phases in order: `doctor -> smoke -> full -> verify -> replay`.
8. Use the exact command, grid, optimizer, epochs, patience, seeds, folds, origins, geometries, and paths from the frozen contract.
9. Run smoke and full training with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Require deterministic float32 MPS and the registered clean Git/source receipt.
10. Permit at most one active training job. Parallel read-only monitoring is allowed; parallel optimizer processes are not.
11. Resume only the same job, from its registered epoch-boundary artifact, after validating model, optimizer, scaler, RNG, history, patience, step count, contract, and source hashes.
12. Retain every registered checkpoint. Never select or discard a seed, fold, origin, or geometry by loss or outcome.
13. After full training, retain the exact ordered 36 local checkpoints and the byte-identical local `checkpoint_manifest.json` in every storage mode. In `external` mode, copy them to the same external filesystem with the backup utility's `checkpoints` operation and bind source/backup paths, hashes, and sizes. In `owner_waiver` mode, verify/replay bind the exact `backup_policy_receipt.json` and must never imply that an external copy exists.
14. Verify the complete Cartesian grid, checkpoint roundtrips, absence of orphan resume artifacts, and the data-access audit.
15. Run the exact full command once more as an idempotence check. It must create zero jobs and execute zero optimizer steps.

## Stop conditions

Stop without modifying the experiment when any of these occurs:

- the parent does not authorize this exact training stage;
- tracked source is dirty or a source/input hash drifts;
- MPS is unavailable, fallback is enabled, or the doctor fails;
- another optimizer process or process lock is active;
- the frozen storage mode, waiver, or storage-policy receipt drifts;
- a resume belongs to another job or is not at an epoch boundary;
- an outcome, target asset, forbidden label, prediction, policy, performance metric, or PnL is accessed;
- the grid, checkpoint set, or replay is incomplete.

Do not repair a frozen run by editing hyperparameters, grid axes, data windows, objectives, thresholds, or accounting. Return the failed invariant and the last valid lifecycle state.

## Handoff

Report the frozen contract hash, Git/source receipt, storage mode and waiver hash when applicable, device, exact command, completed/expected jobs, optimizer steps, resume status, local checkpoint verification, replay result, tests, and next action authorized by the resulting receipt. Training loss is engineering telemetry, not evidence of alpha.
