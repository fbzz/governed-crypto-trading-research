---
name: tlm-one-shot-evaluator
description: Operate a preregistered TLM evaluation with outcome-blind preparation and exactly one outcome unseal. Use when Codex is asked to prepare frozen predictions or positions, run pre-outcome behavioral gates, open a registered historical or prospective outcome window, verify an immutable outcome packet, resume an interrupted evaluation, or replay a completed evaluation without retuning or reading the source outcomes twice.
---

# TLM One-Shot Evaluator

Protect the scientific irreversibility boundary between frozen predictions/positions and outcomes. This skill validates lifecycle receipts; it does not design gates or compute new science.

## Required workflow

1. Run `PYTHONPATH=src python3 -m tlm research-status`. Read the active evaluation contract and training completion receipt, then require the exact phase, action, and command from `research/current.yaml`.
2. Read [references/evaluation-contract.md](references/evaluation-contract.md).
3. Run `python3 .agents/skills/tlm-one-shot-evaluator/scripts/validate_evaluation_packet.py --repo-root <repo> --packet <packet>` at the current lifecycle phase. The validator independently rechecks the live research state. Stop on failure.
4. Complete `prepare` without opening any registered outcome column or packet:
   - use every registered checkpoint without selection;
   - freeze predictions, positions, controls, and their hashes;
   - run every registered outcome-blind behavior gate, including turnover, action coverage, concentration, exposure, episode/churn structure, missingness, prediction distribution, quantile crossing, and ensemble disagreement when registered;
   - bind the exact costs, accounting, controls, and outcome-dependent gates into the prepare receipt.
5. Require an explicit user authorization for this exact registered unseal after the prepare packet passes. A generic `continue` or an earlier permission is insufficient.
6. Before reading outcomes, atomically write one authorization receipt bound to the evaluation spec, prepare receipt, and registered contract hash.
7. Read only the registered outcome columns, keys, dates, assets, and maturity window. Atomically write the outcome packet, then write its receipt bound to the authorization receipt and packet hash.
8. Evaluate only the preregistered metrics, costs, accounting, controls, bootstrap, and gates. Preserve every failed cell.
9. Write one completion receipt bound to the spec, prepare receipt, outcome receipt, result artifacts, and immutable pass/retire decision.
10. On interruption or replay, validate and reuse an existing complete outcome packet. Never overwrite it, recreate its authorization, or reread source outcomes.

## Stop conditions

Stop before unseal when preparation is incomplete, any outcome-blind gate fails, a hash drifts, the source receipt is dirty, an outcome was accessed during prepare, or explicit authorization is absent.

Fail closed after unseal when the authorization exists without a complete atomic packet. Preserve the evidence and report the incomplete lifecycle; do not issue a second unseal.

Never change a feature, checkpoint, seed, prediction, position, threshold, cost, accounting rule, control, gate, bootstrap cell, or evaluation window after outcome access. A failed gate retires the family unless the frozen contract states a different action.

## Handoff

Report phase, spec/prepare/outcome/result hashes, whether outcomes remain sealed, unseal count, source-read count, registered cost cells, passed/failed gates, immutable decision, target-asset status, and the only next action authorized by the completion receipt.
