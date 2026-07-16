---
name: tlm-loop-engineer
description: Implement exactly one currently authorized TLM research phase and stop at its frozen gate. Use when the user says "continue", "bora", "next step", "implement V56", "run the next loop", or asks to build or execute an already authorized specification, synthetic harness, dataset, training, or evaluation phase.
---

# TLM Loop Engineer

Implement the smallest complete slice for one authorized phase. Never continue
automatically into the next lifecycle action.

## Preflight one phase

1. Invoke `$tlm-research-governor` and require `ready: true`.
2. Run the phase preflight, replacing `v56` only with the governor's exact
   authorized phase:

   ```bash
   python3 .agents/skills/tlm-loop-engineer/scripts/phase_guard.py \
     preflight --phase v56 --json
   ```

3. Stop if the requested phase differs, if documentation drift exists, or if
   unrelated working-tree changes cannot be attributed.
4. Read the complete experiment contract referenced by
   `research/current.yaml`, the phase section in `TASKS.md`, related
   source/tests, and only the inputs admitted by the access contract.
5. Write a short plan whose last step is the current gate. Do not add a later
   phase to the plan.

## Implement

1. Use one writer. Delegate only read-only contract, leakage, and
   reproducibility review.
2. Make the smallest source, config, CLI/Make, test, and documentation changes
   needed for the current phase.
3. Reuse repository accounting, hashing, manifest, and checkpoint utilities;
   do not implement scientific metric logic inside this skill.
4. Validate any uncertain input path with `$tlm-research-governor` before
   opening it.
5. Preserve deterministic seeds, frozen shapes, costs, gates, hashes, missing
   rows, and failure evidence. Never broaden a scope because an existing file
   or command makes it convenient.
6. Run focused tests, the phase smoke command, then the relevant broader suite.
7. Run the exact command twice when byte-identical or idempotent replay is part
   of the gate.

## Stop at the gate

1. After repository tests create the phase receipts, run:

   ```bash
   python3 .agents/skills/tlm-loop-engineer/scripts/phase_guard.py \
     gate --phase v56 --json
   ```

2. A guard pass confirms only that repository-produced gate evidence and the
   next-action receipt exist. It does not recompute scientific metrics.
3. Update `TASKS.md` and the concise current authorization only after the gate
   result is known. Preserve failures exactly.
4. Report files, commands, test counts, gate result, access ledger, and
   limitations.
5. Stop. Return to `$tlm-research-governor`; never implement the newly
   authorized phase in the same loop.

See [phase-contract.md](references/phase-contract.md) for the mandatory handoff
and stop conditions.
