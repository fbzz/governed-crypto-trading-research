# Autonomous task: complete V47 through trained V49

Read `AGENTS.md`, `TASKS.md`, and `AUTONOMOUS_TRAINING_LOOP.md` completely
before taking any action.

Your terminal objective is to complete V47, V48, and V49 sequentially and stop
with all 36 V49 non-target checkpoints trained, hash-verified, and audit-passing.
Do not enter V50 and do not persist held-out/deployment predictions or compute
predictive/economic evaluation metrics. Ephemeral train/validation forward
outputs and the frozen objective losses are permitted.

## Operating rules

1. Inspect the current Git state and preserve unrelated/user changes.
2. Execute only the current authorized version.
3. Before editing, identify reused source seams, affected tests, allowed inputs,
   forbidden inputs, and the exact gate.
4. Implement the smallest version-scoped change that satisfies the frozen
   contract. Do not make new architecture, split, loss, policy, or training
   choices; those are fixed in `AUTONOMOUS_TRAINING_LOOP.md`.
5. Use read-only subagents for bounded review/tests when useful. Keep one writer.
6. Run focused tests, `make test`, the version command, and the idempotence or
   verification command required by the runbook.
7. Preserve every failure. Never relax a gate or select a seed/fold/origin.
8. After a passing V47, update the contracts to authorize only V48, commit, and
   continue automatically.
9. After a passing V48, update the contracts to authorize only V49, commit, and
   continue automatically.
10. Before V49 reads a real label, commit its complete source/config/tests and
    record the clean Git SHA. Do not edit them during training.
11. Run V49 serially on MPS with CPU fallback disabled. Resume only through the
    exact persisted state and retry an environmental interruption at most once.
12. After a passing V49, update the contracts, commit the completion, and stop.

## Hard stop

Stop without advancing if any scientific/audit gate fails; forbidden data is
materialized; an input or state hash drifts; a model/state becomes non-finite;
resume cannot be verified; MPS fallback would be required; or the same runtime
failure occurs twice.

When stopped, leave reproducible evidence and report the exact blocker. Do not
invent a workaround that changes the frozen experiment.

## Final report

Report:

- V47/V48/V49 decisions and hashes;
- files and commits created;
- tests and commands executed;
- origin/geometry/fold/seed checkpoint count;
- epochs, optimizer steps, and resume events;
- proof that BTC/ETH/SOL and held-out outcomes were not read and that no
  held-out/deployment predictions or PnL were produced;
- the exact next legal action, which may be V50 only after V49 passes.
