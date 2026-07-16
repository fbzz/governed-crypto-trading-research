---
name: tlm-research-governor
description: Govern the TLM experiment lineage and authorization boundary from repository evidence. Use when the user asks "where are we", "what is next", "continue", "next loop", or asks how many families, runs, checkpoints, or evaluations exist; also use before opening research data, implementing a phase, training, evaluating, or unsealing outcomes.
---

# TLM Research Governor

Reconstruct current state from passed artifacts and frozen configs. Report one
authorized next action and prevent work from crossing that boundary.

## Govern the repository

1. Run from the repository root:

   ```bash
   python3 .agents/skills/tlm-research-governor/scripts/govern.py --json
   ```

2. Stop if `ready` is false. Resolve contract or documentation drift without
   reading data or outcomes.
3. State the active family separately from its execution objects. Use the
   taxonomy in [taxonomy.md](references/taxonomy.md); never call every seed,
   checkpoint, or evaluation a separate model.
4. Treat `research/current.yaml`, its referenced experiment contract, and the
   repository `research-status` validator as binding. Historical prose is
   context, not current permission.
5. Report only `authorized_next_action` as executable. Later lifecycle actions
   are a roadmap, not authorization.
6. Before opening any proposed path, validate it against the current phase:

   ```bash
   python3 .agents/skills/tlm-research-governor/scripts/govern.py \
     --check-path PATH [PATH ...]
   ```

7. Use read-only subagents for lineage, leakage, or reproducibility reviews.
   Keep one writer for the authorized phase.

## Enforce the boundary

- Read metadata JSON/YAML, source, tests, and documentation only as needed.
- Never deserialize a Parquet, market panel, label table, prediction table, or
  realized-return table unless the current receipt explicitly authorizes it.
- Keep BTC, ETH, SOL, their symbols, target proxies, target predictions, and
  target PnL sealed until an explicit receipt opens them.
- Do not instantiate a real-data model, fit a scaler, create a real checkpoint,
  train, infer, backtest, evaluate, or unseal outcomes during a synthetic-only
  phase.
- Do not infer permission from a future lifecycle field, unchecked task, stale
  README paragraph, existing implementation, or file presence.
- If repository sources disagree, stop at governance repair. Do not choose the
  most permissive interpretation.

## Hand off

Hand the exact phase token, authorization receipt, permitted inputs, forbidden
accesses, and gate text to `$tlm-loop-engineer`. Do not begin the following
phase after the gate passes; return to this governor for a fresh receipt.
