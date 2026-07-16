---
name: tlm-retirement-autopsy
description: Diagnose why a frozen TLM experiment family failed while preserving its retirement decision and sealed research boundaries. Use when asked for a failure autopsy, post-mortem, evidence review, gate attribution, or explanation of a retired family without retuning, counterfactual PnL, new bootstraps, model inference, or outcome-driven selection.
---

# TLM Retirement Autopsy

Perform a deterministic, read-only diagnosis. Treat retirement as immutable and consumed outcomes as development evidence.

## Workflow

1. Read the active repository contract and the retired family's frozen evaluation result, gate result, receipts, and manifests.
2. Define a hash-locked input allowlist before opening detailed tables. Follow [references/autopsy-contract.md](references/autopsy-contract.md).
3. Lint the contract:

   ```bash
   python3 .agents/skills/tlm-retirement-autopsy/scripts/lint_autopsy_contract.py CONTRACT.json
   ```

4. Verify every allowlisted input before analysis:

   ```bash
   python3 .agents/skills/tlm-retirement-autopsy/scripts/verify_frozen_inputs.py \
     CONTRACT.json --root REPOSITORY_ROOT
   ```

5. Read only allowlisted inputs. Materialize every registered group, including sparse, empty, and failed cells.
6. Attribute failure separately across:
   - signal: ordinal or predictive information;
   - calibration: scale, direction, bias, coverage, or origin drift;
   - churn: transitions, episode duration, concentration, and turnover;
   - cost: gross-to-net decomposition under only registered costs.
7. Re-run the hash verifier after analysis. Fail closed on drift or an undeclared read.
8. Report facts before interpretations. Keep the original decision unchanged and recommend only a new ex-ante family, future unseen confirmation, or termination.

## Hard boundaries

- Do not train, instantiate, infer, recalibrate, or inspect checkpoints.
- Do not alter a threshold, architecture, feature, fold, seed, asset, date, cost, or gate.
- Do not calculate counterfactual PnL, alternative policies, or a new bootstrap/cost grid.
- Do not exclude inconvenient rows or promote a diagnostic slice.
- Do not open raw panels, sealed target assets, or later observations unless explicitly allowlisted by the frozen contract.
- Do not describe beating a weak control as deployable evidence; show absolute gross and net performance separately.
- Mark unavailable evidence as unavailable. Never recreate missing seed, triplet, or context predictions through inference.

The scripts are validators, not authorization. Repository contracts and frozen artifacts remain authoritative.
