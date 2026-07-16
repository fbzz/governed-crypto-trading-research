# TLM research-object taxonomy

- **Family**: one frozen hypothesis, architecture, objective, policy, and
  lifecycle identity. A changed hypothesis or contract creates a new family.
- **Run or job**: one execution cell such as origin x geometry x fold x seed.
  Registered future jobs are not trained models.
- **Checkpoint**: one persisted trained state produced by a run. A checkpoint
  is not an independent family or economic candidate.
- **Evaluation**: one frozen protocol consuming predictions and, only after an
  authorized unseal, outcomes. It produces evidence about a family; it is not a
  checkpoint.
- **Candidate**: a family that passed every preceding gate and was immutably
  registered. A specification, synthetic harness, trained checkpoint, or
  partially passing evaluation is not a candidate.

When counting, report all five nouns explicitly and qualify `registered`,
`started`, `completed`, `retired`, or `authorized`. Never collapse them into a
single count called "models".
