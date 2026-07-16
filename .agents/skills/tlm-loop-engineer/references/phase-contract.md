# One-phase execution contract

The handoff into a loop must identify:

- active family identity;
- completed authorization receipt and its SHA-bound config;
- exact phase token and action enum;
- allowed inputs and forbidden accesses;
- required outputs, tests, smoke/replay commands, and gate text;
- explicitly unauthorized following phase.

The loop is complete only when repository code produces its own audit and
result receipts, focused tests and the relevant broader suite pass, replay
requirements pass, and documentation records the observed result. The guard
does not replace those checks.

Stop immediately when authorization is ambiguous, an input hash drifts, a
forbidden path would be needed, a gate fails, or the current phase completes.
Do not repair a scientific failure by changing its frozen contract.
