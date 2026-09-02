# Requirements Traceability Matrix

The authoritative machine-readable RTM is [`spec/rtm.csv`](../spec/rtm.csv).
It maps each requirement to implementation, verification evidence and current
status.

The matrix uses four status classes:

- `PASS`: implemented and exercised by the named evidence.
- `RECOVERED_AND_RERUN`: recovered from the v2.0 PDF listing, repaired only for
  page-boundary indentation loss, then rerun successfully.
- `SPECIFIED_NOT_IMPLEMENTED`: contract exists but executable solver does not.
- `NOT_ATTEMPTED`: planned integration or external validation has not run.

Requirements concerning physical existence cannot be satisfied by unit tests.
They remain outside the permitted claim boundary until observational evidence
and an accepted physical model exist.

