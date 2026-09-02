# Contributing

Contributions must preserve the distinction between software verification and
physical validation.

1. Open an issue describing the model, source and permitted claim scope.
2. Add primary references and explicit coordinate, signature and topology
   conventions.
3. Add a negative control and at least one analytic or independently computed
   benchmark.
4. Add or update the RTM and implementation ledger.
5. Run `python scripts/verify_release.py` before submitting a pull request.

A test may verify code behavior. It must not be described as experimental proof
of a CTC, time machine or modified-gravity theory.

