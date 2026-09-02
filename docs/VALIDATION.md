# Verification and Validation

## Current executable evidence

| Suite | Tests | Purpose |
|---|---:|---|
| Analytic benchmarks | 19 | known margins, boundaries and quotient identities |
| Blind benchmarks | 19 | sealed parameter recovery and score reporting |
| Negative controls | 20 | reject false positives and over-broad negatives |
| Regression | 25 | 33-model registry behavior, fields, energy and theorem gates |
| Convention tests | 14 | curvature, signs, reductions and symmetry |
| Engine tests | 13 | frames, Killing equations and independent numeric routes |
| Baseline status/gate tests | 8 | scope and claim-level enforcement |
| v2.1 reference-core tests | 31 | T0-T3 identity, EFT, multi-cone, topology and promotion |
| **Total** | **149** | combined release suite |

All 149 tests passed in the final local build environment on 2026-09-02. This
statement is software verification. It does not mean that all registered
metrics are physically realizable or that any CTC has been observed.

## V&V levels

1. **Schema and structural verification:** required for every record.
2. **Analytic control:** compare with an exact limit or known invariant.
3. **Independent numerical route:** required where a certificate depends on
   floating-point computation.
4. **Convergence and perturbation:** resolution, tolerance and domain tests.
5. **Blind/held-out validation:** derivation and evaluation data separated.
6. **External reproduction:** not yet completed for this release.
7. **Observational validation:** not available for CTC existence.

## Acceptance rule

A failing or unexecuted required gate prevents claim promotion. Tests marked as
controls may establish the absence of a particular false-positive mechanism;
they do not prove global chronology protection.

