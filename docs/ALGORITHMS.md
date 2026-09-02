# Algorithms

## A1 — Structural registry gate

`MetricModel.assert_wellformed` checks metric shape and symmetry, declared
symbols, compact-coordinate indices and generator consistency before a model
can enter the registry.

## A2 — Compact Killing-orbit scan

For compact generators `K_i`, the baseline evaluates their Gram matrix and
primitive winding vectors `n`:

```math
q(n)=\frac{n^T K n}{n^Tn}.
```

A negative value is only a candidate until the registered identification,
domain, metric signature, Killing property and time orientation also pass.
Implementation: `ctcfa.causality.compact_orbit_test`.

## A3 — Deck-orbit closure

For a quotient `M/Gamma`, reduced words in registered deck generators are
applied explicitly. Closure is tested as `p1 = gamma.p0`, not by comparing raw
coordinate endpoints. Implementation: `ctcfa.model.Quotient` and
`ctcfa.quotient_ctc`.

## A4 — General closed-loop search

Candidate curves use a truncated Fourier representation plus declared winding.
The optimizer minimizes a smoothed maximum of `g(gamma_dot,gamma_dot)` with
bounded-amplitude and non-degenerate-tangent guards. A separate refined-grid
validator checks the exact maximum, closure, boundedness, orientation and
resolution stability. Optimizer output alone is never a witness.

## A5 — Temporal-function certificate

For a candidate covector `d tau` and every physical inverse effective metric
`H_i`, the certificate requires

```math
H_i^{\mu\nu}\,\partial_\mu\tau\,\partial_\nu\tau\leq-\epsilon
```

throughout the declared domain, together with single-valuedness under all
topological identifications. The v2.1 implementation is in
`ctcfa_v21_ref.cones`.

## A6 — Energy-condition classification

The stress tensor is transformed to an orthonormal frame. Type-I closed forms
are cross-checked against constrained minimization; non-Type-I cases retain a
diagnostic rather than being silently promoted to certificates.
Implementation: `ctcfa.energy`.

## A7 — Causal phase-boundary continuation

A model-specific reduced margin is bracketed and solved with recorded tolerance,
bracket and analytic comparison where available. Normalization singularities
are kept distinct from causal boundaries. Implementation: `ctcfa.continuation`.

## A8 — Multi-cone T0-T3 classifier

`ctcfa_v21_ref.pipeline.classify` applies gates in strict order. Earlier
invalidity dominates later causal evidence: malformed specification, off-shell
background, failed EFT validity, incomplete modes and failed hyperbolicity each
stop promotion.

## A9 — Release verification

`scripts/verify_release.py` compiles every Python file, audits all 33 registry
records, validates their JSON representation against the schema, discovers the
test inventory, optionally runs it, and produces a source-tree digest and JSON
verification record.

