"""
CTC-FA v2.0   --   L0 : Unified model state vector and registry data model.

The v1.0 state vector was

    S = (M, g, topology, identifications, fields, quantum_state, provenance)
    X = (F, P, T, I, S, H, C, E, Q, D, A)

but the kernel only implemented ``(g, coords, params, compact)``; ``topology``
and ``identifications`` were free-text strings (v1.0 FM-007). This module
gives every component of the state vector a machine object, so that a global
causal statement can be *computed* from the identifications rather than
asserted in prose.

New in v2.0
-----------
* :class:`DeckTransformation` -- an explicit coordinate map with a verified
  isometry certificate. ``Gamma`` acting on the universal cover is the object
  that turns "psi is periodic" from a comment into a testable statement.
* :class:`Quotient` -- the deck group, with word enumeration, closure testing
  and a free-action check.
* :class:`CompactDirection` now carries a *verification record*: a compact
  generator is only admissible if it is a Killing vector of the declared
  metric and its orbit closes under the declared identification.
* :class:`DomainConstraint` -- machine-checkable coordinate/parameter domain,
  so that a boundary root landing outside the physical domain is rejected
  automatically (v1.0 FM-014).
* :class:`MatterSpec` -- declared stress-energy, so that L2/L9 can distinguish
  "vacuum by assumption" from "vacuum verified".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import sympy as sp

from .status import CTCStatus, ClaimScope, ModelKind

Number = float | int


# --------------------------------------------------------------------------
# Domain
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainConstraint:
    """A single machine-checkable constraint on coordinates and/or parameters.

    ``expr`` is a sympy expression that must be ``> 0`` (``strict``) or
    ``>= 0`` on the physical domain.
    """

    name: str
    expr: sp.Expr
    strict: bool = True
    note: str = ""

    def evaluate(self, subs: Mapping[sp.Symbol | str, float]) -> float:
        return float(sp.N(self.expr.subs(_normalise_subs(self.expr, subs))))

    def holds(self, subs: Mapping[sp.Symbol | str, float], tol: float = 0.0) -> bool:
        v = self.evaluate(subs)
        return v > tol if self.strict else v >= -tol


def _normalise_subs(expr: sp.Expr, subs: Mapping[Any, float]) -> dict[sp.Symbol, float]:
    """Accept substitutions keyed by symbol *or* by symbol name."""
    by_name = {str(s): s for s in expr.free_symbols}
    out: dict[sp.Symbol, float] = {}
    for k, v in subs.items():
        if isinstance(k, sp.Symbol):
            out[k] = v
        else:
            s = by_name.get(str(k))
            if s is not None:
                out[s] = v
    return out


# --------------------------------------------------------------------------
# Compact generators
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactDirection:
    """A declared compact/periodic generator direction.

    Attributes
    ----------
    index
         Coordinate index of the generator ``xi = d/dx^index``.
    name
         Coordinate name, for reporting.
    period
         Sympy expression for the identification period.
    requires_killing
         If True (default) the L4 verifier *must* certify that ``xi`` is a
         Killing vector before the compact-orbit detector will accept it.
         v1.0 accepted any declared coordinate direction with no such check.
    """

    index: int
    name: str
    period: sp.Expr = 2 * sp.pi
    requires_killing: bool = True
    note: str = ""


@dataclass
class GeneratorCertificate:
    """Result of verifying that a declared compact generator is admissible."""

    name: str
    is_killing: bool
    killing_residual: str
    closes: bool
    note: str = ""

    @property
    def admissible(self) -> bool:
        return self.is_killing and self.closes


# --------------------------------------------------------------------------
# Deck transformations and quotients (L3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeckTransformation:
    """An explicit generator of the deck group ``Gamma``.

    ``image`` is the tuple of new coordinate expressions in terms of the old
    coordinate symbols, i.e. the map ``x -> image(x)``. ``inverse`` is the
    corresponding inverse map. Both are required so that words in
    ``Gamma`` and ``Gamma^{-1}`` can be formed.

    An element is admissible as a deck transformation only if it is an
    isometry of the declared metric; :meth:`Quotient.verify_isometries`
    checks this symbolically via the pullback.
    """

    name: str
    image: tuple[sp.Expr, ...]
    inverse: tuple[sp.Expr, ...]
    orientation_preserving: bool = True
    time_orientation_preserving: bool | None = None
    note: str = ""

    def apply(self, coords: Sequence[sp.Symbol], point: Sequence[Any]) -> tuple[Any, ...]:
        sub = dict(zip(coords, point))
        return tuple(sp.simplify(e.subs(sub)) for e in self.image)

    def apply_numeric(self, coords: Sequence[sp.Symbol],
                      point: Sequence[float]) -> tuple[float, ...]:
        sub = {c: v for c, v in zip(coords, point)}
        return tuple(float(sp.N(e.subs(sub))) for e in self.image)


@dataclass
class Quotient:
    """The global identification structure ``M_universal / Gamma``.

    This object is what makes v2.0 able to *compute* the difference between
    two spacetimes with identical local metrics and different global causal
    structure (the L3 scientific gate). The canonical demonstration in the
    v2.0 registry is the pair

        ``ads4_periodic``        -- Gamma = <t -> t + 2 pi L>   : CTC_PROVED
        ``ads4_universal_cover`` -- Gamma = {e}                 : NO_CTC_CERTIFIED

    which share one local metric tensor and differ only here.
    """

    generators: tuple[DeckTransformation, ...] = ()
    label: str = "trivial"
    acts_freely: bool | None = None
    note: str = ""

    # -- group words ------------------------------------------------------

    def words(self, max_length: int = 2) -> list[tuple[str, ...]]:
        """Enumerate reduced words in the generators up to ``max_length``.

        Words are returned as tuples of signed generator names, for example
        ``("g0", "g0")`` for ``g0^2`` and ``("g0^-1",)`` for the inverse.
        The identity is included as the empty tuple.
        """
        if not self.generators:
            return [()]
        alphabet: list[str] = []
        for g in self.generators:
            alphabet.append(g.name)
            alphabet.append(g.name + "^-1")
        out: list[tuple[str, ...]] = [()]
        frontier: list[tuple[str, ...]] = [()]
        for _ in range(max_length):
            nxt: list[tuple[str, ...]] = []
            for w in frontier:
                for a in alphabet:
                    if w and _is_inverse_pair(w[-1], a):
                        continue # reduced words only
                    nxt.append(w + (a,))
            out.extend(nxt)
            frontier = nxt
        return out

    def _map_for(self, token: str) -> tuple[sp.Expr, ...]:
        inv = token.endswith("^-1")
        base = token[:-3] if inv else token
        for g in self.generators:
            if g.name == base:
                return g.inverse if inv else g.image
        raise KeyError(f"unknown deck generator {token!r}")

    def apply_word(self, coords: Sequence[sp.Symbol], point: Sequence[float],
                   word: Sequence[str],
                   params: Mapping[sp.Symbol, float] | None = None
                   ) -> tuple[float, ...]:
        """Apply a deck word numerically.

        ``params`` supplies numeric values for any *parameter* symbols that
        appear in the generator maps (a boost rapidity, a translation
        length, an identification period). Omitting them is the most common
        way a deck computation silently fails, so it is a required argument
        wherever the registry entry declares parameters.
        """
        base = dict(params or {})
        cur = tuple(float(v) for v in point)
        for token in word:
            image = self._map_for(token)
            sub = dict(base)
            sub.update({c: v for c, v in zip(coords, cur)})
            cur = tuple(float(sp.N(e.subs(sub))) for e in image)
        return cur

    def apply_word_symbolic(self, coords: Sequence[sp.Symbol],
                            point: Sequence[sp.Expr],
                            word: Sequence[str]) -> tuple[sp.Expr, ...]:
        cur = tuple(point)
        for token in word:
            image = self._map_for(token)
            sub = dict(zip(coords, cur))
            cur = tuple(sp.simplify(e.subs(sub)) for e in image)
        return cur

    # -- closure test -----------------------------------------------------

    def closes(self, coords: Sequence[sp.Symbol], p0: Sequence[float],
               p1: Sequence[float], tol: float = 1e-9,
               max_length: int = 2,
               params: Mapping[sp.Symbol, float] | None = None
               ) -> tuple[bool, tuple[str, ...] | None, float]:
        """Does ``p1`` equal ``g . p0`` for some word ``g`` in ``Gamma``?

        Returns ``(closed, word, residual)``. This is the *correct*
        closure test for a curve on a quotient manifold and replaces the
        v1.0 test, which only reduced a coordinate difference modulo a
        declared scalar period and therefore could not express Misner,
        Grant or Gott identifications at all.
        """
        best = (False, None, float("inf"))
        for w in self.words(max_length):
            try:
                 q = self.apply_word(coords, p0, w, params)
            except (TypeError, ValueError, ZeroDivisionError):
                 continue
            res = max(abs(a - b) for a, b in zip(q, p1)) if q else float("inf")
            if res < best[2]:
                 best = (res <= tol, w, res)
        return best

    # -- isometry certificate --------------------------------------------

    def verify_isometries(self, coords: Sequence[sp.Symbol],
                          metric: sp.Matrix) -> dict[str, str]:
        """Symbolically verify ``g*_ab = g_ab`` for every generator.

        The pullback is ``(phi^* g)_{ab} = g_{cd}(phi(x)) dphi^c/dx^a
        dphi^d/dx^b``. Returns a map generator-name -> residual string
        (``"0"`` when the identity is exact).
        """
        out: dict[str, str] = {}
        n = len(coords)
        for gen in self.generators:
            J = sp.Matrix(n, n, lambda a, b: sp.diff(gen.image[a], coords[b]))
            sub = dict(zip(coords, gen.image))
            g_at_image = metric.subs(sub, simultaneous=True)
            pull = sp.simplify(J.T * g_at_image * J)
            res = sp.simplify(pull - metric)
            out[gen.name] = "0" if res.is_zero_matrix else str(res)
        return out


def _is_inverse_pair(a: str, b: str) -> bool:
    return a == b + "^-1" or b == a + "^-1"


# --------------------------------------------------------------------------
# Matter
# --------------------------------------------------------------------------

@dataclass
class MatterSpec:
    """Declared stress-energy content."""

    label: str = "unspecified"
    T: sp.Matrix | None = None
    """Covariant stress-energy ``T_{mu nu}`` if a closed form is registered."""
    vacuum: bool = False
    cosmological_constant: sp.Expr = sp.S.Zero
    note: str = ""


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

@dataclass
class Provenance:
    source_refs: tuple[str, ...] = ()
    convention_note: str = ""
    derivation: str = ""
    implementation_status: str = "implemented"
    external_spec: str = ""


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

@dataclass
class MetricModel:
    """One registered spacetime model.

    Only ``id``, ``title`` and ``kind`` are mandatory; a
    ``EXTERNAL_EXACT_SPEC`` or ``COMPOSITE_GLOBAL`` model legitimately has no
    local metric and must therefore be handled by a global predicate rather
    than by the tensor engines. Calling :meth:`require_metric` on such a
    model raises, which is the mechanism that prevents CTC-FA from inventing
    a fake canonical tensor (v1.0 REQ-016).
    """

    id: str
    title: str
    coords: tuple[sp.Symbol, ...] = ()
    metric: sp.Matrix | None = None
    params: tuple[sp.Symbol, ...] = ()
    signature: tuple[int, int, int] = (1, 0, 3)
    """Declared signature as ``(n_negative, n_null, n_positive)`` in the
    (-,+,+,+) convention. Checked numerically at every evaluated point."""

    compact: tuple[CompactDirection, ...] = ()
    quotient: Quotient = field(default_factory=Quotient)
    kind: ModelKind = ModelKind.LOCAL_METRIC
    theory: str = "GR"
    matter: MatterSpec = field(default_factory=MatterSpec)
    topology: str = "unspecified"
    expected: CTCStatus = CTCStatus.UNRESOLVED
    expected_scope: ClaimScope = ClaimScope.SECTOR
    expected_at_default: CTCStatus | None = None
    """Expected verdict *at the registered default point*, when it differs
    from the family-level expectation. The Godel-type family is the reason
    this field exists: the family as a whole is UNRESOLVED because its causal
    character depends on ``m`` and ``omega``, while the default point
    ``(m, omega) = (1, 1)`` sits inside the chronology-violating region.
    Conflating the two is a category error that a single 'expected' field
    cannot express."""
    domain: tuple[DomainConstraint, ...] = ()
    default_point: Mapping[str, float] = field(default_factory=dict)
    domain_notes: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    metadata: dict[str, Any] = field(default_factory=dict)

    global_margin: Callable[[Mapping[str, float]], float] | None = None
    """Optional global/reduced numerical route-closure predicate. Negative
    means a closed causal route in the *declared reduced model*; this is
    never a global GR theorem (v1.0 section 9.2)."""

    global_margin_scope: ClaimScope = ClaimScope.REDUCED_MODEL
    global_margin_note: str = ""

    temporal_function: sp.Expr | None = None
    """Optional candidate temporal function tau(x) for the L7 negative
    certificate."""

    reference_timelike: tuple[sp.Expr, ...] | None = None
    """Optional timelike reference direction ``u^a`` used to normalise the
    reduced causal margin (L16). Required for charts in which no coordinate
    direction is timelike -- Misner (``g_TT = 0``) and the Ori-2007 style
    null-adapted cores (``g_rr = 0``) are the registry examples. When absent
    the first coordinate direction with a negative diagonal component is
    used."""

    # -- invariants -------------------------------------------------------

    def __post_init__(self) -> None:
        if self.metric is not None:
            self.assert_wellformed()

    def assert_wellformed(self) -> None:
        """Structural gate V0. Raises on any malformed registry entry.

        v1.0 defect D-02 (the van Stockum entry stored an asymmetric matrix)
        is caught here, at registry load, for every model.
        """
        g = self.metric
        if g is None:
            return
        n = len(self.coords)
        if g.shape != (n, n):
            raise ValueError(f"{self.id}: metric shape {g.shape} != ({n},{n}) "
                              f"for coords {self.coords}")
        asym = sp.simplify(g - g.T)
        if not asym.is_zero_matrix:
            raise ValueError(f"{self.id}: metric is not symmetric; g - g^T = {asym}")
        for cd in self.compact:
            if not (0 <= cd.index < n):
                raise ValueError(f"{self.id}: compact index {cd.index} out of range")
            if str(self.coords[cd.index]) != cd.name:
                raise ValueError(
                     f"{self.id}: compact generator name {cd.name!r} does not match "
                     f"coordinate {self.coords[cd.index]!r} at index {cd.index}")
        declared = set(self.coords) | set(self.params)
        free = set()
        for e in g:
            free |= {s for s in e.free_symbols if isinstance(s, sp.Symbol)}
        unknown = {s for s in free if s not in declared}
        if unknown:
            raise ValueError(f"{self.id}: metric contains undeclared symbols {unknown}")

    def require_metric(self) -> sp.Matrix:
        if self.metric is None:
            raise ValueError(
                f"{self.id} is a {self.kind.value} model with no canonical local "
                f"metric. CTC-FA v2.0 refuses to fabricate one. "
                f"External specification: {self.provenance.external_spec or 'see registry record'}")
        return self.metric

    @property
    def has_metric(self) -> bool:
        return self.metric is not None

    @property
    def dim(self) -> int:
        return len(self.coords)

    def coord_index(self, name: str) -> int:
        for i, c in enumerate(self.coords):
            if str(c) == name:
                return i
        raise KeyError(f"{self.id}: no coordinate named {name!r}")

    def symbol(self, name: str) -> sp.Symbol:
        for s in tuple(self.coords) + tuple(self.params):
            if str(s) == name:
                return s
        raise KeyError(f"{self.id}: no symbol named {name!r}")

    def all_symbols(self) -> tuple[sp.Symbol, ...]:
        return tuple(self.coords) + tuple(self.params)

    def substitution(self, values: Mapping[str | sp.Symbol, float]) -> dict[sp.Symbol, float]:
        by_name = {str(s): s for s in self.all_symbols()}
        out: dict[sp.Symbol, float] = {}
        for k, v in values.items():
            s = k if isinstance(k, sp.Symbol) else by_name.get(str(k))
            if s is None:
                raise KeyError(f"{self.id}: unknown symbol {k!r}. "
                               f"Known: {sorted(by_name)}")
            out[s] = v
        return out

    def missing_values(self, values: Mapping[str | sp.Symbol, float]) -> list[str]:
        """Which declared symbols still have no numeric value?"""
        given = {str(k) for k in values}
        return [str(s) for s in self.all_symbols() if str(s) not in given]

    def domain_report(self, values: Mapping[str | sp.Symbol, float],
                       tol: float = 0.0) -> dict[str, Any]:
        rows = []
        ok = True
        n_skipped = 0
        given = {str(k) for k in values}
        for c in self.domain:
            need = {str(s) for s in c.expr.free_symbols}
            if not need <= given:
                 # A constraint whose symbols were not all supplied is NOT
                 # evaluated and must not be reported as a violation; silently
                 # counting it as a failure was what made every partially
                 # specified continuation run look out of domain.
                 rows.append({"name": c.name, "value": None, "ok": None,
                              "skipped": sorted(need - given)})
                 n_skipped += 1
                 continue
            try:
                 v = c.evaluate(self.substitution(values))
            except Exception as exc:                        # noqa: BLE001
                 rows.append({"name": c.name, "value": None, "ok": False,
                              "error": str(exc)})
                 ok = False
                 continue
            good = v > tol if c.strict else v >= -tol
            ok = ok and good
            rows.append({"name": c.name, "value": v, "strict": c.strict, "ok": good})
        return {"ok": ok, "constraints": rows, "n_skipped": n_skipped,
                "complete": n_skipped == 0}

    def as_record(self) -> dict[str, Any]:
        """Machine-readable registry record (validated by metric.schema.json)."""
        return {
            "model_id": self.id,
            "title": self.title,
            "version": "2.0",
            "kind": self.kind.value,
            "theory": self.theory,
            "signature": list(self.signature),
            "signature_convention": "(-,+,+,+)",
            "coordinates": [str(c) for c in self.coords],
            "parameters": [str(p) for p in self.params],
            "metric_defined": self.metric is not None,
            "metric_components": (
                 {f"{i}{j}": str(self.metric[i, j])
                  for i in range(self.dim) for j in range(i, self.dim)
                  if self.metric[i, j] != 0}
                 if self.metric is not None else None),
            "topology": self.topology,
            "identifications": [
                 {"coordinate": c.name, "index": c.index, "period": str(c.period),
                  "requires_killing": c.requires_killing, "note": c.note}
                 for c in self.compact],
            "quotient": {
                 "label": self.quotient.label,
                 "generators": [
                     {"name": g.name,
                      "image": [str(e) for e in g.image],
                      "inverse": [str(e) for e in g.inverse],
                      "time_orientation_preserving": g.time_orientation_preserving,
                      "note": g.note}
                     for g in self.quotient.generators],
                 "acts_freely": self.quotient.acts_freely,
                 "note": self.quotient.note,
            },
            "matter": {"label": self.matter.label, "vacuum": self.matter.vacuum,
                        "Lambda": str(self.matter.cosmological_constant),
                        "T_defined": self.matter.T is not None,
                        "note": self.matter.note},
            "domain": [{"name": c.name, "expression": str(c.expr),
                         "strict": c.strict, "note": c.note} for c in self.domain],
            "default_point": dict(self.default_point),
            "expected_status": self.expected.value,
            "expected_status_at_default_point": (
                 self.expected_at_default.value if self.expected_at_default
                 else self.expected.value),
            "expected_scope": self.expected_scope.value,
            "domain_notes": self.domain_notes,
            "global_margin": self.global_margin is not None,
            "global_margin_scope": self.global_margin_scope.value,
            "global_margin_note": self.global_margin_note,
            "temporal_function": (str(self.temporal_function)
                                   if self.temporal_function is not None else None),
            "provenance": {
                 "source_refs": list(self.provenance.source_refs),
                 "convention_note": self.provenance.convention_note,
                 "derivation": self.provenance.derivation,
                 "implementation_status": self.provenance.implementation_status,
                 "external_spec": self.provenance.external_spec,
            },
            "metadata": self.metadata,
        }


__all__ = [
    "MetricModel", "CompactDirection", "GeneratorCertificate", "DeckTransformation",
    "Quotient", "DomainConstraint", "MatterSpec", "Provenance",
]
