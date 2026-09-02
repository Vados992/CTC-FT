"""
CTC-FA v2.0    --   command line interface.

Section 26.1 of v1.0 specified a CLI and then noted: "The CLI above is a
normative target interface, not a claim that every command already exists in
the single-file v1.0 kernel." In v2.0 every command below exists and runs.

      ctcfa list
      ctcfa show <model>
      ctcfa validate [<model>] [--no-cross-check]
      ctcfa tensors <model> [--point k=v ...] [--what ricci|einstein|kretschmann]
      ctcfa ctc <model> [--point k=v ...]
      ctcfa certify-time <model> [--box r=lo:hi ...]
      ctcfa boundary <model> --parameter p --bracket lo:hi [--fixed k=v ...]
                              [--box r=lo:hi ...] [--analytic X]
      ctcfa energy <model> [--point k=v ...]
      ctcfa horizons <model> [--variable r]
      ctcfa theorems <model>
      ctcfa quotient <model>
      ctcfa matrix
      ctcfa meta
      ctcfa manifest --out results/manifest.json
      ctcfa export-registry --out registry/metrics
      ctcfa selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sympy as sp


def _kv(pairs: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = float(sp.N(sp.sympify(v.strip())))
    return out


def _boxes(pairs: list[str] | None) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for p in pairs or []:
        k, rng = p.split("=", 1)
        lo, hi = rng.split(":", 1)
        out[k.strip()] = (float(sp.N(sp.sympify(lo))), float(sp.N(sp.sympify(hi))))
    return out


def _dump(obj: Any) -> None:
    from .report import to_json
    print(to_json(obj))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ctcfa",
                                 description="CTC-FA v2.0 Closed Timelike Curve "
                                             "Falsification Architecture")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered models")

    p = sub.add_parser("show", help="show one registry record")
    p.add_argument("model")

    p = sub.add_parser("validate", help="run the V0/V1/V2/V3 gates")
    p.add_argument("model", nargs="?")
    p.add_argument("--no-cross-check", action="store_true")

    p = sub.add_parser("tensors", help="curvature quantities")
    p.add_argument("model")
    p.add_argument("--point", action="append")
    p.add_argument("--what", default="ricci",
                   choices=["ricci", "ricci_scalar", "einstein", "kretschmann"])

    p = sub.add_parser("ctc", help="run the appropriate CTC detector")
    p.add_argument("model")
    p.add_argument("--point", action="append")
    p.add_argument("--energy", action="store_true")
    p.add_argument("--theorems", action="store_true")

    p = sub.add_parser("certify-time", help="temporal-function certificate")
    p.add_argument("model")
    p.add_argument("--box", action="append")

    p = sub.add_parser("boundary", help="causal phase-boundary continuation")
    p.add_argument("model")
    p.add_argument("--parameter", required=True)
    p.add_argument("--bracket", required=True, help="lo:hi")
    p.add_argument("--fixed", action="append")
    p.add_argument("--box", action="append")
    p.add_argument("--analytic", type=float, default=None)

    p = sub.add_parser("energy", help="energy-condition report")
    p.add_argument("model")
    p.add_argument("--point", action="append")

    p = sub.add_parser("horizons", help="horizon taxonomy")
    p.add_argument("model")
    p.add_argument("--variable", default="r")
    p.add_argument("--no-curvature", action="store_true")

    p = sub.add_parser("theorems", help="theorem applicability matrix")
    p.add_argument("model")

    p = sub.add_parser("quotient", help="L3 identification certificate")
    p.add_argument("model")

    sub.add_parser("matrix", help="regression matrix over the whole registry")
    sub.add_parser("meta", help="cross-family meta-analysis")

    p = sub.add_parser("manifest", help="build a release manifest")
    p.add_argument("--out", default="results/manifest.json")
    p.add_argument("--root", default=None)

    p = sub.add_parser("export-registry", help="write machine-readable records")
    p.add_argument("--out", default="registry/metrics")

    sub.add_parser("selftest", help="full self-test (validation + matrix)")

    a = ap.parse_args(argv)

    from . import __architect__, __document_id__, __version__
    from . import registry as R

    if a.cmd == "list":
        from .registry import summary
        rows = summary()
        w = max(len(r["model_id"]) for r in rows) + 2
        print(f"CTC-FA v{__version__} | {__document_id__} | "
              f"system architect: {__architect__}")
        print(f"{'model':<{w}}{'kind':<22}{'expected':<20}{'scope':<18}refs")
        print("-" * (w + 22 + 20 + 18 + 12))
        for r in rows:
            print(f"{r['model_id']:<{w}}{r['kind']:<22}{r['expected']:<20}"
                  f"{r['scope']:<18}{r['refs']}")
        print(f"\n{len(rows)} registered models.")
        return 0

    if a.cmd == "show":
        _dump(R.get(a.model).as_record())
        return 0

    if a.cmd == "validate":
        from .validate import validate_model, validate_registry
        if a.model:
            _dump(validate_model(R.get(a.model),
                                  cross_check=not a.no_cross_check).as_dict())
        else:
            rep = validate_registry(cross_check=not a.no_cross_check)
            _dump({"n_models": rep["n_models"], "n_failed": rep["n_failed"],
                    "pass": rep["pass"],
                    "gates": {r["model_id"]: r["gates"]["results"]
                              for r in rep["reports"]}})
        return 0

    if a.cmd == "tensors":
        from .symbolic import (einstein_tensor, kretschmann, ricci_scalar,
                               ricci_tensor)
        m = R.get(a.model)
        g, c = m.require_metric(), m.coords
        if a.what == "ricci":
            res = ricci_tensor(g, c)
        elif a.what == "ricci_scalar":
            res = ricci_scalar(g, c)
        elif a.what == "einstein":
            res = einstein_tensor(g, c)
        else:
            res = kretschmann(g, c)
        pt = _kv(a.point)
        if pt:
            res = sp.Matrix(res).subs(m.substitution(pt)).evalf(20) \
                 if hasattr(res, "shape") else sp.N(res.subs(m.substitution(pt)), 20)
        print(sp.pretty(res) if hasattr(res, "shape") else res)
        return 0

    if a.cmd == "ctc":
        from .report import analyse
        m = R.get(a.model)
        pt = _kv(a.point) or dict(m.default_point)
        _dump(analyse(m, pt, include_energy=a.energy,
                       include_theorems=a.theorems))
        return 0

    if a.cmd == "certify-time":
        from .causality import certify_temporal_function
        m = R.get(a.model)
        box = {m.symbol(k): v for k, v in _boxes(a.box).items()} or None
        _dump(certify_temporal_function(m, box=box).as_dict())
        return 0

    if a.cmd == "boundary":
        from .continuation import causal_phase_boundary
        m = R.get(a.model)
        lo, hi = a.bracket.split(":", 1)
        th = causal_phase_boundary(m, a.parameter,
                                   (float(lo), float(hi)), _kv(a.fixed),
                                   _boxes(a.box), analytic_value=a.analytic)
        _dump(th.as_dict())
        return 0

    if a.cmd == "energy":
        from .energy import energy_report
        m = R.get(a.model)
        _dump(energy_report(m, _kv(a.point) or None).as_dict())
        return 0

    if a.cmd == "horizons":
        from .horizons import horizon_report
        _dump(horizon_report(R.get(a.model), a.variable,
                             with_curvature=not a.no_curvature))
        return 0

    if a.cmd == "theorems":
        from .theorems import applicability_matrix
        _dump(applicability_matrix(R.get(a.model)))
        return 0

    if a.cmd == "quotient":
        from .topology import classify, verify_quotient
        m = R.get(a.model)
        _dump({"classification": classify(m),
               "certificate": verify_quotient(m).as_dict()})
        return 0

    if a.cmd == "matrix":
        from .report import format_matrix, matrix_agreement, regression_matrix
        rows = regression_matrix()
        print(format_matrix(rows))
        ag = matrix_agreement(rows)
        print(f"\nagreement {ag['agree']}/{ag['n']} pass={ag['pass']}")
        for mm in ag["mismatches"]:
            print(" MISMATCH", mm)
        return 0 if ag["pass"] else 1

    if a.cmd == "meta":
        from .meta import cross_family_report
        _dump(cross_family_report(R.models()))
        return 0

    if a.cmd == "manifest":
        from .provenance import build_manifest
        root = Path(a.root) if a.root else Path(__file__).resolve().parent
        man = build_manifest(root, __document_id__, __version__, __architect__,
                             "2026-08-31")
        man.write(a.out)
        print(f"source digest: {man.source_digest}")
        print(f"files: {len(man.files)}")
        print(f"manifest written to {a.out}")
        return 0

    if a.cmd == "export-registry":
        paths = R.export_records(a.out)
        print(f"{len(paths)} records written to {a.out}")
        return 0

    if a.cmd == "selftest":
        from .registry import registry_audit
        from .report import format_matrix, matrix_agreement, regression_matrix
        from .validate import validate_registry
        print(f"CTC-FA v{__version__} self-test")
        aud = registry_audit()
        print(f" registry audit : {'PASS' if aud['pass'] else 'FAIL'} "
              f"({aud['n_models']} models, {aud['n_problems']} problems)")
        val = validate_registry(cross_check=True)
        print(f" V0/V1 gates     : {'PASS' if val['pass'] else 'FAIL'} "
              f"({val['n_models'] - val['n_failed']}/{val['n_models']})")
        rows = regression_matrix()
        ag = matrix_agreement(rows)
        print(f" regression      : {'PASS' if ag['pass'] else 'FAIL'} "
              f"({ag['agree']}/{ag['n']})")
        print()
        print(format_matrix(rows))
        ok = aud["pass"] and val["pass"] and ag["pass"]
        print(f"\nCTC-FA v{__version__} self-test: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    ap.error(f"unknown command {a.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
