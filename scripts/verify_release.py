#!/usr/bin/env python3
"""Compile, audit, schema-check and optionally test a CTC-FA release."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def source_digest() -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] == "results":
            continue
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return digest.hexdigest(), count


def collected_test_count() -> tuple[int, str]:
    result = run([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    counts = [int(value) for value in re.findall(r":\s*(\d+)\s*$", result.stdout, re.MULTILINE)]
    if not counts:
        match = re.search(r"(\d+) tests? collected", result.stdout)
        if match:
            counts = [int(match.group(1))]
    if not counts:
        raise RuntimeError("unable to determine collected test count")
    return sum(counts), result.stdout.strip()


def dependency_versions() -> dict[str, str]:
    names = ("ctcfa", "numpy", "scipy", "sympy", "mpmath", "pytest", "jsonschema")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    compile_ok = all(
        compileall.compile_dir(ROOT / directory, quiet=1, force=True)
        for directory in ("src", "tests", "examples", "scripts")
    )

    from ctcfa import __architect__, __document_id__, __version__
    from ctcfa import registry

    audit = registry.registry_audit()
    schema = json.loads((ROOT / "schemas/metric.schema.json").read_text(encoding="utf-8"))
    schema_errors: list[str] = []
    for model in registry.models():
        try:
            jsonschema.validate(model.as_record(), schema)
        except jsonschema.ValidationError as exc:
            schema_errors.append(f"{model.id}: {exc.message}")

    test_count, inventory = collected_test_count()
    test_result = None
    if not args.skip_tests:
        test_result = run([sys.executable, "-m", "pytest"])
        if test_result.returncode != 0:
            sys.stderr.write(test_result.stdout + test_result.stderr)

    digest, file_count = source_digest()
    passed = (
        compile_ok
        and audit["pass"]
        and not schema_errors
        and test_count == 149
        and (args.skip_tests or (test_result is not None and test_result.returncode == 0))
    )
    summary = {
        "document_id": __document_id__,
        "version": __version__,
        "architect": __architect__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "dependencies": dependency_versions(),
        "compileall": "PASS" if compile_ok else "FAIL",
        "registry": audit,
        "schema_validation": {"status": "PASS" if not schema_errors else "FAIL", "errors": schema_errors},
        "tests": {
            "collected": test_count,
            "expected": 149,
            "execution": "NOT_RUN" if args.skip_tests else ("PASS" if test_result and test_result.returncode == 0 else "FAIL"),
            "inventory": inventory.splitlines(),
        },
        "source_tree": {"sha256": digest, "files_hashed": file_count},
        "overall": "PASS" if passed else "FAIL",
        "scientific_boundary": "Software verification is not physical validation of CTC existence.",
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = args.output if args.output.is_absolute() else ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(destination)
    else:
        print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
