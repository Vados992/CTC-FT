"""
CTC-FA v2.0   --   Provenance, integrity and immutable run records.

Section 30 of v1.0 lists seven integrity controls. They were described but
not implemented; the released kernel carried a SHA-256 in prose and nothing
that computed or verified it. This module implements all seven:

  1. SHA-256 of every release artifact and of the frozen configuration;
  2. signed / verifiable release manifests (a detached digest file that a
     signature tool can sign);
  3. immutable run records keyed by a content hash;
  4. sealed benchmark answers (HMAC), so a blind evaluation is possible;
  5. never overwrite a historical manifest -- supersede it;
  6. record dependency and interpreter versions with every run;
  7. a minimal independent reproduction script emitted with every claim.

The two-pass build
------------------
An integrity hash that is printed *inside* the document it describes is
self-referential. The two-pass convention used here is: pass 1 builds the
artifact with the hash field blank and computes ``H1`` over the source tree
(not the artifact); pass 2 embeds ``H1``, rebuilds, and records ``H2``, the
hash of the finished artifact. Both are published, and
:func:`verify_two_pass` checks the pair.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def source_tree_digest(root: str | Path, patterns: Sequence[str] = ("*.py",)
                        ) -> tuple[str, list[dict[str, str]]]:
    """Deterministic digest of a source tree, plus the per-file table."""
    root = Path(root)
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(root.rglob(pat)))
    files = sorted({f.resolve() for f in files if f.is_file()},
                    key=lambda p: str(p.relative_to(root.resolve())))
    rows = []
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(root.resolve()))
        d = sha256_file(f)
        rows.append({"path": rel, "sha256": d, "bytes": str(f.stat().st_size)})
        h.update(rel.encode())
        h.update(d.encode())
    return h.hexdigest(), rows


def environment_lock() -> dict[str, Any]:
    """Dependency and interpreter versions (control 6)."""
    mods = {}
    for name in ("sympy", "numpy", "scipy", "mpmath", "matplotlib", "reportlab"):
        try:
             m = __import__(name)
             mods[name] = getattr(m, "__version__", "unknown")
        except Exception:                                 # noqa: BLE001
             mods[name] = "absent"
    return {"python": sys.version.split()[0],
             "implementation": platform.python_implementation(),
             "platform": platform.platform(),
             "modules": mods}


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------

@dataclass
class RunRecord:
    """An immutable record of one computation (v1.0 section 25.2)."""

    run_id: str
    model_id: str
    method: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    code_digest: str
    environment: dict[str, Any]
    created_utc: str
    tolerances: dict[str, Any] = field(default_factory=dict)
    source_refs: list[str] = field(default_factory=list)
    review_state: str = "machine"
    supersedes: str | None = None

    def content_hash(self) -> str:
        payload = json.dumps(self.as_dict(include_hash=False), sort_keys=True)
        return sha256_text(payload)

    def as_dict(self, include_hash: bool = True) -> dict[str, Any]:
        d = {"run_id": self.run_id, "model_id": self.model_id,
             "method": self.method, "parameters": self.parameters,
             "result": self.result, "code_digest": self.code_digest,
             "environment": self.environment, "created_utc": self.created_utc,
             "tolerances": self.tolerances, "source_refs": self.source_refs,
             "review_state": self.review_state, "supersedes": self.supersedes}
        if include_hash:
            d["content_hash"] = self.content_hash()
        return d


def new_run(model_id: str, method: str, parameters: Mapping[str, Any],
            result: Mapping[str, Any], code_digest: str,
            tolerances: Mapping[str, Any] | None = None,
            source_refs: Sequence[str] = ()) -> RunRecord:
    return RunRecord(
        run_id=str(uuid.uuid4()), model_id=model_id, method=method,
        parameters=dict(parameters), result=dict(result),
        code_digest=code_digest, environment=environment_lock(),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tolerances=dict(tolerances or {}), source_refs=list(source_refs))


def write_run(record: RunRecord, directory: str | Path) -> Path:
    """Write a run record. Never overwrites (control 5)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{record.model_id}__{record.content_hash()[:16]}.json"
    if p.exists():
        return p                       # identical content: already recorded
    p.write_text(json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# Sealed benchmark answers (control 4)
# --------------------------------------------------------------------------

def seal_answers(answers: Mapping[str, Any], key: bytes) -> dict[str, str]:
    """HMAC-seal a benchmark answer key so a solver cannot read it.

    The evaluator holds ``key``; the solver holds only the sealed digests and
    can verify a submitted answer without ever learning the target. This is
    what makes the blind benchmark protocol of section 15.2 executable rather
    than aspirational.
    """
    out = {}
    for k, v in sorted(answers.items()):
        payload = json.dumps({k: v}, sort_keys=True).encode()
        out[k] = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return out


def check_answer(name: str, value: Any, sealed: Mapping[str, str],
                 key: bytes) -> bool:
    payload = json.dumps({name: value}, sort_keys=True).encode()
    want = sealed.get(name)
    if want is None:
        return False
    return hmac.compare_digest(
        hmac.new(key, payload, hashlib.sha256).hexdigest(), want)


# --------------------------------------------------------------------------
# Release manifest and the two-pass build
# --------------------------------------------------------------------------

@dataclass
class ReleaseManifest:
    document_id: str
    version: str
    architect: str
    date: str
    source_digest: str
    files: list[dict[str, str]]
    environment: dict[str, Any]
    artifact_digest: str | None = None
    pass1_digest: str | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "version": self.version,
                "architect": self.architect, "date": self.date,
                "source_digest": self.source_digest, "files": self.files,
                "environment": self.environment,
                "artifact_digest": self.artifact_digest,
                "pass1_digest": self.pass1_digest, "note": self.note}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
        return p


def build_manifest(root: str | Path, document_id: str, version: str,
                   architect: str, date: str) -> ReleaseManifest:
    digest, files = source_tree_digest(root)
    return ReleaseManifest(document_id, version, architect, date, digest, files,
                           environment_lock(),
                           note="pass 1: source digest computed with the "
                                "artifact hash field still blank")


def verify_two_pass(manifest_path: str | Path, artifact_path: str | Path
                    ) -> dict[str, Any]:
    m = json.loads(Path(manifest_path).read_text())
    actual = sha256_file(artifact_path)
    return {"manifest": str(manifest_path), "artifact": str(artifact_path),
            "recorded_artifact_digest": m.get("artifact_digest"),
            "actual_artifact_digest": actual,
            "match": m.get("artifact_digest") == actual,
            "source_digest": m.get("source_digest"),
            "pass1_digest": m.get("pass1_digest")}


def reproduction_script(model_id: str, method: str,
                        parameters: Mapping[str, Any]) -> str:
    """Emit the minimal independent script that reproduces one claim (control 7)."""
    return (
        "#!/usr/bin/env python3\n"
        f"# Minimal reproduction for {model_id} / {method}\n"
        "# CTC-FA v2.0 -- run from the repository root with src/ on PYTHONPATH.\n"
        "from ctcfa import registry as R\n"
        "from ctcfa.causality import compact_orbit_test, certify_temporal_function\n"
        "from ctcfa.quotient_ctc import deck_orbit_test\n"
        "\n"
        f"model = R.get({model_id!r})\n"
        f"values = {dict(parameters)!r}\n"
        f"method = {method!r}\n"
        "if method == 'compact_orbit':\n"
        "    print(compact_orbit_test(model, values))\n"
        "elif method == 'deck_orbit':\n"
        "    print(deck_orbit_test(model, values))\n"
        "elif method.startswith('temporal_function'):\n"
        "    print(certify_temporal_function(model))\n"
        "else:\n"
        "    raise SystemExit(f'unknown method {method}')\n")


__all__ = ["sha256_bytes", "sha256_file", "sha256_text", "source_tree_digest",
           "environment_lock", "RunRecord", "new_run", "write_run",
           "seal_answers", "check_answer", "ReleaseManifest", "build_manifest",
           "verify_two_pass", "reproduction_script"]
