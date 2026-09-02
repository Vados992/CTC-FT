#!/usr/bin/env python3
"""Generate and check the 33 model JSON records and Markdown index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctcfa import registry

ROOT = Path(__file__).resolve().parents[1]


def products() -> dict[Path, str]:
    generated: dict[Path, str] = {}
    rows = []
    for model in registry.models():
        record = model.as_record()
        path = ROOT / "registry" / "metrics" / f"{model.id}.json"
        generated[path] = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        title = model.title.replace("|", "\\|")
        rows.append(
            f"| `{model.id}` | {title} | `{model.kind.value}` | "
            f"`{model.expected.value}` | `{model.expected_scope.value}` |"
        )
    header = "# Model Registry\n\n"
    header += "Generated from `ctcfa.registry`; do not edit by hand. The expected status is a regression oracle within the registered model, not an empirical observation.\n\n"
    header += "| Model ID | Title | Kind | Expected status | Scope |\n"
    header += "|---|---|---|---|---|\n"
    generated[ROOT / "docs" / "MODEL_REGISTRY.md"] = header + "\n".join(rows) + "\n"
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = products()
    drift: list[str] = []
    for path, content in generated.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                drift.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if drift:
        print("generated registry drift:")
        for path in drift:
            print(f"  {path}")
        return 1
    print(f"registry products {'checked' if args.check else 'written'}: {len(generated) - 1} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
