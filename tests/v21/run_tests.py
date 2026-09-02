"""Dependency-free test runner for the verified CTC-FA v2.1 reference core."""

from __future__ import annotations

import inspect
import sys
import time
import traceback

import test_reference_core as suite


def main() -> int:
    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(suite, inspect.isfunction)
        if name.startswith("test_")
    ]
    start = time.perf_counter()
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"FAIL {name}")
    elapsed = time.perf_counter() - start
    print(f"SUMMARY total={len(tests)} passed={len(tests)-len(failures)} failed={len(failures)} elapsed={elapsed:.6f}s")
    for name, detail in failures:
        print(f"--- {name} ---")
        print(detail)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
