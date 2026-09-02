# Reproducibility

## Supported environment

- CPython 3.11 or 3.12
- NumPy 1.26 to 2.x
- SciPy 1.11 to 1.x
- SymPy 1.13 to 1.x
- mpmath 1.3 to 1.x

The lower bounds reproduce the documented v2.0 dependency contract; upper
bounds prevent unreviewed major-version changes.

## Reproduction sequence

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m compileall -q -f src tests examples scripts
python -m ruff check src tests examples scripts
python -m pytest
python scripts/generate_registry_docs.py --check
python scripts/verify_release.py --skip-tests
```

For a release record containing the test execution itself, omit
`--skip-tests`. Generated verification JSON records Python and dependency
versions, model count, test count, registry status and the source-tree digest.

Floating-point outputs are meaningful only with their domain, tolerance,
normalization and implementation version.

