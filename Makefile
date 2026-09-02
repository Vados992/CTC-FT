# CTC-FA v2.1 -- reproducible build
PY      ?= python3
SRC      = src
EXPORT   = registry/metrics
RESULTS = results

export PYTHONPATH := $(SRC)

.PHONY: all selftest test registry manifest examples verify matrix clean

all: selftest test registry manifest examples

selftest:
	$(PY) -m ctcfa.cli selftest

test:
	$(PY) -m pytest -q

verify:
	$(PY) scripts/verify_release.py --output $(RESULTS)/verification-summary.json

registry:
	$(PY) -m ctcfa.cli export-registry --out $(EXPORT)
	$(PY) -c "import json,glob,jsonschema; s=json.load(open('schemas/metric.schema.json')); \
           [jsonschema.validate(json.load(open(f)), s) for f in glob.glob('$(EXPORT)/*.json')]; \
           print('registry records validate against schemas/metric.schema.json')"

manifest:
	mkdir -p $(RESULTS)
	$(PY) -m ctcfa.cli manifest --out $(RESULTS)/manifest.json --root $(SRC)/ctcfa

examples:
	@for f in examples/ex*.py; do \
	  echo "== $$f"; $(PY) $$f || exit 1; \
	done

matrix:
	$(PY) -m ctcfa.cli matrix

clean:
	$(PY) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').rglob('__pycache__')]"
	$(PY) -c "from pathlib import Path; import shutil; [shutil.rmtree(Path(p), ignore_errors=True) for p in ('.pytest_cache','.ruff_cache','build')]"
