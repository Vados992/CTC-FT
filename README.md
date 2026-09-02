# CTC-FA v2.1

**Closed Timelike Curve Falsification Architecture**

System Architect: **Vadym Tsinderhoz**  
Document ID: `CTC-FA-2026-V2.1-COMPLETE-EN`

[![CI](https://github.com/Vados992/CTC-FT/actions/workflows/ci.yml/badge.svg)](https://github.com/Vados992/CTC-FT/actions/workflows/ci.yml)

CTC-FA is a research-software architecture for testing causal claims about
closed timelike curves across heterogeneous spacetime models. It combines a
33-model general-relativity baseline with a smaller verified v2.1 reference
core for theory/solution identity, field-equation admissibility, effective
causal cones and topology-aware closure.

> This repository is not evidence that a physically realizable time machine
> exists. Software-test success validates implementation behavior only; it does
> not establish physical existence, constructibility or experimental detection.

## Verified release snapshot

| Item | Release status | Evidence |
|---|---|---|
| v2.0 computational baseline recovered from the published listing | `RECOVERED_AND_RERUN` | 118/118 tests pass after indentation-only page-boundary repair |
| Registry | `VERIFIED_STRUCTURAL` | 33/33 records load; zero structural-audit problems |
| v2.1 T0-T3 reference core | `VERIFIED` | 31/31 dependency-light tests pass |
| Combined repository | `VERIFIED_IN_BUILD_ENVIRONMENT` | 149/149 collected tests pass |
| Complete consolidated document | `INCLUDED` | 264-page PDF; SHA-256 `980d88a5562b...e3cf5c7e` |
| Rotating k-essence numerical adapter | `SPECIFIED_NOT_IMPLEMENTED` | contract and analytic controls only |
| Perturbative EdGB numerical adapter | `SPECIFIED_NOT_IMPLEMENTED` | contract and V&V plan only |
| Generic quartic-Horndeski characteristics | `SPECIFIED_NOT_IMPLEMENTED` | implicit interface only |
| Full 3+1 evolution / observational inference | `NOT_IMPLEMENTED` | external integration required |

The machine-readable ledger is in
[`spec/implementation-state.json`](spec/implementation-state.json). Scientific
claims must not be promoted beyond the state recorded there.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Run the registry and a model dossier:

```bash
ctcfa list
ctcfa show misner
ctcfa validate misner
ctcfa ctc misner --point T=1 --point psi=0 --point y=0 --point z=0
ctcfa matrix
```

Create an auditable release report:

```bash
python scripts/verify_release.py --output results/verification-summary.json
```

## Repository map

```text
src/ctcfa/             recovered, rerun v2.0 computational baseline
src/ctcfa_v21_ref/     verified v2.1 T0-T3 reference core
registry/metrics/      33 machine-readable model records
schemas/               JSON Schemas for metrics, runs, theorems and verdicts
tests/                 analytic, blind, control, regression, unit and v2.1 tests
examples/              ten executable baseline examples
docs/                  architecture, algorithms, RTM, V&V and boundaries
spec/                  implementation ledger and traceability data
scripts/               release and registry verification utilities
artifacts/             document checksums and distribution notes
```

The full standalone document is
[`artifacts/CTC-FA_v2.1_COMPLETE_CONSOLIDATED_RESEARCH_ARCHITECTURE_EN_Vadym_Tsinderhoz.pdf`](artifacts/CTC-FA_v2.1_COMPLETE_CONSOLIDATED_RESEARCH_ARCHITECTURE_EN_Vadym_Tsinderhoz.pdf).

## Scientific scope

The baseline can evaluate explicit metrics, registered quotients, compact
Killing directions, candidate temporal functions, selected energy-condition
margins, horizons and model-specific continuation problems. Negative local or
sector results are not silently widened into global no-CTC theorems.

The v2.1 core adds an explicit theory-to-solution contract and multi-mode cone
logic. It deliberately refuses a verdict when the field equations, domain,
topology or effective characteristic bundle are incomplete.

See [Scientific Boundaries](docs/SCIENTIFIC_BOUNDARIES.md) before quoting a
result, and [Source Recovery](docs/SOURCE_RECOVERY.md) for the exact status of
the recovered v2.0 source.

## Citation and rights

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The source is
publicly inspectable but is **not released under an open-source license**.
See [`LICENSE_IP_NOTICE_v2.1.md`](LICENSE_IP_NOTICE_v2.1.md).
