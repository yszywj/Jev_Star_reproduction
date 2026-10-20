# Offline regression CI

## When CI runs

The workflow `Offline regression tests` (`.github/workflows/tests.yml`) runs on every `push` and `pull_request`. It uses a matrix over the two isolated components (`macro`, `micro`) with `fail-fast: false`, so a micro failure never cancels the macro job — both finish and keep their own status and artifacts.

Each job checks out the repo, installs that component's `requirements.txt` with Python 3.10, and runs the exact same command you can run locally:

```powershell
Push-Location micro
& ..\.venvs\micro\Scripts\python.exe -B -m unittest discover -s tests -p 'test_jev*.py'
Pop-Location
```

(Macro analogously, inside `macro/` with its own virtual environment.)

## Reading the check status

On a commit or PR, the **Checks** tab (or the commit's status icon) shows one `regression (micro)` and one `regression (macro)` entry. Green = that component's offline suite passed. A red component keeps its own red state; the other is unaffected.

## Downloading test artifacts

Whether a job passes or fails, it uploads an artifact:

- `offline-tests-micro-<run_id>`
- `offline-tests-macro-<run_id>`

Download from the workflow run page (`Actions` → run → job → **Artifacts**, or the run's summary sidebar). Retention is 30 days. Each artifact contains `<component>/ci-results/`:

- `unittest.log` — the complete unittest output.
- `summary.json` — machine-readable report generated **from the actual run output**.

## summary.json fields

| Field | Meaning |
| --- | --- |
| `component` | `macro` or `micro` |
| `commit_sha` | Git commit the tests ran against |
| `python_version` | Interpreter version |
| `command` | Exact test command |
| `duration_seconds` | Wall time of the test run |
| `total` / `passed` / `failures` / `errors` / `skipped` | Counts parsed from unittest's summary line |
| `status` | `passed` / `failed` (derived from the command's exit code) |
| `failing_tests` | `FAIL:`/`ERROR:` header lines for each non-passing test |
| `run_id` / `source` | GitHub run id and provenance (`github-actions`) |

Logs contain unittest output only — no API keys or secret environment variables are printed.

## Offline tests vs real SC2 validation

These suites are **offline**: mocked observations and synthetic states, no StarCraft II client, no JEV/Astra network calls. Passing them verifies code and algorithm contracts, **not** in-game tactical performance (win rates), which requires separately authorized real-game experiments.

## Local lightweight report

CI does not commit its raw logs back to the repository (that would re-trigger CI and bloat history). If you want an in-repo record, generate one locally after a run and mark it clearly, e.g.:

```json
{"source": "local", "commit_sha": "<git rev-parse HEAD>", "...": "..."}
```
