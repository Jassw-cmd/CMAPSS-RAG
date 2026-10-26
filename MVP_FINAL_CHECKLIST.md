# CMAPSS RAG — Final MVP Checklist

## Confirmed from reported local runs

- [x] Router logic tests passed: route classification, parameter extraction, and hybrid query extraction.
- [x] Smoke test passed: 6 passed, 0 failed.
- [x] Document route tested with RUL and wear questions.
- [x] Unsupported turbine-blade material question tested for abstention.
- [x] Structured Engine 24 RUL at cycle 100 returned 47 cycles.
- [x] Sensor 7 of Engine 24 at cycle 100 returned 553.51.
- [x] Hybrid damage-propagation + RUL question returned both document and structured results.
- [x] Baseline vs Multi-query results recorded; Baseline retained as the default.
- [x] Evidence-label evaluation ran and created JSON/CSV reports.

## Remaining before sharing as a stable MVP

- [ ] Install from `requirements.txt` in a clean Python 3.10+ environment and record actual package versions.
- [ ] Confirm Ollama is installed, running, and has `qwen2.5:3b` available.
- [ ] Run all tests from a new PowerShell session at the project root.
- [ ] Manually review provisional evidence labels against the source document.
- [ ] Manually audit generated claims against their cited evidence; citation syntax validation alone does not prove support.
- [ ] Expand the answerable/unanswerable evaluation set before making quality claims.
- [ ] Confirm no API keys, `.env`, virtual environment, or private/raw data are committed.
- [ ] Archive legacy experimental scripts only after confirming they are not imported or needed for reproduction.

## Clean-environment setup (PowerShell)

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OLLAMA_MODEL = "qwen2.5:3b"
.\.venv\Scripts\python.exe src\test_router_logic.py
.\.venv\Scripts\python.exe src\smoke_test.py
```

The Python dependencies do not install the Ollama application or download the Ollama model. Those are separate local prerequisites.

## Notes

- The requirements file intentionally does not pin versions because the exact working package versions were not captured in the shared console logs. Freeze known-good versions from the actual working environment before distributing a reproducible build.
- `requirements-data-prep.txt` is optional. Use it only if the project's preprocessing scripts require `pypdf` or `pymupdf4llm`.
- The current six-case smoke test is a representative check, not a complete validation of RAG accuracy.
