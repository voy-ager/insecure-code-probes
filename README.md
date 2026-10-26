# insecure-code-probes

Can a linear probe on a code model's activations tell when the model is writing insecure code, and does a probe trained on synthetic examples still work on the model's own generations?

Status: pilot in progress. Models: Qwen2.5-Coder-0.5B (dev), Qwen2.5-Coder-1.5B (main run). Reference checker: Bandit.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
. .venv/bin/activate           # Linux / macOS
pip install -r requirements.txt -r requirements-model.txt
pip install -e ".[dev]"        # the icp package plus ruff, pytest, pip-audit
pytest                         # fast tests
```

Linux, CPU only: install torch from the CPU index first, `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`, to skip the CUDA build.
`pytest -m slow` runs the real-model tests; for now it runs only a placeholder.

## Contributing

- `main` is protected. All work goes through a feature branch (`feat/<issue#>-<slug>`) and a pull request.
- Every PR links its issue (`Closes #N`) and must pass CI (lint, tests, Bandit, pip-audit).
- Only the maintainer merges.
