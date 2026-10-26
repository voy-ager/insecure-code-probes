# insecure-code-probes

Can a linear probe on a code model's activations tell when the model is writing insecure code, and does a probe trained on synthetic examples still work on the model's own generations?

Status: pilot in progress. Models: Qwen2.5-Coder-0.5B (dev), Qwen2.5-Coder-1.5B (main run). Reference checker: Bandit.

## Setup

```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-model.txt
pytest            # fast tests; `pytest -m slow` runs the real-model tests
```

## Contributing

- `main` is protected. All work goes through a feature branch (`feat/<issue#>-<slug>`) and a pull request.
- Every PR links its issue (`Closes #N`) and must pass CI (lint, tests, Bandit, pip-audit).
- Only the maintainer merges.
