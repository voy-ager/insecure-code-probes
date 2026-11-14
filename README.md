# insecure-code-probes

Can a linear probe on a code model's activations tell when the model is writing insecure code, and does a probe trained on synthetic examples still work on the model's own generations?

Status: pilot complete. Models: Qwen2.5-Coder-0.5B (dev), Qwen2.5-Coder-1.5B (main run). Reference checker: Bandit.

## Results

On Qwen2.5-Coder-1.5B-Instruct, the probe trained on hand-written secure/insecure pairs scores the model's own Bandit-labeled generations at AUROC 0.780 [0.665, 0.873]. That is the same level as a bag-of-tokens lexical baseline (0.759) and a prior that only knows which prompt was asked (0.798); the intervals overlap. So we have no evidence that the probe reads more than prompt topic or surface tokens, and we report this as a negative / inconclusive result. Full write-up: [docs/pilot_report.md](docs/pilot_report.md). Table: [results/qwen2.5-coder-1.5b-instruct/summary.md](results/qwen2.5-coder-1.5b-instruct/summary.md). Plot: [auroc_vs_layer.png](results/qwen2.5-coder-1.5b-instruct/auroc_vs_layer.png).

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
. .venv/bin/activate           # Linux / macOS
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # Linux, CPU only: skips the CUDA build
pip install -r requirements.txt -r requirements-model.txt
pip install -e ".[dev]"        # the icp package plus ruff, pytest, pip-audit
pytest                         # fast tests
```

Reproduce the 1.5B main run (about 1.8 h on CPU, about 8 GB RAM):

```
python scripts/run_all.py --config configs/main_1.5b.toml
```

`pytest -m slow` runs the real-model tests; for now it runs only a placeholder.

## Contributing

- `main` is protected. All work goes through a feature branch (`feat/<issue#>-<slug>`) and a pull request.
- Every PR links its issue (`Closes #N`) and must pass CI (lint, tests, Bandit, pip-audit).
- Only the maintainer merges.
