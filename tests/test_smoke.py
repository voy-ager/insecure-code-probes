from pathlib import Path

import pytest

from icp import load_config

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "": ["seed"],
    "model": ["id", "revision", "dtype"],
    "paths": ["templates", "pairs", "prompts", "generations", "activations", "results"],
    "extract": ["max_len", "batch_size", "exclude_first_token", "pooling", "store_dtype"],
    "probe": [
        "outer_folds", "inner_folds", "select_layer_range", "transfer_train_cwes",
        "C", "max_iter", "n_boot",
    ],
    "generate": ["samples_per_prompt", "temperature", "top_p", "max_new_tokens"],
    "judge": ["system_prompt", "prompt"],
}
BANDIT_CWES = {"CWE-78", "CWE-89", "CWE-94", "CWE-502", "CWE-327", "CWE-798", "CWE-295"}


@pytest.mark.parametrize("name", ["pilot_0.5b.toml", "main_1.5b.toml"])
def test_config_has_all_keys(name):
    cfg = load_config(ROOT / "configs" / name)
    for section, keys in REQUIRED.items():
        table = cfg[section] if section else cfg
        missing = [k for k in keys if k not in table]
        assert not missing, f"{name} [{section}] missing {missing}"
    assert cfg["model"]["id"].endswith("-Instruct")
    assert set(cfg["probe"]["transfer_train_cwes"]) == BANDIT_CWES
    assert cfg["extract"]["exclude_first_token"] is True


def test_no_init_under_data():
    assert not list((ROOT / "data").rglob("__init__.py"))


@pytest.mark.slow
def test_slow_marker_is_opt_in():
    """Placeholder: real-model tests use this marker and run only with `pytest -m slow`."""
