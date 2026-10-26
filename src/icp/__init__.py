"""Insecure-code probes."""

import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
