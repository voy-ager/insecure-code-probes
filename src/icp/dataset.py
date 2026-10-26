"""Pairs dataset: schema validation and leakage-free splits (design #1, section 3)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED = [
    "id", "pair_id", "template_id", "cwe", "label", "code", "variation", "hard", "expected_bandit",
]
CWES = {"CWE-78", "CWE-89", "CWE-94", "CWE-502", "CWE-327", "CWE-798", "CWE-295", "CWE-22"}
HARD = {"mention_in_comment", "hidden_token"}  # or null


def load_pairs(path: str | Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError naming the first broken rule and the offending ids."""

    def fail(rule: str, bad: pd.Series) -> None:
        ids = sorted(map(str, bad.unique()))[:5]
        raise ValueError(f"{rule}: {ids}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    notnull = [c for c in REQUIRED if c != "hard"]
    rows = df[notnull].isna().any(axis=1)
    if rows.any():
        fail("null or missing required field", df.loc[rows, "id"])
    if df["id"].duplicated().any():
        fail("duplicate id", df.loc[df["id"].duplicated(), "id"])
    bad = ~df["label"].isin([0, 1])
    if bad.any():
        fail("label not in {0, 1}", df.loc[bad, "id"])
    bad = ~df["cwe"].isin(CWES)
    if bad.any():
        fail("cwe not in canonical set", df.loc[bad, "id"])
    bad = ~(df["hard"].isna() | df["hard"].isin(HARD))
    if bad.any():
        fail(f"hard not in {sorted(HARD)} or null", df.loc[bad, "id"])
    if df["code"].duplicated().any():
        fail("duplicate code", df.loc[df["code"].duplicated(), "id"])
    g = df.groupby("pair_id")
    bad = (g["label"].count() != 2) | (g["label"].sum() != 1)
    if bad.any():
        fail("pair_id needs exactly one secure (0) and one insecure (1) row", bad[bad].index.to_series())
    bad = (g["template_id"].nunique() != 1) | (g["cwe"].nunique() != 1)
    if bad.any():
        fail("pair members must share template_id and cwe", bad[bad].index.to_series())


def grouped_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (train, test) indices; groups = template_id, stratified by cwe.

    Positions are relative to `df`, so inner folds work on `df.iloc[train]`.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(df, df["cwe"], groups=df["template_id"]))


def loco_splits(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-CWE-out: (held-out cwe, train positions, test positions)."""
    held = df["cwe"].to_numpy()
    return [(c, np.flatnonzero(held != c), np.flatnonzero(held == c)) for c in sorted(set(held))]


def shuffle_within_pairs(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Copy of df with the labels swapped in a random half of pair_ids (null control)."""
    pairs = np.array(sorted(df["pair_id"].unique()))  # sorted: independent of row order
    swap = np.random.default_rng(seed).choice(pairs, size=len(pairs) // 2, replace=False)
    out = df.copy()
    rows = out["pair_id"].isin(swap)
    out.loc[rows, "label"] = 1 - out.loc[rows, "label"]
    return out
