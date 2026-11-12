"""Linear probes on pooled activations (design #1, section 5).

`X` is `[N, n_layers + 1, d]` (index 0 = embeddings), rows aligned with the frame `df`.
Every probe is StandardScaler + L2 LogisticRegression, fit on training rows only.
"""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from icp.dataset import grouped_folds
from icp.stats import auroc


def _fit(X: np.ndarray, y: np.ndarray, cfg: dict, seed: int):
    p = cfg["probe"]
    lr = LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=seed)
    return make_pipeline(StandardScaler(), lr).fit(X.astype(np.float32), y)


def fit_score(X_train, y_train, X_test, cfg: dict, seed: int) -> np.ndarray:
    """Probability of label 1 (insecure) for X_test, probe fit on (X_train, y_train) only."""
    return _fit(X_train, y_train, cfg, seed).predict_proba(X_test.astype(np.float32))[:, 1]


def layer_range(n_hidden: int, frac) -> list[int]:
    """Hidden-states indices within `frac` (fractions of n_layers, inclusive); n_hidden = n_layers + 1."""
    n_layers = n_hidden - 1
    return list(range(math.ceil(frac[0] * n_layers), math.floor(frac[1] * n_layers) + 1))


def oof_scores(X: np.ndarray, y: np.ndarray, folds, cfg: dict, seed: int) -> np.ndarray:
    """Out-of-fold scores for one layer's features `X` [N, d]."""
    out = np.full(len(y), np.nan)
    for tr, te in folds:
        out[te] = fit_score(X[tr], y[tr], X[te], cfg, seed)
    return out


def layer_curve(X: np.ndarray, df: pd.DataFrame, folds, cfg: dict, seed: int) -> np.ndarray:
    """OOF scores at every layer: [n_layers + 1, N] (descriptive per-layer curve)."""
    y = df["label"].to_numpy()
    return np.stack([oof_scores(X[:, layer], y, folds, cfg, seed) for layer in range(X.shape[1])])


def select_layer(X: np.ndarray, df: pd.DataFrame, cfg: dict, seed: int, n_splits: int) -> int:
    """Best in-range layer by grouped CV AUROC on these rows only (ties -> lowest layer)."""
    y = df["label"].to_numpy()
    folds = grouped_folds(df, n_splits, seed)
    layers = layer_range(X.shape[1], cfg["probe"]["select_layer_range"])
    aucs = [auroc(y, oof_scores(X[:, layer], y, folds, cfg, seed)) for layer in layers]
    return layers[int(np.argmax(aucs))]


def nested_scores(X: np.ndarray, df: pd.DataFrame, cfg: dict, seed: int) -> tuple[np.ndarray, list[int]]:
    """Nested OOF scores (H1): per outer fold, inner CV on the training templates picks the layer,
    then the outer test fold is scored at that layer. Returns (scores, selected layer per fold)."""
    p = cfg["probe"]
    y = df["label"].to_numpy()
    out, chosen = np.full(len(df), np.nan), []
    for tr, te in grouped_folds(df, p["outer_folds"], seed):
        layer = select_layer(X[tr], df.iloc[tr], cfg, seed, p["inner_folds"])
        out[te] = fit_score(X[tr, layer], y[tr], X[te, layer], cfg, seed)
        chosen.append(layer)
    return out, chosen


def transfer_scores(X: np.ndarray, df: pd.DataFrame, X_gen: np.ndarray, cfg: dict, seed: int) -> tuple[np.ndarray, int]:
    """Layer chosen by full in-dist CV on (X, df) only, then fit on all of it and score X_gen.
    The caller restricts df to the training CWEs (headline: `transfer_train_cwes`)."""
    layer = select_layer(X, df, cfg, seed, cfg["probe"]["outer_folds"])
    return fit_score(X[:, layer], df["label"].to_numpy(), X_gen[:, layer], cfg, seed), layer


def train_rows(df: pd.DataFrame, cwes) -> np.ndarray:
    """Positions of rows whose cwe is in `cwes`."""
    return np.flatnonzero(df["cwe"].isin(list(cwes)).to_numpy())


def topic_prior(gen: pd.DataFrame, by: str = "cwe_area") -> np.ndarray:
    """Transfer baseline: positive rate of the row's `by` value (cwe_area, or `natural` for
    prompt_risky_prior) over the OTHER prompts (leave-one-prompt-out); global rate of the
    other prompts if no other prompt shares the value."""
    out = np.empty(len(gen))
    labels = gen["label"].to_numpy()
    for i, (key, prompt) in enumerate(zip(gen[by], gen["prompt_id"])):
        others = (gen["prompt_id"] != prompt).to_numpy()
        same = others & (gen[by] == key).to_numpy()
        out[i] = labels[same].mean() if same.any() else labels[others].mean()
    return out
