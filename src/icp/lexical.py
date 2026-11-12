"""Lexical and length-only baselines (design #1, section 6): controls for surface-token signal.

Both use folds from `icp.dataset` (never their own split) and return `icp.stats.SCORES_COLUMNS`
frames. Input frames need: id, template_id (-> group), cwe (-> stratum), label, code.
"""

import io
import itertools
import tokenize

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from icp.stats import SCORES_COLUMNS


def tokens(code: str) -> list[str]:
    """Python tokens (comments kept); whitespace split if the code does not tokenize."""
    try:
        return [t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.string.strip()]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code.split()  # shortcut: no partial credit for the tokens before the error


def _length_features(codes) -> np.ndarray:
    return np.array([[len(tokens(c)), len(c.splitlines())] for c in codes], dtype=float)


def _model(method: str, seed: int):
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    if method == "lexical":
        return make_pipeline(CountVectorizer(analyzer=lambda c: _ngrams(tokens(c))), lr)
    return make_pipeline(StandardScaler(), lr)


def _ngrams(toks: list[str]) -> list[str]:
    return toks + [f"{a} {b}" for a, b in itertools.pairwise(toks)]


def _fit_predict(method: str, train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    prep = (lambda d: d["code"].tolist()) if method == "lexical" else (lambda d: _length_features(d["code"]))
    m = _model(method, seed).fit(prep(train), train["label"])
    return m.decision_function(prep(test))


def _frame(test: pd.DataFrame, score: np.ndarray, seed: int) -> pd.DataFrame:
    return pd.DataFrame({
        "id": test["id"].to_numpy(), "group": test["template_id"].to_numpy(),
        "stratum": test["cwe"].to_numpy(), "label": test["label"].to_numpy(),
        "score": score, "seed": seed,
    })[SCORES_COLUMNS]


def fit_score(train: pd.DataFrame, test: pd.DataFrame, method: str, seed: int = 0) -> pd.DataFrame:
    """Train on `train`, score `test` (transfer). method: 'lexical' or 'length'."""
    return _frame(test, _fit_predict(method, train, test, seed), seed)


def cv_scores(df: pd.DataFrame, folds: list, method: str, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold scores; `folds` from `icp.dataset.grouped_folds` as (train, test) positions."""
    parts = [fit_score(df.iloc[tr], df.iloc[te], method, seed) for tr, te in folds]
    return pd.concat(parts, ignore_index=True)
