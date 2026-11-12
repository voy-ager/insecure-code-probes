"""Lexical and length-only baselines (design #1, section 6): controls for surface-token signal.

Both use folds from `icp.dataset` (never their own split) and return `icp.stats.SCORES_COLUMNS`
frames. Input frames need: id, template_id (-> group), cwe (-> stratum), label, code.

Transfer (generated code) frames carry prompt_id and cwe_area instead: the caller maps
prompt_id -> template_id (bootstrap group) and cwe_area -> cwe (stratum) before calling.
The caller writes `scores_lexical_<eval_set>.csv` and `scores_length_<eval_set>.csv`
(eval_set in pairs, transfer, loco_<cwe>) with `icp.stats.write_scores`.
Methods are `lexical` and `length`; the caller appends `_7cwe` etc. to the scores-file method name
where the 7-CWE run must be kept apart (see the `icp.stats` docstring).
`length` counts Qwen tokens (the probe's tokenizer, no special tokens added) and lines, so it is
comparable with the probe's length confound; pass `tok` (see `load_tokenizer`). `lexical` uses stdlib
tokenize and needs no tokenizer.
`seed` is recorded in the output but has no effect: lbfgs logistic regression is deterministic.
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


def load_tokenizer(cfg: dict):
    """Tokenizer only (no weights) at the pinned revision; safetensors-era flags as in icp.activations."""
    from transformers import AutoTokenizer  # lazy: `lexical` alone needs no transformers

    m = cfg["model"]
    return AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], trust_remote_code=False)


def _length_features(codes, tok) -> np.ndarray:
    n_tok = [len(ids) for ids in tok(list(codes), add_special_tokens=False)["input_ids"]]
    return np.array([[n, len(c.splitlines())] for n, c in zip(n_tok, codes)], dtype=float)


def _model(method: str, seed: int):
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    if method == "lexical":
        return make_pipeline(CountVectorizer(analyzer=lambda c: _ngrams(tokens(c))), lr)
    return make_pipeline(StandardScaler(), lr)


def _ngrams(toks: list[str]) -> list[str]:
    return toks + [f"{a} {b}" for a, b in itertools.pairwise(toks)]


METHODS = ("lexical", "length")


def _fit_predict(method: str, train: pd.DataFrame, test: pd.DataFrame, seed: int, tok) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "length" and tok is None:
        raise ValueError("method 'length' needs a Qwen tokenizer (tok=...)")
    if train["label"].nunique() < 2:
        raise ValueError("train fold needs both classes (0 and 1)")
    if not any(tokens(c) for c in train["code"]):
        raise ValueError("train fold has no code tokens (all code empty)")
    prep = (lambda d: d["code"].tolist()) if method == "lexical" else (lambda d: _length_features(d["code"], tok))
    m = _model(method, seed).fit(prep(train), train["label"])
    return m.decision_function(prep(test))


def _frame(test: pd.DataFrame, score: np.ndarray, seed: int) -> pd.DataFrame:
    return pd.DataFrame({
        "id": test["id"].to_numpy(), "group": test["template_id"].to_numpy(),
        "stratum": test["cwe"].to_numpy(), "label": test["label"].to_numpy(),
        "score": score, "seed": seed,
    })[SCORES_COLUMNS]


def fit_score(train: pd.DataFrame, test: pd.DataFrame, method: str, seed: int = 0, tok=None) -> pd.DataFrame:
    """Train on `train`, score `test` (transfer). method: 'lexical' or 'length' (needs `tok`)."""
    return _frame(test, _fit_predict(method, train, test, seed, tok), seed)


def cv_scores(df: pd.DataFrame, folds: list, method: str, seed: int = 0, tok=None) -> pd.DataFrame:
    """Out-of-fold scores; `folds` from `icp.dataset.grouped_folds` as (train, test) positions."""
    if not folds:
        raise ValueError("no folds given")
    parts = [fit_score(df.iloc[tr], df.iloc[te], method, seed, tok) for tr, te in folds]
    return pd.concat(parts, ignore_index=True)
