import numpy as np
import pandas as pd

from icp import dataset, lexical, stats


def toy(codes, labels):
    n = len(codes)
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(n)], "template_id": [f"t{i // 2}" for i in range(n)],
        "cwe": ["CWE-78"] * (n // 2) + ["CWE-89"] * (n - n // 2), "label": labels, "code": codes,
    })


def run(df, method):
    folds = dataset.grouped_folds(df, 4, 0)
    out = lexical.cv_scores(df, folds, method)
    assert sorted(out["id"]) == sorted(df["id"])  # every row scored exactly once, out of fold
    assert list(out.columns) == stats.SCORES_COLUMNS
    return stats.auroc(out["label"], out["score"])


def test_lexical_one_token_determines_label():
    labels = [i % 2 for i in range(80)]
    codes = [f"x = {'shell' if y else 'safe'}({i})  # {i}" for i, y in enumerate(labels)]
    assert run(toy(codes, labels), "lexical") == 1.0


def test_lexical_random_tokens_near_chance():
    rng = np.random.default_rng(0)
    labels = [i % 2 for i in range(200)]
    codes = [" ".join(f"w{j}" for j in rng.integers(0, 50, 8)) for _ in labels]
    assert abs(run(toy(codes, labels), "lexical") - 0.5) < 0.15


def test_length_longer_than_median():
    rng = np.random.default_rng(0)
    lengths = rng.permutation(np.r_[np.arange(1, 41), np.arange(61, 101)])  # margin: folds differ
    labels = [int(n > 50) for n in lengths]
    codes = ["\n".join(f"a{j} = {j}" for j in range(n)) for n in lengths]
    assert run(toy(codes, labels), "length") == 1.0


def test_untokenizable_code_falls_back_to_split():
    assert lexical.tokens("x = (1\n") == ["x", "=", "(1"]
    assert "# c" in lexical.tokens("x = 1  # c")  # comments are kept


def test_fit_score_transfer_schema():
    df = toy(["a b", "c d", "a b", "c d"], [1, 0, 1, 0])
    out = lexical.fit_score(df, df.iloc[:2], "lexical")
    assert out["group"].tolist() == ["t0", "t0"] and out["stratum"].tolist() == ["CWE-78"] * 2
