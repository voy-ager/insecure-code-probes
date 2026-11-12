from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icp import dataset, lexical, load_config, stats

ROOT = Path(__file__).resolve().parents[1]


class StubTok:
    """Whitespace tokenizer with the HF call shape; no downloads."""

    def __call__(self, codes, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [list(range(len(c.split()))) for c in codes]}


def toy(codes, labels):
    n = len(codes)
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(n)], "template_id": [f"t{i // 2}" for i in range(n)],
        "cwe": ["CWE-78"] * (n // 2) + ["CWE-89"] * (n - n // 2), "label": labels, "code": codes,
    })


def run(df, method, tmp_path):
    folds = dataset.grouped_folds(df, 4, 0)
    out = lexical.cv_scores(df, folds, method, tok=StubTok())
    assert sorted(out["id"]) == sorted(df["id"])  # every row scored exactly once, out of fold
    assert list(out.columns) == stats.SCORES_COLUMNS
    stats.write_scores(out, tmp_path / "s.csv")  # unique ids, finite scores, exact schema
    return stats.auroc(out["label"], out["score"])


def test_lexical_one_token_determines_label(tmp_path):
    labels = [i % 2 for i in range(80)]
    codes = [f"x = {'shell' if y else 'safe'}({i})  # {i}" for i, y in enumerate(labels)]
    assert run(toy(codes, labels), "lexical", tmp_path) == 1.0


def test_lexical_random_tokens_near_chance(tmp_path):
    rng = np.random.default_rng(0)
    labels = [i % 2 for i in range(200)]
    codes = [" ".join(f"w{j}" for j in rng.integers(0, 50, 8)) for _ in labels]
    assert abs(run(toy(codes, labels), "lexical", tmp_path) - 0.5) < 0.15


def test_length_longer_than_median(tmp_path):
    rng = np.random.default_rng(0)
    lengths = rng.permutation(np.r_[np.arange(1, 41), np.arange(61, 101)])  # margin: folds differ
    labels = [int(n > 50) for n in lengths]
    codes = ["\n".join(f"a{j} = {j}" for j in range(n)) for n in lengths]
    assert run(toy(codes, labels), "length", tmp_path) == 1.0


def test_untokenizable_code_falls_back_to_split():
    assert lexical.tokens("x = (1\n") == ["x", "=", "(1"]
    assert "# c" in lexical.tokens("x = 1  # c")  # comments are kept


def test_fit_score_transfer_schema():
    df = toy(["a b", "c d", "a b", "c d"], [1, 0, 1, 0])
    out = lexical.fit_score(df, df.iloc[:2], "lexical")
    assert out["group"].tolist() == ["t0", "t0"] and out["stratum"].tolist() == ["CWE-78"] * 2


def test_clear_errors():
    df = toy(["a b", "c d", "a b", "c d"], [1, 0, 1, 0])
    with pytest.raises(ValueError, match="unknown method"):
        lexical.fit_score(df, df, "lexcial")
    with pytest.raises(ValueError, match="needs a Qwen tokenizer"):
        lexical.fit_score(df, df, "length")
    with pytest.raises(ValueError, match="no folds"):
        lexical.cv_scores(df, [], "lexical")
    with pytest.raises(ValueError, match="both classes"):
        lexical.fit_score(df[df["label"] == 1], df, "length", tok=StubTok())
    empty = toy(["", " ", "\n", ""], [1, 0, 1, 0])
    with pytest.raises(ValueError, match="no code tokens"):
        lexical.fit_score(empty, df, "lexical")


def test_length_counts_tokenizer_tokens_not_stdlib_tokens():
    # same stdlib tokens and lines, different whitespace-token counts -> features follow the tokenizer
    f = lexical._length_features(["a+b", "a + b"], StubTok())
    assert f.tolist() == [[1.0, 1.0], [3.0, 1.0]]


@pytest.mark.slow
def test_real_tokenizer_length_separates_length_only_pair():
    tok = lexical.load_tokenizer(load_config(ROOT / "configs" / "pilot_0.5b.toml"))
    short = [f"x{i} = {i}\n" for i in range(40)]
    longer = [f"x{i} = {i}\ny{i} = {i} + {i}\n" for i in range(40)]  # same shape, only longer
    df = toy(short + longer, [0] * 40 + [1] * 40)
    df["template_id"] = [f"t{i % 40}" for i in range(80)]
    out = lexical.cv_scores(df, dataset.grouped_folds(df, 4, 0), "length", tok=tok)
    assert stats.auroc(out["label"], out["score"]) > 0.95
