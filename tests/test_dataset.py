from pathlib import Path

import numpy as np
import pytest

from icp.dataset import (
    grouped_folds,
    load_pairs,
    loco_splits,
    shuffle_within_pairs,
    validate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pairs_tiny.jsonl"


@pytest.fixture
def df():
    return load_pairs(FIXTURE)


def _break_missing_field(d):
    return d.drop(columns="variation")


def _break_null_field(d):
    d.loc[0, "code"] = None
    return d


def _break_label(d):
    d.loc[0, "label"] = 2
    return d


def _break_duplicate_id(d):
    d.loc[1, "id"] = d.loc[0, "id"]
    return d


def _break_pair_two_insecure(d):
    d.loc[0, "label"] = 1
    return d


def _break_pair_three_rows(d):
    d.loc[2, "pair_id"] = d.loc[0, "pair_id"]
    return d


def _break_duplicate_code(d):
    d.loc[1, "code"] = d.loc[0, "code"]
    return d


def _break_cwe(d):
    d.loc[0, "cwe"] = "CWE-328"  # non-canonical: 327 covers 327/328
    return d


def _break_hard(d):
    d.loc[0, "hard"] = "obvious"
    return d


def _break_pair_template(d):
    d.loc[0, "template_id"] = "other-t99"
    return d


def _break_label_bool(d):
    d["label"] = d["label"].astype(object)
    d.loc[1, "label"] = True
    return d


def _break_label_float(d):
    d["label"] = d["label"].astype(float)
    return d


def _break_expected_bandit(d):
    d["expected_bandit"] = d["expected_bandit"].astype(object)
    d.at[0, "expected_bandit"] = "B101"
    return d


def _break_template_two_cwes(d):
    # relabel a whole pair so the pair rule passes; the template now spans two CWEs
    d.loc[d["pair_id"] == d.loc[0, "pair_id"], "cwe"] = "CWE-78"
    return d


def _break_empty(d):
    return d.iloc[0:0]


@pytest.mark.parametrize("breaker, message", [
    (_break_label_bool, "label not int"),
    (_break_label_float, "label not int"),
    (_break_expected_bandit, "expected_bandit not a list"),
    (_break_template_two_cwes, "exactly one cwe"),
    (_break_empty, "no rows"),
    (_break_missing_field, "missing fields"),
    (_break_null_field, "null or missing required field"),
    (_break_label, "label not int"),
    (_break_duplicate_id, "duplicate id"),
    (_break_pair_two_insecure, "exactly one secure"),
    (_break_pair_three_rows, "exactly one secure"),
    (_break_duplicate_code, "duplicate code"),
    (_break_cwe, "cwe not in canonical set"),
    (_break_hard, "hard not in"),
    (_break_pair_template, "share template_id"),
])
def test_validation_rejects_bad_rows(df, breaker, message):
    with pytest.raises(ValueError, match=message):
        validate(breaker(df.copy()))


def _assert_disjoint(d, train, test):
    for col in ("template_id", "pair_id"):
        assert not set(d[col].iloc[train]) & set(d[col].iloc[test]), col


@pytest.mark.parametrize("seed", range(5))
def test_grouped_folds_leak_free_and_cover_every_cwe(df, seed):
    folds = grouped_folds(df, n_splits=3, seed=seed)
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(len(df)))
    for train, test in folds:
        _assert_disjoint(df, train, test)
        assert set(df["cwe"].iloc[test]) == set(df["cwe"])
        inner = df.iloc[train]
        for itrain, itest in grouped_folds(inner, n_splits=2, seed=seed):
            _assert_disjoint(inner, itrain, itest)


def test_loco_splits(df):
    splits = loco_splits(df)
    assert [c for c, _, _ in splits] == sorted(df["cwe"].unique())
    for cwe, train, test in splits:
        _assert_disjoint(df, train, test)
        assert set(df["cwe"].iloc[test]) == {cwe}
        assert cwe not in set(df["cwe"].iloc[train])


def test_shuffle_within_pairs(df):
    out = shuffle_within_pairs(df, seed=0)
    assert (out.groupby("pair_id")["label"].agg(sorted).map(tuple) == (0, 1)).all()
    swapped = out.loc[out["label"] != df["label"], "pair_id"].nunique()
    assert swapped == df["pair_id"].nunique() // 2
    assert out["label"].equals(shuffle_within_pairs(df, seed=0)["label"])
    assert out["label"].equals(shuffle_within_pairs(df.iloc[::-1], seed=0)["label"].sort_index())
    assert not out["label"].equals(df["label"])
    assert df["label"].equals(load_pairs(FIXTURE)["label"])  # input not mutated


@pytest.mark.parametrize("content, message", [
    (b'{"id": "a"}\nnot json\n', r"bad\.jsonl:2: invalid JSON"),
    (b'\n[1, 2]\n', r"bad\.jsonl:2: line is not a JSON object"),
    (b'{"id": "\xff"}\n', r"bad\.jsonl:1: invalid JSON or UTF-8"),
    (b"\n\n", "no rows"),
])
def test_load_pairs_errors_name_file_and_line(tmp_path, content, message):
    p = tmp_path / "bad.jsonl"
    p.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        load_pairs(p)
