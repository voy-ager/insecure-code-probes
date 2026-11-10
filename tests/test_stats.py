import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from icp import stats

N_BOOT = 500


def _data(n_groups=40, per_group=6, signal=0.0, seed=0):
    """Groups with both classes; score = signal * label + noise."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    labels = np.tile([0, 1], n_groups * per_group // 2)
    scores = signal * labels + rng.normal(size=len(labels))
    return labels, scores, groups


def test_auroc_auprc_known_answers():
    labels, scores = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert stats.auroc(labels, scores) == pytest.approx(0.75)
    # ranked: 0.8 (pos, P=1/1), 0.4 (neg), 0.35 (pos, P=2/3) -> (1 + 2/3) / 2
    assert stats.auprc(labels, scores) == pytest.approx(5 / 6)


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    scores = rng.integers(0, 10, 300)  # many ties
    assert stats.auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
    assert stats.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_perfect_scores_give_auroc_one():
    labels, _, groups = _data()
    ci = stats.auroc_ci(labels, labels.astype(float), groups, N_BOOT, seed=0)
    assert (ci.estimate, ci.lo, ci.hi) == (1.0, 1.0, 1.0)


def test_random_scores_ci_covers_half():
    labels, scores, groups = _data(signal=0.0)
    ci = stats.auroc_ci(labels, scores, groups, N_BOOT, seed=0)
    assert ci.lo < 0.5 < ci.hi
    assert ci.lo <= ci.estimate <= ci.hi


def test_fixed_seed_is_deterministic():
    labels, scores, groups = _data(signal=0.5)
    a = stats.auroc_ci(labels, scores, groups, N_BOOT, seed=7)
    assert a == stats.auroc_ci(labels, scores, groups, N_BOOT, seed=7)
    assert a != stats.auroc_ci(labels, scores, groups, N_BOOT, seed=8)


def test_cluster_bootstrap_is_wider_than_row_bootstrap_for_clustered_scores():
    # single-class groups whose rows share one group-level score: ~20 independent units, not 400
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 20)
    labels = groups % 2
    scores = rng.normal(size=20)[groups] + 0.5 * labels + 0.1 * rng.normal(size=len(labels))
    clustered = stats.auroc_ci(labels, scores, groups, N_BOOT, seed=0)
    per_row = stats.auroc_ci(labels, scores, np.arange(len(labels)), N_BOOT, seed=0)
    assert clustered.hi - clustered.lo > per_row.hi - per_row.lo


def test_paired_diff_with_itself_is_exactly_zero():
    labels, scores, groups = _data(signal=0.5)
    ci = stats.paired_diff_ci(scores, scores, labels, groups, N_BOOT, seed=0)
    assert (ci.estimate, ci.lo, ci.hi) == (0.0, 0.0, 0.0)


def test_paired_diff_clearly_better_method():
    labels, noise, groups = _data(signal=0.0)
    good = 2.0 * labels + noise
    ci = stats.paired_diff_ci(good, noise, labels, groups, N_BOOT, seed=0)
    assert ci.lo > 0
    assert ci.estimate == pytest.approx(stats.auroc(labels, good) - stats.auroc(labels, noise))


def test_gap_ci_independent_sets():
    indist = _data(signal=2.0, seed=1)
    transfer = _data(n_groups=28, signal=0.0, seed=2)
    ci = stats.gap_ci(indist, transfer, N_BOOT, seed=0)
    assert ci.estimate == pytest.approx(stats.auroc(*indist[:2]) - stats.auroc(*transfer[:2]))
    assert ci.lo > 0
    assert ci == stats.gap_ci(indist, transfer, N_BOOT, seed=0)


def test_tiny_groups_trigger_redraws():
    # 4 single-row groups: a resample is single-class with probability 1/8
    labels, scores, groups = [0, 1, 0, 1], [0.2, 0.6, 0.4, 0.5], [0, 1, 2, 3]
    assert stats.auroc_ci(labels, scores, groups, N_BOOT, seed=0).n_redraw > 0
    # groups that each hold both classes never need a redraw
    labels, scores, groups = _data()
    assert stats.auroc_ci(labels, scores, groups, N_BOOT, seed=0).n_redraw == 0


def test_single_class_input_rejected():
    with pytest.raises(ValueError):
        stats.auroc_ci([1, 1, 1], [0.1, 0.2, 0.3], [0, 1, 2], N_BOOT, seed=0)


def test_within_group_auroc_skips_single_class_strata():
    labels = [0, 1, 0, 1, 0, 0]
    scores = [0.1, 0.9, 0.9, 0.1, 0.3, 0.4]
    strata = ["a", "a", "b", "b", "c", "c"]  # a: 1.0, b: 0.0, c: one class
    assert stats.within_group_auroc(labels, scores, strata) == (0.5, 2)


def _scores_df():
    return pd.DataFrame({
        "id": ["x-1", "x-2"], "group": ["007", "008"], "stratum": ["CWE-89", "CWE-22"],
        "label": [0, 1], "score": [0.25, 0.75],
    })


def test_scores_round_trip(tmp_path):
    path = tmp_path / "scores.csv"
    df = _scores_df()
    stats.write_scores(df[["score", "label", "stratum", "group", "id"]], path)
    pd.testing.assert_frame_equal(stats.read_scores(path), df)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_write_scores_rejects_wrong_columns(tmp_path, change):
    df = _scores_df()
    df = df.drop(columns="stratum") if change == "missing" else df.assign(layer=3)
    with pytest.raises(ValueError):
        stats.write_scores(df, tmp_path / "scores.csv")
