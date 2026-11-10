"""AUROC/AUPRC, cluster-bootstrap CIs and the shared scores CSV schema (design #1, section 5)."""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SCORES_COLUMNS = ["id", "group", "stratum", "label", "score"]
CI_LEVEL = 0.95
MAX_DRAWS_PER_RESAMPLE = 100  # guards against an endless redraw loop on degenerate data


class CI(NamedTuple):
    estimate: float
    lo: float
    hi: float
    n_redraw: int  # single-class resamples that were redrawn


def write_scores(df: pd.DataFrame, path: str | Path) -> None:
    if sorted(df.columns) != sorted(SCORES_COLUMNS):
        raise ValueError(f"scores columns must be exactly {SCORES_COLUMNS}, got {list(df.columns)}")
    df[SCORES_COLUMNS].to_csv(path, index=False)


def read_scores(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "group": str, "stratum": str})
    if list(df.columns) != SCORES_COLUMNS:
        raise ValueError(f"{path}: columns {list(df.columns)} != {SCORES_COLUMNS}")
    return df


def auroc(labels, scores) -> float:
    """Mann-Whitney AUROC with average ranks for ties; equals sklearn's roc_auc_score but
    ~6x faster, which matters inside bootstrap loops."""
    y = np.asarray(labels) == 1
    _, inverse, counts = np.unique(np.asarray(scores), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(labels, scores) -> float:
    return float(average_precision_score(labels, scores))


def _both_classes(labels: np.ndarray) -> bool:
    return 0 < labels.sum() < len(labels)


class _Clusters:
    """Row indices per group; draws a cluster-bootstrap resample of row indices."""

    def __init__(self, labels, groups):
        self.labels = np.asarray(labels)
        if not _both_classes(self.labels):
            raise ValueError("need both classes (0 and 1) in labels")
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        self.rows = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        picked = rng.integers(0, len(self.rows), len(self.rows))
        return np.concatenate([self.rows[g] for g in picked])


def _bootstrap(
    estimate: float,
    clusters: list[_Clusters],
    stat: Callable[[list[np.ndarray]], float],
    n_boot: int,
    seed: int,
) -> CI:
    """Percentile bootstrap: each clusters set is resampled independently; a resample in
    which any set lacks a class is redrawn and counted."""
    rng = np.random.default_rng(seed)
    values, n_redraw = [], 0
    for _ in range(n_boot):
        for _ in range(MAX_DRAWS_PER_RESAMPLE):
            idx = [c.draw(rng) for c in clusters]
            if all(_both_classes(c.labels[i]) for c, i in zip(clusters, idx)):
                break
            n_redraw += 1
        else:
            raise ValueError(f"{MAX_DRAWS_PER_RESAMPLE} single-class resamples in a row; too few groups")
        values.append(stat(idx))
    tail = (1 - CI_LEVEL) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return CI(estimate, float(lo), float(hi), n_redraw)


def auroc_ci(labels, scores, groups, n_boot: int, seed: int) -> CI:
    """AUROC with a cluster-bootstrap CI (resampling whole groups, e.g. template_id)."""
    c = _Clusters(labels, groups)
    s = np.asarray(scores)
    return _bootstrap(auroc(c.labels, s), [c], lambda i: auroc(c.labels[i[0]], s[i[0]]), n_boot, seed)


def paired_diff_ci(scores_a, scores_b, labels, groups, n_boot: int, seed: int) -> CI:
    """AUROC(a) - AUROC(b) for two methods scored on the same rows; same resample for both."""
    c = _Clusters(labels, groups)
    a, b = np.asarray(scores_a), np.asarray(scores_b)

    def diff(i):
        return auroc(c.labels[i], a[i]) - auroc(c.labels[i], b[i])

    return _bootstrap(diff(np.arange(len(a))), [c], lambda i: diff(i[0]), n_boot, seed)


def gap_ci(set_a, set_b, n_boot: int, seed: int) -> CI:
    """AUROC(set_a) - AUROC(set_b) for two independent sets, each a (labels, scores, groups)
    tuple (e.g. in-dist vs. transfer); each set is resampled by its own groups."""
    ca, cb = _Clusters(set_a[0], set_a[2]), _Clusters(set_b[0], set_b[2])
    sa, sb = np.asarray(set_a[1]), np.asarray(set_b[1])

    def gap(ia, ib):
        return auroc(ca.labels[ia], sa[ia]) - auroc(cb.labels[ib], sb[ib])

    estimate = gap(np.arange(len(sa)), np.arange(len(sb)))
    return _bootstrap(estimate, [ca, cb], lambda i: gap(*i), n_boot, seed)


def within_group_auroc(labels, scores, strata) -> tuple[float, int]:
    """Mean AUROC over strata that contain both classes, and the number of strata used."""
    df = pd.DataFrame({"label": labels, "score": scores, "stratum": strata})
    per = [auroc(g.label, g.score) for _, g in df.groupby("stratum") if g.label.nunique() == 2]
    if not per:
        raise ValueError("no stratum contains both classes")
    return float(np.mean(per)), len(per)
