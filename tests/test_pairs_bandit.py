"""Checks on data/pairs/pairs.jsonl (design #1, section 3). Snippets are parsed, never executed.

Needs icp.dataset (#5) and icp.bandit_label (#4).
"""

import ast
import importlib.util
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import pytest

from icp import bandit_label
from icp.dataset import CWES, load_pairs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_pairs", ROOT / "scripts/build_pairs.py")
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)

BANDIT_CWES = CWES - {"CWE-22"}
PATH_CHECKS = {"is_relative_to", "commonpath"}


@pytest.fixture(scope="module")
def df():
    return load_pairs(ROOT / "data/pairs/pairs.jsonl")  # also validates the schema


@pytest.fixture(scope="module")
def pairs(df):
    """[(secure row, insecure row)] in file order."""
    by_pair = defaultdict(dict)
    for r in df.to_dict("records"):
        by_pair[r["pair_id"]][r["label"]] = r
    return [(p[0], p[1]) for p in by_pair.values()]


@pytest.fixture(scope="module")
def findings(df):
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(df["id"], ex.map(bandit_label.scan, df["code"])))


def test_counts(df):
    assert len(df) == 576
    assert df["pair_id"].nunique() == 288
    assert df.groupby("cwe")["pair_id"].nunique().eq(36).all()


def test_bandit_agreement(df, findings):
    bad = []
    for r in df.to_dict("records"):
        ids = {f.test_id for f in findings[r["id"]]}
        mapped = {f.test_id for f in findings[r["id"]] if f.cwe}
        if r["label"] == 1 and r["cwe"] in BANDIT_CWES:
            ok = bool(r["expected_bandit"]) and set(r["expected_bandit"]) <= ids
        else:  # secure members, and both CWE-22 members: no mapped ID at all
            ok = not mapped
        if not ok:
            bad.append(r["id"])
    assert not bad, bad[:10]


def _path_check_calls(code: str) -> int:
    return sum(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in PATH_CHECKS
        for n in ast.walk(ast.parse(code))
    )


def test_cwe22_structural(pairs):
    for sec, ins in pairs:
        if sec["cwe"] == "CWE-22":
            assert _path_check_calls(sec["code"]) >= 1, sec["id"]
            assert _path_check_calls(ins["code"]) == 0, ins["id"]


def test_same_imports_per_pair(pairs):
    def imports(code):
        return sorted(ast.dump(n) for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.Import, ast.ImportFrom)))

    bad = [s["pair_id"] for s, i in pairs if imports(s["code"]) != imports(i["code"])]
    assert not bad, bad[:10]


def test_at_most_4_changed_lines(pairs):
    bad = [s["pair_id"] for s, i in pairs if bp.changed_lines(s["code"], i["code"]) > 4]
    assert not bad, bad[:10]


def test_mean_token_diff_per_cwe(pairs):
    diffs = defaultdict(list)
    for s, i in pairs:
        diffs[s["cwe"]].append(bp.count_tokens(s["code"]) - bp.count_tokens(i["code"]))
    means = {c: mean(d) for c, d in diffs.items()}
    print("mean token diff (secure - insecure):", {c: round(m, 2) for c, m in sorted(means.items())})
    assert all(abs(m) <= 3 for m in means.values()), means


def test_helper_line_frequency_balanced(df):
    pool = json.loads((bp.TEMPLATES / "_helpers.json").read_text(encoding="utf-8"))["helpers"]
    lines = {ln for entry in pool for ln in entry}
    has = df["code"].map(lambda c: any(ln in lines for ln in c.splitlines()))
    freq = has.groupby(df["label"]).mean()
    assert abs(freq[1] - freq[0]) <= 0.10, freq.to_dict()


def test_rebuild_is_byte_identical(tmp_path):
    out = tmp_path / "pairs.jsonl"
    bp.write(bp.build(), out)
    assert out.read_bytes() == (ROOT / "data/pairs/pairs.jsonl").read_bytes()
