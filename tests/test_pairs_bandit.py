"""Checks on data/pairs/pairs.jsonl (design #1, section 3). Snippets are parsed, never executed.

Bandit mapping: icp.bandit_label.CWE_MAP.
"""

import ast
import io
import itertools
import json
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pytest

from icp import pairs as bp
from icp.bandit_label import CWE_MAP
from icp.dataset import CWES, load_pairs

ROOT = Path(__file__).resolve().parents[1]
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
def findings(df, tmp_path_factory):
    """{snippet id: set of Bandit test IDs}, from one `bandit -r` over a temp dir (parsed, never run)."""
    d = tmp_path_factory.mktemp("snippets")
    for r in df.to_dict("records"):
        (d / f"{r['id']}.py").write_text(r["code"], encoding="utf-8", newline="\n")
    proc = subprocess.run(
        [sys.executable, "-m", "bandit", "-r", str(d), "--ignore-nosec", "-f", "json", "-q"],
        capture_output=True, encoding="utf-8", check=False,
    )
    assert proc.returncode in (0, 1), proc.stderr
    out = {i: set() for i in df["id"]}
    for res in json.loads(proc.stdout)["results"]:
        out[Path(res["filename"]).stem].add(res["test_id"])
    return out


def test_counts(df):
    assert len(df) == 576
    assert df["pair_id"].nunique() == 288
    assert df.groupby("cwe")["pair_id"].nunique().eq(36).all()


def test_bandit_agreement(df, findings):
    bad = []
    for r in df.to_dict("records"):
        ids = findings[r["id"]]
        mapped = {i for i in ids if i in CWE_MAP}
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


def _comments(code: str) -> int:
    return sum(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline))


def test_same_comment_count_per_pair(pairs):
    bad = [s["pair_id"] for s, i in pairs if _comments(s["code"]) != _comments(i["code"])]
    assert not bad, bad[:10]


def test_max_token_diff_per_pair(pairs):
    bad = [s["pair_id"] for s, i in pairs if abs(bp.count_tokens(s["code"]) - bp.count_tokens(i["code"])) > 3]
    assert not bad, bad[:10]


def test_each_pool_line_balanced_between_classes(df):
    pool = json.loads((bp.TEMPLATES / "_helpers.json").read_text(encoding="utf-8"))["helpers"]
    for entry in pool:
        has = df["code"].map(lambda c, first=entry[0]: first in c.splitlines())
        freq = has.groupby(df["label"]).mean()
        assert abs(freq[1] - freq[0]) <= 0.10, (entry[0], freq.to_dict())


def test_hard_templates_share_and_kinds(df):
    hard = df.groupby("template_id")["hard"].apply(lambda h: h.notna().any())
    assert hard.mean() >= 1 / 3
    assert set(df["hard"].dropna()) == {"mention_in_comment", "hidden_token"}
    assert set(df.loc[df["label"] == 0, "hard"].dropna()) == {"mention_in_comment"}
    assert set(df.loc[df["label"] == 1, "hard"].dropna()) == {"hidden_token"}


def test_template_token_jaccard_below_0_7(df):
    def tokens(code):
        return {t.string for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type in (tokenize.NAME, tokenize.STRING, tokenize.OP)}

    first = df[(df["label"] == 1) & df["pair_id"].str.endswith("-v0")]
    by_cwe = defaultdict(list)
    for r in first.to_dict("records"):
        by_cwe[r["cwe"]].append(tokens(r["code"]))
    worst = max(len(a & b) / len(a | b) for ts in by_cwe.values() for a, b in itertools.combinations(ts, 2))
    assert worst < 0.7, worst


def test_rebuild_is_byte_identical(tmp_path):
    out = tmp_path / "pairs.jsonl"
    bp.write(bp.build(), out)
    assert out.read_bytes() == (ROOT / "data/pairs/pairs.jsonl").read_bytes()
