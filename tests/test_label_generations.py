"""Fixture code is only parsed and Bandit-scanned, never executed. Fake PII is built by concatenation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from icp.label_generations import code_tokens, label_rows, spot_sample, text_free
from icp.pii import pii_flags

FIX = Path(__file__).parent / "fixtures" / "generations_tiny.jsonl"
ROWS = [json.loads(line) for line in FIX.read_text(encoding="utf-8").splitlines()]
LABELS = {"a": 1, "b": 0, "c": None, "d": None, "e": 0, "f": 1}


def test_label_rows_on_fixture():
    out = {r["prompt_id"]: r for r in label_rows(ROWS)}
    assert {k: v["label"] for k, v in out.items()} == LABELS
    assert [f["test_id"] for f in out["a"]["bandit_findings"]] == ["B608"]
    assert out["a"]["bandit_findings"][0]["cwe"] == "CWE-89"
    assert out["b"]["bandit_findings"] == [] and out["d"]["bandit_findings"] == []
    assert all(r["pii_flags"] == [] for r in out.values())


def test_code_tokens():
    assert code_tokens("x\n") == 1 and code_tokens("x = 1\n") == 3
    assert code_tokens("# only a comment\n") == 0 and code_tokens("def f(:\n") == 0 + code_tokens("def f(:\n")


def test_cwe22_area_rejected():
    with pytest.raises(ValueError, match="CWE-22"):
        label_rows([{**ROWS[0], "cwe_area": "CWE-22"}])


def test_text_free_has_no_text_but_keeps_labels():
    t = text_free(label_rows(ROWS))
    assert all({"code", "raw_output", "prompt"}.isdisjoint(r) for r in t)
    assert {"id", "label", "bandit_findings", "pii_flags"} <= set(t[0])


def test_spot_sample_stratified_and_seeded():
    rows = label_rows(ROWS)
    s, sizes = spot_sample(rows, n=1, seed=0)
    strata = {x["stratum"] for x in s}
    assert strata == {"pos_B608", "pos_other", "neg_npv"}
    assert (s, sizes) == spot_sample(rows, n=1, seed=0) and sum(sizes.values()) == 4
    assert [x["id"] for x in s if x["stratum"] == "pos_B608"] == ["a-s0"]


EMAIL = "user" + "@" + "example" + ".com"
IP = "192.0.2.1"  # documentation range
URL = "postgres" + "://" + "bob" + ":" + "hunter2" + "@" + "db.local/x"
PREFIXED = "AK" + "IA" + "X" * 16  # obviously fake, built at test time
ENTROPIC = "aB3dE5gH7jK9mN1pQ2sT4vW6yZ0x"  # mixed, high entropy, 28 chars


@pytest.mark.parametrize("text, flag", [
    (f"contact {EMAIL}", "email"),
    (f"host = '{IP}'", "ipv4"),
    (f"url = '{URL}'", "url_credentials"),
    (f"key = '{PREFIXED}'", "key_prefix"),
    ("-----" + "BEGIN RSA PRIVATE KEY" + "-----", "private_key"),
    (f"t = '{ENTROPIC}'", "high_entropy"),
])
def test_pii_flags_detects(text, flag):
    assert flag in pii_flags(text)


@pytest.mark.parametrize("text", [
    "import os\nx = os.environ['HOME']\n",
    "version = '1.2.3'\nurl = 'https://example.org/path'\n",
    "def a_very_long_but_plain_function_name_here(): pass",
    "ip = '999.1.1.1'",
])
def test_pii_flags_clean(text):
    assert pii_flags(text) == []


def test_pii_row_flagged_in_raw_output_only():
    row = {**ROWS[1], "raw_output": f"mail {EMAIL}"}
    assert label_rows([row])[0]["pii_flags"] == ["email"]


def test_labeled_schema_for_judge_and_probe_joins():
    for g, r in zip(ROWS, label_rows(ROWS), strict=True):
        assert {"id", "prompt_id", "cwe_area", "code", "label"} <= set(r)
        assert (r["id"], r["prompt_id"], r["cwe_area"], r["code"]) == (g["id"], g["prompt_id"], g["cwe_area"], g["code"])
        assert r["label"] in (0, 1, None)


def _run(tmp_path, rows):
    gen, res = tmp_path / "gen", tmp_path / "res"
    gen.mkdir(exist_ok=True)
    (gen / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'seed = 0\n[paths]\ngenerations = "{gen.as_posix()}"\nresults = "{res.as_posix()}"\n', encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "label_generations.py"
    subprocess.run([sys.executable, str(script), "--config", str(cfg)], check=True, capture_output=True,  # nosec B603
                   env={**os.environ, "PYTHONPATH": str(script.parents[1] / "src")})
    return gen, res


def test_script_deterministic_and_split(tmp_path):
    gen, res = _run(tmp_path, ROWS)
    names = ["labels.jsonl", "labeled.jsonl"]
    first = [(gen / n).read_bytes() for n in names] + [(res / n).read_bytes() for n in ("transfer_label_counts.csv", "spotcheck.csv")]
    assert not (gen / "labeled.local.jsonl").exists()
    assert (res / "spotcheck.csv").read_text(encoding="utf-8").splitlines()[0].endswith("manual_verdict")
    _run(tmp_path, ROWS)
    assert first == [(gen / n).read_bytes() for n in names] + [(res / n).read_bytes() for n in ("transfer_label_counts.csv", "spotcheck.csv")]
    # a flagged re-run moves the text to the local file and removes the stale clean one
    _run(tmp_path, [{**ROWS[1], "raw_output": f"mail {EMAIL}"}])
    assert (gen / "labeled.local.jsonl").exists() and not (gen / "labeled.jsonl").exists()
    assert EMAIL not in (gen / "labels.jsonl").read_text(encoding="utf-8")
    # and a clean re-run removes the local one
    _run(tmp_path, ROWS)
    assert (gen / "labeled.jsonl").exists() and not (gen / "labeled.local.jsonl").exists()


def test_transfer_cwes_explicit():
    from icp.label_generations import TRANSFER_CWES
    assert len(TRANSFER_CWES) == 7 and "CWE-22" not in TRANSFER_CWES
