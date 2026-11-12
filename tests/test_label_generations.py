"""Fixture code is only parsed and Bandit-scanned, never executed. Fake PII is built by concatenation."""

import json
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
    s = spot_sample(rows, n=1, seed=0)
    strata = {x["stratum"] for x in s}
    assert strata == {"pos_B608", "pos_other", "neg"}
    assert s == spot_sample(rows, n=1, seed=0)
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
