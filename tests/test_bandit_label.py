"""Snippets are strings that are only parsed and Bandit-scanned, never executed."""

import pytest

from icp import bandit_label
from icp.bandit_label import CWE_MAP, MAX_CODE_BYTES, label, scan

# test_id -> insecure snippet that Bandit 1.9.4 flags with that ID
INSECURE = {
    "B602": "import subprocess\nsubprocess.call(cmd, shell=True)\n",
    "B604": "def f(cmd, run):\n    run(cmd, shell=True)\n",
    "B605": "import os\nos.system(cmd)\n",
    "B608": 'cur.execute(f"SELECT * FROM t WHERE id = {x}")\n',
    "B307": "eval(s)\n",
    "B102": "exec(s)\n",
    "B301": "import pickle\npickle.loads(d)\n",
    "B302": "import marshal\nmarshal.loads(d)\n",
    "B506": "import yaml\nyaml.load(d)\n",
    "B324": "import hashlib\nhashlib.md5(d)\n",
    "B303": "from Crypto.Hash import MD5\nMD5.new(d)\n",
    "B304": "from Crypto.Cipher import ARC4\nARC4.new(k)\n",
    "B305": "from cryptography.hazmat.primitives.ciphers import modes\nmodes.ECB()\n",
    "B105": "password = 'placeholder-not-real'\n",
    "B106": "f(password='placeholder-not-real')\n",
    "B107": "def f(password='placeholder-not-real'):\n    pass\n",
    "B501": "import requests\nrequests.get(u, timeout=5, verify=False)\n",
    "B323": "import ssl\nssl._create_unverified_context()\n",
}

SECURE = {
    "CWE-78": "import subprocess\nsubprocess.run(['ls', d], check=True)\n",
    "CWE-89": 'cur.execute("SELECT * FROM t WHERE id = ?", (x,))\n',
    "CWE-94": "import ast\nast.literal_eval(s)\n",
    "CWE-502": "import json\njson.loads(d)\n",
    "CWE-327": "import hashlib\nhashlib.sha256(d)\n",
    "CWE-798": "import os\npassword = os.environ['DB_PASSWORD']\n",
    "CWE-295": "import requests\nrequests.get(u, timeout=5)\n",
}


def test_every_mapped_id_is_covered_by_an_insecure_snippet():
    assert set(INSECURE) == set(CWE_MAP)


@pytest.mark.parametrize("test_id", INSECURE)
def test_insecure_snippet_fires_its_id_and_is_labeled_1(test_id):
    findings = [f for f in scan(INSECURE[test_id]) if f.test_id == test_id]
    assert findings and findings[0].cwe == CWE_MAP[test_id]
    assert label(INSECURE[test_id]) == 1


@pytest.mark.parametrize("cwe", SECURE)
def test_secure_snippet_is_labeled_0(cwe):
    assert label(SECURE[cwe]) == 0


def test_b608_both_forms_labeled_1():
    inline = 'cur.execute(f"SELECT * FROM t WHERE id = {x}")\n'
    built = 'q = f"SELECT * FROM t WHERE id = {x}"\ncur.execute(q)\n'
    assert label(inline) == 1
    assert label(built) == 1
    # the built-then-executed form is the LOW-confidence one that a >=MEDIUM filter would drop
    assert [f.confidence for f in scan(built) if f.test_id == "B608"] == ["LOW"]


def test_low_severity_b105_is_labeled_1():
    (f,) = scan(INSECURE["B105"])
    assert (f.test_id, f.severity) == ("B105", "LOW")
    assert label(INSECURE["B105"]) == 1


def test_unmapped_finding_alone_is_labeled_0():
    code = "import subprocess\n"  # B404, informational
    assert [f.test_id for f in scan(code)] == ["B404"]
    assert label(code) == 0


def test_b202_is_unmapped_so_no_cwe_22():
    code = "import tarfile\ntarfile.open(p).extractall()\n"
    findings = scan(code)
    assert "B202" in {f.test_id for f in findings}
    assert all(f.cwe is None for f in findings)
    assert label(code) == 0


def test_cwe_327_is_the_single_label_for_327_and_328():
    assert "CWE-328" not in CWE_MAP.values()


def test_unparseable_code_is_none():
    assert label("def f(:\n") is None


@pytest.mark.parametrize("marker", ["# nosec", "# nosec B605", "# nosec B999"])
def test_nosec_comment_does_not_hide_a_finding(marker):
    assert label(f"import os\nos.system(cmd)  {marker}\n") == 1


def test_lone_surrogate_is_unlabelable():
    assert label("x = '\ud800'\n") is None


def test_leading_bom_is_stripped():
    assert label("\ufeffimport os\nos.system(cmd)\n") == 1


def test_oversized_snippet_is_skipped():
    assert label("x = 1\n" * (MAX_CODE_BYTES // 6 + 1)) is None


def test_non_ascii_snippet_is_labeled():
    assert label("name = 'caf\u00e9'\nimport os\nos.system(cmd)\n") == 1


def test_bandit_timeout_raises_clear_error(monkeypatch):
    monkeypatch.setattr(bandit_label, "TIMEOUT_S", 0.001)
    with pytest.raises(RuntimeError, match="timed out"):
        scan("x = 1\n")
