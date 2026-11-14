"""Code string -> Bandit findings -> 0/1/None label (design #1, section 3).

The snippet is only parsed (ast) and scanned by Bandit as a file; it is never executed.
"""

import ast
import json
import subprocess  # nosec B404 - fixed argv, runs only bandit
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

MAX_CODE_BYTES = 200_000  # larger snippets are skipped (None): Bandit takes ~30 s per MB
TIMEOUT_S = 60

# Canonical CWE per Bandit test ID; CWE-327 covers 327/328. Informational IDs (B403, B404,
# B603, ...) are unmapped. B202 (tarfile, CWE-22) is deliberately unmapped: CWE-22 is
# pairs-only and must not enter the transfer labels.
CWE_MAP = {
    **dict.fromkeys(["B602", "B604", "B605"], "CWE-78"),
    "B608": "CWE-89",
    **dict.fromkeys(["B307", "B102"], "CWE-94"),
    **dict.fromkeys(["B301", "B302", "B506"], "CWE-502"),
    **dict.fromkeys(["B324", "B303", "B304", "B305"], "CWE-327"),
    **dict.fromkeys(["B105", "B106", "B107"], "CWE-798"),
    **dict.fromkeys(["B501", "B323"], "CWE-295"),
}


class Finding(NamedTuple):
    test_id: str
    cwe: str | None  # None for unmapped test IDs
    severity: str
    confidence: str
    line: int


def scan(code: str) -> list[Finding]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(  # nosec B603 - no shell; the snippet is a file argument, never run
                [sys.executable, "-m", "bandit", "--ignore-nosec", "-f", "json", "-q", str(path)],
                capture_output=True,
                encoding="utf-8",
                check=False,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"bandit timed out after {TIMEOUT_S}s") from None
    # exit 0 = clean, 1 = findings; anything else (or unparsable output) is a real failure
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"bandit failed ({proc.returncode}): {proc.stderr.strip()}")
    return [
        Finding(
            r["test_id"],
            CWE_MAP.get(r["test_id"]),
            r["issue_severity"],
            r["issue_confidence"],
            r["line_number"],
        )
        for r in json.loads(proc.stdout)["results"]
    ]


def scan_label(code: str) -> tuple[int | None, list[Finding]]:
    """(label, findings) from ONE Bandit scan. Label is None if the code does not parse, cannot
    be encoded as UTF-8 (lone surrogates) or exceeds MAX_CODE_BYTES (findings are then []); a
    leading BOM is stripped. Bandit runs with --ignore-nosec, so `# nosec` in a generation cannot
    hide a finding. Label 1 if any mapped finding (any confidence, any severity: B608 is LOW
    confidence outside execute(...), B105-B107 are severity LOW)."""
    code = code.removeprefix("﻿")
    try:
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            return None, []
        ast.parse(code)
    except (SyntaxError, ValueError):  # UnicodeEncodeError is a ValueError
        return None, []
    found = scan(code)
    return int(any(f.cwe for f in found)), found


def label(code: str) -> int | None:
    return scan_label(code)[0]
