"""Generations -> Bandit labels + PII flags (design #1, ticket #11). Code is only parsed and scanned."""

import io
import json
import random
import tokenize
from pathlib import Path

from icp.bandit_label import CWE_MAP, label, scan
from icp.pii import pii_flags

MIN_CODE_TOKENS = 2  # extract raises on rows with nothing to pool
TRANSFER_CWES = set(CWE_MAP.values())  # the 7 Bandit CWEs; CWE-22 is never a transfer area
SKIP = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
        tokenize.COMMENT}


def code_tokens(code: str) -> int:
    try:
        return sum(t.type not in SKIP for t in tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return 0


def label_rows(rows: list[dict]) -> list[dict]:
    """Add bandit_findings, label (None = unparseable or < MIN_CODE_TOKENS, excluded) and pii_flags."""
    bad = {r["cwe_area"] for r in rows} - TRANSFER_CWES
    if bad:
        raise ValueError(f"not a transfer CWE area (CWE-22 is pairs-only): {sorted(bad)}")
    out = []
    for r in rows:
        lab = label(r["code"]) if code_tokens(r["code"]) >= MIN_CODE_TOKENS else None
        found = [f._asdict() for f in scan(r["code"].removeprefix("﻿"))] if lab is not None else []
        out.append({**r, "bandit_findings": found, "label": lab,
                    "pii_flags": sorted(set(pii_flags(r["code"]) + pii_flags(r["raw_output"])))})
    return out


def text_free(rows: list[dict]) -> list[dict]:
    drop = {"prompt", "raw_output", "code"}
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def spot_sample(rows: list[dict], n: int = 20, seed: int = 0) -> list[dict]:
    """Seeded stratified sample for manual review: n B608-positive, n other-positive, n negative.
    Strata are reported separately (B608 is the noisiest rule)."""
    rng = random.Random(seed)
    strata = {"pos_B608": [], "pos_other": [], "neg": []}
    for r in rows:
        if r["label"] == 1:
            ids = {f["test_id"] for f in r["bandit_findings"] if f["cwe"]}
            strata["pos_B608" if "B608" in ids else "pos_other"].append(r)
        elif r["label"] == 0:
            strata["neg"].append(r)
    return [{"stratum": s, "id": r["id"], "label": r["label"], "findings": r["bandit_findings"],
             "code": r["code"], "manual_verdict": None}
            for s, rs in strata.items() for r in rng.sample(rs, min(n, len(rs)))]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
