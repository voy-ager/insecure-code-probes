"""Generations -> Bandit labels + PII flags (design #1, ticket #11). Code is only parsed and scanned."""

import csv
import io
import json
import random
import tokenize
from collections import Counter
from pathlib import Path

from icp.bandit_label import scan_label
from icp.pii import pii_flags

MIN_CODE_TOKENS = 2  # extract raises on rows with nothing to pool
TRANSFER_CWES = {"CWE-78", "CWE-89", "CWE-94", "CWE-502", "CWE-327", "CWE-798", "CWE-295"}
assert len(TRANSFER_CWES) == 7 and "CWE-22" not in TRANSFER_CWES  # CWE-22 is pairs-only
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
        lab, found = scan_label(r["code"]) if code_tokens(r["code"]) >= MIN_CODE_TOKENS else (None, [])
        out.append({**r, "bandit_findings": [f._asdict() for f in found], "label": lab,
                    "pii_flags": sorted(set(pii_flags(r["code"]) + pii_flags(r["raw_output"])))})
    return out


def text_free(rows: list[dict]) -> list[dict]:
    drop = {"prompt", "raw_output", "code"}
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def spot_sample(rows: list[dict], n: int = 50, seed: int = 0) -> tuple[list[dict], dict[str, int]]:
    """Seeded stratified sample for manual review: up to n B608-positive, n other-positive, n negative
    (the negative stratum measures NPV, not recall). Strata are reported separately (B608 is the
    noisiest rule); weight pooled precision by the full stratum sizes, returned as the second value."""
    rng = random.Random(seed)  # nosec B311 - seeded sampling, not security
    strata = {"pos_B608": [], "pos_other": [], "neg_npv": []}
    for r in rows:
        if r["label"] == 1:
            ids = {f["test_id"] for f in r["bandit_findings"] if f["cwe"]}
            strata["pos_B608" if "B608" in ids else "pos_other"].append(r)
        elif r["label"] == 0:
            strata["neg_npv"].append(r)
    sample = [{"stratum": st, "id": r["id"], "label": r["label"],
               "findings": " ".join(sorted({f["test_id"] for f in r["bandit_findings"]})), "manual_verdict": ""}
              for st, rs in strata.items() for r in rng.sample(rs, min(n, len(rs)))]
    return sample, {st: len(rs) for st, rs in strata.items()}


def count_labels(rows: list[dict]) -> list[dict]:
    """cwe_area x label counts, nulls included."""
    c = Counter((r["cwe_area"], "null" if r["label"] is None else r["label"]) for r in rows)
    return [{"cwe_area": a, "label": lab, "n": n} for (a, lab), n in sorted(c.items(), key=str)]


def pii_counts(rows: list[dict]) -> Counter:
    """Flagged rows per flag type."""
    return Counter(f for r in rows for f in r["pii_flags"])


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
