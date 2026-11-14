"""Label generations with Bandit and check them for PII (generated code is never run).

python scripts/label_generations.py --config configs/pilot_0.5b.toml [--spot-check 50]

Writes, to paths.generations: labels.jsonl (no text; one row per generation, every run) and exactly one
of labeled.jsonl (text, PII-clean) or labeled.local.jsonl (text, gitignored, PII-flagged); the other
variant is deleted so a stale file can never disagree with labels.jsonl. Downstream steps (extract,
judge, probes) take their included rows from labels.jsonl / labeled.* with label not null.
Writes to paths.results: transfer_label_counts.csv (cwe_area x label, nulls included) and, with
--spot-check N (default 50, 0 = off), spotcheck.csv (seeded strata B608-positive / other-positive /
negative; the negative stratum measures NPV, not recall; manual_verdict left empty; ids only, look
the code up in labeled.*). Stratum sizes are logged for weighting pooled precision.

Commit rule: generation text is committed only if this PII check AND gitleaks both pass. The gitleaks
half is NOT enforced here; it is enforced by run_all's gate in PR #30. pii.py false positives (version
strings like 1.2.3.4, long identifiers) err on the safe side: the text then stays local.
"""

import argparse
import json
import logging
from pathlib import Path

from icp import load_config
from icp.label_generations import (
    count_labels,
    label_rows,
    pii_counts,
    spot_sample,
    text_free,
    write_csv,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--spot-check", type=int, default=50, help="seeded sample of up to N per stratum (0 = off)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    cfg = load_config(args.config)
    d = Path(cfg["paths"]["generations"])
    with open(d / "generations.jsonl", encoding="utf-8") as f:
        rows = label_rows([json.loads(line) for line in f if line.strip()])
    flagged = [r for r in rows if r["pii_flags"]]
    write_jsonl(d / "labels.jsonl", text_free(rows))
    keep, drop = ("labeled.local.jsonl", "labeled.jsonl") if flagged else ("labeled.jsonl", "labeled.local.jsonl")
    write_jsonl(d / keep, rows)
    (d / drop).unlink(missing_ok=True)
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    write_csv(res / "transfer_label_counts.csv", count_labels(rows), ["cwe_area", "label", "n"])
    log.info("%d rows, %d labeled, %d positive; %d rows with PII flags, per type %s", len(rows),
             sum(r["label"] is not None for r in rows), sum(r["label"] == 1 for r in rows), len(flagged),
             dict(pii_counts(rows)))
    if flagged:
        log.warning("PII check failed: commit labels.jsonl only; text stays in labeled.local.jsonl")
    if args.spot_check:
        sample, sizes = spot_sample(rows, args.spot_check, cfg["seed"])
        for st, n in sizes.items():
            log.info("spot-check stratum %s: %d rows, %d sampled", st, n, sum(x["stratum"] == st for x in sample))
        write_csv(res / "spotcheck.csv", sample, ["stratum", "id", "label", "findings", "manual_verdict"])


if __name__ == "__main__":
    main()
