"""Label generations with Bandit and check them for PII (generated code is never run).

python scripts/label_generations.py --config configs/pilot_0.5b.toml [--spot-check 20]

Always writes labels.jsonl (no text). labeled.jsonl (with code/raw_output) is written only if the
PII check is clean on every row; otherwise the text goes to labeled.local.jsonl (gitignored).
Generation text is committed only if gitleaks ALSO passes on the file.
"""

import argparse
import json
import logging
from pathlib import Path

from icp import load_config
from icp.label_generations import label_rows, spot_sample, text_free, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--spot-check", type=int, default=0, help="write a seeded stratified sample of N per stratum")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    cfg = load_config(args.config)
    d = Path(cfg["paths"]["generations"])
    with open(d / "generations.jsonl", encoding="utf-8") as f:
        rows = label_rows([json.loads(line) for line in f if line.strip()])
    flagged = [r["id"] for r in rows if r["pii_flags"]]
    write_jsonl(d / "labels.jsonl", text_free(rows))
    write_jsonl(d / ("labeled.local.jsonl" if flagged else "labeled.jsonl"), rows)
    log.info("%d rows, %d labeled, %d positive; %d with PII flags %s", len(rows),
             sum(r["label"] is not None for r in rows), sum(r["label"] == 1 for r in rows), len(flagged), flagged)
    if flagged:
        log.warning("PII check failed: commit labels.jsonl only; text stays in labeled.local.jsonl")
    if args.spot_check:
        write_jsonl(d / "spot_check.local.jsonl", spot_sample(rows, args.spot_check, cfg["seed"]))


if __name__ == "__main__":
    main()
