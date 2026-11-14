"""Extract and cache activations for one set.

python scripts/extract.py --config configs/pilot_0.5b.toml --set pairs
"""

import argparse
import json
import logging
from pathlib import Path

from icp import load_config
from icp.activations import extract_cached
from icp.label_generations import labeled_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", required=True, choices=["pairs", "transfer"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    src = Path(paths["pairs"]) if args.set == "pairs" else labeled_path(paths["generations"])
    with open(src, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r["label"] is not None]  # unparseable generations are excluded
    out = extract_cached(args.set, [r["code"] for r in rows], [r["id"] for r in rows], cfg)
    logging.getLogger(__name__).info("%d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
