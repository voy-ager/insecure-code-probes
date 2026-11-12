"""Score one set with the self-judgment baseline.

python scripts/judge.py --config configs/pilot_0.5b.toml --set pairs
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from icp import load_config
from icp.activations import load_model
from icp.dataset import load_pairs
from icp.judge import judge_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", required=True, choices=["pairs", "transfer"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    if args.set == "pairs":
        df = load_pairs(paths["pairs"])
    else:
        with open(Path(paths["generations"]) / "labeled.jsonl", encoding="utf-8") as f:
            df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
        df = df[df["label"].notna()].astype({"label": int})  # unparseable generations are excluded
    model, tok = load_model(cfg)
    out = judge_set(args.set, df, model, tok, cfg)
    logging.getLogger(__name__).info("%d rows scored", len(out))


if __name__ == "__main__":
    main()
