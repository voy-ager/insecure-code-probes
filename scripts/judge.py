"""Score one set with the self-judgment baseline.

python scripts/judge.py --config configs/pilot_0.5b.toml --set pairs
"""

import argparse
import json
import logging

import pandas as pd

from icp import load_config
from icp.activations import load_model
from icp.dataset import load_pairs
from icp.judge import judge_set
from icp.label_generations import labeled_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", required=True, choices=["pairs", "transfer"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    if args.set == "pairs":
        df, group, stratum = load_pairs(paths["pairs"]), "template_id", "cwe"
    else:
        with open(labeled_path(paths["generations"]), encoding="utf-8") as f:
            df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
        df = df[df["label"].notna()].astype({"label": int})  # unparseable generations are excluded
        group, stratum = "prompt_id", "cwe_area"  # labeled.jsonl: id, prompt_id, cwe_area, code, label
    model, tok = load_model(cfg)
    out = judge_set(args.set, df, model, tok, cfg, group, stratum)
    logging.getLogger(__name__).info("%d rows scored", len(out))


if __name__ == "__main__":
    main()
