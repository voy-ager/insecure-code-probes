"""Lexical and length baselines (icp.lexical, #7) on pairs, LOCO and transfer.

python scripts/run_baselines.py --config configs/pilot_0.5b.toml

Writes scores_{lexical,length}_{pairs,loco_<cwe>,transfer}.csv to paths.results. Same folds as the
probe (icp.dataset). Transfer: trained on `[probe] transfer_train_cwes` (like the headline probe),
skipped if generations/labeled.jsonl is missing. Length counts Qwen tokens (tokenizer download only), capped at extract.max_len on transfer.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from icp import load_config
from icp.dataset import grouped_folds, load_pairs, loco_splits
from icp.lexical import METHODS, cv_scores, fit_score, load_tokenizer
from icp.stats import write_scores

log = logging.getLogger(__name__)


def run(cfg: dict, tok) -> None:
    seed, p, paths = cfg["seed"], cfg["probe"], cfg["paths"]
    res = Path(paths["results"])
    res.mkdir(parents=True, exist_ok=True)
    df = load_pairs(paths["pairs"])
    folds = grouped_folds(df, p["outer_folds"], seed)
    gen_path = Path(paths["generations"]) / "labeled.jsonl"
    gen = None
    if gen_path.exists():
        with open(gen_path, encoding="utf-8") as f:
            gen = pd.DataFrame([json.loads(line) for line in f if line.strip()])
        # lexical's frame contract: prompt_id -> template_id (group), cwe_area -> cwe (stratum)
        gen = gen[gen["label"].notna()].astype({"label": int})
        gen = gen.rename(columns={"prompt_id": "template_id", "cwe_area": "cwe"})
    else:
        log.warning("%s missing: transfer baselines skipped", gen_path)
    max_len = cfg.get("extract", {}).get("max_len", 512)

    def capped(codes, add_special_tokens=False):  # transfer code is judged on what the probe saw: <= max_len tokens
        return {"input_ids": [ids[:max_len] for ids in tok(codes, add_special_tokens=add_special_tokens)["input_ids"]]}

    train = df[df["cwe"].isin(p["transfer_train_cwes"])]
    for m in METHODS:
        write_scores(cv_scores(df, folds, m, seed, tok), res / f"scores_{m}_pairs.csv")
        for cwe, tr, te in loco_splits(df):
            write_scores(fit_score(df.iloc[tr], df.iloc[te], m, seed, tok), res / f"scores_{m}_loco_{cwe}.csv")
        if gen is not None:
            write_scores(fit_score(train, gen, m, seed, capped), res / f"scores_{m}_transfer.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    run(cfg, load_tokenizer(cfg))


if __name__ == "__main__":
    main()
