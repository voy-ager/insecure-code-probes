"""Sample model generations for the transfer prompts (code is only parsed and stored, never run).

python scripts/generate.py --config configs/pilot_0.5b.toml
"""

import argparse
import logging
from pathlib import Path

from icp import activations, load_config
from icp.generate import load_prompts, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    model, tok = activations.load_model(cfg)
    out = Path(cfg["paths"]["generations"]) / "generations.jsonl"
    n = run(load_prompts(cfg["paths"]["prompts"]), model, tok, cfg, out)
    logging.getLogger(__name__).info("%d new generations -> %s", n, out)


if __name__ == "__main__":
    main()
