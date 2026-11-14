"""Sample model generations for the transfer prompts (code is only parsed and stored, never run).

python scripts/generate.py --config configs/pilot_0.5b.toml
"""

import argparse
import json
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
    total = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
    # run_all copies this into run_log: timing is valid for the 1.5B projection only if n_cached == 0
    out.with_name("generate_run.json").write_text(
        json.dumps({"n_generated": n, "n_cached": total - n}), encoding="utf-8")
    logging.getLogger(__name__).info("%d new generations -> %s", n, out)


if __name__ == "__main__":
    main()
