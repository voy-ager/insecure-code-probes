"""Write auroc_vs_layer.png, summary.md and per_cwe.md into the results directory.

python scripts/make_report.py --config configs/pilot_0.5b.toml
"""

import argparse
import sys
from pathlib import Path

from icp import load_config
from icp.report import per_cwe_table, plot_auroc_vs_layer, summary_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["results"])
    n_boot, seed = cfg["stats"]["n_boot"], cfg["seed"]
    try:
        notes = plot_auroc_vs_layer(out, out / "auroc_vs_layer.png", n_boot, seed)
        (out / "summary.md").write_text(summary_table(out, n_boot, seed), encoding="utf-8", newline="\n")
        (out / "per_cwe.md").write_text(per_cwe_table(out, n_boot, seed), encoding="utf-8", newline="\n")
    except (FileNotFoundError, ValueError) as e:  # missing probe_layers.csv; malformed scores file (message names it)
        sys.exit(f"make_report: {e}")
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    print(f"wrote auroc_vs_layer.png, summary.md, per_cwe.md to {out}")


if __name__ == "__main__":
    main()
