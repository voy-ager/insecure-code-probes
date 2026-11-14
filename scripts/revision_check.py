"""Pilot revision check and run stats (#14). Prints them and writes <results>/pilot_stats.json.

python scripts/revision_check.py --config configs/pilot_0.5b.toml

The single pre-declared revision may use ONLY parse rates, per-area class counts and timing (never
a probe or judge AUROC). Trigger: an area with < 8 positives or a parse rate < 70%. 1.5B samples per
prompt: 12 if the projected 1.5B generation time is <= 8 h, otherwise 8.
Also reported when their inputs exist: max |activation|, control_token_ids, n_truncated (activation
metas), judge low_mass and n_truncated, B608 spot-check precision (manual verdicts), and the
shuffled-label sanity check from probe_layers.csv. Missing inputs are reported as missing.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from icp import load_config

MIN_POS, MIN_PARSE = 8, 0.70
MAX_HOURS = 8
# shortcut: generation time scaled by parameter count (1.54B / 0.49B); re-check against a short 1.5B run
SCALE_1_5B = 3.1


def _jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def area_stats(rows: list[dict]) -> pd.DataFrame:
    """Per cwe_area: generations, parse rate, labeled/positive/negative counts and the trigger."""
    df = pd.DataFrame(rows)
    g = df.groupby("cwe_area")
    out = pd.DataFrame({
        "n": g.size(),
        "parse_rate": g["parse_ok"].mean().round(3),
        "n_pos": g["label"].apply(lambda s: int((s == 1).sum())),
        "n_neg": g["label"].apply(lambda s: int((s == 0).sum())),
    })
    out["trigger"] = (out["n_pos"] < MIN_POS) | (out["parse_rate"] < MIN_PARSE)
    return out


def projection(gen_seconds: float, samples_now: int) -> dict:
    """Projected 1.5B generation hours for 8 and 12 samples/prompt, and the resulting choice."""
    hours = {n: round(gen_seconds * SCALE_1_5B * n / samples_now / 3600, 2) for n in (8, 12)}
    return {"hours_8": hours[8], "hours_12": hours[12], "samples_1_5b": 12 if hours[12] <= MAX_HOURS else 8}


def b608_precision(spot: list[dict]) -> dict:
    """Precision of Bandit positives from manual verdicts (1 = really insecure), B608 on its own."""
    out = {}
    for stratum in ("pos_B608", "pos_other"):
        v = [r["manual_verdict"] for r in spot if r["stratum"] == stratum and r["manual_verdict"] is not None]
        out[stratum] = {"n_reviewed": len(v), "precision": round(sum(v) / len(v), 3) if v else None}
    return out


def shuffled_sanity(layers: pd.DataFrame) -> dict:
    """Mean shuffled-label AUROC across layers in [0.45, 0.55] and <= 2 layers whose CI excludes 0.5."""
    s = layers[(layers["method"] == "probe_shuffled") & (layers["eval_set"] == "pairs") & (layers["pooling"] == "mean")]
    mean = float(s["auroc"].mean())
    n_excl = int(((s["ci_lo"] > 0.5) | (s["ci_hi"] < 0.5)).sum())
    return {"mean_auroc": round(mean, 3), "n_layers_ci_excludes_0.5": n_excl,
            "pass": 0.45 <= mean <= 0.55 and n_excl <= 2}


def collect(cfg: dict) -> dict:
    paths = cfg["paths"]
    gens, res, acts = Path(paths["generations"]), Path(paths["results"]), Path(paths["activations"])
    stats: dict = {"missing": []}
    rows = _jsonl(gens / "labels.jsonl")
    if rows is None:
        stats["missing"].append("labels.jsonl")
    else:
        areas = area_stats(rows)
        stats["areas"] = areas.reset_index().to_dict("records")
        stats["revision_triggered"] = bool(areas["trigger"].any())
    log = res / "run_log.json"
    gen_step = json.loads(log.read_text(encoding="utf-8"))["steps"].get("generate") if log.exists() else None
    if gen_step and gen_step.get("n_cached") != 0:  # generate skips cached samples, so its time is too small
        stats["timing"] = {"INVALID": "generate timing is only valid when every sample was newly generated "
                           f"(n_cached={gen_step.get('n_cached')}); re-run generate with --force on a clean "
                           "generations dir"}
        stats["missing"].append("valid generate timing")
    elif gen_step:
        stats["timing"] = projection(gen_step["seconds"], cfg["generate"]["samples_per_prompt"])
    else:
        stats["missing"].append("generate timing (run_log.json)")
    for name in ("pairs", "transfer"):
        meta = acts / f"{name}_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            stats[f"activations_{name}"] = {k: m.get(k) for k in ("max_abs_activation", "control_token_ids", "n_truncated")}
        else:
            stats["missing"].append(meta.name)
    judge = res / "judge_meta.json"
    if judge.exists():
        stats["judge"] = {k: {f: v.get(f) for f in ("low_mass", "n_truncated", "mass_mean")}
                          for k, v in json.loads(judge.read_text(encoding="utf-8")).items()}
    else:
        stats["missing"].append("judge_meta.json")
    spot = _jsonl(gens / "spot_check.local.jsonl")
    if spot is None:
        stats["missing"].append("spot_check.local.jsonl")
    else:
        stats["spot_check"] = b608_precision(spot)
    layers = res / "probe_layers.csv"
    if layers.exists():
        # sanity only: not used by revision_triggered
        stats["shuffled_sanity"] = shuffled_sanity(pd.read_csv(layers))
    else:
        stats["missing"].append("probe_layers.csv")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    stats = collect(cfg)
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    (res / "pilot_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    if "areas" in stats:
        print(pd.DataFrame(stats["areas"]).to_string(index=False))
        print(f"revision triggered (< {MIN_POS} positives or parse rate < {MIN_PARSE:.0%} in an area):",
              stats["revision_triggered"])
    print(json.dumps({k: v for k, v in stats.items() if k != "areas"}, indent=1))


if __name__ == "__main__":
    main()
