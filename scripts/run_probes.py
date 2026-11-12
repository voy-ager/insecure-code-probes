"""Probe analyses (design #1, section 5).

python scripts/run_probes.py --config configs/pilot_0.5b.toml

Writes to paths.results: probe_layers.csv, selected_layers.json ({"transfer": L, plus per-fold
nested and LOCO choices; `_last` keys for last-token pooling}), paired_diffs.csv, and
scores_<method>_<eval_set>[_last].csv with eval_set in {pairs, transfer, loco_<cwe>}.

Needs data/activations/<model>/pairs_<pool>.npy (scripts/extract.py --set pairs); the transfer rows
also need generations/labeled.jsonl and transfer_<pool>.npy, and are skipped if those are missing.
Run the lexical/length baselines (#7) first so the H1 paired difference can be computed.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from icp import load_config
from icp.dataset import grouped_folds, load_pairs, loco_splits, shuffle_within_pairs
from icp.probe import (
    fit_score,
    layer_curve,
    nested_scores,
    select_layer,
    topic_prior,
    train_rows,
    transfer_scores,
)
from icp.stats import (
    auprc,
    auroc_ci,
    paired_diff_ci,
    read_scores,
    within_group_auroc,
    write_scores,
)

log = logging.getLogger(__name__)
LAYER_COLUMNS = [
    "method", "eval_set", "pooling", "layer", "auroc", "ci_lo", "ci_hi", "auprc",
    "n_pos", "n_neg", "n_redraw", "seed", "auroc_within", "note",
]
DIFF_COLUMNS = ["method_a", "method_b", "eval_set", "pooling", "diff", "ci_lo", "ci_hi", "n_pos", "n_neg", "n_redraw", "seed"]
PAIRS_ONLY = "CWE-22"


def _aligned(df: pd.DataFrame, act_dir: Path, name: str, pool: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Frame reordered to the activation rows (ids from <name>_meta.json)."""
    ids = json.loads((act_dir / f"{name}_meta.json").read_text(encoding="utf-8"))["ids"]
    X = np.load(act_dir / f"{name}_{pool}.npy")
    if len(ids) != len(X) or set(ids) != set(df["id"]):
        raise ValueError(f"{name}: activation ids do not match the data rows; re-run scripts/extract.py")
    return df.set_index("id").loc[ids].reset_index(), X


def _scores(df, group, stratum, scores, seed) -> pd.DataFrame:
    return pd.DataFrame({
        "id": df["id"].to_numpy(), "group": df[group].to_numpy(), "stratum": df[stratum].to_numpy(),
        "label": df["label"].to_numpy(), "score": scores, "seed": seed,
    })


def _row(method, eval_set, pool, layer, s: pd.DataFrame, n_boot, seed, note="", within="") -> dict:
    ci = auroc_ci(s["label"], s["score"], s["group"], n_boot, seed)
    return {
        "method": method, "eval_set": eval_set, "pooling": pool, "layer": layer,
        "auroc": ci.estimate, "ci_lo": ci.lo, "ci_hi": ci.hi, "auprc": auprc(s["label"], s["score"]),
        "n_pos": int(s["label"].sum()), "n_neg": int((s["label"] == 0).sum()), "n_redraw": ci.n_redraw,
        "seed": int(s["seed"].iloc[0]), "auroc_within": within, "note": note,
    }


def run(cfg: dict) -> None:
    seed, p, n_boot = cfg["seed"], cfg["probe"], cfg["stats"]["n_boot"]
    paths = cfg["paths"]
    act_dir, res = Path(paths["activations"]), Path(paths["results"])
    res.mkdir(parents=True, exist_ok=True)
    rows, diffs, selected = [], [], {}
    for pool in cfg["extract"]["pooling"]:
        suffix = "" if pool == "mean" else f"_{pool}"

        def save(method, eval_set, s, _suffix=suffix):
            write_scores(s, res / f"scores_{method}_{eval_set}{_suffix}.csv")

        df, X = _aligned(load_pairs(paths["pairs"]), act_dir, "pairs", pool)
        y = df["label"].to_numpy()

        # Per-layer OOF curves (descriptive), 8 CWEs: probe, layer-0 control, and the shuffled-label
        # control (labels swapped within pair_id; shuffle seed = config seed, the rows' `seed`).
        folds = grouped_folds(df, p["outer_folds"], seed)
        dfs = shuffle_within_pairs(df, seed)
        shuffled_curve = layer_curve(X, dfs, folds, cfg, seed)
        for layer, sc in enumerate(layer_curve(X, df, folds, cfg, seed)):
            s = _scores(df, "template_id", "cwe", sc, seed)
            rows.append(_row("probe", "pairs", pool, layer, s, n_boot, seed))
            rows.append(_row("probe_shuffled", "pairs", pool, layer,
                             _scores(dfs, "template_id", "cwe", shuffled_curve[layer], seed), n_boot, seed))
            if layer == 0:
                save("probe_layer0", "pairs", s)
                rows.append(_row("probe_layer0", "pairs", pool, 0, s, n_boot, seed))

        # H1: nested 8-CWE run, plus per-CWE rows (CWE-22 marked pairs_only). Nested rows have no
        # single layer; the per-fold choices go to selected_layers.json.
        sc, selected[f"probe_nested{suffix}"] = nested_scores(X, df, cfg, seed)
        nested = _scores(df, "template_id", "cwe", sc, seed)
        save("probe_nested", "pairs", nested)
        rows.append(_row("probe_nested", "pairs", pool, None, nested, n_boot, seed))
        for cwe, g in nested.groupby("stratum"):
            note = "pairs_only" if cwe == PAIRS_ONLY else ""
            rows.append(_row("probe_nested", f"pairs_{cwe}", pool, None, g, n_boot, seed, note))

        # H2 in-dist side: nested run on the 7 transfer-training CWEs only.
        sub = train_rows(df, p["transfer_train_cwes"])
        sc, selected[f"probe_nested_7cwe{suffix}"] = nested_scores(X[sub], df.iloc[sub], cfg, seed)
        s = _scores(df.iloc[sub], "template_id", "cwe", sc, seed)
        save("probe_nested_7cwe", "pairs", s)
        rows.append(_row("probe_nested_7cwe", "pairs", pool, None, s, n_boot, seed))

        # LOCO: layer picked by inner CV on the 7 training CWEs; one scores file per held-out CWE.
        for cwe, tr, te in loco_splits(df):
            layer = select_layer(X[tr], df.iloc[tr], cfg, seed, p["inner_folds"])
            selected.setdefault(f"loco{suffix}", {})[cwe] = layer
            note = "pairs_only" if cwe == PAIRS_ONLY else ""
            for method, at in (("probe_nested", layer), ("probe_layer0", 0)):
                s = _scores(df.iloc[te], "template_id", "cwe", fit_score(X[tr, at], y[tr], X[te, at], cfg, seed), seed)
                save(method, f"loco_{cwe}", s)
                rows.append(_row(method, f"loco_{cwe}", pool, at, s, n_boot, seed, note))

        # H1 paired difference vs. lexical. Integration point (#7): consumes its scores file.
        lex = res / f"scores_lexical_pairs{suffix}.csv"
        if lex.exists():
            b = read_scores(lex).set_index("id").loc[nested["id"]]
            ci = paired_diff_ci(nested["score"], b["score"], nested["label"], nested["group"], n_boot, seed)
            diffs.append({
                "method_a": "probe_nested", "method_b": "lexical", "eval_set": "pairs", "pooling": pool,
                "diff": ci.estimate, "ci_lo": ci.lo, "ci_hi": ci.hi, "n_pos": int(y.sum()),
                "n_neg": int((y == 0).sum()), "n_redraw": ci.n_redraw, "seed": seed,
            })
        else:
            log.warning("%s missing: run the lexical baseline (#7) first for the H1 paired difference", lex)

        gen_path = Path(paths["generations"]) / "labeled.jsonl"
        if not (gen_path.exists() and (act_dir / f"transfer_{pool}.npy").exists()):
            log.warning("transfer inputs missing for %s pooling; transfer rows skipped", pool)
            continue
        with open(gen_path, encoding="utf-8") as f:
            gen = pd.DataFrame([r for r in map(json.loads, filter(str.strip, f)) if r["label"] is not None])
        gen, Xg = _aligned(gen, act_dir, "transfer", pool)
        with open(paths["prompts"], encoding="utf-8") as f:
            natural = {r["prompt_id"]: r["natural"] for r in map(json.loads, filter(str.strip, f))}
        gen["natural"] = gen["prompt_id"].map(natural)
        if gen["natural"].isna().any():
            raise ValueError(f"generations with a prompt_id that has no `natural` field in {paths['prompts']}")

        def transfer_row(method, sc, layer, note="", _gen=gen, _pool=pool):
            s = _scores(_gen, "prompt_id", "cwe_area", sc, seed)
            within, n_areas = within_group_auroc(s["label"], s["score"], s["stratum"])
            log.info("%s transfer: within-area AUROC over %d areas", method, n_areas)
            rows.append(_row(method, "transfer", _pool, layer, s, n_boot, seed, note, within))
            return s

        # H2/H3 headline: 7 CWEs, no CWE-22 rows; layer chosen on the pairs only.
        train = df.iloc[sub]
        if (train["cwe"] == PAIRS_ONLY).any():
            raise ValueError("headline transfer training rows must not contain CWE-22")
        sc, layer = transfer_scores(X[sub], train, Xg, cfg, seed)
        selected[f"transfer{suffix}"] = layer
        head = transfer_row("probe", sc, layer, "selected")
        save("probe", "transfer", head)
        for nat in ("risky", "safe"):  # does the probe rank within natural-risky / natural-safe prompts?
            g = head[(gen["natural"] == nat).to_numpy()]
            if g["label"].nunique() == 2 and g["group"].nunique() > 1:
                rows.append(_row("probe", f"transfer_{nat}", pool, layer, g, n_boot, seed, "selected"))
            else:
                log.warning("transfer_%s: needs both classes and 2+ prompts; row skipped", nat)
        y_shuf = dfs.iloc[sub]["label"].to_numpy()
        for at in range(X.shape[1]):  # descriptive per-layer transfer curves (probe and shuffled control)
            if at != layer:
                transfer_row("probe", fit_score(X[sub, at], train["label"].to_numpy(), Xg[:, at], cfg, seed), at)
            s = transfer_row("probe_shuffled", fit_score(X[sub, at], y_shuf, Xg[:, at], cfg, seed), at)
            if at == layer:  # the shuffled control is compared at the transfer layer on both sets
                save("probe_shuffled", "transfer", s)
                save("probe_shuffled", "pairs", _scores(dfs, "template_id", "cwe", shuffled_curve[at], seed))
        s = transfer_row("probe_layer0", fit_score(X[sub, 0], train["label"].to_numpy(), Xg[:, 0], cfg, seed), 0)
        save("probe_layer0", "transfer", s)
        sc, layer8 = transfer_scores(X, df, Xg, cfg, seed)
        selected[f"transfer_8cwe{suffix}"] = layer8
        save("probe_8cwe", "transfer", transfer_row("probe_8cwe", sc, layer8, "secondary"))
        save("topic_prior", "transfer", transfer_row("topic_prior", topic_prior(gen), ""))
        s = transfer_row("prompt_risky_prior", topic_prior(gen, by="natural"), "")
        save("prompt_risky_prior", "transfer", s)

    pd.DataFrame(rows, columns=LAYER_COLUMNS).to_csv(res / "probe_layers.csv", index=False)
    (res / "selected_layers.json").write_text(json.dumps(selected, indent=1), encoding="utf-8")
    if diffs:
        pd.DataFrame(diffs, columns=DIFF_COLUMNS).to_csv(res / "paired_diffs.csv", index=False)
    log.info("wrote %d rows to %s", len(rows), res / "probe_layers.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config))


if __name__ == "__main__":
    main()
