"""AUROC-vs-layer plot and summary tables from committed result CSVs (design #1, section 8).

Reads `results/<model>/`: `probe_layers.csv`, `scores_<method>_<eval_set>.csv` (icp.stats schema),
and optionally `selected_layers.json` ({"transfer": <layer>}) and `judge_meta.json`
({"low_mass": bool}). Only mean pooling is reported. Missing optional inputs give an `n/a` cell and
a line under "Notes"; they never raise. AUPRC always carries its chance level (prevalence), and
bootstrap CIs report n_redraw / n_boot (more than 5% redraws flags the CI as underpowered).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from icp import stats

NA = "n/a"
UNDERPOWERED = 0.05
LAYER_METHODS = {"probe", "probe_shuffled"}  # drawn as curves; everything else as flat lines
# (label, in-dist (method, layer), transfer (method, layer), method of the in-dist scores file for the gap)
# layer "sel" = transfer layer from selected_layers.json (also used for the shuffled probe in-dist); None = the row has no layer.
ROWS = [
    ("probe (nested in-dist / selected layer on transfer)", ("probe_nested", None), ("probe", "sel"), "probe_nested_7cwe"),
    ("layer-0 probe", ("probe_layer0", 0), ("probe_layer0", 0), "probe_layer0"),
    ("lexical", ("lexical", None), ("lexical", None), "lexical"),
    ("length", ("length", None), ("length", None), "length"),
    ("shuffled-label probe", ("probe_shuffled", "sel"), ("probe_shuffled", "sel"), "probe_shuffled"),
    ("topic prior", None, ("topic_prior", None), None),
    ("judge", ("judge", None), ("judge", None), "judge"),
]


def _load(results: Path) -> pd.DataFrame:
    path = results / "probe_layers.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the probing step first")
    df = pd.read_csv(path)
    return df[df["pooling"] == "mean"] if "pooling" in df else df


def _json(results: Path, name: str, notes: set) -> dict:
    path = results / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    notes.add(f"{name} missing")
    return {}


def _scores(results: Path, method: str, eval_set: str) -> pd.DataFrame | None:
    path = results / f"scores_{method}_{eval_set}.csv"
    return stats.read_scores(path) if path.exists() else None


def _row(df: pd.DataFrame, method: str, eval_set: str, layer) -> pd.Series | None:
    sel = df[(df["method"] == method) & (df["eval_set"] == eval_set)]
    sel = sel[sel["layer"].isna()] if layer is None else sel[sel["layer"] == layer]
    return sel.iloc[0] if len(sel) else None


def _fmt(est: float, lo: float, hi: float, signed: bool = False) -> str:
    return f"{est:{'+' if signed else ''}.3f} [{lo:.3f}, {hi:.3f}]"


def _ci_cell(ci: stats.CI, n_boot: int, label: str, notes: set, signed: bool = False) -> str:
    rate = ci.n_redraw / n_boot
    notes.add(f"{label}: n_redraw/n_boot = {ci.n_redraw}/{n_boot}" + (" UNDERPOWERED" if rate > UNDERPOWERED else ""))
    return _fmt(ci.estimate, ci.lo, ci.hi, signed)


def _safe(fn, label: str, notes: set) -> str:
    try:
        return fn()
    except ValueError as e:  # too few groups, one class, ...
        notes.add(f"{label}: {e}")
        return NA


def _pooled_loco(results: Path, method: str, n_boot: int, seed: int, label: str, notes: set) -> str:
    files = sorted(results.glob(f"scores_{method}_loco_*.csv"))
    if not files:
        notes.add(f"{label}: no scores_{method}_loco_*.csv (LOCO)")
        return NA
    df = pd.concat([stats.read_scores(f) for f in files])
    return _safe(lambda: _ci_cell(stats.auroc_ci(df["label"], df["score"], df["group"], n_boot, seed),
                                  n_boot, f"{label} LOCO", notes), f"{label} LOCO", notes)


def _auroc_cell(r, n_boot: int, label: str, notes: set) -> str:
    if r is None:
        return NA
    if r["n_redraw"] / n_boot > UNDERPOWERED:
        notes.add(f"{label}: n_redraw/n_boot = {int(r['n_redraw'])}/{n_boot} UNDERPOWERED")
    return _fmt(r["auroc"], r["ci_lo"], r["ci_hi"])


def _auprc_cell(r) -> str:
    if r is None:
        return NA
    chance = r["n_pos"] / (r["n_pos"] + r["n_neg"])
    return f"{r['auprc']:.3f} (chance {chance:.3f})"


def _counts(r) -> str:
    return NA if r is None else f"{int(r['n_pos'])}/{int(r['n_neg'])}"


def summary_table(results: str | Path, n_boot: int, seed: int) -> str:
    results = Path(results)
    df, notes = _load(results), set()
    layers = _json(results, "selected_layers.json", notes)
    meta = _json(results, "judge_meta.json", notes)
    head = ["method", "in-dist AUROC [CI]", "LOCO AUROC [CI]", "transfer AUROC [CI]",
            "within-area transfer AUROC", "gap in-dist - transfer [CI]",
            "AUPRC in-dist", "AUPRC transfer", "n_pos/n_neg in-dist; transfer",
            "n_redraw/n_boot in-dist; transfer"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, ind, tr, score_method in ROWS:
        sel = layers.get("transfer")  # "sel" rows (probe, shuffled probe) use the transfer layer on both sets
        rin = rtr = None
        if ind is not None and not (ind[1] == "sel" and sel is None):
            rin = _row(df, ind[0], "indist_cv", sel if ind[1] == "sel" else ind[1])
        if tr is not None and not (tr[1] == "sel" and sel is None):
            rtr = _row(df, tr[0], "transfer", sel if tr[1] == "sel" else tr[1])
        if label == "judge" and meta.get("low_mass"):
            label += " (low_mass: yes/no probability mass below 0.5, not meaningful)"
        tag = label.split(" (")[0]
        loco = _pooled_loco(results, ind[0], n_boot, seed, tag, notes) if ind else NA
        within = gap = NA
        tsc = _scores(results, tr[0], "transfer") if tr else None
        if tsc is not None:
            within = _safe(lambda tsc=tsc: "{:.3f} ({} areas)".format(*stats.within_group_auroc(tsc["label"], tsc["score"], tsc["stratum"])),
                           f"{tag} within-area", notes)
            isc = _scores(results, score_method, "indist_cv") if score_method else None
            if isc is not None:
                gap = _safe(lambda isc=isc, tsc=tsc, tag=tag: _ci_cell(
                    stats.gap_ci((isc["label"], isc["score"], isc["group"]),
                                 (tsc["label"], tsc["score"], tsc["group"]), n_boot, seed),
                    n_boot, f"{tag} gap", notes, signed=True), f"{tag} gap", notes)
        redraw = "; ".join(NA if r is None else f"{int(r['n_redraw'])}/{n_boot}" for r in (rin, rtr))
        cells = [label, _auroc_cell(rin, n_boot, f"{tag} in-dist", notes), loco, _auroc_cell(rtr, n_boot, f"{tag} transfer", notes),
                 within, gap, _auprc_cell(rin), _auprc_cell(rtr), f"{_counts(rin)}; {_counts(rtr)}", redraw]
        lines.append("| " + " | ".join(cells) + " |")
        if rin is None and tr is not None and rtr is None:
            notes.add(f"{tag}: no in-dist or transfer rows in probe_layers.csv")
    if (df["eval_set"] == "transfer").sum() == 0:
        notes.add("no transfer rows in probe_layers.csv: transfer columns are n/a")
    lines += ["", f"Bootstrap: percentile cluster bootstrap, n_boot = {n_boot}, seed = {seed}.",
              "AUPRC is shown with its chance level (positive fraction).", "", "Notes:"]
    lines += [f"- {n}" for n in sorted(notes)]
    return "\n".join(lines) + "\n"


def per_cwe_table(results: str | Path, n_boot: int, seed: int) -> str:
    results = Path(results)
    notes: set = set()
    ind = _scores(results, "probe_nested", "indist_cv")
    lines = ["| CWE | in-dist AUROC [CI] | LOCO AUROC [CI] | note |", "|---|---|---|---|"]
    if ind is None:
        notes.add("scores_probe_nested_indist_cv.csv missing: no per-CWE rows")
    for cwe in sorted(ind["stratum"].unique()) if ind is not None else []:
        sub = ind[ind["stratum"] == cwe]
        a = _safe(lambda sub=sub, cwe=cwe: _ci_cell(stats.auroc_ci(sub["label"], sub["score"], sub["group"], n_boot, seed),
                                                  n_boot, f"{cwe} in-dist", notes), f"{cwe} in-dist", notes)
        lo = _scores(results, "probe_nested", f"loco_{cwe}")
        b = NA
        if lo is None:
            notes.add(f"{cwe}: scores_probe_nested_loco_{cwe}.csv missing (LOCO)")
        else:
            b = _safe(lambda lo=lo, cwe=cwe: _ci_cell(stats.auroc_ci(lo["label"], lo["score"], lo["group"], n_boot, seed),
                                                    n_boot, f"{cwe} LOCO", notes), f"{cwe} LOCO", notes)
        lines.append(f"| {cwe} | {a} | {b} | {'pairs only, not in transfer' if cwe == 'CWE-22' else ''} |")
    lines += ["", "Notes:"] + [f"- {n}" for n in sorted(notes)]
    return "\n".join(lines) + "\n"


def _draw(ax, df, method, eval_set, label, **style) -> bool:
    band = style.pop("band", True)
    sel = df[(df["method"] == method) & (df["eval_set"] == eval_set)]
    if sel.empty:
        return False
    if method in LAYER_METHODS and sel["layer"].notna().any():
        sel = sel.sort_values("layer")
        ax.plot(sel["layer"], sel["auroc"], label=label, **style)
        if band:
            ax.fill_between(sel["layer"], sel["ci_lo"], sel["ci_hi"], alpha=0.2, color=style.get("color"), lw=0)
    else:
        ax.axhline(sel.iloc[0]["auroc"], label=label, **style)
    return True


def plot_auroc_vs_layer(results: str | Path, out: str | Path) -> list[str]:
    """Writes the PNG; returns notes about inputs that were missing."""
    results = Path(results)
    df, notes = _load(results), set()
    layers = _json(results, "selected_layers.json", notes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        (axes[0], "indist_cv", "in-distribution (cross-validated, template groups)",
         [("probe_nested", "nested probe (selected per fold)", {"color": "k", "ls": "-."}),
          ("lexical", "lexical", {"color": "tab:orange", "ls": "--"}),
          ("length", "length only", {"color": "tab:green", "ls": "--"}),
          ("judge", "self-judge", {"color": "tab:red", "ls": "--"})]),
        (axes[1], "transfer", "transfer (generations, Bandit labels)",
         [("lexical", "lexical", {"color": "tab:orange", "ls": "--"}),
          ("length", "length only", {"color": "tab:green", "ls": "--"}),
          ("judge", "self-judge", {"color": "tab:red", "ls": "--"}),
          ("topic_prior", "topic prior", {"color": "tab:purple", "ls": "--"})]),
    ]
    for ax, eval_set, title, extras in panels:
        if not _draw(ax, df, "probe", eval_set, "probe, per layer", color="tab:blue", marker="o", ms=3):
            notes.add(f"no probe rows for {eval_set}: curve not drawn")
        _draw(ax, df, "probe_shuffled", eval_set, "shuffled-label probe", color="gray", alpha=0.4, ls=":", band=False)
        for method, label, style in extras:
            if not _draw(ax, df, method, eval_set, label, **style):
                notes.add(f"{method} {eval_set}: no rows, line not drawn")
        if eval_set == "transfer":
            if "transfer" in layers:
                ax.axvline(layers["transfer"], color="tab:blue", alpha=0.5, lw=1, label=f"transfer layer ({layers['transfer']})")
            else:
                notes.add("selected_layers.json has no 'transfer': layer not marked")
        ax.axhline(0.5, color="black", lw=0.6, label="chance (0.5)")
        ax.set(title=title, xlabel="layer (0 = embeddings)", ylim=(0, 1.02))
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("AUROC (mean pooling, 95% CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return sorted(notes)
