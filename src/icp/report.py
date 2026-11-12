"""AUROC-vs-layer plot and summary tables from committed result CSVs (design #1, section 8).

Reads `results/<model>/` (who writes what):
- `probe_layers.csv` (PROBE rows only; eval_set pairs / transfer / loco_<cwe>) and
  `selected_layers.json` ({"transfer": <layer>}): the probing PR (#9).
- `scores_lexical_*.csv`, `scores_length_*.csv`: the baselines PR (#7).
- `scores_judge_pairs.csv`, `scores_judge_transfer.csv` and `judge_meta.json`
  ({"pairs": {"low_mass": bool, "n_truncated": int}, "transfer": {...}}): the judge PR (#12).
- `scores_topic_prior_transfer.csv`, `scores_prompt_risky_prior_transfer.csv`: #9.
- Score files are `scores_<method>_<eval_set>[_last].csv` with eval_set in {pairs, transfer,
  loco_<cwe>}; per-CWE in-dist files are `scores_probe_nested_pairs_<cwe>.csv`.
Baselines are never read from probe_layers.csv: their AUROC, CI and AUPRC are computed here from
the score files with icp.stats. Only mean pooling is reported (`_last` files are ignored).
Missing optional inputs give an `n/a` cell and a line under "Notes"; a malformed scores file
raises ValueError naming the file. AUPRC always carries its chance level (prevalence), and
bootstrap CIs report n_redraw / n_boot (more than 5% redraws flags the CI as underpowered).
"""

import json
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from icp import stats

NA = "n/a"
UNDERPOWERED = 0.05
SEL = "sel"  # the transfer layer from selected_layers.json
LAYER_METHODS = {"probe", "probe_shuffled"}  # drawn as curves; everything else as flat lines
# (label, in-dist source, transfer source, in-dist scores method for the gap, transfer scores method)
# source = ("pl", method, layer) from probe_layers.csv, or ("sc", method) computed from scores files
ROWS = [
    ("probe (nested in-dist / selected layer on transfer)", ("pl", "probe_nested", None), ("pl", "probe", SEL),
     "probe_nested_7cwe", "probe"),
    ("layer-0 probe", ("pl", "probe_layer0", 0), ("pl", "probe_layer0", 0), "probe_layer0", "probe_layer0"),
    ("lexical", ("sc", "lexical"), ("sc", "lexical"), "lexical", "lexical"),
    ("length", ("sc", "length"), ("sc", "length"), "length", "length"),
    ("shuffled-label probe", ("pl", "probe_shuffled", SEL), ("pl", "probe_shuffled", SEL),
     "probe_shuffled", "probe_shuffled"),
    ("topic prior", None, ("sc", "topic_prior"), None, "topic_prior"),
    ("prompt-risky prior", None, ("sc", "prompt_risky_prior"), None, "prompt_risky_prior"),
    ("judge", ("sc", "judge"), ("sc", "judge"), "judge", "judge"),
]


class Ev(NamedTuple):
    est: float
    lo: float
    hi: float
    auprc: float
    n_pos: int
    n_neg: int
    n_redraw: int


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
    if not path.exists():
        return None
    try:
        return stats.read_scores(path)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def _row(df: pd.DataFrame, method: str, eval_set: str, layer) -> pd.Series | None:
    sel = df[(df["method"] == method) & (df["eval_set"] == eval_set)]
    sel = sel[sel["layer"].isna()] if layer is None else sel[sel["layer"] == layer]
    return sel.iloc[0] if len(sel) else None


def _ev(results, df, src, eval_set, sel, n_boot, seed, notes, ci=True) -> Ev | None:
    """AUROC [CI], AUPRC and counts from probe_layers.csv or computed from a scores file."""
    if src is None:
        return None
    if src[0] == "pl":
        if src[2] == SEL and sel is None:
            return None
        r = _row(df, src[1], eval_set, sel if src[2] == SEL else src[2])
        if r is None:
            return None
        return Ev(r["auroc"], r["ci_lo"], r["ci_hi"], r["auprc"], int(r["n_pos"]), int(r["n_neg"]), int(r["n_redraw"]))
    sc = _scores(results, src[1], eval_set)
    if sc is None:
        notes.add(f"scores_{src[1]}_{eval_set}.csv missing")
        return None
    try:
        if ci:
            c = stats.auroc_ci(sc["label"], sc["score"], sc["group"], n_boot, seed)
        else:
            c = stats.CI(stats.auroc(sc["label"], sc["score"]), float("nan"), float("nan"), 0)
        return Ev(c.estimate, c.lo, c.hi, stats.auprc(sc["label"], sc["score"]),
                  int(sc["label"].sum()), int((sc["label"] == 0).sum()), c.n_redraw)
    except ValueError as e:
        notes.add(f"{src[1]} {eval_set}: {e}")
        return None


def _fmt(est: float, lo: float, hi: float, signed: bool = False) -> str:
    return f"{est:{'+' if signed else ''}.3f} [{lo:.3f}, {hi:.3f}]"


def _note_redraw(label: str, n_redraw: int, n_boot: int, notes: set) -> None:
    flag = " UNDERPOWERED" if n_redraw / n_boot > UNDERPOWERED else ""
    notes.add(f"{label}: n_redraw/n_boot = {n_redraw}/{n_boot}{flag}")


def _ci_cell(ci: stats.CI, n_boot: int, label: str, notes: set, signed: bool = False) -> str:
    _note_redraw(label, ci.n_redraw, n_boot, notes)
    return _fmt(ci.estimate, ci.lo, ci.hi, signed)


def _safe(fn, label: str, notes: set) -> str:
    try:
        return fn()
    except ValueError as e:  # too few groups, one class, ...
        notes.add(f"{label}: {e}")
        return NA


def _auroc_cell(e: Ev | None, n_boot: int, label: str, notes: set) -> str:
    if e is None:
        return NA
    if e.n_redraw / n_boot > UNDERPOWERED:
        _note_redraw(label, e.n_redraw, n_boot, notes)
    return _fmt(e.est, e.lo, e.hi)


def _auprc_cell(e: Ev | None) -> str:
    return NA if e is None else f"{e.auprc:.3f} (chance {e.n_pos / (e.n_pos + e.n_neg):.3f})"


def _files(results: Path, method: str, glob: str) -> list[Path]:
    """Score files of a method matching `glob`, mean pooling only (no `_last`)."""
    return sorted(f for f in results.glob(f"scores_{method}_{glob}.csv") if not f.stem.endswith("_last"))


def _pooled_loco(results: Path, method: str, n_boot: int, seed: int, label: str, notes: set) -> str:
    files = _files(results, method, "loco_*")
    if not files:
        notes.add(f"{label}: no scores_{method}_loco_*.csv (LOCO)")
        return NA
    df = pd.concat([_scores(results, method, f.stem.removeprefix(f"scores_{method}_")) for f in files])
    return _safe(lambda: _ci_cell(stats.auroc_ci(df["label"], df["score"], df["group"], n_boot, seed),
                                  n_boot, f"{label} LOCO", notes), f"{label} LOCO", notes)


def summary_table(results: str | Path, n_boot: int, seed: int) -> str:
    results = Path(results)
    df, notes = _load(results), set()
    sel = _json(results, "selected_layers.json", notes).get("transfer")
    meta = _json(results, "judge_meta.json", notes)
    low_mass = sorted(k for k, v in meta.items() if v.get("low_mass"))
    head = ["method", "in-dist AUROC [CI]", "LOCO AUROC [CI]", "transfer AUROC [CI]",
            "within-area transfer AUROC", "gap in-dist - transfer [CI]",
            "AUPRC in-dist", "AUPRC transfer", "n_pos/n_neg in-dist; transfer",
            "n_redraw/n_boot in-dist; transfer"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, in_src, tr_src, in_method, tr_method in ROWS:
        tag = label
        if label == "judge" and low_mass:
            label += f" (low_mass on {', '.join(low_mass)}: yes/no mass below 0.5, not meaningful)"
        ein = _ev(results, df, in_src, "pairs", sel, n_boot, seed, notes)
        etr = _ev(results, df, tr_src, "transfer", sel, n_boot, seed, notes)
        loco = _pooled_loco(results, in_src[1], n_boot, seed, tag, notes) if in_src else NA
        within = gap = NA
        tsc = _scores(results, tr_method, "transfer")
        if tsc is not None:
            within = _safe(lambda tsc=tsc: "{:.3f} ({} areas)".format(
                *stats.within_group_auroc(tsc["label"], tsc["score"], tsc["stratum"])), f"{tag} within-area", notes)
            isc = _scores(results, in_method, "pairs") if in_method else None
            if isc is not None:
                gap = _safe(lambda isc=isc, tsc=tsc, tag=tag: _ci_cell(
                    stats.gap_ci((isc["label"], isc["score"], isc["group"]),
                                 (tsc["label"], tsc["score"], tsc["group"]), n_boot, seed),
                    n_boot, f"{tag} gap", notes, signed=True), f"{tag} gap", notes)
        redraw = "; ".join(NA if e is None else f"{e.n_redraw}/{n_boot}" for e in (ein, etr))
        counts = "; ".join(NA if e is None else f"{e.n_pos}/{e.n_neg}" for e in (ein, etr))
        cells = [label, _auroc_cell(ein, n_boot, f"{tag} in-dist", notes), loco,
                 _auroc_cell(etr, n_boot, f"{tag} transfer", notes), within, gap,
                 _auprc_cell(ein), _auprc_cell(etr), counts, redraw]
        lines.append("| " + " | ".join(cells) + " |")
    if (df["eval_set"] == "transfer").sum() == 0:
        notes.add("no transfer rows in probe_layers.csv: probe transfer cells are n/a")
    lines += ["", f"Bootstrap: percentile cluster bootstrap, n_boot = {n_boot}, seed = {seed}.",
              "AUPRC is shown with its chance level (positive fraction).",
              ("Pooled LOCO mixes out-of-fold scores of different leave-one-CWE-out models, so score scales differ "
               "across CWEs; see per_cwe.md for per-CWE values."), "", "Notes:"]
    lines += [f"- {n}" for n in sorted(notes)]
    return "\n".join(lines) + "\n"


def per_cwe_table(results: str | Path, n_boot: int, seed: int) -> str:
    results = Path(results)
    notes: set = set()
    cwes = {f.stem.rsplit("_", 1)[-1] for f in _files(results, "probe_nested", "pairs_*")}
    cwes |= {f.stem.split("_loco_", 1)[1] for f in _files(results, "probe_nested", "loco_*")}
    lines = ["| CWE | in-dist AUROC [CI] | LOCO AUROC [CI] | note |", "|---|---|---|---|"]
    if not cwes:
        notes.add("no scores_probe_nested_pairs_<cwe>.csv or scores_probe_nested_loco_<cwe>.csv: no per-CWE rows")

    def cell(eval_set: str, cwe: str, what: str) -> str:
        sc = _scores(results, "probe_nested", eval_set)
        if sc is None:
            notes.add(f"{cwe}: scores_probe_nested_{eval_set}.csv missing ({what})")
            return NA
        return _safe(lambda: _ci_cell(stats.auroc_ci(sc["label"], sc["score"], sc["group"], n_boot, seed),
                                      n_boot, f"{cwe} {what}", notes), f"{cwe} {what}", notes)

    for cwe in sorted(cwes):
        note = "pairs only, not in transfer" if cwe == "CWE-22" else ""
        lines.append(f"| {cwe} | {cell(f'pairs_{cwe}', cwe, 'in-dist')} | {cell(f'loco_{cwe}', cwe, 'LOCO')} | {note} |")
    lines += ["", "Notes:"] + [f"- {n}" for n in sorted(notes)]
    return "\n".join(lines) + "\n"


def _draw(ax, df, method, eval_set, label, **style) -> bool:
    """Per-layer curve with CI band from probe_layers.csv, or a flat line for a layer-less row."""
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


def plot_auroc_vs_layer(results: str | Path, out: str | Path, n_boot: int, seed: int) -> list[str]:
    """Writes the PNG; returns notes about inputs that were missing."""
    results = Path(results)
    df, notes = _load(results), set()
    layers = _json(results, "selected_layers.json", notes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    lexical = (("sc", "lexical"), "lexical", {"color": "tab:orange", "ls": "--"})
    length = (("sc", "length"), "length only", {"color": "tab:green", "ls": "--"})
    judge = (("sc", "judge"), "self-judge", {"color": "tab:red", "ls": "--"})
    panels = [
        (axes[0], "pairs", "in-distribution (cross-validated, template groups)",
         [(("pl", "probe_nested", None), "nested probe (selected per fold)", {"color": "k", "ls": "-."}),
          lexical, length, judge]),
        (axes[1], "transfer", "transfer (generations, Bandit labels)",
         [lexical, length, judge, (("sc", "topic_prior"), "topic prior", {"color": "tab:purple", "ls": "--"})]),
    ]
    for ax, eval_set, title, extras in panels:
        if not _draw(ax, df, "probe", eval_set, "probe, per layer", color="tab:blue", marker="o", ms=3):
            notes.add(f"no probe rows for {eval_set}: curve not drawn")
        _draw(ax, df, "probe_shuffled", eval_set, "shuffled-label probe", color="gray", alpha=0.4, ls=":", band=False)
        for src, label, style in extras:
            e = _ev(results, df, src, eval_set, None, n_boot, seed, notes, ci=False)
            if e is not None:
                ax.axhline(e.est, label=label, **style)
        if eval_set == "transfer":
            if "transfer" in layers:
                ax.axvline(layers["transfer"], color="tab:blue", alpha=0.5, lw=1,
                           label=f"transfer layer ({layers['transfer']})")
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
