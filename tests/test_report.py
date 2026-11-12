import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from icp.report import per_cwe_table, plot_auroc_vs_layer, summary_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/results_tiny"
N_BOOT = 50


def _row(md: str, name: str) -> list[str]:
    line = next(ln for ln in md.splitlines() if ln.startswith(f"| {name}"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_summary_values_match_fixture():
    md = summary_table(FIXTURE, N_BOOT, 0)
    layers = pd.read_csv(FIXTURE / "probe_layers.csv")
    lex = layers[(layers.method == "lexical") & (layers.eval_set == "indist_cv")].iloc[0]
    cells = _row(md, "lexical")
    assert cells[1] == f"{lex.auroc:.3f} [{lex.ci_lo:.3f}, {lex.ci_hi:.3f}]"
    assert cells[6] == f"{lex.auprc:.3f} (chance {lex.n_pos / (lex.n_pos + lex.n_neg):.3f})"  # prevalence next to AUPRC
    # probe transfer cell uses the layer from selected_layers.json (4)
    probe = layers[(layers.method == "probe") & (layers.eval_set == "transfer") & (layers.layer == 4)].iloc[0]
    assert cells_probe(md)[3] == f"{probe.auroc:.3f} [{probe.ci_lo:.3f}, {probe.ci_hi:.3f}]"
    assert f"n_boot = {N_BOOT}" in md
    assert "n_redraw/n_boot" in md
    judge = _row(md, "judge")
    assert "low_mass on transfer" in judge[0]  # judge_meta.json is keyed per set; only transfer is low_mass
    assert judge[5] != "n/a"  # gap from scores_judge_pairs.csv + scores_judge_transfer.csv
    assert "mixes out-of-fold scores" in md
    assert "within-area" in md and "areas)" in cells_probe(md)[4]
    assert cells_probe(md)[2] != "n/a"  # pooled LOCO from the fixture scores


def cells_probe(md: str) -> list[str]:
    return _row(md, "probe (nested")


def test_loco_pool_ignores_last_token_files(tmp_path):
    res = tmp_path / "res"
    shutil.copytree(FIXTURE, res)
    before = _row(summary_table(res, N_BOOT, 0), "probe (nested")[2]
    shutil.copy(res / "scores_probe_nested_loco_CWE-78.csv", res / "scores_probe_nested_loco_CWE-78_last.csv")
    assert _row(summary_table(res, N_BOOT, 0), "probe (nested")[2] == before


def test_per_cwe_marks_cwe22():
    md = per_cwe_table(FIXTURE, N_BOOT, 0)
    assert "pairs only, not in transfer" in _row(md, "CWE-22")[-1]
    assert _row(md, "CWE-22")[2] != "n/a"  # has LOCO scores in the fixture
    assert _row(md, "CWE-295")[2] == "n/a"  # no LOCO scores: n/a, not a crash
    assert "scores_probe_nested_loco_CWE-295.csv missing" in md


def test_plot_is_small_png(tmp_path):
    out = tmp_path / "auroc_vs_layer.png"
    notes = plot_auroc_vs_layer(FIXTURE, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size < 200_000
    assert notes == []


def test_missing_inputs_give_na_and_notes(tmp_path):
    res = tmp_path / "res"
    shutil.copytree(FIXTURE, res)
    for name in ["judge_meta.json", "selected_layers.json"]:
        (res / name).unlink()
    for f in res.glob("scores_*_transfer.csv"):
        f.unlink()
    for f in res.glob("scores_*_loco_*.csv"):
        f.unlink()
    layers = pd.read_csv(res / "probe_layers.csv")
    layers[layers.eval_set != "transfer"].to_csv(res / "probe_layers.csv", index=False)
    md = summary_table(res, N_BOOT, 0)
    row = _row(md, "probe (nested")
    assert row[2] == row[3] == row[4] == row[5] == "n/a"
    assert "selected_layers.json missing" in md and "judge_meta.json missing" in md
    assert "no transfer rows" in md
    assert "low_mass" not in md
    notes = plot_auroc_vs_layer(res, tmp_path / "p.png")
    assert any("transfer" in n for n in notes)


def test_missing_probe_layers_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="probing step"):
        summary_table(tmp_path, N_BOOT, 0)


def test_script_writes_all_three(tmp_path, monkeypatch):
    res = tmp_path / "results"
    shutil.copytree(FIXTURE, res)
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'seed = 0\n[paths]\nresults = "{res.as_posix()}"\n[stats]\nn_boot = {N_BOOT}\n', encoding="utf-8")
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts/make_report.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", ["make_report.py", "--config", str(cfg)])
    script.main()
    assert all((res / n).stat().st_size > 0 for n in ["auroc_vs_layer.png", "summary.md", "per_cwe.md"])
