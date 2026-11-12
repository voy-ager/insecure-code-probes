import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from icp import stats
from icp.report import per_cwe_table, plot_auroc_vs_layer, summary_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/results_tiny"
N_BOOT = 50


def _write_cfg(cfg: Path, res: Path) -> None:
    lines = ["seed = 0", "[paths]", f'results = "{res.as_posix()}"', "[stats]", f"n_boot = {N_BOOT}"]
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(md: str, name: str) -> list[str]:
    line = next(ln for ln in md.splitlines() if ln.startswith(f"| {name}"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_summary_values_match_fixture():
    md = summary_table(FIXTURE, N_BOOT, 0)
    layers = pd.read_csv(FIXTURE / "probe_layers.csv")
    # baselines are computed from the score files with icp.stats, not read from probe_layers.csv
    assert not set(layers.method) & {"lexical", "length", "judge", "topic_prior"}
    lex = stats.read_scores(FIXTURE / "scores_lexical_pairs.csv")
    ci = stats.auroc_ci(lex["label"], lex["score"], lex["group"], N_BOOT, 0)
    cells = _row(md, "lexical")
    assert cells[1] == f"{ci.estimate:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]"
    prev = lex["label"].mean()
    assert cells[6] == f"{stats.auprc(lex['label'], lex['score']):.3f} (chance {prev:.3f})"  # prevalence next to AUPRC
    assert cells[8] == f"{lex['label'].sum()}/{(lex['label'] == 0).sum()}; 30/54"
    assert _row(md, "prompt-risky prior")[3] != "n/a" and _row(md, "topic prior")[3] != "n/a"
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
    assert _row(md, "CWE-22")[1] != "n/a" and _row(md, "CWE-22")[2] != "n/a"  # pairs_<cwe> and loco_<cwe> files
    assert _row(md, "CWE-295")[1] != "n/a"
    assert _row(md, "CWE-295")[2] == "n/a"  # no LOCO scores: n/a, not a crash
    assert "scores_probe_nested_loco_CWE-295.csv missing" in md


def test_plot_is_small_png(tmp_path):
    out = tmp_path / "auroc_vs_layer.png"
    notes = plot_auroc_vs_layer(FIXTURE, out, N_BOOT, 0)
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
    notes = plot_auroc_vs_layer(res, tmp_path / "p.png", N_BOOT, 0)
    assert any("transfer" in n for n in notes)


def test_missing_probe_layers_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="probing step"):
        summary_table(tmp_path, N_BOOT, 0)


def test_script_writes_all_three(tmp_path, monkeypatch):
    res = tmp_path / "results"
    shutil.copytree(FIXTURE, res)
    cfg = tmp_path / "c.toml"
    _write_cfg(cfg, res)
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts/make_report.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", ["make_report.py", "--config", str(cfg)])
    script.main()
    assert all((res / n).stat().st_size > 0 for n in ["auroc_vs_layer.png", "summary.md", "per_cwe.md"])


def test_script_exits_with_message_naming_a_malformed_scores_file(tmp_path, monkeypatch):
    res = tmp_path / "results"
    shutil.copytree(FIXTURE, res)
    bad = res / "scores_lexical_pairs.csv"
    pd.DataFrame({"id": ["a"], "group": ["g"], "stratum": ["s"], "label": [1], "score": [float("nan")], "seed": [0]}).to_csv(bad, index=False)
    cfg = tmp_path / "c.toml"
    _write_cfg(cfg, res)
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts/make_report.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", ["make_report.py", "--config", str(cfg)])
    with pytest.raises(SystemExit) as e:
        script.main()
    assert "scores_lexical_pairs.csv" in str(e.value) and "finite" in str(e.value)
