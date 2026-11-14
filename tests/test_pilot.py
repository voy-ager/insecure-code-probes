"""run_all orchestration and the #14 revision check, with stubs only (no model, no subprocess)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text(f'seed = 0\n[paths]\nresults = "{(tmp_path / "res").as_posix()}"\n'
                 f'generations = "{(tmp_path / "gen").as_posix()}"\n'
                 f'activations = "{(tmp_path / "acts").as_posix()}"\n'
                 '[generate]\nsamples_per_prompt = 8\n', encoding="utf-8")
    (tmp_path / "gen").mkdir()
    return p


def _fake_run(calls):
    def run(argv):
        calls.append(argv[0])
        return {"seconds": 1.0, "peak_ram_mb": 1}
    return run


def test_run_all_order_and_resume(cfg_file, tmp_path, monkeypatch):
    mod = _load("run_all")
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)  # no gitleaks: gate reports "not run"
    calls = []
    mod.main(["--config", str(cfg_file)], run=_fake_run(calls))
    assert calls == ["scripts/build_pairs.py", "scripts/extract.py", "scripts/generate.py",
                     "scripts/label_generations.py", "scripts/revision_check.py", "scripts/extract.py",
                     "scripts/run_baselines.py", "scripts/judge.py", "scripts/judge.py", "scripts/run_probes.py",
                     "scripts/make_report.py", "scripts/revision_check.py"]
    state = json.loads((tmp_path / "res" / "run_log.json").read_text())
    assert state["steps"]["gate"] == {"pii_clean": True, "gitleaks": "not run"}

    calls.clear()
    mod.main(["--config", str(cfg_file)], run=_fake_run(calls))
    assert calls == []  # everything done: resumed
    mod.main(["--config", str(cfg_file), "--from", "run_probes"], run=_fake_run(calls))
    assert calls == ["scripts/run_probes.py", "scripts/make_report.py", "scripts/revision_check.py"]


def test_run_all_stops_at_failed_step_and_resumes_there(cfg_file, tmp_path, monkeypatch):
    mod = _load("run_all")
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    calls = []

    def flaky(argv):
        if argv[0] == "scripts/generate.py" and "scripts/generate.py" not in calls:
            calls.append(argv[0])
            raise SystemExit("step failed")
        return _fake_run(calls)(argv)

    with pytest.raises(SystemExit):
        mod.main(["--config", str(cfg_file)], run=flaky)
    calls.clear()
    mod.main(["--config", str(cfg_file)], run=_fake_run(calls))
    assert calls[0] == "scripts/generate.py"


def test_gate_stops_on_pii_flags(cfg_file, tmp_path):
    mod = _load("run_all")
    (tmp_path / "gen" / "labeled.local.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="PII"):
        mod.main(["--config", str(cfg_file)], run=_fake_run([]))


def test_area_stats_triggers():
    rc = _load("revision_check")
    rows = ([{"cwe_area": "CWE-89", "parse_ok": True, "label": i % 2} for i in range(20)]  # 10 pos: ok
            + [{"cwe_area": "CWE-78", "parse_ok": True, "label": int(i < 7)} for i in range(20)]  # 7 pos
            + [{"cwe_area": "CWE-94", "parse_ok": i < 13, "label": 1 if i < 13 else None} for i in range(20)])  # 65%
    a = rc.area_stats(rows)
    assert a.loc["CWE-89", "n_pos"] == 10 and not a.loc["CWE-89", "trigger"]
    assert a.loc["CWE-78", "trigger"] and a.loc["CWE-94", "trigger"]
    assert a.loc["CWE-94", "parse_rate"] == 0.65


def test_projection_rule():
    rc = _load("revision_check")
    fast = rc.projection(3600, 8)  # 1 h at 0.5B, 8 samples -> 3.1 h / 4.65 h at 1.5B
    assert fast["samples_1_5b"] == 12 and fast["hours_8"] == 3.1
    assert rc.projection(3 * 3600, 8)["samples_1_5b"] == 8  # 12 samples -> 13.95 h > 8 h


def test_spot_check_and_shuffled_sanity():
    rc = _load("revision_check")
    spot = [{"stratum": "pos_B608", "manual_verdict": v} for v in (1, 1, 0, None)]
    assert rc.b608_precision(spot)["pos_B608"] == {"n_reviewed": 3, "precision": 0.667}
    layers = pd.DataFrame({"method": "probe_shuffled", "eval_set": "pairs", "pooling": "mean",
                           "auroc": [0.5, 0.52, 0.47, 0.6], "ci_lo": [0.4, 0.42, 0.37, 0.51],
                           "ci_hi": [0.6, 0.62, 0.57, 0.7]})
    assert rc.shuffled_sanity(layers) == {"mean_auroc": 0.522, "n_layers_ci_excludes_0.5": 1, "pass": True}


def test_collect_reports_missing_inputs(cfg_file):
    from icp import load_config

    stats = _load("revision_check").collect(load_config(cfg_file))
    assert "labels.jsonl" in stats["missing"] and "probe_layers.csv" in stats["missing"]


def test_run_baselines_writes_contract_files(tmp_path):
    from icp.stats import read_scores

    rows = []
    for cwe in ["CWE-22", "CWE-78", "CWE-89"]:
        for t in range(5):
            for lab in (0, 1):
                tid = f"{cwe}-t{t}"
                rows.append({"id": f"{tid}-{lab}", "pair_id": tid, "template_id": tid, "cwe": cwe, "label": lab,
                             "code": f"x{t} = {'eval' if lab else 'int'}(s)  # {cwe}\n", "variation": {},
                             "hard": None, "expected_bandit": []})
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    gen = tmp_path / "gen"
    gen.mkdir()
    g = [{"id": f"g{i}", "prompt_id": f"p{i // 2}", "cwe_area": "CWE-78", "code": f"y = {'eval' if i % 2 else 'int'}(s)\n",
          "label": i % 2} for i in range(8)] + [{"id": "bad", "prompt_id": "p0", "cwe_area": "CWE-78", "code": "(", "label": None}]
    (gen / "labeled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in g), encoding="utf-8")
    cfg = {"seed": 0, "probe": {"outer_folds": 5, "transfer_train_cwes": ["CWE-78", "CWE-89"]},
           "paths": {"pairs": str(tmp_path / "pairs.jsonl"), "generations": str(gen), "results": str(tmp_path / "res")}}

    def stub_tok(codes, add_special_tokens=False):  # whitespace "tokenizer": CI downloads nothing
        return {"input_ids": [c.split() for c in codes]}

    _load("run_baselines").run(cfg, stub_tok)
    res = tmp_path / "res"
    for m in ("lexical", "length"):
        for es in ("pairs", "transfer", "loco_CWE-22", "loco_CWE-78", "loco_CWE-89"):
            read_scores(res / f"scores_{m}_{es}.csv")  # validates the schema
    assert len(read_scores(res / "scores_lexical_transfer.csv")) == 8  # unparseable row dropped
