import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icp import probe
from icp.dataset import CWES, grouped_folds
from icp.stats import (
    SCORES_COLUMNS,
    auroc,
    auroc_ci,
    read_scores,
    within_group_auroc,
    write_scores,
)

CFG = {
    "seed": 0,
    "probe": {
        "outer_folds": 5, "inner_folds": 4, "select_layer_range": [0.25, 0.75], "C": 1.0, "max_iter": 2000,
        "transfer_train_cwes": sorted(CWES - {"CWE-22"}),
    },
    "stats": {"n_boot": 200},
}
N_HIDDEN, D = 9, 6  # 8 layers + embeddings; in-range layers 2..6


def make_df(n_tmpl=5, n_var=2) -> pd.DataFrame:
    rows = []
    for cwe in sorted(CWES):
        for t in range(n_tmpl):
            for v in range(n_var):
                for label in (0, 1):
                    tid = f"{cwe}-t{t}"
                    rows.append({"id": f"{tid}-v{v}-{label}", "pair_id": f"{tid}-v{v}", "template_id": tid,
                                 "cwe": cwe, "label": label})
    return pd.DataFrame(rows)


def make_x(df, signal_layer=None, seed=1) -> np.ndarray:
    X = np.random.default_rng(seed).normal(size=(len(df), N_HIDDEN, D))
    if signal_layer is not None:
        X[:, signal_layer, 0] += 3 * (2 * df["label"].to_numpy() - 1)
    return X


def test_layer_range_is_inclusive_fraction_of_n_layers():
    assert probe.layer_range(25, [0.25, 0.75]) == list(range(6, 19))
    assert probe.layer_range(N_HIDDEN, [0.25, 0.75]) == [2, 3, 4, 5, 6]


def test_nested_finds_planted_layer():
    df = make_df()
    scores, chosen = probe.nested_scores(make_x(df, signal_layer=4), df, CFG, 0)
    assert chosen == [4] * 5
    assert auroc(df["label"], scores) > 0.9


def test_nested_no_signal_ci_covers_chance():
    df = make_df()
    scores, _ = probe.nested_scores(make_x(df), df, CFG, 0)
    ci = auroc_ci(df["label"], scores, df["template_id"], CFG["stats"]["n_boot"], 0)
    assert ci.lo <= 0.5 <= ci.hi


def test_nested_selection_never_sees_outer_test_fold():
    df = make_df()
    X = make_x(df, signal_layer=4)
    _, chosen = probe.nested_scores(X, df, CFG, 0)
    for k, (_, te) in enumerate(grouped_folds(df, CFG["probe"]["outer_folds"], 0)):
        X2 = X.copy()
        X2[te] = make_x(df.iloc[te], signal_layer=2, seed=7)  # a stronger-looking layer only in this test fold
        X2[te, 2, 0] *= 10
        assert probe.nested_scores(X2, df, CFG, 0)[1][k] == chosen[k]


def test_transfer_layer_uses_pairs_only():
    df = make_df()
    X = make_x(df, signal_layer=5)
    assert list(inspect.signature(probe.transfer_layer).parameters) == ["X", "df", "cfg", "seed"]  # no generations
    layer = probe.transfer_layer(X, df, CFG, 0)
    gen = make_x(pd.DataFrame({"label": [0, 1] * 15}), signal_layer=5, seed=2)
    assert layer == 5
    assert auroc([0, 1] * 15, probe.transfer_scores(X, df, gen, layer, CFG, 0)) > 0.9


def test_topic_prior_and_within_area():
    gen = pd.DataFrame({"prompt_id": ["a1", "a2", "b1", "b2"] * 2, "cwe_area": ["A", "A", "B", "B"] * 2,
                        "label": [1, 1, 0, 0] * 2})
    assert auroc(gen["label"], probe.topic_prior(gen)) == 1.0
    gen["natural"] = ["risky", "risky", "safe", "safe"] * 2
    assert auroc(gen["label"], probe.topic_prior(gen, by="natural")) == 1.0
    # areas A and B are single-class, so only C counts
    strata = ["A", "A", "B", "B", "C", "C"]
    _, n_used = within_group_auroc([1, 1, 0, 0, 1, 0], [0.9, 0.8, 0.1, 0.2, 0.7, 0.3], strata)
    assert n_used == 1


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_probes.py"
    spec = importlib.util.spec_from_file_location("run_probes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(tmp_path, pooling, gen_labels) -> tuple[dict, Path]:
    """Synthetic pairs, generations, prompts, activations (signal in layer 4) and lexical scores."""
    df = make_df()
    acts, gens, res = tmp_path / "acts", tmp_path / "gens", tmp_path / "res"
    for d in (acts, gens, res):
        d.mkdir()
    with open(tmp_path / "pairs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({**r, "code": f"x = {r['id']!r}\n", "variation": {}, "hard": None,
                                "expected_bandit": []}) + "\n" for r in df.to_dict("records"))
    order = df["id"].tolist()[::-1]  # activation rows in a different order than pairs.jsonl
    gen = pd.DataFrame({"id": [f"g{i}" for i in range(48)], "prompt_id": [f"p{i // 4}" for i in range(48)],
                        "cwe_area": [f"A{i // 16}" for i in range(48)], "label": gen_labels})
    for k, pool in enumerate(pooling):
        np.save(acts / f"pairs_{pool}.npy", make_x(df.set_index("id").loc[order].reset_index(), 4, seed=1 + k))
        np.save(acts / f"transfer_{pool}.npy", make_x(gen, 4, seed=11 + k))
    (acts / "pairs_meta.json").write_text(json.dumps({"ids": order}), encoding="utf-8")
    (acts / "transfer_meta.json").write_text(json.dumps({"ids": gen["id"].tolist()}), encoding="utf-8")
    bad = {"id": "bad", "prompt_id": "p0", "cwe_area": "A0", "label": None}  # unparseable: dropped
    with open(gens / "labeled.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in [*gen.to_dict("records"), bad])
    with open(tmp_path / "prompts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"prompt_id": f"p{i}", "natural": ["risky", "safe"][i % 2]}) + "\n"
                     for i in range(12))
    lex = pd.DataFrame({"id": df["id"], "group": df["template_id"], "stratum": df["cwe"], "label": df["label"],
                        "score": np.random.default_rng(3).random(len(df)), "seed": 0})
    write_scores(lex, res / "scores_lexical_pairs.csv")
    paths = {"pairs": tmp_path / "pairs.jsonl", "prompts": tmp_path / "prompts.jsonl", "activations": acts,
             "generations": gens, "results": res}
    return {**CFG, "extract": {"pooling": pooling}, "paths": {k: str(v) for k, v in paths.items()}}, res


def test_run_probes_end_to_end(tmp_path, monkeypatch):
    cfg, res = _setup(tmp_path, ["mean", "last"], [0, 1] * 24)
    mod = _load_script()
    seen = {"nested_scores": [], "transfer_layer": [], "transfer_scores": []}
    for name in seen:
        real = getattr(mod, name)

        def spy(X, d, *a, _real=real, _name=name, **k):
            seen[_name].append("CWE-22" in set(d["cwe"]))
            return _real(X, d, *a, **k)

        monkeypatch.setattr(mod, name, spy)
    mod.run(cfg)

    # Per pooling, each spy sees an 8-CWE and a 7-CWE training frame; the 7-CWE ones (H2 in-dist
    # side, headline transfer layer and headline fit) contain no CWE-22 row.
    assert seen["nested_scores"] == [True, False] * 2
    assert seen["transfer_layer"] == [False, True] * 2
    assert seen["transfer_scores"] == [False, True] * 2

    out = pd.read_csv(res / "probe_layers.csv")
    assert out[["n_pos", "n_neg", "n_redraw", "seed"]].notna().all().all()
    assert set(out.pooling) == {"mean", "last"}
    head = out[(out.method == "probe") & (out.eval_set == "transfer") & (out.note == "selected")]
    assert list(head.layer) == [4, 4] and (head.auroc > 0.9).all()
    sel = json.loads((res / "selected_layers.json").read_text())
    assert sel["transfer"] == sel["transfer_last"] == 4
    assert out[out.method.str.startswith("probe_nested") & (out.eval_set == "pairs")].layer.isna().all()
    assert len(list(res.glob("scores_probe_nested_loco_CWE-*.csv"))) == 2 * len(CWES)  # mean + _last
    assert (out[out.eval_set.str.endswith("CWE-22")].note == "pairs_only").all()
    assert set(out.method) >= {"probe", "probe_nested", "probe_nested_7cwe", "probe_layer0", "probe_shuffled",
                               "probe_8cwe", "topic_prior", "prompt_risky_prior", "prompt_risky_rate"}
    assert {"transfer_risky", "transfer_safe"} <= set(out.eval_set)
    shuffled = out[(out.method == "probe_shuffled") & (out.eval_set == "pairs")]
    assert len(shuffled) == 2 * N_HIDDEN and (shuffled.auroc < 0.75).all()  # within-pair shuffle removes the signal
    for name in ["probe_transfer_last", "probe_shuffled_pairs", "probe_shuffled_pairs_last"]:
        assert (res / f"scores_{name}.csv").exists()
    for f in res.glob("scores_*.csv"):
        assert list(pd.read_csv(f).columns) == SCORES_COLUMNS
    risky = pd.read_csv(res / "scores_prompt_risky_prior_transfer.csv")
    assert set(risky.score) == {0.0, 1.0}  # plain indicator, no labels used
    diffs = pd.read_csv(res / "paired_diffs.csv")
    assert len(diffs) == 2 and diffs[["n_pos", "n_neg", "n_redraw", "seed"]].notna().all().all()


def test_run_probes_single_class_transfer_keeps_outputs(tmp_path):
    cfg, res = _setup(tmp_path, ["mean"], [0] * 48)
    _load_script().run(cfg)
    out = pd.read_csv(res / "probe_layers.csv")
    assert len(out) > 0 and not out.eval_set.str.startswith("transfer").any()
    assert json.loads((res / "selected_layers.json").read_text())["transfer"] == 4  # chosen without generations
    assert (res / "scores_probe_shuffled_pairs.csv").exists()


def test_paired_diff_rejects_mismatched_lexical_labels(tmp_path):
    cfg, res = _setup(tmp_path, ["mean"], [0, 1] * 24)
    lex = read_scores(res / "scores_lexical_pairs.csv")
    lex["label"] = 1 - lex["label"]
    write_scores(lex, res / "scores_lexical_pairs.csv")
    with pytest.raises(ValueError, match="labels do not match"):
        _load_script().run(cfg)
    assert (res / "probe_layers.csv").exists()  # rows computed before the failure are kept
