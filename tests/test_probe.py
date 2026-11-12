import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from icp import probe
from icp.dataset import CWES, grouped_folds
from icp.stats import SCORES_COLUMNS, auroc, auroc_ci, within_group_auroc, write_scores

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
    te = grouped_folds(df, CFG["probe"]["outer_folds"], 0)[0][1]
    X2 = X.copy()
    X2[te] = make_x(df.iloc[te], signal_layer=2, seed=7)  # a different, stronger-looking layer only in the test fold
    X2[te, 2, 0] *= 10
    assert probe.nested_scores(X2, df, CFG, 0)[1][0] == chosen[0]


def test_transfer_scores_cannot_change_the_layer():
    df = make_df()
    X = make_x(df, signal_layer=5)
    gen_a = np.random.default_rng(2).normal(size=(30, N_HIDDEN, D))
    gen_b = make_x(pd.DataFrame({"label": [0, 1] * 15}), signal_layer=2)
    (_, la), (_, lb) = probe.transfer_scores(X, df, gen_a, CFG, 0), probe.transfer_scores(X, df, gen_b, CFG, 0)
    assert la == lb == 5


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


def test_run_probes_end_to_end(tmp_path, monkeypatch):
    df = make_df()
    acts, gens, res = tmp_path / "acts", tmp_path / "gens", tmp_path / "res"
    for d in (acts, gens, res):
        d.mkdir()
    with open(tmp_path / "pairs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({**r, "code": f"x = {r['id']!r}\n", "variation": {}, "hard": None,
                                "expected_bandit": []}) + "\n" for r in df.to_dict("records"))
    order = df["id"].tolist()[::-1]  # activation rows in a different order than pairs.jsonl
    np.save(acts / "pairs_mean.npy", make_x(df.set_index("id").loc[order].reset_index(), signal_layer=4))
    (acts / "pairs_meta.json").write_text(json.dumps({"ids": order}), encoding="utf-8")
    gen = pd.DataFrame({"id": [f"g{i}" for i in range(48)], "prompt_id": [f"p{i // 4}" for i in range(48)],
                        "cwe_area": [f"A{i // 16}" for i in range(48)], "label": [0, 1] * 24})
    with open(gens / "labeled.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in gen.to_dict("records") + [{"id": "bad", "prompt_id": "p0", "cwe_area": "A0", "label": None}])
    with open(tmp_path / "prompts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"prompt_id": f"p{i}", "natural": ["risky", "safe"][i % 2]}) + "\n"
                     for i in range(12))
    np.save(acts / "transfer_mean.npy", make_x(gen, signal_layer=4))
    (acts / "transfer_meta.json").write_text(json.dumps({"ids": gen["id"].tolist()}), encoding="utf-8")
    lex = pd.DataFrame({"id": df["id"], "group": df["template_id"], "stratum": df["cwe"], "label": df["label"],
                        "score": np.random.default_rng(3).random(len(df)), "seed": 0})
    write_scores(lex, res / "scores_lexical_indist_cv.csv")
    cfg = {**CFG, "extract": {"pooling": ["mean"]},
           "paths": {"pairs": str(tmp_path / "pairs.jsonl"), "prompts": str(tmp_path / "prompts.jsonl"),
                      "activations": str(acts), "generations": str(gens),
                     "results": str(res)}}

    mod = _load_script()
    seen = {"nested": [], "transfer": []}
    for name, key in (("nested_scores", "nested"), ("transfer_scores", "transfer")):
        real = getattr(mod, name)

        def spy(X, d, *a, _real=real, _key=key, **k):
            seen[_key].append(set(d["cwe"]))
            return _real(X, d, *a, **k)

        monkeypatch.setattr(mod, name, spy)
    mod.run(cfg)

    # nested calls: 8-CWE H1, shuffled, 7-CWE; transfer calls: headline, then secondary 8-CWE
    assert "CWE-22" not in seen["nested"][2] and "CWE-22" not in seen["transfer"][0]
    assert "CWE-22" in seen["transfer"][1] and "CWE-22" in seen["nested"][0]

    out = pd.read_csv(res / "probe_layers.csv")
    assert out[["n_pos", "n_neg", "n_redraw", "seed"]].notna().all().all()
    head = out[(out.method == "probe") & (out.eval_set == "transfer") & (out.note == "selected")]
    assert len(head) == 1 and head.layer.iloc[0] == "4" and head.auroc.iloc[0] > 0.9
    assert (out[out.eval_set.str.endswith("CWE-22")].note == "pairs_only").all()
    assert set(out.method) >= {"probe", "probe_nested", "probe_nested_7cwe", "probe_layer0", "probe_shuffled",
                               "probe_8cwe", "topic_prior", "prompt_risky_prior"}
    assert {"transfer_risky", "transfer_safe"} <= set(out.eval_set)
    shuffled = out[out.method == "probe_shuffled"].auroc.iloc[0]
    assert shuffled < 0.75  # within-pair shuffle removes the planted signal
    for f in res.glob("scores_*.csv"):
        assert list(pd.read_csv(f).columns) == SCORES_COLUMNS
    assert len(pd.read_csv(res / "paired_diffs.csv")) == 1

