import copy
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # CPU torch arrives in CI with #3
transformers = pytest.importorskip("transformers")

from icp import activations, load_config

ROOT = Path(__file__).resolve().parents[1]
CODES = [
    "import os\nos.system ( cmd )",
    "x = 1",
    "def f ( a , b ) :\n    return a + b + 1",
]
IDS = ["a", "b", "c"]


def tiny_cfg(tmp_path, **extract):
    return {
        "model": {"id": "tiny", "revision": "0" * 40, "dtype": "float32"},
        "paths": {"activations": str(tmp_path / "act")},
        "extract": {"max_len": 64, "batch_size": 2, "exclude_first_token": True,
                    "pooling": ["mean", "last"], **extract},
    }


@pytest.fixture(scope="module")
def tiny():
    """Random-init tiny Qwen2 and a whitespace word-level stub tokenizer; nothing downloaded."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = sorted({w for c in CODES for w in c.split()})
    vocab = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config).eval(), tok


def test_load_model_uses_safe_pinned_flags():
    cfg = load_config(ROOT / "configs" / "pilot_0.5b.toml")
    with mock.patch.object(activations, "AutoTokenizer") as at, \
            mock.patch.object(activations, "AutoModelForCausalLM") as am:
        activations.load_model(cfg)
    for fp in (at.from_pretrained, am.from_pretrained):
        args, kw = fp.call_args
        assert args == (cfg["model"]["id"],)
        assert kw["revision"] == cfg["model"]["revision"]
        assert kw["use_safetensors"] is True
        assert kw["trust_remote_code"] is False
    assert am.from_pretrained.call_args.kwargs["dtype"] is torch.float32


def test_shapes_dtype_and_batch_equals_single(tiny, tmp_path):
    model, tok = tiny
    cfg = tiny_cfg(tmp_path)
    batched, truncated = activations.extract(CODES, IDS, model, tok, cfg)
    assert truncated == []
    for pool, a in batched.items():
        assert a.dtype == np.float16
        assert a.shape == (3, 3, 32)  # [N, n_layers + 1, d]
        single = np.concatenate([activations.extract([c], [i], model, tok, cfg)[0][pool]
                                 for c, i in zip(CODES, IDS)])
        np.testing.assert_allclose(a.astype(np.float32), single.astype(np.float32), atol=1e-4)


@pytest.mark.parametrize("drop_first", [True, False])
def test_mean_and_last_pooling_spans(tiny, tmp_path, drop_first):
    model, tok = tiny
    code = CODES[2]
    out, _ = activations.extract([code], ["c"], model, tok, tiny_cfg(tmp_path, exclude_first_token=drop_first))
    with torch.inference_mode():
        ids = tok([code], return_tensors="pt")["input_ids"]
        h = torch.stack(model(ids, output_hidden_states=True).hidden_states, 1)[0]
    start = 1 if drop_first else 0
    np.testing.assert_allclose(out["mean"][0], h[:, start:].mean(1).half().numpy(), atol=1e-3)
    np.testing.assert_allclose(out["last"][0], h[:, -1].half().numpy(), atol=1e-3)


def test_truncation_is_reported(tiny, tmp_path):
    model, tok = tiny
    out, truncated = activations.extract(CODES, IDS, model, tok, tiny_cfg(tmp_path, max_len=4))
    assert truncated == ["a", "c"]
    assert out["mean"].shape[0] == 3


def test_cache_hit_skips_model_and_change_misses(tiny, tmp_path):
    cfg = tiny_cfg(tmp_path)
    load = mock.Mock(return_value=tiny)
    out = activations.extract_cached("pairs", CODES, IDS, cfg, load=load)
    assert sorted(p.name for p in out.iterdir()) == ["pairs_last.npy", "pairs_mean.npy", "pairs_meta.json"]
    assert np.load(out / "pairs_mean.npy").shape == (3, 3, 32)
    activations.extract_cached("pairs", CODES, IDS, cfg, load=load)
    assert load.call_count == 1
    activations.extract_cached("pairs", CODES[:2] + ["x = 2"], IDS, cfg, load=load)  # data changed
    other = copy.deepcopy(cfg)
    other["model"]["revision"] = "1" * 40
    activations.extract_cached("pairs", CODES, IDS, other, load=load)  # revision changed
    assert load.call_count == 3


@pytest.mark.slow
def test_real_model_pooling_spans():
    """Real Qwen2.5-Coder-0.5B-Instruct at the pinned revision (downloads ~1 GB to the HF cache)."""
    cfg = load_config(ROOT / "configs" / "pilot_0.5b.toml")
    model, tok = activations.load_model(cfg)
    codes = ["import os\n\nos.system(cmd)\n", "x = 1\n", "def f(a, b):\n    return a + b\n"]
    ids = ["s1", "s2", "s3"]
    special = set(tok.all_special_ids)
    assert not any(special & set(t) for t in tok(codes)["input_ids"])  # code alone: no BOS/EOS added
    base, truncated = activations.extract(codes, ids, model, tok, cfg)
    assert truncated == []
    assert base["mean"].shape == (3, model.config.num_hidden_layers + 1, model.config.hidden_size)
    # special tokens in the text (EOS, and the pad token's text, unpadded) are masked out of both poolings
    tail = tok.eos_token + tok.pad_token
    assert {tok.eos_token_id, tok.pad_token_id} <= set(tok(tail)["input_ids"])
    with_special, _ = activations.extract([c + tail for c in codes], ids, model, tok, cfg)
    for pool in ("mean", "last"):
        np.testing.assert_allclose(with_special[pool].astype(np.float32), base[pool].astype(np.float32),
                                   rtol=1e-2, atol=1e-2)
    short = copy.deepcopy(cfg)
    short["extract"]["max_len"] = 5
    assert activations.extract(codes, ids, model, tok, short)[1] == ["s1", "s3"]
