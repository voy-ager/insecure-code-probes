import json
import re
from pathlib import Path

import pytest

from icp.generate import (
    extract_code,
    fence_closed,
    load_prompts,
    parse_ok,
    prompt_seed,
    run,
)

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = load_prompts(ROOT / "data" / "prompts" / "transfer_prompts.jsonl")
CWES = {"CWE-78", "CWE-89", "CWE-94", "CWE-502", "CWE-327", "CWE-798", "CWE-295"}  # no CWE-22


def test_prompts_cover_exactly_the_seven_bandit_cwes():
    areas = {}
    for p in PROMPTS:
        areas.setdefault(p["cwe_area"], []).append(p["natural"])
    assert set(areas) == CWES
    for a, nat in areas.items():
        assert len(nat) == 4 and nat.count("risky") == 2 and nat.count("safe") == 2, a


def test_prompt_ids_unique_and_wording_neutral():
    ids = [p["prompt_id"] for p in PROMPTS]
    assert len(ids) == len(set(ids))
    assert not any(re.search(r"secur|insecur|vulnerab|unsafe", p["prompt"], re.IGNORECASE) for p in PROMPTS)


def test_seeds_differ_across_prompts_and_are_stable():
    seeds = [prompt_seed(0, p["prompt_id"]) for p in PROMPTS]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [prompt_seed(0, p["prompt_id"]) for p in PROMPTS]
    assert prompt_seed(0, "a") != prompt_seed(1, "a")


@pytest.mark.parametrize("raw, code", [
    ("Sure:\n```python\nx = 1\n```\nDone.", "x = 1"),
    ("x = 1\ny = 2", "x = 1\ny = 2"),
    ("```python\nA = 1\n```\ntext\n```python\nB = 2\n```", "A = 1"),
    ("```\nz = 3\n```", "z = 3"),
    ("```PY\nz = 3\n```", "z = 3"),
    ("```python\ndef f():\n    return 1", "def f():\n    return 1"),  # cut off before closing fence
    ("", ""),
    ("```bash\npip install x\n```\nThen:\n```python\nimport x\n```", "import x"),
    ("```sh\npip install x\n```\n```py\nimport x\n```", "import x"),
    ("```bash\npip install x\n```\nThen run it.", ""),  # lone non-python block: no code
    ("```python3\nA = 1\n```", "A = 1"),
    ("```python\r\nA = 1\r\nB = 2\r\n```\r\n", "A = 1\nB = 2"),  # CRLF
    ("Use `x` and ```inline``` here\n```python\nA = 1\n```", "A = 1"),  # fences only at line start
    ("```bash\nls\n```\n```python\ndef f():\n    return 1", "def f():\n    return 1"),  # python cut off
])
def test_extract_code(raw, code):
    assert extract_code(raw) == code


def test_parse_ok():
    assert parse_ok("x = 1")
    assert not parse_ok("def f(:")
    assert not parse_ok("")
    assert not parse_ok("x = 1\0")  # ast.parse raises ValueError on null bytes
    assert not parse_ok(extract_code("I cannot help with that."))


def test_generate_module_never_executes_code():
    src = (ROOT / "src" / "icp" / "generate.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "generate.py").read_text(encoding="utf-8")
    pat = r"\b(exec|eval|(?<!re.)compile|__import__|import_module|subprocess|os\.system|runpy)\b"
    assert not re.search(pat, src + script)


def test_fence_closed_mixed_batch():
    texts = [
        "```python\nx = 1\n```",  # done
        "```python\nx = 1\n",  # still writing
        "```bash\npip install x\n```\n",  # a single non-python fence: keep going
        "```bash\npip install x\n```\n```py\nx = 1\n```",  # python after bash: done
        "plain text",
    ]
    assert fence_closed(texts) == [True, False, False, True, False]


@pytest.fixture(scope="module")
def tiny():
    """Random-init tiny Qwen2 plus a stub word-level tokenizer with a chat template; nothing downloaded."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["Write", "a", "Python", "function", "that", "parses", "text", "```", "x", "=", "1", "<|user|>", "<|assistant|>"]
    vocab = {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(sorted(set(words)))}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}<|user|> {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=128)
    return transformers.Qwen2ForCausalLM(cfg).eval(), tok


def _cfg(k=2):
    return {"seed": 0, "generate": {"samples_per_prompt": k, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 4}}


PR = [{"prompt_id": f"p{i}", "cwe_area": "CWE-78", "prompt": "Write a Python function", "natural": "safe"}
      for i in (1, 2)]


def test_run_writes_schema_and_is_seed_reproducible(tiny, tmp_path):
    model, tok = tiny
    out = tmp_path / "g" / "generations.jsonl"
    assert run(PR[:1], model, tok, _cfg(3), out) == 3
    assert run(PR, model, tok, _cfg(3), out) == 3  # p1 skipped
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["p1-s0", "p1-s1", "p1-s2", "p2-s0", "p2-s1", "p2-s2"]
    assert set(rows[0]) == {"id", "prompt_id", "cwe_area", "prompt", "sample_idx", "seed", "raw_output",
                            "code", "parse_ok"}
    assert rows[0]["seed"] != rows[3]["seed"]
    out2 = tmp_path / "g2.jsonl"  # same seed -> same samples
    run(PR[:1], model, tok, _cfg(3), out2)
    again = [json.loads(line) for line in out2.read_text(encoding="utf-8").splitlines()]
    assert [r["raw_output"] for r in again] == [r["raw_output"] for r in rows[:3]]


def test_resume_after_torn_line_and_partial_prompt(tiny, tmp_path):
    model, tok = tiny
    out = tmp_path / "generations.jsonl"
    assert run(PR, model, tok, _cfg(), out) == 4
    lines = out.read_text(encoding="utf-8").splitlines()
    # crash simulation: p2 lost its second row and the last line is half-written
    out.write_text("\n".join(lines[:3]) + "\n" + lines[3][:20], encoding="utf-8")
    assert run(PR, model, tok, _cfg(), out) == 2  # p2 regenerated, p1 kept
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["p1-s0", "p1-s1", "p2-s0", "p2-s1"]
    assert run(PR, model, tok, _cfg(), out) == 0


def test_resume_refuses_changed_settings(tiny, tmp_path):
    model, tok = tiny
    out = tmp_path / "generations.jsonl"
    run(PR[:1], model, tok, _cfg(2), out)
    with pytest.raises(ValueError, match="differ"):
        run(PR, model, tok, _cfg(3), out)


@pytest.mark.slow
def test_real_model_one_prompt_two_samples(tmp_path, capsys):
    """Real 0.5B Instruct (HF cache, ~1 GB download on first use) through icp.activations.load_model."""
    import time

    from icp import activations, load_config

    cfg = load_config(ROOT / "configs" / "pilot_0.5b.toml")
    cfg["generate"]["samples_per_prompt"] = 2
    model, tok = activations.load_model(cfg)
    t0 = time.time()
    out = tmp_path / "generations.jsonl"
    assert run(PROMPTS[:1], model, tok, cfg, out) == 2
    with capsys.disabled():
        print(f"\nseconds per sample: {(time.time() - t0) / 2:.1f}")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(r["raw_output"] for r in rows)
