"""Transfer-set generation (design #1, ticket #10): sample code, parse it, store it. Never run it."""

import ast
import hashlib
import json
import os
import re
from pathlib import Path

FENCE_LINE = re.compile(r"^```[ \t]*(\w*)[ \t]*$")
PYTHON_TAGS = {"", "python", "py", "python3"}


def code_blocks(text: str) -> list[tuple[str, str, bool]]:
    """Fenced blocks as (tag, body, closed). Fences count only at line start and any line starting
    with ``` closes the block (limitation: triple backticks inside code end it early). One matcher
    for both extraction and the generation stop criterion."""
    blocks, cur = [], None
    for line in text.replace("\r\n", "\n").split("\n"):
        if cur is None:
            m = FENCE_LINE.match(line)
            if m:
                cur = (m.group(1).lower(), [])
        elif line.startswith("```"):
            blocks.append((cur[0], "\n".join(cur[1]), True))
            cur = None
        else:
            cur[1].append(line)
    if cur is not None:
        blocks.append((cur[0], "\n".join(cur[1]), False))
    return blocks


def extract_code(raw: str) -> str:
    """Body of the first python/py/untagged block (an unterminated one, cut off by max_new_tokens,
    counts). No fences at all: the whole output. Only non-python fences (bash, sh, ...): no code."""
    blocks = code_blocks(raw)
    if not blocks:
        return raw.strip()
    return next((body.strip() for tag, body, _ in blocks if tag in PYTHON_TAGS), "")


def fence_closed(texts: list[str]) -> list[bool]:
    """Per sequence: has a python/untagged block opened and closed? (stop criterion for generation)"""
    return [any(c and tag in PYTHON_TAGS for tag, _, c in code_blocks(t)) for t in texts]


def parse_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    return bool(code.strip())


def prompt_seed(seed: int, prompt_id: str) -> int:
    """Stable per-prompt seed from (config seed, prompt_id); differs across prompts."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{prompt_id}".encode()).digest()[:4], "big")


def load_prompts(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(prompt: str, model, tok, g: dict, seed: int) -> list[str]:
    """`samples_per_prompt` completions of one prompt through the chat template, one batched call."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    n = ids["input_ids"].shape[1]

    class FenceEnd(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return torch.tensor(fence_closed(tok.batch_decode(input_ids[:, n:])))

    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(
            **ids, do_sample=True, temperature=g["temperature"], top_p=g["top_p"],
            max_new_tokens=g["max_new_tokens"], num_return_sequences=g["samples_per_prompt"],
            pad_token_id=tok.pad_token_id, stopping_criteria=StoppingCriteriaList([FenceEnd()]))
    return tok.batch_decode(out[:, n:], skip_special_tokens=True)


def _read_rows(path: Path) -> list[dict]:
    """Complete JSON lines only: a trailing partial line from a crashed run is dropped."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run(prompts: list[dict], model, tok, cfg: dict, out_path: str | Path) -> int:
    """Append rows to generations.jsonl one prompt at a time. Resume keeps only prompts that have
    exactly samples_per_prompt rows (partial ones are regenerated) and refuses changed settings."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"seed": cfg["seed"], "model": cfg.get("model", {}).get("revision"), "generate": cfg["generate"]}
    meta = out.with_name(out.stem + ".meta.json")
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")) != settings:
        raise ValueError(f"generate settings differ from those in {meta.name}; use a new output directory")
    meta.write_text(json.dumps(settings, indent=1), encoding="utf-8", newline="\n")
    k = cfg["generate"]["samples_per_prompt"]
    keep, done = [], set()
    if out.exists():
        by_prompt: dict[str, list[dict]] = {}
        for r in _read_rows(out):
            by_prompt.setdefault(r["prompt_id"], []).append(r)
        for pid, rs in by_prompt.items():
            if len(rs) == k:
                keep += rs
                done.add(pid)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8", newline="\n")
        os.replace(tmp, out)  # drops partial prompts and any torn last line
    n = 0
    for p in prompts:
        if p["prompt_id"] in done:
            continue
        seed = prompt_seed(cfg["seed"], p["prompt_id"])
        raws = sample(p["prompt"], model, tok, cfg["generate"], seed)
        rows = []
        for i, raw in enumerate(raws):
            code = extract_code(raw)
            rows.append({"id": f"{p['prompt_id']}-s{i}", "prompt_id": p["prompt_id"], "cwe_area": p["cwe_area"],
                         "prompt": p["prompt"], "sample_idx": i, "seed": seed, "raw_output": raw,
                         "code": code, "parse_ok": parse_ok(code)})
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        n += len(rows)
    return n
