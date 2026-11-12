"""Transfer-set generation (design #1, ticket #10): sample code, parse it, store it. Never run it."""

import ast
import hashlib
import json
import re
from pathlib import Path

FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
OPEN_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*)", re.DOTALL | re.IGNORECASE)


def extract_code(raw: str) -> str:
    """First fenced block; an unterminated first fence (cut off by max_new_tokens) counts; else the whole output."""
    m = FENCE.search(raw) or OPEN_FENCE.search(raw)
    return (m.group(1) if m else raw).strip()


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

    class FenceEnd(StoppingCriteria):  # per-sequence stop once a code block has opened and closed
        def __call__(self, input_ids, scores, **kw):
            return torch.tensor([t.count("```") >= 2 for t in tok.batch_decode(input_ids[:, n:])])

    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(
            **ids, do_sample=True, temperature=g["temperature"], top_p=g["top_p"],
            max_new_tokens=g["max_new_tokens"], num_return_sequences=g["samples_per_prompt"],
            pad_token_id=tok.pad_token_id, stopping_criteria=StoppingCriteriaList([FenceEnd()]))
    return tok.batch_decode(out[:, n:], skip_special_tokens=True)


def run(prompts: list[dict], model, tok, cfg: dict, out_path: str | Path) -> int:
    """Append rows to generations.jsonl, one prompt at a time; prompts already present are skipped."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["prompt_id"] for line in f if line.strip()}
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
        with open(out, "a", encoding="utf-8", newline="\n") as f:  # whole prompt written at once = resumable unit
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        n += len(rows)
    return n
