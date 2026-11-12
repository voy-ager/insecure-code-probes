"""Activation extraction (design #1, section 4): safe loading, per-layer residuals, pooling, cache."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import tokenizers
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

POOLING_VERSION = 2  # bump on any change to masking/pooling so cached arrays are recomputed
NOTES = {
    "layers": "index 0 = embeddings; in HF Qwen2 the last index is after the final RMSNorm, "
              "the others are raw residual-stream outputs",
    "truncated": "for truncated rows, `last` is the token at max_len, not the end of the code",
}


def load_model(cfg: dict):
    """Model and tokenizer at the pinned revision; safetensors only, no remote code. CPU, eval."""
    m = cfg["model"]
    safe = {"use_safetensors": True, "trust_remote_code": False}  # revision explicit for Bandit B615
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], **safe)
    model = AutoModelForCausalLM.from_pretrained(
        m["id"], revision=m["revision"], dtype=getattr(torch, m["dtype"]), **safe)
    return model.eval(), tok


def extract(codes: list[str], ids: list[str], model, tok, cfg: dict) -> tuple[dict, dict]:
    """Pooled hidden states per layer: {pool: [N, n_layers + 1, d]} in `store_dtype` (see NOTES).

    Tokens excluded from pooling: padding and every added token (special or not, e.g. the Coder
    FIM / file_sep / tool_call tokens), plus the first kept token of `mean` if `exclude_first_token`.
    Raises ValueError, naming the ids, if any row has nothing left to pool (e.g. a single-token
    snippet with `exclude_first_token`; filter such rows upstream) or if the stored values overflow.
    Also returns info: truncated ids, ids whose code contains added-token text, max |activation|.
    """
    x = cfg["extract"]
    max_len, bs, drop_first = x["max_len"], x["batch_size"], x["exclude_first_token"]
    store = np.dtype(x["store_dtype"])
    # added_tokens_decoder covers all control tokens; special_tokens_mask misses those in the text
    masked = torch.tensor(sorted(set(tok.added_tokens_decoder) | set(tok.all_special_ids)))
    control = [i for i, c in zip(ids, codes)
               if any(t.content in c for t in tok.added_tokens_decoder.values())]
    if control:
        log.warning("code contains added-token text (masked from pooling): %s", control)
    full = tok(codes)["input_ids"]
    truncated = [i for i, t in zip(ids, full) if len(t) > max_len]
    if truncated:
        log.warning("truncated to %d tokens: %s", max_len, truncated)
    n_kept = [int((~torch.isin(torch.tensor(t[:max_len], dtype=torch.long), masked)).sum()) for t in full]
    empty = [i for i, n in zip(ids, n_kept) if n < 1 + drop_first]
    if empty:
        raise ValueError(f"no tokens left to pool (exclude_first_token={drop_first}): {empty}")
    out = {p: [] for p in x["pooling"]}
    for s in range(0, len(codes), bs):
        enc = tok(codes[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, padding_side="right")
        keep = enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], masked)
        rows = torch.arange(len(keep))
        with torch.inference_mode():
            hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       output_hidden_states=True).hidden_states
        h = torch.stack(hs, 1).float()  # [B, L+1, T, d]
        if "mean" in out:
            m = keep.clone()
            if drop_first:  # Qwen2-family first-position activations are huge outliers
                m[rows, m.int().argmax(1)] = False
            out["mean"].append((h * m[:, None, :, None]).sum(2) / m.sum(1)[:, None, None])
        if "last" in out:
            last = (keep * torch.arange(keep.shape[1])).argmax(1)  # last kept token
            out["last"].append(h[rows, :, last])
    arrays = {p: torch.cat(v).numpy().astype(store) for p, v in out.items()}
    for p, a in arrays.items():
        bad = ~np.isfinite(a).all(axis=(1, 2))
        if bad.any():
            raise ValueError(f"{p} overflows {store}: {[i for i, b in zip(ids, bad) if b]}")
    info = {
        "truncated": truncated,
        "n_truncated": len(truncated),
        "control_token_ids": control,
        "max_abs_activation": {p: float(np.abs(a.astype(np.float32)).max()) for p, a in arrays.items()},
    }
    return arrays, info


def extract_cached(name: str, codes: list[str], ids: list[str], cfg: dict, load=load_model) -> Path:
    """Write `<name>_<pool>.npy` + `<name>_meta.json` under paths.activations; skip if meta matches."""
    if not codes:
        raise ValueError(f"no rows to extract for {name!r}")
    out = Path(cfg["paths"]["activations"])
    x = cfg["extract"]
    key = {
        "ids": ids,
        "model": cfg["model"]["id"],
        "revision": cfg["model"]["revision"],
        "data_sha256": hashlib.sha256(json.dumps([ids, codes]).encode()).hexdigest(),
        "compute_dtype": cfg["model"]["dtype"],
        "store_dtype": x["store_dtype"],
        "max_len": x["max_len"],
        "exclude_first_token": x["exclude_first_token"],
        "pooling": x["pooling"],
        "pooling_version": POOLING_VERSION,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
    }
    meta_path = out / f"{name}_meta.json"
    files = [out / f"{name}_{p}.npy" for p in x["pooling"]]
    if meta_path.exists() and all(f.exists() for f in files):
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in key} == key:
            log.info("cache hit: %s", meta_path)
            return out
    model, tok = load(cfg)
    arrays, info = extract(codes, ids, model, tok, cfg)
    out.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # no stale meta next to half-written arrays
    for p, a in arrays.items():
        np.save(out / f"{name}_{p}.npy", a)
    meta_path.write_text(json.dumps({**key, **info, "notes": NOTES}, indent=1), encoding="utf-8")
    return out
