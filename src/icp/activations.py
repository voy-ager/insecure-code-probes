"""Activation extraction (design #1, section 4): safe loading, per-layer residuals, pooling, cache."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def load_model(cfg: dict):
    """Model and tokenizer at the pinned revision; safetensors only, no remote code. CPU, eval."""
    m = cfg["model"]
    safe = {"use_safetensors": True, "trust_remote_code": False}  # revision explicit for Bandit B615
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], **safe)
    model = AutoModelForCausalLM.from_pretrained(
        m["id"], revision=m["revision"], dtype=getattr(torch, m["dtype"]), **safe)
    return model.eval(), tok


def extract(codes: list[str], ids: list[str], model, tok, cfg: dict) -> tuple[dict, list[str]]:
    """Pooled residual stream per layer: {pool: float16 [N, n_layers + 1, d]} (index 0 = embeddings).

    Also returns the ids truncated at `max_len`.
    """
    x = cfg["extract"]
    max_len, bs, drop_first = x["max_len"], x["batch_size"], x["exclude_first_token"]
    lens = [len(t) for t in tok(codes)["input_ids"]]
    truncated = [i for i, n in zip(ids, lens) if n > max_len]
    if truncated:
        log.warning("truncated to %d tokens: %s", max_len, truncated)
    out = {p: [] for p in x["pooling"]}
    for s in range(0, len(codes), bs):
        enc = tok(codes[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, padding_side="right")
        # by id: special_tokens_mask misses special tokens that appear in the code text itself
        special = torch.isin(enc["input_ids"], torch.tensor(tok.all_special_ids))
        keep = enc["attention_mask"].bool() & ~special
        pos = torch.arange(keep.shape[1])
        with torch.inference_mode():
            hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       output_hidden_states=True).hidden_states
        h = torch.stack(hs, 1).float()  # [B, L+1, T, d]
        if "mean" in out:
            m = keep.clone()
            if drop_first:  # Qwen2-family first-position activations are huge outliers
                m[torch.arange(len(m)), m.int().argmax(1)] = False
            n = m.sum(1)
            if (n == 0).any():
                raise ValueError(f"no tokens left to mean-pool: {ids[s:s + bs]}")
            mean = (h * m[:, None, :, None]).sum(2) / n[:, None, None]
            out["mean"].append(mean.to(torch.float16).numpy())
        if "last" in out:
            last = (keep * pos).argmax(1)  # last kept (non-pad, non-special) token
            out["last"].append(h[torch.arange(len(h)), :, last].to(torch.float16).numpy())
    return {p: np.concatenate(v) for p, v in out.items()}, truncated


def extract_cached(name: str, codes: list[str], ids: list[str], cfg: dict, load=load_model) -> Path:
    """Write `<name>_<pool>.npy` + `<name>_meta.json` under paths.activations; skip if meta matches."""
    out = Path(cfg["paths"]["activations"])
    x = cfg["extract"]
    key = {
        "ids": ids,
        "model": cfg["model"]["id"],
        "revision": cfg["model"]["revision"],
        "data_sha256": hashlib.sha256(json.dumps([ids, codes]).encode()).hexdigest(),
        "compute_dtype": cfg["model"]["dtype"],
        "store_dtype": "float16",
        "max_len": x["max_len"],
        "exclude_first_token": x["exclude_first_token"],
        "pooling": x["pooling"],
    }
    meta_path = out / f"{name}_meta.json"
    files = [out / f"{name}_{p}.npy" for p in x["pooling"]]
    if meta_path.exists() and all(f.exists() for f in files):
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        old.pop("truncated", None)
        if old == key:
            log.info("cache hit: %s", meta_path)
            return out
    model, tok = load(cfg)
    arrays, truncated = extract(codes, ids, model, tok, cfg)
    out.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # no stale meta next to half-written arrays
    for p, a in arrays.items():
        np.save(out / f"{name}_{p}.npy", a)
    meta_path.write_text(json.dumps({**key, "truncated": truncated}, indent=1), encoding="utf-8")
    return out
