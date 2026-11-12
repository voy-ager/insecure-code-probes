"""Self-judgment baseline (design #1, H3): the model's own yes/no verdict as a continuous score.

Score = logsumexp(No-variant logits) - logsumexp(Yes-variant logits) at the first answer token,
so higher = "less secure" (label 1 = insecure). Raw logits, not renormalised: P(Yes)+P(No) is
reported as the mass check, and a low mass means the score is not meaningful.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from icp.stats import SCORES_COLUMNS, write_scores

YES = ["Yes", "yes", "YES", " Yes", " yes", " YES"]
NO = ["No", "no", "NO", " No", " no", " NO"]
LOW_MASS = 0.5


def variant_ids(tok, words: list[str]) -> dict[str, int]:
    """Words that encode to exactly one token, deduplicated by id (first spelling wins)."""
    out: dict[str, int] = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out.values():
            out[w] = ids[0]
    if not out:
        raise ValueError(f"no single-token variant among {words}")
    return out


def render(tok, code: str, cfg: dict) -> str:
    """Chat prompt with the configured system prompt; code is cut to extract.max_len tokens."""
    ids = tok(code, add_special_tokens=False)["input_ids"]
    if len(ids) > cfg["extract"]["max_len"]:
        code = tok.decode(ids[: cfg["extract"]["max_len"]])
    j = cfg["judge"]
    msgs = [{"role": "system", "content": j["system_prompt"]},
            {"role": "user", "content": f"{j['prompt']}\n\n```python\n{code}\n```"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score_logits(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]):
    """logits [B, V] -> (score, mass) arrays; mass = P(Yes variants) + P(No variants)."""
    lg = logits.float()
    lse = torch.logsumexp(lg, -1)
    ly, ln = torch.logsumexp(lg[:, yes_ids], -1), torch.logsumexp(lg[:, no_ids], -1)
    return (ln - ly).numpy(), (torch.exp(ly - lse) + torch.exp(ln - lse)).numpy()


def judge(codes: list[str], model, tok, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scores and masses per code, plus the resolved variants."""
    yes, no = variant_ids(tok, YES), variant_ids(tok, NO)
    scores, masses = [], []
    bs = cfg["extract"]["batch_size"]
    for s in range(0, len(codes), bs):
        prompts = [render(tok, c, cfg) for c in codes[s:s + bs]]
        # add_special_tokens=False: the chat template already carries any special tokens
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        pos = (enc["attention_mask"].cumsum(1) - 1).clamp(min=0)  # left padding must not shift positions
        with torch.inference_mode():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           position_ids=pos).logits[:, -1]
        sc, ms = score_logits(logits, list(yes.values()), list(no.values()))
        scores.append(sc)
        masses.append(ms)
    return np.concatenate(scores), np.concatenate(masses), {"yes": yes, "no": no}


def meta(masses: np.ndarray, variants: dict, cfg: dict) -> dict:
    mean = float(masses.mean())
    return {
        "variants": variants,
        "system_prompt": cfg["judge"]["system_prompt"],
        "prompt": cfg["judge"]["prompt"],
        "n": len(masses),
        "mass_mean": mean,
        "mass_p10": float(np.percentile(masses, 10)),
        "low_mass": mean < LOW_MASS,
    }


def judge_set(name: str, df: pd.DataFrame, model, tok, cfg: dict) -> pd.DataFrame:
    """Score `df` (id, template_id, cwe, label, code); write scores_judge_<name>.csv and judge_meta.json."""
    score, mass, variants = judge(df["code"].tolist(), model, tok, cfg)
    out = pd.DataFrame({
        "id": df["id"].to_numpy(), "group": df["template_id"].to_numpy(),
        "stratum": df["cwe"].to_numpy(), "label": df["label"].to_numpy(),
        "score": score, "seed": cfg["seed"],
    })[SCORES_COLUMNS]
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    write_scores(out, res / f"scores_judge_{name}.csv")
    mp = res / "judge_meta.json"
    allm = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    allm[name] = meta(mass, variants, cfg)
    mp.write_text(json.dumps(allm, indent=1), encoding="utf-8")
    return out
