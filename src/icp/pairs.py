"""Render data/pairs/templates/*.json into data/pairs/pairs.jsonl (deterministic, no RNG).

Snippets are only rendered to strings here; they are never imported or executed.
"""

import difflib
import io
import json
import string
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "data/pairs/templates"
OUT = ROOT / "data/pairs/pairs.jsonl"
NAMES = "abc"
MAX_TOKEN_DIFF = 3  # per pair, secure minus insecure; the shorter member gets one pool entry


def count_tokens(code: str) -> int:
    skip = {tokenize.ENCODING, tokenize.ENDMARKER}
    return sum(t.type not in skip for t in tokenize.generate_tokens(io.StringIO(code).readline))


def changed_lines(a: str, b: str) -> int:
    ops = difflib.SequenceMatcher(None, a.splitlines(), b.splitlines(), autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def render(tpl: list[str], names: dict, doc: str | None, helpers: list[str]) -> str:
    out = []
    for line in tpl:
        slot = line.strip()
        if slot == "${helpers}":
            out += helpers
        elif slot == "${doc}":
            if doc:
                out.append(line[: len(line) - len(line.lstrip())] + f'"""{doc}"""')
        else:
            out.append(string.Template(line).substitute(names))
    return "\n".join(out) + "\n"


def build() -> list[dict]:
    pool = json.loads((TEMPLATES / "_helpers.json").read_text(encoding="utf-8"))["helpers"]
    rows, g, pad_next = [], 0, 0
    for path in sorted(TEMPLATES.glob("cwe*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for t in data["templates"]:
            for v in range(6):
                i, with_doc = v % 3, v >= 3
                names = {k: opts[i] for k, opts in t["names"].items()}
                doc = t["doc"] if with_doc else None
                # shared helpers: 0-2 pool entries, same for both members, fixed rotation
                shared_ix = [(g * 3 + k) % len(pool) for k in range((g + g // 6) % 3)]
                shared = [ln for ix in shared_ix for ln in pool[ix]]

                def code(key: str, extra: tuple = (), shared=shared, names=names, doc=doc, t=t) -> str:
                    return render(t[key], names, doc, shared + list(extra))

                sec, ins = code("secure"), code("insecure")
                diff = count_tokens(sec) - count_tokens(ins)
                pad = side = None
                if abs(diff) > MAX_TOKEN_DIFF:
                    short = "insecure" if diff > 0 else "secure"
                    best = None
                    for k in range(len(pool)):  # rotate from where the last pad stopped
                        ix = (pad_next + k) % len(pool)
                        if ix in shared_ix:
                            continue
                        cand = code(short, pool[ix])
                        new = abs(count_tokens(code("secure" if short == "insecure" else "insecure")) - count_tokens(cand))
                        if best is None or new < best[0]:
                            best = (new, ix, cand)
                        if new <= MAX_TOKEN_DIFF:
                            break
                    _, ix, cand = best
                    pad_next = ix + 1
                    pad, side = ix, short
                    sec, ins = (sec, cand) if short == "insecure" else (cand, ins)
                pair_id = f"{t['template_id']}-v{v}"
                var = {"names": NAMES[i], "docstring": with_doc, "helpers": len(shared_ix), "pad": pad, "pad_side": side}
                for label, name, snippet, expected in (
                    (0, "secure", sec, []),
                    (1, "insecure", ins, t["expected_bandit"]),
                ):
                    rows.append({
                        "id": f"{pair_id}-{name}", "pair_id": pair_id, "template_id": t["template_id"],
                        "cwe": data["cwe"], "label": label, "code": snippet, "variation": var,
                        "hard": t["hard"][name], "expected_bandit": expected,
                    })
                g += 1
    return rows


def write(rows: list[dict], path: Path = OUT) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

