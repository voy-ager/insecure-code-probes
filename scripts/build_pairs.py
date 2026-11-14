"""Render data/pairs/templates/*.json into data/pairs/pairs.jsonl (see icp.pairs)."""

from icp.dataset import load_pairs
from icp.pairs import OUT, ROOT, build, write

if __name__ == "__main__":
    write(build())
    df = load_pairs(OUT)  # validates the schema
    print(f"wrote {len(df)} snippets / {df['pair_id'].nunique()} pairs to {OUT.relative_to(ROOT)}")
