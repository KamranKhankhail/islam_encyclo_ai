"""
Build Count Index Artifacts

Defaults to the repo's bundled Qur'an artifacts under output/processed.
Override paths only when regenerating from alternate data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from count_index import QuranCountIndex

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "output" / "processed" / "quran_complete.json"
DEFAULT_OUT_PATH = PROJECT_ROOT / "output" / "processed" / "count_index.pkl.gz"


def main():
    p = argparse.ArgumentParser(description="Build count_index.pkl.gz using local repo data by default.")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to quran_complete.json (defaults to repo artifact).")
    p.add_argument("--out", default=str(DEFAULT_OUT_PATH), help="Output path for count_index.pkl.gz (defaults to repo artifact location).")
    p.add_argument("--max_verses", type=int, default=None, help="Optional dev limit")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idx = QuranCountIndex.build_from_quran_json(args.data, max_verses=args.max_verses)
    idx.save(out_path)
    print(f"OK: wrote {out_path}")
    print(f"  verses: {len(idx.verse_keys)}")
    for field, fi in idx.fields.items():
        print(f"  {field:16s} tokens={len(fi.token_counts):6d} postings={len(fi.token_postings):6d}")


if __name__ == "__main__":
    main()
