"""
Build Count Index Artifacts

Usage:
  python build_count_index.py --data /path/to/quran_complete.json --out count_index.pkl.gz

This creates a gzipped pickle containing token counts + postings.
"""
from __future__ import annotations

import argparse
from count_index import QuranCountIndex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to quran_complete.json")
    p.add_argument("--out", required=True, help="Output path for count_index.pkl.gz")
    p.add_argument("--max_verses", type=int, default=None, help="Optional dev limit")
    args = p.parse_args()

    idx = QuranCountIndex.build_from_quran_json(args.data, max_verses=args.max_verses)
    idx.save(args.out)
    print(f"OK: wrote {args.out}")
    print(f"  verses: {len(idx.verse_keys)}")
    for field, fi in idx.fields.items():
        print(f"  {field:16s} tokens={len(fi.token_counts):6d} postings={len(fi.token_postings):6d}")


if __name__ == "__main__":
    main()
