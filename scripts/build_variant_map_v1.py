from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from text_norm_v1 import norm_ar, norm_lat, sort_casefold


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _is_arabic_token(token: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in token)


def _clean_token(token: str) -> str:
    return str(token).strip()


def _extract_lat_tokens(text: str) -> Iterable[str]:
    import re

    for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or ""):
        yield m.group(0)


def _extract_ar_tokens(text: str) -> Iterable[str]:
    import re

    for m in re.finditer(r"[\u0600-\u06FF]+", text or ""):
        yield m.group(0)


def build_variant_map(
    verses: List[Dict[str, Any]],
    seed: Dict[str, List[str]],
    max_per_key: int,
) -> Dict[str, List[str]]:
    variants: Dict[str, List[str]] = {}
    seen: Dict[str, Set[str]] = {}

    def add_variant(key: str, value: str) -> None:
        k = _clean_token(key)
        v = _clean_token(value)
        if not k or not v or v == k:
            return
        s = seen.setdefault(k, set())
        if v in s:
            return
        s.add(v)
        variants.setdefault(k, []).append(v)

    # 1) Seed variants
    canonical_keys: Set[str] = set()
    for raw_key, raw_variants in seed.items():
        key = _clean_token(raw_key)
        if not key:
            continue
        if _is_arabic_token(key):
            canon = norm_ar(key)
        else:
            canon = norm_lat(key)
        if not canon:
            continue
        canonical_keys.add(canon)
        for v in raw_variants or []:
            add_variant(canon, v)

    # 2) Rule-based transliteration variants (latin-only, minimal list)
    lat_rules = {
        "ibrahim": ["ibraheem"],
        "musa": ["moosa"],
        "lut": ["lot"],
        "zakat": ["zakah", "zakaat"],
        "salah": ["salat", "salaah"],
        "dua": ["du'a", "duaa"],
        "iman": ["eeman"],
    }
    for canon, vals in lat_rules.items():
        if canon in canonical_keys:
            for v in vals:
                add_variant(canon, v)

    # 3) Arabic surface-form mining for Arabic canonicals
    arabic_canon = {k for k in canonical_keys if _is_arabic_token(k)}
    if arabic_canon:
        for verse in verses:
            ar = str(verse.get("arabic") or "")
            for tok in _extract_ar_tokens(ar):
                tok_norm = norm_ar(tok)
                if tok_norm in arabic_canon:
                    add_variant(tok_norm, tok)

    # 4) Mining from searchable_text (latin + arabic tokens)
    if canonical_keys:
        latin_canon = {k for k in canonical_keys if not _is_arabic_token(k)}
        for verse in verses:
            st = str(verse.get("searchable_text") or "")
            if latin_canon:
                for tok in _extract_lat_tokens(st):
                    tok_norm = norm_lat(tok)
                    if tok_norm in latin_canon:
                        add_variant(tok_norm, tok)
            if arabic_canon:
                for tok in _extract_ar_tokens(st):
                    tok_norm = norm_ar(tok)
                    if tok_norm in arabic_canon:
                        add_variant(tok_norm, tok)

    # 5) Finalize: unique + sorted + capped
    out: Dict[str, List[str]] = {}
    for key in sorted(canonical_keys, key=lambda s: s.casefold()):
        vals = variants.get(key, [])
        uniq_sorted = sort_casefold({v for v in vals if v})
        if max_per_key > 0:
            uniq_sorted = uniq_sorted[:max_per_key]
        out[key] = uniq_sorted

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="Path to quran_complete.json")
    ap.add_argument("--seed", dest="seed_path", required=True, help="Path to seed_variants_v1.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output variant_map_v1.json path")
    ap.add_argument("--max-per-key", type=int, default=12)
    args = ap.parse_args()

    input_path = Path(args.input_path)
    seed_path = Path(args.seed_path)
    out_path = Path(args.out_path)

    verses = _load_json(input_path)
    if not isinstance(verses, list) or not verses:
        raise ValueError("quran_complete.json must be a non-empty list")

    seed = _load_json(seed_path)
    if not isinstance(seed, dict):
        raise ValueError("seed_variants_v1.json must be an object")

    variants = build_variant_map(verses, seed, int(args.max_per_key))

    payload = {
        "schema": "askquran_variant_map_v1",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": "ie-ai-py",
            "input": str(input_path.name),
        },
        "variants": variants,
    }

    _write_json(out_path, payload)


if __name__ == "__main__":
    main()
