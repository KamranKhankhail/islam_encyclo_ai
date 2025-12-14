"""
Qur'an Count Index (Offline)

Build-time index for fast counts and verse lookups:
- token -> total occurrences across verses
- token -> postings list of verse_ids where token appears

Supports multiple fields:
  - arabic (normalized Arabic script; also covers Urdu script tokens)
  - transliteration (latin)
  - english (latin)
  - urdu (normalized Arabic script)

This index is intentionally simple and deterministic. It is meant to be:
- Small enough to ship inside the app (or generated once and cached).
- Fast to query (O(1) dict lookup + small postings traversal).
"""

from __future__ import annotations

import gzip
import json
import re
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Any

from arabic_normalizer import normalize_for_search, tokenize_arabic


_RE_LATIN_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9']{1,}")


def _is_arabic_script(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            return True
    return False


def _tokenize_latin(text: str) -> List[str]:
    if not text:
        return []
    t = text.lower()
    # keep apostrophes in tokens (e.g., "qur'an")
    return _RE_LATIN_TOKEN.findall(t)


@dataclass
class _FieldIndex:
    token_counts: Dict[str, int]
    token_postings: Dict[str, List[int]]  # verse_ids

    def __init__(self):
        self.token_counts = {}
        self.token_postings = {}

    def add_tokens(self, verse_id: int, tokens: List[str]) -> None:
        if not tokens:
            return
        # counts: include duplicates (exact occurrences)
        for tok in tokens:
            self.token_counts[tok] = self.token_counts.get(tok, 0) + 1

        # postings: per verse unique
        seen = set(tokens)
        for tok in seen:
            self.token_postings.setdefault(tok, []).append(verse_id)


@dataclass
class CountResult:
    token: str
    total: int
    by_field: Dict[str, int]
    verse_keys: List[str]


@dataclass
class QuranCountIndex:
    version: int
    verse_keys: List[str]
    fields: Dict[str, _FieldIndex]

    @classmethod
    def build_from_quran_json(
        cls,
        quran_path: str,
        fields: Optional[List[str]] = None,
        max_verses: Optional[int] = None,
    ) -> "QuranCountIndex":
        """
        Build index from quran_complete.json (list of verse objects).

        Args:
            quran_path: path to JSON array
            fields: which fields to index. Defaults to ["arabic", "transliteration", "english", "urdu"].
            max_verses: optional limit for faster dev.
        """
        if fields is None:
            fields = ["arabic", "transliteration", "english", "urdu"]

        with open(quran_path, "r", encoding="utf-8") as f:
            verses = json.load(f)

        if max_verses is not None:
            verses = verses[: int(max_verses)]

        verse_keys: List[str] = []
        field_indexes: Dict[str, _FieldIndex] = {name: _FieldIndex() for name in fields}

        for vid, v in enumerate(verses):
            vk = v.get("verse_key") or v.get("id")
            if not vk:
                continue
            verse_keys.append(vk)

            if "arabic" in field_indexes:
                ar = v.get("arabic", "")
                ar_norm = normalize_for_search(ar)
                toks = tokenize_arabic(ar_norm) if ar_norm else []
                field_indexes["arabic"].add_tokens(vid, toks)

            if "urdu" in field_indexes:
                ur = v.get("translation_ur_builtin", "")
                ur_norm = normalize_for_search(ur)
                toks = tokenize_arabic(ur_norm) if ur_norm else []
                field_indexes["urdu"].add_tokens(vid, toks)

            if "english" in field_indexes:
                en = v.get("translation_en_builtin", "")
                toks = _tokenize_latin(en)
                field_indexes["english"].add_tokens(vid, toks)

            if "transliteration" in field_indexes:
                tr = " ".join([v.get("transliteration", ""), v.get("transliteration_alt", "")]).strip()
                toks = _tokenize_latin(tr)
                field_indexes["transliteration"].add_tokens(vid, toks)

        return cls(version=1, verse_keys=verse_keys, fields=field_indexes)

    # ---------------------
    # Persistence
    # ---------------------

    def save(self, path: str) -> None:
        """
        Save as gzipped pickle for speed and compactness.
        For mobile shipping, convert to a binary format (e.g., FlatBuffers) later.
        """
        payload = {
            "version": self.version,
            "verse_keys": self.verse_keys,
            "fields": {
                k: {
                    "token_counts": v.token_counts,
                    "token_postings": v.token_postings,
                }
                for k, v in self.fields.items()
            },
        }
        with gzip.open(path, "wb") as f:
            f.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str) -> "QuranCountIndex":
        with gzip.open(path, "rb") as f:
            payload = pickle.loads(f.read())

        fields: Dict[str, _FieldIndex] = {}
        for k, v in payload["fields"].items():
            fi = _FieldIndex()
            fi.token_counts = v["token_counts"]
            fi.token_postings = v["token_postings"]
            fields[k] = fi

        return cls(version=int(payload["version"]), verse_keys=list(payload["verse_keys"]), fields=fields)

    # ---------------------
    # Query API
    # ---------------------

    def count_token(self, token: str, fields: Optional[List[str]] = None, limit_verses: int = 50) -> CountResult:
        """
        Count occurrences of a token and return postings (verse_keys) for display.
        """
        tok = self._normalize_token(token)
        by_field: Dict[str, int] = {}

        # Choose fields based on script, but also allow latin token to appear in english+transliteration.
        candidates = fields or (["arabic", "urdu"] if _is_arabic_script(tok) else ["english", "transliteration"])

        verse_ids: List[int] = []
        seen = set()

        for field in candidates:
            fi = self.fields.get(field)
            if not fi:
                continue
            c = int(fi.token_counts.get(tok, 0))
            if c:
                by_field[field] = c
            for vid in fi.token_postings.get(tok, []):
                if vid not in seen:
                    seen.add(vid)
                    verse_ids.append(int(vid))
                if len(verse_ids) >= limit_verses:
                    break
            if len(verse_ids) >= limit_verses:
                break

        total = sum(by_field.values())
        verse_keys = []
        for vid in verse_ids[:limit_verses]:
            if 0 <= vid < len(self.verse_keys):
                verse_keys.append(self.verse_keys[vid])

        return CountResult(token=tok, total=total, by_field=by_field, verse_keys=verse_keys)

    def lookup_token_verses(self, token: str, limit: int = 50) -> List[str]:
        """
        Return verse_keys where token appears, using the appropriate field(s).
        """
        tok = self._normalize_token(token)

        if _is_arabic_script(tok):
            candidates = ["arabic", "urdu"]
        else:
            candidates = ["english", "transliteration"]

        verse_ids: List[int] = []
        seen = set()

        for field in candidates:
            fi = self.fields.get(field)
            if not fi:
                continue
            for vid in fi.token_postings.get(tok, []):
                if vid not in seen:
                    seen.add(vid)
                    verse_ids.append(int(vid))
                if len(verse_ids) >= limit:
                    break
            if len(verse_ids) >= limit:
                break

        # Map to verse_keys (safe bounds)
        out = []
        for vid in verse_ids[:limit]:
            if 0 <= vid < len(self.verse_keys):
                out.append(self.verse_keys[vid])
        return out

    def _normalize_token(self, token: str) -> str:
        t = (token or "").strip().strip(' "\'“”‘’')
        if not t:
            return ""
        # Arabic normalization (safe for Urdu)
        if _is_arabic_script(t):
            return normalize_for_search(t)
        # Latin
        t = t.lower()
        t = re.sub(r"[^a-z0-9']+", "", t)
        return t

def load_count_index(path: str) -> QuranCountIndex:
    """
    Load a count index from gzipped pickle (default) or JSON(.gz) payload.
    """
    p = Path(path)
    suffixes = "".join(p.suffixes[-2:]) if len(p.suffixes) >= 2 else p.suffix

    if suffixes in {".json.gz", ".json"}:
        opener = gzip.open if suffixes == ".json.gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        fields: Dict[str, _FieldIndex] = {}
        for k, v in payload["fields"].items():
            fi = _FieldIndex()
            fi.token_counts = v["token_counts"]
            fi.token_postings = v["token_postings"]
            fields[k] = fi
        return QuranCountIndex(version=int(payload["version"]), verse_keys=list(payload["verse_keys"]), fields=fields)

    return QuranCountIndex.load(str(p))


if __name__ == "__main__":
    # Minimal self-test (build speed sanity)
    import argparse
    project_root = Path(__file__).resolve().parent.parent
    default_quran = project_root / "output" / "processed" / "quran_complete.json"
    default_out = project_root / "output" / "processed" / "count_index.pkl.gz"

    p = argparse.ArgumentParser(description="Build count index using local repo data by default.")
    p.add_argument("--data", default=str(default_quran))
    p.add_argument("--out", default=str(default_out))
    p.add_argument("--max_verses", type=int, default=None)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idx = QuranCountIndex.build_from_quran_json(args.data, max_verses=args.max_verses)
    idx.save(out_path)
    print(f"Saved {out_path} | verse_keys={len(idx.verse_keys)} | fields={list(idx.fields.keys())}")
