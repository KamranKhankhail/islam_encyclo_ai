#!/usr/bin/env python3
"""
Build Mobile Pack v1 for React-Native (portable, integrity-checked).

Inputs (from output/processed):
- quran_complete.json
- metadata.json
- verse_embeddings_e5.npy
- verse_keys_e5.json
- embeddings_meta_e5.json

Outputs (to output/mobile_pack/v1):
- verses_mobile_v1.ndjson        (Realm-import friendly, one JSON per line)
- verse_index_v1.json            (verse_key -> row index)
- aliases_v1.json                (normalized alias map for fast alias resolution)
- verse_embeddings_e5.f16.bin    (raw float16 row-major embeddings)
- pack_manifest_v1.json          (sha256 + size + counts)
- copies of: metadata.json, verse_keys_e5.json, embeddings_meta_e5.json

Design goals:
- Keep runtime super light on mobile
- Preserve strict alignment between verse_keys_e5 and packed verse rows
- Make translation selection policy explicit and auditable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Mobile v1 configuration defaults
# ----------------------------

# Display translation preference lists (used when policy="fixed")
EN_DISPLAY_PREF = ["sahih-international"]
UR_DISPLAY_PREF = ["fatah-muhammad-jalandhari"]

# Translation selection policy:
# - "fixed": prefer EN_DISPLAY_PREF/UR_DISPLAY_PREF, fallback to builtin, then any available
# - "random_seeded": deterministic random per verse (seed + verse_key hash), from available translators + builtin
DEFAULT_EN_POLICY = "random_seeded"
DEFAULT_UR_POLICY = "fixed"

# Searchable text cap to keep pack size bounded (mobile optimization)
DEFAULT_MAX_SEARCHABLE_CHARS = 1600

# Include transliteration in packed verse payload? (increases size; keep off unless you need it in UI/search)
DEFAULT_INCLUDE_TRANSLITERATION = True


# ----------------------------
# Utilities
# ----------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def ndjson_write(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def stable_int_hash(s: str) -> int:
    # Deterministic across runs/processes/platforms
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


# ----------------------------
# Arabic/Multilingual normalization
# ----------------------------

class FallbackNormalizer:
    """Fallback normalizer if arabic_normalizer.py is not importable."""

    _TATWEEL = "\u0640"

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFKC", str(text))
        t = t.casefold()
        # Remove combining marks
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        # Remove tatweel
        t = t.replace(self._TATWEEL, "")
        # Normalize Arabic alef variants
        t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا"))
        # Normalize Yeh variants
        t = t.replace("ى", "ي").replace("ئ", "ي").replace("ی", "ي")
        # Normalize waw-hamza
        t = t.replace("ؤ", "و")
        # Collapse whitespace
        t = " ".join(t.split())
        return t.strip()


def load_normalizer(repo_root: Path):
    """
    Prefer project arabic_normalizer.py. If unavailable, fallback.
    """
    try:
        # Ensure repo_root is on sys.path so `import arabic_normalizer` works.
        for p in (repo_root, repo_root / "scripts"):
            ps = str(p)
            if ps not in sys.path:
                sys.path.insert(0, ps)

        import arabic_normalizer  # type: ignore

        # Expect ArabicNormalizer exists (as in your provided file)
        norm = arabic_normalizer.ArabicNormalizer()  # type: ignore
        return lambda s: norm.normalize_for_search(s)
    except Exception:
        fb = FallbackNormalizer()
        return fb.normalize


# ----------------------------
# Translation selection
# ----------------------------

def _pick_fixed(
    translations: Dict[str, str],
    preferred: List[str],
    builtin_text: str,
) -> Tuple[str, str]:
    # preferred translators first
    for k in preferred:
        v = translations.get(k)
        if isinstance(v, str) and v.strip():
            return k, v.strip()
    # builtin fallback
    if isinstance(builtin_text, str) and builtin_text.strip():
        return "builtin", builtin_text.strip()
    # any available
    for k, v in translations.items():
        if isinstance(v, str) and v.strip():
            return k, v.strip()
    return "", ""


def _pick_random_seeded(
    verse_key: str,
    translations: Dict[str, str],
    builtin_text: str,
    base_seed: int,
) -> Tuple[str, str]:
    choices: List[Tuple[str, str]] = []

    # include builtin as a pseudo-translator if present
    if isinstance(builtin_text, str) and builtin_text.strip():
        choices.append(("builtin", builtin_text.strip()))

    for k, v in translations.items():
        if isinstance(v, str) and v.strip():
            choices.append((k, v.strip()))

    if not choices:
        return "", ""

    r = random.Random((base_seed + stable_int_hash(verse_key)) & 0xFFFFFFFF)
    return r.choice(choices)


def pick_translation(
    *,
    verse_key: str,
    translations: Dict[str, str],
    builtin_text: str,
    policy: str,
    preferred: List[str],
    base_seed: int,
) -> Tuple[str, str]:
    if policy == "fixed":
        return _pick_fixed(translations, preferred, builtin_text)
    if policy == "random_seeded":
        return _pick_random_seeded(verse_key, translations, builtin_text, base_seed)
    raise ValueError(f"Unknown translation policy: {policy}")


# ----------------------------
# Alias map extraction
# ----------------------------

def load_default_alias_map(repo_root: Path) -> Dict[str, str]:
    """
    Loads DEFAULT_ALIAS_MAP from hybrid_search_e5.py.
    We do this so mobile aliases match exactly what your engine uses.
    """
    try:
        # Ensure repo root + scripts in sys.path, then import.
        for p in (repo_root, repo_root / "scripts"):
            ps = str(p)
            if ps not in sys.path:
                sys.path.insert(0, ps)

        import hybrid_search_e5  # type: ignore

        amap = getattr(hybrid_search_e5, "DEFAULT_ALIAS_MAP", None)
        if not isinstance(amap, dict) or not amap:
            return {}
        # normalize to strings
        out: Dict[str, str] = {}
        for k, v in amap.items():
            out[str(k)] = str(v)
        return out
    except Exception:
        return {}


# ----------------------------
# Main pack builder
# ----------------------------

@dataclass(frozen=True)
class PackPaths:
    processed_dir: Path
    out_dir: Path

    quran_complete: Path
    metadata: Path
    emb_npy: Path
    emb_keys: Path
    emb_meta: Path


def resolve_paths(repo_root: Path, processed_dir: Optional[Path], out_dir: Optional[Path]) -> PackPaths:
    proc = processed_dir or (repo_root / "output" / "processed")
    out = out_dir or (repo_root / "output" / "mobile_pack" / "v1")
    data_dir = repo_root / "data"

    metadata_default = proc / "metadata.json"
    if not metadata_default.exists():
        metadata_default = data_dir / "metadata.json"

    return PackPaths(
        processed_dir=proc,
        out_dir=out,
        quran_complete=proc / "quran_complete.json",
        metadata=metadata_default,
        emb_npy=proc / "verse_embeddings_e5.npy",
        emb_keys=proc / "verse_keys_e5.json",
        emb_meta=proc / "embeddings_meta_e5.json",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=str, default=None, help="Override output/processed path")
    ap.add_argument("--out-dir", type=str, default=None, help="Override output/mobile_pack/v1 path")
    ap.add_argument("--pack-version", type=str, default="v1", help="Pack semantic version string")
    ap.add_argument("--en-policy", type=str, default=DEFAULT_EN_POLICY, choices=["fixed", "random_seeded"])
    ap.add_argument("--ur-policy", type=str, default=DEFAULT_UR_POLICY, choices=["fixed", "random_seeded"])
    ap.add_argument("--seed", type=int, default=0, help="Base seed used when policy=random_seeded (0 means derive from embeddings sha256)")
    ap.add_argument("--max-searchable-chars", type=int, default=DEFAULT_MAX_SEARCHABLE_CHARS)
    ap.add_argument("--include-transliteration", action="store_true", default=DEFAULT_INCLUDE_TRANSLITERATION)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(
        repo_root,
        Path(args.processed_dir) if args.processed_dir else None,
        Path(args.out_dir) if args.out_dir else None,
    )

    # Hard precondition checks
    for p in [paths.quran_complete, paths.metadata, paths.emb_npy, paths.emb_keys, paths.emb_meta]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    paths.out_dir.mkdir(parents=True, exist_ok=True)

    # Load normalizer
    normalize = load_normalizer(repo_root)

    # Load inputs
    verses: List[Dict[str, Any]] = read_json(paths.quran_complete)
    if not isinstance(verses, list) or not verses:
        raise ValueError("quran_complete.json must be a non-empty list")

    emb_keys: List[str] = read_json(paths.emb_keys)
    if not isinstance(emb_keys, list) or not emb_keys:
        raise ValueError("verse_keys_e5.json must be a non-empty list")

    if len(emb_keys) != len(verses):
        # Not always fatal, but for this pack we require strict alignment.
        raise ValueError(f"verse_keys_e5.json ({len(emb_keys)}) != verses ({len(verses)})")

    emb_meta = read_json(paths.emb_meta)
    emb_sha = str(emb_meta.get("sha256", "")) if isinstance(emb_meta, dict) else ""

    # Determine seed for random_seeded policy
    # If seed=0, derive seed from embeddings sha256 to keep the “intentional randomness” stable per embedding build.
    base_seed = int(args.seed)
    if base_seed == 0:
        if emb_sha:
            base_seed = stable_int_hash(emb_sha)
        else:
            base_seed = 1337

    # Build verse lookup for O(1) by verse_key
    by_key: Dict[str, Dict[str, Any]] = {}
    for v in verses:
        k = v.get("verse_key") or v.get("id")
        if isinstance(k, str) and k:
            by_key[k] = v

    # Ensure all embedding keys exist in verse data
    missing = [k for k in emb_keys if k not in by_key]
    if missing:
        raise ValueError(f"{len(missing)} embedding keys missing in quran_complete.json. Example: {missing[0]}")

    # Alias map
    alias_map = load_default_alias_map(repo_root)
    write_json(paths.out_dir / "aliases_v1.json", alias_map)

    # Copy metadata + embedding json files (verbatim)
    shutil.copy2(paths.metadata, paths.out_dir / "metadata.json")
    shutil.copy2(paths.emb_keys, paths.out_dir / "verse_keys_e5.json")
    shutil.copy2(paths.emb_meta, paths.out_dir / "embeddings_meta_e5.json")

    # Load embeddings .npy and export raw float16 binary
    emb = np.load(paths.emb_npy)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {emb.shape}")

    n, d = emb.shape
    if n != len(emb_keys):
        raise ValueError(f"Embeddings rows ({n}) != verse_keys count ({len(emb_keys)})")

    # Ensure float16 row-major contiguous
    if emb.dtype != np.float16:
        emb = emb.astype(np.float16)
    if not emb.flags["C_CONTIGUOUS"]:
        emb = np.ascontiguousarray(emb)

    emb_bin_path = paths.out_dir / "verse_embeddings_e5.f16.bin"
    emb.tofile(emb_bin_path)

    # Build packed verse rows in embedding order to preserve alignment
    out_rows: List[Dict[str, Any]] = []
    verse_index: Dict[str, int] = {}

    max_st_chars = int(args.max_searchable_chars)
    include_tr = bool(args.include_transliteration)

    for idx, vk in enumerate(emb_keys):
        v = by_key[vk]

        s = int(v.get("surah") or 0)
        a = int(v.get("ayah") or 0)
        j = int(v.get("juz") or 0)

        ar = str(v.get("arabic") or "").strip()

        # Get translations dicts
        en_map = v.get("translations_english") or {}
        ur_map = v.get("translations_urdu") or {}

        if not isinstance(en_map, dict):
            en_map = {}
        if not isinstance(ur_map, dict):
            ur_map = {}

        en_builtin = str(v.get("translation_en_builtin") or "")
        ur_builtin = str(v.get("translation_ur_builtin") or "")

        en_t, en = pick_translation(
            verse_key=vk,
            translations=en_map,
            builtin_text=en_builtin,
            policy=args.en_policy,
            preferred=EN_DISPLAY_PREF,
            base_seed=base_seed,
        )
        ur_t, ur = pick_translation(
            verse_key=vk,
            translations=ur_map,
            builtin_text=ur_builtin,
            policy=args.ur_policy,
            preferred=UR_DISPLAY_PREF,
            base_seed=base_seed,
        )

        # Transliteration (optional)
        tr = ""
        if include_tr:
            tr = str(v.get("transliteration") or "").strip()

        # Build searchable text (st): normalized, bounded
        # Include: verse key, surah names, arabic, selected translations, optional transliteration
        sn_en = str(v.get("surah_name_english") or "")
        sn_ar = str(v.get("surah_name_arabic") or "")
        sn_tr = str(v.get("surah_name_transliteration") or "")

        st_parts = [
            vk,
            sn_en,
            sn_tr,
            sn_ar,
            ar,
            en,
            ur,
        ]
        if include_tr and tr:
            st_parts.append(tr)

        # Normalize and cap
        st_raw = " ".join([p for p in st_parts if p])
        st_norm = normalize(st_raw)
        if len(st_norm) > max_st_chars:
            st_norm = st_norm[:max_st_chars].rstrip()

        row: Dict[str, Any] = {
            "k": vk,     # verse_key
            "s": s,      # surah
            "a": a,      # ayah
            "j": j,      # juz
            "ar": ar,    # Arabic text
            "en": en,    # selected English display translation
            "ur": ur,    # selected Urdu display translation
            "st": st_norm,  # normalized searchable text
            # translator keys used (auditability, minimal overhead)
            "et": en_t,
            "ut": ur_t,
        }
        if include_tr:
            row["tr"] = tr

        out_rows.append(row)
        verse_index[vk] = idx

    # Write NDJSON + verse index
    ndjson_write(paths.out_dir / "verses_mobile_v1.ndjson", out_rows)
    write_json(paths.out_dir / "verse_index_v1.json", verse_index)

    # Manifest: sha256 + bytes
    manifest_files = [
        "pack_manifest_v1.json",  # placeholder; written last
        "verses_mobile_v1.ndjson",
        "verse_index_v1.json",
        "aliases_v1.json",
        "metadata.json",
        "verse_keys_e5.json",
        "embeddings_meta_e5.json",
        "verse_embeddings_e5.f16.bin",
    ]

    # Compute file hashes (except manifest itself, computed last)
    file_entries: List[Dict[str, Any]] = []
    for name in manifest_files:
        if name == "pack_manifest_v1.json":
            continue
        p = paths.out_dir / name
        file_entries.append({
            "name": name,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    # Pack metadata
    pack_manifest = {
        "pack_version": str(args.pack_version),
        "schema": "askquran_mobile_pack_v1",
        "verse_count": len(out_rows),
        "embedding": {
            "model": str(emb_meta.get("model", "")) if isinstance(emb_meta, dict) else "",
            "dim": int(d),
            "rows": int(n),
            "dtype": "float16",
            "source_npy": "verse_embeddings_e5.npy",
            "sha256_array": emb_sha,
        },
        "translation_policy": {
            "en_policy": args.en_policy,
            "ur_policy": args.ur_policy,
            "en_pref": EN_DISPLAY_PREF,
            "ur_pref": UR_DISPLAY_PREF,
            "seed": int(base_seed),
            "note": "seed is applied per-verse via CRC32(verse_key) to ensure stable selection",
        },
        "searchable_text": {
            "max_chars": int(max_st_chars),
            "includes_transliteration": bool(include_tr),
            "normalizer": "arabic_normalizer.ArabicNormalizer.normalize_for_search (fallback if unavailable)",
        },
        "files": file_entries,
    }

    # Write manifest
    manifest_path = paths.out_dir / "pack_manifest_v1.json"
    write_json(manifest_path, pack_manifest)

    # Optional: include manifest hash too (not required, but useful)
    # (If you do, you must recompute after writing; we leave it out to avoid recursion confusion.)

    total_bytes = sum(e["bytes"] for e in file_entries) + manifest_path.stat().st_size
    print("=" * 80)
    print("✓ MOBILE PACK v1 BUILT")
    print(f"Out: {paths.out_dir}")
    print(f"Verses: {len(out_rows)}")
    print(f"Embeddings: rows={n} dim={d} dtype=float16 bin_bytes={emb_bin_path.stat().st_size}")
    print(f"Total pack size (bytes): {total_bytes}")
    print("=" * 80)


if __name__ == "__main__":
    main()
