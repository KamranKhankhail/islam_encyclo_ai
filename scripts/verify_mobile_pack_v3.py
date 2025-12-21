#!/usr/bin/env python3
"""
Verify Mobile Pack v3 (hard fail on any mismatch).

Validates:
- sha256(file) matches pack_manifest_v3.json for every listed file
- core semantic checks for verse pack alignment
- optional query_encoder artifacts when metadata.query_encoder exists
- optional embeddings_meta sha256_array if processed .npy exists

Run:
  python scripts/verify_mobile_pack_v3.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_lines(path: Path) -> int:
    # streaming line count (fast, low memory)
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=str, default=None, help="Override pack dir (default: output/mobile_pack/v3)")
    ap.add_argument("--processed-dir", type=str, default=None, help="Optional processed dir (default: output/processed)")
    ap.add_argument("--strict-array-hash", action="store_true", default=False, help="Verify embeddings sha256_array vs .npy (requires processed dir)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pack_dir = Path(args.pack_dir) if args.pack_dir else (repo_root / "output" / "mobile_pack" / "v3")
    processed_dir = Path(args.processed_dir) if args.processed_dir else (repo_root / "output" / "processed")

    manifest_path = pack_dir / "pack_manifest_v3.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("pack_manifest_v3.json must be a JSON object")

    verse_count = int(manifest.get("verse_count") or 0)
    if verse_count <= 0:
        raise ValueError("manifest verse_count must be > 0")

    emb = manifest.get("embedding") or {}
    if not isinstance(emb, dict):
        raise ValueError("manifest.embedding must be an object")

    rows = int(emb.get("rows") or 0)
    dim = int(emb.get("dim") or 0)
    dtype = str(emb.get("dtype") or "")
    if rows != verse_count:
        raise ValueError(f"embedding.rows ({rows}) != verse_count ({verse_count})")
    if dim <= 0:
        raise ValueError("embedding.dim must be > 0")
    if dtype != "float16":
        raise ValueError(f"embedding.dtype must be float16, got {dtype}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest.files must be a non-empty list")

    file_entries_by_name: Dict[str, Dict[str, Any]] = {}
    actual_by_name: Dict[str, Dict[str, Any]] = {}

    # 1) File integrity (sha256 + bytes)
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("manifest.files entries must be objects")
        name = str(entry.get("name") or "")
        exp_sha = str(entry.get("sha256") or "")
        exp_bytes = int(entry.get("bytes") or -1)
        if not name or not exp_sha or exp_bytes < 0:
            raise ValueError(f"Invalid file entry: {entry}")
        if name in file_entries_by_name:
            raise ValueError(f"Duplicate manifest entry: {name}")

        p = pack_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing packed file: {p}")

        actual_bytes = p.stat().st_size
        if actual_bytes != exp_bytes:
            raise ValueError(f"Size mismatch for {name}: manifest={exp_bytes}, actual={actual_bytes}")

        actual_sha = sha256_file(p)
        if actual_sha != exp_sha:
            raise ValueError(f"SHA256 mismatch for {name}: manifest={exp_sha}, actual={actual_sha}")

        file_entries_by_name[name] = entry
        actual_by_name[name] = {"bytes": actual_bytes, "sha256": actual_sha}

    def require_manifest_file(rel_path: str, *, expect_json: bool) -> None:
        entry = file_entries_by_name.get(rel_path)
        if entry is None:
            raise ValueError(f"Missing manifest entry for: {rel_path}")

        actual = actual_by_name.get(rel_path)
        if actual is None:
            raise ValueError(f"Missing actual file entry for: {rel_path}")

        exp_bytes = int(entry.get("bytes") or -1)
        exp_sha = str(entry.get("sha256") or "")
        if actual["bytes"] != exp_bytes:
            raise ValueError(f"Size mismatch for {rel_path}: manifest={exp_bytes}, actual={actual['bytes']}")
        if actual["sha256"] != exp_sha:
            raise ValueError(f"SHA256 mismatch for {rel_path}: manifest={exp_sha}, actual={actual['sha256']}")

        if expect_json:
            read_json(pack_dir / rel_path)

    # Optional: validate query encoder artifacts referenced by metadata.json
    metadata_path = pack_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must be a JSON object")

        query_encoder = metadata.get("query_encoder")
        if query_encoder is not None:
            if not isinstance(query_encoder, dict):
                raise ValueError("metadata.query_encoder must be an object")
            model_file = str(query_encoder.get("model_file") or "")
            tokenizer_file = str(query_encoder.get("tokenizer_file") or "")
            if not model_file or not tokenizer_file:
                raise ValueError("metadata.query_encoder must include model_file and tokenizer_file")

            require_manifest_file(model_file, expect_json=False)
            require_manifest_file(tokenizer_file, expect_json=True)

            meta_file = query_encoder.get("meta_file")
            if meta_file:
                require_manifest_file(str(meta_file), expect_json=True)

    # 2) Required semantic files exist
    verses_path = pack_dir / "verses_mobile_v1.ndjson"
    idx_path = pack_dir / "verse_index_v1.json"
    keys_path = pack_dir / "verse_keys_e5.json"
    emb_bin_path = pack_dir / "verse_embeddings_e5.f16.bin"

    for p in [verses_path, idx_path, keys_path, emb_bin_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required pack file: {p}")

    # 3) Count lines
    n_lines = count_lines(verses_path)
    if n_lines != verse_count:
        raise ValueError(f"NDJSON line count mismatch: expected={verse_count}, actual={n_lines}")

    # 4) Embeddings bin size
    expected_bin_bytes = rows * dim * 2
    actual_bin_bytes = emb_bin_path.stat().st_size
    if actual_bin_bytes != expected_bin_bytes:
        raise ValueError(f"Embeddings bin bytes mismatch: expected={expected_bin_bytes}, actual={actual_bin_bytes}")

    # 5) Verse key alignment: verse_index must map verse_keys[i] -> i
    verse_keys = read_json(keys_path)
    if not isinstance(verse_keys, list) or len(verse_keys) != verse_count:
        raise ValueError(f"verse_keys_e5.json must be list of length {verse_count}")

    verse_index = read_json(idx_path)
    if not isinstance(verse_index, dict) or len(verse_index) != verse_count:
        raise ValueError(f"verse_index_v1.json must be object of size {verse_count}")

    for i, vk in enumerate(verse_keys):
        if not isinstance(vk, str) or not vk:
            raise ValueError(f"Invalid verse key at row {i}: {vk}")
        got = verse_index.get(vk)
        if got != i:
            raise ValueError(f"Index alignment mismatch: key={vk} expected={i} got={got}")

    # 6) Optional: strict verify sha256_array vs processed npy (if requested)
    if args.strict_array_hash:
        npy_path = processed_dir / "verse_embeddings_e5.npy"
        meta_path = pack_dir / "embeddings_meta_e5.json"
        if not npy_path.exists():
            raise FileNotFoundError(f"--strict-array-hash requires: {npy_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"--strict-array-hash requires: {meta_path}")

        # Compute sha256 over raw array bytes in a deterministic way:
        # load npy, ensure float16, C-contiguous, then hash .tobytes()
        import numpy as np  # local import to keep base deps minimal

        arr = np.load(npy_path)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        h = hashlib.sha256()
        h.update(arr.tobytes(order="C"))
        actual_arr_sha = h.hexdigest()

        meta = read_json(meta_path)
        exp_arr_sha = str(meta.get("sha256") or "")
        if exp_arr_sha and actual_arr_sha != exp_arr_sha:
            raise ValueError(f"sha256_array mismatch: meta={exp_arr_sha}, actual={actual_arr_sha}")

    print("OK: MOBILE PACK v3 VERIFIED")


if __name__ == "__main__":
    main()
