import json, hashlib
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]  # repo root
SRC = BASE / "output" / "processed"
OUT = BASE / "askquran_pack" / "v1"
OUT.mkdir(parents=True, exist_ok=True)

FILES_TO_COPY = [
    "quran_compact.json",
    "metadata.json",
    "verse_keys_e5.json",
    "embeddings_meta_e5.json",
]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    # 1) Copy JSON files
    manifest = {"version": "v1", "files": {}}

    for name in FILES_TO_COPY:
        src = SRC / name
        if not src.exists():
            raise FileNotFoundError(f"Missing: {src}")
        dst = OUT / name
        dst.write_bytes(src.read_bytes())
        manifest["files"][name] = {
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        }

    # 2) Convert embeddings .npy -> float16 raw binary (smaller + faster to load)
    emb_npy = SRC / "verse_embeddings_e5.npy"
    if not emb_npy.exists():
        raise FileNotFoundError(f"Missing: {emb_npy}")

    emb = np.load(emb_npy)              # expected shape: (N, D)
    emb_f16 = emb.astype(np.float16)    # float16 halves size

    emb_bin = OUT / "verse_embeddings_e5.f16.bin"
    emb_f16.tofile(emb_bin)

    manifest["files"][emb_bin.name] = {
        "bytes": emb_bin.stat().st_size,
        "sha256": sha256_file(emb_bin),
        "dtype": "float16",
        "shape": list(emb_f16.shape),
        "row_major": True,
    }

    # 3) Write manifest
    mf = OUT / "askquran_pack_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK:", OUT)
    print("Total MB:", sum(v["bytes"] for v in manifest["files"].values()) / (1024 * 1024))

if __name__ == "__main__":
    main()
