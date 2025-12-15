########################## improved chatGPT's embeddings implementation.
"""
Quran Semantic Embeddings Builder (E5) — Mobile-Friendly, Deterministic, GPU-First

Outputs (aligned by row index):
- verse_embeddings_e5.npy      : (N, D) float16/float32 normalized embeddings
- verse_keys_e5.json           : list[str] stable mapping row->verse_key
- embeddings_meta_e5.json      : build metadata + sha256 integrity hash
- eval_report_e5.json          : evaluation metrics on a golden query set (optional)

Design principles:
1) Retrieval-optimized "passage:" text: bounded, high-signal fields (Arabic, transliteration, curated translations)
2) Keep all translations for UI display in original JSON; DO NOT dump all translations into embeddings (truncation/noise)
3) Deterministic ordering + integrity hashing for safe mobile shipping
4) GPU-first encoding with clear logs
"""

import json
import time
import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ----------------------------
# Configuration (tune safely)
# ----------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"

################################################################
# Requires: 3GB phone RAM, any 64-bit ARM processor
# MODEL_NAME = jeffh/intfloat-multilingual-e5-small-int8
# Recommended stack for mobile:
# - ONNX Runtime Mobile (best performance)
# - PyTorch Mobile (easier development)
# - TensorFlow Lite (good for Android)
# ✅ Can run on:
# - Samsung Galaxy A13 (4GB RAM)
# - iPhone SE 2nd gen (3GB RAM)
# - Pixel 4a (6GB RAM)
# - Most phones from 2019+
#
# # For even low-end devices, Consider these smaller alternatives:
# "thenlper/gte-small"  # ~60MB
# "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB
# "intfloat/multilingual-e5-base"  # Larger but more accurate
################################################################


# Translator preferences (your request)
EN_PREF = ["sahih-international"]
UR_PREF = ["fatah-muhammad-jalandhari"]

# Passage construction
INCLUDE_ARABIC = True
INCLUDE_TRANSLITERATION = True
INCLUDE_TRANSLITERATION_ALT = False

# Include these preferred translations (bounded, high-signal)
INCLUDE_EN_PREFS = True  # includes EN_PREF in passage if present
INCLUDE_UR_PREFS = True  # includes UR_PREF in passage if present

# Optional: include an extra stable baseline (helps some English queries); keep bounded
INCLUDE_SAHIH_INTERNATIONAL = False  # set True if you want a common baseline

# Hard cap to avoid token truncation/noise (chars is a practical proxy)
MAX_PASSAGE_CHARS = 1200

# Performance settings
BATCH_SIZE_GPU = 128
BATCH_SIZE_CPU = 32

# Mobile size optimization:
# - float16 halves size and is usually fine for cosine retrieval at this scale.
# - If you want maximum numeric fidelity, set EXPORT_DTYPE="float32".
EXPORT_DTYPE = "float16"  # "float16" or "float32"

# Evaluation (golden queries)
# Put expected verse_key(s) as list (some queries map to multiple acceptable verses)
GOLDEN_QUERIES = [
    {"q": "patience", "expected": ["2:153"]},
    {"q": "what does quran say about patience", "expected": ["2:153"]},
    {"q": "ayat ul kursi", "expected": ["2:255"]},
    {"q": "surah 2 ayah 255", "expected": ["2:255"]},
    {"q": "صبر", "expected": ["2:153"]},
    {"q": "نماز", "expected": ["2:43", "2:110", "4:103"]},  # example: multiple acceptable (you can refine)
]


# ----------------------------
# Utilities
# ----------------------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s.strip()


def safe_get(d: Dict[str, Any], *keys: str, default: str = "") -> str:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur if isinstance(cur, str) else default


def pick_translation(translations: Dict[str, str], pref: List[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (key, text) for each preferred translator present and non-empty, in pref order.
    """
    out: List[Tuple[str, str]] = []
    if not isinstance(translations, dict):
        return out
    for k in pref:
        v = translations.get(k)
        v = normalize_text(v or "")
        if v:
            out.append((k, v))
    return out


def sha256_of_array(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(arr)
    return hashlib.sha256(arr_c.tobytes()).hexdigest()


def write_json(path: Path, obj: Any, *, indent: int | None = None) -> None:
    """
    Write JSON bytes with stable UTF-8 + LF newlines (no text-mode translation).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=indent)
    path.write_bytes(data.encode("utf-8"))


def detect_device_and_log() -> Tuple[str, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    device = "cpu"

    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = cuda_ok

        if cuda_ok:
            device = "cuda"
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = str(torch.cuda.get_device_capability(0))
    except Exception as e:
        info["torch_error"] = str(e)

    return device, info


def clamp_passage(passage: str, max_chars: int) -> str:
    if len(passage) <= max_chars:
        return passage
    # Keep beginning (most important structure) + tail (often contains ref anchors)
    head = passage[: int(max_chars * 0.75)]
    tail = passage[-int(max_chars * 0.25):]
    return normalize_text(head + " … " + tail)


# ----------------------------
# Passage Builder (Best Practice)
# ----------------------------

def build_passage(verse: Dict[str, Any]) -> str:
    """
    Construct a bounded, high-signal passage for E5.
    Keep it structured + compact to avoid truncation and noise.
    """
    parts: List[str] = []

    verse_key = normalize_text(verse.get("verse_key", "") or verse.get("id", ""))
    surah_name_en = normalize_text(verse.get("surah_name_english", ""))
    surah_name_ar = normalize_text(verse.get("surah_name_arabic", ""))
    surah_translit = normalize_text(verse.get("surah_name_transliteration", ""))

    # Arabic
    if INCLUDE_ARABIC:
        ar = normalize_text(verse.get("arabic", ""))
        if ar:
            parts.append(f"ar: {ar}")

    # Transliteration(s)
    if INCLUDE_TRANSLITERATION:
        tr = normalize_text(verse.get("transliteration", ""))
        if tr:
            parts.append(f"tr: {tr}")

    if INCLUDE_TRANSLITERATION_ALT:
        tr2 = normalize_text(verse.get("transliteration_alt", ""))
        if tr2:
            parts.append(f"tr_alt: {tr2}")

    # Preferred translations
    translations_en = verse.get("translations_english") or {}
    translations_ur = verse.get("translations_urdu") or {}

    if INCLUDE_EN_PREFS:
        for k, text in pick_translation(translations_en, EN_PREF):
            parts.append(f"en({k}): {text}")
        # fallback builtin if none of preferred present
        if not any(p.startswith("en(") for p in parts):
            builtin = normalize_text(verse.get("translation_en_builtin", ""))
            if builtin:
                parts.append(f"en: {builtin}")

    if INCLUDE_UR_PREFS:
        for k, text in pick_translation(translations_ur, UR_PREF):
            parts.append(f"ur({k}): {text}")
        if not any(p.startswith("ur(") for p in parts):
            builtin = normalize_text(verse.get("translation_ur_builtin", ""))
            if builtin:
                parts.append(f"ur: {builtin}")

    if INCLUDE_SAHIH_INTERNATIONAL:
        si = normalize_text(safe_get(verse, "translations_english", "sahih-international"))
        if si:
            parts.append(f"en(sahih-international): {si}")

    # Lightweight anchors: ref + surah names (helps navigation/search intent)
    ref_bits = []
    if verse_key:
        ref_bits.append(verse_key)
    if surah_name_en:
        ref_bits.append(surah_name_en)
    if surah_translit:
        ref_bits.append(surah_translit)
    if surah_name_ar:
        ref_bits.append(surah_name_ar)
    if ref_bits:
        parts.append("ref: " + " | ".join(ref_bits))

    combined = " | ".join(parts)
    combined = normalize_text(combined)
    combined = clamp_passage(combined, MAX_PASSAGE_CHARS)

    return "passage: " + combined


# ----------------------------
# Evaluation Harness
# ----------------------------

def evaluate_retrieval(
    model: SentenceTransformer,
    device: str,
    embeddings: np.ndarray,
    verse_keys: List[str],
    queries: List[Dict[str, Any]],
    topk_list: List[int] = [1, 3, 5, 10],
) -> Dict[str, Any]:
    """
    Computes Recall@K and MRR@10 on the provided golden set.
    Assumes embeddings are already normalized.
    """
    key_to_index = {k: i for i, k in enumerate(verse_keys)}
    report: Dict[str, Any] = {"n": len(queries), "metrics": {}, "details": []}

    # Ensure float32 for stable dot products during eval
    emb = embeddings.astype(np.float32, copy=False)

    recalls = {k: 0 for k in topk_list}
    mrr10_sum = 0.0

    for item in queries:
        q_raw = str(item.get("q", "")).strip()
        expected_keys = item.get("expected", [])
        if isinstance(expected_keys, str):
            expected_keys = [expected_keys]
        expected_keys = [str(x) for x in expected_keys]

        q = "query: " + normalize_text(q_raw)
        q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32, copy=False)

        sims = emb @ q_emb
        top10 = np.argsort(-sims)[:10]
        top10_keys = [verse_keys[i] for i in top10]

        # Recall@K
        for k in topk_list:
            topk = top10_keys[:k]
            hit = any(e in topk for e in expected_keys)
            if hit:
                recalls[k] += 1

        # MRR@10
        rr = 0.0
        for rank, vk in enumerate(top10_keys, start=1):
            if vk in expected_keys:
                rr = 1.0 / rank
                break
        mrr10_sum += rr

        report["details"].append({
            "query": q_raw,
            "expected": expected_keys,
            "top10": top10_keys,
            "mrr10_rr": rr,
        })

    n = max(len(queries), 1)
    report["metrics"]["mrr@10"] = mrr10_sum / n
    for k in topk_list:
        report["metrics"][f"recall@{k}"] = recalls[k] / n

    return report


# ----------------------------
# Main Build
# ----------------------------

def main():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    out_dir = base_dir / "output" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_dir / "verse_embeddings_e5.npy"
    keys_path = out_dir / "verse_keys_e5.json"
    meta_path = out_dir / "embeddings_meta_e5.json"
    eval_path = out_dir / "eval_report_e5.json"

    print("=" * 80)
    print("QURAN E5 EMBEDDINGS BUILD — GPU-FIRST, MOBILE-FRIENDLY, DETERMINISTIC")
    print("=" * 80)

    print("\n1) Loading dataset...")
    if not data_path.exists():
        raise FileNotFoundError(f"Input not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        verses = json.load(f)

    if not isinstance(verses, list) or not verses:
        raise ValueError("quran_complete.json must be a non-empty list")

    print(f"   ✓ Verses loaded: {len(verses)}")

    print("\n2) Loading model...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    load_s = time.time() - t0
    print(f"   ✓ Model: {MODEL_NAME}")
    print(f"   ✓ Load time: {load_s:.2f}s")

    print("\n3) Selecting device (GPU-first)...")
    device, torch_info = detect_device_and_log()
    print(f"   ✓ Device selected: {device}")
    if torch_info:
        for k, v in torch_info.items():
            print(f"     - {k}: {v}")

    # Move model to device if possible (ensures GPU is used when available)
    try:
        model = model.to(device)
    except Exception as e:
        print(f"   ! Warning: could not move model to {device}: {e}")

    print("\n4) Building passages (retrieval-optimized, bounded)...")
    verse_keys: List[str] = []
    passages: List[str] = []

    # Deterministic ordering: sort by verse_key if present
    # (If file is already stable, this still protects you against upstream reorder.)
    def sort_key(v: Dict[str, Any]) -> str:
        return str(v.get("verse_key") or v.get("id") or "")

    verses_sorted = sorted(verses, key=sort_key)

    for idx, v in enumerate(verses_sorted):
        vk = normalize_text(v.get("verse_key", "") or v.get("id", "") or f"unknown:{idx}")
        verse_keys.append(vk)
        passages.append(build_passage(v))

    print(f"   ✓ Passages built: {len(passages)}")
    print(f"   ✓ Example:\n     {passages[0][:140]}...")

    print("\n5) Encoding embeddings...")
    batch_size = BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Normalize embeddings: True (cosine via dot product)")

    t1 = time.time()
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encode_s = time.time() - t1

    # Enforce dtype for export
    if EXPORT_DTYPE == "float16":
        embeddings = embeddings.astype(np.float16, copy=False)
    elif EXPORT_DTYPE == "float32":
        embeddings = embeddings.astype(np.float32, copy=False)
    else:
        raise ValueError("EXPORT_DTYPE must be 'float16' or 'float32'")

    print(f"\n   ✓ Encode time: {encode_s:.2f}s")
    print(f"   ✓ Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"   ✓ Speed: {len(passages) / max(encode_s, 1e-9):.1f} verses/sec")

    print("\n6) Saving artifacts...")
    np.save(embeddings_path, embeddings)
    write_json(keys_path, verse_keys)

    sha = sha256_of_array(embeddings)
    meta = {
        "model": MODEL_NAME,
        "device": device,
        "torch_info": torch_info,
        "count": len(verse_keys),
        "dim": int(embeddings.shape[1]),
        "normalized": True,
        "dtype": str(embeddings.dtype),
        "max_passage_chars": MAX_PASSAGE_CHARS,
        "include": {
            "arabic": INCLUDE_ARABIC,
            "transliteration": INCLUDE_TRANSLITERATION,
            "transliteration_alt": INCLUDE_TRANSLITERATION_ALT,
            "en_prefs": EN_PREF if INCLUDE_EN_PREFS else [],
            "ur_prefs": UR_PREF if INCLUDE_UR_PREFS else [],
            "include_sahih_international": INCLUDE_SAHIH_INTERNATIONAL,
        },
        "paths": {
            "input_json": str(data_path),
            "embeddings_npy": str(embeddings_path),
            "keys_json": str(keys_path),
        },
        "sha256": sha,
        "created_at_unix": int(time.time()),
        "encode_seconds": encode_s,
        "load_seconds": load_s,
        "batch_size": batch_size,
    }

    write_json(meta_path, meta, indent=2)

    size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Embeddings: {embeddings_path} ({size_mb:.2f} MB)")
    print(f"   ✓ Keys:       {keys_path}")
    print(f"   ✓ Meta:       {meta_path}")
    print(f"   ✓ SHA256:     {sha}")

    print("\n7) Verification (shape + sha256)...")
    loaded = np.load(embeddings_path)
    if loaded.shape != embeddings.shape:
        raise AssertionError("Shape mismatch after reload")
    if sha256_of_array(loaded) != sha:
        raise AssertionError("SHA256 mismatch after reload")
    print("   ✓ Verification passed")

    print("\n8) Evaluation (golden queries)...")
    try:
        report = evaluate_retrieval(
            model=model,
            device=device,
            embeddings=loaded,
            verse_keys=verse_keys,
            queries=GOLDEN_QUERIES,
            topk_list=[1, 3, 5, 10],
        )
        write_json(eval_path, report, indent=2)

        print(f"   ✓ Eval saved: {eval_path}")
        print("   Metrics:")
        for k, v in report["metrics"].items():
            print(f"     - {k}: {v:.4f}")

        # Print a short human-readable preview
        for d in report["details"][:3]:
            print(f"\n   Query: {d['query']}")
            print(f"   Expected: {d['expected']}")
            print(f"   Top10: {d['top10'][:5]} ...")
            print(f"   RR@10: {d['mrr10_rr']:.4f}")

    except Exception as e:
        print(f"   ! Evaluation skipped due to error: {e}")

    print("\n" + "=" * 80)
    print("✓ BUILD COMPLETE — READY FOR MOBILE SHIPPING")
    print("=" * 80)

    print("\nNext (mentor guidance):")
    print("1) On-device search: encode query as 'query: ...', normalize, dot with embeddings")
    print("2) Hybrid pipeline for 'flawless' UX: exact parser → lexical (searchable_text) → E5 dense")
    print("3) Expand GOLDEN_QUERIES continuously using real user searches + expected verse_keys")


if __name__ == "__main__":
    main()

########################## original claude's embeddings implementation.
# """
# Generate Semantic Embeddings using Multilingual E5-Small
# THE BEST model for retrieval tasks (research-backed choice)

# Key differences from MiniLM:
# 1. E5 is trained specifically for retrieval (not paraphrase)
# 2. Requires "passage: " prefix for documents
# 3. Requires "query: " prefix for search queries
# 4. Superior cross-lingual performance (Arabic↔English↔Urdu)
# 5. Higher NDCG@10 on retrieval benchmarks (+4.6 points)
# """

# import json
# import numpy as np
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from tqdm import tqdm
# import time


# def main():
#     # Paths
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
#     output_path = base_dir / "output" / "processed" / "verse_embeddings_e5.npy"
    
#     print("="*70)
#     print("GENERATING E5 SEMANTIC EMBEDDINGS (Research-Optimized)")
#     print("="*70)
#     print("\n📊 Model Choice: multilingual-e5-small")
#     print("   Why E5 > MiniLM for retrieval:")
#     print("   ✓ Trained on query-passage pairs (not paraphrases)")
#     print("   ✓ +4.6 NDCG@10 improvement on retrieval benchmarks")
#     print("   ✓ Superior cross-lingual semantic matching")
#     print("   ✓ Same size (471MB) but better performance")
    
#     # Load Quran data
#     print("\n1. Loading Quran data...")
#     with open(data_path, 'r', encoding='utf-8') as f:
#         verses = json.load(f)
#     print(f"   ✓ Loaded {len(verses)} verses")
    
#     # Load E5 model
#     print("\n2. Loading E5 embedding model...")
#     print("   Model: intfloat/multilingual-e5-small")
#     print("   (First run downloads ~471MB, takes 30-60 seconds...)")
    
#     start = time.time()
#     model = SentenceTransformer('intfloat/multilingual-e5-small')
#     load_time = time.time() - start
#     print(f"   ✓ Model loaded in {load_time:.1f}s")
    
#     # Prepare texts for embedding
#     print("\n3. Preparing verse texts with E5 format...")
#     print("   Note: E5 requires 'passage: ' prefix for documents")
    
#     passages = []
    
#     for verse in verses:
#         # Combine Arabic + primary translations
#         arabic = verse.get('arabic', '')
        
#         # Get best available English translation
#         english = (
#             verse.get('translations_english', {}).get('sahih-international') or
#             verse.get('translation_en_builtin') or
#             ''
#         )
        
#         # Get best available Urdu translation
#         urdu = (
#             verse.get('translations_urdu', {}).get('maulana-abu-al-maududi') or
#             verse.get('translation_ur_builtin') or
#             ''
#         )
        
#         # Combine: Arabic + English + Urdu
#         combined = f"{arabic} {english} {urdu}".strip()
        
#         # CRITICAL: E5 requires "passage: " prefix for document encoding
#         # This is what makes E5 excel at retrieval vs paraphrase tasks
#         passage = f"passage: {combined}"
#         passages.append(passage)
    
#     print(f"   ✓ Prepared {len(passages)} passages")
#     print(f"   ✓ Format example: 'passage: {passages[0][:50]}...'")
    
#     # Generate embeddings
#     print("\n4. Generating E5 embeddings...")
#     print("   (This will take 5-10 minutes for 6236 verses)")
#     print("   Your RTX 4080 will accelerate this significantly!")
#     print("   Progress:")
    
#     # Batch processing for speed
#     batch_size = 32  # Adjust based on your GPU memory
    
#     start = time.time()
#     embeddings = model.encode(
#         passages,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=True,  # Important for cosine similarity
#         device='cuda'  # Use your RTX 4080!
#     )
#     encode_time = time.time() - start
    
#     print(f"\n   ✓ Generated embeddings in {encode_time:.1f}s")
#     print(f"     Shape: {embeddings.shape}")
#     print(f"     Embedding dimension: {embeddings.shape[1]}")
#     print(f"     Speed: {len(passages)/encode_time:.1f} verses/second")
    
#     # Save embeddings
#     print("\n5. Saving embeddings...")
#     np.save(output_path, embeddings)
    
#     file_size = output_path.stat().st_size / (1024 * 1024)
#     print(f"   ✓ Saved to: {output_path}")
#     print(f"     File size: {file_size:.2f} MB")
    
#     # Verify
#     print("\n6. Verification...")
#     loaded = np.load(output_path)
#     assert loaded.shape == embeddings.shape, "Shape mismatch!"
#     assert np.allclose(loaded[:10], embeddings[:10]), "Data mismatch!"
#     print(f"   ✓ Verification passed")
    
#     # Sanity test with E5's query format
#     print("\n7. Sanity test (E5 retrieval performance)...")
    
#     # CRITICAL: E5 requires "query: " prefix for search queries
#     test_queries = [
#         "query: patience",
#         "query: what does quran say about prayer",
#         "query: صبر",  # Arabic
#     ]
    
#     for test_query in test_queries:
#         print(f"\n   Testing: '{test_query}'")
        
#         # Encode query with E5 format
#         query_embedding = model.encode(
#             [test_query], 
#             normalize_embeddings=True,
#             convert_to_numpy=True
#         )[0]
        
#         # Compute cosine similarities
#         similarities = np.dot(embeddings, query_embedding)
#         top_5_idx = np.argsort(-similarities)[:5]
        
#         print(f"   Top 5 semantically similar verses:")
#         for i, idx in enumerate(top_5_idx, 1):
#             verse = verses[idx]
#             sim = similarities[idx]
#             print(f"     {i}. {verse['verse_key']} (similarity: {sim:.3f})")
#             en_text = verse.get('translation_en_builtin', '')[:50]
#             print(f"        {en_text}...")
    
#     print("\n" + "="*70)
#     print("✓ E5 EMBEDDINGS GENERATION COMPLETE")
#     print("="*70)
#     print("\n📈 Expected Performance Improvement:")
#     print("   Baseline (BM25):        91.7%")
#     print("   With E5 Semantic:       94-96% (target)")
#     print("   Cross-lingual boost:    +3-5% on Urdu/Arabic queries")
#     print("\nNext: Update search engine to use E5 embeddings + query format")
    

# if __name__ == "__main__":
#     main()
