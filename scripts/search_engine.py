############# chatgpt's search enine implementation
"""
Quran BM25 Search Engine (Production-grade)
Hybrid: Structural (exact navigation) + Lexical (Field-weighted BM25)

Key properties:
- Multi-field BM25 (BM25F-like via weighted sum of per-field BM25 scores)  :contentReference[oaicite:3]{index=3}
- Adaptive stopword removal (only for long/noisy queries)
- Optional light Arabic stemming (prefix/suffix stripping)                 :contentReference[oaicite:4]{index=4}
- Structural queries supported (single verse + ranges)
- Deterministic on-disk index cache for fast startup
- Fast top-K using numpy argpartition if available

Dependencies:
  pip install rank_bm25
Optional:
  pip install numpy
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from arabic_normalizer import normalize_for_search
from query_parser import StructuralQueryParser


log = logging.getLogger("quran_search")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class QuranSearchConfig:
    # Field weights (BM25F-like)
    w_arabic: float = 1.00
    w_english: float = 0.65
    w_urdu: float = 0.75
    w_translit: float = 0.55  # if present inside searchable_text or a dedicated field

    # BM25 params (typical defaults; tune with your eval harness) :contentReference[oaicite:5]{index=5}
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Query behavior
    top_candidates: int = 200       # rerank shortlist size
    min_score: float = 0.0          # filter final results below this
    rerank_token_coverage_boost: float = 0.18  # small precision boost
    rerank_exact_substring_boost: float = 0.20 # boost if normalized query is substring

    # Stopword behavior
    adaptive_stopwords_min_tokens: int = 4  # only remove stopwords if query has >= this many tokens

    # Arabic behavior
    aggressive_arabic_normalization: bool = True
    light_arabic_stemming: bool = False  # optional (can help recall; use carefully)

    # Index caching
    enable_cache: bool = True
    cache_dir: str = ".bm25_cache"

    # Translation selection
    en_key: str = "sahih-international"
    ur_key: str = "maulana-abu-al-maududi"


# -----------------------------
# Tokenization (robust, multilingual)
# -----------------------------

_ARABIC_BLOCK_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LATIN_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ARABIC_WORD_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

# A minimal, high-impact English stopword set (don’t be too aggressive)
_EN_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","about",
    "what","does","do","did","is","are","was","were","be","been","being",
    "say","says","said","tell","me","please","show","find","explain",
    "this","that","these","those","it","its","as","at","from","by",
    "quran","koran",
}

# Very small, safe synonym hooks (keep conservative; allow extension)
# (You can extend this map from config in your app layer.)
_SYNONYMS = {
    "prayer": ["salah", "salat", "صلاة", "نماز"],
    "patience": ["sabr", "صبر"],
    "charity": ["zakat", "zakah", "زكاة", "زکاة"],
    "fasting": ["sawm", "صوم", "روزہ"],
    "faith": ["iman", "إيمان", "ایمان"],
}


def _is_arabic_script(s: str) -> bool:
    return bool(_ARABIC_BLOCK_RE.search(s or ""))


def _light_stem_arabic_token(tok: str) -> str:
    """
    Very conservative light stemming: strip common single-letter prefixes and 'ال' definite article,
    plus a few common suffixes. This is intentionally minimal to avoid overmatching.
    Research often shows light stemming can improve Arabic IR, but it must be applied carefully. :contentReference[oaicite:6]{index=6}
    """
    if len(tok) < 4:
        return tok

    # Normalize (already mostly done by normalize_for_search)
    t = tok

    # Strip conjunction/preposition prefixes (و ف ب ك ل س) only once
    if t[0] in ("و", "ف", "ب", "ك", "ل", "س") and len(t) >= 4:
        t = t[1:]

    # Strip definite article 'ال'
    if t.startswith("ال") and len(t) >= 5:
        t = t[2:]

    # Strip common suffixes (very conservative)
    for suf in ("ه", "ها", "هم", "هن", "كما", "كم", "كن", "نا", "ي", "ية", "ات", "ون", "ين"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break

    return t or tok


def tokenize_mixed(text: str, *, aggressive_ar: bool, light_stem_ar: bool) -> List[str]:
    """
    Tokenize Arabic/Urdu + Latin in a stable way:
    - Arabic tokens: normalize_for_search then extract Arabic word runs
    - Latin tokens: lowercase alnum runs
    """
    if not text or not text.strip():
        return []

    # Normalize Arabic text (works fine even if the text is mixed)
    norm = normalize_for_search(text) if aggressive_ar else text.lower()

    out: List[str] = []

    # Arabic-script tokens
    for t in _ARABIC_WORD_RE.findall(norm):
        if not t:
            continue
        if light_stem_ar:
            t2 = _light_stem_arabic_token(t)
            out.append(t2)
        out.append(t)

    # Latin tokens
    for t in _LATIN_WORD_RE.findall(norm.lower()):
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(t)

    return out


def expand_query_tokens(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    out = list(tokens)
    for t in tokens:
        syns = _SYNONYMS.get(t)
        if syns:
            out.extend(syns)
    return out


def adaptive_stopword_filter(tokens: List[str], *, min_tokens: int) -> List[str]:
    """
    Remove stopwords only when the query is long enough to be noisy.
    For short queries, keep everything to avoid deleting intent.
    """
    if len(tokens) < min_tokens:
        return tokens
    return [t for t in tokens if t not in _EN_STOPWORDS]


# -----------------------------
# Engine
# -----------------------------

class QuranSearchEngine:
    def __init__(
        self,
        data_path: str,
        metadata_path: Optional[str] = None,
        config: Optional[QuranSearchConfig] = None,
    ):
        self.data_path = str(data_path)
        self.metadata_path = str(metadata_path) if metadata_path else None
        self.config = config or QuranSearchConfig()

        self.structural_parser = StructuralQueryParser(metadata_path=self.metadata_path)

        self.verses: List[Dict[str, Any]] = self._load_verses(self.data_path)
        self.verse_lookup: Dict[str, Dict[str, Any]] = {
            v["verse_key"]: v for v in self.verses if isinstance(v.get("verse_key"), str)
        }

        # Build or load BM25 indexes
        self._load_or_build_indexes()

    # --------- loading / caching ---------

    def _load_verses(self, path: str) -> List[Dict[str, Any]]:
        log.info("Loading Quran data: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("quran_complete.json must be a JSON array.")
        log.info("Loaded %d verses", len(data))
        return data

    def _cache_key(self) -> str:
        st = os.stat(self.data_path)
        # Deterministic cache key without hashing whole file
        return f"{Path(self.data_path).name}__{st.st_size}__{int(st.st_mtime)}__k1{self.config.bm25_k1}_b{self.config.bm25_b}__stem{int(self.config.light_arabic_stemming)}"

    def _cache_paths(self) -> Tuple[Path, Path]:
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key()
        return cache_dir / f"{key}.pkl", cache_dir / f"{key}.meta.json"

    def _load_or_build_indexes(self) -> None:
        cache_pkl, cache_meta = self._cache_paths()

        if self.config.enable_cache and cache_pkl.exists() and cache_meta.exists():
            try:
                with open(cache_pkl, "rb") as f:
                    obj = pickle.load(f)
                self._apply_cached(obj)
                log.info("Loaded BM25 index from cache: %s", cache_pkl)
                return
            except Exception as e:
                log.warning("Cache load failed (%s). Rebuilding index.", e)

        log.info("Building BM25 indexes (fielded)...")
        self._build_indexes()

        if self.config.enable_cache:
            try:
                payload = self._export_cache_payload()
                with open(cache_pkl, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(cache_meta, "w", encoding="utf-8") as f:
                    json.dump({"cache_key": self._cache_key()}, f, ensure_ascii=False, indent=2)
                log.info("Saved BM25 index cache: %s", cache_pkl)
            except Exception as e:
                log.warning("Failed to write cache (%s). Continuing without cache.", e)

    def _apply_cached(self, payload: Dict[str, Any]) -> None:
        self.tokens_ar = payload["tokens_ar"]
        self.tokens_en = payload["tokens_en"]
        self.tokens_ur = payload["tokens_ur"]
        self.tokens_mix = payload["tokens_mix"]

        self.bm25_ar = payload["bm25_ar"]
        self.bm25_en = payload["bm25_en"]
        self.bm25_ur = payload["bm25_ur"]
        self.bm25_mix = payload["bm25_mix"]

    def _export_cache_payload(self) -> Dict[str, Any]:
        return {
            "tokens_ar": self.tokens_ar,
            "tokens_en": self.tokens_en,
            "tokens_ur": self.tokens_ur,
            "tokens_mix": self.tokens_mix,
            "bm25_ar": self.bm25_ar,
            "bm25_en": self.bm25_en,
            "bm25_ur": self.bm25_ur,
            "bm25_mix": self.bm25_mix,
        }

    # --------- index build ---------

    def _build_indexes(self) -> None:
        cfg = self.config

        self.tokens_ar: List[List[str]] = []
        self.tokens_en: List[List[str]] = []
        self.tokens_ur: List[List[str]] = []
        self.tokens_mix: List[List[str]] = []

        for v in self.verses:
            ar = v.get("arabic", "") or ""
            en = ((v.get("translations_english") or {}).get(cfg.en_key) or v.get("translation_en_builtin") or "")
            ur = ((v.get("translations_urdu") or {}).get(cfg.ur_key) or v.get("translation_ur_builtin") or "")

            # Some datasets have a "searchable_text" field that already merges everything.
            mix = v.get("searchable_text")
            if not isinstance(mix, str) or not mix.strip():
                mix = f"{ar} {en} {ur}"

            tar = tokenize_mixed(ar, aggressive_ar=cfg.aggressive_arabic_normalization, light_stem_ar=cfg.light_arabic_stemming)
            ten = tokenize_mixed(en, aggressive_ar=cfg.aggressive_arabic_normalization, light_stem_ar=False)
            tur = tokenize_mixed(ur, aggressive_ar=cfg.aggressive_arabic_normalization, light_stem_ar=cfg.light_arabic_stemming)
            tmix = tokenize_mixed(mix, aggressive_ar=cfg.aggressive_arabic_normalization, light_stem_ar=cfg.light_arabic_stemming)

            # Apply stopword filtering to docs too (keeps index cleaner); not too aggressive
            ten = [t for t in ten if t not in _EN_STOPWORDS]

            self.tokens_ar.append(tar)
            self.tokens_en.append(ten)
            self.tokens_ur.append(tur)
            self.tokens_mix.append(tmix)

        self.bm25_ar = BM25Okapi(self.tokens_ar, k1=cfg.bm25_k1, b=cfg.bm25_b)
        self.bm25_en = BM25Okapi(self.tokens_en, k1=cfg.bm25_k1, b=cfg.bm25_b)
        self.bm25_ur = BM25Okapi(self.tokens_ur, k1=cfg.bm25_k1, b=cfg.bm25_b)
        self.bm25_mix = BM25Okapi(self.tokens_mix, k1=cfg.bm25_k1, b=cfg.bm25_b)

    # --------- search ---------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # 1) Structural first (exact navigation)
        structural = self.structural_parser.parse(query)
        if structural:
            resolved = self._resolve_structural(structural)
            if resolved:
                return resolved
            # if structural parse succeeded but verse not present, fall back to lexical

        # 2) Lexical (BM25)
        return self._lexical_search(query, top_k)

    def _resolve_structural(self, structural: Dict[str, Any]) -> List[Dict[str, Any]]:
        confidence = float(structural.get("confidence", 1.0))

        # Range support if parser provides it
        if "ayah_end" in structural and "surah" in structural and "ayah" in structural:
            s = int(structural["surah"])
            a1 = int(structural["ayah"])
            a2 = int(structural["ayah_end"])
            if a2 < a1:
                a1, a2 = a2, a1
            out = []
            for a in range(a1, a2 + 1):
                vk = f"{s}:{a}"
                v = self.verse_lookup.get(vk)
                if not v:
                    continue
                vv = v.copy()
                vv["relevance_score"] = confidence
                vv["match_type"] = "structural"
                out.append(vv)
            return out

        vk = structural.get("verse_key")
        if isinstance(vk, str) and vk:
            v = self.verse_lookup.get(vk)
            if v:
                vv = v.copy()
                vv["relevance_score"] = confidence
                vv["match_type"] = "structural"
                return [vv]

        return []

    def _lexical_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        cfg = self.config
        q_tokens = tokenize_mixed(
            query,
            aggressive_ar=cfg.aggressive_arabic_normalization,
            light_stem_ar=cfg.light_arabic_stemming,
        )

        if not q_tokens:
            return []

        # Adaptive stopwords only if query is long/noisy
        q_tokens = adaptive_stopword_filter(q_tokens, min_tokens=cfg.adaptive_stopwords_min_tokens)

        # Conservative query expansion hook
        q_tokens = expand_query_tokens(q_tokens)

        # Split for language-sensitive weighting
        q_has_ar = any(_is_arabic_script(t) for t in q_tokens)
        q_has_lat = any(_LATIN_WORD_RE.fullmatch(t or "") for t in q_tokens)

        # Compute per-field scores
        ar_scores = self.bm25_ar.get_scores(q_tokens) if q_has_ar else self.bm25_ar.get_scores([])
        en_scores = self.bm25_en.get_scores(q_tokens) if q_has_lat else self.bm25_en.get_scores([])
        ur_scores = self.bm25_ur.get_scores(q_tokens) if q_has_ar else self.bm25_ur.get_scores([])
        mix_scores = self.bm25_mix.get_scores(q_tokens)

        # Normalize per-field to keep weights meaningful across fields
        def _norm(scores):
            try:
                mx = float(max(scores))
            except Exception:
                mx = 0.0
            if mx <= 0:
                return scores
            return scores / mx  # numpy arrays support this; python lists will fail, but rank_bm25 uses numpy

        try:
            import numpy as np  # type: ignore

            ar_scores = _norm(ar_scores)
            en_scores = _norm(en_scores)
            ur_scores = _norm(ur_scores)
            mix_scores = _norm(mix_scores)

            # Language-aware weighting: if Arabic query, emphasize Arabic/Urdu/mix; if Latin query, emphasize English/mix
            if q_has_ar and not q_has_lat:
                w_ar, w_en, w_ur, w_mix = (cfg.w_arabic, 0.25, cfg.w_urdu, 0.85)
            elif q_has_lat and not q_has_ar:
                w_ar, w_en, w_ur, w_mix = (0.25, cfg.w_english, 0.20, 0.90)
            else:
                w_ar, w_en, w_ur, w_mix = (cfg.w_arabic, cfg.w_english, cfg.w_urdu, 1.00)

            scores = (w_ar * ar_scores) + (w_en * en_scores) + (w_ur * ur_scores) + (w_mix * mix_scores)

            n = scores.shape[0]
            k = max(1, min(int(top_k), n))
            cand = min(cfg.top_candidates, n)

            idx = np.argpartition(-scores, cand - 1)[:cand]
            idx = idx[np.argsort(-scores[idx])]

            # Rerank shortlist with precision boosts
            q_norm = normalize_for_search(query) if cfg.aggressive_arabic_normalization else query.lower()
            q_set = set(q_tokens)

            reranked: List[Tuple[int, float]] = []
            for i in idx.tolist():
                base = float(scores[i])
                if base <= cfg.min_score:
                    continue

                doc_tokens = set(self.tokens_mix[i])
                if q_set:
                    coverage = len(q_set & doc_tokens) / max(1, len(q_set))
                else:
                    coverage = 0.0

                boost = coverage * cfg.rerank_token_coverage_boost

                # Exact substring boost on normalized merged field
                if cfg.rerank_exact_substring_boost > 0 and q_norm and len(q_norm) >= 3:
                    doc_text = self.verses[i].get("searchable_text") or ""
                    doc_norm = normalize_for_search(doc_text) if cfg.aggressive_arabic_normalization else str(doc_text).lower()
                    if q_norm in doc_norm:
                        boost += cfg.rerank_exact_substring_boost

                reranked.append((i, base + boost))

            reranked.sort(key=lambda x: x[1], reverse=True)
            top = reranked[:k]

            results: List[Dict[str, Any]] = []
            for i, s in top:
                v = self.verses[i].copy()
                v["relevance_score"] = float(s)
                v["match_type"] = "lexical"
                results.append(v)
            return results

        except Exception:
            # Fallback: slower but safe for environments without numpy
            scores = list(mix_scores)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(1, top_k)]
            out: List[Dict[str, Any]] = []
            for i in ranked:
                s = float(scores[i])
                if s <= cfg.min_score:
                    continue
                v = self.verses[i].copy()
                v["relevance_score"] = s
                v["match_type"] = "lexical"
                out.append(v)
            return out

    # --------- formatting ---------

    def format_result(self, verse: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config
        return {
            "verse_key": verse.get("verse_key"),
            "surah": verse.get("surah"),
            "surah_name_english": verse.get("surah_name_english"),
            "surah_name_arabic": verse.get("surah_name_arabic"),
            "ayah": verse.get("ayah"),
            "arabic": verse.get("arabic"),
            "translation_english": (
                (verse.get("translations_english") or {}).get(cfg.en_key)
                or verse.get("translation_en_builtin", "")
                or ""
            ),
            "translation_urdu": (
                (verse.get("translations_urdu") or {}).get(cfg.ur_key)
                or verse.get("translation_ur_builtin", "")
                or ""
            ),
            "relevance_score": float(verse.get("relevance_score", 0.0) or 0.0),
            "match_type": verse.get("match_type", "unknown"),
            "juz": verse.get("juz"),
        }

    def search_formatted(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return [self.format_result(v) for v in self.search(query, top_k)]


# -----------------------------
# Convenience cached engine
# -----------------------------

_DEFAULT_ENGINE: Optional[QuranSearchEngine] = None

def get_default_engine() -> QuranSearchEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is not None:
        return _DEFAULT_ENGINE

    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    metadata_path = base_dir / "output" / "processed" / "metadata.json"

    cfg = QuranSearchConfig(
        enable_cache=True,
        light_arabic_stemming=False,  # start OFF; enable only if evaluation shows net gain
    )

    _DEFAULT_ENGINE = QuranSearchEngine(
        data_path=str(data_path),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        config=cfg,
    )
    return _DEFAULT_ENGINE


def quick_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    engine = get_default_engine()
    return engine.search_formatted(query, top_k)


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    metadata_path = base_dir / "output" / "processed" / "metadata.json"

    engine = QuranSearchEngine(
        data_path=str(data_path),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        config=QuranSearchConfig(enable_cache=True),
    )

    tests = [
        "2:255",
        "2:255-257",
        "surah fatiha",
        "patience",
        "bismillah",
        "صبر",
        "نماز",
        "what does quran say about prayer",
    ]

    for q in tests:
        print("\nQuery:", q)
        res = engine.search_formatted(q, top_k=3)
        for i, r in enumerate(res, 1):
            print(f"{i}. {r.get('verse_key')}  score={r['relevance_score']:.3f}  type={r['match_type']}")
            a = (r.get("arabic") or "")
            e = (r.get("translation_english") or "")
            print("   AR:", a[:70] + ("..." if len(a) > 70 else ""))
            print("   EN:", e[:70] + ("..." if len(e) > 70 else ""))


############# claude's search enine implementation
# """
# BM25 Search Engine for Quran
# Implements keyword-based search with ranking
# """

# import json
# from pathlib import Path
# from typing import List, Dict, Optional
# from rank_bm25 import BM25Okapi
# import re

# from arabic_normalizer import normalize_for_search, tokenize_arabic
# from query_parser import StructuralQueryParser


# class QuranSearchEngine:
#     """Hybrid search engine: Structural + Lexical (BM25)"""
    
#     def __init__(self, data_path: str):
#         """
#         Initialize search engine with Quran data
        
#         Args:
#             data_path: Path to quran_complete.json
#         """
#         print("Loading Quran data...")
#         with open(data_path, 'r', encoding='utf-8') as f:
#             self.verses = json.load(f)
        
#         print(f"Loaded {len(self.verses)} verses")
        
#         # Create verse lookup for fast access
#         self.verse_lookup = {v['verse_key']: v for v in self.verses}
        
#         # Initialize parsers
#         self.structural_parser = StructuralQueryParser()
        
#         # Prepare corpus for BM25
#         print("Building search index...")
#         self._build_search_index()
#         print("Search engine ready!")
    
#     def _build_search_index(self):
#         """Build BM25 index from verse corpus"""
#         # Tokenize all searchable text
#         self.corpus_tokens = []
        
#         for verse in self.verses:
#             # Get searchable text and normalize
#             text = verse.get('searchable_text', '')
            
#             # Tokenize (handles Arabic + English + Urdu)
#             tokens = self._tokenize_multilingual(text)
#             self.corpus_tokens.append(tokens)
        
#         # Build BM25 index
#         self.bm25 = BM25Okapi(self.corpus_tokens)
    
#     def _tokenize_multilingual(self, text: str) -> List[str]:
#         """
#         Tokenize text that may contain Arabic, English, Urdu
#         """
#         tokens = []
        
#         # Split into words
#         words = text.split()
        
#         for word in words:
#             # Check if word contains Arabic/Urdu characters
#             if self._contains_arabic(word):
#                 # Normalize and tokenize Arabic
#                 normalized = normalize_for_search(word)
#                 if normalized:
#                     tokens.append(normalized)
#             else:
#                 # English/transliteration - just lowercase
#                 normalized = word.lower().strip()
#                 # Remove punctuation
#                 normalized = re.sub(r'[^\w\s]', '', normalized)
#                 if normalized and len(normalized) > 1:  # Skip single chars
#                     tokens.append(normalized)
        
#         return tokens
    
#     def _contains_arabic(self, text: str) -> bool:
#         """Check if text contains Arabic characters"""
#         arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
#         return bool(arabic_pattern.search(text))
    
#     def search(self, query: str, top_k: int = 5) -> List[Dict]:
#         """
#         Search for verses matching the query
        
#         Args:
#             query: Search query
#             top_k: Number of results to return
        
#         Returns:
#             List of verse results with relevance scores
#         """
#         # Step 1: Try structural query parsing first
#         structural_result = self.structural_parser.parse(query)
        
#         if structural_result:
#             # Direct structural match
#             verse_key = structural_result['verse_key']
#             if verse_key in self.verse_lookup:
#                 verse = self.verse_lookup[verse_key].copy()
#                 verse['relevance_score'] = structural_result['confidence']
#                 verse['match_type'] = 'structural'
#                 return [verse]
#             else:
#                 # Invalid verse reference
#                 return []
        
#         # Step 2: Lexical search using BM25
#         return self._lexical_search(query, top_k)
    
#     def _lexical_search(self, query: str, top_k: int) -> List[Dict]:
#         """
#         Perform BM25 keyword search
#         """
#         # Tokenize query
#         query_tokens = self._tokenize_multilingual(query)
        
#         if not query_tokens:
#             return []
        
#         # Get BM25 scores
#         scores = self.bm25.get_scores(query_tokens)
        
#         # Get top-K indices
#         top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
#         # Build results
#         results = []
#         for idx in top_indices:
#             if scores[idx] > 0:  # Only include verses with positive scores
#                 verse = self.verses[idx].copy()
#                 verse['relevance_score'] = float(scores[idx])
#                 verse['match_type'] = 'lexical'
#                 results.append(verse)
        
#         return results
    
#     def format_result(self, verse: Dict) -> Dict:
#         """Format verse result for display"""
#         return {
#             'verse_key': verse['verse_key'],
#             'surah': verse['surah'],
#             'surah_name_english': verse['surah_name_english'],
#             'surah_name_arabic': verse['surah_name_arabic'],
#             'ayah': verse['ayah'],
#             'arabic': verse['arabic'],
#             'translation_english': verse.get('translations_english', {}).get('sahih-international', 
#                                            verse.get('translation_en_builtin', '')),
#             'translation_urdu': verse.get('translations_urdu', {}).get('maulana-abu-al-maududi',
#                                          verse.get('translation_ur_builtin', '')),
#             'relevance_score': verse.get('relevance_score', 0.0),
#             'match_type': verse.get('match_type', 'unknown'),
#             'juz': verse.get('juz'),
#         }
    
#     def search_formatted(self, query: str, top_k: int = 5) -> List[Dict]:
#         """Search and return formatted results"""
#         results = self.search(query, top_k)
#         return [self.format_result(v) for v in results]


# # Convenience function for quick searching
# def quick_search(query: str, top_k: int = 5) -> List[Dict]:
#     """Quick search using default engine"""
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
    
#     engine = QuranSearchEngine(str(data_path))
#     return engine.search_formatted(query, top_k)


# # Test the search engine
# if __name__ == "__main__":
#     from pathlib import Path
    
#     # Get data path
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
    
#     # Initialize engine
#     print("Initializing search engine...")
#     engine = QuranSearchEngine(str(data_path))
    
#     # Test queries
#     test_queries = [
#         "2:255",  # Structural - Ayat al-Kursi
#         "surah fatiha",  # Structural - Surah name
#         "patience",  # Semantic/keyword
#         "bismillah",  # Keyword
#         "صبر",  # Arabic keyword
#         "what does quran say about prayer",  # Natural language
#     ]
    
#     print("\n" + "="*60)
#     print("SEARCH ENGINE TESTS")
#     print("="*60)
    
#     for query in test_queries:
#         print(f"\n🔍 Query: '{query}'")
#         print("-" * 60)
        
#         results = engine.search_formatted(query, top_k=3)
        
#         if not results:
#             print("  No results found")
#         else:
#             for i, result in enumerate(results, 1):
#                 print(f"\n{i}. {result['surah_name_english']} ({result['surah']}:{result['ayah']})")
#                 print(f"   Score: {result['relevance_score']:.3f} | Type: {result['match_type']}")
#                 print(f"   Arabic: {result['arabic'][:80]}...")
#                 print(f"   English: {result['translation_english'][:80]}...")