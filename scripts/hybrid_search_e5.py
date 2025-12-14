########################## improved chatGPT's e5-search implementation.
"""
E5 Hybrid Search Engine - Production Grade (Qur'an scale + mobile aligned)

Best-practice notes:
- E5 requires "query: " prefix for queries and "passage: " for documents.
- Hybrid fusion: RRF is a strong default because it fuses ranks without fragile score normalization.

Pipeline:
1) Structural parser + exact navigation
2) Alias resolver + exact known names
3) BM25 lexical + topN
4) E5 semantic (full scan for Qur'an) + topN
5) Fuse (RRF) + topK
"""

import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from arabic_normalizer import normalize_for_search
from response_format_v1 import (
    DEFAULT_PACK_VERSION,
    build_ayah_result_v1,
    build_search_response_v1,
    qround,
)
from search_engine import QuranSearchEngine, QuranSearchConfig


# Shared defaults
DEFAULT_SEMANTIC_TOPN = 1000
DEFAULT_MAX_RESULTS = 300

# Default alias expansions for common verse nicknames and misspellings
DEFAULT_ALIAS_MAP: Dict[str, str] = {
    # === Ayat al-Kursi (The Throne Verse) ===
    "ayat ul kursi": "2:255",
    "ayatul kursi": "2:255",
    "ayat al kursi": "2:255",
    "ayatulkursi": "2:255",
    "throne verse": "2:255",
    "kursi verse": "2:255",
    "ayat kursi": "2:255",

    # === Last Two Verses of Al-Baqarah ===
    "last two verses of baqarah": "2:285-286",
    "amanarrasul": "2:285-286",
    "aman ar rasul": "2:285-286",
    "aaman ar rasuul": "2:285-286",

    # === Al-Fatihah (The Opening) ===
    "al fatiha": "1:1-7",
    "fatiha": "1:1-7",
    "surah fatiha": "1:1-7",
    "al fatihah": "1:1-7",
    "the opening": "1:1-7",
    "surah al fatiha": "1:1-7",
    "fatiḥa": "1:1-7",

    # === Al-Ikhlas (The Sincerity) ===
    "al ikhlas": "112:1-4",
    "ikhlas": "112:1-4",
    "surah ikhlas": "112:1-4",
    "tawhid": "112:1-4",
    "monotheism": "112:1-4",
    "sincerity": "112:1-4",
    "qul hu allahu ahad": "112:1-4",

    # === Al-Falaq (The Daybreak) ===
    "al falaq": "113:1-5",
    "falaq": "113:1-5",
    "surah falaq": "113:1-5",
    "daybreak": "113:1-5",

    # === An-Nas (Mankind) ===
    "an nas": "114:1-6",
    "nas": "114:1-6",
    "surah nas": "114:1-6",
    "mankind": "114:1-6",

    # === The Four Quls ===
    "four quls": "109,112-114",
    "qul surahs": "109,112-114",
    "char qul": "109,112-114",

    # === Ayat al-Birr (Righteousness Verse) ===
    "ayat al birr": "2:177",
    "birr verse": "2:177",
    "righteousness verse": "2:177",
    "ayatul birr": "2:177",

    # === Light Verse (Ayat an-Nur) ===
    "ayat al nur": "24:35",
    "ayat an nur": "24:35",
    "nur verse": "24:35",
    "light verse": "24:35",
    "ayatun nur": "24:35",

    # === Debt Verse (Ayat ad-Dayn) ===
    "ayat al dayn": "2:282",
    "debt verse": "2:282",
    "longest verse": "2:282",
    "ayat ud dayn": "2:282",

    # === Khilafah Verse ===
    "khilafah verse": "24:55",
    "caliphate verse": "24:55",
    "succession verse": "24:55",

    # === Sabaq Verse (Patience) ===
    "sabr verse": "2:153",
    "patience verse": "2:153",
    "ayat as sabr": "2:153",

    # === Hajj Verse ===
    "hajj verse": "3:97",
    "hajj obligation": "3:97",

    # === Parents Verse ===
    "parents verse": "17:23-24",
    "respect parents": "17:23-24",

    # === Marriage Verses ===
    "marriage verse": "4:1",
    "spouse verse": "30:21",
    "ayat an nisa": "4:1",

    # === Al-Kahf (The Cave) ===
    "al kahf": "18:1-110",
    "kahf": "18:1-110",
    "surah kahf": "18:1-110",
    "the cave": "18:1-110",

    # === Yaseen ===
    "yaseen": "36:1-83",
    "ya seen": "36:1-83",
    "surah yasin": "36:1-83",
    "heart of quran": "36:1-83",

    # === Ar-Rahman ===
    "ar rahman": "55:1-78",
    "rahman": "55:1-78",
    "surah rahman": "55:1-78",

    # === Al-Waqiah ===
    "al waqiah": "56:1-96",
    "waqiah": "56:1-96",
    "surah waqiah": "56:1-96",
    "the event": "56:1-96",

    # === Al-Mulk ===
    "al mulk": "67:1-30",
    "mulk": "67:1-30",
    "surah mulk": "67:1-30",
    "the sovereignty": "67:1-30",
    "tabarak": "67:1-30",

    # === Short Surahs for Prayer ===
    "al kafirun": "109:1-6",
    "kafirun": "109:1-6",
    "surah kafiroon": "109:1-6",

    # === Juz/Para References ===
    "juz amma": "78-114",
    "para 30": "78-114",
    "juz 30": "78-114",
    "amma": "78-114",

    "juz tabarak": "67-77",
    "para 29": "67-77",
    "juz 29": "67-77",

    "sajdah tilawah": "32:15",
    "prostration verse": "32:15",

    # === Morning/Evening Adhkar Verses ===
    "ayat al kursi morning": "2:255",
    "three quls": "112-114",

    # === Common Misspellings/Variations ===
    "ayatul kursiy": "2:255",
    "ayatulkursee": "2:255",
    "ayat el kursi": "2:255",

    "alfatiha": "1:1-7",
    "alfatihah": "1:1-7",
    "surah alfatiha": "1:1-7",

    "alikhlas": "112:1-4",
    "iklas": "112:1-4",
    "qulhu": "112:1-4",

    # === Special Groups ===
    "muawwadhāt": "113-114",
    "al-mu'awwidhatayn": "113-114",
    "the two protectors": "113-114",

    "musabbihat": "57,59,61,62,64",
    "al-musabbihat": "57,59,61,62,64",

    # === First Revelation ===
    "first revelation": "96:1-5",
    "iqra": "96:1-5",
    "alaq": "96:1-5",

    # === Last Revelation ===
    "last revelation": "5:3",
    "today perfected": "5:3",

    # === Specific Famous Verses ===
    "no compulsion": "2:256",
    "la ikraha": "2:256",
    "no compulsion in religion": "2:256",

    "half of knowledge": "2:269",
    "wisdom verse": "2:269",

    "oppression verse": "2:286",
    "allah burdens not": "2:286",

    # === Frequently Recited in Prayer ===
    "sajdah verse": "32:15",
    "sajda verse": "32:15",

    # === Dua/Verses for Protection ===
    "ayat al hifz": "41:36",
    "protection verse": "41:36",

    # === Medical/Sickness Verses ===
    "shifa verses": "17:82",
    "healing verses": "17:82",
    "ruqyah verses": "2:255, 1:1-7, 112:1-4, 113:1-5, 114:1-6",
}


# ----------------------------
# Config
# ----------------------------

@dataclass
class HybridConfig:
    enable_semantic: bool = True

    # Retrieval depths
    bm25_topn: int = 80
    semantic_topn: int = DEFAULT_SEMANTIC_TOPN
    final_topk: int = DEFAULT_MAX_RESULTS
    max_results: int = DEFAULT_MAX_RESULTS

    # Semantic scan mode:
    # - "full": score all embeddings (recommended for Qur'an ~6k)
    # - "bm25": score only BM25 candidates (for huge corpora)
    semantic_scan_mode: str = "full"

    # Fusion
    fusion_method: str = "rrf"  # "rrf" recommended; "weighted_norm" optional
    rrf_k: int = 60             # typical starting constant
    bm25_weight: float = 0.70
    semantic_weight: float = 0.30

    # Embeddings loading/perf
    mmap_embeddings: bool = True
    semantic_chunk_rows: int = 0  # 0 = no chunking; else chunk size for large corpora

    # Query embedding cache (small LRU-ish)
    query_cache_size: int = 256

    # Prefer CUDA for encoding (if available)
    prefer_cuda: bool = True


# ----------------------------
# Engine
# ----------------------------

class E5HybridSearchEngine(QuranSearchEngine):
    def __init__(
        self,
        data_path: str,
        metadata_path: Optional[str] = None,
        config: Optional[QuranSearchConfig] = None,
        embeddings_path: Optional[str] = None,
        verse_keys_path: Optional[str] = None,
        model_name: str = "intfloat/multilingual-e5-small",
        hybrid: Optional[HybridConfig] = None,
        alias_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(data_path, metadata_path, config)

        self.hybrid = hybrid or HybridConfig()
        self.model_name = model_name

        # Verse maps for O(1) lookups
        self._verse_by_key: Dict[str, Dict[str, Any]] = {}
        for v in self.verses:
            vk = v.get("verse_key")
            if vk:
                self._verse_by_key[vk] = v

        # Alias map (normalized)
        self.alias_map: Dict[str, str] = {}
        alias_source = alias_map or DEFAULT_ALIAS_MAP
        if alias_source:
            for k, vk in alias_source.items():
                self.alias_map[self._norm_alias(k)] = vk

        # Semantic components
        self.semantic_model: Optional[SentenceTransformer] = None
        self.verse_embeddings: Optional[np.ndarray] = None
        self._emb_keys_list: Optional[List[str]] = None
        self._emb_row_by_key: Dict[str, int] = {}

        # Small query embedding cache
        self._qcache: Dict[str, np.ndarray] = {}
        self._qcache_order: List[str] = []

        if not self.hybrid.enable_semantic:
            return

        base_dir = Path(data_path).parent
        embeddings_path = str(embeddings_path or (base_dir / "verse_embeddings_e5.npy"))
        verse_keys_path = str(verse_keys_path or (base_dir / "verse_keys_e5.json"))

        print("Loading E5 semantic components...")

        # Load verse_keys and validate uniqueness
        if not Path(verse_keys_path).exists():
            raise FileNotFoundError(f"Missing verse_keys file: {verse_keys_path}")
        with open(verse_keys_path, "r", encoding="utf-8") as f:
            emb_keys = json.load(f)
        if not isinstance(emb_keys, list) or not emb_keys:
            raise ValueError("verse_keys_e5.json must be a non-empty list")

        # Duplicate check
        if len(set(emb_keys)) != len(emb_keys):
            raise ValueError("verse_keys_e5.json contains duplicate verse keys (invalid alignment)")

        self._emb_keys_list = emb_keys
        self._emb_row_by_key = {vk: i for i, vk in enumerate(emb_keys)}

        # Warn if some embedded keys are missing in dataset (should not happen)
        missing_in_data = [vk for vk in emb_keys if vk not in self._verse_by_key]
        if missing_in_data:
            # Not fatal for search (we will skip those rows), but it indicates a pipeline mismatch.
            print(f"  ! Warning: {len(missing_in_data)} embedding keys not found in dataset. Example: {missing_in_data[0]}")

        # Load embeddings
        if not Path(embeddings_path).exists():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

        mmap_mode = "r" if self.hybrid.mmap_embeddings else None
        self.verse_embeddings = np.load(embeddings_path, mmap_mode=mmap_mode)

        if self.verse_embeddings.shape[0] != len(emb_keys):
            raise ValueError(
                f"Embeddings rows ({self.verse_embeddings.shape[0]}) != verse_keys count ({len(emb_keys)})"
            )

        print(f"  ✓ Loaded E5 verse embeddings: {self.verse_embeddings.shape} dtype={self.verse_embeddings.dtype} mmap={bool(mmap_mode)}")
        print(f"  Loading {self.model_name}...")
        self.semantic_model = SentenceTransformer(self.model_name)

        device = "cpu"
        if self.hybrid.prefer_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                device = "cpu"

        try:
            self.semantic_model = self.semantic_model.to(device)
        except Exception:
            pass

        # Warm-up once to smooth out first-query latency
        try:
            import torch

            with torch.inference_mode():
                _ = self.semantic_model.encode(
                    ["query: warmup"],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception:
            _ = self.semantic_model.encode(
                ["query: warmup"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        print(f"  ✓ E5 model ready ({self.model_name}) on {device}")
        print(f"  ✓ Scoring weights: BM25={self.hybrid.bm25_weight:.0%}, E5={self.hybrid.semantic_weight:.0%}")

    # ----------------------------
    # Public API
    # ----------------------------

    @staticmethod
    def _vk_tuple(vk: str) -> Tuple[int, int]:
        try:
            s, a = str(vk).split(":")
            return int(s), int(a)
        except Exception:
            return (10**9, 10**9)

    def _stable_rank(
        self,
        keys: List[str],
        score_by_key: Dict[str, Dict[str, Optional[float]]],
        primary_field: str,
    ) -> List[str]:
        def _score(vk: str) -> float:
            val = score_by_key.get(vk, {}).get(primary_field)
            if val is None:
                return float("-inf")
            try:
                return float(val)
            except Exception:
                return float("-inf")

        return sorted(keys, key=lambda k: (-_score(k), self._vk_tuple(k), k))

    @staticmethod
    def _ensure_score_entry(score_by_key: Dict[str, Dict[str, Optional[float]]], vk: str) -> Dict[str, Optional[float]]:
        if vk not in score_by_key:
            score_by_key[vk] = {"bm25": None, "semantic": None, "rrf": None}
        return score_by_key[vk]

    def _add_score(self, score_by_key: Dict[str, Dict[str, Optional[float]]], vk: str, field: str, value: Any) -> None:
        entry = self._ensure_score_entry(score_by_key, vk)
        entry[field] = qround(value)

    def _format_results(
        self,
        ranked_keys: List[str],
        score_by_key: Dict[str, Dict[str, Optional[float]]],
        ui_lang: str,
        mode: str,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for vk in ranked_keys:
            verse = self._verse_by_key.get(vk)
            if not verse:
                continue
            scores = score_by_key.get(vk, {})
            out.append(
                build_ayah_result_v1(
                    verse=verse,
                    mode=mode,
                    bm25=scores.get("bm25"),
                    semantic=scores.get("semantic"),
                    rrf=scores.get("rrf"),
                    ui_lang=ui_lang,
                )
            )
        return out

    def _cap_limit(self, limit: int) -> int:
        return max(0, min(int(limit), int(self.hybrid.max_results)))

    def _sort_structural_keys(self, verses: List[Dict[str, Any]]) -> List[str]:
        keys = []
        for v in verses:
            vk = v.get("verse_key") or v.get("id")
            if vk:
                keys.append(str(vk))
        keys = list(dict.fromkeys(keys))
        return sorted(keys, key=lambda k: self._vk_tuple(k))

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        ui_lang: str = "en",
        pack_version: str = DEFAULT_PACK_VERSION,
    ) -> Dict[str, Any]:
        raw_query = "" if query is None else str(query)
        normalized_query = normalize_for_search(raw_query) if raw_query else ""
        intent = "lexical"
        limit = self._cap_limit(top_k if top_k is not None else self.hybrid.final_topk)

        if not raw_query.strip():
            return build_search_response_v1(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                ui_lang=ui_lang,
                results=[],
                pack_version=pack_version,
            )

        # 1) Structural parser
        structural = self.structural_parser.parse(raw_query)
        if structural:
            resolved = self._resolve_structural(structural)
            if resolved:
                intent = "structural"
                ranked_keys = self._sort_structural_keys(resolved)[:limit]
                score_by_key = {vk: {"bm25": None, "semantic": None, "rrf": None} for vk in ranked_keys}
                results = self._format_results(ranked_keys, score_by_key, ui_lang, mode=intent)
                return build_search_response_v1(
                    raw_query=raw_query,
                    normalized_query=normalized_query,
                    intent=intent,
                    ui_lang=ui_lang,
                    results=results,
                    pack_version=pack_version,
                )

        # 2) Alias resolver (treated as structural)
        alias_hit = self._resolve_alias(raw_query)
        if alias_hit:
            intent = "structural"
            ranked_keys = self._sort_structural_keys(alias_hit)[:limit]
            score_by_key = {vk: {"bm25": None, "semantic": None, "rrf": None} for vk in ranked_keys}
            results = self._format_results(ranked_keys, score_by_key, ui_lang, mode=intent)
            return build_search_response_v1(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                ui_lang=ui_lang,
                results=results,
                pack_version=pack_version,
            )

        # 3) BM25 lexical
        bm25_n = min(self.hybrid.bm25_topn, len(self.verses))
        bm25_results = self._lexical_search(raw_query, bm25_n)

        score_by_key: Dict[str, Dict[str, Optional[float]]] = {}
        for r in bm25_results:
            vk = r.get("verse_key")
            if not vk:
                continue
            self._add_score(score_by_key, str(vk), "bm25", r.get("relevance_score"))

        if not self.hybrid.enable_semantic or self.semantic_model is None or self.verse_embeddings is None:
            ranked_keys = self._stable_rank(list(score_by_key.keys()), score_by_key, "bm25")[:limit]
            results = self._format_results(ranked_keys, score_by_key, ui_lang, mode=intent)
            return build_search_response_v1(
                raw_query=raw_query,
                normalized_query=normalized_query,
                intent=intent,
                ui_lang=ui_lang,
                results=results,
                pack_version=pack_version,
            )

        # 4) Semantic retrieval
        sem_results = self._semantic_retrieve(raw_query, bm25_results)
        for r in sem_results:
            vk = r.get("verse_key")
            if not vk:
                continue
            self._add_score(score_by_key, str(vk), "semantic", r.get("e5_raw_similarity"))

        # 5) Fusion (RRF with deterministic tie-break)
        intent = "hybrid"
        bm25_rank = {
            vk: idx + 1
            for idx, vk in enumerate(
                self._stable_rank(
                    [k for k, v in score_by_key.items() if v.get("bm25") is not None],
                    score_by_key,
                    "bm25",
                )
            )
        }
        sem_rank = {
            vk: idx + 1
            for idx, vk in enumerate(
                self._stable_rank(
                    [k for k, v in score_by_key.items() if v.get("semantic") is not None],
                    score_by_key,
                    "semantic",
                )
            )
        }

        k_const = max(1, int(self.hybrid.rrf_k))
        for vk in list(score_by_key.keys()):
            rrf_score = 0.0
            rb = bm25_rank.get(vk)
            rs = sem_rank.get(vk)
            if rb is not None:
                rrf_score += float(self.hybrid.bm25_weight) * (1.0 / (k_const + rb))
            if rs is not None:
                rrf_score += float(self.hybrid.semantic_weight) * (1.0 / (k_const + rs))
            self._add_score(score_by_key, vk, "rrf", rrf_score)

        ranked_keys = self._stable_rank(list(score_by_key.keys()), score_by_key, "rrf")[:limit]
        results = self._format_results(ranked_keys, score_by_key, ui_lang, mode=intent)
        return build_search_response_v1(
            raw_query=raw_query,
            normalized_query=normalized_query,
            intent=intent,
            ui_lang=ui_lang,
            results=results,
            pack_version=pack_version,
        )

    def search_formatted(self, query: str, top_k: int = 5, ui_lang: str = "en") -> List[Dict[str, Any]]:
        resp = self.search(query, top_k=top_k, ui_lang=ui_lang)
        return resp.get("results", [])

    # ----------------------------
    # Alias
    # ----------------------------

    def _parse_alias_target(self, target: str) -> List[Dict[str, Any]]:
        """
        Parse alias target strings into structural specs consumable by _resolve_structural.
        Supports ayah ranges, surah ranges/lists, and comma-separated groups.
        """
        specs: List[Dict[str, Any]] = []
        parts = [p.strip() for p in str(target).split(",") if p.strip()]

        for p in parts:
            # Ayah range within a surah, e.g., 2:285-286
            m = re.fullmatch(r"(\d+):(\d+)-(\d+)", p)
            if m:
                s, a1, a2 = map(int, m.groups())
                specs.append({"surah": s, "ayah": a1, "ayah_end": a2})
                continue

            # Single ayah, e.g., 2:255
            m = re.fullmatch(r"(\d+):(\d+)", p)
            if m:
                s, a = m.groups()
                specs.append({"verse_key": f"{int(s)}:{int(a)}"})
                continue

            # Surah range, e.g., 112-114 -> first ayah of each
            m = re.fullmatch(r"(\d+)-(\d+)", p)
            if m:
                s1, s2 = map(int, m.groups())
                if s2 < s1:
                    s1, s2 = s2, s1
                for surah in range(s1, s2 + 1):
                    specs.append({"verse_key": f"{surah}:1"})
                continue

            # Single surah number -> first ayah
            m = re.fullmatch(r"(\d+)", p)
            if m:
                surah = int(m.group(1))
                specs.append({"verse_key": f"{surah}:1"})
                continue

        return specs

    def _norm_alias(self, s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKC", str(s))
        s = s.casefold()
        s = re.sub(r"[\u200c\u200d]", "", s)  # zero-width joiners
        s = " ".join(s.split())
        return s.strip()

    def _resolve_alias(self, query: str) -> Optional[List[Dict[str, Any]]]:
        k = self._norm_alias(query)
        target = self.alias_map.get(k)
        if not target:
            return None
        specs = self._parse_alias_target(target)
        if not specs:
            return None
        results: List[Dict[str, Any]] = []
        for spec in specs:
            resolved = self._resolve_structural(spec)
            if not resolved:
                continue
            for r in resolved:
                out = r.copy()
                out["match_type"] = "alias"
                out["relevance_score"] = 1.0
                results.append(out)
        if not results:
            return None
        return results

    # ----------------------------
    # Semantic retrieval
    # ----------------------------

    def _get_query_embedding(self, query: str) -> np.ndarray:
        # Cache normalized query embedding (very useful for repeated UI queries)
        key = query.strip()
        if key in self._qcache:
            return self._qcache[key]

        e5_query = f"query: {query}"  # required by E5
        try:
            import torch
            with torch.inference_mode():
                q_emb = self.semantic_model.encode(
                    [e5_query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )[0].astype(np.float32, copy=False)
        except Exception:
            q_emb = self.semantic_model.encode(
                [e5_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].astype(np.float32, copy=False)

        # Insert into simple LRU-ish cache
        self._qcache[key] = q_emb
        self._qcache_order.append(key)
        if len(self._qcache_order) > self.hybrid.query_cache_size:
            old = self._qcache_order.pop(0)
            self._qcache.pop(old, None)

        return q_emb

    def _semantic_retrieve(self, query: str, bm25_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mode = (self.hybrid.semantic_scan_mode or "full").lower()
        q_emb = self._get_query_embedding(query)

        if mode == "bm25":
            # Score only BM25 candidates
            keys = []
            for r in bm25_results:
                vk = r.get("verse_key")
                if vk and vk in self._emb_row_by_key:
                    keys.append(vk)
            keys = list(dict.fromkeys(keys))  # dedupe preserve order
            if not keys:
                return []

            rows = np.array([self._emb_row_by_key[vk] for vk in keys], dtype=np.int64)
            emb = self.verse_embeddings[rows]
            sims = emb @ q_emb
            order = np.argsort(-sims)
            out = []
            for j in order[: min(self.hybrid.semantic_topn, len(order))]:
                vk = keys[int(j)]
                verse = self._verse_by_key.get(vk)
                if not verse:
                    continue
                d = verse.copy()
                d["match_type"] = "semantic_e5"
                d["e5_raw_similarity"] = float(sims[int(j)])
                out.append(d)
            return out

        # Default: full scan (best recall for Qur'an scale)
        sims = self._semantic_full_scan(q_emb)

        topn = min(self.hybrid.semantic_topn, sims.shape[0])
        idx = np.argpartition(-sims, topn - 1)[:topn]
        idx = idx[np.argsort(-sims[idx])]

        out = []
        for r in idx:
            vk = self._emb_keys_list[int(r)]
            verse = self._verse_by_key.get(vk)
            if not verse:
                continue
            d = verse.copy()
            d["match_type"] = "semantic_e5"
            d["e5_raw_similarity"] = float(sims[int(r)])
            out.append(d)
        return out

    def _semantic_full_scan(self, q_emb: np.ndarray) -> np.ndarray:
        emb = self.verse_embeddings

        # If chunking requested (useful for very large corpora later)
        chunk = int(self.hybrid.semantic_chunk_rows or 0)
        if chunk <= 0 or emb.shape[0] <= chunk:
            return emb @ q_emb

        # Chunked scan (keeps memory stable)
        sims = np.empty((emb.shape[0],), dtype=np.float32)
        for start in range(0, emb.shape[0], chunk):
            end = min(start + chunk, emb.shape[0])
            sims[start:end] = emb[start:end] @ q_emb
        return sims

    # ----------------------------
    # Fusion (RRF recommended)
    # ----------------------------

    def _fuse(self, bm25: List[Dict[str, Any]], sem: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        method = (self.hybrid.fusion_method or "rrf").lower()

        bm25_rank: Dict[str, int] = {}
        bm25_raw: Dict[str, float] = {}
        for i, r in enumerate(bm25, start=1):
            vk = r.get("verse_key")
            if vk and vk not in bm25_rank:
                bm25_rank[vk] = i
                bm25_raw[vk] = float(r.get("relevance_score", 0.0) or 0.0)

        sem_rank: Dict[str, int] = {}
        sem_raw: Dict[str, float] = {}
        for i, r in enumerate(sem, start=1):
            vk = r.get("verse_key")
            if vk and vk not in sem_rank:
                sem_rank[vk] = i
                sem_raw[vk] = float(r.get("e5_raw_similarity", 0.0) or 0.0)

        keys = set(bm25_rank) | set(sem_rank)
        if not keys:
            return []

        scored: List[Tuple[str, float]] = []

        if method == "rrf":
            k = max(1, int(self.hybrid.rrf_k))
            w_b = float(self.hybrid.bm25_weight)
            w_s = float(self.hybrid.semantic_weight)
            for vk in keys:
                s = 0.0
                rb = bm25_rank.get(vk)
                rs = sem_rank.get(vk)
                if rb is not None:
                    s += w_b * (1.0 / (k + rb))
                if rs is not None:
                    s += w_s * (1.0 / (k + rs))
                scored.append((vk, s))

        elif method == "weighted_norm":
            # More sensitive than RRF; keep for experiments only
            def minmax(d: Dict[str, float]) -> Dict[str, float]:
                if not d:
                    return {}
                vals = np.array(list(d.values()), dtype=np.float32)
                mn, mx = float(vals.min()), float(vals.max())
                denom = (mx - mn) if (mx - mn) > 1e-9 else 1.0
                return {k: (v - mn) / denom for k, v in d.items()}

            b = minmax(bm25_raw)
            s = minmax(sem_raw)

            w_b = float(self.hybrid.bm25_weight)
            w_s = float(self.hybrid.semantic_weight)
            for vk in keys:
                scored.append((vk, w_b * b.get(vk, 0.0) + w_s * s.get(vk, 0.0)))

        else:
            raise ValueError(f"Unknown fusion_method: {self.hybrid.fusion_method}")

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        out: List[Dict[str, Any]] = []
        for vk, score in scored:
            verse = self._verse_by_key.get(vk)
            if not verse:
                continue
            d = verse.copy()
            d["match_type"] = "hybrid_fused"
            d["relevance_score"] = float(score)
            d["bm25_rank"] = bm25_rank.get(vk)
            d["semantic_rank"] = sem_rank.get(vk)
            d["bm25_raw"] = bm25_raw.get(vk, 0.0)
            d["e5_raw_similarity"] = sem_raw.get(vk, 0.0)
            out.append(d)

        return out


# ----------------------------
# Factory / Demo
# ----------------------------

def get_e5_engine(
    semantic_weight: float = 0.30,
    enable_cache: bool = True,
) -> E5HybridSearchEngine:
    base_dir = Path(__file__).parent.parent

    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    metadata_path = base_dir / "output" / "processed" / "metadata.json"

    embeddings_path = base_dir / "output" / "processed" / "verse_embeddings_e5.npy"
    verse_keys_path = base_dir / "output" / "processed" / "verse_keys_e5.json"

    config = QuranSearchConfig(enable_cache=enable_cache)

    hybrid = HybridConfig(
        enable_semantic=True,
        bm25_weight=1.0 - float(semantic_weight),
        semantic_weight=float(semantic_weight),
        fusion_method="rrf",
        rrf_k=60,
        bm25_topn=80,
        semantic_topn=DEFAULT_SEMANTIC_TOPN,
        final_topk=DEFAULT_MAX_RESULTS,
        max_results=DEFAULT_MAX_RESULTS,
        semantic_scan_mode="full",
        mmap_embeddings=True,
        prefer_cuda=True,
        query_cache_size=256,
        semantic_chunk_rows=0,
    )

    return E5HybridSearchEngine(
        data_path=str(data_path),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        config=config,
        embeddings_path=str(embeddings_path),
        verse_keys_path=str(verse_keys_path),
        model_name="intfloat/multilingual-e5-small",
        hybrid=hybrid,
        alias_map=DEFAULT_ALIAS_MAP,
    )


def _format_translation_en(result: Dict[str, Any]) -> str:
    display = result.get("display") or {}
    text = display.get("text") or ""
    return str(text).strip()


def _log_result_block(result: dict, idx: int) -> None:
    verse_key = result.get("verseKey", "n/a")
    score = result.get("score") or {}

    print(f"{idx}. {verse_key}")
    print("   mode={mode} | rrf={rrf} | bm25={bm25} | semantic={semantic}".format(
        mode=score.get('mode'), rrf=score.get('rrf'), bm25=score.get('bm25'), semantic=score.get('semantic'),
    ))

    en_text = _format_translation_en(result)
    if en_text:
        preview = (en_text[:90] + '...') if len(en_text) > 90 else en_text
        print(f"   text: {preview}")


if __name__ == "__main__":
    print("E5 HYBRID SEARCH ENGINE - RESEARCH-OPTIMIZED")
    print("=" * 70)
    print("Loading BM25 index from cache...")

    engine = get_e5_engine(semantic_weight=0.30)

    tests = [
        "2:255",
        "patience",
        "صبر کے بارے میں آیت",
    ]

    for q in tests:
        print(f"\nQuery: '{q}'")
        print("-" * 70)
        t0 = time.time()
        resp = engine.search(q, top_k=3)
        results = resp.get("results", [])
        latency_ms = (time.time() - t0) * 1000.0
        if not results:
            print("  No results found")
            continue
        intent = resp.get("query", {}).get("intent")
        print(f"  intent={intent} total={len(results)}")
        for i, res in enumerate(results, 1):
            _log_result_block(res, i)
        print(f"   (latency: {latency_ms:.2f} ms)")
