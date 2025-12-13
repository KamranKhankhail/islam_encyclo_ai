########################## improved chatGPT's e5-search implementation.
"""
E5 Hybrid Search Engine — Production-Grade (Qur'an-scale + Mobile-aligned)

Best-practice notes:
- E5 requires "query: " prefix for queries and "passage: " for documents. :contentReference[oaicite:3]{index=3}
- Hybrid fusion: RRF is a strong default because it fuses ranks without fragile score normalization. :contentReference[oaicite:4]{index=4}

Pipeline:
1) Structural parser → exact nav
2) Alias resolver → exact known names
3) BM25 lexical → topN
4) E5 semantic (full scan for Qur'an) → topN
5) Fuse (RRF) → topK
"""

import json
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from search_engine import QuranSearchEngine, QuranSearchConfig


# ----------------------------
# Config
# ----------------------------

@dataclass
class HybridConfig:
    enable_semantic: bool = True

    # Retrieval depths
    bm25_topn: int = 80
    semantic_topn: int = 80
    final_topk: int = 5

    # Semantic scan mode:
    # - "full": score all embeddings (recommended for Qur'an ~6k)
    # - "bm25": score only BM25 candidates (for huge corpora)
    semantic_scan_mode: str = "full"

    # Fusion
    fusion_method: str = "rrf"  # "rrf" recommended; "weighted_norm" optional
    rrf_k: int = 60             # typical starting constant :contentReference[oaicite:5]{index=5}
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
        if alias_map:
            for k, vk in alias_map.items():
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

        print(f"  ✓ Loaded embeddings: {self.verse_embeddings.shape} dtype={self.verse_embeddings.dtype} mmap={bool(mmap_mode)}")

        # Load E5 model
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

        print(f"  ✓ E5 model ready ({self.model_name}) on {device}")
        print(f"  ✓ Fusion: {self.hybrid.fusion_method} | BM25={self.hybrid.bm25_weight:.0%}, E5={self.hybrid.semantic_weight:.0%}")

    # ----------------------------
    # Public API
    # ----------------------------

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query or not str(query).strip():
            return []

        top_k = top_k or self.hybrid.final_topk

        # 1) Structural parser
        structural = self.structural_parser.parse(query)
        if structural:
            resolved = self._resolve_structural(structural)
            if resolved:
                return resolved[:top_k]

        # 2) Alias resolver
        alias_hit = self._resolve_alias(query)
        if alias_hit:
            return alias_hit[:top_k]

        # 3) BM25 lexical
        bm25_n = min(self.hybrid.bm25_topn, len(self.verses))
        bm25_results = self._lexical_search(query, bm25_n)

        if not self.hybrid.enable_semantic or self.semantic_model is None or self.verse_embeddings is None:
            return bm25_results[:top_k]

        # 4) Semantic retrieval
        sem_results = self._semantic_retrieve(query, bm25_results)

        # 5) Fuse
        return self._fuse(bm25_results, sem_results, top_k)

    def search_formatted(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return [self.format_result(v) for v in self.search(query, top_k)]

    # ----------------------------
    # Alias
    # ----------------------------

    def _norm_alias(self, s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKC", str(s))
        s = s.casefold()
        s = re.sub(r"[\u200c\u200d]", "", s)  # zero-width joiners
        s = " ".join(s.split())
        return s.strip()

    def _resolve_alias(self, query: str) -> Optional[List[Dict[str, Any]]]:
        k = self._norm_alias(query)
        vk = self.alias_map.get(k)
        if not vk:
            return None
        v = self._verse_by_key.get(vk)
        if not v:
            return None
        out = v.copy()
        out["match_type"] = "alias"
        out["relevance_score"] = 1.0
        return [out]

    # ----------------------------
    # Semantic retrieval
    # ----------------------------

    def _get_query_embedding(self, query: str) -> np.ndarray:
        # Cache normalized query embedding (very useful for repeated UI queries)
        key = query.strip()
        if key in self._qcache:
            return self._qcache[key]

        e5_query = f"query: {query}"  # required by E5 :contentReference[oaicite:6]{index=6}
        q_emb = self.semantic_model.encode(
            [e5_query],
            normalize_embeddings=True,
            convert_to_numpy=True
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

    # Expand over time (this is what makes named queries flawless)
    alias_map = {
        "ayat ul kursi": "2:255",
        "ayatul kursi": "2:255",
        "ayat al kursi": "2:255",
        "ayatulkursi": "2:255",
    }

    hybrid = HybridConfig(
        enable_semantic=True,
        bm25_weight=1.0 - float(semantic_weight),
        semantic_weight=float(semantic_weight),
        fusion_method="rrf",
        rrf_k=60,
        bm25_topn=80,
        semantic_topn=80,
        final_topk=5,
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
        alias_map=alias_map,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("E5 HYBRID SEARCH — PRODUCTION-GRADE DEMO")
    print("=" * 70)

    engine = get_e5_engine(semantic_weight=0.30)

    tests = [
        "2:255",
        "ayat ul kursi",
        "patience",
        "what does quran say about prayer",
        "صبر کے بارے میں آیت",
        "verses about charity and helping the poor",
    ]

    for q in tests:
        t0 = time.time()
        res = engine.search(q, top_k=3)
        ms = (time.time() - t0) * 1000.0
        print(f"\nQuery: {q}  |  {ms:.2f} ms")
        for i, r in enumerate(res, 1):
            print(f"  {i}. {r.get('verse_key')} | score={r.get('relevance_score'):.4f} | {r.get('match_type')}")
            if r.get("match_type") == "hybrid_fused":
                print(f"     bm25_rank={r.get('bm25_rank')} sem_rank={r.get('semantic_rank')} e5={r.get('e5_raw_similarity'):.4f}")




########################## original claude's embeddings implementation.
# """
# E5-Powered Hybrid Search Engine
# Research-optimized for maximum retrieval accuracy

# Key E5 Features:
# 1. "query: " prefix for search queries (CRITICAL!)
# 2. "passage: " prefix already in embeddings
# 3. Asymmetric retrieval architecture
# 4. Superior cross-lingual semantic matching
# """

# import json
# import numpy as np
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from sentence_transformers import SentenceTransformer

# # Import your excellent BM25 engine as base
# from search_engine import QuranSearchEngine, QuranSearchConfig


# class E5HybridSearchEngine(QuranSearchEngine):
#     """
#     E5-powered hybrid search: BM25 + E5 semantic reranking
    
#     Research-backed improvements over MiniLM:
#     - +4.6 NDCG@10 on retrieval benchmarks
#     - Superior cross-lingual understanding
#     - Optimized for query-passage matching (our exact use case)
    
#     Architecture:
#     1. BM25 retrieves top-50 candidates (fast, ~30ms)
#     2. E5 reranks with semantic similarity (~50-80ms)
#     3. Weighted combination (70% BM25, 30% E5)
#     4. Returns best top-K results
#     """
    
#     def __init__(
#         self,
#         data_path: str,
#         metadata_path: Optional[str] = None,
#         config: Optional[QuranSearchConfig] = None,
#         embeddings_path: Optional[str] = None,
#         enable_semantic: bool = True,
#         semantic_weight: float = 0.30,
#     ):
#         # Initialize base BM25 engine (your 91.7% baseline)
#         super().__init__(data_path, metadata_path, config)
        
#         self.enable_semantic = enable_semantic
#         self.semantic_weight = semantic_weight
#         self.bm25_weight = 1.0 - semantic_weight
        
#         if self.enable_semantic:
#             print("Loading E5 semantic components...")
            
#             # Load verse embeddings (generated with "passage: " prefix)
#             if embeddings_path is None:
#                 base_dir = Path(data_path).parent
#                 embeddings_path = base_dir / "verse_embeddings_e5.npy"
            
#             self.verse_embeddings = np.load(embeddings_path)
#             print(f"  ✓ Loaded E5 verse embeddings: {self.verse_embeddings.shape}")
            
#             # Load E5 model
#             print(f"  Loading intfloat/multilingual-e5-small...")
#             self.semantic_model = SentenceTransformer('intfloat/multilingual-e5-small')
#             print(f"  ✓ E5 model ready")
            
#             print(f"  ✓ Scoring weights: BM25={self.bm25_weight:.0%}, E5={self.semantic_weight:.0%}")
#         else:
#             self.verse_embeddings = None
#             self.semantic_model = None
    
#     def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """
#         E5-powered hybrid search
        
#         Args:
#             query: Search query (plain text, no prefix needed - we add it)
#             top_k: Number of final results
        
#         Returns:
#             Ranked list of verses with combined scoring
#         """
#         if not query or not query.strip():
#             return []
        
#         # Step 1: Structural query (exact navigation) - always 100%
#         structural = self.structural_parser.parse(query)
#         if structural:
#             resolved = self._resolve_structural(structural)
#             if resolved:
#                 return resolved
        
#         # Step 2: Get BM25 candidates
#         if self.enable_semantic:
#             # Get more candidates for E5 reranking
#             candidate_k = min(50, len(self.verses))
#             bm25_results = self._lexical_search(query, candidate_k)
#         else:
#             # Pure BM25 fallback
#             return self._lexical_search(query, top_k)
        
#         if not bm25_results:
#             return []
        
#         # Step 3: E5 semantic reranking
#         return self._e5_rerank(query, bm25_results, top_k)
    
#     def _e5_rerank(
#         self, 
#         query: str, 
#         candidates: List[Dict[str, Any]], 
#         top_k: int
#     ) -> List[Dict[str, Any]]:
#         """
#         Rerank BM25 candidates using E5 semantic similarity
        
#         E5 Scoring Formula:
#         - BM25 score (normalized): 70% weight (default)
#         - E5 semantic similarity: 30% weight (default)
        
#         These weights are tunable based on your evaluation results
#         """
#         if not candidates or not self.enable_semantic:
#             return candidates[:top_k]
        
#         # CRITICAL: E5 requires "query: " prefix for search queries
#         # This is what makes E5 excel at retrieval tasks
#         e5_query = f"query: {query}"
        
#         # Generate E5 query embedding
#         query_embedding = self.semantic_model.encode(
#             [e5_query], 
#             normalize_embeddings=True,
#             convert_to_numpy=True
#         )[0]
        
#         # Get candidate indices and BM25 scores
#         candidate_indices = []
#         bm25_scores = []
        
#         for candidate in candidates:
#             verse_key = candidate.get('verse_key')
#             if not verse_key:
#                 continue
            
#             # Find verse index in original dataset
#             idx = None
#             for i, v in enumerate(self.verses):
#                 if v.get('verse_key') == verse_key:
#                     idx = i
#                     break
            
#             if idx is not None:
#                 candidate_indices.append(idx)
#                 bm25_scores.append(candidate.get('relevance_score', 0.0))
        
#         if not candidate_indices:
#             return candidates[:top_k]
        
#         # Normalize BM25 scores to [0, 1]
#         bm25_scores = np.array(bm25_scores)
#         if bm25_scores.max() > 0:
#             bm25_scores_norm = bm25_scores / bm25_scores.max()
#         else:
#             bm25_scores_norm = bm25_scores
        
#         # Compute E5 semantic similarities
#         # Note: verse embeddings already have "passage: " prefix from generation
#         candidate_embeddings = self.verse_embeddings[candidate_indices]
#         e5_scores = np.dot(candidate_embeddings, query_embedding)
        
#         # Normalize E5 scores to [0, 1]
#         # Cosine similarity is in [-1, 1], map to [0, 1]
#         e5_scores_norm = (e5_scores + 1.0) / 2.0
        
#         # Combined scoring with configurable weights
#         combined_scores = (
#             self.bm25_weight * bm25_scores_norm + 
#             self.semantic_weight * e5_scores_norm
#         )
        
#         # Rank by combined score
#         ranked_indices = np.argsort(-combined_scores)[:top_k]
        
#         # Build final results
#         results = []
#         for rank_idx in ranked_indices:
#             original_idx = candidate_indices[rank_idx]
#             verse = self.verses[original_idx].copy()
            
#             # Add detailed scoring for analysis
#             verse['relevance_score'] = float(combined_scores[rank_idx])
#             verse['bm25_score'] = float(bm25_scores_norm[rank_idx])
#             verse['e5_score'] = float(e5_scores_norm[rank_idx])
#             verse['e5_raw_similarity'] = float(e5_scores[rank_idx])
#             verse['match_type'] = 'hybrid_e5'
            
#             results.append(verse)
        
#         return results
    
#     def search_formatted(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Search and return formatted results"""
#         results = self.search(query, top_k)
#         return [self.format_result(v) for v in results]
    
#     def tune_weights(self, bm25_weight: float):
#         """
#         Tune scoring weights based on evaluation results
        
#         Args:
#             bm25_weight: Weight for BM25 score (0-1)
#                         E5 weight will be (1 - bm25_weight)
        
#         Recommended starting points:
#         - bm25_weight=0.70 (default): Balanced
#         - bm25_weight=0.80: Favor lexical precision
#         - bm25_weight=0.60: Favor semantic recall
#         """
#         self.bm25_weight = max(0.0, min(1.0, bm25_weight))
#         self.semantic_weight = 1.0 - self.bm25_weight
#         print(f"Updated weights: BM25={self.bm25_weight:.0%}, E5={self.semantic_weight:.0%}")


# # Convenience functions

# def get_e5_engine(
#     semantic_weight: float = 0.30,
#     enable_cache: bool = True,
# ) -> E5HybridSearchEngine:
#     """
#     Get E5-powered hybrid engine with optimal settings
    
#     Args:
#         semantic_weight: E5 weight in [0, 1], default 0.30 (30%)
#         enable_cache: Use BM25 index caching for fast startup
#     """
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
#     metadata_path = base_dir / "output" / "processed" / "metadata.json"
#     embeddings_path = base_dir / "output" / "processed" / "verse_embeddings_e5.npy"
    
#     config = QuranSearchConfig(enable_cache=enable_cache)
    
#     return E5HybridSearchEngine(
#         data_path=str(data_path),
#         metadata_path=str(metadata_path) if metadata_path.exists() else None,
#         config=config,
#         embeddings_path=str(embeddings_path),
#         enable_semantic=True,
#         semantic_weight=semantic_weight,
#     )


# if __name__ == "__main__":
#     # E5 Hybrid Search Demo
#     print("="*70)
#     print("E5 HYBRID SEARCH ENGINE - RESEARCH-OPTIMIZED")
#     print("="*70)
    
#     engine = get_e5_engine()
    
#     test_queries = [
#         # Structural (should be 100%)
#         "2:255",
        
#         # Keyword (BM25 excels)
#         "patience",
        
#         # Semantic (E5 excels)
#         "what does quran say about prayer",
        
#         # Cross-lingual (E5's strength)
#         "صبر کے بارے میں آیت",  # Urdu semantic
        
#         # Complex semantic
#         "verses about charity and helping the poor",
#     ]
    
#     for query in test_queries:
#         print(f"\n🔍 Query: '{query}'")
#         print("-"*70)
        
#         results = engine.search_formatted(query, top_k=3)
        
#         if not results:
#             print("   No results found")
#             continue
        
#         for i, r in enumerate(results, 1):
#             print(f"\n{i}. {r['surah_name_english']} ({r['verse_key']})")
#             print(f"   📊 Combined: {r['relevance_score']:.3f}")
            
#             if 'bm25_score' in r and 'e5_score' in r:
#                 print(f"      BM25: {r['bm25_score']:.3f} | E5: {r['e5_score']:.3f}")
#                 if 'e5_raw_similarity' in r:
#                     print(f"      E5 raw cosine: {r['e5_raw_similarity']:.3f}")
            
#             print(f"   🏷️  Type: {r['match_type']}")
#             print(f"   🇬🇧 EN: {r['translation_english'][:65]}...")