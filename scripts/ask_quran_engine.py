"""
AskQur'an Orchestrator (Offline, Mobile-First)

Purpose
- Route user queries to the right retrieval mode:
  1) Structural (surah/ayah references)
  2) Quote/fragment lookup ("Where is this from?")
  3) Topic/Story retrieval ("Where to spend zakat?", "What happened to Qawm-e-Lut?")
  4) Counting (exact word/phrase counts; semantic "instruction/theme" counts)

Design principles
- Deterministic, explainable, offline, and fast.
- No brittle hand-crafted rule forest; only high-precision intent routing.
- Retrieval is delegated to E5HybridSearchEngine (BM25F + E5 semantic fusion).
- Evidence is grouped into ayah-ranges for best UX (users read contiguous context).

This module is pure-Python reference code. For production mobile:
- Keep the same interfaces (router → retrieval → grouping → response),
  but move embedding inference + vector ops to ONNX/NNAPI/CoreML and
  memory-map embeddings for speed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from query_parser import StructuralQueryParser
from arabic_normalizer import normalize_for_search

# Prefer the patched engine (adds semantic_search + fixes semantic relevance_score).
try:
    from hybrid_search_e5_v2 import E5HybridSearchEngine
except Exception:
    from hybrid_search_e5 import E5HybridSearchEngine

try:
    from count_index import QuranCountIndex
except Exception:
    QuranCountIndex = None  # type: ignore

# Resolve project-root paths for local artifacts.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "output" / "processed" / "quran_complete.json"
DEFAULT_EMB_PATH = PROJECT_ROOT / "output" / "processed" / "verse_embeddings_e5.npy"
DEFAULT_KEYS_PATH = PROJECT_ROOT / "output" / "processed" / "verse_keys_e5.json"
DEFAULT_COUNT_INDEX_PATH = PROJECT_ROOT / "output" / "processed" / "count_index.pkl.gz"

# Optional numpy import for safe JSON casting (results may contain np scalars).
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


def _to_jsonable(obj: Any) -> Any:
    """Recursively cast engine outputs into JSON-serializable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # Fallback to string to avoid serialization errors.
    return str(obj)


@dataclass(frozen=True)
class AskQuranConfig:
    # Retrieval
    top_k_default: int = 7
    top_k_story: int = 45              # "what happened to..." style queries
    top_k_quote: int = 7               # quote/fragment lookup
    group_gap_ayah: int = 2            # merge near-by ayahs (context window)

    # Count
    max_list_verses_in_count: int = 50
    semantic_count_candidates: int = 200
    semantic_count_min_keep: int = 10

    # Heuristic thresholds for quote detection
    quote_min_len: int = 20
    quote_min_arabic_ratio: float = 0.20


class AskQuranEngine:
    """
    High-level "Ask Qur'an" engine that returns:
      - intent
      - normalized query (optional)
      - results grouped for UX
      - optional "count" response
      - debug fields (optional, safe)

    The engine *does not generate* free-form answers. It retrieves Qur'anic evidence
    (ayahs / ayah-ranges). An LLM layer can be added later, but v1 is evidence-first.
    """

    def __init__(
        self,
        engine: E5HybridSearchEngine,
        count_index: Optional["QuranCountIndex"] = None,
        config: AskQuranConfig = AskQuranConfig(),
    ):
        self.engine = engine
        self.config = config
        self.structural = StructuralQueryParser()
        self.count_index = count_index

        # Precompiled intent regex
        self._re_count = re.compile(
            r"\b(how\s+many\s+times|how\s+many|count|occurrences?|frequency)\b|"
            r"(کتنی\s*بار|کتنی\s*دفعہ|کِنّی\s*وار)|"
            r"(كم\s+مرة|كم\s+مرّة|عدد\s+مرات)",
            re.IGNORECASE,
        )
        self._re_where_from = re.compile(
            r"\b(where\s+is\s+this\s+from|which\s+ayah|which\s+verse|reference)\b|"
            r"(یہ\s+کہاں\s+سے\s+ہے|یہ\s+کس\s+آیت|حوالہ)|"
            r"(من\s+أين\s+هذا|أين\s+هذه\s+الآية|ما\s+سورة|ما\s+آية)",
            re.IGNORECASE,
        )
        self._re_story = re.compile(
            r"\b(what\s+happened|story\s+of|who\s+was|who\s+were|people\s+of)\b|"
            r"(کیا\s+ہوا|قصہ|واقعہ|قوم\s+)\b|"
            r"(ماذا\s+حدث|قصة|قوم)\b",
            re.IGNORECASE,
        )

    # -----------------------
    # Public API
    # -----------------------

    def answer(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Main entry point.

        Returns a dict suitable for UI:
          - intent: structural | quote_lookup | count | topic
          - results: grouped evidence (topic/quote/structural)
          - count: (only when intent=count)
        """
        q = (query or "").strip()
        if not q:
            return {"intent": "empty", "query": query, "results": []}

        # 1) Structural
        structural = self.structural.parse(q)
        if structural:
            return self._answer_structural(structural)

        # 2) Count intent
        if self._re_count.search(q):
            return self._answer_count(q)

        # 3) Explicit "where is this from?"
        if self._re_where_from.search(q) or self._looks_like_quote(q):
            return self._answer_quote_lookup(q, top_k=top_k)

        # 4) Topic / Story
        return self._answer_topic(q, top_k=top_k)

    # -----------------------
    # Intent handlers
    # -----------------------

    def _answer_structural(self, structural: Dict[str, Any]) -> Dict[str, Any]:
        verse_key = structural["verse_key"]
        hits = self.engine.search(verse_key, top_k=1)
        if not hits:
            return {"intent": "structural", "query": verse_key, "results": [], "error": "verse_not_found"}

        verse = hits[0]
        return {
            "intent": "structural",
            "query": verse_key,
            "results": [self._format_verse(verse, score=structural.get("confidence", 1.0), match_type="structural")],
        }

    def _answer_quote_lookup(self, query: str, top_k: Optional[int]) -> Dict[str, Any]:
        k = top_k or self.config.top_k_quote
        # Quote lookup should prioritize semantic similarity over BM25 quirks.
        # If available, use semantic_search; else fall back to fused search.
        results = []
        if hasattr(self.engine, "semantic_search"):
            sem = self.engine.semantic_search(query, top_k=max(15, k))
            results = sem[:k]
        else:
            results = self.engine.search(query, top_k=k)

        grouped = self._group_ranges(results, gap=self.config.group_gap_ayah)
        return {
            "intent": "quote_lookup",
            "query": query,
            "results": grouped,
        }

    def _answer_topic(self, query: str, top_k: Optional[int]) -> Dict[str, Any]:
        is_story = bool(self._re_story.search(query))
        k = top_k or (self.config.top_k_story if is_story else self.config.top_k_default)

        hits = self.engine.search(query, top_k=max(k, 25 if is_story else k))
        grouped = self._group_ranges(hits, gap=self.config.group_gap_ayah)

        return {
            "intent": "story" if is_story else "topic",
            "query": query,
            "results": grouped,
        }

    def _answer_count(self, query: str) -> Dict[str, Any]:
        """
        Two modes:
          A) Exact word/phrase count (deterministic)
          B) Semantic "theme/instruction" count (deterministic for given model+thresholding)

        We determine mode from language + keywords.
        """
        target = self._extract_count_target(query)
        if not target:
            return {"intent": "count", "query": query, "count": None, "results": [], "error": "count_target_not_found"}

        # Heuristic: "asked/commanded/ordered" => semantic count over verses matching the instruction/theme.
        # This is not a brittle rule forest; it's a routing decision.
        if self._looks_like_instruction_count(query):
            return self._semantic_count(query, target)

        # Else: exact token/phrase count.
        return self._exact_count(target)

    # -----------------------
    # Count helpers
    # -----------------------

    def _looks_like_instruction_count(self, q: str) -> bool:
        return bool(re.search(
            r"\b(asked\s+to|commanded\s+to|ordered\s+to|instructed\s+to|enjoined)\b|"
            r"(حکم|فرض|پابند|قائم\s+کرو|ادا\s+کرو|کا\s+حکم)|"
            r"(أمر|مأمور|فرض|أقيموا|آتوا)",
            q,
            re.IGNORECASE,
        ))

    def _extract_count_target(self, q: str) -> str:
        # English-ish
        s = q
        s = re.sub(r"(?i)\bhow\s+many\s+times\b", " ", s)
        s = re.sub(r"(?i)\bhow\s+many\b", " ", s)
        s = re.sub(r"(?i)\bcount\b", " ", s)
        s = re.sub(r"(?i)\boccurrences?\b", " ", s)
        s = re.sub(r"(?i)\bfrequency\b", " ", s)
        s = re.sub(r"(?i)\b(in\s+quran|in\s+the\s+quran|in\s+al\s*qur'an|in\s+the\s+qur'an)\b", " ", s)
        # Urdu / Arabic common fillers
        s = re.sub(r"(کتنی\s*بار|کتنی\s*دفعہ|کِنّی\s*وار|قرآن\s*میں|قرآن\s*مجید\s*میں)", " ", s)
        s = re.sub(r"(كم\s+مرة|كم\s+مرّة|في\s+القرآن|بالقرآن)", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        # remove surrounding quotes
        s = s.strip(' "\'“”‘’')
        return s

    def _exact_count(self, target: str) -> Dict[str, Any]:
        """
        Deterministic count.
        - If the target is a single token: token count (fast via postings if index exists)
        - Else: phrase count via scanning normalized verse text (safe, still fast for 6236 verses)
        """
        target_norm = target.strip()
        if not target_norm:
            return {"intent": "count", "query": target, "count": None, "results": [], "error": "empty_target"}

        is_single_token = len(target_norm.split()) == 1

        if is_single_token and self.count_index is not None:
            c = self.count_index.count_token(target_norm)
            verse_keys = self.count_index.lookup_token_verses(target_norm, limit=self.config.max_list_verses_in_count)
            verses = self._fetch_verses(verse_keys)
            return {
                "intent": "count",
                "query": target,
                "mode": "exact_token",
                "target": target_norm,
                "count": c,
                "results": self._group_ranges(verses, gap=self.config.group_gap_ayah),
            }

        # Fallback: phrase scan (works even without index)
        c, verse_keys = self._scan_phrase(target_norm)
        verses = self._fetch_verses(verse_keys[: self.config.max_list_verses_in_count])
        return {
            "intent": "count",
            "query": target,
            "mode": "exact_phrase",
            "target": target_norm,
            "count": c,
            "results": self._group_ranges(verses, gap=self.config.group_gap_ayah),
        }

    def _semantic_count(self, full_query: str, target: str) -> Dict[str, Any]:
        """
        Semantic count of verses for instruction/theme queries.

        Method (no brittle hardcoding):
        - Retrieve candidates (hybrid)
        - Keep verses above an adaptive threshold based on score distribution

        Note:
        - "count" here means "how many verses match this instruction/theme",
          which is the only defensible offline interpretation without a full
          Arabic imperative parser + fiqh ontology.
        """
        candidates_k = self.config.semantic_count_candidates

        # Prefer semantic_search for clean score distributions; otherwise use fused.
        if hasattr(self.engine, "semantic_search"):
            cand = self.engine.semantic_search(target, top_k=candidates_k)
        else:
            cand = self.engine.search(target, top_k=candidates_k)

        if not cand:
            return {"intent": "count", "query": full_query, "mode": "semantic_theme", "target": target, "count": 0, "results": []}

        # Use relevance_score if present, else fall back to e5_raw_similarity, else 0.
        scores = [float(v.get("relevance_score", v.get("e5_raw_similarity", 0.0))) for v in cand]
        keep_idx = self._adaptive_keep(scores, min_keep=self.config.semantic_count_min_keep)

        kept = [cand[i] for i in keep_idx]
        verse_keys = [v["verse_key"] for v in kept]

        return {
            "intent": "count",
            "query": full_query,
            "mode": "semantic_theme",
            "target": target,
            "count": len(verse_keys),
            "results": self._group_ranges(kept, gap=self.config.group_gap_ayah),
            "debug": {
                "candidates": len(cand),
                "kept": len(kept),
                "thresholding": "adaptive",
            },
        }

    def _adaptive_keep(self, scores: List[float], min_keep: int = 10) -> List[int]:
        """
        Adaptive thresholding:
        - Keep all results >= (mean + 0.35*std), but at least top `min_keep`.
        - Also cap at 200 to avoid overload.
        """
        import math

        n = len(scores)
        if n == 0:
            return []

        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        thresh = mean + 0.35 * std

        idx = [i for i, s in enumerate(scores) if s >= thresh]
        if len(idx) < min_keep:
            idx = list(range(min_keep))
        idx = idx[: min(200, n)]
        return idx

    def _scan_phrase(self, phrase: str) -> Tuple[int, List[str]]:
        """
        Phrase scan over Arabic + built-in English/Urdu + transliteration.
        Uses normalized comparison to reduce false negatives.
        """
        phrase_norm = self._normalize_phrase(phrase)
        if not phrase_norm:
            return 0, []

        count = 0
        verse_keys: List[str] = []

        for v in getattr(self.engine, "verses", []):
            txt = " ".join([
                v.get("arabic", ""),
                v.get("transliteration", ""),
                v.get("translation_en_builtin", ""),
                v.get("translation_ur_builtin", ""),
            ])
            norm = self._normalize_phrase(txt)
            if phrase_norm in norm:
                count += norm.count(phrase_norm)
                verse_keys.append(v["verse_key"])

        return count, verse_keys

    def _normalize_phrase(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        # normalize Arabic script and lowercase latin
        t = normalize_for_search(t)
        t = t.lower()
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # -----------------------
    # Quote detection
    # -----------------------

    def _looks_like_quote(self, q: str) -> bool:
        if len(q) < self.config.quote_min_len:
            return False
        # If user explicitly asks "where is this from", that's already handled.
        # Here we detect if they pasted text.
        arabic_chars = sum(1 for ch in q if self._is_arabic_char(ch))
        ratio = arabic_chars / max(1, len(q))
        if ratio >= self.config.quote_min_arabic_ratio:
            return True
        # Very long latin text without question words also often indicates pasted translation.
        if len(q) > 80 and not re.search(r"\b(what|where|how|why|which)\b", q, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _is_arabic_char(ch: str) -> bool:
        o = ord(ch)
        return (
            0x0600 <= o <= 0x06FF
            or 0x0750 <= o <= 0x077F
            or 0x08A0 <= o <= 0x08FF
            or 0xFB50 <= o <= 0xFDFF
            or 0xFE70 <= o <= 0xFEFF
        )

    # -----------------------
    # Grouping / formatting
    # -----------------------

    def _group_ranges(self, verses: List[Dict[str, Any]], gap: int = 2) -> List[Dict[str, Any]]:
        """
        Group verses into contiguous ayah ranges per surah for better UX.

        Input list is assumed ranked (highest score first).
        We preserve salience by scoring ranges using max score in the range.
        """
        if not verses:
            return []

        # We'll keep best score per verse from current ordering.
        scored: Dict[str, float] = {}
        for v in verses:
            vk = v.get("verse_key")
            if not vk:
                continue
            s = float(v.get("relevance_score", v.get("e5_raw_similarity", 0.0)) or 0.0)
            if vk not in scored or s > scored[vk]:
                scored[vk] = s

        # Group by surah
        by_surah: Dict[int, List[Dict[str, Any]]] = {}
        for v in verses:
            surah = int(v.get("surah", 0) or 0)
            if surah <= 0:
                continue
            by_surah.setdefault(surah, []).append(v)

        groups: List[Dict[str, Any]] = []
        for surah, vs in by_surah.items():
            # sort by ayah for grouping
            vs_sorted = sorted(vs, key=lambda x: int(x.get("ayah", 0) or 0))
            cur: List[Dict[str, Any]] = []
            for v in vs_sorted:
                ayah = int(v.get("ayah", 0) or 0)
                if not cur:
                    cur = [v]
                    continue
                prev_ayah = int(cur[-1].get("ayah", 0) or 0)
                if ayah <= prev_ayah + gap:
                    cur.append(v)
                else:
                    groups.append(self._finalize_range(cur, scored))
                    cur = [v]
            if cur:
                groups.append(self._finalize_range(cur, scored))

        # Rank groups by max verse score; tie-break by smaller ranges then earlier occurrence.
        groups.sort(key=lambda g: (g["score"], -g["end_ayah"] + g["start_ayah"]), reverse=True)
        return groups

    def _finalize_range(self, vs: List[Dict[str, Any]], score_map: Dict[str, float]) -> Dict[str, Any]:
        surah = int(vs[0]["surah"])
        ayahs = [int(v["ayah"]) for v in vs]
        start_ayah, end_ayah = min(ayahs), max(ayahs)

        verse_keys = [v["verse_key"] for v in vs]
        score = max(score_map.get(vk, 0.0) for vk in verse_keys)

        # Provide formatted verses within range (sorted by ayah)
        formatted = [self._format_verse(v, score=score_map.get(v["verse_key"], 0.0), match_type=v.get("match_type")) for v in sorted(vs, key=lambda x: int(x["ayah"]))]

        return {
            "surah": surah,
            "surah_name_english": vs[0].get("surah_name_english"),
            "surah_name_arabic": vs[0].get("surah_name_arabic"),
            "start_ayah": start_ayah,
            "end_ayah": end_ayah,
            "verse_keys": verse_keys,
            "score": float(score),
            "verses": formatted,
        }

    def _fetch_verses(self, verse_keys: List[str]) -> List[Dict[str, Any]]:
        lookup = getattr(self.engine, "verse_lookup", {})
        out = []
        for vk in verse_keys:
            v = lookup.get(vk)
            if v:
                out.append(v)
        return out

    def _format_verse(self, verse: Dict[str, Any], score: float, match_type: Optional[str]) -> Dict[str, Any]:
        # Minimal stable schema for mobile UI (can be extended safely).
        return {
            "verse_key": verse.get("verse_key"),
            "surah": verse.get("surah"),
            "ayah": verse.get("ayah"),
            "surah_name_english": verse.get("surah_name_english"),
            "surah_name_arabic": verse.get("surah_name_arabic"),
            "arabic": verse.get("arabic"),
            "transliteration": verse.get("transliteration"),
            "translation_en": verse.get("translation_en_builtin") or verse.get("translation_english") or "",
            "translation_ur": verse.get("translation_ur_builtin") or verse.get("translation_urdu") or "",
            "juz": verse.get("juz"),
            "score": float(score),
            "match_type": match_type or verse.get("match_type") or "unknown",
        }


def main():
    # Defaults to local repo artifacts; can be overridden via CLI.
    import argparse

    p = argparse.ArgumentParser(description="Ask Qur'an offline orchestrator (uses local repo artifacts by default).")
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to quran_complete.json (defaults to repo artifact).")
    p.add_argument("--emb", default=str(DEFAULT_EMB_PATH), help="Path to verse_embeddings_e5.npy (defaults to repo artifact).")
    p.add_argument("--keys", default=str(DEFAULT_KEYS_PATH), help="Path to verse_keys_e5.json (defaults to repo artifact).")
    p.add_argument("--count_index", default=str(DEFAULT_COUNT_INDEX_PATH), help="Optional path to count_index.pkl.gz (defaults to repo artifact if present).")
    args = p.parse_args()

    engine = E5HybridSearchEngine(
        data_path=args.data,
        embeddings_path=args.emb,
        verse_keys_path=args.keys,
    )

    count_idx = None
    if args.count_index:
        from count_index import QuranCountIndex
        try:
            count_idx = QuranCountIndex.load(args.count_index)
        except FileNotFoundError:
            count_idx = None

    ask = AskQuranEngine(engine=engine, count_index=count_idx)

    print("=" * 70)
    print("Ask Qur'an (offline) - interactive")
    print("Type ':q' to quit")
    print("=" * 70)

    while True:
        q = input("\n> ").strip()
        if not q or q == ":q":
            break
        ans = ask.answer(q)
        payload = _to_jsonable(ans)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
