
############## chatGPT's evalution framework for search engine - V2 - console logs persisted from original claude's script
"""
Evaluation Framework for Search Engine
Measures precision, recall, and accuracy (original console output preserved)

Upgrades (internals only):
- Supports expected as: "2:255" OR ["2:153","3:200"] OR {"2:153":3,"3:200":2}
- Adds rank-aware metrics: MAP@K, nDCG@K, latency stats
- Keeps original console output lines and summary text
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from search_engine import QuranSearchEngine


VerseKey = str
Grades = Dict[VerseKey, float]


def _as_grades(expected: Any) -> Grades:
    if expected is None:
        return {}
    if isinstance(expected, str):
        k = expected.strip()
        return {k: 1.0} if k else {}
    if isinstance(expected, list):
        out: Grades = {}
        for x in expected:
            if isinstance(x, str) and x.strip():
                out[x.strip()] = 1.0
        return out
    if isinstance(expected, dict):
        out: Grades = {}
        for k, v in expected.items():
            if not isinstance(k, str) or not k.strip():
                continue
            try:
                out[k.strip()] = float(v)
            except Exception:
                out[k.strip()] = 1.0
        return out
    return {}


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _precision_at_1(retrieved: List[VerseKey], relevant: set[VerseKey]) -> float:
    return 1.0 if retrieved and retrieved[0] in relevant else 0.0


def _recall_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    return 1.0 if any(x in relevant for x in retrieved[:k]) else 0.0


def _rr_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
    for i, x in enumerate(retrieved[:k], start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def _ap_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, x in enumerate(retrieved[:k], start=1):
        if x in relevant:
            hits += 1
            sum_prec += hits / i
    denom = min(len(relevant), k)
    return sum_prec / denom if denom > 0 else 0.0


def _dcg_at_k(retrieved: List[VerseKey], grades: Grades, k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = float(grades.get(doc_id, 0.0))
        if rel <= 0:
            continue
        gain = (2.0 ** rel) - 1.0
        dcg += gain / math.log2(i + 1.0)
    return dcg


def _ndcg_at_k(retrieved: List[VerseKey], grades: Grades, k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = _dcg_at_k(retrieved, grades, k)
    ideal = [vk for vk, _ in sorted(grades.items(), key=lambda kv: kv[1], reverse=True)]
    idcg = _dcg_at_k(ideal, grades, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def _p95(values: List[float]) -> float:
    """
    Robust p95:
    - For small N, returns max.
    - For larger N, uses statistics.quantiles (inclusive style is okay for monitoring).
    """
    if not values:
        return 0.0
    if len(values) < 20:
        return max(values)
    # statistics.quantiles returns cut points; n=20 gives 5% increments; index 18 ~= 95th
    return statistics.quantiles(values, n=20)[18]


class SearchEvaluator:
    """Evaluate search engine performance"""

    def __init__(self, engine: QuranSearchEngine, test_queries_path: str):
        self.engine = engine

        with open(test_queries_path, "r", encoding="utf-8") as f:
            self.test_queries = json.load(f)

        print(f"Loaded {len(self.test_queries)} test queries")

    def evaluate_query(self, query_obj: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
        """
        Evaluate a single query (preserves original output fields)
        """
        query = query_obj["query"]
        expected = query_obj["expected"]
        query_type = query_obj.get("type", "unknown")

        grades = _as_grades(expected)
        relevant = set(grades.keys())

        # Perform search + latency
        t0 = time.perf_counter()
        results = self.engine.search(query, top_k=k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_keys = _dedupe_preserve_order([
            str(r.get("verse_key")) for r in results
            if isinstance(r, dict) and r.get("verse_key")
        ])

        found_at_position = None
        for i, vk in enumerate(retrieved_keys):
            if vk in relevant:
                found_at_position = i + 1
                break

        metrics = {
            # Original fields
            "query": query,
            "expected": expected,
            "type": query_type,
            "found": found_at_position is not None,
            "position": found_at_position,
            "precision_at_1": _precision_at_1(retrieved_keys, relevant),
            "recall_at_k": _recall_at_k(retrieved_keys, relevant, k),
            "mrr": (1.0 / found_at_position) if found_at_position else 0.0,

            # Extra (saved to JSON)
            "map_at_k": _ap_at_k(retrieved_keys, relevant, k),
            "ndcg_at_k": _ndcg_at_k(retrieved_keys, grades, k),
            "latency_ms": latency_ms,
            "retrieved_keys": retrieved_keys[:k],
            "relevant_keys": list(relevant),
        }

        return metrics

    def evaluate_all(self, k: int = 5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate all test queries (console output preserved)
        """
        print(f"\nEvaluating {len(self.test_queries)} queries...")
        print("=" * 60)

        all_metrics: List[Dict[str, Any]] = []

        for query_obj in self.test_queries:
            metrics = self.evaluate_query(query_obj, k)
            all_metrics.append(metrics)

            # ORIGINAL progress line format
            status = "✓" if metrics["found"] else "✗"
            qtxt = str(query_obj.get("query", ""))[:40]
            print(f"{status} {qtxt:<40} | Found: {metrics['found']} @ pos {metrics['position']}")

        total = len(all_metrics)
        found_count = sum(1 for m in all_metrics if m["found"])

        aggregate = {
            # Original aggregate fields
            "total_queries": total,
            "found_count": found_count,
            "accuracy": found_count / total if total > 0 else 0.0,
            "precision_at_1": sum(m["precision_at_1"] for m in all_metrics) / total if total else 0.0,
            "recall_at_k": sum(m["recall_at_k"] for m in all_metrics) / total if total else 0.0,
            "mrr": sum(m["mrr"] for m in all_metrics) / total if total else 0.0,

            # Extra aggregates
            "map_at_k": sum(m["map_at_k"] for m in all_metrics) / total if total else 0.0,
            "ndcg_at_k": sum(m["ndcg_at_k"] for m in all_metrics) / total if total else 0.0,
        }

        # Latency summary (requested to print)
        lat = [m["latency_ms"] for m in all_metrics]
        if lat:
            aggregate["latency_ms"] = {
                "mean": statistics.mean(lat),
                "p50": statistics.median(lat),
                "p95": _p95(lat),
                "min": min(lat),
                "max": max(lat),
            }

        # Break down by query type (original style: only accuracy)
        query_types = set(m["type"] for m in all_metrics)
        aggregate["by_type"] = {}

        for qtype in query_types:
            type_metrics = [m for m in all_metrics if m["type"] == qtype]
            if type_metrics:
                type_total = len(type_metrics)
                type_found = sum(1 for m in type_metrics if m["found"])
                aggregate["by_type"][qtype] = {
                    "total": type_total,
                    "found": type_found,
                    "accuracy": type_found / type_total if type_total > 0 else 0.0,
                }

        return aggregate, all_metrics

    def print_report(self, aggregate: Dict[str, Any], detailed: bool = False):
        """Print evaluation report (console output preserved, plus latency block)"""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print(f"\nOverall Metrics:")
        print(f"  Total Queries: {aggregate['total_queries']}")
        print(f"  Found: {aggregate['found_count']} / {aggregate['total_queries']}")
        print(f"  Accuracy: {aggregate['accuracy']*100:.1f}%")
        print(f"  Precision@1: {aggregate['precision_at_1']*100:.1f}%")
        print(f"  Recall@K: {aggregate['recall_at_k']*100:.1f}%")
        print(f"  MRR: {aggregate['mrr']:.3f}")

        # Added latency block (as requested)
        lat = aggregate.get("latency_ms")
        if lat:
            print(f"\nLatency (ms):")
            print(f"  Mean: {lat['mean']:.2f}")
            print(f"  P50 : {lat['p50']:.2f}")
            print(f"  P95 : {lat['p95']:.2f}")
            print(f"  Min : {lat['min']:.2f}")
            print(f"  Max : {lat['max']:.2f}")

        if aggregate.get("by_type"):
            print(f"\nAccuracy by Query Type:")
            for qtype, metrics in aggregate["by_type"].items():
                acc = metrics["accuracy"] * 100
                print(f"  {qtype:20s}: {acc:5.1f}% ({metrics['found']}/{metrics['total']})")

        print("\n" + "=" * 60)

        # Assessment (original thresholds/messages)
        accuracy = aggregate["accuracy"] * 100
        if accuracy >= 85:
            print("✓ EXCELLENT: Accuracy meets target (≥85%)")
        elif accuracy >= 70:
            print("⚠ GOOD: Accuracy acceptable for MVP (≥70%)")
        else:
            print("✗ NEEDS IMPROVEMENT: Accuracy below target (<70%)")

        print("=" * 60)


def run_evaluation():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    test_queries_path = base_dir / "output" / "processed" / "test_queries.json"

    print("Initializing search engine...")
    engine = QuranSearchEngine(str(data_path))

    evaluator = SearchEvaluator(engine, str(test_queries_path))

    aggregate, detailed = evaluator.evaluate_all(k=5)
    evaluator.print_report(aggregate)

    output_path = base_dir / "output" / "processed" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "detailed": detailed}, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {output_path}")
    return aggregate


if __name__ == "__main__":
    run_evaluation()



############## chatGPT's evalution framework for search engine - V2
# """
# Evaluation Framework for Quran Search Engine (BM25 / Hybrid)
# Comprehensive offline evaluation for ranked retrieval + latency + error analysis.

# test_queries.json supports:
#   - {"query": "...", "expected": "2:153", "type": "..."}
#   - {"query": "...", "expected": ["2:153","3:200"], "type": "..."}
#   - {"query": "...", "expected": {"2:153":3,"3:200":2}, "type": "..."}

# Outputs:
#   - aggregate metrics (by K and by type)
#   - detailed per-query metrics
#   - error analysis for misses

# Run:
#   python evaluate_search.py
# """

# from __future__ import annotations

# import json
# import math
# import statistics
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple, Union

# from search_engine import QuranSearchEngine

# VerseKey = str
# Grades = Dict[VerseKey, float]


# def as_grades(expected: Any) -> Grades:
#     if expected is None:
#         return {}
#     if isinstance(expected, str):
#         k = expected.strip()
#         return {k: 1.0} if k else {}
#     if isinstance(expected, list):
#         out: Grades = {}
#         for x in expected:
#             if isinstance(x, str) and x.strip():
#                 out[x.strip()] = 1.0
#         return out
#     if isinstance(expected, dict):
#         out: Grades = {}
#         for k, v in expected.items():
#             if not isinstance(k, str) or not k.strip():
#                 continue
#             try:
#                 out[k.strip()] = float(v)
#             except Exception:
#                 out[k.strip()] = 1.0
#         return out
#     return {}


# def dedupe_preserve_order(items: List[str]) -> List[str]:
#     seen = set()
#     out = []
#     for x in items:
#         if x in seen:
#             continue
#         seen.add(x)
#         out.append(x)
#     return out


# def precision_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if k <= 0:
#         return 0.0
#     r = retrieved[:k]
#     if not r:
#         return 0.0
#     return sum(1 for x in r if x in relevant) / len(r)


# def recall_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if k <= 0 or not relevant:
#         return 0.0
#     r = retrieved[:k]
#     return sum(1 for x in r if x in relevant) / len(relevant)


# def f1(p: float, r: float) -> float:
#     if p <= 0 or r <= 0:
#         return 0.0
#     return 2 * p * r / (p + r)


# def success_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if k <= 0:
#         return 0.0
#     return 1.0 if any(x in relevant for x in retrieved[:k]) else 0.0


# def rr_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     for i, x in enumerate(retrieved[:k], start=1):
#         if x in relevant:
#             return 1.0 / i
#     return 0.0


# def ap_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if not relevant or k <= 0:
#         return 0.0
#     hits = 0
#     sum_prec = 0.0
#     for i, x in enumerate(retrieved[:k], start=1):
#         if x in relevant:
#             hits += 1
#             sum_prec += hits / i
#     denom = min(len(relevant), k)
#     return sum_prec / denom if denom > 0 else 0.0


# def dcg_at_k(retrieved: List[VerseKey], grades: Grades, k: int) -> float:
#     dcg = 0.0
#     for i, doc in enumerate(retrieved[:k], start=1):
#         rel = float(grades.get(doc, 0.0))
#         if rel <= 0:
#             continue
#         gain = (2.0 ** rel) - 1.0
#         dcg += gain / math.log2(i + 1.0)
#     return dcg


# def ndcg_at_k(retrieved: List[VerseKey], grades: Grades, k: int) -> float:
#     if k <= 0:
#         return 0.0
#     dcg = dcg_at_k(retrieved, grades, k)
#     ideal = [vk for vk, _ in sorted(grades.items(), key=lambda kv: kv[1], reverse=True)]
#     idcg = dcg_at_k(ideal, grades, k)
#     return (dcg / idcg) if idcg > 0 else 0.0


# class SearchEvaluator:
#     def __init__(self, engine: QuranSearchEngine, test_queries_path: str):
#         self.engine = engine
#         with open(test_queries_path, "r", encoding="utf-8") as f:
#             self.test_queries = json.load(f)
#         if not isinstance(self.test_queries, list):
#             raise ValueError("test_queries.json must be an array.")
#         print(f"Loaded {len(self.test_queries)} test queries")

#     def _validate(self, obj: Any) -> Optional[str]:
#         if not isinstance(obj, dict):
#             return "Entry is not an object."
#         q = obj.get("query")
#         if not isinstance(q, str) or not q.strip():
#             return "Missing/empty 'query'."
#         if "expected" not in obj:
#             return "Missing 'expected'."
#         grades = as_grades(obj.get("expected"))
#         if not grades:
#             return "Expected is empty/unparseable."
#         return None

#     def evaluate_query(self, query_obj: Dict[str, Any], k_values: List[int]) -> Dict[str, Any]:
#         query = query_obj["query"].strip()
#         qtype = query_obj.get("type", "unknown")

#         grades = as_grades(query_obj.get("expected"))
#         relevant = set(grades.keys())

#         max_k = max(k_values) if k_values else 5

#         t0 = time.perf_counter()
#         results = self.engine.search(query, top_k=max_k)
#         latency_ms = (time.perf_counter() - t0) * 1000.0

#         retrieved = dedupe_preserve_order([
#             str(r.get("verse_key")) for r in results
#             if isinstance(r, dict) and r.get("verse_key")
#         ])

#         positions = {vk: (retrieved.index(vk) + 1) for vk in relevant if vk in retrieved}
#         first_pos = min(positions.values()) if positions else None

#         per_k: Dict[str, Any] = {}
#         for k in sorted(set(k_values)):
#             p = precision_at_k(retrieved, relevant, k)
#             r = recall_at_k(retrieved, relevant, k)
#             per_k[str(k)] = {
#                 "success@k": success_at_k(retrieved, relevant, k),
#                 "precision@k": p,
#                 "recall@k": r,
#                 "f1@k": f1(p, r),
#                 "mrr@k": rr_at_k(retrieved, relevant, k),
#                 "ap@k": ap_at_k(retrieved, relevant, k),
#                 "ndcg@k": ndcg_at_k(retrieved, grades, k),
#             }

#         return {
#             "query": query,
#             "type": qtype,
#             "expected": query_obj.get("expected"),
#             "relevant_count": len(relevant),
#             "retrieved_keys": retrieved[:max_k],
#             "found_any": first_pos is not None,
#             "first_relevant_position": first_pos,
#             "all_relevant_positions": positions,
#             "latency_ms": latency_ms,
#             "metrics_by_k": per_k,
#         }

#     def evaluate_all(self, k_values: List[int] = [1, 3, 5, 10]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
#         print("\nEvaluating queries...")
#         print("=" * 90)

#         detailed: List[Dict[str, Any]] = []
#         skipped: List[Dict[str, Any]] = []

#         for idx, q in enumerate(self.test_queries):
#             err = self._validate(q)
#             if err:
#                 skipped.append({"index": idx, "error": err, "raw": q})
#                 print(f"SKIP {idx:03d}: {err}")
#                 continue

#             m = self.evaluate_query(q, k_values)
#             detailed.append(m)

#             k_main = str(max(k_values))
#             hit = m["metrics_by_k"][k_main]["success@k"]
#             pos = m["first_relevant_position"]
#             print(f"{'HIT' if hit else 'MISS':4s} | {m['type'][:14]:14s} | {m['query'][:55]:55s} | pos={pos} | {m['latency_ms']:.1f}ms")

#             # Error analysis hint for misses
#             if not hit:
#                 print(f"      top={m['retrieved_keys'][:5]}  expected={list(as_grades(q.get('expected')).keys())[:5]}")

#         aggregate = self._aggregate(detailed, k_values)
#         aggregate["skipped"] = skipped
#         return aggregate, detailed

#     def _aggregate(self, detailed: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, Any]:
#         total = len(detailed)
#         if total == 0:
#             return {"total_queries": 0, "by_k": {}, "by_type": {}, "latency_ms": {}}

#         by_k: Dict[str, Any] = {}
#         for k in sorted(set(k_values)):
#             ks = str(k)
#             by_k[ks] = {
#                 "success@k": statistics.mean(d["metrics_by_k"][ks]["success@k"] for d in detailed),
#                 "precision@k": statistics.mean(d["metrics_by_k"][ks]["precision@k"] for d in detailed),
#                 "recall@k": statistics.mean(d["metrics_by_k"][ks]["recall@k"] for d in detailed),
#                 "f1@k": statistics.mean(d["metrics_by_k"][ks]["f1@k"] for d in detailed),
#                 "mrr@k": statistics.mean(d["metrics_by_k"][ks]["mrr@k"] for d in detailed),
#                 "map@k": statistics.mean(d["metrics_by_k"][ks]["ap@k"] for d in detailed),
#                 "ndcg@k": statistics.mean(d["metrics_by_k"][ks]["ndcg@k"] for d in detailed),
#             }

#         lat = [d["latency_ms"] for d in detailed]
#         lat_summary = {
#             "mean": statistics.mean(lat),
#             "p50": statistics.median(lat),
#             "p95": statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat),
#             "min": min(lat),
#             "max": max(lat),
#         }

#         types = sorted(set(d.get("type", "unknown") for d in detailed))
#         by_type: Dict[str, Any] = {}
#         for t in types:
#             subset = [d for d in detailed if d.get("type", "unknown") == t]
#             if not subset:
#                 continue
#             by_type[t] = {
#                 "count": len(subset),
#                 "latency_mean": statistics.mean(d["latency_ms"] for d in subset),
#                 "by_k": {
#                     str(k): {
#                         "success@k": statistics.mean(d["metrics_by_k"][str(k)]["success@k"] for d in subset),
#                         "mrr@k": statistics.mean(d["metrics_by_k"][str(k)]["mrr@k"] for d in subset),
#                         "map@k": statistics.mean(d["metrics_by_k"][str(k)]["ap@k"] for d in subset),
#                         "ndcg@k": statistics.mean(d["metrics_by_k"][str(k)]["ndcg@k"] for d in subset),
#                     }
#                     for k in sorted(set(k_values))
#                 },
#             }

#         return {
#             "total_queries": total,
#             "by_k": by_k,
#             "by_type": by_type,
#             "latency_ms": lat_summary,
#         }

#     def print_report(self, aggregate: Dict[str, Any], k_values: List[int]) -> None:
#         print("\n" + "=" * 90)
#         print("EVALUATION REPORT")
#         print("=" * 90)

#         print(f"Total evaluated: {aggregate.get('total_queries', 0)}")

#         lat = aggregate.get("latency_ms", {})
#         if lat:
#             print("\nLatency (ms): "
#                   f"mean={lat['mean']:.2f}  p50={lat['p50']:.2f}  p95={lat['p95']:.2f}  min={lat['min']:.2f}  max={lat['max']:.2f}")

#         print("\nMetrics by K:")
#         by_k = aggregate.get("by_k", {})
#         for k in sorted(set(k_values)):
#             ks = str(k)
#             m = by_k.get(ks)
#             if not m:
#                 continue
#             print(f"  K={k:>2d} | "
#                   f"Success@K={m['success@k']*100:5.1f}%  "
#                   f"P@K={m['precision@k']*100:5.1f}%  "
#                   f"R@K={m['recall@k']*100:5.1f}%  "
#                   f"MRR@K={m['mrr@k']:.3f}  "
#                   f"MAP@K={m['map@k']:.3f}  "
#                   f"nDCG@K={m['ndcg@k']:.3f}")

#         if aggregate.get("by_type"):
#             print("\nBy type (count, latency_mean):")
#             for t, tm in aggregate["by_type"].items():
#                 print(f"  {t:16s}  n={tm['count']:>2d}  latency={tm['latency_mean']:.2f}ms")

#         skipped = aggregate.get("skipped") or []
#         if skipped:
#             print(f"\nSkipped malformed queries: {len(skipped)}")
#             for s in skipped[:5]:
#                 print(f"  - idx={s['index']}: {s['error']}")

#         print("=" * 90)


# def run_evaluation():
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
#     test_queries_path = base_dir / "output" / "processed" / "test_queries.json"
#     out_path = base_dir / "output" / "processed" / "evaluation_results.json"

#     print("Initializing search engine...")
#     engine = QuranSearchEngine(str(data_path))

#     evaluator = SearchEvaluator(engine, str(test_queries_path))

#     k_values = [1, 3, 5, 10]
#     aggregate, detailed = evaluator.evaluate_all(k_values=k_values)
#     evaluator.print_report(aggregate, k_values=k_values)

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump({"aggregate": aggregate, "detailed": detailed}, f, ensure_ascii=False, indent=2)

#     print(f"\nSaved detailed results to: {out_path}")
#     return aggregate


# if __name__ == "__main__":
#     run_evaluation()




############## chatGPT's evalution framework for search engine
# """
# Evaluation Framework for Quran Search Engine
# Comprehensive offline evaluation for ranked retrieval.

# Metrics (standard IR):
# - Success@K (Hit@K)
# - Precision@K, Recall@K, F1@K
# - MRR@K (reciprocal rank of first relevant within top K)
# - MAP@K (mean average precision within top K)
# - nDCG@K (supports graded relevance)

# Also tracks latency:
# - mean, p50, p95 per query batch

# References for metric definitions:
# - Precision/Recall/MAP/MRR/NDCG are standard IR evaluation measures. :contentReference[oaicite:2]{index=2}
# """

# from __future__ import annotations

# import json
# import math
# import statistics
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple, Union

# from search_engine import QuranSearchEngine


# VerseKey = str
# Grades = Dict[VerseKey, float]


# def _as_grades(expected: Any) -> Grades:
#     """
#     Normalize expected relevance from test_queries.json into a dict:
#       - "2:255"                -> {"2:255": 1.0}
#       - ["2:153","3:200"]      -> {"2:153": 1.0, "3:200": 1.0}
#       - {"2:43":3,"29:45":2}   -> {"2:43": 3.0, "29:45": 2.0}
#     """
#     if expected is None:
#         return {}
#     if isinstance(expected, str):
#         return {expected.strip(): 1.0} if expected.strip() else {}
#     if isinstance(expected, list):
#         out: Grades = {}
#         for x in expected:
#             if isinstance(x, str) and x.strip():
#                 out[x.strip()] = 1.0
#         return out
#     if isinstance(expected, dict):
#         out = {}
#         for k, v in expected.items():
#             if not isinstance(k, str) or not k.strip():
#                 continue
#             try:
#                 out[k.strip()] = float(v)
#             except Exception:
#                 out[k.strip()] = 1.0
#         return out
#     return {}


# def _dedupe_preserve_order(items: List[str]) -> List[str]:
#     seen = set()
#     out = []
#     for x in items:
#         if x in seen:
#             continue
#         seen.add(x)
#         out.append(x)
#     return out


# def _precision_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if k <= 0:
#         return 0.0
#     r = retrieved[:k]
#     if not r:
#         return 0.0
#     hit = sum(1 for x in r if x in relevant)
#     return hit / len(r)


# def _recall_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if k <= 0 or not relevant:
#         return 0.0
#     r = retrieved[:k]
#     hit = sum(1 for x in r if x in relevant)
#     return hit / len(relevant)


# def _f1(p: float, r: float) -> float:
#     if p <= 0 or r <= 0:
#         return 0.0
#     return 2 * p * r / (p + r)


# def _success_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     if k <= 0:
#         return 0.0
#     return 1.0 if any(x in relevant for x in retrieved[:k]) else 0.0


# def _reciprocal_rank_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     """
#     RR@K: reciprocal rank of the first relevant item within top K (else 0).
#     MRR is mean over queries. :contentReference[oaicite:3]{index=3}
#     """
#     for i, x in enumerate(retrieved[:k], start=1):
#         if x in relevant:
#             return 1.0 / i
#     return 0.0


# def _average_precision_at_k(retrieved: List[VerseKey], relevant: set[VerseKey], k: int) -> float:
#     """
#     AP@K: average precision computed over top K only (useful when you only retrieve top K).
#     MAP@K is mean of AP@K across queries. :contentReference[oaicite:4]{index=4}
#     """
#     if not relevant or k <= 0:
#         return 0.0
#     hits = 0
#     sum_prec = 0.0
#     for i, x in enumerate(retrieved[:k], start=1):
#         if x in relevant:
#             hits += 1
#             sum_prec += hits / i
#     # Normalize by the number of relevant items (bounded by K)
#     denom = min(len(relevant), k)
#     return sum_prec / denom if denom > 0 else 0.0


# def _dcg_at_k(retrieved: List[VerseKey], grades: Grades, k: int) -> float:
#     """
#     DCG@K with graded relevance.
#     Uses gain = (2^rel - 1) and log2 discount. :contentReference[oaicite:5]{index=5}
#     """
#     dcg = 0.0
#     for i, doc_id in enumerate(retrieved[:k], start=1):
#         rel = float(grades.get(doc_id, 0.0))
#         if rel <= 0:
#             continue
#         gain = (2.0 ** rel) - 1.0
#         dcg += gain / math.log2(i + 1.0)
#     return dcg


# def _ndcg_at_k(retrieved: List[VerseKey], grades: Grades, k: int) -> float:
#     """
#     nDCG@K = DCG@K / IDCG@K. :contentReference[oaicite:6]{index=6}
#     """
#     if k <= 0:
#         return 0.0
#     dcg = _dcg_at_k(retrieved, grades, k)
#     # Ideal ranking is by grade descending
#     ideal = sorted(grades.items(), key=lambda kv: kv[1], reverse=True)
#     ideal_list = [vk for vk, _ in ideal]
#     idcg = _dcg_at_k(ideal_list, grades, k)
#     if idcg <= 0:
#         return 0.0
#     return dcg / idcg


# class SearchEvaluator:
#     """Evaluate search engine performance (ranked retrieval)."""

#     def __init__(self, engine: QuranSearchEngine, test_queries_path: str):
#         self.engine = engine
#         with open(test_queries_path, "r", encoding="utf-8") as f:
#             self.test_queries = json.load(f)

#         if not isinstance(self.test_queries, list):
#             raise ValueError("test_queries.json must be a JSON array of query objects.")

#         print(f"Loaded {len(self.test_queries)} test queries")

#     def _validate_query_obj(self, q: Dict[str, Any]) -> Optional[str]:
#         if not isinstance(q, dict):
#             return "Query entry is not an object."
#         if "query" not in q or not isinstance(q["query"], str) or not q["query"].strip():
#             return "Missing/empty 'query' string."
#         if "expected" not in q:
#             return "Missing 'expected'. (Use a verse_key, list, or dict of graded relevance.)"
#         grades = _as_grades(q.get("expected"))
#         if not grades:
#             return "Expected relevance is empty/unparseable."
#         return None

#     def evaluate_query(self, query_obj: Dict[str, Any], k_values: List[int]) -> Dict[str, Any]:
#         query = query_obj["query"].strip()
#         qtype = query_obj.get("type", "unknown")

#         grades = _as_grades(query_obj.get("expected"))
#         relevant_set = set(grades.keys())

#         max_k = max(k_values) if k_values else 5

#         # Run search + latency timing
#         t0 = time.perf_counter()
#         results = self.engine.search(query, top_k=max_k)
#         latency_ms = (time.perf_counter() - t0) * 1000.0

#         retrieved_keys = _dedupe_preserve_order([
#             str(r.get("verse_key")) for r in results if isinstance(r, dict) and r.get("verse_key")
#         ])

#         # Positions of relevant docs
#         positions = {vk: (retrieved_keys.index(vk) + 1) for vk in relevant_set if vk in retrieved_keys}
#         first_pos = min(positions.values()) if positions else None

#         per_k: Dict[str, Any] = {}
#         for k in sorted(set(k_values)):
#             p = _precision_at_k(retrieved_keys, relevant_set, k)
#             r = _recall_at_k(retrieved_keys, relevant_set, k)
#             per_k[str(k)] = {
#                 "success@k": _success_at_k(retrieved_keys, relevant_set, k),
#                 "precision@k": p,
#                 "recall@k": r,
#                 "f1@k": _f1(p, r),
#                 "mrr@k": _reciprocal_rank_at_k(retrieved_keys, relevant_set, k),
#                 "ap@k": _average_precision_at_k(retrieved_keys, relevant_set, k),
#                 "ndcg@k": _ndcg_at_k(retrieved_keys, grades, k),
#             }

#         return {
#             "query": query,
#             "type": qtype,
#             "expected": query_obj.get("expected"),
#             "relevant_count": len(relevant_set),
#             "retrieved_count": len(retrieved_keys),
#             "found_any": first_pos is not None,
#             "first_relevant_position": first_pos,
#             "all_relevant_positions": positions,  # verse_key -> rank
#             "latency_ms": latency_ms,
#             "metrics_by_k": per_k,
#         }

#     def evaluate_all(self, k_values: List[int] = [1, 3, 5, 10]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
#         print(f"\nEvaluating {len(self.test_queries)} queries...")
#         print("=" * 80)

#         detailed: List[Dict[str, Any]] = []
#         errors: List[Dict[str, Any]] = []

#         for idx, q in enumerate(self.test_queries):
#             err = self._validate_query_obj(q)
#             if err:
#                 errors.append({"index": idx, "error": err, "raw": q})
#                 print(f"ERROR {idx:04d}: {err}")
#                 continue

#             m = self.evaluate_query(q, k_values)
#             detailed.append(m)

#             # Lightweight progress line
#             k_main = str(max(k_values))
#             hit = m["metrics_by_k"][k_main]["success@k"]
#             pos = m["first_relevant_position"]
#             print(f"{'HIT' if hit else 'MISS':4s} {q['query'][:50]:50s}  first_pos={pos}  latency={m['latency_ms']:.1f}ms")

#         aggregate = self._aggregate(detailed, k_values)
#         aggregate["input_errors"] = errors
#         return aggregate, detailed

#     def _aggregate(self, detailed: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, Any]:
#         total = len(detailed)
#         if total == 0:
#             return {"total_queries": 0, "by_k": {}, "by_type": {}, "latency_ms": {}}

#         by_k: Dict[str, Any] = {}
#         for k in sorted(set(k_values)):
#             ks = str(k)
#             by_k[ks] = {
#                 "success@k": statistics.mean(d["metrics_by_k"][ks]["success@k"] for d in detailed),
#                 "precision@k": statistics.mean(d["metrics_by_k"][ks]["precision@k"] for d in detailed),
#                 "recall@k": statistics.mean(d["metrics_by_k"][ks]["recall@k"] for d in detailed),
#                 "f1@k": statistics.mean(d["metrics_by_k"][ks]["f1@k"] for d in detailed),
#                 "mrr@k": statistics.mean(d["metrics_by_k"][ks]["mrr@k"] for d in detailed),
#                 "map@k": statistics.mean(d["metrics_by_k"][ks]["ap@k"] for d in detailed),
#                 "ndcg@k": statistics.mean(d["metrics_by_k"][ks]["ndcg@k"] for d in detailed),
#             }

#         latencies = [d["latency_ms"] for d in detailed]
#         lat_summary = {
#             "mean": statistics.mean(latencies),
#             "p50": statistics.median(latencies),
#             "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
#             "max": max(latencies),
#             "min": min(latencies),
#         }

#         # Breakdown by type
#         types = sorted(set(d.get("type", "unknown") for d in detailed))
#         by_type: Dict[str, Any] = {}
#         for t in types:
#             subset = [d for d in detailed if d.get("type", "unknown") == t]
#             if not subset:
#                 continue
#             by_type[t] = {
#                 "count": len(subset),
#                 "by_k": {
#                     str(k): {
#                         "success@k": statistics.mean(d["metrics_by_k"][str(k)]["success@k"] for d in subset),
#                         "mrr@k": statistics.mean(d["metrics_by_k"][str(k)]["mrr@k"] for d in subset),
#                         "map@k": statistics.mean(d["metrics_by_k"][str(k)]["ap@k"] for d in subset),
#                         "ndcg@k": statistics.mean(d["metrics_by_k"][str(k)]["ndcg@k"] for d in subset),
#                     }
#                     for k in sorted(set(k_values))
#                 },
#                 "latency_ms_mean": statistics.mean(d["latency_ms"] for d in subset),
#             }

#         return {
#             "total_queries": total,
#             "by_k": by_k,
#             "by_type": by_type,
#             "latency_ms": lat_summary,
#         }

#     def print_report(self, aggregate: Dict[str, Any], k_values: List[int]) -> None:
#         print("\n" + "=" * 80)
#         print("EVALUATION REPORT (Offline)")
#         print("=" * 80)

#         print(f"Total evaluated queries: {aggregate.get('total_queries', 0)}")

#         # Latency
#         lat = aggregate.get("latency_ms", {})
#         if lat:
#             print("\nLatency (ms):")
#             print(f"  Mean: {lat.get('mean', 0):.2f}")
#             print(f"  P50 : {lat.get('p50', 0):.2f}")
#             print(f"  P95 : {lat.get('p95', 0):.2f}")
#             print(f"  Min : {lat.get('min', 0):.2f}")
#             print(f"  Max : {lat.get('max', 0):.2f}")

#         # Metrics by K
#         print("\nMetrics by K:")
#         by_k = aggregate.get("by_k", {})
#         for k in sorted(set(k_values)):
#             ks = str(k)
#             m = by_k.get(ks, {})
#             if not m:
#                 continue
#             print(f"  K={k:>2d} | "
#                   f"Success@K={m['success@k']*100:5.1f}%  "
#                   f"P@K={m['precision@k']*100:5.1f}%  "
#                   f"R@K={m['recall@k']*100:5.1f}%  "
#                   f"MRR@K={m['mrr@k']:.3f}  "
#                   f"MAP@K={m['map@k']:.3f}  "
#                   f"nDCG@K={m['ndcg@k']:.3f}")

#         # By type
#         if aggregate.get("by_type"):
#             print("\nBreakdown by query type:")
#             for t, tm in aggregate["by_type"].items():
#                 print(f"  {t:20s} count={tm['count']:>4d}  latency_mean={tm['latency_ms_mean']:.2f}ms")

#         # Input errors
#         errs = aggregate.get("input_errors") or []
#         if errs:
#             print(f"\nInput errors (skipped): {len(errs)}")
#             print("  First few:")
#             for e in errs[:5]:
#                 print(f"   - idx={e['index']}: {e['error']}")

#         print("=" * 80)


# def run_evaluation():
#     """Run complete evaluation"""
#     # Paths
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
#     test_queries_path = base_dir / "output" / "processed" / "test_queries.json"
#     out_path = base_dir / "output" / "processed" / "evaluation_results.json"

#     print("Initializing search engine...")
#     engine = QuranSearchEngine(str(data_path))

#     evaluator = SearchEvaluator(engine, str(test_queries_path))

#     k_values = [1, 3, 5, 10]
#     aggregate, detailed = evaluator.evaluate_all(k_values=k_values)
#     evaluator.print_report(aggregate, k_values=k_values)

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump({"aggregate": aggregate, "detailed": detailed}, f, ensure_ascii=False, indent=2)

#     print(f"\nSaved: {out_path}")
#     return aggregate


# if __name__ == "__main__":
#     run_evaluation()


############## claude's evalution framework for search engine
# """
# Evaluation Framework for Search Engine
# Measures precision, recall, and accuracy
# """

# import json
# from pathlib import Path
# from typing import List, Dict
# from search_engine import QuranSearchEngine


# class SearchEvaluator:
#     """Evaluate search engine performance"""
    
#     def __init__(self, engine: QuranSearchEngine, test_queries_path: str):
#         """
#         Initialize evaluator
        
#         Args:
#             engine: Initialized QuranSearchEngine
#             test_queries_path: Path to test_queries.json
#         """
#         self.engine = engine
        
#         # Load test queries
#         with open(test_queries_path, 'r', encoding='utf-8') as f:
#             self.test_queries = json.load(f)
        
#         print(f"Loaded {len(self.test_queries)} test queries")
    
#     def evaluate_query(self, query_obj: Dict, k: int = 5) -> Dict:
#         """
#         Evaluate a single query
        
#         Args:
#             query_obj: Query object with 'query' and 'expected' fields
#             k: Number of results to retrieve
        
#         Returns:
#             Evaluation metrics for this query
#         """
#         query = query_obj['query']
#         expected_verse = query_obj['expected']
#         query_type = query_obj.get('type', 'unknown')
        
#         # Perform search
#         results = self.engine.search(query, top_k=k)
        
#         # Check if expected verse is in results
#         found_at_position = None
#         for i, result in enumerate(results):
#             if result['verse_key'] == expected_verse:
#                 found_at_position = i + 1  # 1-indexed
#                 break
        
#         # Calculate metrics
#         metrics = {
#             'query': query,
#             'expected': expected_verse,
#             'type': query_type,
#             'found': found_at_position is not None,
#             'position': found_at_position,
#             'precision_at_1': 1.0 if found_at_position == 1 else 0.0,
#             'recall_at_k': 1.0 if found_at_position is not None else 0.0,
#             'mrr': 1.0 / found_at_position if found_at_position else 0.0,  # Mean Reciprocal Rank
#         }
        
#         return metrics
    
#     def evaluate_all(self, k: int = 5) -> Dict:
#         """
#         Evaluate all test queries
        
#         Args:
#             k: Number of results to retrieve per query
        
#         Returns:
#             Aggregated metrics
#         """
#         print(f"\nEvaluating {len(self.test_queries)} queries...")
#         print("="*60)
        
#         all_metrics = []
        
#         for query_obj in self.test_queries:
#             metrics = self.evaluate_query(query_obj, k)
#             all_metrics.append(metrics)
            
#             # Print progress
#             status = "✓" if metrics['found'] else "✗"
#             print(f"{status} {query_obj['query'][:40]:<40} | Found: {metrics['found']} @ pos {metrics['position']}")
        
#         # Calculate aggregate metrics
#         total = len(all_metrics)
#         found_count = sum(1 for m in all_metrics if m['found'])
        
#         aggregate = {
#             'total_queries': total,
#             'found_count': found_count,
#             'accuracy': found_count / total if total > 0 else 0.0,
#             'precision_at_1': sum(m['precision_at_1'] for m in all_metrics) / total,
#             'recall_at_k': sum(m['recall_at_k'] for m in all_metrics) / total,
#             'mrr': sum(m['mrr'] for m in all_metrics) / total,  # Mean Reciprocal Rank
#         }
        
#         # Break down by query type
#         query_types = set(m['type'] for m in all_metrics)
#         aggregate['by_type'] = {}
        
#         for qtype in query_types:
#             type_metrics = [m for m in all_metrics if m['type'] == qtype]
#             if type_metrics:
#                 type_total = len(type_metrics)
#                 type_found = sum(1 for m in type_metrics if m['found'])
#                 aggregate['by_type'][qtype] = {
#                     'total': type_total,
#                     'found': type_found,
#                     'accuracy': type_found / type_total if type_total > 0 else 0.0
#                 }
        
#         return aggregate, all_metrics
    
#     def print_report(self, aggregate: Dict, detailed: bool = False):
#         """Print evaluation report"""
#         print("\n" + "="*60)
#         print("EVALUATION REPORT")
#         print("="*60)
        
#         print(f"\nOverall Metrics:")
#         print(f"  Total Queries: {aggregate['total_queries']}")
#         print(f"  Found: {aggregate['found_count']} / {aggregate['total_queries']}")
#         print(f"  Accuracy: {aggregate['accuracy']*100:.1f}%")
#         print(f"  Precision@1: {aggregate['precision_at_1']*100:.1f}%")
#         print(f"  Recall@K: {aggregate['recall_at_k']*100:.1f}%")
#         print(f"  MRR: {aggregate['mrr']:.3f}")
        
#         if aggregate.get('by_type'):
#             print(f"\nAccuracy by Query Type:")
#             for qtype, metrics in aggregate['by_type'].items():
#                 acc = metrics['accuracy'] * 100
#                 print(f"  {qtype:20s}: {acc:5.1f}% ({metrics['found']}/{metrics['total']})")
        
#         print("\n" + "="*60)
        
#         # Assessment
#         accuracy = aggregate['accuracy'] * 100
#         if accuracy >= 85:
#             print("✓ EXCELLENT: Accuracy meets target (≥85%)")
#         elif accuracy >= 70:
#             print("⚠ GOOD: Accuracy acceptable for MVP (≥70%)")
#         else:
#             print("✗ NEEDS IMPROVEMENT: Accuracy below target (<70%)")
        
#         print("="*60)


# def run_evaluation():
#     """Run complete evaluation"""
#     # Paths
#     base_dir = Path(__file__).parent.parent
#     data_path = base_dir / "output" / "processed" / "quran_complete.json"
#     test_queries_path = base_dir / "output" / "processed" / "test_queries.json"
    
#     # Initialize engine
#     print("Initializing search engine...")
#     engine = QuranSearchEngine(str(data_path))
    
#     # Initialize evaluator
#     evaluator = SearchEvaluator(engine, str(test_queries_path))
    
#     # Run evaluation
#     aggregate, detailed = evaluator.evaluate_all(k=5)
    
#     # Print report
#     evaluator.print_report(aggregate)
    
#     # Save results
#     output_path = base_dir / "output" / "processed" / "evaluation_results.json"
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump({
#             'aggregate': aggregate,
#             'detailed': detailed
#         }, f, ensure_ascii=False, indent=2)
    
#     print(f"\nDetailed results saved to: {output_path}")
    
#     return aggregate


# if __name__ == "__main__":
#     run_evaluation()