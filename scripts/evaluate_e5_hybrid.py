"""
E5 Hybrid Search Evaluation (Production-Grade)

Compares:
- BM25 baseline
- E5 Hybrid (BM25 + E5 semantic + fusion)

Notes:
- This script evaluates *our system* on *our test set*.
- It does NOT prove "E5 > MiniLM" unless you also run a MiniLM hybrid variant on the same set.
- For semantic/topic queries, strict single-answer "accuracy" can be misleading; prefer Recall@K / MRR / nDCG where available.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

from hybrid_search_e5 import E5HybridSearchEngine, HybridConfig
from search_engine import QuranSearchConfig
from evaluate import SearchEvaluator


# ---- Config ----

DEFAULT_K = 5
SEMANTIC_WEIGHTS_TO_TRY = [0.20, 0.30, 0.40]  # set to [0.30] if you want fixed only
SAVE_REPORT = True


def _metric(agg: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """
    Safely extract a metric even if evaluator naming differs.
    Example: accuracy vs recall@k, mrr vs mrr@10, etc.
    """
    for k in keys:
        if k in agg and isinstance(agg[k], (int, float)):
            return float(agg[k])
        # nested forms occasionally appear
        if k in agg and isinstance(agg[k], dict) and "mean" in agg[k]:
            try:
                return float(agg[k]["mean"])
            except Exception:
                pass
    return float(default)


def _lat_mean_ms(agg: Dict[str, Any]) -> float:
    lat = agg.get("latency_ms", {})
    if isinstance(lat, dict):
        return float(lat.get("mean", 0.0) or 0.0)
    return 0.0


def _warmup(engine: E5HybridSearchEngine) -> None:
    """
    Warmup to avoid first-query GPU/kernel overhead affecting latency stats.
    """
    try:
        engine.search("2:255", top_k=1)
        engine.search("patience", top_k=1)
    except Exception:
        # warmup should never crash evaluation; ignore if something is unavailable
        pass


def _build_engine(
    data_path: str,
    metadata_path: Optional[str],
    enable_semantic: bool,
    semantic_weight: float,
    enable_cache: bool,
) -> E5HybridSearchEngine:
    config = QuranSearchConfig(enable_cache=enable_cache)

    hybrid = HybridConfig(
        enable_semantic=enable_semantic,
        bm25_weight=1.0 - float(semantic_weight),
        semantic_weight=float(semantic_weight),
        fusion_method="rrf",
        rrf_k=60,
        bm25_topn=80,
        semantic_topn=80,
        final_topk=DEFAULT_K,
        semantic_scan_mode="full",   # Qur’an-scale: best recall
        mmap_embeddings=True,
        prefer_cuda=True,
        query_cache_size=256,
        semantic_chunk_rows=0,
    )

    # Keep alias/common-name map consistent across evaluations
    alias_map = {
        "ayat ul kursi": "2:255",
        "ayatul kursi": "2:255",
        "ayat al kursi": "2:255",
        "ayatulkursi": "2:255",
    }

    return E5HybridSearchEngine(
        data_path=data_path,
        metadata_path=metadata_path,
        config=config,
        hybrid=hybrid,
        alias_map=alias_map,
    )


def _evaluate(engine: E5HybridSearchEngine, test_queries_path: str, k: int) -> Tuple[Dict[str, Any], Any]:
    evaluator = SearchEvaluator(engine, test_queries_path)
    aggregate, detailed = evaluator.evaluate_all(k=k)
    evaluator.print_report(aggregate)
    return aggregate, detailed


def main():
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    metadata_path = base_dir / "output" / "processed" / "metadata.json"
    test_queries_path = base_dir / "output" / "processed" / "test_queries.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing: {data_path}")
    if not test_queries_path.exists():
        raise FileNotFoundError(f"Missing: {test_queries_path}")

    meta_path_str = str(metadata_path) if metadata_path.exists() else None

    print("=" * 70)
    print("E5 HYBRID SEARCH EVALUATION (PRODUCTION-GRADE)")
    print("=" * 70)
    print(f"Dataset: {data_path}")
    print(f"Queries:  {test_queries_path}")
    print(f"K:       {DEFAULT_K}")
    print()

    # ---------------------------
    # Baseline: BM25 only
    # ---------------------------
    print("=" * 70)
    print("BASELINE: BM25 ONLY")
    print("=" * 70)

    engine_bm25 = _build_engine(
        data_path=str(data_path),
        metadata_path=meta_path_str,
        enable_semantic=False,
        semantic_weight=0.0,  # irrelevant when semantic disabled
        enable_cache=True,
    )
    _warmup(engine_bm25)

    agg_bm25, det_bm25 = _evaluate(engine_bm25, str(test_queries_path), k=DEFAULT_K)

    bm25_acc = _metric(agg_bm25, "accuracy", f"recall@{DEFAULT_K}", default=0.0)
    bm25_mrr = _metric(agg_bm25, "mrr", "mrr@10", "mrr@5", default=0.0)
    bm25_lat = _lat_mean_ms(agg_bm25)

    # ---------------------------
    # Hybrid: try weights grid
    # ---------------------------
    best = None
    all_runs: List[Dict[str, Any]] = []

    for w in SEMANTIC_WEIGHTS_TO_TRY:
        print("\n" + "=" * 70)
        print(f"HYBRID: BM25 + E5 (semantic_weight={w:.2f})")
        print("=" * 70)

        engine_e5 = _build_engine(
            data_path=str(data_path),
            metadata_path=meta_path_str,
            enable_semantic=True,
            semantic_weight=w,
            enable_cache=True,
        )
        _warmup(engine_e5)

        agg_e5, det_e5 = _evaluate(engine_e5, str(test_queries_path), k=DEFAULT_K)

        e5_acc = _metric(agg_e5, "accuracy", f"recall@{DEFAULT_K}", default=0.0)
        e5_mrr = _metric(agg_e5, "mrr", "mrr@10", "mrr@5", default=0.0)
        e5_lat = _lat_mean_ms(agg_e5)

        run = {
            "semantic_weight": w,
            "aggregate": agg_e5,
            "summary": {
                "acc_like": e5_acc,
                "mrr_like": e5_mrr,
                "lat_mean_ms": e5_lat,
                "delta_acc_like": e5_acc - bm25_acc,
                "delta_mrr_like": e5_mrr - bm25_mrr,
                "delta_lat_ms": e5_lat - bm25_lat,
            },
        }
        all_runs.append(run)

        # Choose best primarily by acc-like, then MRR, then latency
        if best is None:
            best = run
        else:
            b = best["summary"]
            r = run["summary"]
            if (r["acc_like"], r["mrr_like"], -r["lat_mean_ms"]) > (b["acc_like"], b["mrr_like"], -b["lat_mean_ms"]):
                best = run

    # ---------------------------
    # Final comparison print
    # ---------------------------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"BM25 acc-like: {bm25_acc*100:.1f}% | mrr-like: {bm25_mrr:.3f} | mean latency: {bm25_lat:.2f}ms")

    if best:
        b = best["summary"]
        w = best["semantic_weight"]
        print(f"BEST HYBRID (w={w:.2f}) acc-like: {b['acc_like']*100:.1f}% | mrr-like: {b['mrr_like']:.3f} | mean latency: {b['lat_mean_ms']:.2f}ms")
        print(f"Δ acc-like: {b['delta_acc_like']*100:+.1f}% | Δ mrr-like: {b['delta_mrr_like']:+.3f} | Δ latency: {b['delta_lat_ms']:+.2f}ms")

    # Per-type breakdown if evaluator provides it
    by_type_bm25 = agg_bm25.get("by_type", {})
    if isinstance(by_type_bm25, dict) and best and isinstance(best["aggregate"].get("by_type", {}), dict):
        print("\nACCURACY BY TYPE (if present in evaluator):")
        print(f"{'Type':<20} {'BM25':>8} {'Hybrid':>8} {'Δ':>8}")
        print("-" * 50)

        by_type_e5 = best["aggregate"].get("by_type", {})
        all_types = sorted(set(by_type_bm25.keys()) | set(by_type_e5.keys()))
        for t in all_types:
            a_b = _metric(by_type_bm25.get(t, {}), "accuracy", f"recall@{DEFAULT_K}", default=0.0) * 100
            a_e = _metric(by_type_e5.get(t, {}), "accuracy", f"recall@{DEFAULT_K}", default=0.0) * 100
            print(f"{t:<20} {a_b:7.1f}% {a_e:7.1f}% {a_e-a_b:+7.1f}%")

    # Save report
    result = {
        "k": DEFAULT_K,
        "baseline": {"aggregate": agg_bm25},
        "hybrid_runs": all_runs,
        "best_hybrid": best,
        "generated_at_unix": int(time.time()),
    }

    if SAVE_REPORT:
        out_path = base_dir / "output" / "processed" / "eval_hybrid_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {out_path}")

    return result


if __name__ == "__main__":
    main()
