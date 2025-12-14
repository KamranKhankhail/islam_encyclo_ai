from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from hybrid_search_e5 import E5HybridSearchEngine, HybridConfig  # noqa: E402
from search_engine import QuranSearchConfig  # noqa: E402


def make_engine() -> E5HybridSearchEngine:
    base = Path(__file__).parent.parent
    data_path = base / "output" / "processed" / "quran_complete.json"
    metadata_path = base / "output" / "processed" / "metadata.json"

    hybrid = HybridConfig(
        enable_semantic=False,
        bm25_topn=50,
        semantic_topn=10,
        final_topk=50,
        max_results=50,
    )
    cfg = QuranSearchConfig(enable_cache=False)

    return E5HybridSearchEngine(
        data_path=str(data_path),
        metadata_path=str(metadata_path) if metadata_path.exists() else None,
        config=cfg,
        embeddings_path=None,
        verse_keys_path=None,
        hybrid=hybrid,
    )


def test_result_schema_fields():
    engine = make_engine()
    resp = engine.search("2:255")
    results = resp.get("results", [])
    assert results, "expected at least one result"

    r = results[0]
    expected_keys = {"verseKey", "surah", "ayah", "arabic", "display", "meta", "score"}
    assert expected_keys == set(r.keys())

    display = r["display"]
    assert {"lang", "text"} == set(display.keys())

    meta = r["meta"]
    assert {"juz", "ruku", "page"} <= set(meta.keys())

    score = r["score"]
    assert {"mode", "rrf", "bm25", "semantic"} == set(score.keys())
