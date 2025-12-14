from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from hybrid_search_e5 import E5HybridSearchEngine, HybridConfig  # noqa: E402
from search_engine import QuranSearchConfig  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    base = Path(__file__).parent.parent
    data_path = base / "output" / "processed" / "quran_complete.json"
    metadata_path = base / "output" / "processed" / "metadata.json"

    hybrid = HybridConfig(
        enable_semantic=False,
        bm25_topn=80,
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


@pytest.mark.parametrize(
    "query",
    [
        "2:255",
        "surah fatiha",
        "where to spend zakat",
        "صبر",
        "ayat ul kursu",
    ],
)
def test_deterministic_order(engine, query):
    resp1 = engine.search(query)
    resp2 = engine.search(query)

    assert resp1["version"] == 1
    for key in ["version", "packVersion", "query", "page", "results"]:
        assert key in resp1

    keys1 = [r["verseKey"] for r in resp1.get("results", [])]
    keys2 = [r["verseKey"] for r in resp2.get("results", [])]
    assert keys1 == keys2
