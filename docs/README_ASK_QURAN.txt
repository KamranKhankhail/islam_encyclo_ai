Islam Encyclo - Ask Qur'an v1 (Offline Evidence Engine)

What this adds (on top of your BM25 + E5 hybrid retrieval):
1) Intent routing:
   - structural references: 2:255, surah fatiha, etc.
   - quote lookup: user pastes Arabic/translation and asks 'where is this from?'
   - topic/story queries: 'where to spend zakat?', 'what happened to qaum e loot?'
   - count queries:
       a) exact word/phrase count (deterministic)
       b) semantic theme/instruction count (deterministic for given model + thresholding)

2) UX-first grouping:
   - results are grouped into contiguous ayah ranges, per surah, so users read context.

3) Optional count index artifacts:
   - gzipped pickle index for token counts + postings for fast counts and verse listing

Files included:
- hybrid_search_e5.py  (patched: semantic results now include relevance_score; exposes semantic_search)
- ask_quran_engine.py     (orchestrator)
- count_index.py          (count index build/load/query)
- build_count_index.py    (builder for count index)

Local paths (defaults wired into scripts):
- Data: output/processed/quran_complete.json
- Embeddings: output/processed/verse_embeddings_e5.npy
- Verse keys: output/processed/verse_keys_e5.json
- Count index (optional): output/processed/count_index.pkl.gz (created by builder)

How to run (Python reference, no args needed when repo artifacts are present):
1) Build count index (optional but recommended):
   python build_count_index.py

2) Run Ask Qur'an interactive (uses defaults above):
   python ask_quran_engine.py

Notes for mobile:
- Keep interfaces identical: router → retrieval → grouping → response.
- Replace Python embedding inference with on-device ONNX/NNAPI/CoreML.
- Memory-map verse embeddings (float16) for constant memory usage.
- Count index can be converted to a compact binary format later.
