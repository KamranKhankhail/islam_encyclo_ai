Ask Qur’an v1 (Offline) — Progress + Next Steps

Context
You are building Islam Encyclo AI v1: “Ask Qur’an” fully offline on mobile.
Targets include:
- Structural lookup: “surah 2 ayah 255”, “2:255-257”
- Quote/source: “Where is this from?” (Arabic/Urdu/English/translit snippets)
- Analytics: “How many times in Qur’an X?” (multi-language)
- Topic/story: “What happened with qoum e loot?” (retrieve verses + coherent offline answer)

App Constraints
- Current app size: ~50 MB
- Allowed growth: up to ~300 MB
- Must run fully offline on device with excellent UX


1) What has been achieved so far (Production-grade baseline)

1.1 Data and Metadata
- metadata.json with complete surah list (1–114):
  - number, Arabic name, English name, transliteration, total_verses, revelation_type
- juz_mappings (1–30) start/end verse_key boundaries
- Validator script created for metadata integrity and range sanity

1.2 Arabic Normalization (High quality)
- Arabic normalizer improved for search usage:
  - diacritics/tashkeel removal
  - tatweel removal
  - robust letter normalization (hamza/alef/yeh/teh variants)
  - whitespace normalization
- Tokenization support

1.3 Structural Query Parsing (High coverage)
- StructuralQueryParser supports:
  - numeric formats: 2:255, 2-255, “2 255”
  - “surah 2 ayah 255”, “verse 2:255”
  - surah name variations
  - verse nicknames (Ayat al Kursi etc.)
- Structural match returns direct verse with high confidence

1.4 BM25F-like Offline Search Engine (Keyword + Field weights)
- QuranSearchEngine (hybrid structural + lexical)
- BM25 (rank_bm25) index over verse corpus
- Multi-language tokenization (Arabic vs non-Arabic)
- Field weighting and language-aware scoring (as per current implementation improvements)
- Fast lookup via verse_key map

1.5 Evaluation Framework (Professional metrics + latency)
- Evaluator produces:
  - Accuracy, Precision@1, Recall@K, MRR
  - Latency mean/p50/p95/min/max
  - Breakdown by query type

1.6 Current Evaluation Outcome
Overall Metrics:
- Total Queries: 12
- Found: 11/12
- Accuracy: 91.7%
- Precision@1: 83.3%
- Recall@K: 91.7%
- MRR: 0.861
Latency (ms): Mean 36.41, P50 49.56, P95 67.82
Only miss: semantic_ur ("صبر کے بارے میں آیت")

Status Summary
- The MVP search engine is already production-capable for:
  - structural lookup
  - exact/partial matches
  - many English semantic-ish cases via translation overlap
- Remaining gaps are mainly:
  - Urdu semantic phrasing mismatch
  - broader “topic/story/analytics” capability (beyond retrieval)


2) Claude’s proposed next step (Day 3): Semantic Search Boost

Objective
- Add lightweight semantic embeddings to improve semantic matching and cross-language paraphrases.
- Expected improvement: accuracy ~91.7% -> 93–95% (typical)
- Specifically targets semantic_ur failure and paraphrased/topic queries.

Claude’s Architecture
- Hybrid:
  - BM25 generates top N candidates (e.g., 50)
  - Semantic reranker scores these candidates using embeddings
  - Return top K after combined scoring

Model suggestion
- paraphrase-multilingual-MiniLM-L12-v2
- Generate verse embeddings offline, cache to disk
- Runtime: embed query, compute similarity against candidates

Deliverables (Claude)
- Embedding generation script
- Hybrid search engine (BM25 + semantic rerank)
- Evaluation comparing BM25-only vs Hybrid

Mobile consideration (important)
Claude’s plan is sound conceptually but must be implemented mobile-first:
- Use ONNX/quantized model suitable for on-device
- Precompute verse embeddings and ship them in app bundle
- Rerank only small candidate set for UX smoothness


3) After implementing Claude’s semantic step: My recommended next steps (to reach Ask Qur’an v1 offline)

Guiding Principle
Do not expand endless rules. Build durable foundations that scale:
Retrieve (BM25F + Vector) -> Rerank (learned) -> Execute (tools) with logging.

3.1 Step 1: Make Hybrid Retrieval Mobile-Ready
Goal: deliver consistent offline performance.
- Export embedding model to ONNX and quantize (int8/float16) for on-device inference
- Precompute verse embeddings offline:
  - store float16 vectors
  - optionally store per-field embeddings (Arabic/English/Urdu/translit) later
- Implement vector search strategy:
  - simplest: semantic rerank only on BM25 candidates
  - upgrade: ANN top-N retrieval + union with BM25 (same API)

Acceptance Criteria
- Semantic_ur query now returns expected verse in top K
- Overall latency still within target (p95 < 150 ms on device)
- No UI jitter: BM25 results shown immediately, semantic rerank updates once


3.2 Step 2: Add Concordance/Occurrence Index (Exact analytics)
Unlock queries like:
- “how many times in quran sabar?”
- “list occurrences of صبر with verse refs”
- multilingual counts (Arabic/Urdu/English/translit)

What to build
- Offline inverted index per field/language:
  - token -> total_count
  - token -> list of (verse_key, count_in_verse)
  - optional positions for context snippets
- This is exact and near-instant; not a fuzzy search problem.

Acceptance Criteria
- count(term) returns exact counts
- list_occurrences(term) returns verse refs + optional snippets
- Works offline with low latency (<20 ms typical)


3.3 Step 3: Intent Routing (Search vs Tools vs Structural)
Goal: one entry point for “Ask Qur’an”.

Intent classes (v1)
- structural (surah/ayah/range)
- quote_source ("where is this from?")
- analytics_count ("how many times")
- topic_story ("what happened with qoum e loot")
- general_search

Implementation approach
- Start with safe deterministic detection for structural + analytics keywords
- Move toward a small lightweight classifier later (still offline)

Acceptance Criteria
- Queries are routed correctly to:
  - structural resolver
  - retrieval pipeline
  - concordance tools


3.4 Step 4: Quote/Source Identification Mode
Goal: “Where is this from?”

Approach
- Use same hybrid retrieval pipeline
- Add match confidence:
  - exact/near-exact substring match score
  - semantic similarity score
  - lexical overlap
- Return best verse_key + optional top-3 alternatives + confidence

Acceptance Criteria
- Works for Arabic diacritics variants and translation snippets
- Stable confidence thresholding to avoid wrong “confident” answers


3.5 Step 5: Topic/Story Answering (Offline)
Goal: “What happened with qoum e loot?”

Approach
- Retrieve top N verses using hybrid search
- Cluster by surah/range
- Produce an offline answer format:
  - short structured summary
  - primary verse references (ranges)
  - key excerpts (translations)

Implementation note
- For v1, summary can be template-based and citation-driven
- Later, you can add an on-device small generator without changing architecture

Acceptance Criteria
- Answer includes citations (verse_key) always
- Returns coherent set of verses rather than a single verse


3.6 Step 6: Learning Loop (Offline-first)
Goal: system improves over time without hardcoded expansions.

On-device personalization
- Log local signals:
  - clicks, dwell time, copy/share, reformulation
- Maintain small reranker adjustments locally (optional v1)

Global improvement path
- Ship periodic model/index updates (app updates or downloadable packs)

Acceptance Criteria
- Reranker can be updated independently from app code
- Evaluation suite expands and regressions are prevented


4) Recommended Evaluation Upgrades After Hybrid
To avoid misleading “single-expected verse” limits for broad semantic queries:
- Allow multiple expected answers (list/graded) for topic/story queries
- Track:
  - Success@K, nDCG@K, MAP@K
- Keep Precision@1 strict for structural/canonical queries


5) What the novice AI engineer should do next (strict execution order)

Phase A (Claude step, mobile-first)
1) Implement semantic rerank on top of BM25 candidates
2) Export/quantize embedding model for mobile
3) Precompute verse embeddings offline and package them
4) Re-run evaluation and confirm semantic_ur passes

Phase B (Ask Qur’an foundations)
5) Build concordance index for exact counts + occurrences
6) Add intent router and tool interfaces
7) Implement quote/source mode
8) Implement topic/story mode with citations
9) Expand evaluation suite to cover analytics and stories


Definition of Done for Ask Qur’an v1 (Offline)
- Structural queries: 100%
- Quote/source queries: high confidence with citations
- Count/occurrence queries: exact and instant
- Topic/story queries: citation-backed, coherent results
- Fully offline on device; p95 latency within UX target
- Architecture remains stable for future upgrades (better models, reranker training, on-device generation)

