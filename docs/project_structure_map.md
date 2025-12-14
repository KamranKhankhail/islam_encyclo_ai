# IslamEncycloAI Project Structure

Snapshot of the current repo layout (excludes `__pycache__`, `.git/`, and the contents of `venv/`). Update this map when moving or adding major files.

```
IslamEncycloAI/
|-- .bm25_cache/
|-- .pytest_cache/
|-- ask_quran_v1_patch/
|   \-- README_ASK_QURAN.txt
|-- data/
|   |-- metadata.json
|   |-- quran_arabic.json
|   |-- quran_english_translations/
|   |   |-- ask_quran_v_1_offline_roadmap.md
|   |   \-- *.json (20 English translations)
|   \-- quran_urdu_translations/
|       \-- *.json (7 Urdu translations)
|-- docs/
|   |-- progress/
|   |   |-- day-1-progress.md
|   |   |-- day-1-complete.md
|   |   |-- day-2-complete.md
|   |   \-- day-3-complete.md
|   |-- claude-v1-day3-chatgpt-next-recommendations.md
|   |-- quran_e_5_embeddings_generation_build_release_documentation.md
|   \-- tuning-via-chatgpt.md
|-- eval/
|   \-- queries.jsonl
|-- output/
|   \-- processed/
|       |-- dataset_stats.json
|       |-- embeddings_meta_e5.json
|       |-- evaluation_results.json
|       |-- eval_hybrid_report.json
|       |-- eval_report_e5.json
|       |-- quran_compact.json
|       |-- quran_complete.json
|       |-- quran_complete.zip
|       |-- spell_index_en.json
|       |-- test_queries.json
|       |-- verse_embeddings_e5.npy
|       |-- verse_keys_e5.json
|       \-- verse_lookup.json
|-- scripts/
|   |-- arabic_normalizer.py
|   |-- ask_quran_engine.py
|   |-- build_count_index.py
|   |-- count_index.py
|   |-- evaluate.py
|   |-- evaluate_e5_hybrid.py
|   |-- generate_embeddings_e5.py
|   |-- hybrid_search_e5.py
|   |-- hybrid_search_e5_13122025.py
|   |-- metadata_verifier.py
|   |-- prepare_data.py
|   |-- query_parser.py
|   |-- search_engine.py
|   \-- test_search_interactive.py
|-- tests/
|   \-- test_data.py
|-- venv/ (Python virtual environment; contents not listed)
\-- .gitignore
```
