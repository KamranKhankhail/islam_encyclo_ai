Understood. The next step is to stop guessing and start **measuring**—with a small, high-quality **test collection** (queries + relevance judgments) and an **offline evaluation harness** that lets you tune BM25 parameters/field weights/synonyms/stemming safely.

This is standard IR practice: build a labeled set, compute rank-aware metrics (MRR/nDCG/MAP/Recall@K), tune, and lock in regressions. ([Pinecone][1])

Below is a complete, production-usable plan and code.

---

## 1) What you will build

### A. Gold dataset (human-judged)

`eval/queries.jsonl` (one JSON per line):

```json
{"qid":"q0001","query":"patience","lang":"en","relevance":{"2:153":2,"3:200":1},"notes":"Core verses about sabr"}
{"qid":"q0002","query":"صبر","lang":"ar","relevance":{"2:153":2,"3:200":1}}
{"qid":"q0003","query":"2:255","lang":"struct","relevance":{"2:255":3}}
```

Grades are 0–3:

* 3 = exact / primary
* 2 = strongly relevant
* 1 = weakly relevant
* (missing) = not relevant

This allows **nDCG@K** (graded ranking quality) and still works for MRR/MAP/Recall. ([Pinecone][1])

### B. Pooling workflow (reduces labeling workload)

Instead of judging against all 6236 verses, you label only a **candidate pool**:

* Run 2–4 engine configs
* Collect top 50 per query
* Union + dedupe
* Label that pool

Pooling is a classic evaluation approach in IR practice. ([Stanford University][2])

---

## 2) Metrics you should track (and why)

For your app UX, the key “feels good” objective is: **users see a correct verse quickly**.

* **MRR@K**: how early the first relevant result appears (best for “find the answer fast”). ([Pinecone][1])
* **nDCG@K**: rewards putting the “most relevant” verses higher (graded relevance). ([Pinecone][1])
* **Recall@K**: ensures you don’t miss relevant verses in top K (important for religious content discovery). ([Pinecone][1])
* **MAP@K**: balances precision across multiple relevant items. ([Pinecone][1])

---

## 3) The evaluation harness (drop-in scripts)

### `eval/run_eval.py`

This loads your engine, runs queries, and prints metrics.

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
from typing import Any, Dict, List, Tuple, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dcg(rels: List[float]) -> float:
    # DCG with log2 discount starting at rank 1
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2.0 ** r - 1.0) / math.log2(i + 1.0)
    return s


def ndcg_at_k(ranked_keys: List[str], qrel: Dict[str, float], k: int) -> float:
    rels = [float(qrel.get(vk, 0.0)) for vk in ranked_keys[:k]]
    ideal = sorted([float(v) for v in qrel.values()], reverse=True)[:k]
    if not ideal:
        return 0.0
    denom = dcg(ideal)
    if denom <= 0:
        return 0.0
    return dcg(rels) / denom


def mrr_at_k(ranked_keys: List[str], qrel: Dict[str, float], k: int) -> float:
    for i, vk in enumerate(ranked_keys[:k], start=1):
        if qrel.get(vk, 0.0) > 0:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_keys: List[str], qrel: Dict[str, float], k: int) -> float:
    relevant = {vk for vk, g in qrel.items() if float(g) > 0}
    if not relevant:
        return 0.0
    got = sum(1 for vk in ranked_keys[:k] if vk in relevant)
    return got / float(len(relevant))


def ap_at_k(ranked_keys: List[str], qrel: Dict[str, float], k: int) -> float:
    relevant = {vk for vk, g in qrel.items() if float(g) > 0}
    if not relevant:
        return 0.0
    hits = 0
    s = 0.0
    for i, vk in enumerate(ranked_keys[:k], start=1):
        if vk in relevant:
            hits += 1
            s += hits / float(i)
    return s / float(len(relevant))


def mean(xs: List[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def import_engine(engine_module: str, engine_class: str):
    mod = importlib.import_module(engine_module)
    cls = getattr(mod, engine_class)
    return cls


def build_engine(cls, data_path: str, metadata_path: Optional[str], engine_kwargs_json: Optional[str]):
    kwargs = {}
    if engine_kwargs_json:
        kwargs = json.loads(engine_kwargs_json)
        if not isinstance(kwargs, dict):
            raise ValueError("--engine-kwargs must decode to a JSON object")
    # Convention: your engine __init__(data_path, metadata_path=None, ...)
    try:
        return cls(data_path=data_path, metadata_path=metadata_path, **kwargs)
    except TypeError:
        # Fallback: allow positional signature (data_path, metadata_path)
        return cls(data_path, metadata_path, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, help="Path to eval/queries.jsonl")
    p.add_argument("--data", required=True, help="Path to quran_complete.json")
    p.add_argument("--metadata", default=None, help="Optional path to metadata.json")

    p.add_argument("--engine-module", required=True, help="Python module name, e.g. quran_search_engine")
    p.add_argument("--engine-class", default="QuranSearchEngine", help="Class name in module")
    p.add_argument("--engine-kwargs", default=None, help="JSON dict of extra kwargs to engine constructor")

    p.add_argument("--k", type=int, default=10, help="K for @K metrics")
    p.add_argument("--pool-k", type=int, default=50, help="How many results to retrieve per query")
    args = p.parse_args()

    rows = load_jsonl(args.queries)
    EngineCls = import_engine(args.engine_module, args.engine_class)
    engine = build_engine(EngineCls, args.data, args.metadata, args.engine_kwargs)

    mrrs: List[float] = []
    ndcgs: List[float] = []
    recalls: List[float] = []
    maps: List[float] = []

    per_query_out: List[Dict[str, Any]] = []

    for row in rows:
        qid = row.get("qid")
        query = row.get("query")
        qrel = row.get("relevance") or {}
        if not isinstance(qrel, dict):
            raise ValueError(f"{qid}: relevance must be an object mapping verse_key -> grade")

        results = engine.search(query, top_k=args.pool_k)
        ranked_keys = [r.get("verse_key") for r in results if isinstance(r, dict) and isinstance(r.get("verse_key"), str)]

        mrr = mrr_at_k(ranked_keys, qrel, args.k)
        nd = ndcg_at_k(ranked_keys, qrel, args.k)
        rc = recall_at_k(ranked_keys, qrel, args.k)
        ap = ap_at_k(ranked_keys, qrel, args.k)

        mrrs.append(mrr)
        ndcgs.append(nd)
        recalls.append(rc)
        maps.append(ap)

        per_query_out.append({
            "qid": qid,
            "query": query,
            "MRR@k": mrr,
            "nDCG@k": nd,
            "Recall@k": rc,
            "MAP@k": ap,
            "top": ranked_keys[:args.k],
        })

    summary = {
        f"MRR@{args.k}": mean(mrrs),
        f"nDCG@{args.k}": mean(ndcgs),
        f"Recall@{args.k}": mean(recalls),
        f"MAP@{args.k}": mean(maps),
        "num_queries": len(rows),
    }

    print(json.dumps({"summary": summary, "per_query": per_query_out}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `eval/build_pool.py`

Generates a candidate pool for annotation (top-N per query).

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def import_engine(engine_module: str, engine_class: str):
    mod = importlib.import_module(engine_module)
    return getattr(mod, engine_class)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--metadata", default=None)
    p.add_argument("--engine-module", required=True)
    p.add_argument("--engine-class", default="QuranSearchEngine")
    p.add_argument("--engine-kwargs", default=None)
    p.add_argument("--topn", type=int, default=50)
    p.add_argument("--out", required=True, help="Output JSONL pool file")
    args = p.parse_args()

    rows = load_jsonl(args.queries)
    EngineCls = import_engine(args.engine_module, args.engine_class)
    kwargs = json.loads(args.engine_kwargs) if args.engine_kwargs else {}
    engine = EngineCls(data_path=args.data, metadata_path=args.metadata, **kwargs)

    with open(args.out, "w", encoding="utf-8") as out:
        for row in rows:
            qid = row.get("qid")
            query = row.get("query")
            results = engine.search(query, top_k=args.topn)

            # write a pool record that is easy to judge
            pool = []
            for r in results:
                if not isinstance(r, dict):
                    continue
                pool.append({
                    "verse_key": r.get("verse_key"),
                    "arabic": (r.get("arabic") or "")[:160],
                    "en": ( (r.get("translations_english") or {}).get("sahih-international") or r.get("translation_en_builtin") or "" )[:220],
                    "ur": ( (r.get("translations_urdu") or {}).get("maulana-abu-al-maududi") or r.get("translation_ur_builtin") or "" )[:220],
                })

            out.write(json.dumps({
                "qid": qid,
                "query": query,
                "candidates": pool,
                "relevance": row.get("relevance", {}),  # keep existing labels if any
                "notes": row.get("notes", ""),
            }, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `eval/annotate_pool.py` (fast CLI labeling)

This keeps you moving: choose grade with keystrokes.

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="Pool file from build_pool.py")
    p.add_argument("--out", required=True, help="Writes updated pool (with relevance grades)")
    args = p.parse_args()

    rows = load_jsonl(args.pool)

    for row in rows:
        qid = row.get("qid")
        query = row.get("query")
        rel: Dict[str, int] = row.get("relevance") or {}
        if not isinstance(rel, dict):
            rel = {}

        print("\n" + "=" * 90)
        print(f"QID: {qid} | Query: {query}")
        print("Grade: 3=primary 2=strong 1=weak 0=not  s=skip  q=quit")
        print("=" * 90)

        for c in row.get("candidates", []):
            vk = c.get("verse_key")
            if not isinstance(vk, str) or not vk:
                continue

            cur = int(rel.get(vk, 0) or 0)
            print("\n" + "-" * 90)
            print(f"{vk}  (current: {cur})")
            print("AR:", c.get("arabic", ""))
            print("EN:", c.get("en", ""))
            print("UR:", c.get("ur", ""))

            while True:
                ans = input("grade> ").strip().lower()
                if ans in ("q", "quit"):
                    row["relevance"] = rel
                    save_jsonl(args.out, rows)
                    print("Saved and exiting.")
                    return 0
                if ans in ("s", "skip", ""):
                    break
                if ans in ("0", "1", "2", "3"):
                    rel[vk] = int(ans)
                    break
                print("Enter 0/1/2/3 or s or q")

        row["relevance"] = rel

    save_jsonl(args.out, rows)
    print("Done. Saved:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 4) How to use this (exact commands)

1. Create initial query set (start with 120–200 queries):

* 30 Arabic keyword queries (1–3 tokens)
* 30 English queries (keywords + natural language)
* 30 Urdu queries
* 20 transliteration queries (e.g., “bismillah”, “sabr”, “taqwa”)
* 20 “hard” queries (ambiguous / multi-topic)
* 20 structural queries (single + ranges)

2. Pool candidates (top 50):

```bash
python eval/build_pool.py \
  --queries eval/queries.jsonl \
  --data output/processed/quran_complete.json \
  --metadata output/processed/metadata.json \
  --engine-module quran_search_engine \
  --engine-class QuranSearchEngine \
  --topn 50 \
  --out eval/pool.jsonl
```

3. Annotate:

```bash
python eval/annotate_pool.py --pool eval/pool.jsonl --out eval/pool_labeled.jsonl
```

4. Convert labeled pool into the gold `queries.jsonl` you’ll actually use (simple approach: copy the `relevance` maps back into `queries.jsonl`, or just evaluate directly from `pool_labeled.jsonl` by treating it as queries input).

5. Run evaluation:

```bash
python eval/run_eval.py \
  --queries eval/queries.jsonl \
  --data output/processed/quran_complete.json \
  --metadata output/processed/metadata.json \
  --engine-module quran_search_engine \
  --engine-class QuranSearchEngine \
  --k 10 \
  --pool-k 50
```

---

## 5) Tuning protocol (what to tune, in what order)

Tune in this order because each step changes the trade-off surface:

1. **Field weights / BM25F-style blending** (biggest gain for mixed-language corpora) ([ACM Digital Library][3])
2. **k1 / b** (sensitive to field lengths and tokenization) ([ACM Digital Library][3])
3. **Stopword policy** (adaptive vs always)
4. **Arabic light stemming** (only if your eval improves—Arabic IR literature shows stemming can help, but can introduce errors/overmatching) ([SpringerLink][4])
5. **Synonym expansion** (keep conservative; verify it doesn’t “query drift”) ([Georgetown - Home][5])

Lock each improvement with:

* `baseline.json` metrics
* `current.json` metrics
* a “regression list” of queries that got worse (per-query output from `run_eval.py` enables this)

---

## 6) If you want the *real* “best” outcome

Add one more artifact: **an evaluation dashboard** that plots metrics across versions and shows “worst-regressed queries.” That’s what prevents quality from slipping as you add features.

If you want, I’ll provide:

* `eval/grid_tune.py` (grid search over config weights/k1/b)
* `eval/compare_runs.py` (diff per-query regressions)
* a minimal HTML report generator (no dependencies)

Tell me whether your current engine module is named exactly `quran_search_engine.py` (and what the class name is), and I’ll tailor the tuning scripts to your exact constructor/config structure without placeholders.

[1]: https://www.pinecone.io/learn/offline-evaluation/?utm_source=chatgpt.com "Evaluation Measures in Information Retrieval"
[2]: https://web.stanford.edu/class/cs276/handouts/lecture12-bm25etc.pdf?utm_source=chatgpt.com "lecture12-bm25etc.pdf"
[3]: https://dl.acm.org/doi/abs/10.1561/1500000019?utm_source=chatgpt.com "The Probabilistic Relevance Framework: BM25 and Beyond"
[4]: https://link.springer.com/chapter/10.1007/978-1-4020-6046-5_12?utm_source=chatgpt.com "Light Stemming for Arabic Information Retrieval"
[5]: https://ir.cs.georgetown.edu/downloads/p340-aljlayl.pdf?utm_source=chatgpt.com "Improving the Retrieval Effectiveness via a Light Stemming ..."
