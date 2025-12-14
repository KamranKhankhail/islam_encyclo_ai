"""
Interactive Search Testing - AskQuran Response Format v1
"""

from hybrid_search_e5 import get_e5_engine


def print_result(result: dict, index: int) -> None:
    score = result.get("score") or {}
    display = result.get("display") or {}

    print(f"\n{'-'*60}")
    print(f"Result #{index}  {result.get('verseKey')}")
    print(f"{'-'*60}")
    print(
        "mode={mode} | rrf={rrf} | bm25={bm25} | semantic={semantic}".format(
            mode=score.get("mode"),
            rrf=score.get("rrf"),
            bm25=score.get("bm25"),
            semantic=score.get("semantic"),
        )
    )
    print(f"display [{display.get('lang')}]: {display.get('text')}")
    print("arabic:")
    print(f"  {result.get('arabic')}")


def main() -> None:
    engine = get_e5_engine()

    print("=" * 60)
    print("ISLAM ENCYCLO AI - INTERACTIVE SEARCH")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        resp = engine.search(query, top_k=10)
        results = resp.get("results", [])
        intent = resp.get("query", {}).get("intent")

        if not results:
            print("No results found.")
            continue

        print(f"\nFound {len(results)} results (intent={intent}):")
        for i, result in enumerate(results, 1):
            print_result(result, i)


if __name__ == "__main__":
    main()
