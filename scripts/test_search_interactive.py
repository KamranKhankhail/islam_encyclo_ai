"""
Interactive Search Testing
Test search engine with custom queries
"""

from pathlib import Path
from search_engine import QuranSearchEngine


def print_result(result, index):
    """Pretty print a search result"""
    print(f"\n{'─'*60}")
    print(f"Result #{index}")
    print(f"{'─'*60}")
    print(f"📍 Location: {result['surah_name_english']} {result['surah']}:{result['ayah']}")
    print(f"📊 Score: {result['relevance_score']:.3f} | Type: {result['match_type']}")
    if result.get('juz'):
        print(f"📖 Juz: {result['juz']}")
    print(f"\n🔤 Arabic:")
    print(f"   {result['arabic']}")
    print(f"\n🇬🇧 English:")
    print(f"   {result['translation_english']}")
    print(f"\n🇵🇰 Urdu:")
    print(f"   {result['translation_urdu']}")


def main():
    """Interactive search loop"""
    # Initialize engine
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    
    print("="*60)
    print("ISLAM ENCYCLO AI - INTERACTIVE SEARCH")
    print("="*60)
    print("\nInitializing search engine...")
    
    engine = QuranSearchEngine(str(data_path))
    
    print("✓ Ready!")
    print("\nTips:")
    print("  - Try structural queries: '2:255', 'surah fatiha'")
    print("  - Try keywords: 'patience', 'prayer', 'صبر'")
    print("  - Try questions: 'what does quran say about...?'")
    print("  - Type 'quit' or 'exit' to stop")
    print("="*60)
    
    while True:
        # Get query from user
        query = input("\n🔍 Enter query: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Search
        results = engine.search_formatted(query, top_k=5)
        
        if not results:
            print("\n❌ No results found. Try different keywords.")
            continue
        
        print(f"\n✓ Found {len(results)} results:")
        
        # Display results
        for i, result in enumerate(results, 1):
            print_result(result, i)
        
        # Ask if user wants to see more details
        if len(results) > 3:
            show_more = input("\n👉 Show all results? (y/n): ").lower()
            if show_more == 'y':
                for i, result in enumerate(results[3:], 4):
                    print_result(result, i)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()