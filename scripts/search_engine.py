"""
BM25 Search Engine for Quran
Implements keyword-based search with ranking
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import re

from arabic_normalizer import normalize_for_search, tokenize_arabic
from query_parser import StructuralQueryParser


class QuranSearchEngine:
    """Hybrid search engine: Structural + Lexical (BM25)"""
    
    def __init__(self, data_path: str):
        """
        Initialize search engine with Quran data
        
        Args:
            data_path: Path to quran_complete.json
        """
        print("Loading Quran data...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.verses = json.load(f)
        
        print(f"Loaded {len(self.verses)} verses")
        
        # Create verse lookup for fast access
        self.verse_lookup = {v['verse_key']: v for v in self.verses}
        
        # Initialize parsers
        self.structural_parser = StructuralQueryParser()
        
        # Prepare corpus for BM25
        print("Building search index...")
        self._build_search_index()
        print("Search engine ready!")
    
    def _build_search_index(self):
        """Build BM25 index from verse corpus"""
        # Tokenize all searchable text
        self.corpus_tokens = []
        
        for verse in self.verses:
            # Get searchable text and normalize
            text = verse.get('searchable_text', '')
            
            # Tokenize (handles Arabic + English + Urdu)
            tokens = self._tokenize_multilingual(text)
            self.corpus_tokens.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus_tokens)
    
    def _tokenize_multilingual(self, text: str) -> List[str]:
        """
        Tokenize text that may contain Arabic, English, Urdu
        """
        tokens = []
        
        # Split into words
        words = text.split()
        
        for word in words:
            # Check if word contains Arabic/Urdu characters
            if self._contains_arabic(word):
                # Normalize and tokenize Arabic
                normalized = normalize_for_search(word)
                if normalized:
                    tokens.append(normalized)
            else:
                # English/transliteration - just lowercase
                normalized = word.lower().strip()
                # Remove punctuation
                normalized = re.sub(r'[^\w\s]', '', normalized)
                if normalized and len(normalized) > 1:  # Skip single chars
                    tokens.append(normalized)
        
        return tokens
    
    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        return bool(arabic_pattern.search(text))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for verses matching the query
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of verse results with relevance scores
        """
        # Step 1: Try structural query parsing first
        structural_result = self.structural_parser.parse(query)
        
        if structural_result:
            # Direct structural match
            verse_key = structural_result['verse_key']
            if verse_key in self.verse_lookup:
                verse = self.verse_lookup[verse_key].copy()
                verse['relevance_score'] = structural_result['confidence']
                verse['match_type'] = 'structural'
                return [verse]
            else:
                # Invalid verse reference
                return []
        
        # Step 2: Lexical search using BM25
        return self._lexical_search(query, top_k)
    
    def _lexical_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform BM25 keyword search
        """
        # Tokenize query
        query_tokens = self._tokenize_multilingual(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include verses with positive scores
                verse = self.verses[idx].copy()
                verse['relevance_score'] = float(scores[idx])
                verse['match_type'] = 'lexical'
                results.append(verse)
        
        return results
    
    def format_result(self, verse: Dict) -> Dict:
        """Format verse result for display"""
        return {
            'verse_key': verse['verse_key'],
            'surah': verse['surah'],
            'surah_name_english': verse['surah_name_english'],
            'surah_name_arabic': verse['surah_name_arabic'],
            'ayah': verse['ayah'],
            'arabic': verse['arabic'],
            'translation_english': verse.get('translations_english', {}).get('sahih-international', 
                                           verse.get('translation_en_builtin', '')),
            'translation_urdu': verse.get('translations_urdu', {}).get('maulana-abu-al-maududi',
                                         verse.get('translation_ur_builtin', '')),
            'relevance_score': verse.get('relevance_score', 0.0),
            'match_type': verse.get('match_type', 'unknown'),
            'juz': verse.get('juz'),
        }
    
    def search_formatted(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search and return formatted results"""
        results = self.search(query, top_k)
        return [self.format_result(v) for v in results]


# Convenience function for quick searching
def quick_search(query: str, top_k: int = 5) -> List[Dict]:
    """Quick search using default engine"""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    
    engine = QuranSearchEngine(str(data_path))
    return engine.search_formatted(query, top_k)


# Test the search engine
if __name__ == "__main__":
    from pathlib import Path
    
    # Get data path
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "output" / "processed" / "quran_complete.json"
    
    # Initialize engine
    print("Initializing search engine...")
    engine = QuranSearchEngine(str(data_path))
    
    # Test queries
    test_queries = [
        "2:255",  # Structural - Ayat al-Kursi
        "surah fatiha",  # Structural - Surah name
        "patience",  # Semantic/keyword
        "bismillah",  # Keyword
        "صبر",  # Arabic keyword
        "what does quran say about prayer",  # Natural language
    ]
    
    print("\n" + "="*60)
    print("SEARCH ENGINE TESTS")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        results = engine.search_formatted(query, top_k=3)
        
        if not results:
            print("  No results found")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['surah_name_english']} ({result['surah']}:{result['ayah']})")
                print(f"   Score: {result['relevance_score']:.3f} | Type: {result['match_type']}")
                print(f"   Arabic: {result['arabic'][:80]}...")
                print(f"   English: {result['translation_english'][:80]}...")