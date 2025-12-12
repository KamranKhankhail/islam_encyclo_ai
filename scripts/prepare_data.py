"""
Islam Encyclo AI - Quran Data Preparation Pipeline
Processes Arabic Quran + 28 translations into unified search-ready format
"""

import json
import os
import glob
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected verse count (Hafs standard)
EXPECTED_VERSES = 6236


def load_json(filepath):
    """Load JSON file with UTF-8 encoding"""
    print(f"Loading: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_verse_key(verse_key):
    """Parse '1:1' into surah=1, ayah=1"""
    parts = verse_key.split(':')
    return int(parts[0]), int(parts[1])


def get_juz_for_verse(verse_key, juz_mappings):
    """Determine which Juz a verse belongs to"""
    surah, ayah = parse_verse_key(verse_key)
    
    for juz_info in juz_mappings:
        start_s, start_a = parse_verse_key(juz_info['start'])
        end_s, end_a = parse_verse_key(juz_info['end'])
        
        # Check if verse falls within this Juz range
        if (surah > start_s or (surah == start_s and ayah >= start_a)) and \
           (surah < end_s or (surah == end_s and ayah <= end_a)):
            return juz_info['juz']
    
    return None


def load_arabic_base():
    """Load the Arabic Quran base file"""
    print("\n" + "="*60)
    print("STEP 1: Loading Arabic Base Data")
    print("="*60)
    
    arabic_data = load_json(DATA_DIR / "quran_arabic.json")
    print(f"✓ Loaded {len(arabic_data)} verses")
    
    # Validate count
    if len(arabic_data) != EXPECTED_VERSES:
        print(f"⚠ WARNING: Expected {EXPECTED_VERSES} verses, got {len(arabic_data)}")
    
    return arabic_data


def load_translations():
    """Load all Urdu and English translation files"""
    print("\n" + "="*60)
    print("STEP 2: Loading Translation Files")
    print("="*60)
    
    translations = {
        'urdu': {},
        'english': {}
    }
    
    # Load Urdu translations
    urdu_dir = DATA_DIR / "quran_urdu_translations"
    urdu_files = glob.glob(str(urdu_dir / "*urdu-simple.json"))
    print(f"\nFound {len(urdu_files)} Urdu translation files:")
    
    for filepath in urdu_files:
        filename = os.path.basename(filepath)
        translator_name = filename.replace('-urdu-simple.json', '')
        
        data = load_json(filepath)
        translations['urdu'][translator_name] = data
        print(f"  ✓ {translator_name}: {len(data)} verses")
    
    # Load English translations
    english_dir = DATA_DIR / "quran_english_translations"
    english_files = glob.glob(str(english_dir / "*en-simple.json"))
    print(f"\nFound {len(english_files)} English translation files:")
    
    for filepath in english_files:
        filename = os.path.basename(filepath)
        translator_name = filename.replace('-en-simple.json', '')
        
        data = load_json(filepath)
        translations['english'][translator_name] = data
        print(f"  ✓ {translator_name}: {len(data)} verses")
    
    return translations


def load_metadata():
    """Load Surah names and Juz mappings"""
    print("\n" + "="*60)
    print("STEP 3: Loading Metadata")
    print("="*60)
    
    metadata = load_json(DATA_DIR / "metadata.json")
    
    # Create lookup dictionaries
    surah_lookup = {s['number']: s for s in metadata['surahs']}
    
    print(f"✓ Loaded metadata for {len(surah_lookup)} Surahs")
    print(f"✓ Loaded {len(metadata['juz_mappings'])} Juz mappings")
    
    return surah_lookup, metadata['juz_mappings']


def build_unified_dataset(arabic_data, translations, surah_lookup, juz_mappings):
    """Merge all data into unified format"""
    print("\n" + "="*60)
    print("STEP 4: Building Unified Dataset")
    print("="*60)
    
    unified = []
    errors = []
    
    for verse in arabic_data:
        verse_key = verse['verse_key']
        surah, ayah = parse_verse_key(verse_key)
        
        # Get surah metadata
        surah_info = surah_lookup.get(surah, {})
        
        # Get Juz
        juz = get_juz_for_verse(verse_key, juz_mappings)
        
        # Build unified verse entry
        unified_verse = {
            'id': verse_key,
            'verse_key': verse_key,
            'surah': surah,
            'ayah': ayah,
            
            # Surah metadata
            'surah_name_arabic': surah_info.get('name_arabic', ''),
            'surah_name_english': surah_info.get('name_english', ''),
            'surah_name_transliteration': surah_info.get('name_transliteration', ''),
            'revelation_type': surah_info.get('revelation_type', ''),
            
            # Arabic text
            'arabic': verse['text_uthmani'],
            'transliteration': verse.get('transliteration', ''),
            'transliteration_alt': verse.get('transliteration_alt', ''),
            
            # Built-in translations from Arabic file
            'translation_en_builtin': verse.get('translation_en', ''),
            'translation_ur_builtin': verse.get('translation_ur', ''),
            
            # Additional translations
            'translations_urdu': {},
            'translations_english': {},
            
            # Metadata
            'juz': juz,
            'hasanat': verse.get('hasanat', 0),
            
            # Search fields (will populate later)
            'searchable_text': ''
        }
        
        # Add all Urdu translations
        for translator_name, trans_data in translations['urdu'].items():
            if verse_key in trans_data:
                unified_verse['translations_urdu'][translator_name] = trans_data[verse_key]['t']
            else:
                errors.append(f"Missing Urdu translation: {translator_name} - {verse_key}")
        
        # Add all English translations
        for translator_name, trans_data in translations['english'].items():
            if verse_key in trans_data:
                unified_verse['translations_english'][translator_name] = trans_data[verse_key]['t']
            else:
                errors.append(f"Missing English translation: {translator_name} - {verse_key}")
        
        # Create searchable text (combine key fields for search)
        searchable_parts = [
            unified_verse['arabic'],
            unified_verse['transliteration'],
            unified_verse['translation_en_builtin'],
            unified_verse['translation_ur_builtin'],
            unified_verse['surah_name_english'],
            unified_verse['surah_name_arabic']
        ]
        
        # Add primary translations for search
        if unified_verse['translations_english'].get('sahih-international'):
            searchable_parts.append(unified_verse['translations_english']['sahih-international'])
        if unified_verse['translations_urdu'].get('maulana-abu-al-maududi'):
            searchable_parts.append(unified_verse['translations_urdu']['maulana-abu-al-maududi'])
        
        unified_verse['searchable_text'] = ' '.join(filter(None, searchable_parts))
        
        unified.append(unified_verse)
    
    print(f"✓ Processed {len(unified)} verses")
    
    if errors:
        print(f"\n⚠ {len(errors)} warnings (missing translations in some files)")
        # Show first 5 errors as sample
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more")
    
    return unified


def validate_dataset(dataset):
    """Run comprehensive validation checks"""
    print("\n" + "="*60)
    print("STEP 5: Validating Dataset")
    print("="*60)
    
    checks = []
    
    # Check 1: Total count
    expected = EXPECTED_VERSES
    actual = len(dataset)
    status = "✓" if actual == expected else "✗"
    checks.append(f"{status} Verse count: {actual} (expected {expected})")
    
    # Check 2: No duplicates
    verse_keys = [v['verse_key'] for v in dataset]
    duplicates = len(verse_keys) - len(set(verse_keys))
    status = "✓" if duplicates == 0 else "✗"
    checks.append(f"{status} Duplicates: {duplicates} (should be 0)")
    
    # Check 3: Sequential numbering
    missing_verses = []
    for surah in range(1, 115):
        surah_verses = [v for v in dataset if v['surah'] == surah]
        if surah_verses:
            ayah_numbers = sorted([v['ayah'] for v in surah_verses])
            expected_ayahs = list(range(1, len(ayah_numbers) + 1))
            if ayah_numbers != expected_ayahs:
                missing_verses.append(f"Surah {surah}")
    
    status = "✓" if len(missing_verses) == 0 else "✗"
    checks.append(f"{status} Sequential numbering: {len(missing_verses)} gaps")
    
    # Check 4: No null Arabic text
    null_arabic = sum(1 for v in dataset if not v['arabic'])
    status = "✓" if null_arabic == 0 else "✗"
    checks.append(f"{status} Null Arabic text: {null_arabic} (should be 0)")
    
    # Check 5: All verses have at least one translation
    no_translations = sum(1 for v in dataset if not v['translations_english'] and not v['translations_urdu'])
    status = "✓" if no_translations == 0 else "✗"
    checks.append(f"{status} Verses without translations: {no_translations}")
    
    # Check 6: Juz assignment
    null_juz = sum(1 for v in dataset if v['juz'] is None)
    status = "✓" if null_juz == 0 else "✗"
    checks.append(f"{status} Verses without Juz: {null_juz}")
    
    # Print all checks
    for check in checks:
        print(f"  {check}")
    
    # Overall result
    all_passed = all("✓" in check for check in checks)
    
    print("\n" + "-"*60)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("✗ SOME VALIDATION CHECKS FAILED - Review above")
    print("-"*60)
    
    return all_passed


def generate_sample_queries():
    """Generate test queries for evaluation"""
    print("\n" + "="*60)
    print("STEP 6: Generating Sample Test Queries")
    print("="*60)
    
    test_queries = [
        # Exact matches
        {"query": "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیْمِ", "expected": "1:1", "type": "exact_arabic"},
        {"query": "Alhamdu lillahi rabbi alAAalameen", "expected": "1:2", "type": "transliteration"},
        
        # Structural queries
        {"query": "surah 2 ayah 255", "expected": "2:255", "type": "structural"},
        {"query": "ayat ul kursi", "expected": "2:255", "type": "common_name"},
        
        # Semantic queries (English)
        {"query": "what does Quran say about patience", "expected": "2:153", "type": "semantic_en"},
        {"query": "verse about prayer", "expected": "2:45", "type": "semantic_en"},
        
        # Semantic queries (Urdu)
        {"query": "صبر کے بارے میں آیت", "expected": "2:153", "type": "semantic_ur"},
        
        # Partial matches
        {"query": "Allah is with those who are patient", "expected": "2:153", "type": "partial_en"},
        {"query": "اللہ صبر کرنے والوں کے ساتھ", "expected": "2:153", "type": "partial_ar"},
        
        # Translation search
        {"query": "In the name of Allah", "expected": "1:1", "type": "translation"},
        
        # Surah name search
        {"query": "surah fatiha", "expected": "1:1", "type": "surah_name"},
        {"query": "surah baqara", "expected": "2:1", "type": "surah_name"},
    ]
    
    output_path = OUTPUT_DIR / "test_queries.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Generated {len(test_queries)} test queries")
    print(f"  Saved to: {output_path}")
    
    return test_queries


def save_output(dataset):
    """Save processed data to output files"""
    print("\n" + "="*60)
    print("STEP 7: Saving Output Files")
    print("="*60)
    
    # Save complete unified dataset
    output_path = OUTPUT_DIR / "quran_complete.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved complete dataset: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Save compact version (for production)
    compact_path = OUTPUT_DIR / "quran_compact.json"
    with open(compact_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, separators=(',', ':'))
    print(f"✓ Saved compact version: {compact_path}")
    print(f"  Size: {os.path.getsize(compact_path) / (1024*1024):.2f} MB")
    
    # Save verse lookup (fast access by verse_key)
    lookup = {v['verse_key']: v for v in dataset}
    lookup_path = OUTPUT_DIR / "verse_lookup.json"
    with open(lookup_path, 'w', encoding='utf-8') as f:
        json.dump(lookup, f, ensure_ascii=False, separators=(',', ':'))
    print(f"✓ Saved verse lookup: {lookup_path}")
    
    # Generate statistics
    stats = {
        'total_verses': len(dataset),
        'total_surahs': len(set(v['surah'] for v in dataset)),
        'urdu_translators': len(dataset[0]['translations_urdu']),
        'english_translators': len(dataset[0]['translations_english']),
        'generated_at': __import__('datetime').datetime.now().isoformat()
    }
    
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved statistics: {stats_path}")


def main():
    """Main execution pipeline"""
    print("\n" + "█"*60)
    print("ISLAM ENCYCLO AI - DATA PREPARATION PIPELINE")
    print("█"*60)
    
    try:
        # Load all data
        arabic_data = load_arabic_base()
        translations = load_translations()
        surah_lookup, juz_mappings = load_metadata()
        
        # Build unified dataset
        dataset = build_unified_dataset(arabic_data, translations, surah_lookup, juz_mappings)
        
        # Validate
        validation_passed = validate_dataset(dataset)
        
        if not validation_passed:
            print("\n⚠ WARNING: Validation failed, but continuing to save output")
            print("  Please review the errors above before proceeding")
        
        # Generate test queries
        generate_sample_queries()
        
        # Save output
        save_output(dataset)
        
        # Final summary
        print("\n" + "█"*60)
        print("✓ DATA PREPARATION COMPLETE")
        print("█"*60)
        print(f"\nProcessed {len(dataset)} verses successfully")
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("  1. Run validation tests: pytest tests/test_data.py -v")
        print("  2. Inspect output files in output/processed/")
        print("  3. Proceed to Day 2: Building search engine")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)