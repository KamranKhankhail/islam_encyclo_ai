"""
Islam Encyclo AI - Data Validation Tests
Ensures data integrity and quality
"""

import pytest
import json
import unicodedata
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "processed"
COMPLETE_DATA = OUTPUT_DIR / "quran_complete.json"
EXPECTED_VERSES = 6236


@pytest.fixture
def dataset():
    """Load the processed dataset"""
    with open(COMPLETE_DATA, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_file_exists():
    """Test that output file was created"""
    assert COMPLETE_DATA.exists(), "quran_complete.json not found"


def test_verse_count(dataset):
    """Verify we have exactly 6236 verses"""
    assert len(dataset) == EXPECTED_VERSES, \
        f"Expected {EXPECTED_VERSES} verses, got {len(dataset)}"


def test_no_duplicates(dataset):
    """Ensure no duplicate verse keys"""
    verse_keys = [v['verse_key'] for v in dataset]
    unique_keys = set(verse_keys)
    assert len(verse_keys) == len(unique_keys), \
        f"Found {len(verse_keys) - len(unique_keys)} duplicate verses"


def test_no_null_arabic(dataset):
    """All verses must have Arabic text"""
    null_count = sum(1 for v in dataset if not v.get('arabic'))
    assert null_count == 0, f"Found {null_count} verses without Arabic text"


def test_verse_structure(dataset):
    """Verify each verse has required fields"""
    required_fields = [
        'id', 'verse_key', 'surah', 'ayah', 
        'arabic', 'surah_name_english', 'searchable_text'
    ]
    
    for verse in dataset[:10]:  # Test first 10 verses
        for field in required_fields:
            assert field in verse, f"Verse {verse.get('verse_key')} missing field: {field}"
            assert verse[field] is not None, f"Verse {verse.get('verse_key')} has null {field}"


def test_arabic_encoding(dataset):
    """Verify Arabic text is properly encoded"""
    # Test first verse (Bismillah)
    first_verse = dataset[0]
    assert 'بِسْمِ' in first_verse['arabic'], \
        "Arabic text encoding error - Bismillah not found"
    
    # Should not contain mojibake
    for verse in dataset[:100]:
        assert '�' not in verse['arabic'], \
            f"Encoding error in verse {verse['verse_key']}"


def test_surah_numbering(dataset):
    """Verify surah numbers are valid (1-114)"""
    for verse in dataset:
        assert 1 <= verse['surah'] <= 114, \
            f"Invalid surah number: {verse['surah']}"


def test_ayah_numbering(dataset):
    """Verify ayah numbers start at 1 and are sequential"""
    for surah_num in range(1, 115):
        surah_verses = [v for v in dataset if v['surah'] == surah_num]
        
        if not surah_verses:
            pytest.fail(f"No verses found for Surah {surah_num}")
        
        ayah_numbers = sorted([v['ayah'] for v in surah_verses])
        expected = list(range(1, len(ayah_numbers) + 1))
        
        assert ayah_numbers == expected, \
            f"Surah {surah_num} has non-sequential ayahs: {ayah_numbers}"


def test_translations_present(dataset):
    """Verify translations exist"""
    for verse in dataset[:10]:  # Test first 10
        # Should have at least one translation
        has_translation = (
            len(verse.get('translations_english', {})) > 0 or 
            len(verse.get('translations_urdu', {})) > 0 or
            verse.get('translation_en_builtin') or
            verse.get('translation_ur_builtin')
        )
        assert has_translation, \
            f"Verse {verse['verse_key']} has no translations"


def test_juz_assignment(dataset):
    """Verify all verses have Juz assigned"""
    null_juz = [v for v in dataset if v.get('juz') is None]
    assert len(null_juz) == 0, \
        f"{len(null_juz)} verses missing Juz assignment"


def test_searchable_text(dataset):
    """Verify searchable text is populated"""
    empty_search = [v for v in dataset if not v.get('searchable_text')]
    assert len(empty_search) == 0, \
        f"{len(empty_search)} verses have empty searchable_text"


def normalize_arabic(text):
    """Normalize Arabic text for reliable comparison"""
    # Remove diacritics (combining marks)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Normalize letter variations (Urdu/Persian influenced)
    text = text.replace('ی', 'ي')  # Farsi Yeh → Arabic Yeh
    text = text.replace('ک', 'ك')  # Farsi Kaf → Arabic Kaf
    text = text.replace('ہ', 'ه')  # Urdu Heh → Arabic Heh
    text = text.replace('ے', 'ي')  # Urdu Yeh → Arabic Yeh
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def test_famous_verses(dataset):
    """Spot check famous verses for correctness"""
    verse_lookup = {v['verse_key']: v for v in dataset}
    
    # Test Ayat al-Kursi (2:255)
    ayat_kursi = verse_lookup.get('2:255')
    assert ayat_kursi is not None, "Ayat al-Kursi (2:255) not found"
    
    # Normalize for reliable checking
    text_clean = normalize_arabic(ayat_kursi['arabic'])
    
    # Check for core words (normalized)
    assert 'الله' in text_clean, f"Missing 'Allah' - got: {text_clean[:50]}"
    assert 'الحي' in text_clean, f"Missing 'Al-Hayy' - got: {text_clean[:50]}"
    assert 'القيوم' in text_clean, f"Missing 'Al-Qayyum' - got: {text_clean[:50]}"
    assert 'الكرسي' in text_clean or 'كرسي' in text_clean, f"Missing 'Kursi' - got: {text_clean[:100]}"
    assert len(ayat_kursi['arabic']) > 400, f"Ayat al-Kursi too short: {len(ayat_kursi['arabic'])} chars"
    
    # Test Al-Ikhlas (112:1)
    ikhlas = verse_lookup.get('112:1')
    assert ikhlas is not None, "Surah Al-Ikhlas (112:1) not found"
    
    text_clean = normalize_arabic(ikhlas['arabic'])
    assert 'الله' in text_clean, f"Missing 'Allah' - got: {text_clean}"
    assert 'احد' in text_clean or 'أحد' in text_clean, f"Missing 'Ahad' - got: {text_clean}"



def test_surah_al_fatiha(dataset):
    """Detailed test of Surah Al-Fatiha (first 7 verses)"""
    fatiha = [v for v in dataset if v['surah'] == 1]
    
    assert len(fatiha) == 7, f"Al-Fatiha should have 7 verses, got {len(fatiha)}"
    
    # Verify they're in order
    for i, verse in enumerate(sorted(fatiha, key=lambda x: x['ayah']), 1):
        assert verse['ayah'] == i, f"Al-Fatiha ayah {i} out of order"


def test_longest_surah(dataset):
    """Verify Al-Baqarah (longest surah) has 286 verses"""
    baqarah = [v for v in dataset if v['surah'] == 2]
    assert len(baqarah) == 286, \
        f"Al-Baqarah should have 286 verses, got {len(baqarah)}"


def test_shortest_surah(dataset):
    """Verify Al-Kawthar (shortest surah) has 3 verses"""
    kawthar = [v for v in dataset if v['surah'] == 108]
    assert len(kawthar) == 3, \
        f"Al-Kawthar should have 3 verses, got {len(kawthar)}"


# Run this file with: pytest tests/test_data.py -v