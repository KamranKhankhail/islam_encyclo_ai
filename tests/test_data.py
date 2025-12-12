import json
import pytest


def test_verse_count():
    """Verify we have exactly 6236 verses"""
    with open('../output/processed/quran_complete.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert len(data) == 6236, f"Expected 6236 verses, got {len(data)}"


def test_no_nulls():
    """Ensure no missing data"""
    with open('../output/processed/quran_complete.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for verse in data:
        assert verse['arabic'] is not None
        assert verse['surah'] is not None
        assert verse['ayah'] is not None
