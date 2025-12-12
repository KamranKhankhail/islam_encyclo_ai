####### chat GPT normalizer
"""
Arabic Text Normalizer (Search-Oriented, Qur'an-Friendly)

Goals:
- Deterministic, high-performance normalization for Arabic search and indexing
- Removes diacritics/combining marks robustly (covers Qur'anic marks too)
- Normalizes common Arabic letter variants + common Urdu/Persian keyboard variants
- Configurable "aggressive" behavior without silently destroying meaning by default

Notes:
- Arabic has no case, but we casefold to normalize any embedded Latin text.
- For maximum matching recall, enable aggressive options; for better precision, keep them off.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional


# --- Fast regexes (compiled once) ---
_RE_MULTI_WS = re.compile(r"\s+")
_RE_ARABIC_TOKENS = re.compile(
    r"[0-9A-Za-z]+|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+",
    flags=re.UNICODE,
)

# Invisibility / directionality chars that can break matching
_INVISIBLE_CHARS = {
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi embeddings
    "\u2060",  # WORD JOINER
    "\ufeff",  # BOM
}

# Tatweel
_TATWEEL = "\u0640"

# Arabic-Indic digits + Extended Arabic-Indic digits
_DIGIT_MAP = str.maketrans({
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

# Common punctuation normalization (optional usage)
_PUNCT_TO_SPACE = str.maketrans({
    "،": " ", "؛": " ", "؟": " ",
    ",": " ", ";": " ", "?": " ",
    "“": " ", "”": " ", "\"": " ",
    "‘": " ", "’": " ", "'": " ",
    "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ",
    "ـ": " ",  # tatweel as space (though we also remove it)
})


@dataclass(frozen=True)
class ArabicNormalizationConfig:
    # Core cleanup
    unicode_nfkc: bool = True
    remove_invisibles: bool = True
    remove_tatweel: bool = True
    remove_combining_marks: bool = True  # removes tashkeel + Qur'anic combining marks

    # Search-focused canonicalization
    normalize_alef: bool = True          # أ إ آ ٱ -> ا
    normalize_yeh: bool = True           # ى ئ ی -> ي
    normalize_waw_hamza: bool = True     # ؤ -> و
    normalize_teh_marbuta_to_heh: bool = False  # ة -> ه (precision-loss; keep off by default)
    normalize_persian_urdu_letters: bool = True # ک -> ك, ہ/ھ -> ه, etc.
    drop_standalone_hamza: bool = False  # ء -> '' (precision-loss; keep off by default)

    # Utility
    normalize_digits_to_ascii: bool = True
    normalize_punctuation_to_spaces: bool = True
    normalize_whitespace: bool = True

    # Final shaping
    casefold_latin: bool = True  # affects only Latin; Arabic is unchanged


class ArabicNormalizer:
    """Normalize Arabic text for consistent searching and indexing."""

    def __init__(self, config: Optional[ArabicNormalizationConfig] = None):
        self.config = config or ArabicNormalizationConfig()

        # Build translation table once (fast path)
        trans = {}

        # Alef variants -> ا
        if self.config.normalize_alef:
            for ch in ("أ", "إ", "آ", "ٱ"):
                trans[ord(ch)] = "ا"

        # Yeh variants -> ي (IMPORTANT FIX)
        if self.config.normalize_yeh:
            # ى (alef maksura) -> ي ; ئ -> ي ; Persian Yeh (ی) -> ي
            for ch in ("ى", "ئ", "ی"):
                trans[ord(ch)] = "ي"

        # Waw with hamza -> و
        if self.config.normalize_waw_hamza:
            trans[ord("ؤ")] = "و"

        # Teh marbuta -> ه (optional, off by default due to precision loss)
        if self.config.normalize_teh_marbuta_to_heh:
            for ch in ("ة", "ۃ"):
                trans[ord(ch)] = "ه"

        # Persian/Urdu letter variants commonly typed by users
        if self.config.normalize_persian_urdu_letters:
            trans.update({
                ord("ک"): "ك",  # Keheh -> Kaf
                ord("گ"): "ك",  # Gaf (approx; optional, helps matching in mixed queries)
                ord("ہ"): "ه",  # Heh goal -> Heh
                ord("ھ"): "ه",  # Do chashmi heh -> Heh
                ord("ۀ"): "ه",  # Heh with yeh above -> Heh (common in Persian)
                ord("ۂ"): "ه",  # Heh goal with hamza above
                ord("ە"): "ه",  # Ae -> approximate to Heh for search
            })

        # Standalone hamza drop (optional)
        if self.config.drop_standalone_hamza:
            trans[ord("ء")] = ""

        # Punctuation -> space
        if self.config.normalize_punctuation_to_spaces:
            trans.update(_PUNCT_TO_SPACE)

        self._translate_table = trans

    def _maybe_nfkc(self, text: str) -> str:
        # NFKC collapses Arabic presentation forms (ligatures) into standard codepoints.
        return unicodedata.normalize("NFKC", text) if self.config.unicode_nfkc else text

    def _remove_invisibles(self, text: str) -> str:
        if not self.config.remove_invisibles:
            return text
        # Fast path: single pass filter
        return "".join(ch for ch in text if ch not in _INVISIBLE_CHARS)

    def _remove_tatweel(self, text: str) -> str:
        return text.replace(_TATWEEL, "") if self.config.remove_tatweel else text

    def _remove_combining_marks(self, text: str) -> str:
        if not self.config.remove_combining_marks:
            return text
        # Removes tashkeel AND Qur'anic combining marks robustly:
        # any char in categories Mn (Nonspacing Mark) or Me (Enclosing Mark)
        return "".join(
            ch for ch in text
            if unicodedata.category(ch) not in ("Mn", "Me")
        )

    def _normalize_digits(self, text: str) -> str:
        return text.translate(_DIGIT_MAP) if self.config.normalize_digits_to_ascii else text

    def _normalize_whitespace(self, text: str) -> str:
        if not self.config.normalize_whitespace:
            return text
        return _RE_MULTI_WS.sub(" ", text).strip()

    def normalize_full(self, text: Optional[str]) -> str:
        """Full normalization pipeline based on config."""
        if not text:
            return ""

        if not isinstance(text, str):
            text = str(text)

        text = self._maybe_nfkc(text)
        text = self._remove_invisibles(text)

        if self.config.remove_combining_marks:
            text = self._remove_combining_marks(text)

        if self.config.remove_tatweel:
            text = self._remove_tatweel(text)

        # Apply translation table (fast)
        if self._translate_table:
            text = text.translate(self._translate_table)

        if self.config.normalize_digits_to_ascii:
            text = self._normalize_digits(text)

        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Casefold for any embedded Latin text (Arabic unaffected)
        if self.config.casefold_latin:
            text = text.casefold()

        return text

    def normalize_for_search(self, text: Optional[str]) -> str:
        """
        Convenience wrapper: by default uses the instance config.
        For a more aggressive search mode, instantiate with a more aggressive config.
        """
        return self.normalize_full(text)

    def tokenize(self, text: Optional[str]) -> List[str]:
        """
        Tokenize Arabic + Latin alphanumerics reliably.
        Avoids Arabic word-boundary pitfalls of \\b.
        """
        normalized = self.normalize_for_search(text)
        return _RE_ARABIC_TOKENS.findall(normalized)


# --- Convenience: standard + aggressive presets ---
DEFAULT_NORMALIZER = ArabicNormalizer()

AGGRESSIVE_SEARCH_NORMALIZER = ArabicNormalizer(
    ArabicNormalizationConfig(
        # keep the safe defaults, but optionally turn on extra recall knobs:
        normalize_teh_marbuta_to_heh=True,
        drop_standalone_hamza=True,
    )
)


def normalize_arabic(text: Optional[str], aggressive: bool = False) -> str:
    """Normalize Arabic text (default: precision-friendly)."""
    if aggressive:
        return AGGRESSIVE_SEARCH_NORMALIZER.normalize_full(text)
    return DEFAULT_NORMALIZER.normalize_full(text)


def normalize_for_search(text: Optional[str]) -> str:
    """
    Normalize for search with a safer default (precision-friendly).
    If you want maximum recall, call normalize_arabic(text, aggressive=True).
    """
    return DEFAULT_NORMALIZER.normalize_for_search(text)


def tokenize_arabic(text: Optional[str], aggressive: bool = False) -> List[str]:
    """Tokenize Arabic text after normalization."""
    norm = AGGRESSIVE_SEARCH_NORMALIZER if aggressive else DEFAULT_NORMALIZER
    return norm.tokenize(text)


if __name__ == "__main__":
    test_cases = [
        "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیْمِ",
        "بسم الله الرحمن الرحيم",
        "أَلْحَمْدُ لِلّٰهِ",
        "الحمد لله",
        "علي vs على",
        "مساء vs مسا ء",   # hamza sensitivity example
        "کیا یہ درست ہے؟", # Urdu/Persian letters + punctuation
        "ﷲ",              # Allah ligature (presentation form)
    ]

    print("Arabic Normalization Tests")
    print("=" * 60)

    for t in test_cases:
        print("\nOriginal:   ", t)
        print("Default:    ", normalize_arabic(t, aggressive=False))
        print("Aggressive: ", normalize_arabic(t, aggressive=True))
        print("Tokens(def):", tokenize_arabic(t, aggressive=False))

####### claude ai normalizer
# """
# Arabic Text Normalizer
# Handles diacritics, hamza variations, and other Arabic-specific normalization
# """

# import re
# import pyarabic.araby as araby


# class ArabicNormalizer:
#     """Normalize Arabic text for consistent searching"""
    
#     def __init__(self):
#         # Arabic diacritics (tashkeel)
#         self.tashkeel = [
#             '\u064B',  # Tanween Fath
#             '\u064C',  # Tanween Damm
#             '\u064D',  # Tanween Kasr
#             '\u064E',  # Fatha
#             '\u064F',  # Damma
#             '\u0650',  # Kasra
#             '\u0651',  # Shadda
#             '\u0652',  # Sukun
#             '\u0653',  # Maddah
#             '\u0654',  # Hamza Above
#             '\u0655',  # Hamza Below
#             '\u0656',  # Subscript Alef
#             '\u0657',  # Inverted Damma
#             '\u0658',  # Mark Noon Ghunna
#             '\u0670',  # Superscript Alef
#         ]
        
#         # Hamza variations
#         self.hamza_variants = {
#             'أ': 'ا',  # Alef with Hamza Above
#             'إ': 'ا',  # Alef with Hamza Below
#             'آ': 'ا',  # Alef with Madda
#             'ٱ': 'ا',  # Alef Wasla
#             'ء': '',   # Hamza alone (can be removed in some contexts)
#         }
        
#         # Alef variations
#         self.alef_variants = {
#             'أ': 'ا',
#             'إ': 'ا',
#             'آ': 'ا',
#             'ٱ': 'ا',
#         }
        
#         # Yeh variations
#         self.yeh_variants = {
#             'ي': 'ى',  # Yeh
#             'ى': 'ى',  # Alef Maksura
#             'ئ': 'ى',  # Yeh with Hamza
#         }
        
#         # Teh Marbuta
#         self.teh_variants = {
#             'ة': 'ه',  # Teh Marbuta → Heh
#             'ۃ': 'ه',  # Urdu Teh Marbuta
#         }
    
#     def remove_diacritics(self, text):
#         """Remove all Arabic diacritics (tashkeel)"""
#         return araby.strip_tashkeel(text)
    
#     def normalize_hamza(self, text):
#         """Normalize all hamza variations to plain alef"""
#         for variant, normalized in self.hamza_variants.items():
#             text = text.replace(variant, normalized)
#         return text
    
#     def normalize_alef(self, text):
#         """Normalize all alef variations"""
#         for variant, normalized in self.alef_variants.items():
#             text = text.replace(variant, normalized)
#         return text
    
#     def normalize_yeh(self, text):
#         """Normalize yeh variations"""
#         for variant, normalized in self.yeh_variants.items():
#             text = text.replace(variant, normalized)
#         return text
    
#     def normalize_teh(self, text):
#         """Normalize teh marbuta"""
#         for variant, normalized in self.teh_variants.items():
#             text = text.replace(variant, normalized)
#         return text
    
#     def remove_tatweel(self, text):
#         """Remove tatweel (elongation character)"""
#         return text.replace('ـ', '')
    
#     def normalize_spaces(self, text):
#         """Normalize whitespace"""
#         # Replace multiple spaces with single space
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()
    
#     def normalize_full(self, text, aggressive=False):
#         """
#         Full normalization pipeline
        
#         Args:
#             text: Input Arabic text
#             aggressive: If True, normalize hamza, yeh, teh variations
        
#         Returns:
#             Normalized text
#         """
#         if not text:
#             return ""
        
#         # Remove diacritics (always)
#         text = self.remove_diacritics(text)
        
#         # Remove tatweel
#         text = self.remove_tatweel(text)
        
#         if aggressive:
#             # Normalize letter variations
#             text = self.normalize_hamza(text)
#             text = self.normalize_alef(text)
#             text = self.normalize_yeh(text)
#             text = self.normalize_teh(text)
        
#         # Normalize spaces
#         text = self.normalize_spaces(text)
        
#         return text
    
#     def normalize_for_search(self, text):
#         """
#         Normalize text specifically for search queries
#         More aggressive normalization for better matching
#         """
#         return self.normalize_full(text, aggressive=True).lower()
    
#     def tokenize(self, text):
#         """
#         Split Arabic text into tokens (words)
#         """
#         normalized = self.normalize_for_search(text)
#         # Split on whitespace and punctuation
#         tokens = re.findall(r'\b\w+\b', normalized)
#         return tokens


# # Convenience functions
# _normalizer = ArabicNormalizer()

# def normalize_arabic(text, aggressive=False):
#     """Normalize Arabic text"""
#     return _normalizer.normalize_full(text, aggressive)

# def normalize_for_search(text):
#     """Normalize for search (aggressive)"""
#     return _normalizer.normalize_for_search(text)

# def tokenize_arabic(text):
#     """Tokenize Arabic text"""
#     return _normalizer.tokenize(text)


# # Test normalization
# if __name__ == "__main__":
#     test_cases = [
#         "بِسْمِ اللّٰهِ الرَّحْمٰنِ الرَّحِیْمِ",  # With diacritics
#         "بسم الله الرحمن الرحيم",  # Without diacritics
#         "أَلْحَمْدُ لِلّٰهِ",  # With hamza
#         "الحمد لله",  # Simple
#     ]
    
#     normalizer = ArabicNormalizer()
    
#     print("Arabic Normalization Tests:")
#     print("="*60)
    
#     for text in test_cases:
#         print(f"\nOriginal: {text}")
#         print(f"No diacritics: {normalizer.remove_diacritics(text)}")
#         print(f"Full normal: {normalizer.normalize_full(text, aggressive=True)}")
#         print(f"For search: {normalizer.normalize_for_search(text)}")
#         print(f"Tokens: {normalizer.tokenize(text)}")