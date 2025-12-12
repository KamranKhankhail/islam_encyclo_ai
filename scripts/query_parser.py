############ chatgpt query parser
"""
Structural Query Parser (Production-Grade)
Parses queries like:
- "2:255", "2-255", "2/255", "٢:٢٥٥"
- "surah 2 ayah 255"
- "surah baqarah ayah 255"
- "verse 2:255"
- nicknames like "ayatul kursi"
Optionally validates ayah ranges per-surah via metadata_path.

Design goals:
- High precision (avoid false positives in long semantic queries)
- High recall for explicit structural intent
- Robust digit normalization (Arabic-Indic + Extended Arabic-Indic)
- Alias matching via token n-grams (NOT substring matching)
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional, Dict, List, Tuple, Any


class StructuralQueryParser:
    """Parse structural queries into verse references."""

    # Arabic-Indic + Extended Arabic-Indic digits → ASCII digits
    _DIGIT_TRANS = str.maketrans({
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    })

    # Explicit structural intent tokens (English + common Arabic/Urdu)
    _STRUCTURE_TOKENS = {
        "surah", "sura", "surat", "surah:", "sura:", "chapter", "chap", "ch",
        "ayah", "aya", "aayah", "ayat", "aayat", "verse", "verses",
        "سورة", "سورۃ", "سورت", "سورہ",
        "آية", "اية", "آیت", "آیہ",
    }

    # Nickname normalization helper
    _NICKNAME_NORM_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)

    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize parser with surah name mappings and optional per-surah ayah counts.

        metadata_path (optional):
            JSON file containing either:
            - {"surah_ayah_counts": {"1": 7, "2": 286, ...}}
            - {"surah_ayah_counts": [7, 286, ...]}  # index 0 = surah 1
            - {"1": 7, "2": 286, ...}               # direct mapping
        """

        # Surah name variations (English transliterations + common variants)
        # NOTE: English-meaning aliases are separated into "loose" aliases to avoid false positives.
        strict_aliases: Dict[str, int] = {
            "fatiha": 1, "fatihah": 1, "al fatiha": 1, "al fatihah": 1,

            "baqara": 2, "baqarah": 2, "al baqara": 2, "al baqarah": 2, "bakra": 2,

            "imran": 3, "ali imran": 3, "aal e imran": 3, "ale imran": 3,

            "nisa": 4, "an nisa": 4, "nisaa": 4,

            "maidah": 5, "al maidah": 5, "maida": 5,

            "anam": 6, "al anam": 6,

            "araf": 7, "al araf": 7, "aaraf": 7,

            "anfal": 8, "al anfal": 8,

            "tawba": 9, "at tawba": 9, "taubah": 9, "tauba": 9, "toba": 9,

            "yunus": 10, "yoonus": 10,
            "hud": 11,
            "yusuf": 12, "yousuf": 12,
            "rad": 13, "ar rad": 13, "raad": 13,
            "ibrahim": 14, "ibraheem": 14,
            "hijr": 15, "al hijr": 15,
            "nahl": 16, "an nahl": 16,
            "isra": 17, "al isra": 17, "israa": 17, "bani israel": 17,
            "kahf": 18, "al kahf": 18,
            "maryam": 19, "mariam": 19,
            "taha": 20, "ta ha": 20, "taa ha": 20,
            "anbiya": 21, "al anbiya": 21, "anbiyaa": 21,
            "hajj": 22, "al hajj": 22, "haj": 22,
            "muminun": 23, "al muminun": 23, "muminoon": 23,
            "nur": 24, "an nur": 24, "noor": 24,
            "furqan": 25, "al furqan": 25, "furqaan": 25,
            "shuara": 26, "ash shuara": 26, "shu ara": 26,
            "naml": 27, "an naml": 27,
            "qasas": 28, "al qasas": 28,
            "ankabut": 29, "al ankabut": 29, "ankaboot": 29,
            "rum": 30, "ar rum": 30, "room": 30,
            "luqman": 31, "lukman": 31, "lokman": 31,
            "sajda": 32, "as sajda": 32, "sajdah": 32,
            "ahzab": 33, "al ahzab": 33, "ahzaab": 33,
            "saba": 34,
            "fatir": 35, "faatir": 35,
            "yasin": 36, "ya sin": 36, "yaseen": 36,
            "saffat": 37, "as saffat": 37, "saaffaat": 37,
            "sad": 38, "saad": 38,
            "zumar": 39, "az zumar": 39,
            "ghafir": 40, "ghaafir": 40, "mumin": 40,
            "fussilat": 41, "ha mim": 41,
            "shura": 42, "ash shura": 42, "shoora": 42,
            "zukhruf": 43, "az zukhruf": 43,
            "dukhan": 44, "ad dukhan": 44,
            "jathiya": 45, "al jathiya": 45, "jaathiya": 45,
            "ahqaf": 46, "al ahqaf": 46,
            "muhammad": 47,
            "fath": 48, "al fath": 48,
            "hujurat": 49, "al hujurat": 49, "hujraat": 49,
            "qaf": 50, "qaaf": 50,
            "dhariyat": 51, "adh dhariyat": 51, "dhaariyat": 51,
            "tur": 52, "at tur": 52, "toor": 52,
            "najm": 53, "an najm": 53,
            "qamar": 54, "al qamar": 54,
            "rahman": 55, "ar rahman": 55, "rahmaan": 55,
            "waqia": 56, "al waqia": 56, "waaqi ah": 56,
            "hadid": 57, "al hadid": 57,
            "mujadila": 58, "al mujadila": 58, "mujaadila": 58,
            "hashr": 59, "al hashr": 59,
            "mumtahana": 60, "al mumtahana": 60, "mumtahina": 60,
            "saff": 61, "as saff": 61, "saf": 61,
            "jumua": 62, "al jumua": 62, "jumu ah": 62,
            "munafiqun": 63, "al munafiqun": 63, "munaafiqoon": 63,
            "taghabun": 64, "at taghabun": 64, "taghaabun": 64,
            "talaq": 65, "at talaq": 65, "talaaq": 65,
            "tahrim": 66, "at tahrim": 66, "tahreem": 66,
            "mulk": 67, "al mulk": 67,
            "qalam": 68, "al qalam": 68,
            "haqqa": 69, "al haqqa": 69, "haaqqa": 69,
            "maarij": 70, "al maarij": 70, "ma arij": 70,
            "nuh": 71, "nooh": 71,
            "jinn": 72, "al jinn": 72,
            "muzzammil": 73, "al muzzammil": 73,
            "muddaththir": 74, "al muddaththir": 74, "muddassir": 74,
            "qiyama": 75, "al qiyama": 75, "qiyaama": 75, "qiyamah": 75,
            "insan": 76, "al insan": 76, "insaan": 76, "dahr": 76,
            "mursalat": 77, "al mursalat": 77, "mursalaat": 77,
            "naba": 78, "an naba": 78, "nabaa": 78,
            "naziat": 79, "an naziat": 79, "naazi aat": 79,
            "abasa": 80,
            "takwir": 81, "at takwir": 81,
            "infitar": 82, "al infitar": 82, "infitaar": 82,
            "mutaffifin": 83, "al mutaffifin": 83, "mutaffifeen": 83,
            "inshiqaq": 84, "al inshiqaq": 84, "inshiqaaq": 84,
            "buruj": 85, "al buruj": 85, "burooj": 85,
            "tariq": 86, "at tariq": 86, "taariq": 86,
            "ala": 87, "al ala": 87, "a la": 87, "alaa": 87,
            "ghashiya": 88, "al ghashiya": 88, "ghaashiya": 88,
            "fajr": 89, "al fajr": 89,
            "balad": 90, "al balad": 90,
            "shams": 91, "ash shams": 91,
            "layl": 92, "al layl": 92, "lail": 92,
            "duha": 93, "ad duha": 93, "dhuhaa": 93,
            "sharh": 94, "ash sharh": 94, "inshirah": 94,
            "tin": 95, "at tin": 95, "teen": 95,
            "alaq": 96, "al alaq": 96, "iqra": 96,
            "qadr": 97, "al qadr": 97,
            "bayyina": 98, "al bayyina": 98, "bayyinah": 98,
            "zalzala": 99, "az zalzala": 99, "zilzaal": 99,
            "adiyat": 100, "al adiyat": 100, "aadiyaat": 100,
            "qaria": 101, "al qaria": 101, "qaari ah": 101,
            "takathur": 102, "at takathur": 102, "takaathur": 102,
            "asr": 103, "al asr": 103,
            "humaza": 104, "al humaza": 104,
            "fil": 105, "al fil": 105,
            "quraysh": 106, "quraish": 106, "qureish": 106,
            "maun": 107, "al maun": 107, "maa un": 107,
            "kawthar": 108, "al kawthar": 108, "kauthar": 108, "kausar": 108,
            "kafirun": 109, "al kafirun": 109, "kaafiroon": 109, "kafiroon": 109,
            "nasr": 110, "an nasr": 110,
            "masad": 111, "al masad": 111, "lahab": 111,
            "ikhlas": 112, "al ikhlas": 112, "ikhlaas": 112,
            "falaq": 113, "al falaq": 113,
            "nas": 114, "an nas": 114, "naas": 114,
        }

        # English-meaning aliases (high false-positive risk) — only match when intent is explicit.
        loose_aliases: Dict[str, int] = {
            "jonas": 10,
            "joseph": 12,
            "abraham": 14,
            "cave": 18,
            "prophets": 21,
            "believers": 23,
            "light": 24,
            "stories": 28,
            "spider": 29,
            "romans": 30,
            "star": 53,
            "moon": 54,
            "iron": 57,
            "friday": 62,
            "divorce": 65,
            "dominion": 67,
            "pen": 68,
            "smoke": 44,
            "fig": 95,
            "elephant": 105,
            "dawn": 113,
            "mankind": 114,
            "time": 103,
            "sun": 91,
            "night": 92,
            "power": 97,
            "victory": 48,
            "help": 110,
            "sincerity": 112,
        }

        # Common verse nicknames
        self.verse_nicknames: Dict[str, str] = {
            "ayat al kursi": "2:255",
            "ayatul kursi": "2:255",
            "ayat kursi": "2:255",
            "ayat ul kursi": "2:255",
            "ayatul-kursi": "2:255",
            "ayat-ul-kursi": "2:255",
            "throne verse": "2:255",
            "kursi": "2:255",
        }

        # Build token-ngram index for aliases (precision-safe)
        self._strict_alias_index, self._strict_max_len = self._build_alias_index(strict_aliases)
        self._loose_alias_index, self._loose_max_len = self._build_alias_index(loose_aliases)

        # Optional per-surah ayah counts
        self._ayah_counts = self._load_ayah_counts(metadata_path)

        # Compile patterns (fast path)
        # Accept a wide set of separators, including Arabic comma and fullwidth colon.
        sep = r"[:;،,\-–—/\.：]"
        # Range connector: "-", "to", "—", "–"
        rng = r"(?:\-|–|—|\bto\b)"

        self._patterns: List[Tuple[re.Pattern, str]] = [
            # "2:255" / "2-255" / "2/255" / "2.255" / "2：255"
            (re.compile(rf"^\s*(\d{{1,3}})\s*{sep}\s*(\d{{1,3}})\s*$", re.IGNORECASE), "ref_pair"),

            # "2:255-257" (range within same surah)
            (re.compile(rf"^\s*(\d{{1,3}})\s*{sep}\s*(\d{{1,3}})\s*{rng}\s*(\d{{1,3}})\s*$", re.IGNORECASE), "ref_range_same_surah"),

            # "surah 2 ayah 255" (supports Arabic/Urdu tokens too, after normalization)
            (re.compile(r"(?:(?:surah|sura|surat|chapter|ch)\s*)?(\d{1,3})\s+(?:ayah|aya|aayah|ayat|aayat|verse)\s*(?:[:\-]?\s*)?(\d{1,3})", re.IGNORECASE), "surah_ayah_words"),

            # "verse 2:255"
            (re.compile(rf"(?:verse|ayah|aya|aayah|ayat|aayat)\s+(\d{{1,3}})\s*{sep}\s*(\d{{1,3}})", re.IGNORECASE), "verse_ref_words"),

            # "surah 2 verse 255"
            (re.compile(r"(?:surah|sura|surat|chapter|ch)\s+(\d{1,3})\s+(?:verse|ayah|aya|aayah|ayat|aayat)\s+(\d{1,3})", re.IGNORECASE), "chapter_verse_words"),

            # "2 255" (two numbers)
            (re.compile(r"^\s*(\d{1,3})\s+(\d{1,3})\s*$", re.IGNORECASE), "ref_space_pair"),

            # Single number: "2" (treat as surah only when query is exactly one token)
            (re.compile(r"^\s*(\d{1,3})\s*$", re.IGNORECASE), "surah_only_number"),
        ]

    # -----------------------------
    # Public API
    # -----------------------------

    def parse(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Parse a query and extract structural information.

        Returns dict:
            For single verse:
                {'type','surah','ayah','verse_key','confidence'}
            For range:
                includes 'ayah_end' and 'verse_key_end'
        Returns None if not confidently structural.
        """
        if not query or not query.strip():
            return None

        raw = query
        q_num = self._normalize_numbers_nfkc(raw)
        q_norm = self._normalize_for_matching(raw)  # token-safe form

        # 1) Verse nicknames (high confidence)
        nick = self._match_nickname(q_norm)
        if nick:
            surah, ayah = nick
            return {
                "type": "nickname",
                "surah": surah,
                "ayah": ayah,
                "verse_key": f"{surah}:{ayah}",
                "confidence": 1.0,
            }

        # 2) Explicit numeric / word patterns (high confidence)
        pat = self._match_patterns(q_num)
        if pat:
            return pat

        # 3) Surah name / alias parsing (medium confidence; precision-protected)
        alias_res = self._match_surah_alias(q_norm)
        if alias_res:
            surah, confidence = alias_res

            # Try to get ayah number if specified
            ayah = self._extract_ayah_number(q_num)
            if ayah is None:
                # If query looks like "surah baqarah 255" (number present but no keyword),
                # accept a trailing number as ayah when the query is short.
                maybe_ayah = self._extract_trailing_number_as_ayah(q_norm)
                ayah = maybe_ayah

            if ayah is None:
                # Surah-only intent
                return {
                    "type": "surah_name",
                    "surah": surah,
                    "ayah": 1,
                    "verse_key": f"{surah}:1",
                    "confidence": min(confidence, 0.75),
                }

            if not self._validate_ref(surah, ayah):
                return None

            return {
                "type": "surah_name",
                "surah": surah,
                "ayah": ayah,
                "verse_key": f"{surah}:{ayah}",
                "confidence": min(0.9, confidence + 0.1),
            }

        return None

    def is_structural_query(self, query: str) -> bool:
        return self.parse(query) is not None

    # -----------------------------
    # Normalization
    # -----------------------------

    def _normalize_numbers_nfkc(self, text: str) -> str:
        # Keep punctuation for regex patterns, but normalize digits and unicode forms.
        t = unicodedata.normalize("NFKC", text)
        t = t.translate(self._DIGIT_TRANS)
        return t.strip().casefold()

    def _normalize_for_matching(self, text: str) -> str:
        # Token-safe: punctuation → spaces, collapse whitespace, digits normalized.
        t = self._normalize_numbers_nfkc(text)

        # Replace most punctuation with spaces (keep only tokens)
        t = re.sub(r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", " ", t, flags=re.UNICODE)
        t = re.sub(r"\s+", " ", t).strip()

        # Normalize hyphens/underscores to spaces
        t = t.replace("_", " ").replace("-", " ")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # -----------------------------
    # Nicknames
    # -----------------------------

    def _match_nickname(self, q_norm: str) -> Optional[Tuple[int, int]]:
        # Normalize to stable nickname form (latin-only punctuation stripped)
        q_simple = self._NICKNAME_NORM_RE.sub(" ", q_norm)
        q_simple = re.sub(r"\s+", " ", q_simple).strip()

        for nickname, verse_ref in self.verse_nicknames.items():
            n = self._NICKNAME_NORM_RE.sub(" ", nickname.casefold())
            n = re.sub(r"\s+", " ", n).strip()
            if n and (n == q_simple or f" {n} " in f" {q_simple} "):
                s, a = verse_ref.split(":")
                return int(s), int(a)
        return None

    # -----------------------------
    # Patterns
    # -----------------------------

    def _match_patterns(self, q_num: str) -> Optional[Dict[str, Any]]:
        for rx, ptype in self._patterns:
            m = rx.search(q_num)
            if not m:
                continue

            if ptype == "ref_range_same_surah":
                surah = int(m.group(1))
                a1 = int(m.group(2))
                a2 = int(m.group(3))
                if a2 < a1:
                    a1, a2 = a2, a1
                if not self._validate_ref(surah, a1) or not self._validate_ref(surah, a2):
                    continue
                return {
                    "type": ptype,
                    "surah": surah,
                    "ayah": a1,
                    "ayah_end": a2,
                    "verse_key": f"{surah}:{a1}",
                    "verse_key_end": f"{surah}:{a2}",
                    "confidence": 1.0,
                }

            if ptype == "surah_only_number":
                surah = int(m.group(1))
                if 1 <= surah <= 114:
                    # Only accept if query is exactly that number (high precision)
                    if q_num.strip() == str(surah):
                        return {
                            "type": ptype,
                            "surah": surah,
                            "ayah": 1,
                            "verse_key": f"{surah}:1",
                            "confidence": 0.85,
                        }
                continue

            # Standard pair patterns
            surah = int(m.group(1))
            ayah = int(m.group(2))
            if not self._validate_ref(surah, ayah):
                continue
            return {
                "type": ptype,
                "surah": surah,
                "ayah": ayah,
                "verse_key": f"{surah}:{ayah}",
                "confidence": 1.0,
            }

        return None

    # -----------------------------
    # Alias matching (token n-grams)
    # -----------------------------

    def _build_alias_index(self, aliases: Dict[str, int]) -> Tuple[Dict[Tuple[str, ...], int], int]:
        idx: Dict[Tuple[str, ...], int] = {}
        max_len = 1
        for k, v in aliases.items():
            key_norm = self._normalize_for_matching(k)
            toks = tuple(key_norm.split())
            if not toks:
                continue
            idx[toks] = v
            max_len = max(max_len, len(toks))
        return idx, max_len

    def _match_surah_alias(self, q_norm: str) -> Optional[Tuple[int, float]]:
        toks = q_norm.split()
        if not toks:
            return None

        has_struct_intent = any(t in self._STRUCTURE_TOKENS for t in toks)
        token_len = len(toks)

        # Helper: scan for longest match first
        def scan(index: Dict[Tuple[str, ...], int], max_len: int) -> Optional[int]:
            for i in range(len(toks)):
                for L in range(min(max_len, len(toks) - i), 0, -1):
                    seg = tuple(toks[i:i+L])
                    if seg in index:
                        return index[seg]
            return None

        # Strict aliases: allow in broader contexts (still token-based, so safe)
        s = scan(self._strict_alias_index, self._strict_max_len)
        if s is not None:
            # Confidence depends on explicit intent and query length
            if has_struct_intent:
                return s, 0.9
            if token_len <= 2:
                return s, 0.8
            # If longer query without intent, keep conservative to avoid false positives
            return s, 0.65

        # Loose aliases (English meaning): ONLY if intent is explicit or query is very short
        l = scan(self._loose_alias_index, self._loose_max_len)
        if l is not None:
            if has_struct_intent:
                return l, 0.75
            if token_len == 1:
                return l, 0.6
            return None

        return None

    # -----------------------------
    # Ayah extraction + validation
    # -----------------------------

    def _extract_ayah_number(self, q_num: str) -> Optional[int]:
        # Explicit forms: "ayah 255", "verse:255", "ayat-255"
        m = re.search(r"(?:ayah|aya|aayah|ayat|aayat|verse)\s*[:\-]?\s*(\d{1,3})", q_num, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def _extract_trailing_number_as_ayah(self, q_norm: str) -> Optional[int]:
        # If query is short and ends with a number, treat it as ayah (e.g., "baqarah 255").
        toks = q_norm.split()
        if len(toks) <= 3 and toks and toks[-1].isdigit():
            n = int(toks[-1])
            if 1 <= n <= 286:
                return n
        return None

    def _validate_ref(self, surah: int, ayah: int) -> bool:
        if not (1 <= surah <= 114):
            return False
        if not (1 <= ayah <= 286):
            return False

        # Per-surah validation if available
        if self._ayah_counts:
            max_ayah = self._ayah_counts.get(surah)
            if isinstance(max_ayah, int) and max_ayah > 0:
                return ayah <= max_ayah

        return True

    def _load_ayah_counts(self, metadata_path: Optional[str]) -> Optional[Dict[int, int]]:
        if not metadata_path:
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 1) Your current metadata format: {"surahs":[{number, total_verses, ...}, ...], ...}
            if isinstance(data, dict) and isinstance(data.get("surahs"), list):
                counts: Dict[int, int] = {}
                for s in data["surahs"]:
                    if not isinstance(s, dict):
                        continue
                    n = s.get("number")
                    tv = s.get("total_verses")
                    if isinstance(n, int) and isinstance(tv, int) and 1 <= n <= 114 and tv > 0:
                        counts[n] = tv
                return counts or None

            # 2) Back-compat: {"surah_ayah_counts": {...}} or {"surah_ayah_counts":[...]}
            if isinstance(data, dict) and "surah_ayah_counts" in data:
                data = data["surah_ayah_counts"]

            # 3) Back-compat list: [7,286,...] index 0 => surah 1
            if isinstance(data, list):
                counts = {i: v for i, v in enumerate(data, start=1) if isinstance(v, int) and v > 0}
                return counts or None

            # 4) Back-compat mapping: {"1":7,"2":286,...} or {1:7,2:286,...}
            if isinstance(data, dict):
                counts = {}
                for k, v in data.items():
                    try:
                        si = int(k)
                    except Exception:
                        continue
                    if isinstance(v, int) and v > 0:
                        counts[si] = v
                return counts or None

        except Exception:
            return None

        return None



# Convenience functions
_parser = StructuralQueryParser()

def parse_structural_query(query: str) -> Optional[Dict[str, Any]]:
    return _parser.parse(query)

def is_structural(query: str) -> bool:
    return _parser.is_structural_query(query)


if __name__ == "__main__":
    test_queries = [
        "2:255",
        "٢:٢٥٥",
        "2-255",
        "2/255",
        "verse 2:255",
        "surah 2 ayah 255",
        "surah baqarah ayah 255",
        "surah baqarah 255",
        "ayat ul kursi",
        "kursi",
        "surah ikhlas",
        "112:1",
        "2:255-257",
        "light",  # loose alias (only matches if short; returns low confidence)
        "what does quran say about light",  # should NOT become structural
        "how to use pen in school",         # should NOT become structural
        "surah light",                      # should become structural (explicit intent)
    ]

    parser = StructuralQueryParser()

    print("Structural Query Parser Tests:")
    print("=" * 70)

    for q in test_queries:
        r = parser.parse(q)
        print(f"\nQuery: {q}")
        if r:
            print(f"  ✓ STRUCTURAL: {r}")
        else:
            print("  ✗ NOT structural (needs semantic search)")

############ claude query parser
# """
# Structural Query Parser
# Handles queries like "surah 2 ayah 255", "verse 2:255", etc.
# """

# import re
# from typing import Optional, Dict, List


# class StructuralQueryParser:
#     """Parse structural queries into verse references"""
    
#     def __init__(self, metadata_path=None):
#         """
#         Initialize parser with surah name mappings
#         """
#         # Surah name variations (English)
#         self.surah_names = {
#             'fatiha': 1, 'fatihah': 1, 'al-fatiha': 1, 'al-fatihah': 1,
#             'baqara': 2, 'baqarah': 2, 'al-baqara': 2, 'al-baqarah': 2, 'bakra': 2,
#             'imran': 3, 'ali imran': 3, 'aal-e-imran': 3, 'ale-imran': 3,
#             'nisa': 4, 'an-nisa': 4, 'nisaa': 4,
#             'maidah': 5, 'al-maidah': 5, 'maida': 5,
#             'anam': 6, 'al-anam': 6,
#             'araf': 7, 'al-araf': 7, 'aaraf': 7,
#             'anfal': 8, 'al-anfal': 8,
#             'tawba': 9, 'at-tawba': 9, 'taubah': 9, 'tauba': 9, 'toba': 9,
#             'yunus': 10, 'yoonus': 10, 'jonas': 10,
#             'hud': 11,
#             'yusuf': 12, 'yousuf': 12, 'joseph': 12,
#             'rad': 13, 'ar-rad': 13, 'raad': 13,
#             'ibrahim': 14, 'ibraheem': 14, 'abraham': 14,
#             'hijr': 15, 'al-hijr': 15,
#             'nahl': 16, 'an-nahl': 16,
#             'isra': 17, 'al-isra': 17, 'israa': 17, 'bani israel': 17,
#             'kahf': 18, 'al-kahf': 18, 'cave': 18,
#             'maryam': 19, 'mariam': 19, 'mary': 19,
#             'taha': 20, 'ta-ha': 20, 'taa-ha': 20,
#             'anbiya': 21, 'al-anbiya': 21, 'anbiyaa': 21, 'prophets': 21,
#             'hajj': 22, 'al-hajj': 22, 'haj': 22,
#             'muminun': 23, 'al-muminun': 23, 'muminoon': 23, 'believers': 23,
#             'nur': 24, 'an-nur': 24, 'noor': 24, 'light': 24,
#             'furqan': 25, 'al-furqan': 25, 'furqaan': 25,
#             'shuara': 26, 'ash-shuara': 26, 'shu\'ara': 26,
#             'naml': 27, 'an-naml': 27, 'ant': 27,
#             'qasas': 28, 'al-qasas': 28, 'stories': 28,
#             'ankabut': 29, 'al-ankabut': 29, 'ankaboot': 29, 'spider': 29,
#             'rum': 30, 'ar-rum': 30, 'room': 30, 'romans': 30,
#             'luqman': 31, 'lukman': 31, 'lokman': 31,
#             'sajda': 32, 'as-sajda': 32, 'sajdah': 32,
#             'ahzab': 33, 'al-ahzab': 33, 'ahzaab': 33,
#             'saba': 34, 'sheba': 34,
#             'fatir': 35, 'faatir': 35,
#             'yasin': 36, 'ya-sin': 36, 'yaseen': 36,
#             'saffat': 37, 'as-saffat': 37, 'saaffaat': 37,
#             'sad': 38, 'saad': 38,
#             'zumar': 39, 'az-zumar': 39,
#             'ghafir': 40, 'ghaafir': 40, 'mumin': 40,
#             'fussilat': 41, 'ha mim': 41,
#             'shura': 42, 'ash-shura': 42, 'shoora': 42,
#             'zukhruf': 43, 'az-zukhruf': 43,
#             'dukhan': 44, 'ad-dukhan': 44, 'smoke': 44,
#             'jathiya': 45, 'al-jathiya': 45, 'jaathiya': 45,
#             'ahqaf': 46, 'al-ahqaf': 46,
#             'muhammad': 47,
#             'fath': 48, 'al-fath': 48, 'fat-h': 48, 'victory': 48,
#             'hujurat': 49, 'al-hujurat': 49, 'hujraat': 49,
#             'qaf': 50, 'qaaf': 50,
#             'dhariyat': 51, 'adh-dhariyat': 51, 'dhaariyat': 51,
#             'tur': 52, 'at-tur': 52, 'toor': 52,
#             'najm': 53, 'an-najm': 53, 'star': 53,
#             'qamar': 54, 'al-qamar': 54, 'moon': 54,
#             'rahman': 55, 'ar-rahman': 55, 'rahmaan': 55,
#             'waqia': 56, 'al-waqia': 56, 'waaqi\'ah': 56,
#             'hadid': 57, 'al-hadid': 57, 'iron': 57,
#             'mujadila': 58, 'al-mujadila': 58, 'mujaadila': 58,
#             'hashr': 59, 'al-hashr': 59,
#             'mumtahana': 60, 'al-mumtahana': 60, 'mumtahina': 60,
#             'saff': 61, 'as-saff': 61, 'saf': 61,
#             'jumua': 62, 'al-jumua': 62, 'jumu\'ah': 62, 'friday': 62,
#             'munafiqun': 63, 'al-munafiqun': 63, 'munaafiqoon': 63,
#             'taghabun': 64, 'at-taghabun': 64, 'taghaabun': 64,
#             'talaq': 65, 'at-talaq': 65, 'talaaq': 65, 'divorce': 65,
#             'tahrim': 66, 'at-tahrim': 66, 'tahreem': 66,
#             'mulk': 67, 'al-mulk': 67, 'dominion': 67,
#             'qalam': 68, 'al-qalam': 68, 'pen': 68,
#             'haqqa': 69, 'al-haqqa': 69, 'haaqqa': 69,
#             'maarij': 70, 'al-maarij': 70, 'ma\'arij': 70,
#             'nuh': 71, 'nooh': 71, 'noah': 71,
#             'jinn': 72, 'al-jinn': 72,
#             'muzzammil': 73, 'al-muzzammil': 73,
#             'muddaththir': 74, 'al-muddaththir': 74, 'muddassir': 74,
#             'qiyama': 75, 'al-qiyama': 75, 'qiyaama': 75, 'qiyamah': 75,
#             'insan': 76, 'al-insan': 76, 'insaan': 76, 'dahr': 76,
#             'mursalat': 77, 'al-mursalat': 77, 'mursalaat': 77,
#             'naba': 78, 'an-naba': 78, 'nabaa': 78,
#             'naziat': 79, 'an-naziat': 79, 'naazi\'aat': 79,
#             'abasa': 80,
#             'takwir': 81, 'at-takwir': 81,
#             'infitar': 82, 'al-infitar': 82, 'infitaar': 82,
#             'mutaffifin': 83, 'al-mutaffifin': 83, 'mutaffifeen': 83,
#             'inshiqaq': 84, 'al-inshiqaq': 84, 'inshiqaaq': 84,
#             'buruj': 85, 'al-buruj': 85, 'burooj': 85,
#             'tariq': 86, 'at-tariq': 86, 'taariq': 86,
#             'ala': 87, 'al-ala': 87, 'a\'la': 87, 'alaa': 87,
#             'ghashiya': 88, 'al-ghashiya': 88, 'ghaashiya': 88,
#             'fajr': 89, 'al-fajr': 89,
#             'balad': 90, 'al-balad': 90,
#             'shams': 91, 'ash-shams': 91, 'sun': 91,
#             'layl': 92, 'al-layl': 92, 'lail': 92, 'night': 92,
#             'duha': 93, 'ad-duha': 93, 'dhuhaa': 93,
#             'sharh': 94, 'ash-sharh': 94, 'inshirah': 94,
#             'tin': 95, 'at-tin': 95, 'teen': 95, 'fig': 95,
#             'alaq': 96, 'al-alaq': 96, 'iqra': 96,
#             'qadr': 97, 'al-qadr': 97, 'power': 97,
#             'bayyina': 98, 'al-bayyina': 98, 'bayyinah': 98,
#             'zalzala': 99, 'az-zalzala': 99, 'zilzaal': 99,
#             'adiyat': 100, 'al-adiyat': 100, 'aadiyaat': 100,
#             'qaria': 101, 'al-qaria': 101, 'qaari\'ah': 101,
#             'takathur': 102, 'at-takathur': 102, 'takaathur': 102,
#             'asr': 103, 'al-asr': 103, 'time': 103,
#             'humaza': 104, 'al-humaza': 104,
#             'fil': 105, 'al-fil': 105, 'elephant': 105,
#             'quraysh': 106, 'quraish': 106, 'qureish': 106,
#             'maun': 107, 'al-maun': 107, 'maa\'un': 107,
#             'kawthar': 108, 'al-kawthar': 108, 'kauthar': 108, 'kausar': 108,
#             'kafirun': 109, 'al-kafirun': 109, 'kaafiroon': 109, 'kafiroon': 109,
#             'nasr': 110, 'an-nasr': 110, 'help': 110,
#             'masad': 111, 'al-masad': 111, 'lahab': 111,
#             'ikhlas': 112, 'al-ikhlas': 112, 'ikhlaas': 112, 'sincerity': 112,
#             'falaq': 113, 'al-falaq': 113, 'dawn': 113,
#             'nas': 114, 'an-nas': 114, 'naas': 114, 'mankind': 114,
#         }
        
#         # Common verse nicknames
#         self.verse_nicknames = {
#             'ayat al kursi': '2:255',
#             'ayatul kursi': '2:255',
#             'ayat kursi': '2:255',
#             'throne verse': '2:255',
#             'kursi': '2:255',
#         }
        
#         # Patterns for different query formats
#         self.patterns = [
#             # Format: "2:255" or "2-255"
#             (r'^(\d{1,3})[:;،\-](\d{1,3})$', 'colon_format'),
            
#             # Format: "surah 2 ayah 255"
#             (r'surah?\s+(\d{1,3})\s+(?:ayah?|verse|aayat)\s+(\d{1,3})', 'surah_ayah'),
            
#             # Format: "verse 2:255"
#             (r'(?:verse|ayah?|aayat)\s+(\d{1,3})[:;،\-](\d{1,3})', 'verse_colon'),
            
#             # Format: "surah 2 verse 255" or "chapter 2 verse 255"
#             (r'(?:surah?|chapter)\s+(\d{1,3})\s+(?:verse|ayah?|aayat)\s+(\d{1,3})', 'chapter_verse'),
            
#             # Format: "2 255" (just numbers)
#             (r'^(\d{1,3})\s+(\d{1,3})$', 'space_format'),
#         ]
    
#     def parse(self, query: str) -> Optional[Dict]:
#         """
#         Parse a query and extract structural information
        
#         Returns:
#             Dict with 'type', 'surah', 'ayah' if structural query found
#             None if not a structural query
#         """
#         query_lower = query.lower().strip()
        
#         # Check for verse nicknames
#         for nickname, verse_ref in self.verse_nicknames.items():
#             if nickname in query_lower:
#                 surah, ayah = verse_ref.split(':')
#                 return {
#                     'type': 'nickname',
#                     'surah': int(surah),
#                     'ayah': int(ayah),
#                     'verse_key': verse_ref,
#                     'confidence': 1.0
#                 }
        
#         # Check for surah names
#         surah_num = self._extract_surah_name(query_lower)
#         if surah_num:
#             # If only surah name, return first ayah
#             ayah_num = self._extract_ayah_number(query_lower)
#             return {
#                 'type': 'surah_name',
#                 'surah': surah_num,
#                 'ayah': ayah_num or 1,
#                 'verse_key': f"{surah_num}:{ayah_num or 1}",
#                 'confidence': 0.9 if ayah_num else 0.7  # Lower confidence if no ayah specified
#             }
        
#         # Try pattern matching
#         for pattern, pattern_type in self.patterns:
#             match = re.search(pattern, query_lower, re.IGNORECASE)
#             if match:
#                 surah = int(match.group(1))
#                 ayah = int(match.group(2))
                
#                 # Validate ranges
#                 if not (1 <= surah <= 114):
#                     continue
#                 if not (1 <= ayah <= 286):  # Max ayahs in longest surah
#                     continue
                
#                 return {
#                     'type': pattern_type,
#                     'surah': surah,
#                     'ayah': ayah,
#                     'verse_key': f"{surah}:{ayah}",
#                     'confidence': 1.0
#                 }
        
#         return None
    
#     def _extract_surah_name(self, query: str) -> Optional[int]:
#         """Extract surah number from surah name"""
#         # Remove common prefixes
#         query = re.sub(r'\b(surah?|surat|chapter|sura)\b', '', query).strip()
        
#         # Check against known names
#         for name, number in self.surah_names.items():
#             if name in query:
#                 return number
        
#         return None
    
#     def _extract_ayah_number(self, query: str) -> Optional[int]:
#         """Extract ayah number from text"""
#         # Look for patterns like "ayah 5", "verse 5", "aayat 5"
#         match = re.search(r'(?:ayah?|verse|aayat)\s+(\d{1,3})', query, re.IGNORECASE)
#         if match:
#             return int(match.group(1))
#         return None
    
#     def is_structural_query(self, query: str) -> bool:
#         """Check if query is a structural query"""
#         return self.parse(query) is not None


# # Convenience function
# _parser = StructuralQueryParser()

# def parse_structural_query(query: str) -> Optional[Dict]:
#     """Parse structural query"""
#     return _parser.parse(query)

# def is_structural(query: str) -> bool:
#     """Check if structural query"""
#     return _parser.is_structural_query(query)


# # Test the parser
# if __name__ == "__main__":
#     test_queries = [
#         "2:255",
#         "surah 2 ayah 255",
#         "verse 2:255",
#         "ayat ul kursi",
#         "surah baqara",
#         "surah fatiha verse 1",
#         "what does quran say about patience",  # Not structural
#         "112:1",
#         "surah ikhlas",
#     ]
    
#     parser = StructuralQueryParser()
    
#     print("Structural Query Parser Tests:")
#     print("="*60)
    
#     for query in test_queries:
#         result = parser.parse(query)
#         print(f"\nQuery: {query}")
#         if result:
#             print(f"  ✓ STRUCTURAL: {result}")
#         else:
#             print(f"  ✗ NOT structural (needs semantic search)")