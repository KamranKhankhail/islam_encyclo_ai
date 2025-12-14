import re
from typing import Any, Dict, List, Tuple, Optional

DEFAULT_PACK_VERSION = "askquran.v1"
DEFAULT_EN_ORDER = ["sahih-international", "yusuf-ali", "pickthall", "shakir"]
DEFAULT_UR_ORDER = ["maududi", "jalandhry", "junagarhi"]
SCORE_PRECISION = 6

_AR_BLOCK = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_lang_hint(raw_query: str) -> str:
    text = raw_query or ""
    has_ar = bool(_AR_BLOCK.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "ar"
    return "en"


def qround(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), SCORE_PRECISION)
    except Exception:
        return None


def _get_translation_block(verse: Dict[str, Any], lang: str) -> Dict[str, str]:
    translations = verse.get("translations")
    if isinstance(translations, dict):
        block = translations.get(lang)
        if isinstance(block, dict):
            return block

    if lang.startswith("ur"):
        block = verse.get("translations_urdu")
    else:
        block = verse.get("translations_english")

    return block if isinstance(block, dict) else {}


def pick_display_text(verse: Dict[str, Any], ui_lang: str) -> Tuple[str, str]:
    lang_code = "ur" if str(ui_lang or "").lower().startswith("ur") else "en"
    order = DEFAULT_UR_ORDER if lang_code == "ur" else DEFAULT_EN_ORDER

    translations = _get_translation_block(verse, lang_code)

    for key in order:
        text = translations.get(key)
        if text:
            t = str(text).strip()
            if t:
                return lang_code, t

    # Fallback: first non-empty translation deterministically (sorted keys)
    for key in sorted(translations.keys()):
        text = translations.get(key)
        if text:
            t = str(text).strip()
            if t:
                return lang_code, t

    # Fallback to builtin translations if available
    if lang_code == "ur":
        builtin = str(verse.get("translation_ur_builtin") or "").strip()
        if builtin:
            return lang_code, builtin
    else:
        builtin = str(verse.get("translation_en_builtin") or "").strip()
        if builtin:
            return lang_code, builtin

    return lang_code, ""


def _parse_verse_key(vk: Optional[str]) -> Tuple[int, int]:
    try:
        s, a = str(vk).split(":")
        return int(s), int(a)
    except Exception:
        return (0, 0)


def build_ayah_result_v1(
    verse: Dict[str, Any],
    mode: str,
    bm25: Optional[float] = None,
    semantic: Optional[float] = None,
    rrf: Optional[float] = None,
    ui_lang: str = "en",
) -> Dict[str, Any]:
    verse_key = verse.get("verse_key") or verse.get("id") or ""
    surah, ayah = _parse_verse_key(verse_key)

    meta = verse.get("meta") or {}
    meta_obj = {
        "juz": int(verse.get("juz") or meta.get("juz") or 0),
        "ruku": int(verse.get("ruku") or meta.get("ruku") or 0),
        "page": int(verse.get("page") or meta.get("page") or 0),
    }

    disp_lang, disp_text = pick_display_text(verse, ui_lang)

    return {
        "verseKey": verse_key,
        "surah": surah,
        "ayah": ayah,
        "arabic": verse.get("arabic") or "",
        "display": {"lang": disp_lang, "text": disp_text},
        "meta": meta_obj,
        "score": {
            "mode": mode,
            "rrf": qround(rrf),
            "bm25": qround(bm25),
            "semantic": qround(semantic),
        },
    }


def build_search_response_v1(
    raw_query: str,
    normalized_query: str,
    intent: str,
    ui_lang: str,
    results: List[Dict[str, Any]],
    pack_version: str = DEFAULT_PACK_VERSION,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "packVersion": pack_version,
        "query": {
            "raw": raw_query,
            "normalized": normalized_query,
            "intent": intent,
            "langHint": detect_lang_hint(raw_query),
        },
        "page": {
            "offset": 0,
            "limit": len(results),
            "total": len(results),
            "nextOffset": None,
        },
        "results": results,
    }
