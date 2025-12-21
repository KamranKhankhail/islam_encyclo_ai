from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


_RE_MULTI_WS = re.compile(r"\s+")
_RE_AR_CLEAN = re.compile(r"[^0-9\u0600-\u06FF]+")

_AR_DIACRITICS_RE = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06ED]"
)

_UR_PUNCT_TRANSLATE = str.maketrans({
    "\u060C": " ",  # Arabic comma
    "\u061B": " ",  # Arabic semicolon
    "\u061F": " ",  # Arabic question mark
    "\u06D4": " ",  # Urdu full stop
    "\u066B": " ",  # Arabic decimal separator
    "\u066C": " ",  # Arabic thousands separator
    "\u2018": " ",
    "\u2019": " ",
    "\u201C": " ",
    "\u201D": " ",
})

_AR_TRANSLATE = {
    # Alef variants -> Alef
    ord("\u0622"): "\u0627",
    ord("\u0623"): "\u0627",
    ord("\u0625"): "\u0627",
    ord("\u0671"): "\u0627",
    # Yeh variants -> Yeh
    ord("\u0649"): "\u064A",
    ord("\u0626"): "\u064A",
    ord("\u06CC"): "\u064A",
    # Waw with hamza -> Waw
    ord("\u0624"): "\u0648",
    # Taa marbuta -> Heh
    ord("\u0629"): "\u0647",
    # Standalone hamza removed
    ord("\u0621"): None,
}

_LAT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "by",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "that",
    "this",
    "these",
    "those",
    "as",
    "it",
}


def uniq_stable(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def sort_casefold(items: Iterable[str]) -> List[str]:
    return sorted(list(items), key=lambda s: s.casefold())


def norm_ar(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = _AR_DIACRITICS_RE.sub("", t)
    t = t.replace("\u0640", "")
    t = t.translate(_AR_TRANSLATE)
    t = _RE_MULTI_WS.sub(" ", t).strip()
    return t


def _keep_apostrophe_inside_words(text: str) -> str:
    out: List[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == "'":
            prev_ok = i > 0 and text[i - 1].isalnum()
            next_ok = i + 1 < n and text[i + 1].isalnum()
            out.append("'" if prev_ok and next_ok else " ")
        else:
            out.append(ch)
    return "".join(out)


def norm_lat(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).lower()
    t = _keep_apostrophe_inside_words(t)
    t = re.sub(r"[^a-z0-9']+", " ", t)
    t = _RE_MULTI_WS.sub(" ", t).strip()
    return t


def norm_ur(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = t.translate(_UR_PUNCT_TRANSLATE)
    t = _RE_MULTI_WS.sub(" ", t).strip()
    return t


def tokenize_ar(s: str) -> List[str]:
    t = norm_ar(s)
    parts = [p for p in t.split(" ") if p]
    out: List[str] = []
    for part in parts:
        cleaned = _RE_AR_CLEAN.sub("", part)
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out


def tokenize_lat(s: str) -> List[str]:
    t = norm_lat(s)
    tokens = [p for p in t.split(" ") if p]
    out: List[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok in _LAT_STOPWORDS:
            continue
        out.append(tok)
    return out


def tokenize_ur(s: str) -> List[str]:
    t = norm_ur(s)
    tokens = [p for p in t.split(" ") if p]
    out: List[str] = []
    for tok in tokens:
        cleaned = _RE_AR_CLEAN.sub("", tok)
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out


def _self_test() -> None:
    sample_ar = "\u0627\u0644\u0642\u0631\u0622\u0646\u064f \u0627\u0644\u0643\u0631\u064a\u0645\u0640"
    sample_lat = "Du'a for Ibrahim, Musa and Lut!"
    sample_ur = "\u0646\u0645\u0627\u0632\u060c \u0632\u06a9\u0627\u062a\u061f"

    def _safe(s: str) -> str:
        return s.encode("unicode_escape").decode("ascii")

    print("norm_ar:", _safe(norm_ar(sample_ar)))
    print("tokenize_ar:", [ _safe(t) for t in tokenize_ar(sample_ar) ])
    print("norm_lat:", _safe(norm_lat(sample_lat)))
    print("tokenize_lat:", tokenize_lat(sample_lat))
    print("norm_ur:", _safe(norm_ur(sample_ur)))
    print("tokenize_ur:", [ _safe(t) for t in tokenize_ur(sample_ur) ])


if __name__ == "__main__":
    _self_test()
