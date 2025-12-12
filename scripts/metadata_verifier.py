#!/usr/bin/env python3
"""
metadata.json Validator (Offline + Online)

Validates:
- Schema/shape (required keys, types)
- Surah integrity (1..114, unique, verse counts, revelation_type)
- Juz integrity (1..30, unique, verse_key format, ranges, contiguity, full coverage)
- Total verse count (warn if not 6236)
- Optional ONLINE verification against Al Quran Cloud API:
  - Surah verse counts
  - Juz boundaries (start/end verse keys)

Usage:
  python validate_metadata.py ./metadata.json
  python validate_metadata.py ./metadata.json --online
  python validate_metadata.py ./metadata.json --online --edition quran-uthmani
  python validate_metadata.py ./metadata.json --json
  python validate_metadata.py ./metadata.json --strict

Exit codes:
  0: OK (no errors; warnings allowed unless --strict)
  2: Validation errors found (or warnings treated as errors in --strict)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Optional online mode dependency
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF]")
VERSE_KEY_RE = re.compile(r"^\s*(\d{1,3})\s*:\s*(\d{1,3})\s*$")


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN" | "INFO"
    code: str
    message: str
    path: str = ""


class Report:
    def __init__(self) -> None:
        self.issues: List[Issue] = []

    def error(self, code: str, message: str, path: str = "") -> None:
        self.issues.append(Issue("ERROR", code, message, path))

    def warn(self, code: str, message: str, path: str = "") -> None:
        self.issues.append(Issue("WARN", code, message, path))

    def info(self, code: str, message: str, path: str = "") -> None:
        self.issues.append(Issue("INFO", code, message, path))

    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)

    def warnings_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "WARN")

    def errors_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": {
                "errors": self.errors_count(),
                "warnings": self.warnings_count(),
                "info": sum(1 for i in self.issues if i.level == "INFO"),
            },
            "issues": [asdict(i) for i in self.issues],
        }


def load_json(path: str, report: Report) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            report.error("E_JSON_ROOT", "Root JSON must be an object.", path="$")
            return None
        return data
    except FileNotFoundError:
        report.error("E_FILE_NOT_FOUND", f"File not found: {path}")
    except json.JSONDecodeError as e:
        report.error("E_JSON_PARSE", f"Invalid JSON: {e}")
    except Exception as e:
        report.error("E_JSON_READ", f"Failed to read JSON: {e}")
    return None


def parse_verse_key(key: str) -> Optional[Tuple[int, int]]:
    m = VERSE_KEY_RE.match(key or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def build_surah_maps(
    surahs: List[Dict[str, Any]],
    report: Report,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, int], List[int]]:
    """
    Returns:
      - surah_by_number
      - verses_by_surah (surah_number -> total_verses)
      - ordered_surah_numbers
    """
    surah_by_number: Dict[int, Dict[str, Any]] = {}
    verses_by_surah: Dict[int, int] = {}
    ordered_numbers: List[int] = []

    seen = set()
    for idx, s in enumerate(surahs):
        path = f"$.surahs[{idx}]"
        if not isinstance(s, dict):
            report.error("E_SURAH_TYPE", "Each surah entry must be an object.", path=path)
            continue

        num = s.get("number")
        if not isinstance(num, int):
            report.error("E_SURAH_NUMBER_TYPE", "Surah.number must be an integer.", path=f"{path}.number")
            continue
        if not (1 <= num <= 114):
            report.error("E_SURAH_NUMBER_RANGE", "Surah.number must be in 1..114.", path=f"{path}.number")
            continue
        if num in seen:
            report.error("E_SURAH_DUP", f"Duplicate surah number: {num}.", path=f"{path}.number")
            continue
        seen.add(num)

        tv = s.get("total_verses")
        if not isinstance(tv, int):
            report.error("E_SURAH_VERSECOUNT_TYPE", "Surah.total_verses must be an integer.", path=f"{path}.total_verses")
            continue
        if tv <= 0:
            report.error("E_SURAH_VERSECOUNT_RANGE", "Surah.total_verses must be > 0.", path=f"{path}.total_verses")
            continue

        # Light sanity checks on strings
        if not is_nonempty_str(s.get("name_arabic")):
            report.warn("W_SURAH_NAME_AR_EMPTY", "Surah.name_arabic is missing/empty.", path=f"{path}.name_arabic")
        else:
            if not ARABIC_LETTERS_RE.search(s["name_arabic"]):
                report.warn("W_SURAH_NAME_AR_NONAR", "Surah.name_arabic does not appear to contain Arabic letters.", path=f"{path}.name_arabic")

        if not is_nonempty_str(s.get("name_english")):
            report.warn("W_SURAH_NAME_EN_EMPTY", "Surah.name_english is missing/empty.", path=f"{path}.name_english")

        if not is_nonempty_str(s.get("name_transliteration")):
            report.warn("W_SURAH_NAME_TR_EMPTY", "Surah.name_transliteration is missing/empty.", path=f"{path}.name_transliteration")

        rt = s.get("revelation_type")
        if rt not in ("Meccan", "Medinan"):
            report.warn(
                "W_SURAH_REVELATION_TYPE",
                "Surah.revelation_type should be 'Meccan' or 'Medinan' (warning only).",
                path=f"{path}.revelation_type",
            )

        surah_by_number[num] = s
        verses_by_surah[num] = tv
        ordered_numbers.append(num)

    ordered_numbers.sort()

    # Strong expectation: all 114 surahs exist
    if len(ordered_numbers) != 114:
        missing = [n for n in range(1, 115) if n not in surah_by_number]
        if missing:
            report.error("E_SURAH_MISSING", f"Missing surah entries for: {missing[:20]}{'...' if len(missing) > 20 else ''}", path="$.surahs")

    # Strong expectation: sequential without gaps
    if ordered_numbers and ordered_numbers != list(range(1, len(ordered_numbers) + 1)):
        # Only warn because some datasets might be partial in other contexts,
        # but for your use case, it should be complete.
        report.warn("W_SURAH_NONSEQUENTIAL", "Surah numbers are not perfectly sequential from 1..114.", path="$.surahs")

    return surah_by_number, verses_by_surah, ordered_numbers


def compute_cumulative_offsets(verses_by_surah: Dict[int, int], report: Report) -> Dict[int, int]:
    """
    Returns offsets where offset[s] = total verses before surah s (1-indexed surahs).
    """
    offsets: Dict[int, int] = {}
    total = 0
    for s in range(1, 115):
        offsets[s] = total
        tv = verses_by_surah.get(s)
        if tv is None:
            # If missing, cannot compute accurate offsets
            report.error("E_OFFSETS_MISSING_SURAH", f"Cannot compute offsets: missing total_verses for surah {s}.", path="$.surahs")
            tv = 0
        total += tv

    # Standard Quran verse count is commonly 6236 (as referenced by Al Quran Cloud docs). :contentReference[oaicite:1]{index=1}
    if total != 6236:
        report.warn("W_TOTAL_VERSE_COUNT", f"Sum(total_verses) = {total}, expected 6236 (common count).", path="$.surahs")
    return offsets


def verse_to_abs(surah: int, ayah: int, offsets: Dict[int, int]) -> int:
    return offsets[surah] + ayah


def abs_to_verse(abs_index: int, verses_by_surah: Dict[int, int], offsets: Dict[int, int]) -> Tuple[int, int]:
    # abs_index is 1-based verse index across whole Quran
    # We search surah by offsets.
    for s in range(114, 0, -1):
        if abs_index > offsets[s]:
            ayah = abs_index - offsets[s]
            return s, ayah
    return 1, abs_index  # fallback (should not happen)


def validate_juz_mappings(
    juz_mappings: Any,
    verses_by_surah: Dict[int, int],
    offsets: Dict[int, int],
    report: Report,
) -> List[Dict[str, Any]]:
    if not isinstance(juz_mappings, list):
        report.error("E_JUZ_TYPE", "juz_mappings must be an array.", path="$.juz_mappings")
        return []

    normalized: List[Dict[str, Any]] = []
    seen = set()

    for idx, j in enumerate(juz_mappings):
        path = f"$.juz_mappings[{idx}]"
        if not isinstance(j, dict):
            report.error("E_JUZ_ENTRY_TYPE", "Each juz mapping must be an object.", path=path)
            continue

        jnum = j.get("juz")
        if not isinstance(jnum, int):
            report.error("E_JUZ_NUM_TYPE", "juz must be an integer.", path=f"{path}.juz")
            continue
        if not (1 <= jnum <= 30):
            report.error("E_JUZ_NUM_RANGE", "juz must be in 1..30.", path=f"{path}.juz")
            continue
        if jnum in seen:
            report.error("E_JUZ_DUP", f"Duplicate juz number: {jnum}.", path=f"{path}.juz")
            continue
        seen.add(jnum)

        start = j.get("start")
        end = j.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            report.error("E_JUZ_STARTEND_TYPE", "start/end must be strings like '2:255'.", path=path)
            continue

        s_parsed = parse_verse_key(start)
        e_parsed = parse_verse_key(end)
        if not s_parsed:
            report.error("E_VERSEKEY_START", f"Invalid start verse key: {start!r}", path=f"{path}.start")
            continue
        if not e_parsed:
            report.error("E_VERSEKEY_END", f"Invalid end verse key: {end!r}", path=f"{path}.end")
            continue

        ss, sa = s_parsed
        es, ea = e_parsed

        # Range checks within Quran
        if not (1 <= ss <= 114 and 1 <= es <= 114):
            report.error("E_VERSEKEY_SURAH_RANGE", "Surah in start/end must be in 1..114.", path=path)
            continue

        ss_tv = verses_by_surah.get(ss)
        es_tv = verses_by_surah.get(es)
        if ss_tv is None or es_tv is None:
            report.error("E_VERSEKEY_SURAH_UNKNOWN", "Start/end surah missing from surahs list.", path=path)
            continue
        if not (1 <= sa <= ss_tv):
            report.error("E_VERSEKEY_AYAH_RANGE_START", f"Start ayah out of range for surah {ss} (1..{ss_tv}).", path=f"{path}.start")
            continue
        if not (1 <= ea <= es_tv):
            report.error("E_VERSEKEY_AYAH_RANGE_END", f"End ayah out of range for surah {es} (1..{es_tv}).", path=f"{path}.end")
            continue

        abs_start = verse_to_abs(ss, sa, offsets)
        abs_end = verse_to_abs(es, ea, offsets)

        if abs_start > abs_end:
            report.error("E_JUZ_ORDER", "Juz start must be <= juz end.", path=path)
            continue

        normalized.append(
            {
                "juz": jnum,
                "start": f"{ss}:{sa}",
                "end": f"{es}:{ea}",
                "_abs_start": abs_start,
                "_abs_end": abs_end,
            }
        )

    normalized.sort(key=lambda x: x["juz"])

    if len(normalized) != 30:
        missing = [n for n in range(1, 31) if n not in {x["juz"] for x in normalized}]
        if missing:
            report.error("E_JUZ_MISSING", f"Missing juz entries for: {missing}", path="$.juz_mappings")

    # Coverage & contiguity checks (only if we have a full ordered set)
    by_juz = {x["juz"]: x for x in normalized}
    if all(n in by_juz for n in range(1, 31)):
        first = by_juz[1]
        last = by_juz[30]
        if first["start"] != "1:1":
            report.error("E_JUZ_FIRST_START", "Juz 1 must start at 1:1.", path="$.juz_mappings[?juz=1].start")
        if last["end"] != "114:6":
            report.error("E_JUZ_LAST_END", "Juz 30 must end at 114:6.", path="$.juz_mappings[?juz=30].end")

        for n in range(2, 31):
            prev = by_juz[n - 1]
            cur = by_juz[n]
            if cur["_abs_start"] != prev["_abs_end"] + 1:
                prev_end = abs_to_verse(prev["_abs_end"], verses_by_surah, offsets)
                expected_start = abs_to_verse(prev["_abs_end"] + 1, verses_by_surah, offsets)
                report.error(
                    "E_JUZ_CONTIGUITY",
                    (
                        f"Juz {n} is not contiguous. "
                        f"Juz {n-1} ends at {prev_end[0]}:{prev_end[1]}, "
                        f"so Juz {n} should start at {expected_start[0]}:{expected_start[1]}, "
                        f"but starts at {cur['start']}."
                    ),
                    path=f"$.juz_mappings[?juz={n}]",
                )

    return normalized


def validate_offline(metadata: Dict[str, Any], report: Report) -> Dict[str, Any]:
    # Required keys
    if "surahs" not in metadata:
        report.error("E_MISSING_SURAHS", "Missing top-level key: 'surahs'.", path="$")
        surahs = []
    else:
        surahs = metadata["surahs"]

    if "juz_mappings" not in metadata:
        report.error("E_MISSING_JUZ", "Missing top-level key: 'juz_mappings'.", path="$")
        juz_mappings = []
    else:
        juz_mappings = metadata["juz_mappings"]

    if not isinstance(surahs, list):
        report.error("E_SURAHS_TYPE", "'surahs' must be an array.", path="$.surahs")
        surahs_list: List[Dict[str, Any]] = []
    else:
        surahs_list = surahs  # type: ignore

    surah_by_num, verses_by_surah, ordered_nums = build_surah_maps(surahs_list, report)
    offsets = compute_cumulative_offsets(verses_by_surah, report)

    normalized_juz = validate_juz_mappings(juz_mappings, verses_by_surah, offsets, report)

    return {
        "surah_by_num": surah_by_num,
        "verses_by_surah": verses_by_surah,
        "offsets": offsets,
        "normalized_juz": normalized_juz,
        "ordered_surah_numbers": ordered_nums,
    }


def http_get_json(url: str, timeout: float = 20.0, retries: int = 2, backoff: float = 0.6) -> Any:
    if requests is None:
        raise RuntimeError("requests is not installed. Run: pip install requests")
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise last_err


def validate_online_alquran_cloud(
    offline_ctx: Dict[str, Any],
    report: Report,
    api_base: str,
    edition: str,
) -> None:
    """
    Online verification using Al Quran Cloud API:
      - GET /v1/surah
      - GET /v1/juz/{juz}/{edition}
    Endpoints are documented publicly. :contentReference[oaicite:2]{index=2}
    """
    verses_by_surah: Dict[int, int] = offline_ctx["verses_by_surah"]
    normalized_juz: List[Dict[str, Any]] = offline_ctx["normalized_juz"]

    # 1) Surah list & verse counts
    surah_url = f"{api_base.rstrip('/')}/surah"
    try:
        payload = http_get_json(surah_url)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) < 114:
            report.warn("W_ONLINE_SURAHS_SHAPE", "Online /surah response shape unexpected.", path=surah_url)
        else:
            mismatches = []
            for s in data:
                try:
                    num = int(s.get("number"))
                    online_count = int(s.get("numberOfAyahs"))
                except Exception:
                    continue
                local_count = verses_by_surah.get(num)
                if local_count is None:
                    mismatches.append((num, online_count, None))
                elif local_count != online_count:
                    mismatches.append((num, online_count, local_count))

            if mismatches:
                sample = mismatches[:10]
                report.error(
                    "E_ONLINE_SURAH_VERSECOUNT_MISMATCH",
                    f"Surah verse-count mismatches vs online source (sample): {sample}{'...' if len(mismatches) > 10 else ''}",
                    path=surah_url,
                )
            else:
                report.info("I_ONLINE_SURAHS_OK", "Online surah verse counts match.", path=surah_url)

    except Exception as e:
        report.warn("W_ONLINE_SURAHS_FAIL", f"Online surah verification failed: {e}", path=surah_url)

    # 2) Juz boundaries
    # We fetch each Juz and compare the first/last ayah's (surah:numberInSurah) to metadata boundaries.
    if len(normalized_juz) >= 1:
        local_by_juz = {j["juz"]: j for j in normalized_juz if isinstance(j.get("juz"), int)}
        boundary_mismatches: List[Tuple[int, str, str]] = []

        for jnum in range(1, 31):
            local = local_by_juz.get(jnum)
            if not local:
                continue
            juz_url = f"{api_base.rstrip('/')}/juz/{jnum}/{edition}"
            try:
                payload = http_get_json(juz_url, timeout=30.0, retries=2)
                data = payload.get("data") if isinstance(payload, dict) else None
                ayahs = data.get("ayahs") if isinstance(data, dict) else None

                if not isinstance(ayahs, list) or not ayahs:
                    report.warn("W_ONLINE_JUZ_SHAPE", f"Online juz {jnum} response missing ayahs.", path=juz_url)
                    continue

                first = ayahs[0]
                last = ayahs[-1]

                fs = int(first.get("surah", {}).get("number"))
                fa = int(first.get("numberInSurah"))
                ls = int(last.get("surah", {}).get("number"))
                la = int(last.get("numberInSurah"))

                online_start = f"{fs}:{fa}"
                online_end = f"{ls}:{la}"

                if online_start != local["start"] or online_end != local["end"]:
                    boundary_mismatches.append((jnum, f"{local['start']}..{local['end']}", f"{online_start}..{online_end}"))

            except Exception as e:
                report.warn("W_ONLINE_JUZ_FAIL", f"Online juz {jnum} verification failed: {e}", path=juz_url)

        if boundary_mismatches:
            report.error(
                "E_ONLINE_JUZ_BOUNDARY_MISMATCH",
                f"Juz boundary mismatches vs online source (sample): {boundary_mismatches[:10]}{'...' if len(boundary_mismatches) > 10 else ''}",
                path=f"{api_base.rstrip('/')}/juz/{{juz}}/{edition}",
            )
        else:
            report.info("I_ONLINE_JUZ_OK", "Online juz boundaries match.", path=f"{api_base.rstrip('/')}/juz/{{juz}}/{edition}")


def render_human(report: Report) -> str:
    lines: List[str] = []
    lines.append("Validation Report")
    lines.append("=" * 80)

    summary = report.to_json()["summary"]
    lines.append(f"Errors:   {summary['errors']}")
    lines.append(f"Warnings: {summary['warnings']}")
    lines.append(f"Info:     {summary['info']}")
    lines.append("-" * 80)

    for i in report.issues:
        loc = f" [{i.path}]" if i.path else ""
        lines.append(f"{i.level:<5} {i.code}{loc}: {i.message}")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Validate metadata.json (offline + optional online verification).")
    p.add_argument("path", help="Path to metadata.json")
    p.add_argument("--online", action="store_true", help="Enable online verification (Al Quran Cloud).")
    p.add_argument("--api-base", default="https://api.alquran.cloud/v1", help="API base URL (default: Al Quran Cloud).")
    p.add_argument("--edition", default="quran-uthmani", help="Edition used for juz verification (default: quran-uthmani).")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    p.add_argument("--strict", action="store_true", help="Treat warnings as errors (non-zero exit if any warnings).")
    args = p.parse_args()

    report = Report()
    data = load_json(args.path, report)
    if data is None:
        print(render_human(report))
        return 2

    offline_ctx = validate_offline(data, report)

    if args.online:
        if requests is None:
            report.error("E_REQUESTS_MISSING", "Online mode requires 'requests'. Install: pip install requests")
        else:
            validate_online_alquran_cloud(
                offline_ctx=offline_ctx,
                report=report,
                api_base=args.api_base,
                edition=args.edition,
            )

    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, indent=2))
    else:
        print(render_human(report))

    if report.has_errors():
        return 2
    if args.strict and report.warnings_count() > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
