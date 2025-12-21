from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml

from schema_validate import validate_json
from text_norm_v1 import norm_ar, norm_lat, norm_ur, sort_casefold, tokenize_ar, tokenize_lat, tokenize_ur


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_verse_key(key: str) -> Tuple[int, int]:
    parts = key.split(":")
    return int(parts[0]), int(parts[1])


def _sort_verse_keys(keys: Iterable[str]) -> List[str]:
    return sorted(set(keys), key=lambda k: _parse_verse_key(k))


def _ensure_trigger_map(raw: Any) -> Dict[str, List[str]]:
    out = {"en": [], "ur": [], "ar": [], "tr": []}
    if not isinstance(raw, dict):
        return out
    for lang in ("en", "ur", "ar", "tr"):
        vals = raw.get(lang) or []
        if isinstance(vals, list):
            out[lang] = [str(v) for v in vals if str(v).strip()]
    return out


def _normalize_trigger(token: str, lang: str) -> str:
    if lang == "ar":
        return norm_ar(token)
    if lang == "ur":
        return norm_ur(token)
    return norm_lat(token)


def _normalize_trigger_list(tokens: Iterable[str]) -> List[str]:
    return sort_casefold({t for t in tokens if t})


def _expand_triggers(
    trigger_map: Dict[str, List[str]],
    variant_map: Dict[str, List[str]],
) -> Dict[str, Set[str]]:
    expanded: Dict[str, Set[str]] = {"en": set(), "ur": set(), "ar": set(), "tr": set()}

    for lang, tokens in trigger_map.items():
        for tok in tokens:
            norm = _normalize_trigger(tok, lang)
            if not norm:
                continue
            expanded[lang].add(norm)
            if norm in variant_map:
                for v in variant_map[norm]:
                    v_norm = _normalize_trigger(v, lang)
                    if v_norm:
                        expanded[lang].add(v_norm)

    return expanded


def _build_verse_tokens(verses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for v in verses:
        vk = str(v.get("verse_key") or v.get("id") or "").strip()
        if not vk:
            continue
        st = str(v.get("searchable_text") or "")
        ar = str(v.get("arabic") or "")
        tr = " ".join([str(v.get("transliteration") or ""), str(v.get("transliteration_alt") or "")]).strip()
        en_builtin = str(v.get("translation_en_builtin") or "")
        ur_builtin = str(v.get("translation_ur_builtin") or "")

        en_extra = " ".join([str(x) for x in (v.get("translations_english") or {}).values()])
        ur_extra = " ".join([str(x) for x in (v.get("translations_urdu") or {}).values()])

        en_text = " ".join([en_builtin, en_extra, st])
        ur_text = " ".join([ur_builtin, ur_extra, st])
        tr_text = " ".join([tr, st])
        ar_text = " ".join([ar, st])

        tokens = {
            "ar": set(tokenize_ar(ar_text)),
            "en": set(tokenize_lat(en_text)),
            "ur": set(tokenize_ur(ur_text)),
            "tr": set(tokenize_lat(tr_text)),
        }

        rows.append({"verse_key": vk, "tokens": tokens})
    return rows


def _matches(tokens: Dict[str, Set[str]], triggers: Dict[str, Set[str]]) -> bool:
    for lang, trig in triggers.items():
        if trig and tokens.get(lang) and tokens[lang].intersection(trig):
            return True
    return False


def _build_topic(
    topic: Dict[str, Any],
    variant_map: Dict[str, List[str]],
    verse_tokens: List[Dict[str, Any]],
) -> Dict[str, Any]:
    topic_id = str(topic.get("id") or "").strip()
    if not topic_id:
        raise ValueError("Topic missing id")

    raw_triggers = _ensure_trigger_map(topic.get("triggers"))
    raw_negative = _ensure_trigger_map(topic.get("negative_triggers"))

    normalized_triggers = _expand_triggers(raw_triggers, variant_map)
    normalized_negative = _expand_triggers(raw_negative, variant_map)

    anchors = [str(v) for v in topic.get("anchors") or [] if str(v).strip()]

    groups_in = topic.get("groups") or []
    group_entries: List[Dict[str, Any]] = []
    group_anchor_union: Set[str] = set()

    for g in groups_in:
        gid = str(g.get("id") or "").strip()
        title = g.get("title") or {}
        g_anchors = [str(v) for v in g.get("anchors") or [] if str(v).strip()]
        group_anchor_union.update(g_anchors)

        g_triggers = _ensure_trigger_map(g.get("triggers"))
        g_norm_triggers = _expand_triggers(g_triggers, variant_map) if g_triggers else {"en": set(), "ur": set(), "ar": set(), "tr": set()}

        g_keys: Set[str] = set(g_anchors)
        if any(g_norm_triggers.values()):
            for row in verse_tokens:
                vk = row["verse_key"]
                if vk in g_keys:
                    continue
                if _matches(row["tokens"], g_norm_triggers):
                    g_keys.add(vk)

        g_sorted = _sort_verse_keys(g_keys)
        if len(g_sorted) > 40:
            g_sorted = g_sorted[:40]

        group_entries.append(
            {
                "id": gid,
                "title": title,
                "verse_keys": g_sorted,
            }
        )

    anchor_set = set(anchors).union(group_anchor_union)
    topic_keys: Set[str] = set(anchor_set)

    for row in verse_tokens:
        vk = row["verse_key"]
        if vk in topic_keys:
            continue
        if _matches(row["tokens"], normalized_negative):
            continue
        if _matches(row["tokens"], normalized_triggers):
            topic_keys.add(vk)

    sorted_candidates = _sort_verse_keys(topic_keys - anchor_set)
    anchor_sorted = _sort_verse_keys(anchor_set)

    if len(anchor_sorted) >= 160:
        final_keys = anchor_sorted
    else:
        keep_n = 160 - len(anchor_sorted)
        final_keys = _sort_verse_keys(anchor_sorted + sorted_candidates[:keep_n])

    out_topic = {
        "id": topic_id,
        "names": topic.get("names") or {},
        "triggers": {k: _normalize_trigger_list(v) for k, v in raw_triggers.items()},
        "verse_keys": final_keys,
    }
    if any(raw_negative.values()):
        out_topic["negative_triggers"] = {k: _normalize_trigger_list(v) for k, v in raw_negative.items()}
    if group_entries:
        out_topic["groups"] = group_entries

    return out_topic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True, help="Path to topic_ontology_v1.yaml")
    ap.add_argument("--variant-map", required=True, help="Path to variant_map_v1.json")
    ap.add_argument("--in", dest="input_path", required=True, help="Path to quran_complete.json")
    ap.add_argument("--out", required=True, help="Output topic_pack_v1.json path")
    args = ap.parse_args()

    ontology_path = Path(args.ontology)
    variant_path = Path(args.variant_map)
    input_path = Path(args.input_path)
    out_path = Path(args.out)

    ontology = _load_yaml(ontology_path)
    if not isinstance(ontology, dict) or "topics" not in ontology:
        raise ValueError("topic_ontology_v1.yaml must have a top-level 'topics' list")

    topics = ontology.get("topics") or []
    if not isinstance(topics, list):
        raise ValueError("topic_ontology_v1.yaml topics must be a list")

    verses = _load_json(input_path)
    if not isinstance(verses, list) or not verses:
        raise ValueError("quran_complete.json must be a non-empty list")

    variant = _load_json(variant_path)
    validate_json("schemas/askquran_variant_map_v1.schema.json", variant)
    variant_map = variant.get("variants") or {}
    if not isinstance(variant_map, dict):
        raise ValueError("variant_map_v1.json variants must be an object")

    verse_tokens = _build_verse_tokens(verses)

    built_topics: List[Dict[str, Any]] = []
    for t in topics:
        if not isinstance(t, dict):
            continue
        built_topics.append(_build_topic(t, variant_map, verse_tokens))

    built_topics.sort(key=lambda t: str(t.get("id", "")).casefold())

    payload = {
        "schema": "askquran_topic_pack_v1",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": "ie-ai-py",
            "ontology": str(ontology_path.name),
            "input": str(input_path.name),
            "variant_map": str(variant_path.name),
        },
        "topics": built_topics,
    }

    validate_json("schemas/askquran_topic_pack_v1.schema.json", payload)
    _write_json(out_path, payload)


if __name__ == "__main__":
    main()
