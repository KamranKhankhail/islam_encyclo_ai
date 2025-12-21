from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import yaml

from text_norm_v1 import norm_ar, norm_lat, norm_ur, sort_casefold, tokenize_ar, tokenize_lat, tokenize_ur


ANCHOR_CAP = 5
GROUP_ANCHOR_CAP = 3
NAME_PLACEHOLDER = "???"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_localized_triggers(path: Path | None) -> Dict[str, Dict[str, List[str]]]:
    if not path or not path.exists():
        return {}
    data = _load_json(path)
    if isinstance(data, dict) and "topics" in data:
        data = data.get("topics") or {}
    if not isinstance(data, dict):
        raise ValueError("localized triggers must be an object or an object with 'topics'")

    out: Dict[str, Dict[str, List[str]]] = {}
    for topic_id, payload in data.items():
        if not isinstance(payload, dict):
            continue
        ar_vals = payload.get("ar") or []
        ur_vals = payload.get("ur") or []
        if not isinstance(ar_vals, list):
            ar_vals = []
        if not isinstance(ur_vals, list):
            ur_vals = []
        out[str(topic_id)] = {
            "ar": [str(v).strip() for v in ar_vals if str(v).strip()],
            "ur": [str(v).strip() for v in ur_vals if str(v).strip()],
        }
    return out


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )


def _parse_verse_key(key: str) -> Tuple[int, int]:
    parts = key.split(":")
    return int(parts[0]), int(parts[1])


def _sort_verse_keys(keys: Iterable[str]) -> List[str]:
    return sorted(set(keys), key=lambda k: _parse_verse_key(k))


def _ensure_trigger_map(raw: Dict[str, Iterable[str]] | None) -> Dict[str, List[str]]:
    out = {"en": [], "ur": [], "ar": [], "tr": []}
    if not isinstance(raw, dict):
        return out
    for lang in ("en", "ur", "ar", "tr"):
        vals = raw.get(lang) or []
        if isinstance(vals, list):
            cleaned = [str(v).strip() for v in vals if str(v).strip()]
            out[lang] = cleaned
    return out


def _normalize_trigger(token: str, lang: str) -> str:
    if lang == "ar":
        return norm_ar(token)
    if lang == "ur":
        return norm_ur(token)
    return norm_lat(token)


def _normalize_trigger_list(tokens: Iterable[str]) -> List[str]:
    return sort_casefold({t for t in (s.strip() for s in tokens) if t})


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
            for v in variant_map.get(norm, []):
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


def _make_names(en: str) -> Dict[str, str]:
    return {"en": en, "ur": NAME_PLACEHOLDER, "ar": NAME_PLACEHOLDER}


def _make_triggers(en: Iterable[str], tr: Iterable[str] | None = None) -> Dict[str, List[str]]:
    return {
        "en": _normalize_trigger_list(en),
        "ur": [],
        "ar": [],
        "tr": _normalize_trigger_list(tr or []),
    }


def _merge_localized_triggers(
    base: Dict[str, List[str]],
    localized: Dict[str, List[str]] | None,
) -> Dict[str, List[str]]:
    if not localized:
        return base
    merged = dict(base)
    merged["ar"] = _normalize_trigger_list(base.get("ar", []) + (localized.get("ar") or []))
    merged["ur"] = _normalize_trigger_list(base.get("ur", []) + (localized.get("ur") or []))
    return merged


def _make_group(
    group_id: str,
    title_en: str,
    en_triggers: Iterable[str],
    tr_triggers: Iterable[str] | None = None,
) -> Dict[str, Any]:
    return {
        "id": group_id,
        "title": _make_names(title_en),
        "triggers": _make_triggers(en_triggers, tr_triggers),
    }


def _topic_definitions(
    localized_triggers: Dict[str, Dict[str, List[str]]] | None = None,
) -> List[Dict[str, Any]]:
    topics: List[Dict[str, Any]] = []

    def add_topic(
        topic_id: str,
        name_en: str,
        en_triggers: Iterable[str],
        tr_triggers: Iterable[str] | None = None,
        negative_triggers: Dict[str, Iterable[str]] | None = None,
        groups: List[Dict[str, Any]] | None = None,
    ) -> None:
        localized = localized_triggers.get(topic_id) if localized_triggers else None
        triggers = _merge_localized_triggers(_make_triggers(en_triggers, tr_triggers), localized)
        topic: Dict[str, Any] = {
            "id": topic_id,
            "names": _make_names(name_en),
            "triggers": triggers,
        }
        if negative_triggers:
            topic["negative_triggers"] = _ensure_trigger_map(negative_triggers)
        if groups:
            topic["groups"] = groups
        topics.append(topic)

    # Core beliefs
    add_topic("topic.faith", "Faith", ["faith", "belief", "believe", "believers"], ["iman", "eemaan"])
    add_topic("topic.disbelief", "Disbelief", ["disbelief", "disbelieve", "disbeliever", "unbelievers", "deny", "deniers", "reject"], ["kufr", "kafir", "kaafir"])
    add_topic("topic.hypocrisy", "Hypocrisy", ["hypocrite", "hypocrisy", "hypocrites"], ["munafiq", "nifaq"])
    add_topic("topic.tawhid", "Tawhid", ["monotheism", "oneness"], ["tawhid"])
    add_topic("topic.shirk", "Shirk", ["associating", "associate", "partners", "partner"], ["shirk"])
    add_topic("topic.angels", "Angels", ["angel", "angels"], ["malaika", "malak"])
    add_topic("topic.jinn", "Jinn", ["jinn", "genie"], ["jinn"])
    add_topic("topic.shaitan", "Shaitan", ["satan", "devil"], ["shaitan", "shaytan"])
    add_topic("topic.quran", "Quran", ["quran", "qur'an"], ["quran"])
    add_topic("topic.revelation", "Revelation", ["revelation", "revealed", "reveal"], ["wahy"])
    add_topic("topic.prophets", "Prophets", ["prophet", "prophets"], ["nabi", "anbiya"])
    add_topic("topic.messengers", "Messengers", ["messenger", "messengers", "apostle", "apostles"], ["rasul", "rusul"])
    add_topic("topic.scripture", "Scripture", ["scripture", "book", "books"], ["kitab"])
    add_topic("topic.torah", "Torah", ["torah"], ["tawrat"])
    add_topic("topic.gospel", "Gospel", ["gospel"], ["injil"])
    add_topic("topic.afterlife", "Afterlife", ["afterlife", "hereafter"], ["akhirah", "akhirat"])
    add_topic("topic.judgment", "Judgment", ["judgment", "judgement", "reckoning", "account"], ["hisab"])
    add_topic("topic.resurrection", "Resurrection", ["resurrection", "raised", "revive", "revival"], [])
    add_topic("topic.paradise", "Paradise", ["paradise", "garden", "gardens"], ["jannah"])
    add_topic("topic.hell", "Hell", ["hell", "fire"], ["jahannam", "nar"])

    # Virtues and character
    add_topic("topic.mercy", "Mercy", ["mercy", "merciful"], ["rahmah", "rahma"])
    add_topic("topic.forgiveness", "Forgiveness", ["forgive", "forgiveness", "forgiving"], ["maghfirah", "ghafur"])
    add_topic("topic.repentance", "Repentance", ["repent", "repentance"], ["tawbah", "tawba"])
    add_topic("topic.gratitude", "Gratitude", ["gratitude", "thankful", "thanks", "grateful"], ["shukr"])
    add_topic("topic.patience", "Patience", ["patience", "patient"], ["sabr"])
    add_topic("topic.trust", "Trust in Allah", ["trust", "rely", "reliance"], ["tawakkul"])
    add_topic("topic.taqwa", "Taqwa (Piety)", ["piety", "righteous", "righteousness", "godfearing"], ["taqwa"])
    add_topic("topic.sincerity", "Sincerity", ["sincere", "sincerity"], ["ikhlas"])
    add_topic("topic.justice", "Justice", ["justice", "just", "equity", "equitable"], ["adl"])
    add_topic("topic.kindness", "Kindness", ["kindness", "kind", "compassion", "benevolence"], ["ihsan"])
    add_topic("topic.truthfulness", "Truthfulness", ["truth", "truthful", "honest", "honesty"], ["sidq"])
    add_topic("topic.knowledge", "Knowledge", ["knowledge", "learn", "learned"], ["ilm"])
    add_topic("topic.wisdom", "Wisdom", ["wisdom", "wise"], ["hikmah"])
    add_topic("topic.humility", "Humility", ["humility", "humble"], ["tawadu"])

    # Worship
    add_topic("topic.salah", "Prayer (Salah)", ["prayer", "pray"], ["salah", "salat"])
    add_topic("topic.zakat", "Zakat", ["alms", "almsgiving"], ["zakat", "zakah"])
    add_topic("topic.charity", "Charity", ["charity", "almsgiving"], ["sadaqah", "sadaqa"])
    add_topic("topic.fasting", "Fasting", ["fast", "fasting"], ["sawm", "siyam"])
    add_topic("topic.hajj", "Hajj", ["hajj", "pilgrimage"], ["hajj"])
    add_topic("topic.dua", "Supplication (Dua)", ["supplication", "invoke"], ["dua", "duaa"])
    add_topic("topic.dhikr", "Remembrance (Dhikr)", ["remembrance"], ["dhikr", "zikr"])
    add_topic("topic.qibla", "Qibla", ["qibla", "direction"], ["qibla"])
    add_topic("topic.mosque", "Mosque", ["mosque", "masjid"], ["masjid"])
    add_topic("topic.purification", "Purification", ["purify", "purification", "cleanse", "ablution"], ["wudu", "tayammum", "taharah"])
    add_topic("topic.sacrifice", "Sacrifice", ["sacrifice", "slaughter", "offering"], ["udhiyah", "qurban"])
    add_topic("topic.ramadan", "Ramadan", ["ramadan", "ramadhan"], ["ramadan"])

    # Rulings, society, and economics
    add_topic("topic.marriage", "Marriage", ["marriage", "marry", "wife", "wives", "husband", "spouse"], ["nikah"])
    add_topic("topic.divorce", "Divorce", ["divorce", "divorced", "separation"], ["talaq"])
    add_topic("topic.inheritance", "Inheritance", ["inherit", "inheritance", "heir", "heirs"], ["mirath"])
    add_topic("topic.orphans", "Orphans", ["orphan", "orphans"], ["yatim"])
    add_topic("topic.parents", "Parents", ["parents", "parent"], [])
    add_topic("topic.women", "Women", ["women", "woman", "wives"], [])
    add_topic("topic.hijab", "Hijab", ["hijab", "veil", "veiling"], ["hijab"])
    add_topic("topic.adultery", "Adultery", ["adultery", "fornication", "unchastity"], ["zina"])
    add_topic("topic.usury", "Usury (Riba)", ["usury", "interest"], ["riba"])
    add_topic("topic.debt", "Debt", ["debt", "debts"], [])
    add_topic("topic.contracts", "Contracts", ["contract", "agreement", "covenant", "pledge"], [])
    add_topic("topic.trade", "Trade", ["trade", "commerce", "buy", "sell"], ["tijarah"])
    add_topic("topic.testimony", "Testimony", ["testimony", "witness", "witnesses"], [])
    add_topic("topic.theft", "Theft", ["theft", "steal", "stolen", "thief"], [])
    add_topic("topic.murder", "Murder", ["murder", "kill", "killing", "slain"], [])
    add_topic("topic.intoxicants", "Intoxicants", ["intoxicants", "wine", "drunk", "alcohol"], ["khamr"])
    add_topic("topic.gambling", "Gambling", ["gambling", "lottery"], ["maisir"])
    add_topic("topic.pork", "Pork", ["pork", "swine"], [])
    add_topic("topic.halal", "Halal", ["lawful", "permitted"], ["halal"])
    add_topic("topic.haram", "Haram", ["forbidden", "prohibited", "unlawful"], ["haram"])
    add_topic("topic.jihad", "Jihad", ["jihad", "strive", "striving"], ["jihad"])
    add_topic("topic.war", "War", ["war", "battle", "fight", "fighting"], ["qital"])
    add_topic("topic.peace", "Peace", ["peace", "reconcile", "reconciliation", "truce"], ["salam"])
    add_topic("topic.migration", "Migration", ["migration", "migrate", "emigrate", "emigrants"], ["hijrah", "hijra"])
    add_topic("topic.slavery", "Slavery", ["slave", "slaves", "bondwoman", "bondwomen", "captives"], [])
    add_topic("topic.bequest", "Bequest", ["bequest", "testament"], ["wasiyyah"])

    # Prophets (entities)
    prophets = [
        ("adam", "Adam", ["adam"], ["adam"]),
        ("idris", "Idris (Enoch)", ["idris", "enoch"], ["idris"]),
        ("nuh", "Nuh (Noah)", ["nuh", "noah"], ["nuh", "nooh"]),
        ("hud", "Hud", ["hud"], ["hud"]),
        ("salih", "Salih", ["salih", "saleh"], ["salih", "saleh"]),
        ("ibrahim", "Ibrahim (Abraham)", ["ibrahim", "abraham"], ["ibrahim", "ibraheem"]),
        ("lut", "Lut (Lot)", ["lut"], ["lut", "loot"]),
        ("ismail", "Ismail (Ishmael)", ["ismail", "ishmael"], ["ismail", "ismael"]),
        ("ishaq", "Ishaq (Isaac)", ["ishaq", "isaac"], ["ishaq", "ishaaq"]),
        ("yaqub", "Yaqub (Jacob)", ["yaqub", "jacob"], ["yaqub", "yaaqoob"]),
        ("yusuf", "Yusuf (Joseph)", ["yusuf", "joseph"], ["yusuf", "yousuf"]),
        ("shuayb", "Shuayb (Jethro)", ["shuayb", "shu'aib", "jethro"], ["shuayb", "shuaib"]),
        ("ayyub", "Ayyub (Job)", ["ayyub", "job"], ["ayyub", "ayub"]),
        ("dhulkifl", "Dhul-Kifl", ["dhulkifl", "kifl"], ["dhulkifl", "kifl"]),
        ("musa", "Musa (Moses)", ["musa", "moses"], ["musa", "moosa"]),
        ("harun", "Harun (Aaron)", ["harun", "aaron"], ["harun", "haroon"]),
        ("dawud", "Dawud (David)", ["dawud", "david"], ["dawud", "daud"]),
        ("sulayman", "Sulayman (Solomon)", ["sulayman", "solomon"], ["sulayman", "sulaiman"]),
        ("ilyas", "Ilyas (Elijah)", ["ilyas", "elijah"], ["ilyas", "ilyaas"]),
        ("alyasa", "Alyasa (Elisha)", ["alyasa", "elisha"], ["alyasa"]),
        ("yunus", "Yunus (Jonah)", ["yunus", "jonah"], ["yunus", "younus"]),
        ("zakariya", "Zakariya (Zechariah)", ["zakariya", "zechariah"], ["zakariya", "zakaria"]),
        ("yahya", "Yahya (John)", ["yahya", "john"], ["yahya", "yahyaa"]),
        ("isa", "Isa (Jesus)", ["isa", "jesus"], ["isa", "eesa"]),
        ("muhammad", "Muhammad", ["muhammad", "mohammad", "ahmad"], ["muhammad", "mohammad"]),
    ]
    for slug, name_en, en_tr, tr_tr in prophets:
        add_topic(f"entity.{slug}", name_en, en_tr, tr_tr)

    # Other named figures
    others = [
        ("maryam", "Maryam (Mary)", ["maryam", "mary", "mariam"], ["maryam", "mariam"]),
        ("luqman", "Luqman", ["luqman"], ["luqman"]),
        ("dhulqarnayn", "Dhul-Qarnayn", ["dhulqarnayn", "zulqarnayn", "qarnayn"], ["dhulqarnayn", "zulqarnayn", "qarnayn"]),
        ("talut", "Talut (Saul)", ["talut", "saul"], ["talut"]),
        ("jalut", "Jalut (Goliath)", ["jalut", "goliath"], ["jalut"]),
        ("qarun", "Qarun (Korah)", ["qarun", "korah"], ["qarun"]),
        ("firawn", "Firawn (Pharaoh)", ["pharaoh", "pharaohs"], ["firawn", "firon"]),
        ("haman", "Haman", ["haman"], ["haman"]),
        ("samiri", "Samiri", ["samiri", "samaritan"], ["samiri"]),
        ("abu_lahab", "Abu Lahab", ["lahab"], ["lahab"]),
        ("azar", "Azar", ["azar"], ["azar"]),
        ("imran", "Imran", ["imran"], ["imran"]),
        ("uzair", "Uzair (Ezra)", ["uzair", "ezra"], ["uzair"]),
    ]
    for slug, name_en, en_tr, tr_tr in others:
        add_topic(f"entity.{slug}", name_en, en_tr, tr_tr)

    # Places
    places = [
        ("makkah", "Makkah (Mecca)", ["mecca", "makkah"], ["makkah"]),
        ("bakkah", "Bakkah", ["bakkah"], ["bakkah"]),
        ("kaaba", "Kaaba", ["kaaba", "ka'bah", "kaba"], ["kaaba", "kaba"]),
        ("masjid_haram", "Sacred Mosque", ["haram", "sanctuary", "sacred", "mosque"], ["haram", "masjid"]),
        ("masjid_aqsa", "Farthest Mosque", ["aqsa", "farthest"], ["aqsa"]),
        ("safa", "Safa", ["safa"], ["safa"]),
        ("marwa", "Marwa", ["marwa"], ["marwa"]),
        ("arafat", "Arafat", ["arafat", "arafah"], ["arafat", "arafah"]),
        ("sinai", "Sinai", ["sinai"], ["sinai", "tur"]),
        ("egypt", "Egypt", ["egypt"], ["misr"]),
        ("madyan", "Madyan", ["madyan", "midian"], ["madyan"]),
    ]
    for slug, name_en, en_tr, tr_tr in places:
        add_topic(f"place.{slug}", name_en, en_tr, tr_tr)

    # Peoples and nations
    peoples = [
        ("bani_israel", "Bani Israel", ["israel", "israelites"], ["israel"]),
        ("aad", "Aad", ["aad"], ["aad"]),
        ("thamud", "Thamud", ["thamud"], ["thamud"]),
        ("madyan", "People of Madyan", ["madyan", "midian"], ["madyan"]),
        ("saba", "Saba (Sheba)", ["saba", "sheba"], ["saba"]),
        ("quraysh", "Quraysh", ["quraysh", "quraish"], ["quraysh", "quraish"]),
        ("people_of_cave", "People of the Cave", ["cave", "caves"], ["kahf"]),
        ("people_of_elephant", "People of the Elephant", ["elephant"], ["fil"]),
        ("people_of_trench", "People of the Trench", ["trench", "ditch"], ["ukhdood"]),
        ("sabbath_breakers", "Sabbath Breakers", ["sabbath"], ["sabt"]),
    ]
    for slug, name_en, en_tr, tr_tr in peoples:
        add_topic(f"people.{slug}", name_en, en_tr, tr_tr)

    # Story topics with groups
    add_topic(
        "story.nuh",
        "Story of Nuh",
        ["nuh", "noah", "ark", "flood"],
        ["nuh", "nooh"],
        groups=[
            _make_group("ark", "Ark", ["ark", "ship", "vessel"]),
            _make_group("flood", "Flood", ["flood", "deluge", "rain", "waves"]),
            _make_group("salvation", "Salvation", ["saved", "drowned", "drown"]),
        ],
    )
    add_topic(
        "story.ibrahim",
        "Story of Ibrahim",
        ["ibrahim", "abraham", "fire", "kaaba", "sacrifice"],
        ["ibrahim", "ibraheem"],
        groups=[
            _make_group("trials", "Trials", ["fire", "sacrifice", "son"]),
            _make_group("house", "Sacred House", ["kaaba", "house", "sanctuary"]),
            _make_group("legacy", "Legacy", ["imam", "leader", "descendants"]),
        ],
    )
    add_topic(
        "story.lut",
        "Story of Lut",
        ["lut", "lewd", "stones"],
        ["lut", "loot"],
        groups=[
            _make_group("actions", "Actions", ["lewd", "lust", "approach"]),
            _make_group("warnings", "Warnings", ["warn", "warning", "messengers"]),
            _make_group("outcome", "Outcome", ["stones", "rain", "overturned"]),
        ],
    )
    add_topic(
        "story.musa",
        "Story of Musa",
        ["musa", "moses", "pharaoh", "sea", "staff"],
        ["musa", "moosa"],
        groups=[
            _make_group("pharaoh", "Pharaoh", ["pharaoh", "firawn"]),
            _make_group("signs", "Signs", ["staff", "rod", "snake", "hand"]),
            _make_group("sea", "Sea", ["sea", "parted", "split"]),
        ],
    )
    add_topic(
        "story.yusuf",
        "Story of Yusuf",
        ["yusuf", "joseph", "dream", "prison"],
        ["yusuf", "yousuf"],
        groups=[
            _make_group("dream", "Dream", ["dream", "vision", "stars"]),
            _make_group("prison", "Prison", ["prison", "imprisoned"]),
            _make_group("reunion", "Reunion", ["brothers", "father", "forgave"]),
        ],
    )
    add_topic(
        "story.isa",
        "Story of Isa",
        ["isa", "jesus", "messiah", "disciples"],
        ["isa", "eesa"],
        groups=[
            _make_group("birth", "Birth", ["virgin", "birth", "cradle"]),
            _make_group("miracles", "Miracles", ["heal", "blind", "leper", "dead"]),
            _make_group("disciples", "Disciples", ["disciples", "helpers"]),
        ],
    )
    add_topic(
        "story.dhulqarnayn",
        "Story of Dhul-Qarnayn",
        ["dhulqarnayn", "zulqarnayn", "qarnayn", "gog", "magog"],
        ["dhulqarnayn", "zulqarnayn", "qarnayn"],
        groups=[
            _make_group("journeys", "Journeys", ["east", "west", "sunset", "sunrise"]),
            _make_group("barrier", "Barrier", ["barrier", "wall", "dam", "gog", "magog"]),
            _make_group("justice", "Justice", ["punish", "reward"]),
        ],
    )
    add_topic(
        "story.talut_jalut",
        "Story of Talut and Jalut",
        ["talut", "jalut", "goliath", "saul"],
        ["talut", "jalut"],
        groups=[
            _make_group("selection", "Selection", ["king", "talut"]),
            _make_group("river", "River Test", ["river"]),
            _make_group("battle", "Battle", ["jalut", "goliath"]),
        ],
    )

    # Story topics without groups
    add_topic("story.adam", "Story of Adam", ["adam"], ["adam"])
    add_topic("story.maryam", "Story of Maryam", ["maryam", "mary", "mariam", "virgin"], ["maryam", "mariam"])
    add_topic("story.yunus", "Story of Yunus", ["yunus", "jonah", "fish"], ["yunus", "younus"])
    add_topic("story.people_of_cave", "Story of the Cave", ["cave", "sleep"], ["kahf"])

    return topics


def _topic_matches(
    verse_tokens: List[Dict[str, Any]],
    triggers: Dict[str, Set[str]],
    negative: Dict[str, Set[str]],
) -> Set[str]:
    matched: Set[str] = set()
    for row in verse_tokens:
        tokens = row["tokens"]
        if _matches(tokens, negative):
            continue
        if _matches(tokens, triggers):
            matched.add(row["verse_key"])
    return matched


def _build_topic_output(
    topic: Dict[str, Any],
    variant_map: Dict[str, List[str]],
    verse_tokens: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    raw_triggers = _ensure_trigger_map(topic.get("triggers"))
    raw_negative = _ensure_trigger_map(topic.get("negative_triggers"))

    expanded_triggers = _expand_triggers(raw_triggers, variant_map)
    expanded_negative = _expand_triggers(raw_negative, variant_map)

    topic_matches = _topic_matches(verse_tokens, expanded_triggers, expanded_negative)
    anchors = _sort_verse_keys(topic_matches)[:ANCHOR_CAP]

    group_defs = topic.get("groups") or []
    groups_out: List[Dict[str, Any]] = []
    empty_groups: List[str] = []
    if isinstance(group_defs, list):
        for g in group_defs:
            g_triggers_raw = _ensure_trigger_map(g.get("triggers"))
            g_triggers = _expand_triggers(g_triggers_raw, variant_map)
            g_matches = []
            if any(g_triggers.values()):
                for row in verse_tokens:
                    vk = row["verse_key"]
                    if vk not in topic_matches:
                        continue
                    if _matches(row["tokens"], g_triggers):
                        g_matches.append(vk)
            g_anchors = _sort_verse_keys(g_matches)[:GROUP_ANCHOR_CAP]
            if not g_anchors:
                empty_groups.append(f"{topic.get('id')}:{g.get('id')}")
                g_anchors = anchors[:GROUP_ANCHOR_CAP] if anchors else []
            groups_out.append(
                {
                    "id": g.get("id"),
                    "title": g.get("title") or {},
                    "anchors": g_anchors,
                }
            )

    out_topic = {
        "id": topic.get("id"),
        "names": topic.get("names") or {},
        "triggers": raw_triggers,
        "anchors": anchors,
    }
    if any(raw_negative.values()):
        out_topic["negative_triggers"] = raw_negative
    if groups_out:
        out_topic["groups"] = groups_out

    empty_topics = []
    if not anchors:
        empty_topics.append(str(topic.get("id")))

    return out_topic, empty_topics, empty_groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="Path to quran_complete.json")
    ap.add_argument("--variant-map", required=True, help="Path to variant_map_v1.json")
    ap.add_argument("--out", required=True, help="Output topic_ontology_v1.yaml path")
    ap.add_argument(
        "--localized-triggers",
        default=None,
        help="Optional path to localized triggers JSON (ar/ur).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    localized_path = Path(args.localized_triggers) if args.localized_triggers else None
    if localized_path is None:
        default_path = repo_root / "topics" / "topic_triggers_ur_ar_v1.json"
        localized_path = default_path if default_path.exists() else None

    verses = _load_json(Path(args.input_path))
    if not isinstance(verses, list) or not verses:
        raise ValueError("quran_complete.json must be a non-empty list")

    variant = _load_json(Path(args.variant_map))
    variant_map = variant.get("variants") or {}
    if not isinstance(variant_map, dict):
        raise ValueError("variant_map_v1.json variants must be an object")

    localized_triggers = _load_localized_triggers(localized_path)
    verse_tokens = _build_verse_tokens(verses)
    topics_in = _topic_definitions(localized_triggers)

    if localized_triggers:
        topic_ids = {str(t.get("id") or "") for t in topics_in}
        unknown = sorted([tid for tid in localized_triggers.keys() if tid not in topic_ids])
        if unknown:
            raise ValueError(f"Localized triggers contain unknown topic ids: {unknown}")

    seen_ids: Set[str] = set()
    topics_out: List[Dict[str, Any]] = []
    empty_topics: List[str] = []
    empty_groups: List[str] = []

    for t in topics_in:
        tid = str(t.get("id") or "")
        if not tid:
            continue
        if tid in seen_ids:
            raise ValueError(f"Duplicate topic id: {tid}")
        seen_ids.add(tid)

        out_topic, empty_t, empty_g = _build_topic_output(t, variant_map, verse_tokens)
        topics_out.append(out_topic)
        empty_topics.extend(empty_t)
        empty_groups.extend(empty_g)

    payload = {"version": 1, "topics": topics_out}
    _write_yaml(Path(args.out), payload)

    print(f"topics_total: {len(topics_out)}")
    if empty_topics:
        print("topics_with_empty_anchors:")
        for tid in empty_topics:
            print(f"- {tid}")
    if empty_groups:
        print("groups_with_empty_anchors:")
        for gid in empty_groups:
            print(f"- {gid}")

    if empty_topics:
        raise SystemExit("One or more topics have zero anchors. Adjust triggers.")


if __name__ == "__main__":
    main()
