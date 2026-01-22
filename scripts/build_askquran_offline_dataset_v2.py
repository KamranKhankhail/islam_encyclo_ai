#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AskQuran Offline Dataset Builder — v2 FINAL
Phase 3: Smart Topic Packs (deterministic curated recall booster)
Phase 4: Answer experience layer (citations-only; no hallucinations)

Run:
  python build_askquran_offline_dataset_v2.py --input quran_complete.json --out ./askquran_dataset_v2

Produces:
  ./askquran_dataset_v2/
    quran_complete.json
    topics/seed_variants_v2.json
    topics/topic_ontology_v2.yaml
    artifacts/variant_map_v2.json
    artifacts/topic_pack_v2.json
    artifacts/answer_templates_v2.json
    tools/validate_dataset_v2.py
    README.txt
    SHA256SUMS.txt
  plus a zip:
    askquran_offline_dataset_v2_FINAL.zip
"""

from __future__ import annotations
import argparse, os, json, re, hashlib, datetime, zipfile, collections, itertools, shutil, unicodedata
from typing import Dict, List, Tuple, Any

import yaml

# -----------------------------
# Normalization & tokenization
# -----------------------------
_AR_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_AR_CLEAN_RE = re.compile(r"[^0-9\u0600-\u06FF]+")
_LAT_NONWORD_RE = re.compile(r"[^a-z0-9']+")
_WS_RE = re.compile(r"\s+")

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

STOP_EN = set(
    """
the a an and or of in on at for from by with is are was were be been being to that this these those as it
""".split()
)

def vk_to_tuple(vk: str) -> Tuple[int,int]:
    s,a = vk.split(":")
    return (int(s), int(a))

def norm_ar(s: str) -> str:
    if not s: return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = _AR_DIACRITICS_RE.sub("", t)
    t = t.replace("\u0640", "")
    t = t.translate(_AR_TRANSLATE)
    t = _WS_RE.sub(" ", t).strip()
    return t

def _keep_apostrophe_inside_words(text: str) -> str:
    out=[]
    n=len(text)
    for i,ch in enumerate(text):
        if ch == "'":
            prev_ok = i > 0 and text[i-1].isalnum()
            next_ok = i + 1 < n and text[i+1].isalnum()
            out.append("'" if prev_ok and next_ok else " ")
        else:
            out.append(ch)
    return "".join(out)

def norm_lat(s: str) -> str:
    if not s: return ""
    t = unicodedata.normalize("NFKC", str(s)).lower()
    t = _keep_apostrophe_inside_words(t)
    t = _LAT_NONWORD_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

def norm_ur(s: str) -> str:
    if not s: return ""
    t = unicodedata.normalize("NFKC", str(s))
    t = t.translate(_UR_PUNCT_TRANSLATE)
    t = _WS_RE.sub(" ", t).strip()
    return t

def tokenize_ar_list(s: str) -> List[str]:
    t = norm_ar(s)
    parts = [p for p in t.split(" ") if p]
    out=[]
    for part in parts:
        cleaned = _AR_CLEAN_RE.sub("", part)
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out

def tokenize_lat_list(s: str) -> List[str]:
    return [t for t in norm_lat(s).split(" ") if len(t) >= 2 and t not in STOP_EN]

def tokenize_ur_list(s: str) -> List[str]:
    t = norm_ur(s)
    parts = [p for p in t.split(" ") if p]
    out=[]
    for part in parts:
        cleaned = _AR_CLEAN_RE.sub("", part)
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out

def phrase_to_tokens(phrase: str, lang: str) -> List[str]:
    if not phrase: return []
    if lang == "ar":
        return tokenize_ar_list(phrase)
    if lang == "ur":
        return tokenize_ur_list(phrase)
    # en/tr
    return tokenize_lat_list(phrase)

def ordered_match(seq: List[str], phrase_tokens: List[str], max_gap: int = 2) -> bool:
    """
    Best-in-class deterministic phrase handling for offline compilation:
    - ordered
    - allows up to max_gap tokens between phrase tokens
    """
    if not phrase_tokens: return False
    i = 0
    last = -1
    for pt in phrase_tokens:
        found = False
        for j in range(i, len(seq)):
            if seq[j] == pt:
                if last >= 0 and (j - last - 1) > max_gap:
                    # too far; continue searching for a closer occurrence
                    continue
                found = True
                last = j
                i = j + 1
                break
        if not found:
            return False
    return True

def uniq_sorted_casefold(items: List[str]) -> List[str]:
    seen=set(); out=[]
    for x in items:
        if not x: continue
        k=x.casefold()
        if k in seen: continue
        seen.add(k); out.append(x)
    out.sort(key=lambda x: x.casefold())
    return out

# -----------------------------
# Text helpers (use all translations for recall)
# -----------------------------
def all_en_text(v: Dict[str,Any]) -> str:
    st = v.get("searchable_text","")
    parts=[v.get("translation_en_builtin","")]
    te=v.get("translations_english") or {}
    for k in sorted(te.keys()):
        parts.append(te[k] or "")
    parts.append(st or "")
    parts.append(v.get("surah_name_english",""))
    parts.append(v.get("surah_name_transliteration",""))
    return " ".join(parts)

def all_ur_text(v: Dict[str,Any]) -> str:
    st = v.get("searchable_text","")
    parts=[v.get("translation_ur_builtin","")]
    tu=v.get("translations_urdu") or {}
    for k in sorted(tu.keys()):
        parts.append(tu[k] or "")
    parts.append(st or "")
    return " ".join(parts)

# -----------------------------
# Entity/stories seeds
# -----------------------------
PROPHET_DEFS = [
    {"canon":"adam","en":["adam"],"ar":["آدم"],"ur":["آدم"]},
    {"canon":"idris","en":["idris"],"ar":["إدريس","ادريس"],"ur":["ادریس","ادريس"]},
    {"canon":"nuh","en":["noah","nuh","nooh"],"ar":["نوح"],"ur":["نوح"]},
    {"canon":"hud","en":["hud"],"ar":["هود"],"ur":["ہود"]},
    {"canon":"salih","en":["salih","saleh"],"ar":["صالح"],"ur":["صالح"]},
    {"canon":"ibrahim","en":["abraham","ibrahim","ibraheem"],"ar":["إبراهيم","ابراهيم"],"ur":["ابراہیم","ابراهيم"]},
    {"canon":"ismail","en":["ishmael","ismail","ismael"],"ar":["إسماعيل","اسماعيل"],"ur":["اسماعیل","اسماعيل"]},
    {"canon":"ishaq","en":["isaac","ishaq"],"ar":["إسحاق","اسحاق"],"ur":["اسحاق"]},
    {"canon":"yaqub","en":["jacob","yaqub","yaaqoob"],"ar":["يعقوب"],"ur":["یعقوب"]},
    {"canon":"yusuf","en":["joseph","yusuf","yoosuf"],"ar":["يوسف"],"ur":["یوسف"]},
    {"canon":"lut","en":["lut","loot"],"ar":["لوط"],"ur":["لوط"]},
    {"canon":"shuayb","en":["shuayb","shuaib","shoayb"],"ar":["شعيب"],"ur":["شعیب"]},
    {"canon":"ayyub","en":["job","ayyub"],"ar":["أيوب","ايوب"],"ur":["ایوب"]},
    {"canon":"dhul_kifl","en":["dhul kifl","dhul-kifl","dhu al kifl","dhu al-kifl"],"ar":["ذو الكفل","ذي الكفل"],"ur":["ذوالکفل","ذو الکفل"]},
    {"canon":"musa","en":["moses","musa","moosa"],"ar":["موسى","موسي"],"ur":["موسی","موسیٰ"]},
    {"canon":"harun","en":["aaron","harun","haroon"],"ar":["هارون"],"ur":["ہارون"]},
    {"canon":"dawud","en":["david","dawud","daud"],"ar":["داود"],"ur":["داؤد","داود"]},
    {"canon":"sulayman","en":["solomon","sulayman","sulaiman"],"ar":["سليمان"],"ur":["سلیمان"]},
    {"canon":"ilyas","en":["elijah","ilyas","ilyaas"],"ar":["إلياس","الياس"],"ur":["الیاس"]},
    {"canon":"alyasa","en":["elisha","alyasa","al-yasa","alyasa'"],"ar":["اليسع"],"ur":["الیسع"]},
    {"canon":"yunus","en":["jonah","yunus"],"ar":["يونس"],"ur":["یونس"]},
    {"canon":"zakariya","en":["zechariah","zakariya","zakaria"],"ar":["زكريا","زکریا"],"ur":["زکریا"]},
    {"canon":"yahya","en":["john","yahya"],"ar":["يحيى"],"ur":["یحییٰ","یحیی"]},
    {"canon":"isa","en":["jesus","isa","eesa"],"ar":["عيسى","عيسي"],"ur":["عیسیٰ","عیسی"]},
    {"canon":"muhammad","en":["muhammad","mohammad","muhammed"],"ar":["محمد"],"ur":["محمد"]},
]

FIGURE_DEFS = [
    {"canon":"maryam","en":["maryam","mary"],"ar":["مريم"],"ur":["مریم"]},
    {"canon":"firawn","en":["pharaoh","firawn"],"ar":["فرعون"],"ur":["فرعون"],"title_en":"Pharaoh"},
    {"canon":"qarun","en":["korah","qarun"],"ar":["قارون"],"ur":["قارون"]},
    {"canon":"haman","en":["haman"],"ar":["هامان"],"ur":["ہامان"]},
    {"canon":"luqman","en":["luqman"],"ar":["لقمان"],"ur":["لقمان"]},
    {"canon":"talut","en":["saul","talut"],"ar":["طالوت"],"ur":["طالوت"]},
    {"canon":"jalut","en":["goliath","jalut"],"ar":["جالوت"],"ur":["جالوت"]},
    {"canon":"dhu_al_qarnayn","en":["dhul qarnayn","dhul-qarnayn","dhu al qarnayn","dhul qarnain","zulqarnain"],"ar":["ذو القرنين","ذي القرنين"],"ur":["ذوالقرنین","ذو القرنین"],"title_en":"Dhul-Qarnayn"},
    {"canon":"ashab_al_kahf","en":["people of the cave","companions of the cave","ashab al kahf"],"ar":["اصحاب الكهف","أصحاب الكهف"],"ur":["اصحاب کہف","اصحاب الکہف"],"title_en":"People of the Cave"},
    {"canon":"ashab_al_fil","en":["people of the elephant","companions of the elephant","ashab al fil"],"ar":["اصحاب الفيل","أصحاب الفيل"],"ur":["اصحاب فیل","اصحاب الفیل"],"title_en":"People of the Elephant"},
    {"canon":"ashab_al_ukhdud","en":["people of the trench","companions of the trench","ashab al ukhdud"],"ar":["اصحاب الاخدود","أصحاب الأخدود"],"ur":["اصحاب الاخدود"],"title_en":"People of the Trench"},
    {"canon":"yajuj_majuj","en":["gog","magog","gog and magog","yajuj","majuj"],"ar":["ياجوج","ماجوج"],"ur":["یاجوج","ماجوج"],"title_en":"Gog and Magog"},
    {"canon":"bani_israel","en":["children of israel","israelites","bani israel"],"ar":["بني اسرائيل","بني إسرائيل"],"ur":["بنی اسرائیل"],"title_en":"Children of Israel"},
    {"canon":"yahud","en":["jews","jewish"],"ar":["يهود"],"ur":["یہود"],"title_en":"Jews"},
    {"canon":"nasara","en":["christians","christian","nasara"],"ar":["نصارى"],"ur":["نصاریٰ","نصارى"],"title_en":"Christians"},
    {"canon":"imran","en":["imran","imraan"],"ar":["عمران"],"ur":["عمران"]},
    {"canon":"azar","en":["azar"],"ar":["آزر","ازر"],"ur":["آزر","ازر"]},
    {"canon":"uzair","en":["uzair","ezra"],"ar":["عزير"],"ur":["عزیر"]},
    {"canon":"samiri","en":["samiri","samaritan"],"ar":["سامري"],"ur":["سامری"]},
    {"canon":"saba","en":["saba","sheba"],"ar":["سبأ","سبا"],"ur":["سبا","سبأ"],"title_en":"Saba"},
]

ENTITY_DEFS = PROPHET_DEFS + FIGURE_DEFS

# -----------------------------
# “FINAL” expanded ontology specs
# NOTE: The builder drops any trigger that does not appear in corpus.
# -----------------------------
def build_seed_variants() -> Dict[str,List[str]]:
    seed: Dict[str,List[str]] = {}

    def add(canon: str, variants: List[str]):
        canon = (canon or "").strip()
        if not canon: return
        seed.setdefault(canon, [])
        seed[canon].extend([v for v in variants if v and v.strip()])

    # Core ibadah/fiqh/aquaid/akhlaq concepts (expand aggressively, but safe)
    add("salah", ["salat","salaah","salah","prayer","prayers"])
    add("zakat", ["zakat","zakah","zakaat","alms","almsgiving"])
    add("sadaqah", ["sadaqah","sadaqa","charity","alms"])
    add("infaq", ["infaq","spend","spending","expend","expenditure","donate","donation"])
    add("fasting", ["fast","fasting","sawm","saum","ramadan","ramadhan"])
    add("hajj", ["hajj","pilgrimage"])
    add("umrah", ["umrah","umra","'umrah"])
    add("dua", ["dua","duaa","du'a","supplication","invocation","implore"])
    add("dhikr", ["dhikr","zikr","remembrance","remember"])
    add("tawbah", ["tawbah","repent","repentance"])
    add("istighfar", ["istighfar","seek forgiveness","forgiveness"])
    add("taqwa", ["taqwa","piety","righteousness","godfearing"])
    add("iman", ["iman","faith","believe","belief"])
    add("kufr", ["kufr","disbelief","unbelief"])
    add("shirk", ["shirk","associate partners","partners"])
    add("munafiq", ["munafiq","hypocrite","hypocrites","hypocrisy"])
    add("akhirah", ["akhirah","hereafter","afterlife"])
    add("qiyamah", ["qiyamah","resurrection","judgment","judgement"])
    add("jannah", ["jannah","paradise","gardens","heaven"])
    add("jahannam", ["jahannam","hell","hellfire"])
    add("riba", ["riba","usury","interest"])
    add("halal", ["halal","lawful","permitted"])
    add("haram", ["haram","forbidden","prohibited","unlawful"])
    add("jihad", ["jihad","striving"])
    add("fitnah", ["fitnah","trial","temptation","persecution"])
    add("sabr", ["sabr","patience","steadfast","persevere","endure"])
    add("shukr", ["shukr","gratitude","thankful","thanks"])
    add("adl", ["justice","equity"])
    add("zulm", ["injustice","oppression","wrongdoing"])
    add("rahmah", ["mercy","compassion","merciful"])
    add("amanah", ["trust","entrust","trustworthiness"])
    add("shura", ["consultation","counsel","shura"])

    # Major fiqh themes (extend)
    add("nikah", ["nikah","marriage","marry"])
    add("talaq", ["talaq","divorce"])
    add("mahr", ["mahr","dower","dowry"])
    add("inheritance", ["inheritance","heirs","will"])
    add("orphans", ["orphan","orphans"])
    add("zina", ["zina","fornication","adultery"])
    add("theft", ["theft","steal","stealing"])
    add("murder", ["murder","kill","killing"])
    add("qisas", ["qisas","retaliation"])
    add("kafarah", ["kaffarah","kafarah","expiation","atonement"])
    add("oaths", ["oath","oaths","swear","swearing"])
    add("vows", ["vow","vows"])
    add("khamr", ["khamr","wine","intoxicants","alcohol"])
    add("maysir", ["maysir","gambling"])
    add("food", ["pork","swine","carcass","blood","lawful food"])

    # Akhlaq virtues/vices (extend)
    add("kibr", ["arrogance","pride","proud"])
    add("hasad", ["envy","jealousy"])
    add("ghibah", ["backbiting","slander","defamation","gossip"])
    add("kidhb", ["lie","lying","falsehood"])
    add("fasad", ["corruption","mischief","spread corruption"])
    add("forgiveness", ["forgive","forgiveness","pardon"])
    add("sidq", ["truthful","truthfulness","honesty"])
    add("haya", ["modesty","chastity"])

    # Arabic anchor keys (high precision)
    for ar in [
        "الصلاة","الزكاة","صدقة","انفاق","الصيام","رمضان","الحج","عمرة","دعاء","ذكر","توبة","استغفر",
        "ايمان","تقوى","شرك","كفر","منافقون","الاخرة","القيامة","الجنة","جهنم","النار","الربا",
        "حلال","حرام","جهاد","فتنة","صبر","شكر","عدل","قسط","ظلم","رحمة","امانة","شورى",
        "نكاح","طلاق","ميراث","يتامى","زنا","سرق","قتل","قصاص","كفارة","يمين","نذر",
        "خمر","ميسر","خنزير","ميتة","دم",
        "غيبة","بهتان","حسد","كبر","كذب","فساد",
        "ربنا","اهل","الكتاب","بني","اسرائيل"
    ]:
        add(norm_ar(ar), [ar])

    # People/stories (expanded set)
    for entry in ENTITY_DEFS:
        canon = entry.get("canon","")
        if not canon:
            continue
        envars = entry.get("en") or []
        arvars = entry.get("ar") or []
        urvars = entry.get("ur") or []
        add(canon, envars)
        for a in arvars:
            add(norm_ar(a), [a])
        for u in urvars:
            add(norm_ur(u), [u])

    return seed

def split_variants(items: List[str]) -> Tuple[List[str], List[str]]:
    tokens=[]
    phrases=[]
    for raw in (items or []):
        v = str(raw).strip()
        if not v:
            continue
        if re.search(r"[\s\-]", v):
            phrases.append(v)
        else:
            tokens.append(v)
    return tokens, phrases

def load_localized_triggers(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    topics = data.get("topics") if isinstance(data, dict) else None
    return topics if isinstance(topics, dict) else {}

def merge_localized_triggers(topics: List[Dict[str,Any]], localized: Dict[str, Dict[str, List[str]]]) -> None:
    if not localized:
        return
    for t in topics:
        tid = t.get("id")
        if not tid:
            continue
        extra = localized.get(tid)
        if not isinstance(extra, dict):
            continue
        for lang in ("ar","ur"):
            vals = extra.get(lang)
            if isinstance(vals, list) and vals:
                t["match"]["any_tokens"][lang].extend([str(v) for v in vals if v])

# -----------------------------
# Ontology topics (expanded)
# - Use tokens + phrases + negative triggers.
# - Keep it deterministic; drop triggers not present in corpus.
# -----------------------------
def ontology_specs() -> List[Dict[str,Any]]:
    """
    This is intentionally large but still curated.
    The builder will prune triggers that don't exist in the corpus.
    """
    T = []

    def topic(tid: str, en: str, ur: str, ar: str,
              tok_en=None, tok_ur=None, tok_ar=None, tok_tr=None,
              phr_en=None, phr_ur=None, phr_ar=None, phr_tr=None,
              neg_tok_en=None, neg_tok_ur=None, neg_tok_ar=None, neg_tok_tr=None,
              neg_phr_en=None, neg_phr_ur=None, neg_phr_ar=None, neg_phr_tr=None,
              kind="topic", groups=None):
        T.append({
            "id": tid, "kind": kind,
            "names": {"en": en, "ur": ur, "ar": ar},
            "match": {
                "any_tokens": {"en": tok_en or [], "ur": tok_ur or [], "ar": tok_ar or [], "tr": tok_tr or []},
                "any_phrases": {"en": phr_en or [], "ur": phr_ur or [], "ar": phr_ar or [], "tr": phr_tr or []},
            },
            "negative": {
                "any_tokens": {"en": neg_tok_en or [], "ur": neg_tok_ur or [], "ar": neg_tok_ar or [], "tr": neg_tok_tr or []},
                "any_phrases": {"en": neg_phr_en or [], "ur": neg_phr_ur or [], "ar": neg_phr_ar or [], "tr": neg_phr_tr or []},
            },
            "groups": groups
        })

    # Ibadah (core)
    topic("topic.salah","Prayer (Salah)","نماز","الصلاة",
          tok_en=["prayer","pray","salat","salah"], tok_ur=["نماز","صلوٰۃ","صلاة"],
          tok_ar=["الصلاة","اقيموا","يقيمون"], tok_tr=["salat","salah"],
          phr_ar=["اقيموا الصلاة","يقيمون الصلاة"])
    topic("topic.zakat","Zakat","زکوٰۃ","الزكاة",
          tok_en=["zakat","zakah","alms","almsgiving"], tok_ur=["زکوٰۃ","زکاة"],
          tok_ar=["الزكاة","زكاة"], tok_tr=["zakat","zakah"])
    topic("topic.sadaqah_infaq","Charity / Spending (Sadaqah/Infaq)","صدقہ / انفاق","الصدقة / الإنفاق",
          tok_en=["charity","sadaqah","spend","spending","expend","alms"],
          tok_ur=["صدقہ","صدقات","انفاق","خرچ"], tok_ar=["صدقة","صدقات","انفاق","ينفقون","انفقوا"],
          neg_tok_en=["inheritance"], neg_tok_ar=["ميراث"])
    topic("topic.fasting","Fasting / Ramadan","روزہ / رمضان","الصيام",
          tok_en=["fasting","fast","ramadan","sawm","saum"], tok_ur=["روزہ","رمضان","صوم"],
          tok_ar=["الصيام","صيام","رمضان"], tok_tr=["sawm","saum","ramadan"])
    topic("topic.hajj","Hajj / Pilgrimage","حج","الحج",
          tok_en=["hajj","pilgrimage"], tok_ur=["حج"], tok_ar=["الحج","حج"], tok_tr=["hajj"])
    topic("topic.umrah","Umrah","عمرہ","عمرة",
          tok_en=["umrah","umra"], tok_ur=["عمرہ"], tok_ar=["عمرة"], tok_tr=["umrah"])
    topic("topic.dua","Supplication (Dua)","دعا","الدعاء",
          tok_en=["supplication","invocation","dua","duaa","implore"], tok_ur=["دعا","دعائیں"],
          tok_ar=["دعاء","ادعوا","يدعون","ربنا"], tok_tr=["dua","duaa"],
          phr_ar=["ربنا","قال رب"])
    topic("topic.dhikr","Remembrance (Dhikr)","ذکر","الذكر",
          tok_en=["remembrance","remember","dhikr","zikr"], tok_ur=["ذکر","یاد"],
          tok_ar=["ذكر","اذكروا","اذكر"], tok_tr=["dhikr","zikr"])

    # Aqeedah + eschatology
    topic("topic.tawhid","Oneness of Allah (Tawhid)","توحید","التوحيد",
          tok_en=["oneness","alone"], tok_ur=["توحید"], tok_ar=["احد"], tok_tr=["tawhid"],
          phr_ar=["قل هو الله احد"])
    topic("topic.shirk","Shirk (Associating partners)","شرک","الشرك",
          tok_en=["shirk","associate","partners"], tok_ur=["شرک"], tok_ar=["شرك","مشركين"], tok_tr=["shirk"])
    topic("topic.iman","Faith (Iman)","ایمان","الإيمان",
          tok_en=["faith","believe","belief","iman"], tok_ur=["ایمان","مومن","مؤمن"], tok_ar=["ايمان","آمنوا","مؤمنين"], tok_tr=["iman"])
    topic("topic.kufr","Disbelief (Kufr)","کفر","الكفر",
          tok_en=["disbelief","unbelief","kufr"], tok_ur=["کفر","کافر"], tok_ar=["كفر","كافرين","كافرون"], tok_tr=["kufr"])
    topic("topic.nifaq","Hypocrisy (Nifaq)","نفاق","النفاق",
          tok_en=["hypocrite","hypocrisy","munafiq"], tok_ur=["منافق","نفاق"], tok_ar=["منافقون","منافقين"], tok_tr=["munafiq"])
    topic("topic.akhirah","Hereafter (Akhirah)","آخرت","الآخرة",
          tok_en=["hereafter","afterlife","akhirah"], tok_ur=["آخرت"], tok_ar=["الاخرة"], tok_tr=["akhirah"])
    topic("topic.qiyamah","Day of Judgment / Resurrection","قیامت","القيامة",
          tok_en=["resurrection","judgment","judgement"], tok_ur=["قیامت","حساب"], tok_ar=["القيامة","حساب"],
          phr_ar=["يوم القيامة"], phr_en=["day judgment","day resurrection"])
    topic("topic.jannah","Paradise (Jannah)","جنت","الجنة",
          tok_en=["paradise","jannah","gardens"], tok_ur=["جنت"], tok_ar=["الجنة","جنات"], tok_tr=["jannah"])
    topic("topic.jahannam","Hell (Jahannam)","جہنم","جهنم",
          tok_en=["hell","hellfire","jahannam"], tok_ur=["جہنم","دوزخ"], tok_ar=["جهنم","النار","سعير","جحيم","سقر"], tok_tr=["jahannam"])
    topic("topic.angels","Angels","فرشتے","الملائكة",
          tok_en=["angel","angels","malaika"], tok_ur=["فرشتہ","فرشتے","ملائکہ"], tok_ar=["ملائكة","الملائكة"], tok_tr=["malaika"])
    topic("topic.jinn","Jinn","جن","الجن",
          tok_en=["jinn","djinn","genie"], tok_ur=["جن","جنات"], tok_ar=["جن","الجن"], tok_tr=["jinn"])
    topic("topic.shaytan","Satan / Shaytan","شیطان / ابلیس","الشيطان / إبليس",
          tok_en=["satan","shaytan","iblis"], tok_ur=["شیطان","ابلیس"], tok_ar=["شيطان","الشيطان","إبليس","ابليس"], tok_tr=["shaytan","iblis"])

    # Akhlaq — virtues (expanded)
    topic("topic.sabr","Patience (Sabr)","صبر","الصبر",
          tok_en=["patience","steadfast"], tok_ur=["صبر","صابر"], tok_ar=["صبر","الصابرين","اصبروا","صابرين"])
    topic("topic.shukr","Gratitude (Shukr)","شکر","الشكر",
          tok_en=["gratitude","thankful","thanks"], tok_ur=["شکر","شکور"], tok_ar=["شكر","اشكروا","شاكرين"])
    topic("topic.adl_qist","Justice (Adl/Qist)","عدل","العدل",
          tok_en=["justice","equity"], tok_ur=["عدل","انصاف"], tok_ar=["عدل","قسط","اقسطوا"])
    topic("topic.rahmah","Mercy (Rahmah)","رحمت","الرحمة",
          tok_en=["mercy","compassion"], tok_ur=["رحمت"], tok_ar=["رحمة","رحمن","رحيم"])
    topic("topic.amanah","Trust (Amanah)","امانت","الأمانة",
          tok_en=["trust","trustworthiness"], tok_ur=["امانت"], tok_ar=["امانة"])
    topic("topic.shura","Consultation (Shura)","مشورہ","الشورى",
          tok_en=["consultation"], tok_ur=["مشورہ","شوریٰ"], tok_ar=["شورى"])
    topic("topic.afw_maghfirah","Forgiveness","معافی","المغفرة",
          tok_en=["forgive","forgiveness","pardon"], tok_ur=["معافی","بخش"], tok_ar=["غفر","مغفرة","غفور","رحيم"])

    # Akhlaq — vices (expanded, stricter negatives possible)
    topic("topic.kibr","Arrogance / Pride","تکبر","الكبر",
          tok_en=["arrogance","pride","proud"], tok_ur=["تکبر","غرور"], tok_ar=["كبر","متكبر"])
    topic("topic.hasad","Envy (Hasad)","حسد","الحسد",
          tok_en=["envy","jealousy"], tok_ur=["حسد"], tok_ar=["حسد"])
    topic("topic.ghibah_buhtan","Backbiting / Slander","غیبت / بہتان","الغيبة",
          tok_en=["backbiting","slander","defamation","gossip"], tok_ur=["غیبت","بہتان"], tok_ar=["غيبة","بهتان"])
    topic("topic.kidhb","Lying / Falsehood","جھوٹ","الكذب",
          tok_en=["lying","falsehood"], tok_ur=["جھوٹ"], tok_ar=["كذب","كاذبين"])
    topic("topic.fasad","Corruption (Fasad)","فساد","الفساد",
          tok_en=["corruption","mischief"], tok_ur=["فساد"], tok_ar=["فساد","مفسدين"])

    # Fiqh / muamalat / hudud etc (expanded)
    topic("topic.riba","Riba / Usury","سود","الربا",
          tok_en=["riba","usury","interest"], tok_ur=["سود","ربا"], tok_ar=["الربا"])
    topic("topic.inheritance","Inheritance / Wills","وراثت / وصیت","الميراث",
          tok_en=["inheritance","heirs","will"], tok_ur=["وراثت","وصیت","وارث"], tok_ar=["يوصيكم","وصية"])
    topic("topic.orphans","Orphans","یتیم","اليتامى",
          tok_en=["orphan","orphans"], tok_ur=["یتیم","یتیموں"], tok_ar=["يتيم","يتامى","اليتامى"])
    topic("topic.nikah","Marriage (Nikah)","نکاح","النكاح",
          tok_en=["marriage","marry","wives","husbands"], tok_ur=["نکاح","شادی"], tok_ar=["ازواج","زوج"])
    topic("topic.talaq","Divorce (Talaq)","طلاق","الطلاق",
          tok_en=["divorce"], tok_ur=["طلاق"], tok_ar=["طلاق","طلقتم"])
    topic("topic.zina","Zina / Fornication","زنا","الزنا",
          tok_en=["zina","fornication","adultery"], tok_ur=["زنا"], tok_ar=["زنا"])
    topic("topic.theft","Theft","چوری","السرقة",
          tok_en=["theft","steal","stealing"], tok_ur=["چوری"], tok_ar=["سرق","سارق"])
    topic("topic.murder","Murder / Killing","قتل","القتل",
          tok_en=["murder","kill","killing"], tok_ur=["قتل"], tok_ar=["قتل","قاتل"])
    topic("topic.qisas","Qisas / Retaliation","قصاص","القصاص",
          tok_en=["retaliation","qisas"], tok_ur=["قصاص"], tok_ar=["قصاص"])
    topic("topic.khamr","Intoxicants (Khamr)","شراب","الخمر",
          tok_en=["intoxicants","wine","alcohol"], tok_ur=["شراب","نشہ","خمر"], tok_ar=["خمر"])
    topic("topic.maysir","Gambling (Maysir)","جوا","الميسر",
          tok_en=["gambling"], tok_ur=["جوا","قمار"], tok_ar=["ميسر"])
    topic("topic.food_prohibitions","Food Prohibitions","حرام غذا","الطعام",
          tok_en=["pork","swine","carcass","blood"], tok_ur=["خنزیر","مردار","خون"], tok_ar=["خنزير","ميتة","دم","حرم"])

    # Places / sacred sites
    topic("topic.makkah","Makkah","مکہ","مكة",
          tok_en=["makkah","mecca","bakkah"], tok_ur=["مکہ"], tok_ar=["مكة","بكة"], tok_tr=["makkah","mecca","bakkah"])
    topic("topic.madinah","Madinah / Yathrib","مدینہ","المدينة",
          tok_en=["madinah","medina","yathrib"], tok_ur=["مدینہ","یثرب"], tok_ar=["المدينة","مدينة","يثرب"], tok_tr=["madinah","medina","yathrib"])
    topic("topic.kaaba","Kaaba","کعبہ","الكعبة",
          tok_en=["kaaba","ka'bah","kabah","ka'ba"], tok_ur=["کعبہ","کعبه"], tok_ar=["الكعبة","كعبة"], tok_tr=["kaaba","kaba"])
    topic("topic.masjid_haram","Sacred Mosque (Masjid al-Haram)","مسجد الحرام","المسجد الحرام",
          phr_en=["masjid al haram","al masjid al haram","sacred mosque","holy mosque"],
          phr_ur=["مسجد الحرام"], phr_ar=["المسجد الحرام"], phr_tr=["masjid al haram","al masjid al haram"])
    topic("topic.masjid_aqsa","Al-Aqsa Mosque","مسجد اقصیٰ","المسجد الأقصى",
          phr_en=["masjid al aqsa","al aqsa","aqsa mosque"], phr_ur=["مسجد اقصیٰ","اقصیٰ"],
          phr_ar=["المسجد الاقصى","المسجد الأقصى","الاقصى"], phr_tr=["masjid al aqsa","al aqsa"])
    topic("topic.safa_marwah","Safa & Marwah","صفا و مروہ","الصفا والمروة",
          tok_en=["safa","marwah"], tok_ur=["صفا","مروہ"], tok_ar=["الصفا","المروة"],
          phr_en=["safa and marwah"], phr_ar=["الصفا والمروة"])
    topic("topic.arafat","Arafat","عرفات","عرفات",
          tok_en=["arafat","arafah"], tok_ur=["عرفات","عرفہ"], tok_ar=["عرفات"], tok_tr=["arafat","arafah"])
    topic("topic.mashar_haram","Mash'ar al-Haram","مشعر الحرام","المشعر الحرام",
          phr_en=["mashar al haram","mash'ar al haram","sacred monument"], phr_ur=["مشعر الحرام"],
          phr_ar=["المشعر الحرام"], phr_tr=["mashar al haram","mash'ar al haram"])
    topic("topic.sinai","Mount Sinai (Tur)","طور سیناء","طور سيناء",
          tok_en=["sinai","tur","tuur"], tok_ur=["طور","سیناء"], tok_ar=["طور","سيناء"],
          phr_en=["mount sinai","mount tur"], phr_ar=["طور سيناء"], phr_tr=["mount sinai","mount tur"])
    topic("topic.egypt","Egypt (Misr)","مصر","مصر",
          tok_en=["egypt","misr"], tok_ur=["مصر"], tok_ar=["مصر"], tok_tr=["misr"])
    topic("topic.babylon","Babylon","بابل","بابل",
          tok_en=["babylon","babel"], tok_ur=["بابل"], tok_ar=["بابل"], tok_tr=["babylon"])

    # Multi-token phrase topics (phrase-aware)
    topic("topic.ahl_al_kitab","People of the Book","اہلِ کتاب","أهل الكتاب",
          phr_ar=["اهل الكتاب"], phr_en=["people book","people scripture"])
    topic("topic.bani_israel","Children of Israel","بنی اسرائیل","بني إسرائيل",
          phr_ar=["بني اسرائيل"], phr_en=["children israel"])
    topic("people.lut","People of Lut","قوم لوط","قوم لوط",
          tok_en=["lut"], tok_ur=["لوط"], tok_ar=["لوط"], tok_tr=["lut","loot"],
          phr_ar=["قوم لوط"], phr_en=["people lut","people of lut","people of lot"])

    # Entities/stories: auto-generated downstream (precision policy)
    # This is handled in code below from the PROPHETS+FIGURES list.

    return T

# -----------------------------
# Main build pipeline
# -----------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    default_localized = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "topics", "topic_triggers_ur_ar_v1.json"))
    if not os.path.exists(default_localized):
        default_localized = ""
    ap.add_argument("--input", required=True, help="path to quran_complete.json")
    ap.add_argument("--out", required=True, help="output dir, e.g. ./askquran_dataset_v2")
    ap.add_argument("--localized-triggers", default=default_localized, help="optional localized trigger JSON")
    ap.add_argument("--cap-topic", type=int, default=2000, help="max verse_keys per topic (0 disables cap)")
    ap.add_argument("--cap-entity", type=int, default=3000, help="max verse_keys per entity/story (0 disables cap)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    topics_dir = os.path.join(out_dir, "topics")
    artifacts_dir = os.path.join(out_dir, "artifacts")
    tools_dir = os.path.join(out_dir, "tools")
    os.makedirs(topics_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(tools_dir, exist_ok=True)

    # Load corpus
    with open(args.input, "r", encoding="utf-8") as f:
        verses = json.load(f)
    verse_by_key = {v["verse_key"]: v for v in verses}
    ordered_keys = sorted(verse_by_key.keys(), key=vk_to_tuple)
    cap_topic = int(args.cap_topic)
    cap_entity = int(args.cap_entity)

    # Precompute token sequences + sets
    per_verse = {}
    en_all=set(); ur_all=set(); tr_all=set(); ar_all=set()
    for vk in ordered_keys:
        v = verse_by_key[vk]
        en_seq = tokenize_lat_list(all_en_text(v))
        ur_seq = tokenize_ur_list(all_ur_text(v))
        tr_seq = tokenize_lat_list((v.get("transliteration","") or "") + " " + (v.get("transliteration_alt","") or ""))
        ar_seq = tokenize_ar_list(v.get("arabic","") or "")
        per_verse[vk] = {"seq":{"en":en_seq,"ur":ur_seq,"tr":tr_seq,"ar":ar_seq},
                         "set":{"en":set(en_seq),"ur":set(ur_seq),"tr":set(tr_seq),"ar":set(ar_seq)}}
        en_all.update(en_seq); ur_all.update(ur_seq); tr_all.update(tr_seq); ar_all.update(ar_seq)

    def token_exists(tok: str, lang: str) -> bool:
        if not tok: return False
        if lang=="ar": return tok in ar_all
        if lang=="ur": return tok in ur_all
        if lang=="tr": return tok in tr_all
        return tok in en_all

    def normalize_token_list(lst: List[str], lang: str) -> List[str]:
        out=[]
        for t in (lst or []):
            t=t.strip()
            if not t: continue
            if lang=="ar": tn=norm_ar(t)
            elif lang=="ur": tn=norm_ur(t)
            else:
                tn=norm_lat(t)
                if tn in STOP_EN: continue
            if tn and token_exists(tn, lang):
                out.append(tn)
        return sorted(set(out), key=lambda x:x.casefold())

    def normalize_phrase_list(lst: List[str], lang: str) -> List[str]:
        out=[]
        for p in (lst or []):
            pts = phrase_to_tokens(p, lang)
            if not pts: continue
            # must match at least one verse
            ok=False
            for vk in ordered_keys:
                if ordered_match(per_verse[vk]["seq"][lang], pts, max_gap=2):
                    ok=True; break
            if ok:
                out.append(" ".join(pts))
        return sorted(set(out), key=lambda x:x.casefold())

    # Build seed variants (and prune to corpus)
    seed = build_seed_variants()
    def variant_exists_surface(v: str) -> bool:
        v=v.strip()
        if not v: return False
        if " " in v:
            for lang in ("en","tr","ar","ur"):
                pts = phrase_to_tokens(v, lang)
                if not pts: continue
                for vk in ordered_keys:
                    if ordered_match(per_verse[vk]["seq"][lang], pts, max_gap=2):
                        return True
            return False
        if re.search(r"[\u0600-\u06FF]", v):
            return (norm_ar(v) in ar_all) or (norm_ur(v) in ur_all)
        vv = norm_lat(v)
        return (vv in en_all) or (vv in tr_all)

    seed_filtered={}
    for canon, vars_ in seed.items():
        kept=[vv for vv in (vars_ or []) if variant_exists_surface(vv)]
        seed_filtered[canon]=uniq_sorted_casefold(kept)

    with open(os.path.join(topics_dir,"seed_variants_v2.json"),"w",encoding="utf-8") as f:
        json.dump(seed_filtered, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Variant map (capped)
    ar_norm_to_surface = collections.defaultdict(set)
    for vk in ordered_keys:
        raw = verse_by_key[vk].get("arabic","") or ""
        for tok in re.split(r"\s+", raw):
            if not tok: continue
            n = norm_ar(tok)
            if n and len(n) >= 2:
                ar_norm_to_surface[n].add(tok)

    MAX_VARIANTS = 16
    variants_out={}
    for canon, vars_ in seed_filtered.items():
        out=list(vars_)
        if re.search(r"[\u0600-\u06FF]", canon):
            out.extend(sorted(ar_norm_to_surface.get(canon,set())))
        out=uniq_sorted_casefold(out)
        if len(out) > MAX_VARIANTS:
            out = out[:MAX_VARIANTS]
        variants_out[canon]=out

    variant_map={
        "schema":"askquran_variant_map_v2",
        "version":2,
        "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "max_variants_per_key": MAX_VARIANTS,
        "variants": dict(sorted(variants_out.items(), key=lambda kv: kv[0])),
    }
    with open(os.path.join(artifacts_dir,"variant_map_v2.json"),"w",encoding="utf-8") as f:
        json.dump(variant_map, f, ensure_ascii=False, indent=2)

    variants_loaded = variant_map["variants"]
    def expand_token(tok_norm: str, lang: str) -> List[str]:
        out=[tok_norm]
        if tok_norm in variants_loaded:
            for vv in variants_loaded[tok_norm]:
                out.append(norm_ar(vv) if lang=="ar" else (norm_ur(vv) if lang=="ur" else norm_lat(vv)))
        out=[x for x in out if x]
        return sorted(set(out), key=lambda x:x.casefold())

    # Build ontology (normalize triggers, drop missing)
    base = ontology_specs()
    localized = load_localized_triggers(args.localized_triggers)
    if localized:
        merge_localized_triggers(base, localized)

    def pick_first(values: List[str]) -> str:
        for v in (values or []):
            vv = str(v).strip()
            if vv:
                return vv
        return ""

    def with_canon(items: List[str], canon: str) -> List[str]:
        out = list(items or [])
        canon_val = canon.replace("_"," ").strip()
        if canon_val and canon_val not in out:
            out.append(canon_val)
        return out

    # Expand entity/story topics (broad coverage, but precision policy applied later)
    for entry in ENTITY_DEFS:
        canon = str(entry.get("canon","")).strip()
        if not canon:
            continue
        en_vars = with_canon(entry.get("en") or [], canon)
        ar_vars = entry.get("ar") or []
        ur_vars = entry.get("ur") or []
        tr_vars = entry.get("tr") or en_vars

        en_tok, en_phr = split_variants(en_vars)
        ar_tok, ar_phr = split_variants(ar_vars)
        ur_tok, ur_phr = split_variants(ur_vars)
        tr_tok, tr_phr = split_variants(tr_vars)

        name_en = entry.get("title_en") or (en_vars[0].title() if en_vars else canon.replace("_"," ").title())
        name_ar = pick_first(ar_vars)
        name_ur = pick_first(ur_vars) or name_ar

        base.append({
            "id": f"entity.{canon}",
            "kind": "entity",
            "names": {"en": name_en, "ur": name_ur, "ar": name_ar},
            "match": {
                "any_tokens":{"en":en_tok, "ur":ur_tok, "ar":ar_tok, "tr":tr_tok},
                "any_phrases":{"en":en_phr, "ur":ur_phr, "ar":ar_phr, "tr":tr_phr}
            },
            "negative": {"any_tokens":{"en":[], "ur":[], "ar":[], "tr":[]},
                         "any_phrases":{"en":[], "ur":[], "ar":[], "tr":[]}},
            "groups": None
        })
        base.append({
            "id": f"story.{canon}",
            "kind": "story",
            "names": {"en": f"Mentions of {name_en}", "ur": name_ur, "ar": name_ar},
            "match": {
                "any_tokens":{"en":en_tok, "ur":ur_tok, "ar":ar_tok, "tr":tr_tok},
                "any_phrases":{"en":en_phr, "ur":ur_phr, "ar":ar_phr, "tr":tr_phr}
            },
            "negative": {"any_tokens":{"en":[], "ur":[], "ar":[], "tr":[]},
                         "any_phrases":{"en":[], "ur":[], "ar":[], "tr":[]}},
            "groups": [
                {"id":"passages","title":{"en":"Key passages","ur":"","ar":""}},
                {"id":"mentions","title":{"en":"Other mentions","ur":"","ar":""}},
            ]
        })

    topics=[]
    for t in base:
        t["match"]["any_tokens"]["en"] = normalize_token_list(t["match"]["any_tokens"]["en"], "en")
        t["match"]["any_tokens"]["ur"] = normalize_token_list(t["match"]["any_tokens"]["ur"], "ur")
        t["match"]["any_tokens"]["ar"] = normalize_token_list(t["match"]["any_tokens"]["ar"], "ar")
        t["match"]["any_tokens"]["tr"] = normalize_token_list(t["match"]["any_tokens"]["tr"], "tr")

        t["match"]["any_phrases"]["en"] = normalize_phrase_list(t["match"]["any_phrases"]["en"], "en")
        t["match"]["any_phrases"]["ur"] = normalize_phrase_list(t["match"]["any_phrases"]["ur"], "ur")
        t["match"]["any_phrases"]["ar"] = normalize_phrase_list(t["match"]["any_phrases"]["ar"], "ar")
        t["match"]["any_phrases"]["tr"] = normalize_phrase_list(t["match"]["any_phrases"]["tr"], "tr")

        t["negative"]["any_tokens"]["en"] = normalize_token_list(t["negative"]["any_tokens"]["en"], "en")
        t["negative"]["any_tokens"]["ur"] = normalize_token_list(t["negative"]["any_tokens"]["ur"], "ur")
        t["negative"]["any_tokens"]["ar"] = normalize_token_list(t["negative"]["any_tokens"]["ar"], "ar")
        t["negative"]["any_tokens"]["tr"] = normalize_token_list(t["negative"]["any_tokens"]["tr"], "tr")

        t["negative"]["any_phrases"]["en"] = normalize_phrase_list(t["negative"]["any_phrases"]["en"], "en")
        t["negative"]["any_phrases"]["ur"] = normalize_phrase_list(t["negative"]["any_phrases"]["ur"], "ur")
        t["negative"]["any_phrases"]["ar"] = normalize_phrase_list(t["negative"]["any_phrases"]["ar"], "ar")
        t["negative"]["any_phrases"]["tr"] = normalize_phrase_list(t["negative"]["any_phrases"]["tr"], "tr")

        # drop empty topics
        has_any = False
        for k in ("any_tokens","any_phrases"):
            for lang in ("en","ur","ar","tr"):
                if t["match"][k][lang]:
                    has_any=True
                    break
            if has_any: break
        if has_any:
            topics.append(t)

    topics.sort(key=lambda x: x["id"])
    ontology = {"schema":"askquran_topic_ontology_v2","version":2,"topics":topics}
    with open(os.path.join(topics_dir,"topic_ontology_v2.yaml"),"w",encoding="utf-8") as f:
        yaml.safe_dump(ontology, f, allow_unicode=True, sort_keys=False)

    # Compile topic pack
    def neg_hit(t, vk):
        seq = per_verse[vk]["seq"]; st = per_verse[vk]["set"]
        for lang in ("en","ur","ar","tr"):
            for tok in t["negative"]["any_tokens"][lang]:
                if tok in st[lang]:
                    return True
        for lang in ("en","ur","ar","tr"):
            for ph in t["negative"]["any_phrases"][lang]:
                if ordered_match(seq[lang], ph.split(" "), max_gap=2):
                    return True
        return False

    def pos_hit(t, vk, langs):
        seq = per_verse[vk]["seq"]; st = per_verse[vk]["set"]
        for lang in langs:
            for tok in t["match"]["any_tokens"][lang]:
                for cand in expand_token(tok, lang):
                    if cand in st[lang]:
                        return True
            for ph in t["match"]["any_phrases"][lang]:
                if ordered_match(seq[lang], ph.split(" "), max_gap=2):
                    return True
        return False

    def cap_sorted(vks: List[str], cap: int) -> List[str]:
        vks_sorted = sorted(set(vks), key=vk_to_tuple)
        if cap <= 0:
            return vks_sorted
        return vks_sorted if len(vks_sorted) <= cap else vks_sorted[:cap]

    compiled=[]
    for t in topics:
        kind = t.get("kind","topic")
        langs = ("ar","tr") if kind in ("entity","story") else ("en","ur","ar","tr")

        matched=[]
        for vk in ordered_keys:
            if pos_hit(t, vk, langs) and not neg_hit(t, vk):
                matched.append(vk)

        cap = cap_entity if kind in ("entity","story") else cap_topic
        verse_keys = cap_sorted(matched, cap)

        groups_out=None
        if t.get("groups"):
            blocks=[]; cur=[]; prev=None
            for k in verse_keys:
                s,a = vk_to_tuple(k)
                if prev and prev[0]==s and a==prev[1]+1:
                    cur.append(k)
                else:
                    if cur: blocks.append(cur)
                    cur=[k]
                prev=(s,a)
            if cur: blocks.append(cur)
            blocks.sort(key=lambda b:(-len(b), vk_to_tuple(b[0])))
            passages = cap_sorted(list(itertools.chain.from_iterable(blocks[:6])), 180)
            pset=set(passages)
            mentions = cap_sorted([k for k in verse_keys if k not in pset], 320)
            groups_out=[
                {"id":"passages","title":t["groups"][0]["title"],"verse_keys":passages},
                {"id":"mentions","title":t["groups"][1]["title"],"verse_keys":mentions},
            ]

        compiled.append({
            "id": t["id"],
            "kind": kind,
            "names": t["names"],
            "match": t["match"],
            "negative": t["negative"],
            "verse_keys": verse_keys,
            "groups": groups_out
        })

    topic_pack = {
        "schema":"askquran_topic_pack_v2",
        "version":2,
        "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "topics": compiled
    }
    with open(os.path.join(artifacts_dir,"topic_pack_v2.json"),"w",encoding="utf-8") as f:
        json.dump(topic_pack, f, ensure_ascii=False, indent=2)

    # Answer templates (citations-only)
    templates=[]
    for t in compiled:
        if t.get("groups"):
            templates.append({
                "id": t["id"],
                "title": t["names"],
                "sections":[{"id":g["id"],"title":g["title"],"verse_keys":g["verse_keys"]} for g in t["groups"]]
            })
    answer_templates={
        "schema":"askquran_answer_templates_v2",
        "version":2,
        "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
        "templates": sorted(templates, key=lambda x:x["id"])
    }
    with open(os.path.join(artifacts_dir,"answer_templates_v2.json"),"w",encoding="utf-8") as f:
        json.dump(answer_templates, f, ensure_ascii=False, indent=2)

    # Copy input corpus into output folder (locked dataset)
    shutil.copy2(args.input, os.path.join(out_dir, "quran_complete.json"))

    # Validator tool
    validate_py = r'''#!/usr/bin/env python3
import os, json, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def vk_to_tuple(vk: str):
    s,a = vk.split(":"); return (int(s), int(a))
def is_sorted(lst):
    return all(vk_to_tuple(lst[i]) <= vk_to_tuple(lst[i+1]) for i in range(len(lst)-1))
with open(os.path.join(ROOT,"quran_complete.json"),"r",encoding="utf-8") as f:
    verses=json.load(f)
verse_by_key={v["verse_key"]:v for v in verses}
with open(os.path.join(ROOT,"artifacts","topic_pack_v2.json"),"r",encoding="utf-8") as f:
    tp=json.load(f)
topics=tp["topics"]
ids=[t["id"] for t in topics]
dups=[k for k,v in collections.Counter(ids).items() if v>1]
if dups: raise SystemExit(f"Duplicate topic ids: {dups}")
for t in topics:
    if not is_sorted(t["verse_keys"]): raise SystemExit(f"Unsorted verse_keys in {t['id']}")
    for vk in t["verse_keys"]:
        if vk not in verse_by_key: raise SystemExit(f"Unknown verse_key {vk} in {t['id']}")
print(f"OK: {len(topics)} topics; all verse_keys valid/sorted.")
'''
    with open(os.path.join(tools_dir,"validate_dataset_v2.py"),"w",encoding="utf-8") as f:
        f.write(validate_py)

    # README
    readme = f"""AskQuran Offline Dataset v2 (FINAL)

Outputs:
- topics/seed_variants_v2.json
- topics/topic_ontology_v2.yaml
- artifacts/variant_map_v2.json
- artifacts/topic_pack_v2.json
- artifacts/answer_templates_v2.json

Key properties:
- deterministic compilation
- phrase-aware matching (ordered, max_gap=2)
- negative triggers supported
- entity/story precision: Arabic+transliteration evidence only
- corpus-verified triggers only (dropped if absent)
- localized triggers supported (Arabic/Urdu map optional)
- configurable caps for topic/entity verse lists

Build date (UTC): {datetime.datetime.utcnow().replace(microsecond=0).isoformat()}Z
"""
    with open(os.path.join(out_dir,"README.txt"),"w",encoding="utf-8") as f:
        f.write(readme)

    # SHA256SUMS
    rels = [
        "quran_complete.json",
        "topics/seed_variants_v2.json",
        "topics/topic_ontology_v2.yaml",
        "artifacts/variant_map_v2.json",
        "artifacts/topic_pack_v2.json",
        "artifacts/answer_templates_v2.json",
        "tools/validate_dataset_v2.py",
        "README.txt",
    ]
    lines=[]
    for rel in rels:
        p=os.path.join(out_dir, rel)
        lines.append(f"{sha256_file(p)}  {rel}")
    lines.sort()
    with open(os.path.join(out_dir,"SHA256SUMS.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ZIP
    zip_path = os.path.join(os.path.dirname(out_dir), "askquran_offline_dataset_v2_FINAL.zip")
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn.endswith((".json",".yaml",".txt",".py")):
                    ap = os.path.join(root, fn)
                    rp = os.path.relpath(ap, out_dir)
                    z.write(ap, arcname=rp)

    print("DONE")
    print(f"out_dir:  {out_dir}")
    print(f"zip:      {zip_path}")
    print(f"topics:   {len(compiled)}")
    print(f"variants: {len(variant_map['variants'])}")

if __name__ == "__main__":
    main()
