from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from schema_validate import validate_json


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True, help="Path to topic_ontology_v1.yaml")
    ap.add_argument("--topic-pack", required=True, help="Path to topic_pack_v1.json")
    ap.add_argument("--out", required=True, help="Output answer_templates_v1.json path")
    args = ap.parse_args()

    ontology_path = Path(args.ontology)
    pack_path = Path(args.topic_pack)
    out_path = Path(args.out)

    ontology = _load_yaml(ontology_path)
    if not isinstance(ontology, dict):
        raise ValueError("topic_ontology_v1.yaml must be an object")

    topics = ontology.get("topics") or []
    if not isinstance(topics, list):
        raise ValueError("topic_ontology_v1.yaml topics must be a list")

    pack = _load_json(pack_path)
    validate_json("schemas/askquran_topic_pack_v1.schema.json", pack)
    pack_topics = pack.get("topics") or []
    if not isinstance(pack_topics, list):
        raise ValueError("topic_pack_v1.json topics must be a list")

    pack_by_id = {str(t.get("id")): t for t in pack_topics}

    templates: List[Dict[str, Any]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        groups = topic.get("groups") or []
        if not groups:
            continue

        tid = str(topic.get("id") or "").strip()
        if not tid:
            raise ValueError("Topic missing id in ontology")

        pack_topic = pack_by_id.get(tid)
        if not pack_topic:
            raise ValueError(f"Topic {tid} missing in topic_pack_v1.json")

        pack_groups = {str(g.get("id")): g for g in (pack_topic.get("groups") or []) if isinstance(g, dict)}

        sections: List[Dict[str, Any]] = []
        for g in groups:
            gid = str(g.get("id") or "").strip()
            title = g.get("title") or {}
            pack_group = pack_groups.get(gid)
            if not pack_group:
                raise ValueError(f"Group {tid}:{gid} missing in topic_pack_v1.json")
            sections.append(
                {
                    "id": gid,
                    "title": title,
                    "verse_keys": pack_group.get("verse_keys") or [],
                }
            )

        templates.append(
            {
                "id": tid,
                "title": topic.get("names") or {},
                "sections": sections,
            }
        )

    payload = {
        "schema": "askquran_answer_templates_v1",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": {
            "name": "ie-ai-py",
            "ontology": str(ontology_path.name),
            "topic_pack": str(pack_path.name),
        },
        "templates": templates,
    }

    validate_json("schemas/askquran_answer_templates_v1.schema.json", payload)
    _write_json(out_path, payload)


if __name__ == "__main__":
    main()
