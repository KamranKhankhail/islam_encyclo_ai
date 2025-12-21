from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from schema_validate import validate_json


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("topic_pack", help="Path to topic_pack_v1.json")
    ap.add_argument("--in", dest="input_path", required=True, help="Path to quran_complete.json")
    args = ap.parse_args()

    pack_path = Path(args.topic_pack)
    input_path = Path(args.input_path)

    pack = _load_json(pack_path)
    validate_json("schemas/askquran_topic_pack_v1.schema.json", pack)

    verses = _load_json(input_path)
    verse_keys: Set[str] = {str(v.get("verse_key") or v.get("id") or "") for v in verses}

    topics = pack.get("topics") or []
    if not isinstance(topics, list):
        raise ValueError("topics must be a list")

    for topic in topics:
        tid = str(topic.get("id") or "")
        keys = topic.get("verse_keys") or []
        if not isinstance(keys, list) or len(keys) == 0:
            raise ValueError(f"Topic {tid} has no verse_keys")
        if len(keys) > 160:
            raise ValueError(f"Topic {tid} exceeds cap: {len(keys)}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"Topic {tid} has duplicate verse_keys")
        for k in keys:
            if k not in verse_keys:
                raise ValueError(f"Topic {tid} contains unknown verse_key: {k}")

        groups = topic.get("groups") or []
        if isinstance(groups, list):
            for g in groups:
                gid = str(g.get("id") or "")
                gkeys = g.get("verse_keys") or []
                if not isinstance(gkeys, list) or len(gkeys) == 0:
                    raise ValueError(f"Group {tid}:{gid} has no verse_keys")
                if len(gkeys) > 40:
                    raise ValueError(f"Group {tid}:{gid} exceeds cap: {len(gkeys)}")
                if len(gkeys) != len(set(gkeys)):
                    raise ValueError(f"Group {tid}:{gid} has duplicate verse_keys")
                for k in gkeys:
                    if k not in verse_keys:
                        raise ValueError(f"Group {tid}:{gid} contains unknown verse_key: {k}")

    print("topic_pack_v1 verified")


if __name__ == "__main__":
    main()
