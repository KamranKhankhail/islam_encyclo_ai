from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

from schema_validate import validate_json


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer_templates", help="Path to answer_templates_v1.json")
    args = ap.parse_args()

    path = Path(args.answer_templates)
    data = _load_json(path)
    validate_json("schemas/askquran_answer_templates_v1.schema.json", data)

    templates = data.get("templates") or []
    if not isinstance(templates, list):
        raise ValueError("templates must be a list")

    for tmpl in templates:
        tid = str(tmpl.get("id") or "")
        sections = tmpl.get("sections") or []
        if not isinstance(sections, list):
            raise ValueError(f"Template {tid} sections must be a list")
        for sec in sections:
            sid = str(sec.get("id") or "")
            keys = sec.get("verse_keys") or []
            if not isinstance(keys, list) or len(keys) == 0:
                raise ValueError(f"Section {tid}:{sid} has no verse_keys")
            if len(keys) != len(set(keys)):
                raise ValueError(f"Section {tid}:{sid} has duplicate verse_keys")

    print("answer_templates_v1 verified")


if __name__ == "__main__":
    main()
