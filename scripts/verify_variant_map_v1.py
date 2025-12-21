from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from schema_validate import validate_json


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_unique(values: List[str]) -> None:
    if len(values) != len(set(values)):
        dupes = [v for v in values if values.count(v) > 1]
        raise ValueError(f"Duplicate variants detected: {sorted(set(dupes))}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/verify_variant_map_v1.py <variant_map_v1.json>")

    path = Path(sys.argv[1])
    data = _load_json(path)

    validate_json("schemas/askquran_variant_map_v1.schema.json", data)

    variants = data.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("variants must be an object")

    for key, vals in variants.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Variant map keys must be non-empty strings")
        if not isinstance(vals, list):
            raise ValueError(f"variants[{key}] must be an array")
        if len(vals) > 12:
            raise ValueError(f"variants[{key}] exceeds cap: {len(vals)}")
        for v in vals:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"variants[{key}] contains empty string")
        _ensure_unique(vals)

    print("variant_map_v1 verified")


if __name__ == "__main__":
    main()
