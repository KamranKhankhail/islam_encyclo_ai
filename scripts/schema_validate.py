from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

import jsonschema


def _json_pointer(path: Iterable[Any]) -> str:
    parts: List[str] = []
    for p in path:
        parts.append(str(p).replace("~", "~0").replace("/", "~1"))
    return "/" + "/".join(parts) if parts else "/"


def validate_json(schema_path: str | Path, data: Any) -> None:
    schema_file = Path(schema_path)
    with schema_file.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)

    errors = sorted(
        validator.iter_errors(data),
        key=lambda e: (list(e.path), e.message),
    )
    if not errors:
        return

    messages = []
    for err in errors:
        ptr = _json_pointer(err.path)
        messages.append(f"{ptr}: {err.message}")

    raise ValueError("Schema validation failed:\n" + "\n".join(messages))


__all__ = ["validate_json"]
