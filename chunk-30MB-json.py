#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic JSON Chunker for Windows (GUI file picker)

Goal:
- Pick ANY .json / .jsonl / .ndjson file via Windows file-browse dialog
- Write chunk files each STRICTLY < 30 MB
- Preserve the original "shape" as much as possible:
  - If top-level is a LIST: outputs multiple JSON LIST chunks (slices of items)
  - If top-level is a DICT and it has a top-level ARRAY value: outputs multiple JSON DICT chunks,
    each keeping the same dict keys/order, with only that array sliced per chunk
  - If top-level is a DICT with NO top-level arrays: outputs multiple JSON DICT chunks,
    each containing a subset of keys (still a dict)
  - If input is NDJSON/JSONL: outputs multiple .jsonl chunks
  - If top-level is a SCALAR:
      - if <= 30MB: one file
      - if a huge STRING > 30MB: splits into multiple files using a small wrapper format
        (this is the only case where exact original structure cannot be preserved)

Recommended:
- Install ijson for streaming large files (avoids loading huge JSON into RAM):
    pip install ijson

Run:
    python json_chunker_windows.py
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import decimal
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Optional streaming parser (strongly recommended) ---
try:
    import ijson  # type: ignore
except Exception:
    ijson = None


# -------------------- CONFIG --------------------
MB = 1024 * 1024
MAX_CHUNK_BYTES = 30 * MB  # strict upper bound requested
SAFETY_MARGIN_BYTES = 128 * 1024  # stay safely below 30MB
TARGET_BYTES = MAX_CHUNK_BYTES - SAFETY_MARGIN_BYTES  # internal threshold

JSON_DUMPS_KW = dict(ensure_ascii=False, separators=(",", ":"))  # compact JSON for smaller chunks


# -------------------- GUI (Windows file browse) --------------------
def pick_file_windows() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print("ERROR: tkinter is not available in this Python install:", e, file=sys.stderr)
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select JSON / JSONL file to chunk",
        filetypes=[
            ("JSON / JSONL", "*.json *.jsonl *.ndjson"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def info_box(title: str, message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        print(f"{title}: {message}")
        return

    root = tk.Tk()
    root.withdraw()
    root.update()
    messagebox.showinfo(title, message)
    root.destroy()


def error_box(title: str, message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        print(f"{title}: {message}", file=sys.stderr)
        return

    root = tk.Tk()
    root.withdraw()
    root.update()
    messagebox.showerror(title, message)
    root.destroy()


# -------------------- HELPERS --------------------
def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unique_output_dir(input_path: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0]
    parent = os.path.dirname(os.path.abspath(input_path))
    out = os.path.join(parent, f"{base}_chunks")
    if not os.path.exists(out):
        return out
    # avoid clobbering existing runs
    i = 2
    while True:
        candidate = f"{out}_{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def normalize_number(x: Any) -> Any:
    # ijson may return decimal.Decimal for numbers; json.dumps can't serialize Decimal by default.
    if isinstance(x, decimal.Decimal):
        # preserve integers exactly; floats as float (may lose precision for very large decimals)
        try:
            if x == x.to_integral_value():
                return int(x)
        except Exception:
            pass
        return float(x)
    return x


def dumps_bytes(obj: Any) -> bytes:
    return json.dumps(obj, **JSON_DUMPS_KW).encode("utf-8")


def file_size(path: str) -> int:
    return os.path.getsize(path)


@dataclass
class OutputSummary:
    output_dir: str
    files: List[str]
    strategy: str
    notes: List[str]


# -------------------- STREAM PARSER CORE (ijson) --------------------
def _parse_events(fp):
    # yields (event, value) and ignores prefix
    for _, event, value in ijson.parse(fp):  # type: ignore[attr-defined]
        yield event, value


def _skip_value(it, start_event: str) -> None:
    # consumes events for current value if it is a container
    if start_event not in ("start_map", "start_array"):
        return
    depth = 1
    while depth > 0:
        ev, _ = next(it)
        if ev in ("start_map", "start_array"):
            depth += 1
        elif ev in ("end_map", "end_array"):
            depth -= 1


def _read_value(it, first_event: str, first_value: Any) -> Any:
    # build python object from event stream for a single value
    if first_event == "null":
        return None
    if first_event == "boolean":
        return bool(first_value)
    if first_event == "number":
        return normalize_number(first_value)
    if first_event == "string":
        return first_value
    if first_event == "start_array":
        arr: List[Any] = []
        while True:
            ev, val = next(it)
            if ev == "end_array":
                break
            arr.append(_read_value(it, ev, val))
        return arr
    if first_event == "start_map":
        obj: Dict[str, Any] = {}
        while True:
            ev, val = next(it)
            if ev == "end_map":
                break
            if ev != "map_key":
                raise ValueError(f"Malformed JSON stream: expected map_key, got {ev}")
            key = val
            ev2, val2 = next(it)
            obj[key] = _read_value(it, ev2, val2)
        return obj

    raise ValueError(f"Unsupported event: {first_event}")


# -------------------- CHUNK WRITERS --------------------
class JsonArrayChunkWriter:
    def __init__(self, output_dir: str, base_name: str, ext: str = ".json") -> None:
        self.output_dir = output_dir
        self.base_name = base_name
        self.ext = ext
        self.part = 1
        self.fp = None
        self.bytes_written = 0
        self.first_item = True
        self.files: List[str] = []

    def _open_new(self) -> None:
        if self.fp:
            self._close_current()
        name = f"{self.base_name}.part{self.part:04d}{self.ext}"
        path = os.path.join(self.output_dir, name)
        self.fp = open(path, "wb")
        self.fp.write(b"[")
        self.bytes_written = 1
        self.first_item = True
        self.files.append(path)
        self.part += 1

    def _close_current(self) -> None:
        if not self.fp:
            return
        self.fp.write(b"]")
        self.bytes_written += 1
        self.fp.close()
        self.fp = None

    def add(self, item: Any) -> None:
        if not self.fp:
            self._open_new()

        item_b = dumps_bytes(item)
        sep_b = b"" if self.first_item else b","

        # ensure closing bracket fits too
        projected = self.bytes_written + len(sep_b) + len(item_b) + 1
        if (not self.first_item) and projected > TARGET_BYTES:
            # rotate file
            self._open_new()
            item_b = dumps_bytes(item)
            sep_b = b""

        # If a single item doesn't fit into an empty chunk, we still write it (cannot split without changing structure)
        self.fp.write(sep_b)
        self.fp.write(item_b)
        self.bytes_written += len(sep_b) + len(item_b)
        self.first_item = False

    def finish(self) -> None:
        if self.fp:
            self._close_current()


class JsonDictWithArrayChunkWriter:
    """
    Writes chunks of a dict that contains one target array key.
    All other keys are repeated in each chunk.
    """
    def __init__(
        self,
        output_dir: str,
        base_name: str,
        key_order: List[str],
        fixed_values: Dict[str, Any],
        target_array_key: str,
    ) -> None:
        self.output_dir = output_dir
        self.base_name = base_name
        self.key_order = key_order
        self.fixed_values = fixed_values
        self.target = target_array_key

        self.part = 1
        self.fp = None
        self.bytes_written = 0
        self.first_item = True
        self.files: List[str] = []

        self.prefix, self.suffix = self._build_prefix_suffix()
        overhead = len(self.prefix) + len(self.suffix)
        if overhead >= TARGET_BYTES:
            raise ValueError(
                f"Non-array wrapper fields are too large to fit under 30MB. Overhead={overhead} bytes."
            )

    def _pair_bytes(self, k: str, v: Any) -> bytes:
        return dumps_bytes(k) + b":" + dumps_bytes(v)

    def _build_prefix_suffix(self) -> Tuple[bytes, bytes]:
        before: List[bytes] = []
        after: List[bytes] = []
        seen_target = False

        for k in self.key_order:
            if k == self.target:
                seen_target = True
                continue
            if k not in self.fixed_values:
                # should not happen; fixed_values is for keys other than target
                continue
            p = self._pair_bytes(k, self.fixed_values[k])
            if not seen_target:
                before.append(p)
            else:
                after.append(p)

        # prefix: { + before + , + "target":[
        prefix = b"{"
        if before:
            prefix += b",".join(before) + b","
        prefix += dumps_bytes(self.target) + b":["

        # suffix: ] + ,after + }
        suffix = b"]"
        if after:
            suffix += b"," + b",".join(after)
        suffix += b"}"
        return prefix, suffix

    def _open_new(self) -> None:
        if self.fp:
            self._close_current()
        name = f"{self.base_name}.part{self.part:04d}.json"
        path = os.path.join(self.output_dir, name)
        self.fp = open(path, "wb")
        self.fp.write(self.prefix)
        self.bytes_written = len(self.prefix)
        self.first_item = True
        self.files.append(path)
        self.part += 1

    def _close_current(self) -> None:
        if not self.fp:
            return
        self.fp.write(self.suffix)
        self.bytes_written += len(self.suffix)
        self.fp.close()
        self.fp = None

    def add_array_item(self, item: Any) -> None:
        if not self.fp:
            self._open_new()

        item_b = dumps_bytes(item)
        sep_b = b"" if self.first_item else b","

        projected = self.bytes_written + len(sep_b) + len(item_b) + len(self.suffix)
        if (not self.first_item) and projected > TARGET_BYTES:
            self._open_new()
            item_b = dumps_bytes(item)
            sep_b = b""

        self.fp.write(sep_b)
        self.fp.write(item_b)
        self.bytes_written += len(sep_b) + len(item_b)
        self.first_item = False

    def finish(self) -> None:
        if self.fp:
            self._close_current()


# -------------------- CHUNK STRATEGIES --------------------
def chunk_ndjson(path: str, out_dir: str) -> OutputSummary:
    base = os.path.splitext(os.path.basename(path))[0]
    safe_makedirs(out_dir)

    files: List[str] = []
    part = 1
    cur_path = os.path.join(out_dir, f"{base}.part{part:04d}.jsonl")
    fp = open(cur_path, "wb")
    cur_bytes = 0

    def rotate():
        nonlocal part, fp, cur_path, cur_bytes
        fp.close()
        files.append(cur_path)
        part += 1
        cur_path = os.path.join(out_dir, f"{base}.part{part:04d}.jsonl")
        fp = open(cur_path, "wb")
        cur_bytes = 0

    notes: List[str] = []
    try:
        with open(path, "rb") as fin:
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                # Validate JSON per line (keeps structure)
                obj = json.loads(line.decode("utf-8", errors="strict"))
                out_line = dumps_bytes(obj) + b"\n"
                if cur_bytes > 0 and cur_bytes + len(out_line) > TARGET_BYTES:
                    rotate()
                fp.write(out_line)
                cur_bytes += len(out_line)
    finally:
        fp.close()

    if os.path.exists(cur_path) and (not files or files[-1] != cur_path):
        if file_size(cur_path) > 0:
            files.append(cur_path)
        else:
            try:
                os.remove(cur_path)
            except Exception:
                pass

    # sanity check
    for f in files:
        if file_size(f) >= MAX_CHUNK_BYTES:
            notes.append(f"WARNING: {os.path.basename(f)} is >= 30MB (line too large to split).")

    return OutputSummary(out_dir, files, "ndjson", notes)


def chunk_top_level_array_stream(path: str, out_dir: str) -> OutputSummary:
    base = os.path.splitext(os.path.basename(path))[0]
    safe_makedirs(out_dir)

    writer = JsonArrayChunkWriter(out_dir, base, ext=".json")
    notes: List[str] = []

    with open(path, "rb") as fp:
        it = iter(_parse_events(fp))
        ev, val = next(it)
        if ev != "start_array":
            raise ValueError("Not a top-level JSON array.")
        while True:
            ev, val = next(it)
            if ev == "end_array":
                break
            item = _read_value(it, ev, val)
            writer.add(item)

    writer.finish()

    # sanity check
    for f in writer.files:
        if file_size(f) >= MAX_CHUNK_BYTES:
            notes.append(f"WARNING: {os.path.basename(f)} is >= 30MB (single item too large).")

    return OutputSummary(out_dir, writer.files, "top_level_array", notes)


def find_first_top_level_array_key_stream(path: str) -> Optional[str]:
    with open(path, "rb") as fp:
        it = iter(_parse_events(fp))
        ev, _ = next(it)
        if ev != "start_map":
            return None
        while True:
            ev, val = next(it)
            if ev == "end_map":
                return None
            if ev != "map_key":
                continue
            key = val
            ev2, _v2 = next(it)
            if ev2 == "start_array":
                return key
            _skip_value(it, ev2)


def build_wrapper_skip_target_array(
    path: str, target_key: str
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parses the top-level dict, builds all values EXCEPT target_key's array, which is skipped.
    Returns: (key_order, fixed_values) where fixed_values excludes target_key.
    """
    key_order: List[str] = []
    fixed: Dict[str, Any] = {}

    with open(path, "rb") as fp:
        it = iter(_parse_events(fp))
        ev, _ = next(it)
        if ev != "start_map":
            raise ValueError("Not a top-level JSON object (dict).")

        while True:
            ev, val = next(it)
            if ev == "end_map":
                break
            if ev != "map_key":
                continue
            key = val
            key_order.append(key)
            ev2, v2 = next(it)
            if key == target_key and ev2 == "start_array":
                _skip_value(it, ev2)
            else:
                fixed[key] = _read_value(it, ev2, v2)

    # Ensure target key exists in order (even if not in fixed)
    if target_key not in key_order:
        raise ValueError(f"Target key '{target_key}' not found in the top-level object.")

    # Remove target key from fixed_values (writer expects only non-target fixed keys)
    if target_key in fixed:
        fixed.pop(target_key, None)

    return key_order, fixed


def stream_target_array_items(path: str, target_key: str):
    """
    Generator that yields items of the array value at top-level key == target_key.
    """
    with open(path, "rb") as fp:
        it = iter(_parse_events(fp))
        ev, _ = next(it)
        if ev != "start_map":
            raise ValueError("Not a top-level JSON object (dict).")

        while True:
            ev, val = next(it)
            if ev == "end_map":
                break
            if ev != "map_key":
                continue
            key = val
            ev2, v2 = next(it)
            if key == target_key:
                if ev2 != "start_array":
                    raise ValueError(f"Key '{target_key}' exists but is not an array.")
                # yield items until end_array
                while True:
                    ev3, v3 = next(it)
                    if ev3 == "end_array":
                        return
                    yield _read_value(it, ev3, v3)
            else:
                _skip_value(it, ev2)


def chunk_dict_by_array_key_stream(path: str, out_dir: str, target_key: str) -> OutputSummary:
    base = os.path.splitext(os.path.basename(path))[0]
    safe_makedirs(out_dir)
    notes: List[str] = []

    key_order, fixed = build_wrapper_skip_target_array(path, target_key)
    writer = JsonDictWithArrayChunkWriter(out_dir, base, key_order, fixed, target_key)

    for item in stream_target_array_items(path, target_key):
        writer.add_array_item(item)

    writer.finish()

    for f in writer.files:
        if file_size(f) >= MAX_CHUNK_BYTES:
            notes.append(f"WARNING: {os.path.basename(f)} is >= 30MB (single item too large).")

    return OutputSummary(out_dir, writer.files, f"dict_chunk_by_top_level_array:{target_key}", notes)


def chunk_top_level_dict_by_keys_stream(path: str, out_dir: str) -> OutputSummary:
    """
    Fallback for dicts that have no top-level array: chunk by distributing key/value pairs into dict chunks.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    safe_makedirs(out_dir)
    notes: List[str] = []

    files: List[str] = []
    part = 1
    current: Dict[str, Any] = {}
    current_order: List[str] = []

    def current_bytes_estimate(d: Dict[str, Any]) -> int:
        return len(dumps_bytes(d))

    def flush():
        nonlocal part, current, current_order
        if not current_order:
            return
        out_path = os.path.join(out_dir, f"{base}.part{part:04d}.json")
        # preserve insertion order
        obj = {k: current[k] for k in current_order}
        with open(out_path, "wb") as f:
            f.write(dumps_bytes(obj))
        files.append(out_path)
        part += 1
        current = {}
        current_order = []

    with open(path, "rb") as fp:
        it = iter(_parse_events(fp))
        ev, _ = next(it)
        if ev != "start_map":
            raise ValueError("Not a top-level JSON object (dict).")

        while True:
            ev, val = next(it)
            if ev == "end_map":
                break
            if ev != "map_key":
                continue
            key = val
            ev2, v2 = next(it)
            value = _read_value(it, ev2, v2)

            # tentative add
            current[key] = value
            current_order.append(key)
            if current_bytes_estimate({k: current[k] for k in current_order}) > TARGET_BYTES:
                # rollback this key, flush, then add to new chunk
                current_order.pop()
                current.pop(key, None)
                flush()
                current[key] = value
                current_order.append(key)

                # if even alone too big, we still write it (cannot split without changing structure)
                if current_bytes_estimate({key: value}) > TARGET_BYTES:
                    notes.append(
                        f"WARNING: Single key '{key}' chunk may be >=30MB; cannot split without changing JSON semantics."
                    )

    flush()

    for f in files:
        if file_size(f) >= MAX_CHUNK_BYTES:
            notes.append(f"WARNING: {os.path.basename(f)} is >= 30MB (single value too large).")

    return OutputSummary(out_dir, files, "top_level_dict_chunk_by_keys", notes)


def chunk_scalar_fallback(path: str, out_dir: str) -> OutputSummary:
    base = os.path.splitext(os.path.basename(path))[0]
    safe_makedirs(out_dir)

    raw = open(path, "rb").read()
    # Try parse scalar (or any JSON) fully; if this succeeds and output fits, write single chunk.
    obj = json.loads(raw.decode("utf-8"))
    b = dumps_bytes(obj)
    notes: List[str] = []

    if len(b) <= TARGET_BYTES:
        out_path = os.path.join(out_dir, f"{base}.part0001.json")
        with open(out_path, "wb") as f:
            f.write(b)
        return OutputSummary(out_dir, [out_path], "scalar_single", notes)

    # Only safe split case: huge string
    if isinstance(obj, str):
        # wrapper format for chunked strings
        total_parts = math.ceil(len(obj.encode("utf-8")) / TARGET_BYTES)
        files: List[str] = []
        start = 0
        part = 1

        # split by UTF-8 bytes safely: operate in bytes and decode back per chunk
        obj_bytes = obj.encode("utf-8")
        while start < len(obj_bytes):
            end = min(start + TARGET_BYTES, len(obj_bytes))
            chunk_bytes = obj_bytes[start:end]

            # ensure valid utf-8 boundaries
            while end < len(obj_bytes):
                try:
                    chunk_str = chunk_bytes.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
                    chunk_bytes = obj_bytes[start:end]
            else:
                chunk_str = chunk_bytes.decode("utf-8", errors="replace")

            payload = {
                "__chunked__": True,
                "type": "string",
                "part": part,
                "total_parts": total_parts,
                "value_part": chunk_str,
            }
            out_path = os.path.join(out_dir, f"{base}.part{part:04d}.json")
            with open(out_path, "wb") as f:
                f.write(dumps_bytes(payload))
            files.append(out_path)

            part += 1
            start = end

        notes.append(
            "NOTE: Top-level scalar string exceeded 30MB, so it was chunked using a wrapper format "
            "(exact original scalar cannot be preserved in multiple valid JSON files)."
        )
        return OutputSummary(out_dir, files, "scalar_string_wrapper_chunks", notes)

    raise ValueError(
        "Top-level JSON is a scalar >30MB that cannot be split into multiple valid JSON files "
        "without changing structure (only huge strings are split with a wrapper)."
    )


# -------------------- NON-STREAMING FALLBACK (no ijson) --------------------
def chunk_without_ijson(path: str, out_dir: str) -> OutputSummary:
    """
    Works for 'most' JSON files but loads everything into RAM.
    Provided as a fallback when ijson is not installed.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    safe_makedirs(out_dir)
    notes = ["WARNING: ijson not installed; using json.load() (loads entire file into RAM)."]

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # NDJSON not supported well without streaming; treat only by extension upstream.
    if isinstance(obj, list):
        writer = JsonArrayChunkWriter(out_dir, base, ext=".json")
        for item in obj:
            writer.add(item)
        writer.finish()
        return OutputSummary(out_dir, writer.files, "top_level_array_no_ijson", notes)

    if isinstance(obj, dict):
        # choose first top-level list value as target array if present
        target_key = None
        for k, v in obj.items():
            if isinstance(v, list):
                target_key = k
                break

        if target_key:
            key_order = list(obj.keys())
            fixed = {k: v for k, v in obj.items() if k != target_key}
            writer = JsonDictWithArrayChunkWriter(out_dir, base, key_order, fixed, target_key)
            for item in obj[target_key]:
                writer.add_array_item(item)
            writer.finish()
            return OutputSummary(out_dir, writer.files, f"dict_chunk_by_top_level_array_no_ijson:{target_key}", notes)

        # else chunk by keys
        files: List[str] = []
        part = 1
        cur: Dict[str, Any] = {}
        for k, v in obj.items():
            cur[k] = v
            if len(dumps_bytes(cur)) > TARGET_BYTES and len(cur) > 1:
                # remove k, flush, add k to new chunk
                cur.pop(k, None)
                out_path = os.path.join(out_dir, f"{base}.part{part:04d}.json")
                with open(out_path, "wb") as f:
                    f.write(dumps_bytes(cur))
                files.append(out_path)
                part += 1
                cur = {k: v}

        if cur:
            out_path = os.path.join(out_dir, f"{base}.part{part:04d}.json")
            with open(out_path, "wb") as f:
                f.write(dumps_bytes(cur))
            files.append(out_path)

        return OutputSummary(out_dir, files, "top_level_dict_chunk_by_keys_no_ijson", notes)

    # scalar
    return chunk_scalar_fallback(path, out_dir)


# -------------------- MAIN --------------------
def main() -> None:
    path = pick_file_windows()
    if not path:
        return

    if not os.path.isfile(path):
        error_box("JSON Chunker", "Selected path is not a file.")
        return

    out_dir = unique_output_dir(path)
    safe_makedirs(out_dir)

    ext = os.path.splitext(path)[1].lower()
    is_jsonl = ext in (".jsonl", ".ndjson")

    start_ts = time.time()
    try:
        if is_jsonl:
            summary = chunk_ndjson(path, out_dir)
        else:
            if ijson is None:
                summary = chunk_without_ijson(path, out_dir)
            else:
                # Determine top-level by first event
                with open(path, "rb") as fp:
                    it = iter(_parse_events(fp))
                    ev, _ = next(it)

                if ev == "start_array":
                    summary = chunk_top_level_array_stream(path, out_dir)
                elif ev == "start_map":
                    # Prefer chunking a top-level array value if present (best preservation & best for LM Studio)
                    target_key = find_first_top_level_array_key_stream(path)
                    if target_key:
                        summary = chunk_dict_by_array_key_stream(path, out_dir, target_key)
                    else:
                        summary = chunk_top_level_dict_by_keys_stream(path, out_dir)
                else:
                    # scalar / unusual
                    summary = chunk_scalar_fallback(path, out_dir)

        # Write a simple manifest
        manifest = {
            "source_file": os.path.abspath(path),
            "output_dir": os.path.abspath(summary.output_dir),
            "strategy": summary.strategy,
            "max_chunk_mb": 30,
            "safety_margin_kb": SAFETY_MARGIN_BYTES // 1024,
            "files": [os.path.basename(p) for p in summary.files],
            "sizes_mb": [round(file_size(p) / MB, 3) for p in summary.files],
            "notes": summary.notes,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_ts

        # Prepare message
        msg_lines = [
            "Done.",
            f"Strategy: {summary.strategy}",
            f"Output folder: {summary.output_dir}",
            f"Chunks written: {len(summary.files)}",
            f"Elapsed: {elapsed:.2f}s",
            "",
            "Largest chunks:",
        ]
        # Show top 5 by size
        sizes = sorted([(file_size(p), p) for p in summary.files], reverse=True)[:5]
        for sz, p in sizes:
            msg_lines.append(f"- {os.path.basename(p)}  ({sz/MB:.2f} MB)")

        if summary.notes:
            msg_lines.append("")
            msg_lines.append("Notes / Warnings:")
            msg_lines.extend([f"- {n}" for n in summary.notes])

        info_box("JSON Chunker", "\n".join(msg_lines))

    except Exception as e:
        error_box("JSON Chunker - Error", str(e))


if __name__ == "__main__":
    main()
