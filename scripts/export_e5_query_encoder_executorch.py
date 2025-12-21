#!/usr/bin/env python3
"""
Export E5 query encoder to ExecuTorch (XNNPACK) and tokenizer JSON.

Run:
  python scripts/export_e5_query_encoder_executorch.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - explicit error for missing deps
    raise RuntimeError("Missing dependency: numpy. Install numpy before running.") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover - explicit error for missing deps
    raise RuntimeError("Missing dependency: torch. Install PyTorch before running.") from exc

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover - explicit error for missing deps
    raise RuntimeError("Missing dependency: transformers. Install transformers before running.") from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - explicit error for missing deps
    raise RuntimeError(
        "Missing dependency: sentence-transformers. Install sentence-transformers before running."
    ) from exc


MODEL_ID = "intfloat/multilingual-e5-small"
MAX_TOKENS = 128
EMBED_DIM = 384

TEST_QUERIES: List[str] = [
    "query: patience in adversity",
    "query: what does the Quran say about mercy",
    "query: \u0627\u0644\u062d\u0645\u062f \u0644\u0644\u0647",
    "query: \u0627\u0644\u0635\u0644\u0627\u0629 \u0641\u064a \u0625\u0644\u0633\u0644\u0627\u0645",
    "query: \u0635\u0628\u0631 \u06a9\u0627 \u0627\u062c\u0631 \u06a9\u06cc\u0627 \u06c1\u06d2\u061f",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(payload)


def get_package_version(package: str, attr_name: str = "__version__") -> str:
    try:
        module = __import__(package)
        version = getattr(module, attr_name, None)
        if isinstance(version, str) and version:
            return version

        try:
            import importlib.metadata as importlib_metadata

            version = importlib_metadata.version(package)
            if isinstance(version, str) and version:
                print(f"Version for {package} resolved via importlib.metadata.")
                return version
        except Exception as meta_exc:
            raise RuntimeError(
                f"Package {package} has no usable {attr_name} and importlib.metadata lookup failed."
            ) from meta_exc

        raise RuntimeError(f"Package {package} has no usable {attr_name}.")
    except Exception as exc:
        raise RuntimeError(f"Could not determine version for package: {package}.") from exc


class E5QueryEncoder(torch.nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
        masked = last_hidden * mask
        sum_emb = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean = sum_emb / denom
        norm = torch.sqrt(torch.sum(mean * mean, dim=-1, keepdim=True)).clamp(min=1e-12)
        normalized = mean / norm
        return normalized.float()


def build_tokenizer(output_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("AutoTokenizer did not load a fast tokenizer; required for tokenizer.json export.")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise RuntimeError("Tokenizer missing pad_token_id; cannot enable fixed-length padding.")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError("Fast tokenizer missing backend_tokenizer; cannot export tokenizer.json.")
    backend.enable_truncation(max_length=MAX_TOKENS)
    backend.enable_padding(length=MAX_TOKENS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(output_path))
    return tokenizer


def tokenize_queries(tokenizer, queries: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(queries),
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise RuntimeError("Tokenizer output missing input_ids or attention_mask.")
    return encoded["input_ids"], encoded["attention_mask"]


def compare_embeddings(label: str, ref: np.ndarray, cand: np.ndarray) -> None:
    if ref.shape != cand.shape:
        raise RuntimeError(f"{label}: shape mismatch {ref.shape} vs {cand.shape}.")
    if ref.ndim != 2 or ref.shape[1] != EMBED_DIM:
        raise RuntimeError(f"{label}: unexpected embedding shape {ref.shape}; expected (*, {EMBED_DIM}).")
    for idx in range(ref.shape[0]):
        a = ref[idx]
        b = cand[idx]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        max_diff = float(np.max(np.abs(a - b)))
        if cos < 0.999 and max_diff > 1e-3:
            raise RuntimeError(
                f"{label}: mismatch at index {idx} (cos={cos:.6f}, max_diff={max_diff:.6f})."
            )


def export_executorch(model: torch.nn.Module, example_inputs: Tuple[torch.Tensor, torch.Tensor], out_path: Path) -> None:
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError("torch.export.export is required (PyTorch 2.1+).")
    try:
        from executorch.exir import to_edge  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: executorch. Install ExecuTorch before exporting.") from exc

    try:
        from executorch.exir import to_edge_transform_and_lower  # type: ignore
        has_to_edge_transform = True
    except Exception:
        to_edge_transform_and_lower = None
        has_to_edge_transform = False

    partitioner = None
    try:
        from executorch.exir.backend.partitioner import XnnpackPartitioner  # type: ignore

        partitioner = XnnpackPartitioner()
        print("ExecuTorch: using XnnpackPartitioner from executorch.exir.backend.partitioner")
    except Exception:
        try:
            from executorch.backends.xnnpack.partitioner import XnnpackPartitioner  # type: ignore

            partitioner = XnnpackPartitioner()
            print("ExecuTorch: using XnnpackPartitioner from executorch.backends.xnnpack.partitioner")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ExecuTorch XNNPACK backend not available (XnnpackPartitioner import failed).") from exc

    exported = torch.export.export(model, example_inputs)
    if has_to_edge_transform:
        print("ExecuTorch: using to_edge_transform_and_lower")
        edge_prog = to_edge_transform_and_lower(exported, partitioner=[partitioner])
    else:
        print("ExecuTorch: using to_edge + transform")
        edge_prog = to_edge(exported)
        if hasattr(edge_prog, "transform"):
            edge_prog = edge_prog.transform([partitioner])
        elif hasattr(edge_prog, "transform_and_lower"):
            edge_prog = edge_prog.transform_and_lower([partitioner])
        else:
            raise RuntimeError("ExecuTorch EdgeProgramManager missing transform/transform_and_lower.")

    if not hasattr(edge_prog, "to_executorch"):
        raise RuntimeError("ExecuTorch EdgeProgramManager missing to_executorch().")
    exec_prog = edge_prog.to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(exec_prog, "write_to_file"):
        exec_prog.write_to_file(str(out_path))
    elif hasattr(exec_prog, "save"):
        exec_prog.save(str(out_path))
    else:
        raise RuntimeError("ExecuTorch program missing write_to_file/save; cannot serialize .pte.")


def run_self_test(
    model: E5QueryEncoder,
    tokenizer,
    output_path: Path,
) -> None:
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    sentence_model = SentenceTransformer(MODEL_ID, device=str(device))
    sentence_model.max_seq_length = MAX_TOKENS

    st_embeddings = sentence_model.encode(
        TEST_QUERIES,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    input_ids, attention_mask = tokenize_queries(tokenizer, TEST_QUERIES)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        torch_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

    if torch_embeddings.dtype != torch.float32:
        raise RuntimeError(f"Model output dtype must be float32, got {torch_embeddings.dtype}.")

    torch_np = torch_embeddings.cpu().numpy()
    compare_embeddings("SentenceTransformer vs torch module", st_embeddings, torch_np)

    norms = np.linalg.norm(torch_np, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1e-3:
        raise RuntimeError("Torch module output not L2-normalized to 1 within tolerance.")

    if torch_np.shape[1] != EMBED_DIM:
        raise RuntimeError(f"Unexpected embedding dim {torch_np.shape[1]} (expected {EMBED_DIM}).")

    try:
        from executorch.runtime import Runtime  # type: ignore
    except Exception:
        print("ExecuTorch runtime not available; skipping .pte runtime check.")
        return

    runtime = Runtime()
    program = runtime.load_program(str(output_path))

    input_ids_one = input_ids[:1]
    attention_mask_one = attention_mask[:1]
    with torch.no_grad():
        torch_one = model(input_ids=input_ids_one, attention_mask=attention_mask_one).cpu().numpy()

    if hasattr(program, "forward"):
        et_out = program.forward((input_ids_one, attention_mask_one))
    elif hasattr(program, "run_method"):
        et_out = program.run_method("forward", (input_ids_one, attention_mask_one))
    else:
        raise RuntimeError("ExecuTorch runtime module missing forward/run_method.")

    if isinstance(et_out, (list, tuple)):
        et_out = et_out[0]
    if isinstance(et_out, torch.Tensor):
        et_np = et_out.cpu().numpy()
    elif isinstance(et_out, np.ndarray):
        et_np = et_out
    else:
        raise RuntimeError("ExecuTorch runtime output is not a torch.Tensor or numpy.ndarray.")

    if et_np.ndim == 1:
        et_np = et_np[None, :]

    compare_embeddings("ExecuTorch runtime vs torch module", torch_one, et_np)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "output" / "processed" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output_dir / "e5_tokenizer.json"
    pte_path = output_dir / "e5_query_encoder_xnnpack.pte"
    meta_path = output_dir / "e5_query_encoder_meta.json"

    print(f"Loading model: {MODEL_ID}")
    model = E5QueryEncoder(MODEL_ID)
    model.eval()
    model.to("cpu")

    print("Building tokenizer.json with fixed max_length=128")
    tokenizer = build_tokenizer(tokenizer_path)

    print("Exporting ExecuTorch program with XNNPACK partitioner")
    example_input_ids = torch.zeros((1, MAX_TOKENS), dtype=torch.long)
    example_attention = torch.ones((1, MAX_TOKENS), dtype=torch.long)
    export_executorch(model, (example_input_ids, example_attention), pte_path)

    print("Running self-test comparisons")
    run_self_test(model, tokenizer, pte_path)

    meta = {
        "model_id": MODEL_ID,
        "dim": EMBED_DIM,
        "max_tokens": MAX_TOKENS,
        "normalized": True,
        "pte_sha256": sha256_file(pte_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "versions": {
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "executorch": get_package_version("executorch"),
        },
        "created_at_unix": int(time.time()),
    }
    write_json(meta_path, meta)
    print(f"Wrote {pte_path}")
    print(f"Wrote {tokenizer_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
