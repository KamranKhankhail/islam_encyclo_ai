#!/usr/bin/env python3
"""
Export E5 query encoder to ExecuTorch (.pte) lowered for XNNPACK + tokenizer.json.

Repo root (Windows):
  D:\\projects\\IslamEncycloAI\\IslamEncycloAI

Run:
  python scripts/export_e5_query_encoder_executorch.py

Outputs:
  output/processed/models/e5_query_encoder_xnnpack.pte
  output/processed/models/e5_tokenizer.json
  output/processed/models/e5_query_encoder_meta.json

Notes:
- This script is strict by design: it refuses to "silently succeed" with bad embeddings.
- ExecuTorch .pte compatibility depends on the ExecuTorch runtime version used on-device.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Hard constants (do not drift)
# ----------------------------
MODEL_ID = "intfloat/multilingual-e5-small"
MAX_TOKENS = 128
EMBED_DIM = 384

# Deterministic fixed test set (multilingual, query-prefixed)
TEST_QUERIES: List[str] = [
    "query: patience in adversity",
    "query: what does the Quran say about mercy",
    "query: الحمد لله",
    "query: الصلاة في الإسلام",
    "query: صبر کا اجر کیا ہے؟",
]

# Tolerances for self-test
MIN_COSINE = 0.999
MAX_ABS_DIFF = 1e-3


# ----------------------------
# Dependency checks (explicit)
# ----------------------------
def _require_import(name: str):
    try:
        return __import__(name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {name}. Install it before running.") from exc


np = _require_import("numpy")
torch = _require_import("torch")

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency: transformers. Install transformers before running.") from exc

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: sentence-transformers. Install sentence-transformers before running."
    ) from exc


# ----------------------------
# Utilities
# ----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_compact(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_package_version(package: str) -> str:
    # Avoids importing importlib.metadata on old environments unnecessarily.
    try:
        mod = __import__(package)
        v = getattr(mod, "__version__", None)
        if isinstance(v, str) and v:
            return v
    except Exception:
        pass
    try:
        import importlib.metadata as md  # py3.8+

        return md.version(package)
    except Exception:
        return "unknown"


def now_unix() -> int:
    return int(time.time())


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # F.normalize avoids some edge-case differences across versions.
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


# ----------------------------
# Model: E5 query encoder
# ----------------------------
class E5QueryEncoder(torch.nn.Module):
    """
    Mean-pooling with attention mask + L2 normalize, matching SentenceTransformers behavior.
    Output: float32 shape [B, 384], normalized to unit length.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        # Reduce export complexity (no cache, no hidden states)
        if hasattr(self.encoder, "config"):
            self.encoder.config.use_cache = False
            self.encoder.config.output_hidden_states = False
            self.encoder.config.output_attentions = False

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, T, H]

        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)  # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)  # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        mean = summed / denom

        normed = _l2_normalize(mean).to(dtype=torch.float32)
        return normed


# ----------------------------
# Tokenizer export (tokenizer.json)
# ----------------------------
def build_tokenizer_json(output_path: Path):
    """
    Exports HF fast tokenizer backend as tokenizer.json with fixed pad/truncate to MAX_TOKENS.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("AutoTokenizer did not load a fast tokenizer; required for tokenizer.json export.")

    if tok.pad_token_id is None or tok.pad_token is None:
        raise RuntimeError("Tokenizer missing pad_token_id/pad_token; cannot enable fixed-length padding.")

    backend = getattr(tok, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError("Fast tokenizer missing backend_tokenizer; cannot export tokenizer.json.")

    # Persisted into tokenizer.json
    backend.enable_truncation(max_length=MAX_TOKENS, strategy="longest_first")
    backend.enable_padding(
        length=MAX_TOKENS,
        direction="right",
        pad_id=int(tok.pad_token_id),
        pad_type_id=int(getattr(tok, "pad_token_type_id", 0) or 0),
        pad_token=str(tok.pad_token),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(output_path))
    return tok


def validate_tokenizer_json(tokenizer_path: Path) -> None:
    """
    Validates that tokenizer.json is valid JSON and contains padding + truncation config.
    Structure may vary slightly by tokenizers version; we do a robust presence check.
    """
    obj = read_json(tokenizer_path)
    if not isinstance(obj, dict):
        raise RuntimeError("tokenizer.json root must be an object/dict.")

    trunc = obj.get("truncation")
    pad = obj.get("padding")
    if trunc is None or pad is None:
        raise RuntimeError("tokenizer.json missing truncation/padding configuration.")

    # Best-effort checks (structure differs across tokenizers versions).
    # Truncation: max_length must be MAX_TOKENS somewhere.
    trunc_str = json.dumps(trunc, separators=(",", ":"), ensure_ascii=False)
    pad_str = json.dumps(pad, separators=(",", ":"), ensure_ascii=False)

    if str(MAX_TOKENS) not in trunc_str:
        raise RuntimeError(f"tokenizer.json truncation config does not appear to include max_length={MAX_TOKENS}.")
    if str(MAX_TOKENS) not in pad_str:
        raise RuntimeError(f"tokenizer.json padding config does not appear to include length={MAX_TOKENS}.")


# ----------------------------
# Encoding helpers for tests
# ----------------------------
def tokenize_batch(tok, texts: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tok(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise RuntimeError("Tokenizer output missing input_ids/attention_mask.")
    if encoded["input_ids"].shape[-1] != MAX_TOKENS:
        raise RuntimeError(f"Tokenizer produced seq_len={encoded['input_ids'].shape[-1]} expected {MAX_TOKENS}.")
    return encoded["input_ids"], encoded["attention_mask"]


def assert_close_embeddings(label: str, ref: "np.ndarray", cand: "np.ndarray") -> None:
    if ref.shape != cand.shape:
        raise RuntimeError(f"{label}: shape mismatch {ref.shape} vs {cand.shape}.")
    if ref.ndim != 2 or ref.shape[1] != EMBED_DIM:
        raise RuntimeError(f"{label}: expected (*,{EMBED_DIM}) got {ref.shape}.")

    for i in range(ref.shape[0]):
        a = ref[i]
        b = cand[i]
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        cos = float(np.dot(a, b) / denom)
        max_diff = float(np.max(np.abs(a - b)))

        # STRICT: fail if either tolerance is violated
        if cos < MIN_COSINE or max_diff > MAX_ABS_DIFF:
            raise RuntimeError(
                f"{label}: mismatch at idx={i} cos={cos:.6f} (min {MIN_COSINE}) "
                f"max_abs_diff={max_diff:.6f} (max {MAX_ABS_DIFF})."
            )


def assert_unit_norm(label: str, emb: "np.ndarray") -> None:
    norms = np.linalg.norm(emb, axis=1)
    max_err = float(np.max(np.abs(norms - 1.0)))
    if max_err > 1e-3:
        raise RuntimeError(f"{label}: embeddings not unit-normalized (max |norm-1|={max_err:.6f}).")


# ----------------------------
# ExecuTorch export
# ----------------------------
def export_to_executorch_xnnpack(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, torch.Tensor],
    out_pte: Path,
) -> None:
    if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
        raise RuntimeError("torch.export.export is required (PyTorch 2.1+).")

    try:
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: executorch (exir). Install executorch before exporting.") from exc

    # XNNPACK partitioner: preferred import path per docs
    partitioner = None
    import_errors: List[str] = []
    for p in (
        "executorch.backends.xnnpack.partition.xnnpack_partitioner",
        "executorch.backends.xnnpack.partition.xnnpack_partitioner",  # keep duplicate-free; retained for clarity
        "executorch.backends.xnnpack.partitioner",  # older
        "executorch.exir.backend.partitioner",  # much older
    ):
        try:
            mod = __import__(p, fromlist=["XnnpackPartitioner"])
            partitioner = getattr(mod, "XnnpackPartitioner")()
            break
        except Exception as e:
            import_errors.append(f"{p}: {e!r}")

    if partitioner is None:
        raise RuntimeError("Failed to import XnnpackPartitioner. Errors:\n" + "\n".join(import_errors))

    # XNNPACK edge compile config (recommended where available)
    compile_config = None
    try:
        # Documented in ExecuTorch LLM export examples for XNNPACK edge compile config.
        from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config  # type: ignore

        compile_config = get_xnnpack_edge_compile_config()
        print("ExecuTorch: using get_xnnpack_edge_compile_config()")
    except Exception:
        # Safe fallback (also shown in XNNPACK lowering tutorials)
        compile_config = EdgeCompileConfig(_check_ir_validity=False)
        print("ExecuTorch: using EdgeCompileConfig(_check_ir_validity=False) fallback")

    exported = torch.export.export(model, example_inputs)

    edge_mgr = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner],
        compile_config=compile_config,
    )

    exec_prog = edge_mgr.to_executorch()

    out_pte.parent.mkdir(parents=True, exist_ok=True)
    with out_pte.open("wb") as f:
        # Common save patterns across versions
        if hasattr(exec_prog, "write_to_file"):
            exec_prog.write_to_file(f)
        elif hasattr(exec_prog, "buffer"):
            f.write(exec_prog.buffer)  # type: ignore[attr-defined]
        else:
            raise RuntimeError("ExecuTorch program missing write_to_file/buffer; cannot serialize .pte.")


# ----------------------------
# ExecuTorch runtime validation (official API)
# ----------------------------
def run_executorch_runtime_check(pte_path: Path, inputs: Tuple[torch.Tensor, torch.Tensor]) -> "np.ndarray":
    try:
        from executorch.runtime import Runtime, Verification  # type: ignore
    except Exception as exc:
        raise RuntimeError("ExecuTorch runtime not available. Install executorch runtime bindings.") from exc

    rt = Runtime.get()
    program = rt.load_program(pte_path, verification=Verification.Minimal)

    if not hasattr(program, "method_names"):
        raise RuntimeError("Unexpected ExecuTorch Program API: missing method_names.")

    if "forward" not in set(program.method_names):
        raise RuntimeError(f"Exported program missing 'forward'. Found methods={program.method_names}.")

    forward = program.load_method("forward")
    outputs = forward.execute(inputs)  # returns list of tensors per docs

    if not isinstance(outputs, (list, tuple)) or len(outputs) < 1:
        raise RuntimeError("ExecuTorch forward returned no outputs.")

    out0 = outputs[0]
    if not isinstance(out0, torch.Tensor):
        raise RuntimeError(f"ExecuTorch output[0] is not torch.Tensor, got {type(out0)}")

    arr = out0.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


# ----------------------------
# End-to-end self-test
# ----------------------------
def run_self_test(model: E5QueryEncoder, tok, pte_path: Path) -> None:
    device = torch.device("cpu")
    model.eval().to(device)

    # Reference: SentenceTransformer encode
    st = SentenceTransformer(MODEL_ID, device="cpu")
    st.max_seq_length = MAX_TOKENS

    st_emb = st.encode(
        TEST_QUERIES,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if st_emb.ndim != 2 or st_emb.shape[1] != EMBED_DIM:
        raise RuntimeError(f"SentenceTransformer returned shape {st_emb.shape} expected (*,{EMBED_DIM}).")

    # Torch module encode (same tokenizer)
    input_ids, attention_mask = tokenize_batch(tok, TEST_QUERIES)
    input_ids = input_ids.to(device).contiguous()
    attention_mask = attention_mask.to(device).contiguous()

    with torch.inference_mode():
        torch_out = model(input_ids=input_ids, attention_mask=attention_mask)

    if torch_out.dtype != torch.float32:
        raise RuntimeError(f"Torch model output must be float32, got {torch_out.dtype}.")

    torch_np = torch_out.detach().cpu().numpy()
    assert_unit_norm("Torch module", torch_np)
    assert_close_embeddings("SentenceTransformer vs Torch module", st_emb, torch_np)

    # ExecuTorch runtime check: single query
    input_ids_one = input_ids[:1].contiguous()
    attention_one = attention_mask[:1].contiguous()

    with torch.inference_mode():
        ref_one = model(input_ids=input_ids_one, attention_mask=attention_one).detach().cpu().numpy()

    et_one = run_executorch_runtime_check(pte_path, (input_ids_one, attention_one))
    assert_unit_norm("ExecuTorch runtime", et_one)
    assert_close_embeddings("ExecuTorch runtime vs Torch module (1 item)", ref_one, et_one)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    # Determinism: fixed seeds; CPU-only.
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "output" / "processed" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    pte_path = out_dir / "e5_query_encoder_xnnpack.pte"
    tok_path = out_dir / "e5_tokenizer.json"
    meta_path = out_dir / "e5_query_encoder_meta.json"

    print(f"Repo root: {repo_root}")
    print(f"Model: {MODEL_ID}")
    print(f"MAX_TOKENS: {MAX_TOKENS}, DIM: {EMBED_DIM}")
    print(f"Output dir: {out_dir}")

    # 1) Tokenizer export
    t0 = time.time()
    tok = build_tokenizer_json(tok_path)
    validate_tokenizer_json(tok_path)
    tok_ms = int((time.time() - t0) * 1000)
    print(f"Tokenizer exported + validated in {tok_ms} ms: {tok_path}")

    # 2) Build model
    t1 = time.time()
    model = E5QueryEncoder(MODEL_ID).eval().to("cpu")
    model_ms = int((time.time() - t1) * 1000)
    print(f"Model loaded in {model_ms} ms")

    # 3) Export to ExecuTorch with XNNPACK
    t2 = time.time()
    example_input_ids = torch.zeros((1, MAX_TOKENS), dtype=torch.long).contiguous()
    example_attention = torch.ones((1, MAX_TOKENS), dtype=torch.long).contiguous()
    export_to_executorch_xnnpack(model, (example_input_ids, example_attention), pte_path)
    export_ms = int((time.time() - t2) * 1000)
    print(f"Exported .pte in {export_ms} ms: {pte_path}")

    # 4) Self-test (strict)
    print("Running strict self-test (SentenceTransformer vs Torch vs ExecuTorch runtime)...")
    t3 = time.time()
    run_self_test(model, tok, pte_path)
    test_ms = int((time.time() - t3) * 1000)
    print(f"Self-test OK in {test_ms} ms")

    # 5) Metadata
    meta: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "dim": EMBED_DIM,
        "max_tokens": MAX_TOKENS,
        "normalized": True,
        "files": {
            "pte": {
                "path": str(pte_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(pte_path),
                "bytes": pte_path.stat().st_size,
            },
            "tokenizer": {
                "path": str(tok_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(tok_path),
                "bytes": tok_path.stat().st_size,
            },
        },
        "versions": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": get_package_version("torch"),
            "transformers": get_package_version("transformers"),
            "executorch": get_package_version("executorch"),
            "sentence_transformers": get_package_version("sentence_transformers"),
            "numpy": get_package_version("numpy"),
        },
        "timings_ms": {
            "tokenizer_export": tok_ms,
            "model_load": model_ms,
            "executorch_export": export_ms,
            "self_test": test_ms,
        },
        "created_at_unix": now_unix(),
    }
    write_json_compact(meta_path, meta)

    print("Done.")
    print(f"Wrote: {pte_path}")
    print(f"Wrote: {tok_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
