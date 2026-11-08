"""Phase 3: checkpointed multilingual-e5-small embeddings.

Embeds the shared retrieval units (data/processed/corpus_chunks.jsonl) with
batch size 16, fp16 on MPS, the ``passage: `` prefix, mean pooling over the
attention mask, and L2 normalization. The same recipe with ``query: `` is used
for question sets. Also embeds the 1,501 truncated over-length parents used by
the truncated-vs-split ablation.

Checkpointing: vectors are written to a float16 memmap in file order. A JSON
sidecar records how many units are done, so an interrupted run resumes at the
last batch boundary instead of starting over.

Before a run is marked complete:
  * every stored vector is checked for NaN/INF;
  * the first 100 units are embedded a second time and compared with the
    stored vectors (max abs difference and min cosine similarity).

Usage:
    cd bangla-rag
    .venv/bin/python experiments/phase3_embed_dense.py
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
BATCH_SIZE = 16
MAX_SEQ_LEN = 512
DTYPE = "fp16"
CHECKPOINT_EVERY = 2048
CHECK_N = 100

PROCESSED = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS = PROJECT_ROOT / "results"

UNITS = PROCESSED / "corpus_chunks.jsonl"
TRUNCATED = PROCESSED / "overlength_parents_truncated.jsonl"
QUERY_SETS = {
    "miracl_train": PROCESSED / "miracl_train_questions.jsonl",
    "miracl_dev": PROCESSED / "miracl_dev_questions.jsonl",
    "tydi_train": PROCESSED / "tydi_train_questions.jsonl",
    "tydi_validation": PROCESSED / "tydi_validation_questions.jsonl",
}

OUT_UNITS = EMB_DIR / "corpus_units.f16.npy"
OUT_UNITS_CKPT = EMB_DIR / "corpus_units.checkpoint.json"
OUT_TRUNC = EMB_DIR / "overlength_truncated.f16.npy"
OUT_TRUNC_IDS = EMB_DIR / "overlength_truncated_ids.json"
OUT_CHECKS = RESULTS / "phase3_embed_checks.json"


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.relative_to(PROJECT_ROOT)),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model = model.to(device).eval()
    dim = int(model.config.hidden_size)
    return tokenizer, model, device, dim


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device,
    batch_size: int = BATCH_SIZE,
    show: bool = False,
) -> np.ndarray:
    """Mean-pool (attention mask) then L2-normalize; returns float32."""
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        normed = F.normalize(pooled, p=2, dim=1)
        out.append(normed.cpu().numpy().astype(np.float32))
        if show and (start // batch_size) % 100 == 0:
            print(f"    {start + len(batch):,}/{len(texts):,}", flush=True)
    if not out:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(out, axis=0)


def iter_unit_texts(skip: int = 0, limit: int | None = None):
    yielded = 0
    seen = 0
    with UNITS.open(encoding="utf-8") as f:
        for line in f:
            if seen < skip:
                seen += 1
                continue
            if limit is not None and yielded >= limit:
                return
            rec = json.loads(line)
            yielded += 1
            yield PASSAGE_PREFIX + rec["text"]


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def embed_units_simple(tokenizer, model, device, dim: int) -> dict:
    """Batched loop with checkpoints (used by main)."""
    n_units = count_lines(UNITS)
    fingerprint = file_fingerprint(UNITS)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    start_count = 0
    ckpt = {}
    if OUT_UNITS_CKPT.exists() and OUT_UNITS.exists():
        ckpt = json.loads(OUT_UNITS_CKPT.read_text(encoding="utf-8"))
        same_input = ckpt.get("input") == fingerprint
        same_shape = (
            ckpt.get("n_units") == n_units
            and ckpt.get("dim") == dim
            and ckpt.get("model") == MODEL_NAME
            and ckpt.get("batch_size") == BATCH_SIZE
            and ckpt.get("dtype") == DTYPE
        )
        if same_input and same_shape:
            start_count = int(ckpt.get("count", 0))
            start_count -= start_count % BATCH_SIZE
        else:
            start_count = 0

    if start_count >= n_units and ckpt.get("complete"):
        print(f"[ok] units already complete ({n_units:,})")
        return {"n_units": n_units, "resumed": True, "encode_seconds": 0.0}
    if start_count:
        print(f"[resume] {start_count:,}/{n_units:,} units already embedded")

    if OUT_UNITS.exists() and start_count > 0:
        mm = np.lib.format.open_memmap(
            OUT_UNITS, mode="r+", dtype=np.float16, shape=(n_units, dim)
        )
    else:
        mm = np.lib.format.open_memmap(
            OUT_UNITS, mode="w+", dtype=np.float16, shape=(n_units, dim)
        )

    pending: list[str] = []
    done = start_count
    since_ckpt = 0
    t0 = time.time()
    for idx, text in enumerate(iter_unit_texts(skip=start_count)):
        pending.append(text)
        if len(pending) == BATCH_SIZE:
            vec = embed_texts(pending, tokenizer, model, device)
            if not np.isfinite(vec).all():
                raise RuntimeError(f"non-finite embedding at unit offset {done}")
            mm[done:done + len(pending)] = vec.astype(np.float16)
            done += len(pending)
            since_ckpt += len(pending)
            pending = []
            if since_ckpt >= CHECKPOINT_EVERY and done < n_units:
                mm.flush()
                write_json_atomic(OUT_UNITS_CKPT, {
                    "model": MODEL_NAME,
                    "prefix": PASSAGE_PREFIX,
                    "batch_size": BATCH_SIZE,
                    "dtype": DTYPE,
                    "max_seq_len": MAX_SEQ_LEN,
                    "pooling": "mean",
                    "normalized": True,
                    "dim": dim,
                    "n_units": n_units,
                    "count": done,
                    "complete": False,
                    "input": fingerprint,
                })
                since_ckpt = 0
                rate = (done - start_count) / max(time.time() - t0, 1e-9)
                eta = (n_units - done) / max(rate, 1e-9)
                print(f"  [ckpt] {done:,}/{n_units:,} | {rate:.1f}/s | "
                      f"ETA {eta/60:.1f} min", flush=True)
    if pending:
        vec = embed_texts(pending, tokenizer, model, device)
        if not np.isfinite(vec).all():
            raise RuntimeError(f"non-finite embedding at unit offset {done}")
        mm[done:done + len(pending)] = vec.astype(np.float16)
        done += len(pending)

    if done != n_units:
        raise RuntimeError(f"embedded {done} units but expected {n_units}")
    mm.flush()
    elapsed = time.time() - t0
    write_json_atomic(OUT_UNITS_CKPT, {
        "model": MODEL_NAME,
        "prefix": PASSAGE_PREFIX,
        "batch_size": BATCH_SIZE,
        "dtype": DTYPE,
        "max_seq_len": MAX_SEQ_LEN,
        "pooling": "mean",
        "normalized": True,
        "dim": dim,
        "n_units": n_units,
        "count": done,
        "complete": True,
        "input": fingerprint,
        "encode_seconds": round(elapsed, 1),
    })
    return {
        "n_units": n_units,
        "resumed": start_count > 0,
        "start_count": start_count,
        "encode_seconds": round(elapsed, 1),
    }


def verify_units(tokenizer, model, device, dim: int) -> dict:
    n_units = count_lines(UNITS)
    mm = np.load(OUT_UNITS, mmap_mode="r")
    if mm.shape != (n_units, dim):
        raise RuntimeError(f"unexpected unit embedding shape {mm.shape}")

    nonfinite = 0
    zero_norm = 0
    for start in range(0, n_units, 10000):
        block = np.asarray(mm[start:start + 10000], dtype=np.float32)
        nonfinite += int((~np.isfinite(block)).sum())
        zero_norm += int((np.linalg.norm(block, axis=1) < 0.5).sum())
    if nonfinite:
        raise RuntimeError(f"{nonfinite} non-finite values in unit embeddings")

    sample = list(iter_unit_texts(limit=CHECK_N))
    again = embed_texts(sample, tokenizer, model, device)
    stored = np.asarray(mm[:CHECK_N], dtype=np.float32)
    max_abs = float(np.abs(again - stored).max())
    cos = (again * stored).sum(axis=1)
    min_cos = float(cos.min())
    ok = max_abs < 1e-3 and min_cos > 0.999
    return {
        "n_units": n_units,
        "dim": dim,
        "nonfinite_values": nonfinite,
        "vectors_with_norm_<0.5": zero_norm,
        "determinism_check_n": CHECK_N,
        "determinism_max_abs_diff": max_abs,
        "determinism_min_cosine": min_cos,
        "determinism_ok": ok,
        "checks_ok": bool(ok and nonfinite == 0 and zero_norm == 0),
    }


def embed_query_sets(tokenizer, model, device) -> dict:
    report = {}
    for name, path in QUERY_SETS.items():
        ids: list[str] = []
        texts: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids.append(rec["question_id"])
                texts.append(QUERY_PREFIX + rec["question"])
        t0 = time.time()
        vec = embed_texts(texts, tokenizer, model, device, show=False)
        if not np.isfinite(vec).all():
            raise RuntimeError(f"non-finite query embedding in {name}")
        norms = np.linalg.norm(vec, axis=1)
        out_npy = EMB_DIR / f"queries_{name}.f16.npy"
        np.save(out_npy, vec.astype(np.float16))
        (EMB_DIR / f"queries_{name}.json").write_text(
            json.dumps({"question_ids": ids}, ensure_ascii=False),
            encoding="utf-8",
        )
        report[name] = {
            "n_queries": len(texts),
            "path": str(out_npy.relative_to(PROJECT_ROOT)),
            "encode_seconds": round(time.time() - t0, 1),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
            "nonfinite_values": int((~np.isfinite(vec)).sum()),
        }
        print(f"[ok] queries {name}: {len(texts):,} in "
              f"{report[name]['encode_seconds']}s")
    return report


def embed_truncated(tokenizer, model, device) -> dict:
    ids: list[str] = []
    texts: list[str] = []
    with TRUNCATED.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["passage_id"])
            texts.append(PASSAGE_PREFIX + rec["text"])
    t0 = time.time()
    vec = embed_texts(texts, tokenizer, model, device)
    if not np.isfinite(vec).all():
        raise RuntimeError("non-finite truncated-parent embedding")
    np.save(OUT_TRUNC, vec.astype(np.float16))
    OUT_TRUNC_IDS.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] truncated parents: {len(texts):,} in "
          f"{time.time()-t0:.1f}s")
    return {
        "n_truncated": len(texts),
        "path": str(OUT_TRUNC.relative_to(PROJECT_ROOT)),
        "ids_path": str(OUT_TRUNC_IDS.relative_to(PROJECT_ROOT)),
        "encode_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available; refusing to run the Mac fp16 pass")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"[ok] loading {MODEL_NAME}")
    tokenizer, model, device, dim = load_model()
    print(f"[ok] device={device} dtype={DTYPE} dim={dim} "
          f"batch_size={BATCH_SIZE} max_seq={MAX_SEQ_LEN}")

    unit_stats = embed_units_simple(tokenizer, model, device, dim)
    print(f"[ok] unit pass done: {unit_stats}")
    checks = verify_units(tokenizer, model, device, dim)
    print(f"[ok] checks: {checks}")
    if not checks["checks_ok"]:
        raise SystemExit("embedding checks failed; not marking complete")

    query_stats = embed_query_sets(tokenizer, model, device)
    trunc_stats = embed_truncated(tokenizer, model, device)

    payload = {
        "model": MODEL_NAME,
        "device": str(device),
        "dtype": DTYPE,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LEN,
        "pooling": "mean over attention mask",
        "normalization": "L2",
        "prefixes": {"units": PASSAGE_PREFIX, "queries": QUERY_PREFIX},
        "units": unit_stats,
        "truncated": trunc_stats,
        "queries": query_stats,
        "checks": checks,
        "peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3), 2
        ),
    }
    write_json_atomic(OUT_CHECKS, payload)
    print(f"\n[ok] wrote {OUT_CHECKS.relative_to(PROJECT_ROOT)}")
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
