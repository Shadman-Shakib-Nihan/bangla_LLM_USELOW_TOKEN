"""Phase 2 gate: time multilingual-e5-small on 1,000 corpus passages.

Does NOT embed the full slice. Loads the model, times a warmed-up run over 1,000
seeded-sampled passages on MPS (fp16), measures throughput and peak RSS, and
extrapolates the full-slice cost and embedding storage.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/time_embedding.py
"""

from __future__ import annotations

import json
import os
import random
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Keep model weights inside the project cache (gitignored).
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus.jsonl"
OUT = PROJECT_ROOT / "results" / "phase2_embedding_timing.json"
MODEL_NAME = "intfloat/multilingual-e5-small"
N_TIMING = 1000
N_WARMUP = 16
BATCH_SIZE = 16
SEED = 7
MAX_SEQ = 512


def sample_texts(path: Path, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            total += 1
            text = json.loads(line)["text"]
            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randint(0, total - 1)
                if j < n:
                    reservoir[j] = text
    return reservoir


def main() -> int:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available; refusing to report Mac timing")

    texts = sample_texts(CORPUS, N_TIMING, SEED)
    passages = ["passage: " + t for t in texts]
    print(f"[ok] sampled {len(passages)} passages (seed={SEED})")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device="mps")
    model.max_seq_length = MAX_SEQ
    dtype = "fp32"
    try:
        model.half()
        dtype = "fp16"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] fp16 not available, using fp32: {exc}")
    load_s = time.time() - t0
    dim = model.get_sentence_embedding_dimension()
    print(f"[ok] loaded {MODEL_NAME} in {load_s:.1f}s | dim={dim} | dtype={dtype}")

    # Warm-up (not timed).
    model.encode(
        passages[:N_WARMUP], batch_size=BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=False, device="mps", convert_to_numpy=True,
    )
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t1 = time.time()
    emb = model.encode(
        passages, batch_size=BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=False, device="mps", convert_to_numpy=True,
    )
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elapsed = time.time() - t1
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is bytes on macOS.
    peak_rss_gib = rss_after / (1024 ** 3)
    # MPS allocated memory, if available.
    mps_alloc_gib = (
        torch.mps.driver_allocated_memory() / (1024 ** 3)
        if hasattr(torch.mps, "driver_allocated_memory") else None
    )

    total_passages = sum(1 for _ in CORPUS.open(encoding="utf-8"))
    throughput = len(passages) / elapsed
    full_seconds = total_passages / throughput
    bytes_per_vec = 2 if dtype == "fp16" else 4
    storage_mib = total_passages * dim * bytes_per_vec / (1024 ** 2)

    result = {
        "model": MODEL_NAME,
        "device": "mps",
        "dtype": dtype,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ,
        "seed": SEED,
        "n_timed": len(passages),
        "model_load_seconds": round(load_s, 2),
        "encode_seconds_1000": round(elapsed, 3),
        "throughput_passages_per_s": round(throughput, 2),
        "embedding_dim": dim,
        "corpus_passages": total_passages,
        "estimated_full_encode_seconds": round(full_seconds, 1),
        "estimated_full_encode_minutes": round(full_seconds / 60, 1),
        "estimated_embedding_storage_mib": round(storage_mib, 1),
        "peak_rss_gib": round(peak_rss_gib, 2),
        "mps_driver_allocated_gib": round(mps_alloc_gib, 2) if mps_alloc_gib else None,
        "torch": torch.__version__,
        "embedding_shape": list(emb.shape),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== multilingual-e5-small timing (MPS) ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"\n[ok] wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
