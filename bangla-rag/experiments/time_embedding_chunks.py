"""Phase 3 gate: time multilingual-e5-small on 1,000 retrieval units (chunks).

Samples 1,000 units from data/processed/corpus_chunks.jsonl, times a warmed-up
MPS fp16 run, and extrapolates the full split-index workload plus the 1,501
truncated parents needed for the dense ablation.

Does NOT embed the full corpus.

Usage:
    cd bangla-rag
    .venv/bin/python experiments/time_embedding_chunks.py
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

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

CORPUS = PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl"
ABLATION = PROJECT_ROOT / "data" / "processed" / "overlength_parents.jsonl"
OUT = PROJECT_ROOT / "results" / "phase3_embedding_timing_chunks.json"
MODEL_NAME = "intfloat/multilingual-e5-small"
N_TIMING = 1000
N_WARMUP = 16
BATCH_SIZE = 16
SEED = 7
MAX_SEQ = 512
PREFIX = "passage: "


def sample_texts(path: Path, n: int, seed: int) -> tuple[list[str], int]:
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
    return reservoir, total


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> int:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available; refusing to report Mac timing")

    texts, total_units = sample_texts(CORPUS, N_TIMING, SEED)
    ablation_units = count_lines(ABLATION)
    passages = [PREFIX + t for t in texts]
    print(f"[ok] sampled {len(passages)} units from {total_units:,} "
          f"(seed={SEED}); ablation units={ablation_units:,}")

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
    dim = model.get_embedding_dimension()
    print(f"[ok] loaded {MODEL_NAME} in {load_s:.1f}s | dim={dim} | dtype={dtype}")

    model.encode(
        passages[:N_WARMUP], batch_size=BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=False, device="mps", convert_to_numpy=True,
    )
    torch.mps.synchronize()

    t1 = time.time()
    emb = model.encode(
        passages, batch_size=BATCH_SIZE, normalize_embeddings=True,
        show_progress_bar=False, device="mps", convert_to_numpy=True,
    )
    torch.mps.synchronize()
    elapsed = time.time() - t1

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    mps_alloc_gib = torch.mps.driver_allocated_memory() / (1024 ** 3)

    throughput = len(passages) / elapsed
    split_seconds = total_units / throughput
    ablation_seconds = ablation_units / throughput
    bytes_per_vec = 2 if dtype == "fp16" else 4
    storage_mib = (total_units + ablation_units) * dim * bytes_per_vec / (1024 ** 2)

    result = {
        "model": MODEL_NAME,
        "corpus": str(CORPUS.relative_to(PROJECT_ROOT)),
        "device": "mps",
        "dtype": dtype,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ,
        "seed": SEED,
        "n_timed": len(passages),
        "model_load_seconds": round(load_s, 2),
        "encode_seconds_1000": round(elapsed, 3),
        "throughput_units_per_s": round(throughput, 2),
        "embedding_dim": dim,
        "split_index_units": total_units,
        "ablation_units": ablation_units,
        "estimated_split_encode_seconds": round(split_seconds, 1),
        "estimated_split_encode_minutes": round(split_seconds / 60, 1),
        "estimated_ablation_encode_seconds": round(ablation_seconds, 1),
        "estimated_total_encode_seconds": round(split_seconds + ablation_seconds, 1),
        "estimated_total_encode_minutes": round(
            (split_seconds + ablation_seconds) / 60, 1
        ),
        "estimated_embedding_storage_mib": round(storage_mib, 1),
        "peak_rss_gib": round(peak_rss_gib, 2),
        "mps_driver_allocated_gib": round(mps_alloc_gib, 2),
        "torch": torch.__version__,
        "embedding_shape": list(emb.shape),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== multilingual-e5-small timing (chunks, MPS) ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"\n[ok] wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
