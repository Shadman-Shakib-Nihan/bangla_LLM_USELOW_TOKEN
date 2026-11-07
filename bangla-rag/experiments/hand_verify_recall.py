"""Hand-verify Recall@k on three toy queries and show the arithmetic.

Writes results/phase3_recall_handcheck.txt. This is a sanity check of the
metric implementation, not a real experiment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import mrr, ndcg_at_k, recall_at_k, recall_fraction_at_k

QUERIES = {
    "q1": {
        "ranking": ["d1", "d2", "d3", "d4", "d5"],
        "gold": {"d3"},
    },
    "q2": {
        "ranking": ["d2", "d4", "d1", "d5", "d3"],
        "gold": {"d1", "d5"},
    },
    "q3": {
        "ranking": ["d5", "d1", "d2", "d3", "d4"],
        "gold": {"d2", "d4"},
    },
}
K_VALUES = (1, 3, 5)


def main() -> int:
    lines: list[str] = []
    lines.append("Hand verification of Recall@k (hit rate), MRR, nDCG@5")
    lines.append("=" * 60)
    lines.append("Recall@k = 1 if any gold ID is in the top-k list, else 0.")
    lines.append("nDCG uses binary relevance; IDCG = sum 1/log2(i+1) for i<=|gold|.")
    lines.append("")

    for qid, data in QUERIES.items():
        ranking = data["ranking"]
        gold = data["gold"]
        lines.append(f"Query {qid}")
        lines.append(f"  ranking = {ranking}")
        lines.append(f"  gold    = {sorted(gold)}")
        for k in K_VALUES:
            top = ranking[:k]
            hits = sorted(set(top) & gold)
            hit = 1.0 if hits else 0.0
            frac = len(hits) / len(gold)
            lines.append(
                f"  top-{k} = {top} -> hits={hits} "
                f"hit_rate={hit:.0f} recall_fraction={frac:.3f}"
            )
            assert recall_at_k(ranking, gold, k) == hit
            assert recall_fraction_at_k(ranking, gold, k) == frac

        first_gold_rank = next(
            (i for i, d in enumerate(ranking, start=1) if d in gold), None
        )
        mrr_manual = 1.0 / first_gold_rank if first_gold_rank else 0.0
        dcg = sum(
            1.0 / math.log2(i + 1)
            for i, d in enumerate(ranking[:5], start=1) if d in gold
        )
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), 5) + 1))
        ndcg_manual = dcg / idcg if idcg else 0.0
        lines.append(
            f"  first gold at rank {first_gold_rank} -> MRR = 1/{first_gold_rank} "
            f"= {mrr_manual:.4f}"
        )
        lines.append(
            f"  nDCG@5: DCG={dcg:.4f}, IDCG={idcg:.4f}, "
            f"nDCG={ndcg_manual:.4f}"
        )
        assert mrr(ranking, gold) == mrr_manual
        assert abs(ndcg_at_k(ranking, gold, 5) - ndcg_manual) < 1e-12
        lines.append("  all asserted values match src.metrics")
        lines.append("")

    text = "\n".join(lines)
    out = PROJECT_ROOT / "results" / "phase3_recall_handcheck.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[ok] wrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
