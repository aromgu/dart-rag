"""검색 기법 비교실험: dense+sparse(bge-m3) 하이브리드 vs 전통 BM25.

data/eval/golden_qa.json(154문항, 77개 기업)을 기준으로 두 방식의
Recall@5/Recall@10/MRR을 계산해서 data/eval/retrieval_comparison.json에 저장한다.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.retrieval_metrics import reciprocal_rank, recall_at_k
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.retriever import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "retrieval_comparison.json"


def main() -> None:
    items = json.loads(GOLDEN.read_text())
    print(f"골드셋 {len(items)}문항 로드", flush=True)

    retriever = Retriever()
    bm25 = BM25Retriever()

    results = {"hybrid": [], "bm25": []}
    t0 = time.time()
    for i, item in enumerate(items, 1):
        q, corp, expected = item["question"], item["corp_name"], item["expected_chunk_id"]

        hybrid_ids = retriever.search_ids(q, top_k=10, corp_name=corp, candidate_n=100)
        bm25_ids = bm25.search(q, top_k=10, corp_name=corp)

        results["hybrid"].append(
            {
                "recall@5": recall_at_k(hybrid_ids, expected, 5),
                "recall@10": recall_at_k(hybrid_ids, expected, 10),
                "rr": reciprocal_rank(hybrid_ids, expected),
            }
        )
        results["bm25"].append(
            {
                "recall@5": recall_at_k(bm25_ids, expected, 5),
                "recall@10": recall_at_k(bm25_ids, expected, 10),
                "rr": reciprocal_rank(bm25_ids, expected),
            }
        )

        if i % 20 == 0 or i == len(items):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(items) - i)
            print(f"[{i}/{len(items)}] 경과 {elapsed:.1f}초, 예상 잔여 {eta:.1f}초", flush=True)

    def _agg(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "recall@5": sum(r["recall@5"] for r in rows) / n,
            "recall@10": sum(r["recall@10"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / n,
            "n": n,
        }

    summary = {"hybrid": _agg(results["hybrid"]), "bm25": _agg(results["bm25"])}
    OUTPUT.write_text(json.dumps({"summary": summary, "per_item": results}, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
