"""cross-encoder 재순위가 검색 품질을 얼마나 더 올리는지 골드셋으로 검증한다.

하이브리드로 top-20 후보를 찾은 뒤 cross-encoder(bge-reranker-base)로
top-5까지 재정렬한 결과를, 하이브리드가 바로 뽑은 top-5(재순위 없음)와 비교한다.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.retrieval_metrics import recall_at_k_by_value, reciprocal_rank_by_value
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "rerank_comparison.json"

RERANK_CANDIDATE_N = 20
TOP_K = 5


def main() -> None:
    items = json.loads(GOLDEN.read_text())
    print(f"골드셋 {len(items)}문항 로드", flush=True)

    retriever = Retriever()
    reranker = Reranker()

    results = {"no_rerank": [], "rerank": []}
    t0 = time.time()

    for i, item in enumerate(items, 1):
        q, corp, expected_value = item["question"], item["corp_name"], item["expected_value"]

        candidates = retriever.search(q, top_k=RERANK_CANDIDATE_N, corp_name=corp, candidate_n=100)
        no_rerank_hit_5 = any(expected_value in c["text"] for c in candidates[:5])
        no_rerank_rr = next(
            (1.0 / (rank + 1) for rank, c in enumerate(candidates) if expected_value in c["text"]), 0.0
        )

        reranked = reranker.rerank(q, list(candidates), top_k=TOP_K)
        rerank_hit_5 = any(expected_value in c["text"] for c in reranked[:5])
        rerank_rr = next(
            (1.0 / (rank + 1) for rank, c in enumerate(reranked) if expected_value in c["text"]), 0.0
        )

        results["no_rerank"].append({"value_recall@5": int(no_rerank_hit_5), "value_rr": no_rerank_rr})
        results["rerank"].append({"value_recall@5": int(rerank_hit_5), "value_rr": rerank_rr})

        if i % 20 == 0 or i == len(items):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(items) - i)
            print(f"[{i}/{len(items)}] 경과 {elapsed:.1f}초, 예상 잔여 {eta:.1f}초", flush=True)

    def _agg(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "value_recall@5": sum(r["value_recall@5"] for r in rows) / n,
            "value_mrr": sum(r["value_rr"] for r in rows) / n,
            "n": n,
        }

    summary = {label: _agg(rows) for label, rows in results.items()}
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
