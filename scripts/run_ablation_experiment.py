"""검색 조건별 ablation 비교실험.

dense 단독 / bge-m3 sparse 단독 / BM25 단독 / dense+sparse(RRF) / dense+BM25(RRF)
5가지 조합을 같은 골드셋(154문항)으로 비교한다. 모델은 한 번만 로드해서 재사용한다.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.session import SessionLocal
from src.evaluation.retrieval_metrics import (
    reciprocal_rank,
    reciprocal_rank_by_value,
    recall_at_k,
    recall_at_k_by_value,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.retriever import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "ablation_comparison.json"

CANDIDATE_N = 100
TOP_K = 10


def main() -> None:
    items = json.loads(GOLDEN.read_text())
    print(f"골드셋 {len(items)}문항 로드", flush=True)

    retriever = Retriever()
    bm25 = BM25Retriever()

    conditions = ["dense", "sparse", "bm25", "dense+sparse", "dense+bm25"]
    results = {c: [] for c in conditions}

    t0 = time.time()
    for i, item in enumerate(items, 1):
        q, corp, expected = item["question"], item["corp_name"], item["expected_chunk_id"]

        dense_vec, sparse_vec = retriever.embedder.embed_query(q)
        dense_ids = retriever._dense_search(dense_vec, CANDIDATE_N, corp)
        sparse_ids = retriever._sparse_search(sparse_vec, CANDIDATE_N, corp)
        bm25_ids = bm25.search(q, top_k=CANDIDATE_N, corp_name=corp)

        ranked = {
            "dense": dense_ids[:TOP_K],
            "sparse": sparse_ids[:TOP_K],
            "bm25": bm25_ids[:TOP_K],
            "dense+sparse": [cid for cid, _ in retriever._rrf_fuse([dense_ids, sparse_ids], TOP_K)],
            "dense+bm25": [cid for cid, _ in retriever._rrf_fuse([dense_ids, bm25_ids], TOP_K)],
        }

        # 같은 수치가 여러 표에 동시에 등장하는 경우가 흔해서(실측 확인),
        # "정답 chunk_id 하나와 정확히 일치"뿐 아니라 "정답 수치가 실제로
        # 담긴 청크를 찾았는가"도 같이 측정한다 - 검색된 청크들 텍스트를 한 번에 조회.
        unique_ids = {cid for ids in ranked.values() for cid in ids}
        with SessionLocal() as session:
            rows = session.execute(
                text("SELECT id, text FROM chunks WHERE id = ANY(:ids)"), {"ids": list(unique_ids)}
            ).fetchall()
        id_to_text = dict(rows)
        expected_value = item["expected_value"]

        for c in conditions:
            ids = ranked[c]
            results[c].append(
                {
                    "recall@5": recall_at_k(ids, expected, 5),
                    "recall@10": recall_at_k(ids, expected, 10),
                    "rr": reciprocal_rank(ids, expected),
                    "value_recall@5": recall_at_k_by_value(ids, expected_value, 5, id_to_text),
                    "value_recall@10": recall_at_k_by_value(ids, expected_value, 10, id_to_text),
                    "value_rr": reciprocal_rank_by_value(ids, expected_value, id_to_text),
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
            "value_recall@5": sum(r["value_recall@5"] for r in rows) / n,
            "value_recall@10": sum(r["value_recall@10"] for r in rows) / n,
            "value_mrr": sum(r["value_rr"] for r in rows) / n,
            "n": n,
        }

    summary = {c: _agg(results[c]) for c in conditions}
    OUTPUT.write_text(json.dumps({"summary": summary, "per_item": results}, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
