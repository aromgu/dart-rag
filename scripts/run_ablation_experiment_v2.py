"""비교실험 2차: BM25+breadcrumb, weighted-sum 융합을 추가로 검증한다.

docs/experiments.md 1차 실험에서 발견한 원인(BM25 색인엔 회사명 신호가 아예
없음)을 breadcrumb 포함 BM25로 직접 검증하고, RRF 대신 가중합(weighted-sum)
융합이 "약한 신호가 강한 dense를 희석시키는" 문제를 완화하는지 비교한다.
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
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "ablation_comparison_v2.json"

CANDIDATE_N = 100
TOP_K = 10
DENSE_WEIGHT = 0.7  # 1차 실험에서 dense 단독이 가장 강했던 결과를 반영한 가중치
SPARSE_WEIGHT = 0.3


def main() -> None:
    items = json.loads(GOLDEN.read_text())
    print(f"골드셋 {len(items)}문항 로드", flush=True)

    retriever = Retriever()
    bm25_plain = BM25Retriever(include_breadcrumb=False)
    bm25_bc = BM25Retriever(include_breadcrumb=True)

    conditions = ["bm25+breadcrumb", "dense+bm25bc(rrf)", "dense+sparse(weighted)", "dense+bm25bc(weighted)"]
    results = {c: [] for c in conditions}

    t0 = time.time()
    for i, item in enumerate(items, 1):
        q, corp, expected = item["question"], item["corp_name"], item["expected_chunk_id"]
        expected_value = item["expected_value"]

        dense_vec, sparse_vec = retriever.embedder.embed_query(q)
        dense_scored = retriever._dense_search_scored(dense_vec, CANDIDATE_N, corp)
        sparse_scored = retriever._sparse_search_scored(sparse_vec, CANDIDATE_N, corp)
        dense_ids = [cid for cid, _ in dense_scored]
        bm25bc_scored = bm25_bc.search_scored(q, top_k=CANDIDATE_N, corp_name=corp)
        bm25bc_ids = [cid for cid, _ in bm25bc_scored]

        ranked = {
            "bm25+breadcrumb": bm25_bc.search(q, top_k=TOP_K, corp_name=corp),
            "dense+bm25bc(rrf)": [cid for cid, _ in retriever._rrf_fuse([dense_ids, bm25bc_ids], TOP_K)],
            "dense+sparse(weighted)": [
                cid for cid, _ in retriever._weighted_fuse(
                    [(dense_scored, DENSE_WEIGHT), (sparse_scored, SPARSE_WEIGHT)], TOP_K
                )
            ],
            "dense+bm25bc(weighted)": [
                cid for cid, _ in retriever._weighted_fuse(
                    [(dense_scored, DENSE_WEIGHT), (bm25bc_scored, SPARSE_WEIGHT)], TOP_K
                )
            ],
        }

        unique_ids = {cid for ids in ranked.values() for cid in ids}
        with SessionLocal() as session:
            rows = session.execute(
                text("SELECT id, text FROM chunks WHERE id = ANY(:ids)"), {"ids": list(unique_ids)}
            ).fetchall()
        id_to_text = dict(rows)

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
