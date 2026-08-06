"""가중합 융합의 dense/sparse 가중치를 스윕해서 최적값을 찾는다.

검색만 하고(생성 없음) 골드셋 154문항 전체로 돌린다 - 문항당 dense/sparse
점수는 가중치와 무관하게 한 번만 계산해서 재사용하므로 가중치를 몇 개를 스윕하든
추가 GPU/DB 비용이 거의 없다.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.session import SessionLocal
from src.evaluation.retrieval_metrics import recall_at_k_by_value, reciprocal_rank_by_value
from src.retrieval.retriever import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "weight_sweep.json"

CANDIDATE_N = 100
TOP_K = 10
DENSE_WEIGHTS = [0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    items = json.loads(GOLDEN.read_text())
    print(f"골드셋 {len(items)}문항 로드", flush=True)

    retriever = Retriever()

    per_weight_rows = {dw: [] for dw in DENSE_WEIGHTS}
    t0 = time.time()

    for i, item in enumerate(items, 1):
        q, corp, expected_value = item["question"], item["corp_name"], item["expected_value"]

        dense_vec, sparse_vec = retriever.embedder.embed_query(q)
        dense_scored = retriever._dense_search_scored(dense_vec, CANDIDATE_N, corp)
        sparse_scored = retriever._sparse_search_scored(sparse_vec, CANDIDATE_N, corp)

        # 이 문항에서 나올 수 있는 모든 후보 id의 텍스트를 한 번에 조회
        unique_ids = {cid for cid, _ in dense_scored} | {cid for cid, _ in sparse_scored}
        with SessionLocal() as session:
            rows = session.execute(
                text("SELECT id, text FROM chunks WHERE id = ANY(:ids)"), {"ids": list(unique_ids)}
            ).fetchall()
        id_to_text = dict(rows)

        for dw in DENSE_WEIGHTS:
            fused = retriever._weighted_fuse([(dense_scored, dw), (sparse_scored, 1 - dw)], TOP_K)
            ids = [cid for cid, _ in fused]
            per_weight_rows[dw].append(
                {
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
            "value_recall@5": sum(r["value_recall@5"] for r in rows) / n,
            "value_recall@10": sum(r["value_recall@10"] for r in rows) / n,
            "value_mrr": sum(r["value_rr"] for r in rows) / n,
            "n": n,
        }

    summary = {f"dense={dw}/sparse={1 - dw:.1f}": _agg(rows) for dw, rows in per_weight_rows.items()}
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
