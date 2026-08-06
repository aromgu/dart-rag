"""확장된 골드셋(199문항, 당기순이익 포함)으로 프로덕션 검색 설정을 재검증한다.

Retriever.search()를 그대로 쓴다(dense+sparse 가중합 + breadcrumb 포함
cross-encoder 재순위 - 지금 프로덕션 기본값). 계정과목(metric)별로 나눠서
당기순이익이 매출액/영업이익과 다르게 나오는지 확인한다.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.retrieval_metrics import recall_at_k_by_value, reciprocal_rank_by_value
from src.retrieval.retriever import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "expanded_golden_eval.json"

TOP_K = 5


def main() -> None:
    items = json.loads(GOLDEN.read_text())
    print(f"골드셋 {len(items)}문항 로드", flush=True)

    retriever = Retriever()  # 프로덕션 기본값: 가중합 + reranking

    rows_by_metric: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()

    for i, item in enumerate(items, 1):
        q, corp, expected_value, metric = (
            item["question"],
            item["corp_name"],
            item["expected_value"],
            item["metric"],
        )
        chunks = retriever.search(q, top_k=TOP_K, corp_name=corp)
        id_to_text = {i: c["text"] for i, c in enumerate(chunks)}
        ids = list(range(len(chunks)))

        rows_by_metric[metric].append(
            {
                "value_recall@5": recall_at_k_by_value(ids, expected_value, 5, id_to_text),
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
            "value_mrr": sum(r["value_rr"] for r in rows) / n,
            "n": n,
        }

    summary = {metric: _agg(rows) for metric, rows in rows_by_metric.items()}
    all_rows = [r for rows in rows_by_metric.values() for r in rows]
    summary["전체"] = _agg(all_rows)

    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
