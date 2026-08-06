"""검색→생성까지 이어진 end-to-end 답변 정확도 비교.

지금까지는 검색(Recall@k/MRR)만 비교했는데, 검색 품질 차이가 실제 LLM
최종 답변 정확도에도 이어지는지는 따로 확인한 적이 없다. 골드셋 일부를
샘플링해서 (a) 검색 조건(하이브리드 vs dense 단독)과 (b) top_k(LLM에 넘기는
청크 수)를 같이 비교한다. 정답 판정은 결정론적 방식(생성된 답변에 정답
수치가 실제로 포함됐는가)을 쓴다 - docs/study.md "RAG 평가 기법" 참고.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.generator import Generator
from src.retrieval.retriever import Retriever

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "e2e_ablation.json"

SAMPLE_N = 30
CANDIDATE_N = 100

CONDITIONS = [
    ("hybrid_k3", "hybrid", 3),
    ("hybrid_k5", "hybrid", 5),
    ("hybrid_k10", "hybrid", 10),
    ("dense_k5", "dense", 5),
]


def _dense_only_chunks(retriever: Retriever, query: str, top_k: int, corp_name: str) -> list[dict]:
    dense_vec, _ = retriever.embedder.embed_query(query)
    scored = retriever._dense_search_scored(dense_vec, CANDIDATE_N, corp_name)[:top_k]
    return retriever._fetch_chunks(scored)


def main() -> None:
    all_items = json.loads(GOLDEN.read_text())
    random.seed(42)
    items = random.sample(all_items, SAMPLE_N)
    print(f"골드셋 {len(all_items)}문항 중 {SAMPLE_N}문항 샘플링", flush=True)

    retriever = Retriever()
    generator = Generator()

    results = {label: [] for label, _, _ in CONDITIONS}
    t0 = time.time()
    total = len(items) * len(CONDITIONS)
    done = 0

    for item in items:
        q, corp, expected_value = item["question"], item["corp_name"], item["expected_value"]

        for label, mode, top_k in CONDITIONS:
            if mode == "hybrid":
                chunks = retriever.search(q, top_k=top_k, corp_name=corp, candidate_n=CANDIDATE_N)
            else:
                chunks = _dense_only_chunks(retriever, q, top_k, corp)

            answer = generator.generate(q, chunks)
            correct = expected_value in answer
            results[label].append({"question": q, "answer": answer, "correct": correct})

            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] 경과 {elapsed:.1f}초, 예상 잔여 {eta:.1f}초", flush=True)

    summary = {
        label: {
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "n": len(rows),
        }
        for label, rows in results.items()
    }

    OUTPUT.write_text(json.dumps({"summary": summary, "per_item": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
