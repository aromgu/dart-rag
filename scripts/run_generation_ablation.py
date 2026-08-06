"""질의 재작성(query rewriting)과 자기검증 재시도(self-verification retry)가
end-to-end 답변 정확도에 도움이 되는지 비교한다.

docs/experiments.md 실험 4에서 확인한 병목(검색은 맞는 청크를 찾아왔는데
생성이 문맥 안에서 엉뚱한 숫자를 고름)을 겨냥한 두 가지 대응책을 검증한다.
run_e2e_ablation.py와 같은 골드셋 샘플(seed=42, 30문항)을 쓴다.
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
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "generation_ablation.json"

SAMPLE_N = 30
TOP_K = 5

CONDITIONS = ["baseline", "verify", "rewrite", "rewrite+verify"]


def main() -> None:
    all_items = json.loads(GOLDEN.read_text())
    random.seed(42)
    items = random.sample(all_items, SAMPLE_N)
    print(f"골드셋 {len(all_items)}문항 중 {SAMPLE_N}문항 샘플링 (동일 seed)", flush=True)

    retriever = Retriever()  # reranking 기본 적용
    generator = Generator()

    results = {c: [] for c in CONDITIONS}
    t0 = time.time()
    total = len(items) * len(CONDITIONS)
    done = 0

    for item in items:
        q, corp, expected_value = item["question"], item["corp_name"], item["expected_value"]

        for cond in CONDITIONS:
            use_rewrite = "rewrite" in cond
            use_verify = "verify" in cond

            search_query = generator.rewrite_query(q) if use_rewrite else q
            chunks = retriever.search(search_query, top_k=TOP_K, corp_name=corp)
            answer = (
                generator.generate_verified(q, chunks) if use_verify else generator.generate(q, chunks)
            )
            correct = expected_value in answer
            results[cond].append(
                {"question": q, "search_query": search_query, "answer": answer, "correct": correct}
            )

            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] 경과 {elapsed:.1f}초, 예상 잔여 {eta:.1f}초", flush=True)

    summary = {
        c: {"accuracy": sum(r["correct"] for r in rows) / len(rows), "n": len(rows)}
        for c, rows in results.items()
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "per_item": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
