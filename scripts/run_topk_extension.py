"""top_k를 10 이상으로 더 올렸을 때 최종 답변 정확도의 한계점을 찾는다.

run_e2e_ablation.py와 같은 골드셋 샘플(seed=42, 30문항)·같은 프롬프트로
top_k=15/20/25만 추가로 측정한다(k=3/5/10은 이미 측정 완료 - docs/experiments.md 실험 5).
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
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "topk_extension.json"

SAMPLE_N = 30
TOP_KS = [15, 20, 25]


def main() -> None:
    all_items = json.loads(GOLDEN.read_text())
    random.seed(42)  # run_e2e_ablation.py와 동일한 샘플을 재현
    items = random.sample(all_items, SAMPLE_N)
    print(f"골드셋 {len(all_items)}문항 중 {SAMPLE_N}문항 샘플링 (동일 seed)", flush=True)

    retriever = Retriever()
    generator = Generator()

    results = {f"hybrid_k{k}": [] for k in TOP_KS}
    t0 = time.time()
    total = len(items) * len(TOP_KS)
    done = 0

    for item in items:
        q, corp, expected_value = item["question"], item["corp_name"], item["expected_value"]

        for k in TOP_KS:
            chunks = retriever.search(q, top_k=k, corp_name=corp, candidate_n=100)
            answer = generator.generate(q, chunks)
            correct = expected_value in answer
            results[f"hybrid_k{k}"].append({"question": q, "answer": answer, "correct": correct})

            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] 경과 {elapsed:.1f}초, 예상 잔여 {eta:.1f}초", flush=True)

    summary = {label: {"accuracy": sum(r["correct"] for r in rows) / len(rows), "n": len(rows)} for label, rows in results.items()}
    OUTPUT.write_text(json.dumps({"summary": summary, "per_item": results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
