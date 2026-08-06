"""표 재포맷(_reformat_table_text) + few-shot 예시 적용 후 end-to-end 정확도를
개선 전 baseline(53.3%, docs/experiments.md 실험 11)과 비교한다.

같은 골드셋 샘플(seed=42, 30문항), 같은 프로덕션 설정(가중합+재순위, top_k=5)을 쓴다.
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
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "reformat_fewshot_eval.json"

SAMPLE_N = 30
TOP_K = 5


def main() -> None:
    all_items = json.loads(GOLDEN.read_text())
    random.seed(42)
    items = random.sample([it for it in all_items if it["metric"] in ("매출액", "영업이익")], SAMPLE_N)
    print(f"30문항 샘플링 (동일 seed, 매출액/영업이익만 - 실험 11과 동일 모집단)", flush=True)

    retriever = Retriever()
    generator = Generator()

    rows = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        q, corp, expected_value = item["question"], item["corp_name"], item["expected_value"]
        chunks = retriever.search(q, top_k=TOP_K, corp_name=corp)
        answer = generator.generate(q, chunks)
        correct = expected_value in answer
        rows.append({"question": q, "answer": answer, "correct": correct})

        if i % 10 == 0 or i == len(items):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(items) - i)
            print(f"[{i}/{len(items)}] 경과 {elapsed:.1f}초, 예상 잔여 {eta:.1f}초", flush=True)

    accuracy = sum(r["correct"] for r in rows) / len(rows)
    result = {"accuracy": accuracy, "n": len(rows), "baseline_before": 0.5333333333333333}
    OUTPUT.write_text(json.dumps({"summary": result, "per_item": rows}, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
