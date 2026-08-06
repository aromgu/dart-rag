"""청킹 파라미터가 검색 품질에 미치는 영향을 소규모(5개 문서)로 저렴하게 비교.

99개 문서 전체 재청킹+재임베딩(~45분)은 비싸서, 골드셋에 있는 회사 중 5개
문서만 골라 두 가지 target_chars 설정으로 각각 재청킹+재임베딩하고, 완전히
별도의 메모리 인덱스(프로덕션 DB 건드리지 않음)로 검색 품질을 비교한다.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import chunk_blocks
from src.embedding.embedder import Embedder
from src.ingestion.documents import extract_documents, parse_report_blocks
from src.retrieval.retriever import Retriever

DOC_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "documents"
REPORTS_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "latest_annual_reports.csv"
GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "chunking_ablation.json"

N_COMPANIES = 15
TOP_K = 5

# 1차 파일럿(5문서/10문항)에서 900과 1500이 동률이고 400이 더 나빴던 걸 보고,
# 극단값 대신 기본값(900) 주변을 촘촘하게 스윕해서 진짜 최적점을 좁힌다.
VARIANTS = {
    "600": dict(target_chars=600, table_max_chars=1500, small_table_chars=200),
    "900 (기본값)": dict(target_chars=900, table_max_chars=1500, small_table_chars=200),
    "1200": dict(target_chars=1200, table_max_chars=1500, small_table_chars=200),
}


def _load_rcept_no_by_corp() -> dict[str, str]:
    import csv

    with REPORTS_CSV.open(encoding="utf-8") as f:
        return {row["corp_name"]: row["rcept_no"] for row in csv.DictReader(f)}


def main() -> None:
    golden = json.loads(GOLDEN.read_text())
    rcept_by_corp = _load_rcept_no_by_corp()

    companies = sorted({it["corp_name"] for it in golden})[:N_COMPANIES]
    print(f"대상 회사({N_COMPANIES}개): {companies}", flush=True)

    questions = [it for it in golden if it["corp_name"] in companies]
    print(f"평가 문항 {len(questions)}개", flush=True)

    # 문서별 블록은 청킹 파라미터와 무관하므로 한 번만 파싱해서 재사용한다.
    blocks_by_corp = {}
    for corp in companies:
        rcept_no = rcept_by_corp[corp]
        zip_bytes = (DOC_DIR / f"{rcept_no}.zip").read_bytes()
        docs = extract_documents(zip_bytes)
        blocks = []
        for _, content in sorted(docs.items()):
            blocks.extend(parse_report_blocks(content))
        blocks_by_corp[corp] = blocks

    embedder = Embedder()
    results = {}

    for variant_name, params in VARIANTS.items():
        t0 = time.time()
        print(f"\n=== {variant_name} ===", flush=True)

        chunks_by_corp = {corp: chunk_blocks(blocks_by_corp[corp], **params) for corp in companies}
        total_chunks = sum(len(v) for v in chunks_by_corp.values())
        print(f"청크 {total_chunks}개, 임베딩 시작...", flush=True)

        dense_by_corp, sparse_by_corp, text_by_corp = {}, {}, {}
        for corp in companies:
            chunks = chunks_by_corp[corp]
            dense_vecs, sparse_vecs = embedder.embed_chunks(chunks, corp_name=corp)
            dense_by_corp[corp] = np.array(dense_vecs)
            sparse_by_corp[corp] = sparse_vecs
            text_by_corp[corp] = [c["text"] for c in chunks]
        print(f"임베딩 완료 ({time.time() - t0:.1f}초)", flush=True)

        correct = 0
        for item in questions:
            corp, expected_value = item["corp_name"], item["expected_value"]
            dense_vec, sparse_vec = embedder.embed_query(item["question"])

            dense_scores = dense_by_corp[corp] @ np.array(dense_vec)
            dense_scored = list(enumerate(dense_scores.tolist()))

            sparse_scored = [
                (i, sum(sparse_vec.get(tok, 0.0) * w for tok, w in sv.items()))
                for i, sv in enumerate(sparse_by_corp[corp])
            ]

            fused = Retriever._weighted_fuse([(dense_scored, 0.7), (sparse_scored, 0.3)], TOP_K)
            hit = any(expected_value in text_by_corp[corp][idx] for idx, _ in fused)
            correct += int(hit)

        acc = correct / len(questions)
        results[variant_name] = {"value_recall@5": acc, "total_chunks": total_chunks, "n": len(questions)}
        print(f"{variant_name}: value_recall@5={acc:.3f} ({correct}/{len(questions)}), 총 {time.time() - t0:.1f}초", flush=True)

    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
