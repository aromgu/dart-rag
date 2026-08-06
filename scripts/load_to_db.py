"""파싱 -> 청킹 -> 임베딩 -> DB 적재를 잇는 로더.

data/processed/latest_annual_reports.csv의 99개 공시를 순회하며
Company/Disclosure/Chunk 행을 채운다. 문서는 이미 data/raw/documents/에
받아둔 zip을 읽으므로 DART API를 다시 호출하지 않는다.

캐싱/중복방지: 공시별로 이미 저장된 (section_path, chunk_index)를 먼저
조회해서, 새로 생긴 청크만 임베딩/삽입한다 - 재실행해도 이미 끝난 공시는
임베딩(GPU) 비용이 들지 않는다.

--limit N 으로 문서 수를 제한해서 소규모로 먼저 검증할 수 있다.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import chunk_blocks
from src.db import models  # noqa: F401
from src.db.models import Chunk, Company, Disclosure
from src.db.session import SessionLocal
from src.embedding.embedder import Embedder
from src.ingestion.documents import extract_documents, parse_report_blocks

DOC_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "documents"
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "latest_annual_reports.csv"


def _load_local_blocks(rcept_no: str) -> list[dict]:
    """이미 받아둔 zip을 읽어서 블록 리스트로 변환한다 (API 재호출 없음)."""
    zip_bytes = (DOC_DIR / f"{rcept_no}.zip").read_bytes()
    docs = extract_documents(zip_bytes)
    blocks: list[dict] = []
    for _, content in sorted(docs.items()):
        blocks.extend(parse_report_blocks(content))
    return blocks


def _get_or_create_company(session, corp_code: str, corp_name: str, stock_code: str) -> Company:
    company = session.get(Company, corp_code)
    if company is None:
        company = Company(corp_code=corp_code, corp_name=corp_name, stock_code=stock_code)
        session.add(company)
    return company


def _get_or_create_disclosure(session, row: dict, company: Company) -> Disclosure:
    disclosure = session.get(Disclosure, row["rcept_no"])
    if disclosure is None:
        rcept_dt = time.strptime(row["rcept_dt"], "%Y%m%d")
        disclosure = Disclosure(
            rcept_no=row["rcept_no"],
            corp_code=company.corp_code,
            report_nm=row["report_nm"],
            rcept_dt=time.strftime("%Y-%m-%d", rcept_dt),
        )
        session.add(disclosure)
    return disclosure


def load_document(session, embedder: Embedder, row: dict) -> tuple[int, int]:
    """공시 1건을 적재한다. (신규 삽입 청크 수, 스킵한 청크 수)를 반환한다."""
    company = _get_or_create_company(session, row["corp_code"], row["corp_name"], row["stock_code"])
    disclosure = _get_or_create_disclosure(session, row, company)
    session.flush()  # 아래 조회/FK가 방금 만든 행을 볼 수 있게

    existing = {
        (tuple(section_path), chunk_index)
        for section_path, chunk_index in session.query(Chunk.section_path, Chunk.chunk_index)
        .filter(Chunk.rcept_no == disclosure.rcept_no)
        .all()
    }

    blocks = _load_local_blocks(row["rcept_no"])
    chunks = chunk_blocks(blocks)

    new_chunks = [c for c in chunks if (tuple(c["section_path"]), c["chunk_index"]) not in existing]
    skipped = len(chunks) - len(new_chunks)

    if not new_chunks:
        return 0, skipped

    dense_vecs, sparse_vecs = embedder.embed_chunks(new_chunks, corp_name=row["corp_name"])
    for chunk, dense, sparse in zip(new_chunks, dense_vecs, sparse_vecs):
        session.add(
            Chunk(
                rcept_no=disclosure.rcept_no,
                section_path=chunk["section_path"],
                chunk_type=chunk["type"],
                chunk_index=chunk["chunk_index"],
                text=chunk["text"],
                embedding=dense,
                sparse_embedding=sparse,
            )
        )
    session.commit()
    return len(new_chunks), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 문서 수를 제한 (검증용)")
    args = parser.parse_args()

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    print(f"총 {len(rows)}개 공시 적재 시작 (임베딩 모델 로딩 중...)", flush=True)
    t_load = time.time()
    embedder = Embedder()
    print(f"모델 로드 완료: {time.time() - t_load:.1f}초", flush=True)

    session = SessionLocal()
    total_new, total_skipped = 0, 0
    durations = []

    try:
        for i, row in enumerate(rows, 1):
            t0 = time.time()
            n_new, n_skip = load_document(session, embedder, row)
            dt = time.time() - t0
            durations.append(dt)
            total_new += n_new
            total_skipped += n_skip

            avg = sum(durations) / len(durations)
            eta = avg * (len(rows) - i)
            print(
                f"[{i}/{len(rows)}] {row['corp_name']}: {dt:.1f}초, "
                f"신규 {n_new}개 / 스킵 {n_skip}개 (누적 신규 {total_new}개) "
                f"- 예상 잔여시간 {eta / 60:.1f}분",
                flush=True,
            )
    finally:
        session.close()

    print(f"완료: 신규 {total_new}개, 스킵(이미 적재됨) {total_skipped}개")


if __name__ == "__main__":
    main()
