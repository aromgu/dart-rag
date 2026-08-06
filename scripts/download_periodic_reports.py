"""회사당 가장 최근 사업보고서 1건씩의 원본 문서(zip)를 data/raw/documents/에 저장한다.

최초엔 정기공시 전체(724건)를 받으려 했지만 노이즈가 많고 개수가 과해서, 회사당
가장 최근 사업보고서([기재정정] 포함 최신본) 1건으로 범위를 좁혔다 - 자세한 경위는
docs/eda.md 참고. data/processed/latest_annual_reports.csv를 먼저 만들어야 한다
(scripts/build_latest_annual_reports.py).

이미 받은 rcept_no는 건너뛰어서 중간에 끊겨도 재실행하면 이어받기가 된다.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.client import DartClient
from src.ingestion.documents import fetch_document_zip
from src.ingestion.exceptions import DartApiError

TARGETS_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "latest_annual_reports.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "documents"


def main():
    targets = pd.read_csv(TARGETS_CSV, dtype=str)
    print(f"대상 사업보고서: {len(targets)}건")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = DartClient()

    failed = []
    skipped = 0
    downloaded = 0

    for _, row in tqdm(list(targets.iterrows())):
        rcept_no = row["rcept_no"]
        out_path = OUTPUT_DIR / f"{rcept_no}.zip"
        if out_path.exists():
            skipped += 1
            continue

        try:
            zip_bytes = fetch_document_zip(rcept_no, client)
            # 중간에 kill되거나 죽어도 out_path에는 완성된 파일만 남도록 임시파일에 쓰고 rename한다
            tmp_path = out_path.with_suffix(".zip.tmp")
            tmp_path.write_bytes(zip_bytes)
            tmp_path.rename(out_path)
            downloaded += 1
        except (DartApiError, requests.exceptions.RequestException) as e:
            # 한 건이 느리거나 실패해도 몇 시간짜리 배치 전체가 죽지 않도록 넓게 잡아서 넘어간다
            failed.append((rcept_no, row["corp_name"], row["report_nm"], str(e)))
            time.sleep(1)

    print(f"새로 다운로드: {downloaded}건, 이미 있어서 건너뜀: {skipped}건, 실패: {len(failed)}건")
    for rcept_no, corp_name, report_nm, msg in failed:
        print(f"  {corp_name} / {report_nm} ({rcept_no}): {msg}")


if __name__ == "__main__":
    main()
