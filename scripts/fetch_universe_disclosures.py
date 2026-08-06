"""EDA용: kospi100_corp_codes.csv의 100개 기업에 대해 최근 1년치 공시 목록을 수집해서 저장한다."""

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.client import DartClient
from src.ingestion.disclosures import search_disclosures
from src.ingestion.exceptions import DartApiError

RESOLVED_UNIVERSE = Path(__file__).resolve().parents[1] / "data" / "processed" / "kospi100_corp_codes.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "disclosures_1y.csv"

BGN_DE = "20250101"
END_DE = "20260805"


def main():
    universe = pd.read_csv(RESOLVED_UNIVERSE, dtype=str)
    client = DartClient()

    all_rows = []
    failed = []

    for corp_code, corp_name in tqdm(list(zip(universe["corp_code"], universe["corp_name"]))):
        try:
            rows = search_disclosures(client, corp_code=corp_code, bgn_de=BGN_DE, end_de=END_DE)
            all_rows.extend(rows)
        except DartApiError as e:
            failed.append((corp_code, corp_name, str(e)))
            time.sleep(1)

    df = pd.DataFrame(all_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"총 공시 건수: {len(df)}")
    print(f"실패한 기업: {len(failed)}")
    for corp_code, corp_name, msg in failed:
        print(f"  {corp_name}({corp_code}): {msg}")


if __name__ == "__main__":
    main()
