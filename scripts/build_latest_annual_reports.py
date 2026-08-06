"""data/processed/disclosures_1y.csv에서 회사당 가장 최근 사업보고서 1건만 뽑아
data/processed/latest_annual_reports.csv로 저장한다.

[기재정정]/[첨부정정]/[첨부추가]가 붙은 정정본은 원본보다 항상 나중에 접수되므로,
접수일(rcept_dt) 기준으로 가장 최근 것을 고르면 자동으로 최신 정정본이 선택된다.
"해외증권거래소등에신고한사업보고서등의국내신고"처럼 이름에 "사업보고서"가 들어가지만
실제로는 다른 종류의 공시인 건 정규식으로 제외한다.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISCLOSURES = Path(__file__).resolve().parents[1] / "data" / "processed" / "disclosures_1y.csv"
UNIVERSE = Path(__file__).resolve().parents[1] / "data" / "processed" / "kospi100_corp_codes.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "latest_annual_reports.csv"

ANNUAL_REPORT_PATTERN = r"^(\[기재정정\]|\[첨부정정\]|\[첨부추가\])?사업보고서\s*\("


def main():
    df = pd.read_csv(DISCLOSURES, dtype=str)
    annual_reports = df[df["report_nm"].str.match(ANNUAL_REPORT_PATTERN, na=False)]

    latest = annual_reports.sort_values("rcept_dt").groupby("corp_name", as_index=False).last()
    latest.to_csv(OUTPUT, index=False)

    universe_names = set(pd.read_csv(UNIVERSE, dtype=str)["corp_name"])
    missing = universe_names - set(latest["corp_name"])

    print(f"사업보고서류 전체: {len(annual_reports)}건")
    print(f"회사당 최신 1건: {len(latest)}건 ({latest['corp_name'].nunique()}개 기업)")
    if missing:
        print(f"사업보고서가 없는 기업: {sorted(missing)}")


if __name__ == "__main__":
    main()
