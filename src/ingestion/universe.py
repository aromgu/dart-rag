from pathlib import Path

import pandas as pd

from src.ingestion.client import DartClient
from src.ingestion.corp_code import load_corp_codes

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
UNIVERSE_SEED = DATA_DIR / "kospi100_universe.csv"
UNIVERSE_RESOLVED = DATA_DIR / "processed" / "kospi100_corp_codes.csv"


def load_universe_seed() -> pd.DataFrame:
    """추적 대상으로 큐레이션한 코스피 대형주 종목명 리스트를 로드한다 (data/kospi100_universe.csv)."""
    return pd.read_csv(UNIVERSE_SEED, dtype=str)


def resolve_universe(client: DartClient | None = None, corp_codes: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """큐레이션한 종목명 리스트를 DART corp_code 목록과 매칭한다.

    (resolved_df, unmatched_names)를 반환한다. resolved_df는 매칭된 회사마다
    DART corp_code, corp_name, stock_code를 담은 한 행씩을 가진다.
    """
    seed = load_universe_seed()
    corp_codes = corp_codes if corp_codes is not None else load_corp_codes(client)

    merged = seed.merge(corp_codes, on="corp_name", how="left")

    # DART는 상장폐지/합병된 옛 법인과 현재 상장된 법인이 같은 corp_name을
    # 공유하는 경우가 있어(예: 합병 전 껍데기 법인과 현재 상장 법인),
    # 이름만으로는 매칭이 모호할 수 있다. stock_code가 있는(=현재 상장된)
    # 항목을 우선하고, 그래도 동률이면 modify_date가 가장 최근인 것을 쓴다.
    merged["_is_listed"] = merged["stock_code"].fillna("") != ""
    merged = merged.sort_values(["_is_listed", "modify_date"], ascending=[False, False])
    merged = merged.drop_duplicates(subset="corp_name", keep="first")
    merged = merged.sort_index().drop(columns=["_is_listed"])

    resolved = merged.dropna(subset=["corp_code"]).drop(columns=["source_rank"])
    unmatched = merged.loc[merged["corp_code"].isna(), "corp_name"].tolist()

    return resolved.reset_index(drop=True), unmatched


def save_resolved_universe(resolved: pd.DataFrame) -> None:
    UNIVERSE_RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(UNIVERSE_RESOLVED, index=False)
