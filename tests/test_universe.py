import pandas as pd

from src.ingestion.universe import load_universe_seed, resolve_universe


def test_universe_seed_has_100_unique_companies_no_etfs_or_preferred():
    seed = load_universe_seed()
    assert len(seed) == 100
    assert seed["corp_name"].is_unique
    assert not seed["corp_name"].str.contains("KODEX|TIGER", regex=True).any()
    assert "삼성전자우" not in set(seed["corp_name"])


def test_resolve_universe_matches_by_corp_name_and_reports_unmatched():
    corp_codes = pd.DataFrame(
        [
            {"corp_code": "00126380", "corp_name": "삼성전자", "stock_code": "005930", "modify_date": "20240101"},
            {"corp_code": "00164742", "corp_name": "SK하이닉스", "stock_code": "000660", "modify_date": "20240101"},
        ]
    )
    resolved, unmatched = resolve_universe(corp_codes=corp_codes)

    resolved_names = set(resolved["corp_name"])
    assert {"삼성전자", "SK하이닉스"} <= resolved_names
    assert "corp_code" in resolved.columns
    assert "source_rank" not in resolved.columns

    # 가짜 corp_codes 테이블에 없는 seed 회사들은 전부 unmatched로 나와야 한다
    assert len(unmatched) == len(load_universe_seed()) - 2
    assert "카카오" in unmatched
