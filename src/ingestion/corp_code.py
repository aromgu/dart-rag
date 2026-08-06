import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.ingestion.client import DartClient

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CORP_CODE_CACHE = RAW_DATA_DIR / "corp_code.csv"


def parse_corp_code(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_bytes = zf.read("CORPCODE.xml")

    root = ET.fromstring(xml_bytes)
    rows = [
        {
            "corp_code": elem.findtext("corp_code"),
            "corp_name": elem.findtext("corp_name"),
            "stock_code": (elem.findtext("stock_code") or "").strip(),
            "modify_date": elem.findtext("modify_date"),
        }
        for elem in root.iter("list")
    ]
    return pd.DataFrame(rows)


def load_corp_codes(client: DartClient | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """corp_code -> corp_name 전체 매핑을 로드하고 data/raw/corp_code.csv에 캐싱한다."""
    if CORP_CODE_CACHE.exists() and not force_refresh:
        return pd.read_csv(CORP_CODE_CACHE, dtype=str, keep_default_na=False)

    client = client or DartClient()
    zip_bytes = client.get_bytes("corpCode.xml")
    df = parse_corp_code(zip_bytes)

    CORP_CODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORP_CODE_CACHE, index=False)
    return df


def find_corp_code(corp_name: str, df: pd.DataFrame | None = None, listed_only: bool = True) -> pd.DataFrame:
    """회사명(부분 일치, 대소문자 구분)으로 corp_code를 조회한다."""
    df = df if df is not None else load_corp_codes()
    matches = df[df["corp_name"].str.contains(corp_name, na=False)]
    if listed_only:
        matches = matches[matches["stock_code"] != ""]
    return matches
