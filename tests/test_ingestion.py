import io
import zipfile

import pytest

from src.ingestion.client import DartClient
from src.ingestion.corp_code import find_corp_code, parse_corp_code
from src.ingestion.disclosures import search_disclosures
from src.ingestion.documents import extract_documents, fetch_document_text, xml_to_text
from src.ingestion.exceptions import DartApiError


CORP_CODE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list>
    <corp_code>00126380</corp_code>
    <corp_name>삼성전자</corp_name>
    <stock_code>005930</stock_code>
    <modify_date>20240101</modify_date>
  </list>
  <list>
    <corp_code>00164779</corp_code>
    <corp_name>비상장회사</corp_name>
    <stock_code></stock_code>
    <modify_date>20240101</modify_date>
  </list>
</result>
"""


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


class FakeClient:
    def __init__(self, json_data=None, bytes_data=None, json_error=None, bytes_error=None):
        self._json_data = json_data
        self._bytes_data = bytes_data
        self._json_error = json_error
        self._bytes_error = bytes_error
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, endpoint, **params):
        self.calls.append((endpoint, params))
        if self._json_error:
            raise self._json_error
        return self._json_data

    def get_bytes(self, endpoint, **params):
        self.calls.append((endpoint, params))
        if self._bytes_error:
            raise self._bytes_error
        return self._bytes_data


def test_parse_corp_code():
    zip_bytes = _zip_bytes({"CORPCODE.xml": CORP_CODE_XML.encode("utf-8")})
    df = parse_corp_code(zip_bytes)
    assert list(df["corp_name"]) == ["삼성전자", "비상장회사"]
    assert df.iloc[0]["stock_code"] == "005930"


def test_find_corp_code_filters_unlisted_by_default():
    zip_bytes = _zip_bytes({"CORPCODE.xml": CORP_CODE_XML.encode("utf-8")})
    df = parse_corp_code(zip_bytes)

    matches = find_corp_code("삼성전자", df=df)
    assert len(matches) == 1
    assert matches.iloc[0]["corp_code"] == "00126380"

    all_matches = find_corp_code("회사", df=df, listed_only=False)
    assert len(all_matches) == 1  # "회사"가 들어간 건 "비상장회사"뿐


def test_search_disclosures_paginates_until_last_page():
    client = FakeClient(json_data={"status": "000", "list": [{"rcept_no": "1"}], "total_page": 1})
    results = search_disclosures(client, corp_code="00126380", bgn_de="20240101")
    assert results == [{"rcept_no": "1"}]
    assert client.calls[0][1]["page_no"] == 1


def test_search_disclosures_returns_empty_on_no_data():
    client = FakeClient(json_error=DartApiError("013", "조회된 데이타가 없습니다"))
    assert search_disclosures(client, corp_code="00126380") == []


def test_search_disclosures_reraises_other_errors():
    client = FakeClient(json_error=DartApiError("020", "요청 제한을 초과하였습니다"))
    with pytest.raises(DartApiError):
        search_disclosures(client, corp_code="00126380")


def test_xml_to_text_strips_markup_and_blank_lines():
    xml_str = "<DOCUMENT><P>  본문 내용  </P><P></P></DOCUMENT>"
    assert xml_to_text(xml_str) == "본문 내용"


def test_extract_documents_decodes_euckr():
    raw = "삼성전자".encode("euc-kr")
    zip_bytes = _zip_bytes({"20240101.xml": raw})
    docs = extract_documents(zip_bytes)
    assert docs["20240101.xml"] == "삼성전자"


def test_fetch_document_text_joins_parts_in_filename_order():
    zip_bytes = _zip_bytes(
        {
            "b.xml": "<DOCUMENT><P>둘째</P></DOCUMENT>".encode("euc-kr"),
            "a.xml": "<DOCUMENT><P>첫째</P></DOCUMENT>".encode("euc-kr"),
        }
    )
    client = FakeClient(bytes_data=zip_bytes)
    text = fetch_document_text("20240101000001", client)
    assert text == "첫째\n\n둘째"


def test_client_requires_api_key(monkeypatch):
    monkeypatch.setattr("src.ingestion.client.settings.dart_api_key", "")
    with pytest.raises(ValueError):
        DartClient(api_key="")
