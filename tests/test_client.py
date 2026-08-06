import pytest

from src.ingestion.client import DartClient
from src.ingestion.exceptions import DartApiError


class FakeResponse:
    def __init__(self, content: bytes, headers: dict, json_data=None):
        self.content = content
        self.headers = headers
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, response: FakeResponse):
        self._response = response

    def get(self, url, params=None, timeout=None):
        return self._response


def _client(response: FakeResponse) -> DartClient:
    return DartClient(api_key="dummy", session=FakeSession(response))


def test_get_bytes_returns_content_when_it_looks_like_a_zip():
    response = FakeResponse(b"PK\x03\x04...zip bytes...", headers={"Content-Type": "application/octet-stream"})
    assert _client(response).get_bytes("document.xml", rcept_no="1") == b"PK\x03\x04...zip bytes..."


def test_get_bytes_raises_on_json_error_response():
    response = FakeResponse(
        b'{"status": "013", "message": "..."}',
        headers={"Content-Type": "application/json"},
        json_data={"status": "013", "message": "조회된 데이터가 없습니다"},
    )
    with pytest.raises(DartApiError) as exc_info:
        _client(response).get_bytes("document.xml", rcept_no="1")
    assert exc_info.value.status == "013"


def test_get_bytes_raises_on_xml_error_response():
    # document.xml에서 rcept_no가 존재하지 않으면 DART가 JSON이 아니라 이런 XML을 돌려준다
    xml = '<?xml version="1.0" encoding="UTF-8"?><result><status>014</status><message>파일이 존재하지 않습니다.</message></result>'
    response = FakeResponse(xml.encode("utf-8"), headers={"Content-Type": "application/xml;charset=UTF-8"})
    with pytest.raises(DartApiError) as exc_info:
        _client(response).get_bytes("document.xml", rcept_no="1")
    assert exc_info.value.status == "014"
    assert exc_info.value.message == "파일이 존재하지 않습니다."
