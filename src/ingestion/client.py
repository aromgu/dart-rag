import re

import requests

from src.config import settings
from src.ingestion.exceptions import DartApiError


class DartClient:
    """DART Open API(https://opendart.fss.or.kr)에 대한 얇은 래퍼."""

    BASE_URL = "https://opendart.fss.or.kr/api"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key or settings.dart_api_key
        if not self.api_key:
            raise ValueError("DART_API_KEY is not set (check your .env file)")
        self.session = session or requests.Session()

    def get_json(self, endpoint: str, **params) -> dict:
        response = self._request(endpoint, params)
        data = response.json()
        status = data.get("status")
        if status != "000":
            raise DartApiError(status, data.get("message", ""))
        return data

    def get_bytes(self, endpoint: str, **params) -> bytes:
        response = self._request(endpoint, params)
        content = response.content
        # 에러 응답의 Content-Type이 항상 json은 아니었다(예: document.xml에서 파일이
        # 없는 경우 status=014인 XML을 돌려줌 - Content-Type으로 구분하려다 놓쳤던 버그).
        # zip 파일 시그니처(PK)로 시작하지 않으면 무조건 에러 응답으로 본다.
        if not content.startswith(b"PK"):
            raise DartApiError(*self._parse_error(response, content))
        return content

    @staticmethod
    def _parse_error(response: requests.Response, content: bytes) -> tuple[str, str]:
        if "json" in response.headers.get("Content-Type", ""):
            data = response.json()
            return data.get("status", "unknown"), data.get("message", "")

        text = content.decode("utf-8", errors="replace")
        status_match = re.search(r"<status>(.*?)</status>", text)
        message_match = re.search(r"<message>(.*?)</message>", text)
        return (
            status_match.group(1) if status_match else "unknown",
            message_match.group(1) if message_match else text[:200],
        )

    def _request(self, endpoint: str, params: dict) -> requests.Response:
        query = {k: v for k, v in params.items() if v is not None}
        query["crtfc_key"] = self.api_key
        response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=query, timeout=30)
        response.raise_for_status()
        return response
