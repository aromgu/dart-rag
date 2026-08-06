import io
import re
import zipfile

from bs4 import BeautifulSoup

from src.ingestion.client import DartClient

# 본문 중 "Data & Solution"처럼 이스케이프 안 된 & 가 그대로 들어있는 경우가 있어서
# XML 파서가 그 지점부터 태그 구조를 잃어버린다(예: SK하이닉스 사업보고서). 유효한
# 엔티티 참조(&amp; 등)가 아닌 & 만 골라서 미리 &amp;로 escape해준다.
_BARE_AMPERSAND = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)")

# "<주요자회사>", "<SK스퀘어>", "<Wholesale 부문>"처럼 소제목을 홑화살괄호 대신 <>로
# 그냥 써버려서 가짜 태그로 오인되는 경우가 다양하게 있다(닫는 태그가 없으니 파서가
# 뒤 구조를 다 잃어버림). "다음 글자가 영문자인가"나 "태그명 뒤에 공백이 오는가" 같은
# 문법적 휴리스틱으로는 "<Wholesale 부문>"처럼 진짜 태그(이름 뒤에 속성)처럼 생긴
# 경우를 못 거른다. 대신 99개 문서 전체에서 실제로 등장한 태그 이름 빈도를 세어보면
# 진짜 DART 태그(수백~수백만 회)와 이런 장식용 꺾쇠(대부분 1~10회, 전부 회사/사업
# 용어) 사이에 뚜렷한 빈도 단절이 있어서, 그걸로 뽑은 화이트리스트를 쓴다.
_KNOWN_TAGS = {
    "TD", "TE", "TR", "COL", "TH", "P", "TBODY", "TABLE", "COLGROUP", "TU", "THEAD",
    "SPAN", "PGBRK", "TITLE", "TABLE-GROUP", "SECTION-1", "SECTION-2", "SECTION-3",
    "LIBRARY", "EXTRACTION", "A", "IMAGE", "IMG", "IMG-CAPTION", "DOCUMENT",
    "DOCUMENT-NAME", "FORMULA-VERSION", "COMPANY-NAME", "SUMMARY", "BODY", "COVER",
    "COVER-TITLE", "CORRECTION",
}
_TAG_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _sanitize_xml(xml_str: str) -> str:
    xml_str = _BARE_AMPERSAND.sub("&amp;", xml_str)

    def replace(m: re.Match) -> str:
        # xml_str[m.end():]처럼 부분문자열을 새로 만들면 "<"가 나올 때마다 문서
        # 나머지 전체를 복사하게 돼서 큰 문서(대형 TABLE 문서는 "<"가 수만~수십만
        # 개)에서 O(n^2)로 터진다. re.match(string, pos)로 위치만 넘겨서 복사를 피한다.
        end = m.end()
        if xml_str[end : end + 1] in ("/", "!", "?"):
            return "<"
        name_match = _TAG_NAME.match(xml_str, end)
        if name_match and name_match.group(0).upper() in _KNOWN_TAGS:
            return "<"
        return "&lt;"

    return re.sub("<", replace, xml_str)


def fetch_document_zip(rcept_no: str, client: DartClient | None = None) -> bytes:
    client = client or DartClient()
    return client.get_bytes("document.xml", rcept_no=rcept_no)


def extract_documents(zip_bytes: bytes) -> dict[str, str]:
    """DART 문서 응답 zip을 {파일명: 디코딩된 텍스트}로 풀어낸다. 공시 원문은 EUC-KR 인코딩이다."""
    docs = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            raw = zf.read(name)
            try:
                docs[name] = raw.decode("euc-kr")
            except UnicodeDecodeError:
                docs[name] = raw.decode("utf-8", errors="replace")
    return docs


TABLE_CELL_TAGS = ["TD", "TE", "TU", "TH"]


def _cell_text(cell) -> str:
    """표 셀 하나의 텍스트를 뽑는다.

    셀 안에 표가 통째로 중첩된 경우(실제로 존재함 - 예: 부문별 실적을 한 셀 안에
    미니 표로 넣어놓은 경우)가 있는데, 그냥 get_text()를 쓰면 중첩 표의 모든 행이
    행 구분 없이 한 줄로 뭉개져서 수만 자짜리 의미 없는 텍스트가 된다. 중첩 표는
    먼저 떼어내서 별도로 직렬화한 뒤 대괄호로 붙여서, 바깥 표의 한 행 크기가
    안에 뭐가 들었든 안 폭발하게 한다.
    """
    nested_tables = cell.find_all("TABLE")
    nested_parts = []
    for nested in nested_tables:
        nested_parts.append(_table_to_text(nested).replace("\n", "; "))
        nested.extract()

    text = cell.get_text(" ", strip=True)
    if nested_parts:
        text = (text + " " + " / ".join(f"[{p}]" for p in nested_parts)).strip()
    return text


def _table_to_text(table) -> str:
    """TABLE 태그를 행 단위로 셀을 " | "로 이어붙인 텍스트로 직렬화한다.

    ROWSPAN/COLSPAN을 반영해 그리드를 완벽히 재구성하지는 않는다 - 사업보고서
    표(재무제표, 임원현황 등)를 검색/LLM 컨텍스트로 쓰기에 필요한 수준(행 안에서
    셀들의 상대적 위치)만 보존하면 충분하다고 보고 단순하게 처리한다.
    """
    # find_all("TR")은 기본적으로 재귀적이라 중첩 표 안의 행까지 다 딸려온다.
    # 그러면 중첩 표 내용이 (a) 부모 셀 텍스트에 대괄호로 한 번, (b) 여기서 또
    # 독립된 행으로 한 번, 이렇게 중복된다. 가장 가까운 TABLE 조상이 지금 이
    # table 자신인 행만 걸러서 중복을 막는다.
    lines = []
    rows = [tr for tr in table.find_all("TR") if tr.find_parent("TABLE") is table]
    for tr in rows:
        cells = tr.find_all(TABLE_CELL_TAGS, recursive=False)
        cell_texts = [_cell_text(c) for c in cells]
        if any(cell_texts):
            lines.append(" | ".join(cell_texts))
    return "\n".join(lines)


def parse_report_blocks(xml_str: str) -> list[dict]:
    """DART 사업보고서류 XML(SECTION/TITLE/P/TABLE 구조)을 블록 리스트로 파싱한다.

    각 블록은 {"section_path": [...], "type": "paragraph"|"table", "text": "..."}
    형태다. SECTION-1/2/3의 TITLE을 따라가며 목차 경로를 쌓고, 문단(P)과 표(TABLE)를
    별도 블록으로 분리해 둔다 - naive하게 전체를 get_text()로 펼치면 표 안의 숫자들이
    문맥 없이 뒤섞여버리고, 이후 청킹 단계에서 표 중간이 잘리는 문제가 생기기 때문이다.
    """
    soup = BeautifulSoup(_sanitize_xml(xml_str), "lxml-xml")
    blocks: list[dict] = []

    def walk(node, path: list[str]) -> None:
        for child in node.find_all(recursive=False):
            name = child.name
            if name and name.startswith("SECTION"):
                title = child.find("TITLE", recursive=False)
                title_text = title.get_text(strip=True) if title else ""
                walk(child, path + [title_text] if title_text else path)
            elif name == "TABLE":
                text = _table_to_text(child)
                if text:
                    blocks.append({"section_path": list(path), "type": "table", "text": text})
            elif name == "P":
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append({"section_path": list(path), "type": "paragraph", "text": text})
            elif name == "TITLE":
                continue
            else:
                walk(child, path)

    walk(soup, [])
    return blocks


def xml_to_text(xml_str: str) -> str:
    """DART 공시 XML/HTML 마크업을 걷어내고 공백을 정리한 순수 텍스트로 변환한다."""
    blocks = parse_report_blocks(xml_str)
    return "\n\n".join(block["text"] for block in blocks)


def fetch_document_text(rcept_no: str, client: DartClient | None = None) -> str:
    """공시 원본 문서를 받아와 하나의 순수 텍스트 문자열로 반환한다."""
    zip_bytes = fetch_document_zip(rcept_no, client)
    docs = extract_documents(zip_bytes)
    parts = [xml_to_text(content) for _, content in sorted(docs.items())]
    return "\n\n".join(parts)


def fetch_report_blocks(rcept_no: str, client: DartClient | None = None) -> list[dict]:
    """공시 원본 문서를 받아와 섹션 경로가 붙은 구조화된 블록 리스트로 반환한다."""
    zip_bytes = fetch_document_zip(rcept_no, client)
    docs = extract_documents(zip_bytes)
    blocks: list[dict] = []
    for _, content in sorted(docs.items()):
        blocks.extend(parse_report_blocks(content))
    return blocks
