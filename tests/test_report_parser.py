import time

from src.ingestion.documents import _sanitize_xml, parse_report_blocks, xml_to_text

SAMPLE_XML = """<DOCUMENT>
<BODY>
<SECTION-1 ACLASS="MANDATORY">
<TITLE ATOCID="1">I. 회사의 개요</TITLE>
<SECTION-2>
<TITLE ATOCID="2">1. 회사의 개요</TITLE>
<P>당사는 반도체를 생산하는 회사입니다.</P>
<P></P>
<TABLE-GROUP>
<TABLE BORDER="0">
<TBODY>
<TR><TD>구분</TD><TD>2024</TD><TD>2023</TD></TR>
<TR><TD>매출액</TD><TU AUNIT="KRW">1000</TU><TU AUNIT="KRW">900</TU></TR>
</TBODY>
</TABLE>
</TABLE-GROUP>
</SECTION-2>
</SECTION-1>
</BODY>
</DOCUMENT>
"""


def test_parse_report_blocks_tracks_nested_section_path():
    blocks = parse_report_blocks(SAMPLE_XML)
    paragraph = next(b for b in blocks if b["type"] == "paragraph")
    assert paragraph["section_path"] == ["I. 회사의 개요", "1. 회사의 개요"]
    assert paragraph["text"] == "당사는 반도체를 생산하는 회사입니다."


def test_parse_report_blocks_serializes_table_rows_with_pipes():
    blocks = parse_report_blocks(SAMPLE_XML)
    table = next(b for b in blocks if b["type"] == "table")
    assert table["text"] == "구분 | 2024 | 2023\n매출액 | 1000 | 900"
    assert table["section_path"] == ["I. 회사의 개요", "1. 회사의 개요"]


def test_parse_report_blocks_skips_empty_paragraphs():
    blocks = parse_report_blocks(SAMPLE_XML)
    assert all(b["text"] for b in blocks)


def test_xml_to_text_joins_blocks_with_blank_lines():
    text = xml_to_text(SAMPLE_XML)
    assert "당사는 반도체를 생산하는 회사입니다." in text
    assert "구분 | 2024 | 2023" in text


def test_sanitize_xml_escapes_bare_ampersand():
    xml_str = "<P>Data & Solution</P>"
    assert _sanitize_xml(xml_str) == "<P>Data &amp; Solution</P>"


def test_sanitize_xml_leaves_real_entities_alone():
    xml_str = "<P>A &amp; B, 3 &lt; 5</P>"
    assert _sanitize_xml(xml_str) == xml_str


def test_sanitize_xml_escapes_decorative_angle_brackets():
    # DART 원문에 <주요자회사>, <SK스퀘어>처럼 홑화살괄호 대신 <>를 그냥 써버린
    # 소제목이 섞여 있어서, 닫는 태그가 없는 채로 파서가 뒤 구조를 다 잃어버렸었다.
    xml_str = "<SPAN><주요자회사></SPAN><SPAN><SK스퀘어></SPAN>"
    sanitized = _sanitize_xml(xml_str)
    assert sanitized == "<SPAN>&lt;주요자회사></SPAN><SPAN>&lt;SK스퀘어></SPAN>"


def test_parse_report_blocks_handles_decorative_angle_brackets_in_paragraph():
    xml_str = "<DOCUMENT><BODY><P><SPAN><SK스퀘어></SPAN> 종속회사입니다.</P></BODY></DOCUMENT>"
    blocks = parse_report_blocks(xml_str)
    assert blocks == [{"section_path": [], "type": "paragraph", "text": "<SK스퀘어> 종속회사입니다."}]


def test_sanitize_xml_escapes_decorative_bracket_that_looks_like_tag_plus_attribute():
    # "<Wholesale 부문>"은 진짜 태그(이름 뒤 공백)처럼 보이지만 화이트리스트에 없는
    # 이름이라 가짜로 판별돼야 한다 - 실제로 SK증권 사업보고서에서 파서를 깨뜨렸다.
    xml_str = "<P><Wholesale 부문> 순영업수익을 기록하였습니다.</P>"
    sanitized = _sanitize_xml(xml_str)
    assert sanitized == "<P>&lt;Wholesale 부문> 순영업수익을 기록하였습니다.</P>"


NESTED_TABLE_XML = """<DOCUMENT><BODY>
<TABLE>
<TBODY>
<TR><TD>부문</TD><TD>
  <TABLE><TBODY>
    <TR><TD>DX</TD><TD>100</TD></TR>
    <TR><TD>DS</TD><TD>200</TD></TR>
  </TBODY></TABLE>
</TD></TR>
<TR><TD>합계</TD><TD>300</TD></TR>
</TBODY>
</TABLE>
</BODY></DOCUMENT>
"""


def test_table_to_text_flattens_nested_table_without_exploding_row_length():
    from src.ingestion.documents import _table_to_text
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(NESTED_TABLE_XML, "lxml-xml")
    table = soup.find("TABLE")
    text = _table_to_text(table)
    lines = text.split("\n")

    # 바깥 표는 여전히 2행이어야 한다 (중첩 표의 행이 바깥 행으로 새서 늘어나면 안 됨)
    assert len(lines) == 2
    assert lines[0] == "부문 | [DX | 100; DS | 200]"
    assert lines[1] == "합계 | 300"


def test_sanitize_xml_stays_fast_on_documents_with_many_tags():
    # _sanitize_xml이 "<"를 만날 때마다 xml_str[m.end():]로 부분문자열을 새로
    # 만들면, "<"가 많은(실제 사업보고서는 수만~수십만 개) 대형 문서에서 O(n^2)로
    # 터진다 - 실제로 한 문서가 수 시간 동안 안 끝나는 사고가 있었다. 합리적인
    # 시간 안에 끝나는지로 이 회귀를 잡는다.
    xml_str = "<P>본문 내용입니다.</P>" * 20000  # "<"가 4만 개
    t0 = time.time()
    _sanitize_xml(xml_str)
    assert time.time() - t0 < 2.0
