from src.embedding.embedder import MAX_INPUT_CHARS, _build_input


def test_build_input_prepends_company_and_section_breadcrumb():
    chunk = {"section_path": ["III. 재무에 관한 사항", "2. 연결재무제표"], "type": "table", "text": "매출액 | 100 | 90"}
    result = _build_input(chunk, "삼성전자")
    assert result == "[삼성전자] III. 재무에 관한 사항 > 2. 연결재무제표\n매출액 | 100 | 90"


def test_build_input_handles_empty_section_path():
    chunk = {"section_path": [], "type": "paragraph", "text": "표지 문구"}
    result = _build_input(chunk, "삼성전자")
    assert result == "[삼성전자]\n표지 문구"


def test_build_input_truncates_to_max_chars():
    chunk = {"section_path": [], "type": "paragraph", "text": "가" * 5000}
    result = _build_input(chunk, "삼성전자")
    assert len(result) == MAX_INPUT_CHARS
