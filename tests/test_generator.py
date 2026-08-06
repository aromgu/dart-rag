from src.generation.generator import _build_context, _reformat_table_text, _statement_basis_label


def test_reformat_table_text_merges_continuation_header_row():
    # "제57기" 행 다음에 빈 첫 셀로 시작하는 날짜 행이 이어지는 실제 DART 패턴
    text = (
        "구 분 | 제57기 | 제56기\n"
        " | 2025년 12월말 | 2024년 12월말\n"
        "매출액 | 333,605,938 | 300,870,903"
    )
    result = _reformat_table_text(text)
    assert "매출액: 제57기 2025년 12월말=333,605,938, 제56기 2024년 12월말=300,870,903" in result


def test_reformat_table_text_falls_back_when_row_looks_like_a_period():
    # 열 개수가 안 맞고 행 이름이 날짜처럼 보이는 애매한 행은 원문 그대로 둔다
    # (실측으로 발견한 실제 DART 표 어긋남 케이스 - docs/experiments.md 참고)
    text = (
        "구   분 | 제 64 기 | 제 63 기\n"
        "2025년 12월말 | 2024년 12월말\n"
        "[유동자산] | 9,709,042 | 11,616,873"
    )
    result = _reformat_table_text(text)
    assert "2025년 12월말 | 2024년 12월말" in result  # 원문 그대로
    assert "[유동자산]: 제 64 기=9,709,042, 제 63 기=11,616,873" in result  # 나머지는 정상 재구성


def test_reformat_table_text_leaves_non_table_text_unchanged():
    text = "표가 아닌 그냥 문단 텍스트입니다."
    assert _reformat_table_text(text) == text


def test_build_context_only_reformats_table_chunks():
    chunks = [
        {
            "corp_name": "테스트기업",
            "section_path": ["III"],
            "chunk_type": "table",
            "text": "구분 | 제10기 | 제9기\n매출액 | 100 | 90",
        },
        {
            "corp_name": "테스트기업",
            "section_path": ["III"],
            "chunk_type": "paragraph",
            "text": "구분 | 제10기 | 제9기\n매출액 | 100 | 90",  # 표처럼 생겼어도 paragraph면 안 건드림
        },
    ]
    context = _build_context(chunks)
    assert "매출액: 제10기=100, 제9기=90" in context  # [1]은 재구성됨
    assert context.count("구분 | 제10기 | 제9기\n매출액 | 100 | 90") == 1  # [2]는 원문 그대로 한 번만


def test_statement_basis_label_detects_consolidated_vs_standalone():
    assert _statement_basis_label(["III. 재무에 관한 사항", "2. 연결재무제표"]) == "연결"
    assert _statement_basis_label(["III. 재무에 관한 사항", "4. 재무제표"]) == "별도"
    assert _statement_basis_label(["II. 사업의 내용"]) is None


def test_statement_basis_label_handles_spaced_out_dart_headings():
    # DART 원문에 "연 결 재 무 제 표"처럼 글자 사이 공백이 섞여 나오는 경우가 흔하다
    assert _statement_basis_label(["(첨부)연 결 재 무 제 표"]) == "연결"
    assert _statement_basis_label(["(첨부)재 무 제 표"]) == "별도"


def test_build_context_tags_breadcrumb_with_statement_basis():
    chunks = [
        {
            "corp_name": "테스트기업",
            "section_path": ["III. 재무에 관한 사항", "2. 연결재무제표"],
            "chunk_type": "table",
            "text": "매출액 | 100",
        },
        {
            "corp_name": "테스트기업",
            "section_path": ["III. 재무에 관한 사항", "4. 재무제표"],
            "chunk_type": "table",
            "text": "매출액 | 90",
        },
    ]
    context = _build_context(chunks)
    assert "테스트기업 [연결기준] / III. 재무에 관한 사항 > 2. 연결재무제표" in context
    assert "테스트기업 [별도기준] / III. 재무에 관한 사항 > 4. 재무제표" in context
