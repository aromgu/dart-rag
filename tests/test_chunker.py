from src.chunking.chunker import _split_table_text, chunk_blocks


def _p(section_path, text):
    return {"section_path": section_path, "type": "paragraph", "text": text}


def _t(section_path, text):
    return {"section_path": section_path, "type": "table", "text": text}


def test_chunk_blocks_merges_short_paragraphs_in_same_section():
    blocks = [_p(["I. 개요"], "문장 하나."), _p(["I. 개요"], "문장 둘.")]
    chunks = chunk_blocks(blocks, target_chars=900)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "문장 하나.\n문장 둘."
    assert chunks[0]["section_path"] == ["I. 개요"]
    assert chunks[0]["chunk_index"] == 0


def test_chunk_blocks_splits_when_target_size_exceeded():
    blocks = [_p(["I. 개요"], "가" * 50), _p(["I. 개요"], "나" * 50)]
    chunks = chunk_blocks(blocks, target_chars=60)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "가" * 50
    assert chunks[1]["text"] == "나" * 50
    assert [c["chunk_index"] for c in chunks] == [0, 1]


def test_chunk_blocks_splits_on_section_change_even_under_target_size():
    blocks = [_p(["I. 개요"], "짧은 문장"), _p(["II. 사업"], "다른 섹션 문장")]
    chunks = chunk_blocks(blocks, target_chars=900)
    assert len(chunks) == 2
    assert chunks[0]["section_path"] == ["I. 개요"]
    assert chunks[1]["section_path"] == ["II. 사업"]


def test_chunk_blocks_keeps_large_table_separate_from_surrounding_paragraphs():
    big_table = "구분 | 값\n" + "\n".join(f"항목{i} | {i}" for i in range(40))  # 200자 이상
    blocks = [
        _p(["III. 재무"], "표 앞 설명."),
        _t(["III. 재무"], big_table),
        _p(["III. 재무"], "표 뒤 설명."),
    ]
    chunks = chunk_blocks(blocks, target_chars=900, table_max_chars=10000)
    assert [c["type"] for c in chunks] == ["paragraph", "table", "paragraph"]
    assert chunks[1]["text"] == big_table


def test_chunk_blocks_merges_small_layout_table_with_paragraphs():
    # 표지의 "금융위원회 | / 날짜"류 레이아웃용 작은 표는 문단처럼 버퍼에 합쳐져야 한다
    blocks = [
        _p(["표지"], "사업보고서"),
        _t(["표지"], "금융위원회 |\n한국거래소 귀중 | 2026년 3월 11일"),
        _p(["표지"], "제출대상법인 유형: 주권상장법인"),
    ]
    chunks = chunk_blocks(blocks, target_chars=900, small_table_chars=200)
    assert len(chunks) == 1
    assert chunks[0]["type"] == "paragraph"
    assert "금융위원회" in chunks[0]["text"]


def test_split_table_text_returns_whole_table_when_under_limit():
    text = "구분 | 값\n매출 | 100"
    assert _split_table_text(text, max_chars=1000) == [text]


def test_split_table_text_repeats_header_across_parts():
    header = "구분 | 값"
    rows = [f"항목{i} | {i}" for i in range(20)]
    text = "\n".join([header] + rows)

    parts = _split_table_text(text, max_chars=40)

    assert len(parts) > 1
    for part in parts:
        assert part.startswith(header)
    # 모든 데이터 행이 어딘가엔 남아있어야 한다
    joined = "\n".join(parts)
    for row in rows:
        assert row in joined


def test_chunk_blocks_splits_large_table_into_multiple_chunks_same_section():
    header = "구분 | 값"
    rows = [f"항목{i} | {i}" for i in range(20)]
    table_text = "\n".join([header] + rows)

    chunks = chunk_blocks([_t(["III. 재무"], table_text)], table_max_chars=40, small_table_chars=0)

    assert len(chunks) > 1
    assert all(c["type"] == "table" for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
