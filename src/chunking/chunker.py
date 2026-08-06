"""docs/chunking_strategy.md에 정리된 설계를 그대로 구현한다.

parse_report_blocks()가 만든 (section_path, type, text) 블록 리스트를 받아서:
- 문단(paragraph)은 같은 section_path 안에서 target_chars까지 묶어서 청크로 만들고
- "진짜 데이터 표"(small_table_chars 이상)는 문단과 섞지 않고, table_max_chars를 넘으면
  헤더 행을 반복하며 행 단위로 나눈다
- 표지의 "금융위원회 | / 날짜"류처럼 텍스트 배치용으로만 쓰인 작은 표(small_table_chars
  미만)는 문단과 똑같이 취급해서 버퍼에 합친다 - DART는 TABLE 태그를 순수 레이아웃
  목적으로도 쓰는데(문서당 최대 2,000개+), 전부 독립 청크로 분리하면 자잘한 표가
  전체 청크 수를 압도해버린다(99개 문서 실측: 문서당 청크 중앙값이 2,559개까지 나왔음).
  자세한 경위는 docs/chunking_strategy.md 참고.
"""


def _split_table_text(text: str, max_chars: int) -> list[str]:
    """표 텍스트(행마다 줄바꿈으로 구분됨)를 max_chars 단위로 나눈다.

    첫 행을 헤더로 보고, 나눠진 각 조각 앞에 헤더를 반복해서 붙인다 - 청크 하나만
    검색됐을 때도 숫자가 뭘 의미하는지 헤더 없이 유실되지 않도록 하기 위함이다.
    """
    if len(text) <= max_chars:
        return [text]

    rows = text.split("\n")
    header = rows[0]
    parts: list[str] = []
    current = [header]
    current_len = len(header)

    for row in rows[1:]:
        if current_len + 1 + len(row) > max_chars and len(current) > 1:
            parts.append("\n".join(current))
            current = [header]
            current_len = len(header)
        current.append(row)
        current_len += 1 + len(row)

    if len(current) > 1:
        parts.append("\n".join(current))

    return parts


def chunk_blocks(
    blocks: list[dict],
    target_chars: int = 900,
    table_max_chars: int = 1500,
    small_table_chars: int = 200,
) -> list[dict]:
    """블록 리스트를 청크 리스트로 묶는다.

    각 청크는 {"section_path", "type", "text", "chunk_index"} 형태다. chunk_index는
    같은 section_path 안에서의 순서(0부터)로, 인접 청크를 이어서 보여줄 때 쓴다.
    """
    chunks: list[dict] = []
    section_counters: dict[tuple, int] = {}

    def next_index(section_path: list[str]) -> int:
        key = tuple(section_path)
        idx = section_counters.get(key, 0)
        section_counters[key] = idx + 1
        return idx

    buffer_texts: list[str] = []
    buffer_section: list[str] | None = None
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer_texts, buffer_section, buffer_len
        if buffer_texts:
            chunks.append(
                {
                    "section_path": buffer_section,
                    "type": "paragraph",
                    "text": "\n".join(buffer_texts),
                    "chunk_index": next_index(buffer_section),
                }
            )
        buffer_texts = []
        buffer_section = None
        buffer_len = 0

    for block in blocks:
        if block["type"] == "table" and len(block["text"]) >= small_table_chars:
            flush()
            for part in _split_table_text(block["text"], table_max_chars):
                chunks.append(
                    {
                        "section_path": block["section_path"],
                        "type": "table",
                        "text": part,
                        "chunk_index": next_index(block["section_path"]),
                    }
                )
            continue

        # 문단, 그리고 small_table_chars 미만의 작은(레이아웃용으로 추정되는) 표: 섹션이
        # 바뀌었거나 목표 크기를 넘으면 지금까지 모은 걸 먼저 청크로 내보낸다
        if buffer_section is not None and (
            block["section_path"] != buffer_section or buffer_len + len(block["text"]) > target_chars
        ):
            flush()

        buffer_section = block["section_path"]
        buffer_texts.append(block["text"])
        buffer_len += len(block["text"])

    flush()
    return chunks
