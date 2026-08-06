"""검색 품질을 LLM 없이 객관적으로 측정하는 전통 IR 지표.

골드셋(질문 -> 정답 청크 id)이 있어야 계산 가능하다 - scripts/build_golden_qa.py 참고.
"""


def recall_at_k(retrieved_ids: list[int], expected_id: int, k: int) -> int:
    """정답 청크가 상위 k개 안에 있으면 1, 아니면 0."""
    return 1 if expected_id in retrieved_ids[:k] else 0


def reciprocal_rank(retrieved_ids: list[int], expected_id: int) -> float:
    """정답 청크의 순위 역수(1/rank). 못 찾으면 0."""
    try:
        rank = retrieved_ids.index(expected_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def recall_at_k_by_value(retrieved_ids: list[int], expected_value: str, k: int, id_to_text: dict[int, str]) -> int:
    """상위 k개 청크 중 정답 수치가 실제로 텍스트에 있는 청크가 하나라도 있으면 1.

    같은 수치가 여러 표(재무제표/부문별/제품별 매출 등)에 동시에 등장하는 경우가
    흔해서(실측 확인), "정답 chunk_id 하나와 정확히 일치"보다 이 방식이 실제
    검색 품질을 더 정확히 반영한다.
    """
    for chunk_id in retrieved_ids[:k]:
        if expected_value in id_to_text.get(chunk_id, ""):
            return 1
    return 0


def reciprocal_rank_by_value(retrieved_ids: list[int], expected_value: str, id_to_text: dict[int, str]) -> float:
    """정답 수치가 담긴 첫 청크의 순위 역수(1/rank). 못 찾으면 0."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if expected_value in id_to_text.get(chunk_id, ""):
            return 1.0 / rank
    return 0.0
