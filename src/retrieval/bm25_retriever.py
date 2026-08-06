"""dense+sparse(bge-m3) 하이브리드와 비교하기 위한 전통 BM25 베이스라인.

docs/study.md에서 확인했듯 bge-m3의 sparse는 서브워드(BPE) 토크나이저 특성상
종목코드 같은 순수 숫자 문자열엔 약하다. BM25는 단어 단위 통계라 그 반례를
GPU 없이 검증할 수 있어서 비교 실험 대상으로 택했다.
"""

import re
import time

from rank_bm25 import BM25Okapi
from sqlalchemy import text

from src.db.session import SessionLocal

# BPE 서브워드 분절과 달리 "005930"/"영업이익" 같은 단위를 그대로 보존하는
# 단순 정규식 토큰화 - 전통 BM25는 원래 이렇게 단어 단위로 쓰는 방식이다.
_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")

# 형태소 분석기 없이 조사만 떼어내는 휴리스틱. "매출액은"이 청크 원문의 "매출액"과
# 문자열이 달라 전혀 매칭이 안 되는 문제(실측으로 발견 - 조사 미처리 시 BM25
# recall이 0%까지 떨어짐)를 완화한다. 완벽한 형태소 분석은 아니지만 질문 문장에
# 흔한 조사 위주로 커버한다.
_PARTICLES = sorted(
    ["으로부터", "에서부터", "이라면", "까지", "부터", "에서", "으로", "이며", "이나",
     "와", "과", "의", "은", "는", "이", "가", "을", "를", "에", "로", "도", "만", "나", "며"],
    key=len,
    reverse=True,
)


def _strip_particle(token: str) -> str:
    for p in _PARTICLES:
        if token.endswith(p) and len(token) - len(p) >= 2:
            return token[: -len(p)]
    return token


def _tokenize(s: str) -> list[str]:
    return [_strip_particle(t) for t in _TOKEN.findall(s)]


class BM25Retriever:
    def __init__(self, include_breadcrumb: bool = False):
        """include_breadcrumb=True면 임베딩과 동일하게 회사명/섹션경로를 색인
        텍스트 앞에 붙인다. 원문(chunks.text)엔 회사명이 안 들어있어서(표 셀엔
        "매출액"만 있음) 기본 BM25는 회사 구분 신호가 아예 없다는 게 실측으로
        확인됨(docs/experiments.md) - 이 옵션으로 공정 비교를 한다.
        """
        t0 = time.time()
        print("BM25 인덱스 구축 시작...", flush=True)
        with SessionLocal() as session:
            stmt = text(
                """
                SELECT c.id, c.text, c.section_path, co.corp_name FROM chunks c
                JOIN disclosures d ON c.rcept_no = d.rcept_no
                JOIN companies co ON d.corp_code = co.corp_code
                """
            )
            rows = session.execute(stmt).fetchall()

        self._ids = [r[0] for r in rows]
        self._corp = [r[3] for r in rows]
        if include_breadcrumb:
            texts = [
                f"[{corp_name}] {' > '.join(section_path)}\n{chunk_text}"
                for _, chunk_text, section_path, corp_name in rows
            ]
        else:
            texts = [r[1] for r in rows]
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        print(f"BM25 인덱스 구축 완료: 청크 {len(rows)}개, {time.time() - t0:.1f}초", flush=True)

    def search_scored(self, query: str, top_k: int = 5, corp_name: str | None = None) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        pairs = zip(self._ids, scores, self._corp)
        if corp_name:
            pairs = (p for p in pairs if p[2] == corp_name)
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)[:top_k]
        return [(chunk_id, score) for chunk_id, score, _ in ranked]

    def search(self, query: str, top_k: int = 5, corp_name: str | None = None) -> list[int]:
        return [chunk_id for chunk_id, _ in self.search_scored(query, top_k, corp_name)]
