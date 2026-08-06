# 검색(Retrieval) 전략

관련 코드: [src/retrieval/retriever.py](../src/retrieval/retriever.py) · 배경 지식: [docs/study.md](study.md) "검색(Retrieval) 기법" · 임베딩: [docs/embedding_strategy.md](embedding_strategy.md)

## 1. 전체 흐름

```
질문(텍스트)
  → Embedder.embed_query()로 dense+sparse 벡터화 (bge-m3, breadcrumb 없음)
  → dense 후보 top-N: pgvector HNSW 코사인 유사도 검색
  → sparse 후보 top-N: 메모리 역색인(inverted index)으로 검색
  → 가중합(weighted-sum, dense 0.7 : sparse 0.3)으로 두 순위 리스트 융합
  → 최종 top-k 청크 반환 (원문 텍스트 + 회사명/섹션/공시 메타데이터)
```

dense와 sparse를 **각각 독립적으로** top-N까지 찾은 뒤에 합친다(dense 후보 안에서만 sparse로 재정렬하는 cascade 방식이 아님) — cascade로 하면 dense가 애초에 후보에 넣지 않은 문서는 sparse로도 영원히 못 찾아서, sparse를 쓰는 원래 목적(dense가 놓치는 키워드 매칭 보완)이 무력화되기 때문이다([docs/study.md](study.md) 참고).

## 2. sparse 검색을 메모리 역색인으로 구현하는 이유

pgvector 0.6.0은 sparse 전용 타입(`sparsevec`, 0.7.0+)을 지원하지 않아 DB 인덱스로 sparse 검색을 할 수 없다. 대신 서버(프로세스) 시작 시 `chunks.sparse_embedding` 전체(15.7만 행)를 한 번 읽어서 `{토큰ID: [(청크ID, 가중치), ...]}` 형태의 역색인을 파이썬 메모리에 만들어둔다 — GPU 불필요, 순수 딕셔너리 연산이라 156K 규모에서는 가볍다. 검색 시 질문의 sparse 가중치와 내적(dot product)으로 점수를 매긴다.

## 3. 결합: 가중합 (weighted-sum, dense 0.7 : sparse 0.3)

`score(chunk) = 0.7 * norm(dense_score) + 0.3 * norm(sparse_score)` — 후보군 안에서 각 점수를 min-max 정규화한 뒤 가중합한다(`Retriever._weighted_fuse`).

**원래는 RRF(Reciprocal Rank Fusion)를 기본값으로 뒀었다** — `sum(1/(k+rank))`, 순위만 보므로 dense(코사인)와 sparse(내적)의 스케일이 달라도 보정이 필요 없다는 장점 때문이었다. 그런데 [docs/experiments.md](experiments.md)에서 골드셋 154문항으로 직접 비교해보니, RRF는 (검색 품질이 훨씬 강한) dense를 (상대적으로 약한) sparse와 섞으면서 오히려 dense 단독보다 못한 결과를 냈다 — 약한 신호가 강한 신호를 희석시킨 것. 정규화한 가중합(dense를 더 신뢰하는 0.7:0.3)으로 바꾸니 dense 단독과 동등하거나 더 나은 성능이 나와서 기본 융합 방식을 교체했다. 가중치 0.7:0.3은 실험 결과를 보고 정한 값이라, 질문 유형이 달라지면(예: sparse가 유리한 고유명사/정확한 문구 검색이 많아지면) 재튜닝이 필요할 수 있다.

## 4. 메타데이터 필터

`corp_name`을 넘기면 dense는 SQL `WHERE`로, sparse는 역색인 순회 중 회사명이 다른 청크를 걸러내는 방식으로 동일하게 필터링한다. 질문에서 회사명을 자동으로 추출하는 건(self-query) 범위 밖 — 호출부(향후 API)가 명시적으로 넘겨준다고 가정한다.

## 5. 파라미터 기본값

- `candidate_n=100`: dense/sparse 각각 1차로 찾는 후보 수
- `top_k=5`: 융합 후 최종 반환 개수 (LLM 프롬프트에 넣을 컨텍스트 예산 고려)
- `DENSE_WEIGHT=0.7`, `SPARSE_WEIGHT=0.3`: 가중합 가중치 ([docs/experiments.md](experiments.md) 실험 근거)

## 6. 검증 결과 (2026-08-06)

실제 DB(156,707개 청크)로 4가지를 검증했다:

1. **corp_name 필터**: `corp_name="SK하이닉스"`로 검색 시 결과 전부 SK하이닉스만 나옴을 확인 — dense/sparse 양쪽 모두 필터가 실제로 적용됨.
2. **할루시네이션 방지**: 자료에 없는 질문("삼성전자 대표이사의 개인 취미")에 모델이 숫자를 지어내지 않고 "자료에서 찾을 수 없습니다"라고 정확히 답함.
3. **답변 사실 정합성**: "삼성전자 매출액 333,605,938" "SK하이닉스 영업이익 47,206,319" 둘 다 DB 원문 청크와 SQL로 직접 대조해 정확히 일치함을 확인(재무제표 손익계산서 행과 100% 일치).
4. **dense/sparse가 실제로 다른 후보를 찾는지**: 질의 "005930"(삼성전자 종목코드)으로 테스트한 결과, dense·sparse top-20 후보가 **완전히 겹치지 않음(0/20)**을 확인. 다만 sparse 상위 결과를 원문 대조해보니 "005930"을 포함한 청크가 하나도 없었다 — 원인은 bge-m3 sparse가 서브워드(BPE) 토크나이저를 쓰기 때문에 `005930`이 `00`/`59`/`30`처럼 흔한 2자리 숫자 조각으로 쪼개져서, 그 조각들이 들어간(=거의 모든 숫자 표) 청크에 광범위하게 매칭돼버림(변별력 상실). 반면 `영업이익`(`영업`/`이`/`익`), `SK하이닉스`(`SK`/`하이`/`닉`/`스`) 같은 한글 복합어는 의미 있는 조각으로 나뉘어 실제로 잘 작동한다.

**결론**: sparse는 계정과목명·회사명 같은 한글 복합어 검색엔 설계대로 기여하지만, **순수 숫자 코드(종목코드 등) 검색에는 거의 도움이 안 된다** — 이런 경우는 사용자가 종목코드를 직접 입력해서 질문하는 경우가 드물어(대부분 "삼성전자 매출액"처럼 회사명으로 질문) 실사용 영향은 제한적이라고 판단해 현재 설계를 유지한다. 종목코드로 직접 검색해야 하는 요구가 생기면, 질의에서 숫자 패턴을 추출해 SQL `WHERE stock_code = ...`로 직접 필터링하는 걸 추가하는 게 sparse보다 훨씬 안정적이다.

## 7. Cross-encoder 재순위 (반영됨)

처음엔 "지금 안 하는 것"으로 미뤄뒀는데([docs/study.md](study.md) 참고), [docs/experiments.md](experiments.md) 실험 10에서 breadcrumb를 포함한 `BAAI/bge-reranker-v2-m3` 재순위가 하이브리드 단독보다 크게 나은 걸 확인(value_recall@5 67.5%→77.3%, 이번 세션 최고 기록)해서 프로덕션에 반영했다. `Retriever.search()`는 기본으로 top-`RERANK_POOL`(20)개 후보를 가중합으로 추린 뒤 `Reranker`로 최종 top_k까지 재정렬한다(`search_ids()`는 텍스트가 없는 경량 경로라 재순위 미적용). reranker에 넘기는 텍스트도 dense/sparse와 동일하게 breadcrumb를 붙인다 - 안 붙이면 다른 회사/다른 표의 같은 단어와 혼동한다는 게 실측으로 확인됐다(처음 `bge-reranker-base`+breadcrumb 없이 시도했을 때 오히려 성능이 떨어졌던 원인).

## 8. 지금 안 하는 것

HyDE/multi-query/query decomposition, GraphRAG, 반복형(에이전틱) 검색은 전부 LLM 호출이 추가로 드는 방식이라 제외했다. Query rewriting과 자기검증 재시도는 실제로 구현하고 실측까지 했지만 효과가 없거나 역효과였다([docs/experiments.md](experiments.md) 실험 11 참고, [docs/generation_strategy.md](generation_strategy.md)에 상세) — 답변 품질이 부족하면 이후 단계에서 다른 방식으로 재검토.

## 인터페이스

```python
# src/retrieval/retriever.py
class Retriever:
    def __init__(
        self, embedder: Embedder | None = None, reranker: Reranker | None = None, use_reranker: bool = True
    ): ...
    def search(
        self, query: str, top_k: int = 5, corp_name: str | None = None, candidate_n: int = 100
    ) -> list[dict]:
        """질문을 받아 가중합으로 융합하고 cross-encoder로 재순위까지 마친
        상위 top_k 청크를 반환한다(reranker=None이면 재순위 생략).
        각 결과는 corp_name/report_nm/section_path/chunk_type/text/score를 포함한다."""
```
