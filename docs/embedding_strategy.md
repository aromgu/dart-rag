# 임베딩 전략

관련 코드: [src/embedding/embedder.py](../src/embedding/embedder.py) · 청크 설계: [docs/chunking_strategy.md](chunking_strategy.md)

## 1. 모델: BAAI/bge-m3 (dense, 1024차원)

`config.py`/`.env.example`에 이미 지정돼 있던 모델을 그대로 쓴다. 이 프로젝트 데이터 특성에 맞는 이유:

- **다국어 지원, 한국어 포함** — 사업보고서는 거의 전부 한국어 서술 + 숫자/영문 혼재(예: "Wholesale", "SOCAMM" 같은 영문 용어)라 한국어 전용 모델보다 다국어 모델이 안전
- **최대 8,192 토큰 컨텍스트** — 우리 청크는 문단 900자/표 1,500자 상한이라(대략 350~600 토큰) 여유 있게 들어감. 토큰 초과를 걱정할 필요가 거의 없음
- **dense/sparse/multi-vector를 다 지원하지만 이번 단계는 dense만 사용** — pgvector 컬럼 하나로 바로 저장/검색되고 구현이 단순함. sparse(키워드성 매칭)나 multi-vector(ColBERT류, 정밀도↑ 비용↑)는 나중에 검색 품질이 부족하면 추가 검토

## 2. 임베딩 인풋에 breadcrumb를 붙인다

청크 텍스트 자체는 문맥이 부족하다 — EDA 기준 문단 청크 중앙값이 100자 안팎이고, 표 청크는 `"매출액 | 1,748,877 | 1,699,923"`처럼 그 자체로는 **어느 회사의 어느 항목인지 알 수 없다.** 100개 기업이 다 비슷한 구조의 사업보고서를 내다보니, 문맥 없이 임베딩하면 "삼성전자 매출액"을 물어봤을 때 다른 회사의 비슷한 표 행과 구분이 잘 안 될 위험이 크다.

그래서 **임베딩할 때만** 아래처럼 breadcrumb를 앞에 붙인다 (LLM에 보여줄 `text` 필드 자체는 건드리지 않고, 임베딩 인풋에서만 조합):

```
[삼성전자] III. 재무에 관한 사항 > 2. 연결재무제표
매출액 | 1,748,877 | 1,699,923
```

`corp_name`은 대괄호로, `section_path`는 `>`로 이어붙인다. DB에는 원본 `text`와 이 벡터를 같이 저장하고, LLM 컨텍스트로 넘길 땐 `text`(+메타데이터로 회사/섹션은 프롬프트에서 별도로 알려줌)만 쓰면 된다.

## 3. 정규화 + 유사도

`normalize_embeddings=True`로 인코딩해서 L2 정규화된 벡터를 저장한다. pgvector의 코사인 거리 연산자(`<=>`)와 바로 맞물리고, bge-m3 자체가 코사인 유사도 기준으로 학습된 모델이라 이게 표준 조합이다.

## 4. 배치 처리

`sentence-transformers`의 `encode(texts, batch_size=32, normalize_embeddings=True)`로 한 번에 여러 청크를 인코딩한다. GPU가 있으면(이 환경엔 NVIDIA L4가 있음) 자동으로 GPU를 쓰고, 없으면 CPU로 폴백 - 모델 로드 시 `device`를 명시적으로 고르지 않고 `sentence-transformers` 기본 감지에 맡긴다(불필요한 분기 추가 안 함).

## 5. 안전장치: 길이 초과 방어

청킹 설계상 청크가 모델 최대 길이를 넘을 일은 거의 없어야 하지만(문단 900자/표 1,500자 상한), 혹시 상한을 우회하는 케이스(청킹 버그, 향후 파라미터 변경 등)에 대비해 인코딩 전에 breadcrumb 포함 텍스트를 문자 기준으로 한 번 더 잘라낸다(`max_input_chars`, 기본 4,000자 - bge-m3 8,192 토큰에 비해 넉넉히 여유 있는 보수적인 상한). 토큰 단위로 정확히 자르지 않고 문자 기준으로 넉넉하게 자르는 이유: 정확한 토큰 카운팅은 모델 토크나이저를 매번 돌려야 해서 배치 인코딩 속도에 영향을 주는데, 애초에 이 상한에 걸릴 케이스가 설계상 거의 없어야 하는 방어 코드이므로 정밀함보다 단순함을 택했다.

## 6. 캐싱/재실행

청크 하나당 임베딩은 한 번만 계산하면 된다 - `rcept_no` + `chunk_index`가 안 바뀌는 한 재실행 시 다시 계산할 필요가 없다. 이 캐싱은 `embedding/` 모듈이 아니라 `db/`가 맡는다(청크 저장 시 이미 임베딩된 행은 skip) - 임베딩 모듈 자체는 "청크 리스트를 받아서 벡터 리스트를 반환"하는 순수 함수로만 두고, 저장/중복방지는 DB 레이어 책임으로 분리한다.

## 인터페이스

```python
# src/embedding/embedder.py
class Embedder:
    def __init__(self, model_name: str = settings.embedding_model): ...
    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        """chunk_blocks()가 만든 청크 리스트를 받아 breadcrumb를 붙여 인코딩하고,
        청크당 1024차원 벡터 리스트를 청크와 같은 순서로 반환한다."""
```

`corp_name`은 `chunk_blocks()` 출력에는 없고 상위 파이프라인(문서별로 순회하는 쪽)이 알고 있는 값이라, `embed_chunks(chunks, corp_name=...)`처럼 호출부에서 주입한다.
