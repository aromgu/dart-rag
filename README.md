# dart-rag

한국 DART(전자공시시스템) 사업보고서를 근거로 재무 질문에 답하는 RAG(Retrieval-Augmented Generation) 시스템. KOSPI 시가총액 상위 100개 기업의 최신 사업보고서 99건, 156,707개 청크를 대상으로 한다.

```
"삼성전자의 매출액은 얼마인가요?" → "333,605,938백만원입니다." (근거 청크 포함)
```

## 왜 만들었나

DART 공시는 PDF/XML로 흩어져 있고 표가 많아서 원하는 숫자 하나를 찾으려면 수십 페이지를 뒤져야 한다. 이 프로젝트는 사업보고서 원문을 구조 보존 상태로 파싱·청킹해 벡터 DB에 넣고, 질문 하나로 정확한 재무 수치를 답변+출처와 함께 돌려주는 파이프라인을 처음부터 끝까지 구현했다. 설계 결정마다 "그럴듯해 보이는가"가 아니라 **실제 데이터로 검증**했고, 특히 검색/생성 전략은 11개의 비교실험을 거쳐 데이터 기반으로 골랐다 — 실패한 시도(첫 BM25, query rewriting, 자기검증 루프)도 원인 분석과 함께 정직하게 기록해뒀다.

## 파이프라인

```
DART Open API
    │  수집(ingestion) — 기업당 최신 사업보고서 1건
    ▼
XML 원문 (malformed markup 정제)
    │  파싱(parsing) — SECTION/TABLE 구조 보존, 표는 행 단위로 직렬화
    ▼
구조화된 블록 (문단/표)
    │  청킹(chunking) — 섹션 경계 기준, 표는 헤더 반복 분할
    ▼
청크 156,707개
    │  임베딩(embedding) — BAAI/bge-m3 dense(1024d) + sparse 동시 추출
    ▼
PostgreSQL + pgvector (HNSW 인덱스)
    │  검색(retrieval) — dense+sparse 가중합 융합 → cross-encoder 재순위
    ▼
상위 근거 청크
    │  생성(generation) — Qwen2.5-3B-Instruct, 근거 기반 프롬프트
    ▼
답변 + 출처
    │  API — FastAPI, 무상태 멀티턴
    ▼
클라이언트
```

## 기술적으로 눈여겨볼 점

- **하이브리드 검색을 실측으로 튜닝**: dense 단독 vs sparse 단독 vs BM25 vs 여러 융합 방식을 골드셋 154문항으로 비교. RRF보다 정규화된 가중합(dense 0.7 : sparse 0.3)이 일관되게 낫다는 걸 확인해 기본값으로 채택.
- **"breadcrumb 없인 다 틀린다" 패턴을 세 번 발견**: sparse(서브워드 조각화), BM25(회사명 신호 부재), cross-encoder 재순위(같은 이유) — 전부 회사명/섹션 문맥을 텍스트에 안 붙이면 다른 회사·다른 표와 혼동한다는 동일한 실패 패턴을 각각 직접 진단하고 고쳤다.
- **재순위(cross-encoder) 도입으로 검색 recall@5 67.5%→77.3%**: 처음엔 `bge-reranker-base`로 오히려 성능이 떨어져서(중국어/영어 위주 모델이라 한국어 변별력 약함) `bge-reranker-v2-m3`+breadcrumb로 교체해 반전시켰다.
- **검색보다 생성이 병목이라는 걸 end-to-end로 증명**: 검색 recall은 67~80%대인데 최종 답변 정확도는 그보다 낮았다 — 원인을 오답 케이스 직접 검산으로 추적해서 "숫자를 지어낸 게 아니라 문맥 안의 다른(엉뚱한 연도/계정) 숫자를 고르는 것"임을 확인. 프롬프트에 계정과목/기간 체크리스트를 추가해 비용 0으로 정확도 +10%p 개선.
- **역효과 시도도 정직하게 기록**: query rewriting(암묵적 "당기" 가정을 깨서 역효과), 결정론적 자기검증 재시도(hallucination은 못 잡고 "문맥 내 오선택"은 못 걸러서 무효과) — 둘 다 실측 후 프로덕션에서 제외했다.
- 상세 실험 기록: [docs/experiments.md](docs/experiments.md) (11개 실험, 방법론+결과+원인분석)

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| 수집 | DART Open API, `requests` |
| 파싱 | `BeautifulSoup` + `lxml-xml` (규칙 기반, 모델 미사용) |
| 임베딩 | `BAAI/bge-m3` (dense 1024d + sparse, `FlagEmbedding`) |
| DB | PostgreSQL 16 + `pgvector` (HNSW, 코사인) |
| 검색 | dense+sparse 가중합 하이브리드 + `BAAI/bge-reranker-v2-m3` 재순위 |
| 생성 | `Qwen/Qwen2.5-3B-Instruct` (HuggingFace `transformers`, 로컬 GPU) |
| API | FastAPI (비동기, 무상태 멀티턴) |
| 평가 | 자체 골드셋(154문항) + 결정론적 사실 정합성 체크 + RAGAS(보조) |

## 시작하기

### 요구사항

- Python 3.12
- Docker (PostgreSQL+pgvector용) 또는 로컬 Postgres 16 + pgvector 확장
- NVIDIA GPU (임베딩/재순위/생성 전부 로컬 GPU에서 실행 - 8GB+ VRAM 권장)
- [DART Open API 키](https://opendart.fss.or.kr) (수집 단계에만 필요)

### 설치

```bash
git clone https://github.com/aromgu/dart-rag.git
cd dart-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env에 DART_API_KEY 등 채워넣기
```

### DB 준비

```bash
docker compose up -d
python -m src.db.init_db
```

### 데이터 파이프라인 (수집 → DB 적재)

이미 만들어둔 스크립트로 KOSPI 100개 기업의 최신 사업보고서를 받아 파싱·청킹·임베딩까지 한 번에 처리한다:

```bash
python scripts/build_latest_annual_reports.py   # 대상 문서 목록 생성
python scripts/download_periodic_reports.py     # 원문 zip 다운로드
python scripts/load_to_db.py                    # 파싱→청킹→임베딩→DB 적재 (99개 문서 기준 GPU로 약 45분)
```

## API 실행 및 사용법

### 서버 실행

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

첫 요청 전에 모델(bge-m3, reranker, Qwen2.5-3B)을 한 번 로드한다 - 로그에 `모델 로딩 완료, 서버 준비됨`이 뜨면 준비된 것이다(수십 초~1분 정도 걸림). 이후 요청은 이 로드된 모델을 재사용하므로 빠르다.

### 헬스체크

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 질문하기

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "삼성전자의 매출액은 얼마인가요?",
    "corp_name": "삼성전자"
  }'
```

응답:

```json
{
  "answer": "333,605,938백만원입니다.",
  "sources": [
    {"corp_name": "삼성전자", "section_path": ["III. 재무에 관한 사항", "..."], "chunk_type": "table", "text": "...", "score": 0.64}
  ]
}
```

`corp_name`을 생략하면 전체 기업을 대상으로 검색한다(질문에 회사명이 포함돼 있으면 dense 임베딩이 어느 정도 알아서 좁혀준다).

### 멀티턴(대화) 사용

서버는 무상태다 - 클라이언트가 이전 대화(`history`)를 매 요청마다 같이 보내야 한다. 후속 질문에 회사명이 없으면 `history`의 가장 최근 `corp_name`을 자동으로 이어받는다:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "영업이익은 얼마인가요?",
    "history": [
      {
        "question": "삼성전자의 매출액은 얼마인가요?",
        "answer": "333,605,938백만원입니다.",
        "corp_name": "삼성전자"
      }
    ]
  }'
```

`corp_name`을 명시하지 않았지만 위 히스토리의 "삼성전자"를 이어받아 검색·답변한다.

### API 문서

서버 실행 후 `http://localhost:8000/docs`에서 FastAPI 자동 생성 Swagger UI로 직접 테스트할 수 있다.

## 프로젝트 구조

```
src/
  ingestion/    DART API 클라이언트, 문서 다운로드
  parsing       (documents.py 내) XML → 구조화 블록
  chunking/     블록 → 청크
  embedding/    bge-m3 dense+sparse
  db/           SQLAlchemy 모델, pgvector 설정
  retrieval/    하이브리드 검색 + 재순위
  generation/   Qwen2.5-3B 답변 생성
  api/          FastAPI 서버
  evaluation/   평가 지표(Recall@k/MRR, RAGAS 연동)
scripts/        파이프라인 실행/실험 스크립트
docs/           설계 문서 + 실험 기록 (아래 참고)
tests/          pytest (GPU 없이 도는 것 위주)
```

## 더 읽을거리

| 문서 | 내용 |
|---|---|
| [docs/eda.md](docs/eda.md) | 원본 데이터 탐색 |
| [docs/parsing.md](docs/parsing.md) | XML 구조 보존 파싱 설계 |
| [docs/chunking_strategy.md](docs/chunking_strategy.md) | 청킹 전략 |
| [docs/embedding_strategy.md](docs/embedding_strategy.md) | bge-m3 dense+sparse 임베딩 설계 |
| [docs/retrieval_strategy.md](docs/retrieval_strategy.md) | 하이브리드 검색 + 재순위 설계 |
| [docs/generation_strategy.md](docs/generation_strategy.md) | 생성 프롬프트/디코딩 설계 |
| [docs/api_strategy.md](docs/api_strategy.md) | API/멀티턴/비동기 설계 |
| [docs/experiments.md](docs/experiments.md) | 11개 비교실험 전체 기록 |
| [docs/study.md](docs/study.md) | 파싱/청킹/임베딩/검색/평가 기법 공부 노트 |
