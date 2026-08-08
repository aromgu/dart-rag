# 총정리 (공부용)

이 문서는 dart-rag 프로젝트를 처음부터 끝까지 만들면서 한 일, 만난 문제, 고친 방법, 배운 교훈을 한 자리에 모은 학습용 정리다. 각 단계의 상세 설계는 `docs/`의 개별 문서(`eda.md`, `parsing.md` 등)에, 실험은 `docs/experiments.md`에 더 자세히 있다 - 이 문서는 그 전체를 훑어보는 지도 역할이다.

## 1. 무엇을 만들었나

한국 DART(전자공시시스템)의 KOSPI 100대 기업 최신 사업보고서 99건(청크 156,707개)을 대상으로, 질문 하나에 정확한 재무 수치 + 출처를 답하는 RAG 시스템. 파이프라인은:

```
수집 → EDA → 파싱 → 청킹 → 임베딩 → DB → 검색 → 생성 → API/UI → CI
```

이 순서를 반드시 지켜서 진행했다 - 파싱/청킹 전략을 데이터 특성도 안 보고 짜지 않기 위해서였다("일단 데이터셋 먼저 받아서 EDA부터 해야 하는 거 아니야?"가 초반 방향을 정한 질문이었다).

## 2. 단계별 요약

### 2-1. 수집 (ingestion)
- DART Open API로 KOSPI 시가총액 상위 100개 기업의 corp_code를 확보(KRX 자체 API는 로그인 필요해서 못 씀 → 정적 큐레이션 리스트 사용).
- 회사당 공시 724건 → "가장 최근 사업보고서 1건"으로 좁혀서 99건으로 축소(비용/속도 고려).
- **버그**: `DartClient.get_bytes()`가 JSON 에러만 감지하고 XML 에러(상태 014)는 못 걸러서, 정정보고서 등 3건이 깨진 zip처럼 저장됨 → zip 매직바이트(`PK`)로 검증하도록 수정.

### 2-2. EDA
- 처음엔 8~19개 샘플로 봤다가, "전체 99개로 다시 하자"는 피드백으로 재실행.
- 문단/표 길이, 청크당 문서 분포, 섹션 비중, XML 결함까지 통계로 확인(`scripts/eda_full.py`).

### 2-3. 파싱 (parsing)
- 처음엔 `get_text()`로 전부 평탄화했는데 표가 다 뭉개져서, SECTION/TITLE/P/TABLE 구조를 보존하는 `parse_report_blocks()`로 교체.
- **버그 1**: 본문의 bare `&`가 XML 파싱을 깨서 `_BARE_AMPERSAND` 정규식으로 이스케이프.
- **버그 2**: `<주요자회사>`처럼 장식용 꺾쇠가 태그로 오인됨 → 처음엔 문법 휴리스틱으로 고쳤다가 `<Wholesale 부문>` 같은 케이스에서 또 깨짐 → 실제 99개 문서에서 등장하는 진짜 태그 33개를 빈도 스캔으로 뽑아 **허용목록** 방식으로 전환.
- **버그 3(치명적 성능 버그)**: 태그 허용목록 구현에서 `xml_str[m.end():]`로 매 `<` 문자마다 전체 문자열을 슬라이싱 → O(n²)라서 문서 하나가 7시간 넘게 멈춤. "아직 안 끝났어?"로 발견 → `re.match(string, pos)` 포지셔널 매칭으로 고쳐서 해결(회귀 테스트 추가).
- **버그 4**: 셀 안에 중첩된 표가 있으면 바깥 표의 `find_all("TR")`(기본 recursive)가 중첩 행까지 중복으로 세서 한 "행"이 2만~4만 자가 됨 → 중첩 표를 먼저 추출해 `[...]`로 요약하고, `tr.find_parent("TABLE") is table` 필터로 중복 방지.

### 2-4. 청킹 (chunking)
- 첫 구현은 표 크기와 상관없이 무조건 독립 청크로 분리 → 목표(청크당 900자, 문서당 50~200개)에서 완전히 벗어남(중앙값 100자, 문서당 2559~3011개).
- 원인: 레이아웃용 작은 표까지 전부 청크를 끊어버림 → `small_table_chars=200` 기준으로 작은 표는 문단처럼 병합, 큰 표만 표 특화 분할(`table_max_chars=1500`, 헤더 반복). 결과: 중앙값 2559→1410개/문서.

### 2-5. 임베딩 (embedding)
- `BAAI/bge-m3` (다국어, dense 1024차원). breadcrumb(`[회사명] 섹션경로\n본문`)를 임베딩 인풋에만 붙여서 문맥 보강(원문 텍스트 자체는 안 건드림).
- 처음엔 `sentence-transformers`로 dense만 뽑다가, "하이브리드 비교실험 하고 싶다"는 요청으로 `FlagEmbedding`(`BGEM3FlagModel`)으로 교체 → 같은 forward pass에서 dense+sparse를 동시에 뽑아서 나중 재계산 비용을 없앰.

### 2-6. DB
- PostgreSQL 16 + pgvector, HNSW 인덱스(코사인). `Company`-`Disclosure`-`Chunk` 스키마.
- 156,707개 청크 전부 dense+sparse 포함 적재 완료(GPU 약 45분). `pg_dump`로 백업(823MB, `.gitignore` 처리).

### 2-7. 검색 (retrieval) — 자세한 건 `docs/experiments.md` 참고
- dense+sparse 가중합(0.7:0.3) 융합 + `bge-reranker-v2-m3` cross-encoder 재순위.
- 원래 RRF로 시작했다가 실측으로 가중합이 낫다는 걸 확인해 교체(아래 실험 요약 참고).

### 2-8. 생성 (generation)
- `Qwen2.5-3B-Instruct` (원래 7B 계획했으나 디스크 부족으로 축소 - 아래 참고).
- 프롬프트 체크리스트, 표 재포맷, 연결/별도 명시 태그, 2-shot 예시를 누적 적용해 e2e 정확도를 계속 끌어올림.

### 2-9. API / UI / CI
- FastAPI(`/ask`, `/health`) - 비동기(`run_in_threadpool`), 무상태 멀티턴(클라이언트가 history 동봉, corp_name 자동 이어받기).
- Streamlit 채팅 UI(API를 호출만 하는 얇은 클라이언트).
- GitHub Actions로 push/PR마다 pgvector 서비스 컨테이너 띄워서 전체 테스트 자동 실행.

## 3. 인프라/환경 문제와 디버깅 스토리 (배울 점 많은 것들)

- **디스크 상시 부족**(29GB 중 종종 500MB 이하까지 감소): `hf cache list`/`rm`으로 HF 캐시의 안 쓰는 리비전(예: bge-m3의 pytorch_model.bin 중복, 4.6GB) 정리, `pip cache purge`, `apt-get clean`을 반복 활용. 실제로 쓰는 파일이 뭔지 `strace -e trace=openat`으로 직접 추적해서 확인하는 습관이 여러 번 도움이 됐다(HF 저장소 파일 크기 합산은 여러 포맷 중복 때문에 착시를 일으킴).
- **7B → 3B 모델 축소**: 원래 `Qwen2.5-7B-Instruct` 계획이었는데 디스크 4GB로는 fp16 15GB를 못 받아서 `Qwen2.5-3B-Instruct`(6GB)로 축소. 나중에 reranking(`bge-reranker-v2-m3`, 2.3GB)까지 추가로 필요해지면서 디스크 관리가 프로젝트 내내 중요한 제약이었다.
- **transformers 5.x API 변경으로 두 번 발목**: (1) `apply_chat_template(return_tensors=...)`가 텐서 대신 BatchEncoding을 반환하게 바뀌어서 `return_dict=True`로 명시해야 했음. (2) `FlagEmbedding`의 reranker 래퍼가 최신 transformers에서 제거된 `tokenizer.prepare_for_model()`을 호출해서 깨짐 → `AutoModelForSequenceClassification` 표준 API로 직접 구현해 우회.
- **torch 2.13의 Triton JIT 컴파일이 시스템 컴파일러를 요구**: `gcc`/`python3.12-dev`가 없어서 첫 생성 호출에서 실패 → 설치로 해결(모델 로딩 시점이 아니라 실제 `generate()` 첫 호출에서만 터져서 발견이 늦었음).
- **requirements.txt 버전 drift가 CI를 깨뜨림**: 세션 내내 여러 패키지(ragas, langchain-huggingface, streamlit 등)를 설치하면서 그 전이 의존성(pydantic, sqlalchemy, requests 등)이 조용히 최신 버전으로 올라갔는데 `requirements.txt`는 갱신을 안 해서, GitHub Actions의 클린 설치에서 `langchain-community`가 아주 오래된 버전까지 탐색하다 충돌. `pip freeze` 기준으로 전체를 다시 고정하고, `pip install --dry-run`으로 로컬에서 미리 검증하는 습관으로 재발 방지.
- **GitHub PAT 권한 부족**: `.github/workflows/` 파일을 푸시하려면 토큰에 `workflow` scope가 따로 필요(`repo` scope만으론 거부됨) - 토큰 편집으로 해결.
- **커밋 공동작성자(Co-Authored-By) 실수**: 첫 푸시에 자동으로 붙였다가 사용자 요청으로 `commit --amend` + `force-push`로 제거. 이후 전 커밋에서 계속 뺐다. GitHub Contributors 그래프에는 캐시가 남아 있을 수 있음(force-push해도 즉시 안 사라짐).

## 4. 13개 비교실험 요약 (자세한 표/분석은 `docs/experiments.md`)

| # | 실험 | 핵심 결론 |
|---|---|---|
| 1 | 하이브리드 vs BM25 (1차) | BM25가 조사 미처리로 recall 0% - 버그였다 |
| 2 | 5조건 ablation (dense/sparse/BM25/조합) | dense 단독이 의외로 가장 강함(당시 RRF 기준) |
| 3 | BM25+breadcrumb, 가중합 vs RRF | 가중합이 RRF보다 일관되게 나음 → 기본값 교체 |
| 4 | 검색→생성 end-to-end 정확도 | **병목은 검색이 아니라 생성**이라는 걸 처음 확인 |
| 5 | 프롬프트 개선(체크리스트) | 비용 0으로 +10%p (36.7%→46.7%) |
| 6 | top_k 확장(3~25) | k=10~15가 정점, k=20+ 부터 context 과부하로 하락 |
| 7 | 가중치(dense:sparse) 스윕 | 지표별 트레이드오프 존재, 0.7:0.3 유지 결정 |
| 8 | 청킹 파라미터(소규모 파일럿) | 작으면 손해로 보였지만 표본이 너무 작았음 |
| 9 | 청킹 파라미터(15문서로 확대) | 600~1200자 범위에서 차이 없음 - 실험 8 신호는 노이즈였음 |
| 10 | Cross-encoder 재순위 | breadcrumb 필수 확인 후 recall@5 67.5%→77.3%, 이 세션 최고 검색 기록 |
| 11 | Query rewriting, 자기검증 재시도 | **둘 다 역효과/무효과** - 원인까지 규명 후 프로덕션에서 제외 |
| 12 | 골드셋 확장(당기순이익 추가) | 당기순이익이 매출액/영업이익보다 확실히 약함(문서 전체에 더 자주 등장해서) |
| 13 | 표 재포맷+연결/별도 태그+2-shot | 1차 시도 퇴보(53.3%→43.3%) → 원인 진단 → 태그 추가로 60.0%, **최종 최고 e2e 기록** |

## 5. 최종 설정 (2026-08-06 기준)

- 검색: dense(bge-m3)+sparse(bge-m3) 가중합 0.7:0.3 → top-20 후보 → `bge-reranker-v2-m3` 재순위(breadcrumb 포함) → top-5
- 생성: `Qwen2.5-3B-Instruct`, greedy 디코딩, 표 재포맷+연결/별도 태그+2-shot 프롬프트
- e2e 정확도(골드셋 30문항 샘플 기준): **60.0%** (세션 시작 시점 46.7% 대비 +13.3%p, 전부 비용 0인 프롬프트/컨텍스트 개선만으로)

## 6. 배운 교훈 (메타 레벨)

1. **"그럴듯해 보인다" ≠ "맞다"** - 매번 SQL로 원문과 직접 대조하거나, 오답 사례를 실제로 검산해서 원인을 확정지었다. 가설을 세우고 끝내지 않고 검증까지 하는 습관이 이번 세션에서 가장 많이 반복된 패턴이다.
2. **더 정교한 방법이 항상 이기는 게 아니다** - RRF(이론상 우아함) < 가중합(단순하지만 실측으로 나음), 표 재포맷(더 읽기 쉬움) 1차 시도가 오히려 정확도를 떨어뜨림. 실측 없이 "이게 더 낫겠지"로 결정하면 안 된다는 걸 여러 번 확인했다.
3. **같은 실패 패턴이 반복해서 나타났다** - "breadcrumb(문맥) 없이 텍스트만 주면 다른 회사/다른 표와 혼동한다"는 패턴이 sparse, BM25, cross-encoder, 생성 컨텍스트(연결/별도)에서 각각 독립적으로 재발견됐다. 한 번 배운 교훈을 다음에도 의심하고 확인하는 게 중요했다.
4. **부정적인 결과도 자산이다** - query rewriting, 자기검증 재시도, 표 재포맷 1차 시도는 전부 "실패"였지만, 원인을 규명해서 문서화한 덕분에 프로젝트의 병목(생성 단계의 문맥 내 오선택)을 더 명확히 이해하게 됐다.
5. **인프라 제약(디스크/GPU 비용)이 설계 결정에 실제로 영향을 준다** - 7B→3B 축소, reranker 모델 선택, requirements.txt 관리까지 전부 "이상적인 선택"과 "가용 자원 안에서 가능한 선택" 사이의 타협이었다.
