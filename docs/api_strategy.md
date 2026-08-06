# API 전략

관련 코드: [src/api/main.py](../src/api/main.py) · 검색: [docs/retrieval_strategy.md](retrieval_strategy.md) · 생성: [docs/generation_strategy.md](generation_strategy.md)

## 1. 모델은 요청마다 새로 안 만들고 서버 시작 시 한 번만 로드

`Retriever`(bge-m3, reranker, sparse 역색인 156K행)와 `Generator`(Qwen2.5-3B)는 로딩 자체가 수십 초 걸리는 무거운 객체다 - FastAPI의 `lifespan` 컨텍스트로 서버 프로세스 시작 시 딱 한 번 만들어서 `app.state`에 두고, 매 요청마다 재사용한다.

## 2. 엔드포인트

- `POST /ask`: `{question, top_k?, corp_name?}` → `{answer, sources}` - 검색+생성을 이어서 실행하는 메인 엔드포인트. `sources`엔 답변 근거가 된 청크(회사명/섹션/본문 일부)를 같이 반환해서 사용자가 출처를 확인할 수 있게 한다.
- `GET /health`: 서버·모델 로딩 상태 확인용.

## 3. rewrite_query / generate_verified는 안 쓴다

[docs/experiments.md](experiments.md) 실험 11에서 실측으로 확인했듯 둘 다 도움이 안 됐다(재작성은 역효과, 자기검증은 이 파이프라인의 실패 유형을 못 잡음) - API는 `Generator.generate()`를 그대로 호출한다.

## 4. 멀티턴 (무상태)

서버가 대화를 저장하지 않는다 - 클라이언트가 매 요청마다 `history`(이전 질문/답변 쌍)를 같이 보낸다. 새 세션 저장소(메모리/DB)가 필요 없어서 지금까지의 무상태 설계와 일관된다.

- **회사명 이어받기**: "영업이익은?"처럼 후속 질문엔 회사명이 빠질 수 있다. `corp_name`을 명시하지 않으면 `history`에서 가장 최근에 쓰인 `corp_name`을 그대로 이어받는다(클라이언트가 각 히스토리 턴에 그때 쓴 `corp_name`을 같이 보내줘야 함).
- **검색 질의 보강**: 후속 질문 자체만으로는 dense 임베딩이 문맥을 이해하기 부족할 수 있어서, 최근 대화 질문 텍스트(`_HISTORY_TURNS_FOR_RETRIEVAL`개)를 현재 질문 앞에 그냥 이어붙인다. LLM으로 질문을 재작성하는 방식은 쓰지 않는다 - 실험 11에서 역효과였기 때문.
- **생성**: 이전 턴은 실제 주고받은 그대로 `messages`에 넣고(Qwen2.5-Instruct가 멀티턴 챗 포맷을 기본 지원), 근거자료(`참고자료:...`)는 현재 턴에만 붙인다.

## 5. 비동기 처리

`/ask`는 `async def`이고, GPU를 쓰는 동기 호출(`retriever.search`, `generator.generate`)은 `run_in_threadpool`로 스레드풀에 넘긴다 - 이벤트 루프를 막지 않아서 생성이 오래 걸리는 동안에도 `/health` 같은 다른 요청이 계속 응답될 수 있다. GPU 자체는 한 장이라 실제 생성이 동시에 여러 개 처리되는 건 아니지만(스레드들이 결국 같은 GPU를 순차적으로 씀), 서버 프로세스가 한 요청 때문에 완전히 멈추는 건 막아준다.

## 6. 입력 검증

`top_k`는 1~20으로 제한한다 - [docs/experiments.md](experiments.md) 실험 6에서 top_k가 15~20을 넘으면 문맥 과부하로 답변 정확도가 오히려 떨어지는 게 실측으로 확인돼서, 그 범위를 넘는 값을 API 레벨에서 아예 막는다. `question`은 공백만으로는 안 되고 500자를 넘을 수 없으며, `history`는 최대 20턴까지만 받는다(무제한 이력 전달로 인한 프롬프트 폭주/자원 낭비 방지). 범위를 벗어나면 pydantic이 422로 거부한다.

## 인터페이스

```python
# src/api/main.py
POST /ask
  body: {
    "question": str,
    "top_k": int = 5,
    "corp_name": str | None = None,
    "history": [{"question": str, "answer": str, "corp_name": str | None = None}] = []
  }
  response: {"answer": str, "sources": [{"corp_name", "section_path", "chunk_type", "text", "score"}]}

GET /health
  response: {"status": "ok"}
```
