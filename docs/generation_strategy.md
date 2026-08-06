# 생성(Generation) 전략

관련 코드: [src/generation/generator.py](../src/generation/generator.py) · 검색: [docs/retrieval_strategy.md](retrieval_strategy.md)

## 1. 모델: Qwen2.5-3B-Instruct (HuggingFace, 로컬 GPU)

원래 `Qwen2.5-7B-Instruct`로 계획했으나, 구현 시점에 디스크 여유 공간이 4GB뿐이라 fp16 기준 약 15GB인 7B 가중치를 받을 수 없었다. `pip`/`apt` 캐시 정리와 안 쓰는 HF 캐시 리비전(`hf cache rm`으로 확인 후 삭제) 정리로 9.1GB까지 확보했지만 그래도 7B에는 부족했고, 디스크 자체를 늘리는 건 클라우드 콘솔 작업이라 이 세션에서는 할 수 없어 **`Qwen2.5-3B-Instruct`(fp16 약 6GB)로 축소**했다. `config.py`/`.env`의 `generation_model` 기본값을 이걸로 바꿔뒀다 — 7B로 다시 올리고 싶으면 디스크를 늘린 뒤 이 값만 바꾸면 된다.

## 2. 양자화 여부

3B는 fp16 그대로 로드해도 약 6GB(+임베딩 모델 bge-m3 2.2GB)로 L4 23GB에 충분히 여유가 있어서, 굳이 4bit 양자화로 품질을 깎을 필요가 없다 — `Generator`는 `load_in_4bit` 옵션을 열어두되 기본값은 `False`(fp16)로 한다. 나중에 다시 7B 이상으로 올리면 이 옵션을 켜는 걸 권장한다.

## 3. 프롬프트 설계

검색된 청크를 "참고자료"로 프롬프트에 넣고, **참고자료에 없는 내용은 답하지 말라**고 명시한다 — 재무 수치처럼 틀리면 안 되는 도메인이라 모델이 아는 척(hallucination)하지 않게 강하게 제약한다. 각 참고자료 앞에 회사명/섹션 경로를 붙여서, 여러 회사 청크가 섞여도 모델이 출처를 구분할 수 있게 한다(임베딩 breadcrumb와 같은 이유).

```
[system]
당신은 한국 기업의 DART 공시(사업보고서)를 근거로 답변하는 어시스턴트입니다.
아래 참고자료에 있는 내용만 근거로 답변하세요. 참고자료에 없으면 "자료에서 찾을 수 없습니다"라고 답하세요.

[user]
참고자료:
[1] (삼성전자 / III. 재무에 관한 사항 > 2. 연결재무제표)
매출액 (주30) | 333,605,938 | ...

[2] (...)

질문: 삼성전자 최근 매출액은?
```

`tokenizer.apply_chat_template()`로 Qwen instruct 포맷에 맞춘다(모델마다 채팅 템플릿이 달라서 직접 문자열 조립 대신 이 API를 쓴다 - 프롬프트 포맷이 학습 때와 어긋나면 성능이 떨어짐).

## 4. 디코딩 파라미터

`do_sample=False`(greedy)로 결정론적 출력을 쓴다 - 재무 수치 답변은 매번 같은 질문에 같은 답이 나와야 하는 도메인이라 창의성(temperature)보다 일관성이 우선이다. `max_new_tokens=512`로 과도하게 긴 생성을 방지한다.

## 5. 검색과의 결합

`Generator` 자체는 "질문 + 청크 리스트 → 답변 문자열"만 담당하는 순수한 컴포넌트로 두고, `Retriever.search()`로 얻은 결과를 그대로 입력받는다 - 두 모듈을 분리해서 각각 독립적으로 테스트/교체 가능하게 한다.

## 6. rewrite_query / generate_verified - 구현은 했지만 기본으로 안 씀

질의 재작성과 자기검증 재시도를 구현해서 실측했는데(`docs/experiments.md` 실험 11), 둘 다 도움이 안 됐다 - 재작성은 원래 없던 시점 표현("지난해" 등)을 임의로 추가해서 오히려 정확도를 떨어뜨렸고, 자기검증은 "hallucination"(숫자를 지어냄)은 잡지만 이 파이프라인의 실제 실패 유형인 "문맥 내 다른 숫자를 잘못 고름"은 못 잡아서 효과가 없었다. 메서드는 `Generator`에 남겨뒀지만(`rewrite_query`, `generate_verified`) 기본 흐름(`generate()`)에서는 쓰지 않는다 - API 모듈도 `generate()`를 그대로 쓴다.

## 인터페이스

```python
# src/generation/generator.py
class Generator:
    def __init__(self, model_name: str = settings.generation_model, load_in_4bit: bool = False): ...
    def generate(self, question: str, chunks: list[dict], max_new_tokens: int = 512) -> str:
        """검색된 청크(Retriever.search() 반환 형식)를 근거로 질문에 답변한다."""
```
