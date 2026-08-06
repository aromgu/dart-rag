"""docs/generation_strategy.md의 설계를 구현한다.

Retriever.search()가 반환한 청크 리스트를 근거자료로 프롬프트에 넣고
Qwen2.5-3B-Instruct(HuggingFace, 로컬 GPU)로 답변을 생성한다.

질의 재작성(rewrite_query)과 자기검증 재시도(generate_verified)도 여기서
구현한다 - docs/experiments.md 실험 4에서 확인한 병목(검색은 맞는 청크를
찾아왔는데 생성이 문맥 안에서 엉뚱한 숫자를 고름)을 겨냥한 두 가지 대응이다.
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import settings

REWRITE_SYSTEM_PROMPT = (
    "사용자 질문을 검색에 더 적합하도록 명확하고 구체적으로 다시 쓰세요. "
    "회사명과 재무 항목(계정과목)을 분명히 밝히고, 질문의 의도는 그대로 유지하세요. "
    "다시 쓴 질문 한 문장만 출력하고 다른 설명은 붙이지 마세요."
)

# 답변에 등장한 숫자가 근거자료에 실제로 있는지 확인하는 결정론적(무료) 체크에 쓴다.
_NUMBER = re.compile(r"[\d,]{4,}")

SYSTEM_PROMPT = (
    "당신은 한국 기업의 DART 공시(사업보고서)를 근거로 답변하는 재무 데이터 어시스턴트입니다. "
    "아래 참고자료에 있는 내용만 근거로 답변하세요. 참고자료에 없으면 "
    '"자료에서 찾을 수 없습니다"라고 답하세요.\n\n'
    "참고자료의 표에는 같은 계정과목이라도 여러 값(당기/전기/전전기, 연결/별도재무제표, "
    "부문별·제품별 수치 등)이 함께 나올 수 있습니다. 답변하기 전에 반드시 확인하세요:\n"
    "1. 질문의 계정과목명과 정확히 일치하는 행을 찾으세요 (예: '매출액'과 '매출원가'/'매출총이익'을 혼동하지 마세요).\n"
    "2. 여러 기간(당기/전기)이나 재무제표 종류(연결/별도)가 함께 있으면, 특별한 언급이 없는 한 "
    "가장 최근 기간(당기)의 연결재무제표 값을 답하세요.\n"
    "3. 참고자료의 숫자를 원문 그대로(콤마 포함) 인용하세요 - 단위를 임의로 환산하거나 "
    "한글 숫자(억/만 등)로 바꿔 쓰지 마세요."
)


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        breadcrumb = chunk["corp_name"]
        if chunk.get("section_path"):
            breadcrumb += " / " + " > ".join(chunk["section_path"])
        parts.append(f"[{i}] ({breadcrumb})\n{chunk['text']}")
    return "\n\n".join(parts)


class Generator:
    def __init__(self, model_name: str | None = None, load_in_4bit: bool = False):
        model_name = model_name or settings.generation_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quant_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
            if load_in_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=None if load_in_4bit else torch.float16,
            device_map="auto",
        )

    def _generate_from_messages(self, messages: list[dict], max_new_tokens: int) -> str:
        # transformers 5.x부터 apply_chat_template(return_tensors=...)가 텐서 대신
        # BatchEncoding(dict)을 반환한다 - return_dict=True로 명시하고 **inputs로 풀어준다.
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        # 재무 수치 답변은 매번 같은 질문에 같은 답이 나와야 하는 도메인이라
        # 창의성(temperature)보다 결정론적 출력(greedy)을 우선한다.
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate(
        self,
        question: str,
        chunks: list[dict],
        max_new_tokens: int = 512,
        history: list[dict] | None = None,
    ) -> str:
        """검색된 청크(Retriever.search() 반환 형식)를 근거로 질문에 답변한다.

        history는 [{"question": ..., "answer": ...}, ...] 형태의 이전 대화
        턴이다(멀티턴, 오래된 턴부터 순서대로) - 이전 턴은 실제 주고받은 그대로
        메시지에 넣고, 근거자료는 현재 턴에만 붙인다(과거 턴에 다시 넣지 않음).
        """
        context = _build_context(chunks)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history or []:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": f"참고자료:\n{context}\n\n질문: {question}"})
        return self._generate_from_messages(messages, max_new_tokens)

    def rewrite_query(self, question: str, max_new_tokens: int = 100) -> str:
        """질문을 검색에 더 유리하도록 LLM으로 다듬는다(query rewriting).

        검색 전에 한 번 호출하는 별도 LLM 콜이라 질문당 비용이 하나 더 붙는다.
        """
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self._generate_from_messages(messages, max_new_tokens).strip()

    def generate_verified(
        self, question: str, chunks: list[dict], max_new_tokens: int = 512, max_retries: int = 1
    ) -> str:
        """답변에 나온 숫자가 근거자료에 실제로 있는지 결정론적으로(무료) 검증하고,
        실패하면 재시도한다. LLM 호출은 검증에 실패했을 때만 추가로 든다.
        """
        context = _build_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"참고자료:\n{context}\n\n질문: {question}"},
        ]
        answer = self._generate_from_messages(messages, max_new_tokens)

        for _ in range(max_retries):
            numbers = _NUMBER.findall(answer)
            grounded = not numbers or any(n in context for n in numbers)
            if grounded:
                break
            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "방금 답변에 나온 숫자가 참고자료에서 정확히 확인되지 않았습니다. "
                        '참고자료를 다시 꼼꼼히 확인해서, 정말 없으면 "자료에서 찾을 수 없습니다"라고 '
                        "답하고, 있으면 참고자료의 숫자를 그대로 인용해 다시 답하세요."
                    ),
                }
            )
            answer = self._generate_from_messages(messages, max_new_tokens)

        return answer
