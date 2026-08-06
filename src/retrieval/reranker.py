"""Cross-encoder 재순위(rerank).

dense+sparse 하이브리드(bi-encoder)로 찾은 후보는 질문과 문서를 각각 따로
인코딩해서 비교하지만, cross-encoder는 (질문, 문서) 쌍을 함께 넣어 관련도
점수를 직접 계산해서 더 정밀하다 - 다만 후보 하나하나마다 모델을 돌려야 해서
느려서 1차 검색이 아니라 하이브리드가 찾은 후보군(top-N)을 재정렬하는 데만 쓴다
(docs/study.md "검색 기법" 참고).

FlagEmbedding의 FlagReranker 래퍼는 최신 transformers(5.x)에서 제거된
tokenizer.prepare_for_model()을 호출해서 깨진다 - 표준 AutoModelForSequenceClassification
API로 직접 구현해서 우회한다.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """Retriever.search()가 반환한 청크 후보를 cross-encoder 점수로 재정렬한다.

        원문 텍스트만 넣으면 회사/섹션 문맥이 없어서 다른 회사·다른 표의
        같은 단어("매출액" 열 등)와 혼동한다(실측 확인 - 삼성전자 질문에
        자회사 삼성디스플레이 투자현황표를 1위로 잘못 채점함, docs/experiments.md
        참고) - 임베딩과 동일하게 breadcrumb(회사명+섹션경로)를 붙여서 넣는다.
        """
        if not chunks:
            return []

        def _breadcrumb_text(chunk: dict) -> str:
            breadcrumb = f"[{chunk['corp_name']}]"
            if chunk.get("section_path"):
                breadcrumb += " " + " > ".join(chunk["section_path"])
            return f"{breadcrumb}\n{chunk['text']}"

        pairs = [[query, _breadcrumb_text(c)] for c in chunks]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1).float().cpu().tolist()
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score
        return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
