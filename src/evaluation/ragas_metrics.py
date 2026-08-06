"""docs/study.md "RAG 평가(Evaluation) 기법/툴"에서 정한 대로, ragas의 판정
LLM/임베딩을 OpenAI API 대신 이미 로드된 로컬 모델(Generator의 Qwen2.5-3B,
Embedder의 bge-m3)로 재사용한다 - 모델을 새로 띄우지 않아 GPU/디스크 비용이
추가로 들지 않는다.
"""

from src.evaluation import ragas_compat  # noqa: F401  (ragas import 전에 먼저 실행돼야 함)

from langchain_core.embeddings import Embeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from transformers import pipeline as hf_pipeline

from src.embedding.embedder import Embedder
from src.generation.generator import Generator


class _EmbedderAdapter(Embeddings):
    """Embedder(bge-m3 dense)를 langchain의 Embeddings 인터페이스로 감싼다."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.embed_query(t)[0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)[0]


def build_ragas_metrics(generator: Generator, embedder: Embedder) -> dict:
    """이미 로드된 Generator/Embedder를 재사용해서 ragas 판정 지표 3개를 만든다."""
    text_gen = hf_pipeline(
        "text-generation",
        model=generator.model,
        tokenizer=generator.tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )
    chat_model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen))
    llm = LangchainLLMWrapper(chat_model)
    embeddings = LangchainEmbeddingsWrapper(_EmbedderAdapter(embedder))

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
    }


def score_ragas(metrics: dict, question: str, contexts: list[str], response: str) -> dict[str, float]:
    """질문 하나에 대해 ragas 지표 3개(faithfulness/answer_relevancy/context_precision)를 계산한다."""
    sample = SingleTurnSample(user_input=question, retrieved_contexts=contexts, response=response)
    return {name: metric.single_turn_score(sample) for name, metric in metrics.items()}
