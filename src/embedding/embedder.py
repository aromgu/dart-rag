"""docs/embedding_strategy.md의 설계를 구현한다.

chunk_blocks()가 만든 청크 리스트를 받아 breadcrumb(회사명 + 섹션 경로)를 붙여서
BAAI/bge-m3로 인코딩한다. 저장/중복방지는 db 레이어 책임이라 여기서는 다루지 않는다.
"""

from sentence_transformers import SentenceTransformer

from src.config import settings

# 청킹 단계에서 문단 900자/표 1,500자로 이미 상한을 두고 있어서 거의 걸릴 일이
# 없어야 하는 방어용 상한이다. bge-m3 최대 8,192토큰에 비해 넉넉하게 잡았다 -
# 정확한 토큰 기준으로 자르려면 매번 토크나이저를 돌려야 해서 배치 인코딩 속도가
# 떨어지는데, 애초에 안 걸려야 하는 케이스라 문자 기준의 단순함을 택했다.
MAX_INPUT_CHARS = 4000


def _build_input(chunk: dict, corp_name: str) -> str:
    """청크에 회사명/섹션 경로 breadcrumb를 붙인 임베딩 인풋을 만든다.

    청크 텍스트 자체(예: "매출액 | 1,748,877 | 1,699,923")만으로는 어느 회사의
    어느 항목인지 알 수 없어서, 100개 기업이 비슷한 구조의 사업보고서를 낸 이
    데이터셋에서는 문맥 없이 임베딩하면 회사 간 구분이 잘 안 된다. LLM에 보여줄
    chunk["text"] 자체는 건드리지 않고 임베딩 인풋에서만 붙인다.
    """
    breadcrumb = f"[{corp_name}]"
    if chunk["section_path"]:
        breadcrumb += " " + " > ".join(chunk["section_path"])
    return f"{breadcrumb}\n{chunk['text']}"[:MAX_INPUT_CHARS]


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or settings.embedding_model)

    def embed_chunks(self, chunks: list[dict], corp_name: str, batch_size: int = 32) -> list[list[float]]:
        """청크 리스트를 청크와 같은 순서의 1024차원 벡터 리스트로 변환한다."""
        inputs = [_build_input(chunk, corp_name) for chunk in chunks]
        embeddings = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
