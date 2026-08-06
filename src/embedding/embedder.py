"""docs/embedding_strategy.md의 설계를 구현한다.

chunk_blocks()가 만든 청크 리스트를 받아 breadcrumb(회사명 + 섹션 경로)를 붙여서
BAAI/bge-m3로 인코딩한다. dense(1024차원)와 sparse(lexical weight)를 같은 forward
pass에서 같이 뽑는다 - 나중에 하이브리드 검색을 비교 실험할 때 청크를 GPU로 다시
인코딩하지 않아도 되게 하기 위함(docs/study.md "임베딩과 DB의 관계" 참고).
저장/중복방지는 db 레이어 책임이라 여기서는 다루지 않는다.
"""

from FlagEmbedding import BGEM3FlagModel

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
        self.model = BGEM3FlagModel(model_name or settings.embedding_model, use_fp16=True)

    def embed_chunks(
        self, chunks: list[dict], corp_name: str, batch_size: int = 32
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        """청크 리스트를 (dense 벡터 리스트, sparse 가중치 딕셔너리 리스트)로 변환한다.

        둘 다 청크와 같은 순서를 유지한다. sparse는 {토큰ID(문자열): 가중치} 형태로,
        JSONB 컬럼에 바로 저장 가능하도록 float로 캐스팅한다(원본은 numpy.float16이라
        JSON 직렬화가 안 됨).
        """
        inputs = [_build_input(chunk, corp_name) for chunk in chunks]
        output = self.model.encode(
            inputs,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"].tolist()
        sparse = [{token: float(weight) for token, weight in lw.items()} for lw in output["lexical_weights"]]
        return dense, sparse

    def embed_query(self, query: str) -> tuple[list[float], dict[str, float]]:
        """검색 질문 하나를 (dense 벡터, sparse 가중치 딕셔너리)로 변환한다.

        청크와 달리 회사명/섹션 breadcrumb를 붙이지 않는다 - 질문은 이미 그
        자체로 자연어 텍스트라 breadcrumb로 보강할 문맥이 없다.
        """
        output = self.model.encode(
            [query[:MAX_INPUT_CHARS]], return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        dense = output["dense_vecs"][0].tolist()
        sparse = {token: float(weight) for token, weight in output["lexical_weights"][0].items()}
        return dense, sparse
