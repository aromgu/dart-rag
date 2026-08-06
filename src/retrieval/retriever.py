"""docs/retrieval_strategy.md의 설계를 구현한다.

dense(pgvector HNSW)와 sparse(메모리 역색인)를 각각 독립적으로 검색한 뒤
RRF(Reciprocal Rank Fusion)로 융합한다. sparse는 pgvector 0.6.0이 전용
타입을 지원하지 않아 서버 시작 시 chunks.sparse_embedding 전체를 메모리에
역색인으로 올려두고 검색한다.
"""

import sys
import time
from collections import defaultdict

from sqlalchemy import select

from src.db.models import Chunk, Company, Disclosure
from src.db.session import SessionLocal
from src.embedding.embedder import Embedder
from src.retrieval.reranker import Reranker

RRF_K = 60
# docs/experiments.md 실험 3 결과: RRF는 강한 dense를 약한 sparse와 섞으며
# 손해를 봤지만, 스케일 보정한 가중합은 dense 단독과 동등하거나 더 나았다.
# 가중치는 실험 2에서 dense 단독이 가장 강했던 결과를 반영한 값.
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
# docs/experiments.md 실험 10: breadcrumb를 넣은 cross-encoder 재순위가
# 하이브리드 단독보다 크게 나았다(value_recall@5 67.5%→77.3%). 재순위는
# top_k보다 넉넉한 후보군 안에서 다시 골라야 의미가 있어서, 최종 top_k와
# 별개로 이 정도 후보를 먼저 가중합으로 추린 뒤 재순위한다.
RERANK_POOL = 20


class Retriever:
    def __init__(self, embedder: Embedder | None = None, reranker: Reranker | None = None, use_reranker: bool = True):
        self.embedder = embedder or Embedder()
        self.reranker = reranker if reranker is not None else (Reranker() if use_reranker else None)
        self._postings: dict[str, list[tuple[int, float]]] = {}
        self._chunk_corp: dict[int, str] = {}
        self._load_sparse_index()

    def _load_sparse_index(self) -> None:
        """chunks.sparse_embedding 전체를 읽어 {토큰ID: [(청크ID, 가중치), ...]} 역색인을 만든다."""
        t0 = time.time()
        print("sparse 역색인 구축 시작...", flush=True)
        with SessionLocal() as session:
            stmt = (
                select(Chunk.id, Chunk.sparse_embedding, Company.corp_name)
                .join(Disclosure, Chunk.rcept_no == Disclosure.rcept_no)
                .join(Company, Disclosure.corp_code == Company.corp_code)
            )
            n = 0
            for chunk_id, sparse, corp_name in session.execute(stmt).yield_per(5000):
                self._chunk_corp[chunk_id] = corp_name
                for token, weight in sparse.items():
                    self._postings.setdefault(token, []).append((chunk_id, weight))
                n += 1
                if n % 20000 == 0:
                    print(f"  {n}개 청크 색인 완료 ({time.time() - t0:.1f}초)", flush=True)
        print(
            f"sparse 역색인 구축 완료: 청크 {n}개, 토큰 {len(self._postings)}종, {time.time() - t0:.1f}초",
            flush=True,
        )

    def _dense_search_scored(self, dense_vec: list[float], n: int, corp_name: str | None) -> list[tuple[int, float]]:
        """(청크id, 코사인 유사도) 리스트를 유사도 내림차순으로 반환한다."""
        with SessionLocal() as session:
            dist = Chunk.embedding.cosine_distance(dense_vec)
            stmt = select(Chunk.id, dist)
            if corp_name:
                stmt = stmt.join(Disclosure, Chunk.rcept_no == Disclosure.rcept_no).join(
                    Company, Disclosure.corp_code == Company.corp_code
                ).where(Company.corp_name == corp_name)
            stmt = stmt.order_by(dist).limit(n)
            # pgvector의 cosine_distance = 1 - 코사인유사도라서 유사도로 되돌린다.
            return [(chunk_id, 1.0 - distance) for chunk_id, distance in session.execute(stmt).all()]

    def _dense_search(self, dense_vec: list[float], n: int, corp_name: str | None) -> list[int]:
        return [chunk_id for chunk_id, _ in self._dense_search_scored(dense_vec, n, corp_name)]

    def _sparse_search_scored(
        self, sparse_vec: dict[str, float], n: int, corp_name: str | None
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for token, q_weight in sparse_vec.items():
            for chunk_id, c_weight in self._postings.get(token, ()):
                if corp_name and self._chunk_corp.get(chunk_id) != corp_name:
                    continue
                scores[chunk_id] += q_weight * c_weight
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def _sparse_search(self, sparse_vec: dict[str, float], n: int, corp_name: str | None) -> list[int]:
        return [chunk_id for chunk_id, _ in self._sparse_search_scored(sparse_vec, n, corp_name)]

    @staticmethod
    def _rrf_fuse(rank_lists: list[list[int]], top_k: int, k: int = RRF_K) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for ranked_ids in rank_lists:
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                scores[chunk_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    @staticmethod
    def _weighted_fuse(
        scored_lists: list[tuple[list[tuple[int, float]], float]], top_k: int
    ) -> list[tuple[int, float]]:
        """[(점수 리스트, 가중치), ...]를 받아 각 리스트를 후보군 내에서 min-max
        정규화한 뒤 가중합으로 융합한다. dense(코사인)와 sparse(내적)는 스케일이
        달라서 정규화 없이 그냥 더하면 한쪽이 항상 압도한다 - RRF와 달리
        weighted-sum은 이 보정이 필수다.
        """
        combined: dict[int, float] = defaultdict(float)
        for scored, weight in scored_lists:
            if not scored:
                continue
            values = [s for _, s in scored]
            lo, hi = min(values), max(values)
            span = hi - lo or 1.0
            for chunk_id, score in scored:
                combined[chunk_id] += weight * (score - lo) / span
        return sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def _fetch_chunks(self, ids_with_scores: list[tuple[int, float]]) -> list[dict]:
        if not ids_with_scores:
            return []
        score_by_id = dict(ids_with_scores)
        with SessionLocal() as session:
            stmt = (
                select(Chunk, Company.corp_name, Disclosure.report_nm)
                .join(Disclosure, Chunk.rcept_no == Disclosure.rcept_no)
                .join(Company, Disclosure.corp_code == Company.corp_code)
                .where(Chunk.id.in_(score_by_id.keys()))
            )
            rows = session.execute(stmt).all()
        results = [
            {
                "corp_name": corp_name,
                "report_nm": report_nm,
                "section_path": chunk.section_path,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "score": score_by_id[chunk.id],
            }
            for chunk, corp_name, report_nm in rows
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def _fuse(self, dense_vec: list[float], sparse_vec: dict[str, float], top_k: int, candidate_n: int, corp_name: str | None) -> list[tuple[int, float]]:
        """dense/sparse 후보를 가중합(weighted-sum)으로 융합한다.

        원래 RRF를 썼는데, docs/experiments.md 실험 3에서 가중합(0.7/0.3)이
        RRF보다 일관되게 낫다는 게 실측으로 확인돼서 기본 융합 방식을 바꿨다.
        """
        dense_scored = self._dense_search_scored(dense_vec, candidate_n, corp_name)
        sparse_scored = self._sparse_search_scored(sparse_vec, candidate_n, corp_name)
        return self._weighted_fuse([(dense_scored, DENSE_WEIGHT), (sparse_scored, SPARSE_WEIGHT)], top_k)

    def search_ids(
        self, query: str, top_k: int = 5, corp_name: str | None = None, candidate_n: int = 100
    ) -> list[int]:
        """질문을 받아 dense+sparse 후보를 가중합으로 융합한 상위 top_k 청크 id만 반환한다.

        평가 코드처럼 텍스트 본문 없이 순위만 필요할 때 DB 재조회를 피하려고
        search()에서 _fetch_chunks 호출만 뺀 버전 - 재순위는 텍스트가 있어야
        가능해서(breadcrumb 조합) 여기서는 적용하지 않는다(순수 검색 성능만 보고 싶을 때 사용).
        """
        dense_vec, sparse_vec = self.embedder.embed_query(query)
        fused = self._fuse(dense_vec, sparse_vec, top_k, candidate_n, corp_name)
        return [chunk_id for chunk_id, _ in fused]

    def search(
        self, query: str, top_k: int = 5, corp_name: str | None = None, candidate_n: int = 100
    ) -> list[dict]:
        """질문을 받아 dense+sparse 후보를 가중합으로 융합하고, cross-encoder로
        재순위까지 마친 상위 top_k 청크를 반환한다(reranker=None이면 재순위 생략)."""
        dense_vec, sparse_vec = self.embedder.embed_query(query)
        pool_k = max(top_k, RERANK_POOL) if self.reranker else top_k
        fused = self._fuse(dense_vec, sparse_vec, pool_k, candidate_n, corp_name)
        chunks = self._fetch_chunks(fused)
        if self.reranker:
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)
        return chunks


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "매출액은 얼마인가요"
    corp = sys.argv[2] if len(sys.argv) > 2 else None
    retriever = Retriever()
    for r in retriever.search(query, corp_name=corp):
        print(f"[{r['score']:.4f}] {r['corp_name']} / {' > '.join(r['section_path'])}")
        print(f"  {r['text'][:150]}")
