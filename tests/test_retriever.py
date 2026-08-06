from src.retrieval.retriever import Retriever


def test_rrf_fuse_ranks_items_high_in_both_lists_first():
    dense = [1, 2, 3]
    sparse = [3, 1, 2]
    fused_ids = [chunk_id for chunk_id, _ in Retriever._rrf_fuse([dense, sparse], top_k=3)]
    assert fused_ids[0] in (1, 3)  # 두 리스트 모두에서 상위인 후보가 1위여야 함


def test_rrf_fuse_includes_items_present_in_only_one_list():
    dense = [1, 2]
    sparse = [3]
    fused_ids = [chunk_id for chunk_id, _ in Retriever._rrf_fuse([dense, sparse], top_k=10)]
    assert set(fused_ids) == {1, 2, 3}


def test_sparse_search_scores_by_dot_product_and_respects_corp_filter():
    retriever = Retriever.__new__(Retriever)  # GPU 모델/DB 로딩 없이 순수 로직만 검증
    retriever._postings = {
        "tok_a": [(1, 1.0), (2, 0.5)],
        "tok_b": [(1, 0.2), (3, 1.0)],
    }
    retriever._chunk_corp = {1: "삼성전자", 2: "SK하이닉스", 3: "삼성전자"}

    result = retriever._sparse_search({"tok_a": 1.0, "tok_b": 1.0}, n=10, corp_name=None)
    assert result[0] == 1  # tok_a*1.0 + tok_b*0.2 = 1.2 로 최고점

    filtered = retriever._sparse_search({"tok_a": 1.0, "tok_b": 1.0}, n=10, corp_name="SK하이닉스")
    assert filtered == [2]


def test_weighted_fuse_normalizes_before_weighting():
    # dense 점수(0~1 근처)와 sparse 점수(0~수십)는 스케일이 완전히 다르다 -
    # 정규화 없이 그냥 더하면 sparse가 항상 압도해버려서, 후보군 내 min-max
    # 정규화가 실제로 적용되는지 검증한다.
    dense = [(1, 0.9), (2, 0.1)]
    sparse = [(1, 1.0), (2, 50.0)]  # 정규화 안 하면 2가 압도적으로 유리
    fused = Retriever._weighted_fuse([(dense, 0.7), (sparse, 0.3)], top_k=2)
    fused_ids = [chunk_id for chunk_id, _ in fused]
    assert fused_ids[0] == 1  # 정규화 후엔 dense에서 훨씬 앞선 1이 1위여야 함


def test_weighted_fuse_default_matches_production_weights():
    from src.retrieval.retriever import DENSE_WEIGHT, SPARSE_WEIGHT

    assert DENSE_WEIGHT == 0.7
    assert SPARSE_WEIGHT == 0.3
