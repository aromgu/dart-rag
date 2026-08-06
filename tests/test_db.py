import datetime
import random

from sqlalchemy import select

from src.db.models import Chunk, Company, Disclosure

EMBEDDING_DIM = 1024


def _random_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]


def test_insert_and_query_company_disclosure_chunk(db_session):
    # corp_code/rcept_no는 실제 적재된 운영 데이터와 겹치지 않는 가짜 값을 쓴다 -
    # dev DB에 실제 99개 기업 데이터가 이미 들어있어서 진짜 코드를 쓰면 PK 충돌난다.
    company = Company(corp_code="00000001", corp_name="테스트회사", stock_code="000001")
    disclosure = Disclosure(
        rcept_no="99999999999999",
        corp_code=company.corp_code,
        report_nm="사업보고서 (2024.12)",
        rcept_dt=datetime.date(2025, 3, 11),
    )
    chunk = Chunk(
        rcept_no=disclosure.rcept_no,
        section_path=["III. 재무에 관한 사항", "2. 연결재무제표"],
        chunk_type="table",
        chunk_index=0,
        text="매출액 | 1,748,877 | 1,699,923",
        embedding=_random_vector(1),
    )

    db_session.add_all([company, disclosure, chunk])
    db_session.commit()

    fetched = db_session.scalar(select(Chunk).where(Chunk.rcept_no == "99999999999999"))
    assert fetched.text == "매출액 | 1,748,877 | 1,699,923"
    assert fetched.section_path == ["III. 재무에 관한 사항", "2. 연결재무제표"]
    assert fetched.disclosure.corp_code == "00000001"
    assert fetched.disclosure.company.corp_name == "테스트회사"


def test_vector_cosine_similarity_search_orders_closest_first(db_session):
    company = Company(corp_code="00000001", corp_name="테스트회사", stock_code="000001")
    disclosure = Disclosure(
        rcept_no="99999999999999",
        corp_code=company.corp_code,
        report_nm="사업보고서 (2024.12)",
        rcept_dt=datetime.date(2025, 3, 11),
    )
    db_session.add_all([company, disclosure])

    query_vector = _random_vector(0)
    noise = random.Random(1)
    near = [x + noise.gauss(0, 0.01) for x in query_vector]  # query와 거의 동일한 벡터
    far = _random_vector(999)  # 완전히 다른 랜덤 벡터

    near_chunk = Chunk(
        rcept_no=disclosure.rcept_no, section_path=[], chunk_type="paragraph", chunk_index=0, text="near", embedding=near
    )
    far_chunk = Chunk(
        rcept_no=disclosure.rcept_no, section_path=[], chunk_type="paragraph", chunk_index=1, text="far", embedding=far
    )
    db_session.add_all([near_chunk, far_chunk])
    db_session.commit()

    # rcept_no로 이 테스트가 만든 두 행만 좁혀서 비교한다 - dev DB에는 이미
    # 실제 15만 개 청크가 들어있어서 전체 테이블 기준으로 비교하면 그 데이터가
    # near/far보다 query_vector에 더 가까울 수 있어 순서 비교가 무의미해진다.
    results = db_session.scalars(
        select(Chunk)
        .where(Chunk.rcept_no == disclosure.rcept_no)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
    ).all()
    assert [r.text for r in results] == ["near", "far"]
