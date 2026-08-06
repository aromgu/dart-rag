"""pgvector 익스텐션 활성화 + 테이블 생성 + 임베딩 벡터 인덱스 생성.

python -m src.db.init_db 로 실행한다. docker-compose의 postgres가 떠 있어야 한다.
"""

from sqlalchemy import text

from src.db import models  # noqa: F401  (Base.metadata에 테이블을 등록시키기 위해 import)
from src.db.base import Base
from src.db.session import engine


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        # 코사인 거리(<=>) 기준 근사 최근접 이웃 검색용 HNSW 인덱스.
        # bge-m3 임베딩은 정규화해서 저장하므로 코사인 유사도가 표준 조합이다.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


if __name__ == "__main__":
    init_db()
    print("DB 초기화 완료: 익스텐션 + 테이블(companies, disclosures, chunks) + HNSW 인덱스")
