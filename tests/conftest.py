import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.db import models  # noqa: F401
from src.db.base import Base
from src.db.session import engine


@pytest.fixture
def db_session():
    """실제 postgres에 연결해서 트랜잭션 하나로 감싼 세션을 준다.

    테스트가 끝나면 무조건 롤백해서 dev DB에 테스트 데이터가 남지 않는다.
    DB에 연결이 안 되면(로컬에 postgres가 없는 환경 등) 그 테스트만 skip한다.
    """
    try:
        connection = engine.connect()
    except OperationalError:
        pytest.skip("postgres에 연결할 수 없어 DB 테스트를 건너뜀 (docker-compose up 필요)")

    transaction = connection.begin()
    Base.metadata.create_all(connection)
    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
