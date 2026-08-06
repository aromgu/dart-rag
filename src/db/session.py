from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Session:
    return SessionLocal()
