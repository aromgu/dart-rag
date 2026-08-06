"""RAG 파이프라인 스키마.

Company(회사) 1 --- N Disclosure(공시) 1 --- N Chunk(청크, 임베딩 포함)

DART corp_code/stock_code/rcept_no는 전부 자릿수 고정 숫자 문자열이라 (앞자리 0이
의미 있음) String으로 저장한다 - Integer로 저장하면 "00126380"이 126380이 돼버린다.
"""

import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base

# bge-m3(dense) 임베딩 차원. docs/embedding_strategy.md 참고.
EMBEDDING_DIM = 1024


class Company(Base):
    __tablename__ = "companies"

    corp_code: Mapped[str] = mapped_column(String(8), primary_key=True)
    corp_name: Mapped[str] = mapped_column(String, nullable=False)
    stock_code: Mapped[str] = mapped_column(String(6), nullable=False)

    disclosures: Mapped[list["Disclosure"]] = relationship(back_populates="company")


class Disclosure(Base):
    __tablename__ = "disclosures"

    rcept_no: Mapped[str] = mapped_column(String(14), primary_key=True)
    corp_code: Mapped[str] = mapped_column(ForeignKey("companies.corp_code"), nullable=False)
    report_nm: Mapped[str] = mapped_column(String, nullable=False)
    rcept_dt: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    company: Mapped["Company"] = relationship(back_populates="disclosures")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="disclosure")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # 같은 청크가 재실행 등으로 중복 삽입되는 걸 막는다 - chunk_index는
        # chunk_blocks()에서 section_path 안에서만 순서를 매기므로, 공시+섹션까지
        # 같이 봐야 진짜 유일해진다.
        UniqueConstraint("rcept_no", "section_path", "chunk_index", name="uq_chunk_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rcept_no: Mapped[str] = mapped_column(ForeignKey("disclosures.rcept_no"), nullable=False)
    section_path: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    chunk_type: Mapped[str] = mapped_column(String(16), nullable=False)  # "paragraph" | "table"
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # bge-m3 sparse(lexical weight) 출력. {토큰ID(문자열): 가중치} 형태 - BM25가 아니라
    # 모델이 학습으로 예측하는 SPLADE류 키워드 가중치. dense와 같은 forward pass에서
    # 같이 나오므로 지금 같이 저장해두면 나중에 하이브리드 검색 비교 실험 때 15만 개
    # 청크를 GPU로 다시 인코딩할 필요가 없다. pgvector 0.6.0은 sparsevec을 지원하지
    # 않아 JSONB로 저장하고, 하이브리드 스코어링 로직은 검색(retrieval) 단계에서 구현한다.
    sparse_embedding: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    disclosure: Mapped["Disclosure"] = relationship(back_populates="chunks")
