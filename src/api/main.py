"""docs/api_strategy.md의 설계를 구현한다.

Retriever/Generator는 서버 시작 시 한 번만 로드해서 app.state에 두고 재사용한다.
실행: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.generation.generator import Generator
from src.retrieval.retriever import Retriever


class HistoryTurn(BaseModel):
    question: str
    answer: str
    corp_name: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    corp_name: str | None = None
    history: list[HistoryTurn] = []


class Source(BaseModel):
    corp_name: str
    section_path: list[str]
    chunk_type: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("모델 로딩 중 (Retriever + Generator)...", flush=True)
    retriever = Retriever()
    generator = Generator()
    app.state.retriever = retriever
    app.state.generator = generator
    print("모델 로딩 완료, 서버 준비됨", flush=True)
    yield
    app.state.retriever = None
    app.state.generator = None


app = FastAPI(title="dart-rag API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


_HISTORY_TURNS_FOR_RETRIEVAL = 2  # 검색 질의에 이어붙일 최근 대화 턴 수


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """무상태(stateless) 멀티턴: 클라이언트가 매 요청마다 history를 같이 보낸다.

    후속 질문(예: "영업이익은?")은 회사명이 빠져있을 수 있어서, corp_name을
    명시하지 않았으면 history에서 가장 최근에 쓰인 corp_name을 이어받는다.
    검색 질의는 LLM 재작성(질의 재작성은 docs/experiments.md 실험 11에서
    역효과였음) 대신, 최근 대화 질문 텍스트를 그냥 이어붙여서 문맥을 보강한다.

    retriever.search()/generator.generate()는 GPU를 쓰는 동기(blocking) 호출이라
    run_in_threadpool로 스레드풀에 넘긴다 - 이벤트 루프를 막지 않아서, 생성이
    오래 걸리는 동안에도 /health 같은 다른 요청이 계속 응답될 수 있다.
    """
    retriever: Retriever = app.state.retriever
    generator: Generator = app.state.generator

    corp_name = req.corp_name
    if corp_name is None:
        for turn in reversed(req.history):
            if turn.corp_name:
                corp_name = turn.corp_name
                break

    recent_questions = [t.question for t in req.history[-_HISTORY_TURNS_FOR_RETRIEVAL:]]
    search_query = " ".join([*recent_questions, req.question])

    chunks = await run_in_threadpool(retriever.search, search_query, top_k=req.top_k, corp_name=corp_name)
    history = [{"question": t.question, "answer": t.answer} for t in req.history]
    answer = await run_in_threadpool(generator.generate, req.question, chunks, history=history)

    sources = [
        Source(
            corp_name=c["corp_name"],
            section_path=c["section_path"],
            chunk_type=c["chunk_type"],
            text=c["text"],
            score=c["score"],
        )
        for c in chunks
    ]
    return AskResponse(answer=answer, sources=sources)
