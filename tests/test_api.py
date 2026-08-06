import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.api.main import AskRequest, HistoryTurn, app, ask, health


def test_health_returns_ok():
    assert health() == {"status": "ok"}


def test_ask_returns_answer_and_sources_without_loading_real_models():
    fake_retriever = MagicMock()
    fake_retriever.search.return_value = [
        {
            "corp_name": "삼성전자",
            "section_path": ["III. 재무에 관한 사항"],
            "chunk_type": "table",
            "text": "매출액 | 333,605,938",
            "score": 0.9,
        }
    ]
    fake_generator = MagicMock()
    fake_generator.generate.return_value = "333,605,938입니다."

    app.state.retriever = fake_retriever
    app.state.generator = fake_generator

    response = asyncio.run(ask(AskRequest(question="삼성전자 매출액은?", corp_name="삼성전자")))

    assert response.answer == "333,605,938입니다."
    assert len(response.sources) == 1
    assert response.sources[0].corp_name == "삼성전자"
    assert response.sources[0].text == "매출액 | 333,605,938"
    fake_retriever.search.assert_called_once_with("삼성전자 매출액은?", top_k=5, corp_name="삼성전자")
    fake_generator.generate.assert_called_once_with(
        "삼성전자 매출액은?", fake_retriever.search.return_value, history=[]
    )


def test_ask_carries_forward_corp_name_and_history_when_followup_omits_it():
    fake_retriever = MagicMock()
    fake_retriever.search.return_value = []
    fake_generator = MagicMock()
    fake_generator.generate.return_value = "43,601,051입니다."

    app.state.retriever = fake_retriever
    app.state.generator = fake_generator

    req = AskRequest(
        question="영업이익은 얼마인가요?",
        history=[
            HistoryTurn(
                question="삼성전자의 매출액은 얼마인가요?",
                answer="333,605,938입니다.",
                corp_name="삼성전자",
            )
        ],
    )
    response = asyncio.run(ask(req))

    assert response.answer == "43,601,051입니다."
    # corp_name을 명시 안 했으니 history의 마지막 corp_name("삼성전자")을 이어받아야 함
    fake_retriever.search.assert_called_once_with(
        "삼성전자의 매출액은 얼마인가요? 영업이익은 얼마인가요?", top_k=5, corp_name="삼성전자"
    )
    fake_generator.generate.assert_called_once_with(
        "영업이익은 얼마인가요?",
        [],
        history=[{"question": "삼성전자의 매출액은 얼마인가요?", "answer": "333,605,938입니다."}],
    )


def test_ask_request_rejects_blank_question():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="")


@pytest.mark.parametrize("top_k", [0, -1, 21, 100])
def test_ask_request_rejects_top_k_out_of_range(top_k):
    with pytest.raises(ValidationError):
        AskRequest(question="매출액은?", top_k=top_k)


def test_ask_request_accepts_top_k_boundaries():
    assert AskRequest(question="매출액은?", top_k=1).top_k == 1
    assert AskRequest(question="매출액은?", top_k=20).top_k == 20


def test_ask_request_strips_whitespace_from_question():
    assert AskRequest(question="  매출액은?  ").question == "매출액은?"
