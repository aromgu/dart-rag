"""dart-rag용 간단한 채팅 UI.

RAG 로직은 전혀 안 갖고 있다 - FastAPI(/ask)를 호출하기만 하는 얇은 클라이언트.
대화 이력은 API가 무상태라서 이 세션(st.session_state)에서만 들고 있다가
매 요청마다 API_STRATEGY의 멀티턴 계약대로 함께 보낸다.

실행: streamlit run src/ui/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

API_BASE_URL = os.environ.get("DART_RAG_API_URL", "http://localhost:8000")
UNIVERSE_CSV = Path(__file__).resolve().parents[2] / "data" / "kospi100_universe.csv"


@st.cache_data
def _load_company_names() -> list[str]:
    import csv

    with UNIVERSE_CSV.open(encoding="utf-8") as f:
        return sorted(row["corp_name"] for row in csv.DictReader(f))


def _ask(question: str, corp_name: str | None, history: list[dict]) -> dict:
    payload = {
        "question": question,
        "corp_name": corp_name,
        "history": [
            {"question": h["question"], "answer": h["answer"], "corp_name": h.get("corp_name")}
            for h in history
        ],
    }
    resp = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="dart-rag", page_icon="📊")
st.title("📊 dart-rag")
st.caption("KOSPI 100대 기업 DART 사업보고서 기반 질의응답")

with st.sidebar:
    st.subheader("회사 필터")
    companies = ["(전체)"] + _load_company_names()
    selected = st.selectbox("회사를 지정하면 검색 범위가 좁혀집니다", companies)
    corp_name = None if selected == "(전체)" else selected

    if st.button("대화 초기화"):
        st.session_state.history = []
        st.rerun()

    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3).json()
        st.success(f"API 연결됨 ({API_BASE_URL})")
    except requests.exceptions.RequestException:
        st.error(f"API에 연결할 수 없습니다 ({API_BASE_URL}) - 서버가 켜져 있는지 확인하세요")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            with st.expander(f"근거자료 {len(turn['sources'])}개"):
                for s in turn["sources"]:
                    breadcrumb = f"**{s['corp_name']}** / {' > '.join(s['section_path'])}"
                    st.markdown(breadcrumb)
                    st.text(s["text"][:300])
                    st.divider()

question = st.chat_input("질문을 입력하세요 (예: 삼성전자의 매출액은 얼마인가요?)")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("검색하고 답변을 생성하는 중..."):
            try:
                result = _ask(question, corp_name, st.session_state.history)
            except requests.exceptions.RequestException as e:
                st.error(f"요청 실패: {e}")
                st.stop()
        st.write(result["answer"])
        if result["sources"]:
            with st.expander(f"근거자료 {len(result['sources'])}개"):
                for s in result["sources"]:
                    breadcrumb = f"**{s['corp_name']}** / {' > '.join(s['section_path'])}"
                    st.markdown(breadcrumb)
                    st.text(s["text"][:300])
                    st.divider()

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "corp_name": corp_name,
            "sources": result["sources"],
        }
    )
