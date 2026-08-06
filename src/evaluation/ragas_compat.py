"""ragas 0.4.3 임포트 호환 패치.

langchain-community가 sunset 진행 중이라 vertexai 통합을 빼버렸는데,
ragas가 무조건 `from langchain_community.chat_models.vertexai import
ChatVertexAI`를 import해서 우리는 안 쓰는 VertexAI 때문에 임포트 자체가
깨진다. 더미 모듈로 막아준다 - 반드시 ragas를 import하기 전에 먼저
import해야 한다(`import src.evaluation.ragas_compat`이 부작용으로 패치함).
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - 사용 안 함, import 통과시키기 위한 더미
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
