from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from multi_agent_rag.config import get_settings
from multi_agent_rag.database import crear_base
from multi_agent_rag.graph import Dependencias
from multi_agent_rag.retriever import BM25Retriever
from multi_agent_rag.semantic_layer import CapaSemantica


class FakeChat(BaseChatModel):
    """Chat model con guion: cada llamada consume el siguiente elemento.

    Un elemento puede ser un str, un AIMessage (con tool_calls), un BaseModel (para
    with_structured_output) o una función(mensajes). `por_defecto` se usa cuando el guion se agota.
    """

    guion: list[Any] = []
    por_defecto: Any = None
    llamadas: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _siguiente(self, mensajes):
        self.llamadas += 1
        r = self.guion.pop(0) if self.guion else self.por_defecto
        if r is None:
            raise AssertionError("FakeChat: guion agotado")
        return r(mensajes) if callable(r) else r

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        r = self._siguiente(messages)
        mensaje = r if isinstance(r, AIMessage) else AIMessage(content=str(r))
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(self._siguiente)


def llamar(tool: str, _id: str = "1", **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": _id, "type": "tool_call"}])


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    db = tmp_path_factory.mktemp("datos") / "banco.db"
    crear_base(db, n_clientes=500)
    return replace(get_settings(), db_path=db, max_pasos_agente=8)


@pytest.fixture(scope="session")
def capa(settings):
    return CapaSemantica(settings.ontologia_path)


@pytest.fixture
def crear_deps(settings, capa):
    def _crear(principal: FakeChat, rapido: FakeChat) -> Dependencias:
        return Dependencias(principal, rapido, capa, BM25Retriever.desde_directorio(settings.docs_dir), settings)

    return _crear
