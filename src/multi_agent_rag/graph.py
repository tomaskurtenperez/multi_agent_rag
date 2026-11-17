"""Router liviano + dos agentes especialistas en paralelo + respuesta final + chequeos en código.

    Pregunta → router ─┬→ datos (definiciones + SQL) ─┐
                       └→ documentos (RAG) ───────────┴→ responder → chequear → FIN

- Router: no es un agente; es una sola llamada al LLM con salida estructurada.
- Agente de datos: ReAct con `buscar_definicion` y `ejecutar_sql` (van juntos porque se usan en cadena).
- Agente de documentos: ReAct con `buscar_documentos`.
- Responder: combina ambos resultados y redacta.
- Chequear: código simple, sin LLM (números y citas).

Cuando corren los dos agentes, LangGraph espera a ambos antes de `responder`.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from multi_agent_rag.config import Settings, get_settings
from multi_agent_rag.retriever import BM25Retriever
from multi_agent_rag.semantic_layer import CapaSemantica, normalizar
from multi_agent_rag.tools import Contexto, crear_tool_documentos, crear_tools_datos

# --- 1. Estado compartido ----------------------------------------------------------------


class Estado(TypedDict, total=False):
    pregunta: str
    rol: str
    ruta: list[str]
    resultado_datos: str
    sql_ejecutado: list[str]
    resultado_docs: str
    fuentes: list[dict]
    # Salidas crudas de las tools (la "evidencia"); los dos agentes escriben en paralelo.
    evidencia: Annotated[list[str], operator.add]
    respuesta: str
    advertencias: list[str]


# --- Prompts (cortos y específicos) ------------------------------------------------------

PROMPT_ROUTER = """Decidí qué hace falta para responder la pregunta de un analista de un banco:
- "datos": cifras de la base (clientes, préstamos, mora, saldos, segmentos).
- "documentos": políticas y manuales internos (cobranza, productos de crédito, protección de datos).
Elegí uno, ambos, o ninguno si la pregunta no tiene relación con el banco."""

PROMPT_DATOS = """Sos un analista SQL de un banco (SQLite).
1. Antes de escribir SQL, usá buscar_definicion para cada término de negocio de la pregunta.
2. Usá las expresiones SQL oficiales tal cual y los joins indicados. No inventes definiciones.
3. Ejecutá una consulta agregada con ejecutar_sql: COUNT(DISTINCT clientes.id) para contar clientes en un join y ROUND(x, 2) para montos. Si da error, corregila.
4. Devolvé el resultado, el SQL usado y la definición aplicada (versión y si está certificada).
Ignorá las partes de la pregunta que no se responden con datos."""

PROMPT_DOCS = """Buscá en los documentos internos con buscar_documentos y respondé solo con lo encontrado.
Citá cada dato como (Fuente: <título>, sección <sección>). Si no encontrás nada, decilo.
Ignorá las partes de la pregunta que no se responden con documentos."""

PROMPT_RESPUESTA = """Sos el asistente de datos de un banco. Respondé en español, breve, usando SOLO la información recibida.
- Copiá los números tal cual.
- Aclará la definición oficial usada para los términos de negocio.
- Mantené las citas en formato (Fuente: <título>, sección <sección>).
- Si falta información, decí "No tengo suficiente información" sobre esa parte."""


class Ruta(BaseModel):
    destinos: list[Literal["datos", "documentos"]] = Field(
        description="Agentes necesarios: ninguno, uno o ambos."
    )


# --- Dependencias (inyectables para tests) -----------------------------------------------


@dataclass
class Dependencias:
    llm_principal: Any  # agente de datos y respuesta final
    llm_rapido: Any  # router y agente de documentos (más barato)
    capa: CapaSemantica
    retriever: BM25Retriever
    settings: Settings

    @classmethod
    def desde_config(cls, settings: Settings | None = None) -> "Dependencias":
        from multi_agent_rag.database import crear_base
        from multi_agent_rag.llm import crear_llm

        s = settings or get_settings()
        if not s.db_path.exists():
            crear_base(s.db_path)
        principal = crear_llm(s.llm_model)
        rapido = principal if s.llm_model_rapido == s.llm_model else crear_llm(s.llm_model_rapido)
        return cls(principal, rapido, CapaSemantica(s.ontologia_path), BM25Retriever.desde_directorio(s.docs_dir), s)


# --- Chequeos en código ------------------------------------------------------------------


def _numeros(texto: str) -> set[str]:
    return {re.sub(r"\D", "", m) for m in re.findall(r"\d[\d.,]*\d|\d", texto)}


def numeros_no_respaldados(respuesta: str, evidencia: str, pregunta: str = "") -> list[str]:
    """Números de 2+ dígitos de la respuesta que no aparecen en la salida de ninguna tool."""
    disponibles = _numeros(evidencia) | _numeros(pregunta)
    return sorted(n for n in _numeros(respuesta) if len(n) >= 2 and n not in disponibles)


def _texto(mensaje: Any) -> str:
    contenido = getattr(mensaje, "content", mensaje)
    if isinstance(contenido, list):  # bloques de contenido (p. ej. Anthropic)
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido).strip()
    return str(contenido).strip()


# --- Grafo -------------------------------------------------------------------------------


def construir_grafo(deps: Dependencias | None = None):
    deps = deps or Dependencias.desde_config()
    s = deps.settings
    limite = {"recursion_limit": s.max_pasos_agente}

    agente_datos = create_agent(
        deps.llm_principal,
        tools=crear_tools_datos(deps.capa, s),
        system_prompt=PROMPT_DATOS,
        name="agente_datos",
    )
    agente_docs = create_agent(
        deps.llm_rapido,
        tools=[crear_tool_documentos(deps.retriever, s.top_k_documentos)],
        system_prompt=PROMPT_DOCS,
        context_schema=Contexto,
        name="agente_documentos",
    )

    # 2. Router: una sola llamada con salida estructurada.
    def router(estado: Estado) -> dict:
        ruta: Ruta = deps.llm_rapido.with_structured_output(Ruta).invoke(
            [("system", PROMPT_ROUTER), ("human", estado["pregunta"])]
        )
        return {"ruta": list(dict.fromkeys(ruta.destinos))}

    def despachar(estado: Estado) -> list[Send]:
        # Send manda a uno o a ambos agentes en paralelo; si no hace falta ninguno, directo a responder.
        return [Send(d, estado) for d in estado["ruta"]] or [Send("responder", estado)]

    # 3. Los dos agentes.
    def nodo_datos(estado: Estado) -> dict:
        try:
            r = agente_datos.invoke({"messages": [("user", estado["pregunta"])]}, config=limite)
        except GraphRecursionError:
            return {"resultado_datos": "No se pudo obtener el dato dentro del límite de pasos."}
        mensajes = r["messages"]
        return {
            "resultado_datos": _texto(mensajes[-1]),
            "sql_ejecutado": [
                tc["args"].get("sql", "")
                for m in mensajes for tc in (getattr(m, "tool_calls", None) or [])
                if tc["name"] == "ejecutar_sql"
            ],
            "evidencia": [m.content for m in mensajes if isinstance(m, ToolMessage)],
        }

    def nodo_docs(estado: Estado) -> dict:
        try:
            r = agente_docs.invoke(
                {"messages": [("user", estado["pregunta"])]},
                config=limite,
                context=Contexto(rol=estado.get("rol", "analista")),
            )
        except GraphRecursionError:
            return {"resultado_docs": "No se pudo completar la búsqueda dentro del límite de pasos."}
        mensajes = r["messages"]
        tools = [m for m in mensajes if isinstance(m, ToolMessage)]
        return {
            "resultado_docs": _texto(mensajes[-1]),
            "fuentes": [f for m in tools for f in (m.artifact or [])],
            "evidencia": [m.content for m in tools],
        }

    # 4. Respuesta final y chequeos.
    def responder(estado: Estado) -> dict:
        partes = [f"Pregunta: {estado['pregunta']}"]
        if estado.get("resultado_datos"):
            partes.append(f"Datos:\n{estado['resultado_datos']}")
        if estado.get("resultado_docs"):
            partes.append(f"Documentos:\n{estado['resultado_docs']}")
        if len(partes) == 1:
            partes.append("No se consultó ninguna fuente: la pregunta está fuera del alcance del asistente.")
        mensaje = deps.llm_principal.invoke([("system", PROMPT_RESPUESTA), ("human", "\n\n".join(partes))])
        return {"respuesta": _texto(mensaje)}

    def chequear(estado: Estado) -> dict:
        respuesta, avisos = estado["respuesta"], []
        sueltos = numeros_no_respaldados(respuesta, "\n".join(estado.get("evidencia", [])), estado["pregunta"])
        if sueltos:
            avisos.append(f"Números que no salen de las fuentes: {', '.join(sueltos)}.")
        fuentes = estado.get("fuentes", [])
        if fuentes and not any(normalizar(f["titulo"]) in normalizar(respuesta) for f in fuentes):
            avisos.append("No se pudo verificar la fuente.")
        if not avisos:
            return {"advertencias": []}
        return {"respuesta": f"{respuesta}\n\n(Advertencia: {' '.join(avisos)})", "advertencias": avisos}

    # 5. Armar el grafo.
    g = StateGraph(Estado)
    g.add_node("router", router)
    g.add_node("datos", nodo_datos)
    g.add_node("documentos", nodo_docs)
    g.add_node("responder", responder)
    g.add_node("chequear", chequear)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", despachar, ["datos", "documentos", "responder"])
    g.add_edge("datos", "responder")
    g.add_edge("documentos", "responder")
    g.add_edge("responder", "chequear")
    g.add_edge("chequear", END)
    return g.compile()


def preguntar(pregunta: str, grafo=None, rol: str = "analista") -> dict[str, Any]:
    grafo = grafo or construir_grafo()
    return grafo.invoke({"pregunta": pregunta, "rol": rol, "evidencia": []})
