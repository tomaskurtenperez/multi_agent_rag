"""Tools de los agentes. Los guardrails viven acá adentro, no en los prompts:

- `ejecutar_sql`: solo lectura, tablas permitidas, sin columnas sensibles, límite de filas.
- `buscar_documentos`: filtra por el rol del usuario antes de recuperar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool

from multi_agent_rag import database
from multi_agent_rag.config import Settings
from multi_agent_rag.retriever import BM25Retriever
from multi_agent_rag.semantic_layer import CapaSemantica


@dataclass
class Contexto:
    """Contexto de ejecución del agente de documentos (no lo ve el LLM)."""

    rol: str = "analista"


def crear_tools_datos(capa: CapaSemantica, settings: Settings) -> list:
    tablas = set(capa.tablas)
    sensibles = capa.columnas_sensibles()

    @tool
    def buscar_definicion(termino: str) -> str:
        """Busca la definición oficial de un término de negocio (p. ej. 'cliente activo', 'mora').
        Devuelve la definición, la expresión SQL oficial, las tablas, columnas y joins, y si está certificada."""
        r = capa.resolver(termino, settings.umbral_confianza)
        if not r["conceptos"]:
            return f"'{termino}' no está definido en la capa semántica. No inventes una definición."
        return json.dumps(r, ensure_ascii=False)

    @tool(response_format="content_and_artifact")
    def ejecutar_sql(sql: str) -> tuple[str, dict | None]:
        """Ejecuta UNA consulta SELECT de solo lectura (dialecto SQLite) y devuelve columnas y filas."""
        try:
            r = database.ejecutar_sql(sql, settings.db_path, tablas, sensibles, settings.max_filas)
        except database.ErrorSQL as e:
            return f"Error: {e} Corregí la consulta.", None
        return json.dumps(r, ensure_ascii=False, default=str), r

    return [buscar_definicion, ejecutar_sql]


def crear_tool_documentos(retriever: BM25Retriever, k: int):
    @tool(response_format="content_and_artifact")
    def buscar_documentos(consulta: str, runtime: ToolRuntime[Contexto]) -> tuple[str, list[dict]]:
        """Busca en las políticas y manuales internos del banco. Devuelve fragmentos con su fuente."""
        rol = runtime.context.rol if runtime.context else "analista"
        fragmentos = retriever.buscar(consulta, k=k, rol=rol)
        if not fragmentos:
            return "No se encontraron documentos relevantes (o el rol no tiene acceso).", []
        texto = "\n\n".join(
            f"(Fuente: {f.titulo}, sección \"{f.seccion}\")\n{f.texto}" for f in fragmentos
        )
        return texto, [{"fuente": f.fuente, "titulo": f.titulo, "seccion": f.seccion} for f in fragmentos]

    return buscar_documentos
