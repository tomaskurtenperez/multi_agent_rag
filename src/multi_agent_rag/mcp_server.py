"""Servidor MCP: expone la capa semántica (y el asistente completo) como tools.

Cualquier agente o cliente MCP (Claude Desktop, Claude Code, otros frameworks) puede
consultar las definiciones oficiales sin reimplementarlas. Si cambia una definición en
la ontología, todos los consumidores reciben la versión nueva sin tocar código.

Ejecutar: python -m multi_agent_rag.mcp_server   (transporte stdio)
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from multi_agent_rag.config import get_settings
from multi_agent_rag.semantic_layer import CapaSemantica

settings = get_settings()
capa = CapaSemantica(settings.ontologia_path)
mcp = MCPServer(
    "capa-semantica-banco",
    instructions="Definiciones oficiales de negocio del banco (capa semántica gobernada). "
    "Usá buscar_conceptos antes de escribir SQL o interpretar términos como 'cliente activo' o 'mora'.",
)


@mcp.tool()
def buscar_conceptos(texto: str) -> dict[str, Any]:
    """Resuelve los términos de negocio de un texto: definiciones oficiales, expresiones SQL,
    tablas, relaciones y advertencias de gobierno (conceptos no certificados)."""
    return capa.resolver(texto, settings.umbral_confianza)


@mcp.tool()
def obtener_concepto(concepto_id: str) -> dict[str, Any]:
    """Devuelve un concepto por id (p. ej. 'cliente_activo') con sus metadatos de gobierno."""
    c = capa.obtener(concepto_id)
    if c is None:
        return {"error": f"Concepto '{concepto_id}' inexistente.", "disponibles": sorted(capa.conceptos)}
    return c.model_dump()


@mcp.tool()
def listar_conceptos() -> list[dict[str, Any]]:
    """Lista todos los conceptos de la ontología con su estado de certificación."""
    return [
        {"id": c.id, "nombre": c.nombre, "dominio": c.dominio, "certificado": c.certificado, "confianza": c.confianza}
        for c in capa.conceptos.values()
    ]


@mcp.tool()
def esquema_tablas(tablas: list[str]) -> dict[str, Any]:
    """Describe tablas y columnas (sin columnas sensibles)."""
    validas = [t for t in tablas if t in capa.tablas]
    return capa.esquema(validas)


@mcp.tool()
def preguntar_asistente(pregunta: str) -> dict[str, Any]:
    """Ejecuta el asistente multiagente completo y devuelve la respuesta con sus fuentes."""
    from multi_agent_rag.graph import preguntar

    r = preguntar(pregunta)
    return {"respuesta": r["respuesta"], "fuentes": r.get("fuentes", []), "advertencias": r.get("advertencias", [])}


if __name__ == "__main__":
    mcp.run()
