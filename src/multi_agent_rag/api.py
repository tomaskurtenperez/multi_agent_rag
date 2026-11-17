"""API HTTP (FastAPI). Ejecutar: uvicorn multi_agent_rag.api:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from multi_agent_rag.graph import construir_grafo, preguntar

app = FastAPI(title="RAG multiagente", version="0.2.0")


class Pregunta(BaseModel):
    pregunta: str
    rol: str = "analista"


class Respuesta(BaseModel):
    respuesta: str
    ruta: list[str]
    sql_ejecutado: list[str]
    fuentes: list[dict]
    advertencias: list[str]


@lru_cache
def _grafo():
    return construir_grafo()


@app.get("/salud")
def salud() -> dict[str, str]:
    return {"estado": "ok"}


@app.post("/preguntar", response_model=Respuesta)
def endpoint_preguntar(cuerpo: Pregunta) -> Respuesta:
    r = preguntar(cuerpo.pregunta, _grafo(), cuerpo.rol)
    return Respuesta(
        respuesta=r["respuesta"],
        ruta=r.get("ruta", []),
        sql_ejecutado=r.get("sql_ejecutado", []),
        fuentes=r.get("fuentes", []),
        advertencias=r.get("advertencias", []),
    )
