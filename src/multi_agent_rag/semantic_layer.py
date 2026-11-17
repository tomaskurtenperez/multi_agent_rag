"""Capa semántica: ontología de negocio cargada como grafo de conocimiento.

Los conceptos son nodos y las relaciones aristas. Cada concepto trae su definición
oficial, cómo se calcula en SQL, en qué tablas vive y metadatos de gobierno
(steward, puntaje de confianza, certificación y versión).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar términos de negocio."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


class Concepto(BaseModel):
    id: str
    nombre: str
    definicion: str
    sinonimos: list[str] = []
    tablas: list[str] = []
    expresion_sql: str | None = None
    documento: str | None = None
    dominio: str
    steward: str
    confianza: int
    certificado: bool
    version: str


class Relacion(BaseModel):
    origen: str
    tipo: str
    destino: str
    join: str | None = None


class CapaSemantica:
    def __init__(self, ruta: Path | str):
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        self.version: str = datos.get("version", "")
        self.tablas: dict[str, Any] = datos["tablas"]
        self.conceptos: dict[str, Concepto] = {c["id"]: Concepto(**c) for c in datos["conceptos"]}
        self.grafo = nx.MultiDiGraph()
        for c in self.conceptos.values():
            self.grafo.add_node(c.id, concepto=c)
        for r in datos["relaciones"]:
            rel = Relacion(**r)
            if rel.origen not in self.conceptos or rel.destino not in self.conceptos:
                raise ValueError(f"Relación con concepto inexistente: {rel}")
            self.grafo.add_edge(rel.origen, rel.destino, relacion=rel)

    # --- búsqueda -----------------------------------------------------------------

    def buscar(self, texto: str) -> list[Concepto]:
        """Encuentra los conceptos mencionados en el texto.

        Gana la coincidencia más larga: "clientes activos" resuelve a `cliente_activo`
        y no a `cliente`.
        """
        norm = normalizar(texto)
        candidatos: list[tuple[int, int, str]] = []
        for c in self.conceptos.values():
            for termino in {c.nombre, *c.sinonimos}:
                patron = r"\b" + re.escape(normalizar(termino)) + r"\b"
                candidatos += [(m.start(), m.end(), c.id) for m in re.finditer(patron, norm)]

        candidatos.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        ocupados: list[tuple[int, int]] = []
        elegidos: list[str] = []
        for ini, fin, cid in candidatos:
            if any(ini < f and fin > i for i, f in ocupados):
                continue
            ocupados.append((ini, fin))
            if cid not in elegidos:
                elegidos.append(cid)
        return [self.conceptos[c] for c in elegidos]

    def obtener(self, concepto_id: str) -> Concepto | None:
        return self.conceptos.get(concepto_id)

    def relaciones_de(self, ids: set[str]) -> list[Relacion]:
        """Aristas que tocan al menos uno de los conceptos dados."""
        return [
            d["relacion"]
            for o, dest, d in self.grafo.edges(data=True)
            if o in ids or dest in ids
        ]

    def vecinos(self, ids: set[str]) -> list[Concepto]:
        vecinos: set[str] = set()
        for cid in ids:
            vecinos |= set(self.grafo.successors(cid)) | set(self.grafo.predecessors(cid))
        return [self.conceptos[v] for v in sorted(vecinos - ids)]

    def columnas_sensibles(self) -> set[str]:
        return {
            col
            for t in self.tablas.values()
            for col, meta in t["columnas"].items()
            if meta.get("sensible")
        }

    def esquema(self, tablas: list[str], incluir_sensibles: bool = False) -> dict[str, Any]:
        resultado = {}
        for nombre in tablas:
            t = self.tablas[nombre]
            columnas = {
                col: meta
                for col, meta in t["columnas"].items()
                if incluir_sensibles or not meta.get("sensible")
            }
            resultado[nombre] = {"descripcion": t["descripcion"], "columnas": columnas}
        return resultado

    def resolver(self, texto: str, umbral_confianza: int) -> dict[str, Any]:
        """Contexto semántico completo para un texto: conceptos, relaciones, tablas y avisos."""
        encontrados = self.buscar(texto)
        ids = {c.id for c in encontrados}
        vecinos = self.vecinos(ids)
        tablas = sorted({t for c in [*encontrados, *vecinos] for t in c.tablas})

        advertencias = [
            f"El concepto '{c.nombre}' no está certificado por el data steward "
            f"(confianza {c.confianza}/100). Usarlo con cautela."
            for c in encontrados
            if not c.certificado or c.confianza < umbral_confianza
        ]

        return {
            "version_ontologia": self.version,
            "conceptos": [c.model_dump() for c in encontrados],
            "relacionados": [
                {"id": c.id, "nombre": c.nombre, "definicion": c.definicion, "tablas": c.tablas}
                for c in vecinos
            ],
            "relaciones": [r.model_dump() for r in self.relaciones_de(ids)],
            "tablas": self.esquema(tablas),
            "advertencias": advertencias,
        }
