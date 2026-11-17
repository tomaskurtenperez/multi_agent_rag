"""Recuperación de documentos (BM25 en memoria, sin dependencias ni API keys).

Los Markdown se parten por sección (`## ...`) para que cada fragmento tenga una fuente
citable (título + sección). La línea `Acceso: rol1, rol2` de cada documento define
qué roles pueden verlo; `todos` lo hace público.

En producción se reemplaza por Azure AI Search, pgvector, etc., con la misma interfaz
`buscar(consulta, k, rol)`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from multi_agent_rag.semantic_layer import normalizar

_STOPWORDS = set(
    "a al ante con como cual cuales de del desde donde el en entre es esta este esto la las le les lo los "
    "mas me mi no o para pero por que se si sin sobre su sus te tiene un una uno unos y ya dice hay son ser".split()
)


@dataclass
class Fragmento:
    fuente: str
    titulo: str
    seccion: str
    texto: str
    acceso: set[str] = field(default_factory=lambda: {"todos"})

    def permite(self, rol: str) -> bool:
        return "todos" in self.acceso or rol in self.acceso


def cargar_fragmentos(docs_dir: Path | str) -> list[Fragmento]:
    fragmentos: list[Fragmento] = []
    for ruta in sorted(Path(docs_dir).glob("*.md")):
        lineas = ruta.read_text(encoding="utf-8").splitlines()
        titulo = next((l[2:].strip() for l in lineas if l.startswith("# ")), ruta.stem)
        acceso_linea = next((l for l in lineas if l.lower().startswith("acceso:")), "Acceso: todos")
        acceso = {r.strip().lower() for r in acceso_linea.split(":", 1)[1].split(",") if r.strip()}

        seccion, cuerpo = "Introducción", []

        def cerrar():
            texto = "\n".join(cuerpo).strip()
            if texto:
                fragmentos.append(Fragmento(ruta.name, titulo, seccion, texto, acceso))

        for linea in lineas:
            if linea.startswith("## "):
                cerrar()
                seccion, cuerpo = linea[3:].strip(), []
            elif not linea.startswith("# ") and not linea.lower().startswith("acceso:"):
                cuerpo.append(linea)
        cerrar()
    return fragmentos


def tokenizar(texto: str) -> list[str]:
    """Tokens normalizados con stemming por prefijo (6 letras): simple y efectivo en español."""
    return [t[:6] for t in re.findall(r"[a-z0-9]+", normalizar(texto)) if t not in _STOPWORDS and len(t) > 1]


class BM25Retriever:
    def __init__(self, fragmentos: list[Fragmento], k1: float = 1.5, b: float = 0.75):
        self.fragmentos = fragmentos
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenizar(f"{f.titulo} {f.seccion} {f.texto}")) for f in fragmentos]
        self.largos = [sum(d.values()) for d in self.docs]
        self.promedio = sum(self.largos) / max(len(self.largos), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def desde_directorio(cls, docs_dir: Path | str) -> "BM25Retriever":
        return cls(cargar_fragmentos(docs_dir))

    def buscar(self, consulta: str, k: int = 3, rol: str = "analista") -> list[Fragmento]:
        """Top-k fragmentos que el rol tiene permitido ver."""
        terminos = set(tokenizar(consulta))
        puntajes = []
        for i, doc in enumerate(self.docs):
            if not self.fragmentos[i].permite(rol):
                continue
            norm = 1 - self.b + self.b * self.largos[i] / self.promedio
            s = sum(
                self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + self.k1 * norm)
                for t in terminos if t in doc
            )
            if s > 0:
                puntajes.append((s, i))
        puntajes.sort(reverse=True)
        return [self.fragmentos[i] for _, i in puntajes[:k]]
