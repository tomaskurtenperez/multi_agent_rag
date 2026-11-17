"""Evaluación de regresión del asistente con LLMs reales.

Para cada pregunta de eval/preguntas.jsonl mide:
- exactitud numérica contra una consulta SQL de referencia (ground truth),
- recuperación de la fuente esperada,
- contenido obligatorio / prohibido (p. ej. no filtrar datos personales),
- advertencias de los chequeos en código y latencia.

Uso: python scripts/evaluar.py [--salida eval/resultados.json]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from multi_agent_rag.config import RAIZ
from multi_agent_rag.database import ejecutar_sql
from multi_agent_rag.semantic_layer import normalizar
from multi_agent_rag.graph import Dependencias, construir_grafo, preguntar


def _digitos(texto: str) -> set[str]:
    return {re.sub(r"\D", "", m) for m in re.findall(r"\d[\d.,]*\d|\d", texto)}


def evaluar_caso(caso: dict, grafo, deps: Dependencias) -> dict:
    inicio = time.perf_counter()
    r = preguntar(caso["pregunta"], grafo)
    latencia = round(time.perf_counter() - inicio, 2)
    respuesta = r.get("respuesta", "")
    chequeos: dict[str, bool] = {}

    if sql := caso.get("sql_referencia"):
        esperado = ejecutar_sql(sql, deps.db_path, set(deps.capa.tablas))["filas"][0][0]
        chequeos["numero_correcto"] = re.sub(r"\D", "", f"{esperado}") in _digitos(respuesta)
    if fuente := caso.get("fuente_esperada"):
        chequeos["fuente_recuperada"] = fuente in [d["fuente"] for d in r.get("documentos", [])]
    for texto in caso.get("debe_contener", []):
        chequeos[f"contiene:{texto}"] = normalizar(texto) in normalizar(respuesta)
    for texto in caso.get("no_debe_contener", []):
        chequeos[f"no_contiene:{texto}"] = normalizar(texto) not in normalizar(respuesta)

    return {
        "pregunta": caso["pregunta"],
        "ruta": r.get("ruta", []),
        "sql_ejecutado": r.get("sql_ejecutado", []),
        "respuesta": respuesta,
        "chequeos": chequeos,
        "ok": all(chequeos.values()),
        "advertencias": r.get("advertencias", []),
        "latencia_s": latencia,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=RAIZ / "eval" / "preguntas.jsonl", type=Path)
    parser.add_argument("--salida", default=RAIZ / "eval" / "resultados.json", type=Path)
    args = parser.parse_args()

    deps = Dependencias.desde_config()
    grafo = construir_grafo(deps)
    casos = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]

    resultados = []
    for caso in casos:
        res = evaluar_caso(caso, grafo, deps)
        resultados.append(res)
        marca = "OK  " if res["ok"] else "FALLA"
        print(f"[{marca}] {res['latencia_s']:>5}s  {caso['pregunta']}")
        for nombre, paso in res["chequeos"].items():
            if not paso:
                print(f"         x {nombre}")

    n = len(resultados)
    resumen = {
        "casos": n,
        "exactitud": round(sum(r["ok"] for r in resultados) / n, 3),
        "sin_advertencias": round(sum(not r["advertencias"] for r in resultados) / n, 3),
        "latencia_promedio_s": round(sum(r["latencia_s"] for r in resultados) / n, 2),
    }
    print("\n" + json.dumps(resumen, indent=2, ensure_ascii=False))
    args.salida.write_text(
        json.dumps({"resumen": resumen, "resultados": resultados}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
