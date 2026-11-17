"""Línea de comandos: `mar preguntar "..."`, `mar chat`, `mar init-db`."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

consola = Console()


def _mostrar(r: dict, verbose: bool) -> None:
    if verbose:
        consola.print(f"[cyan]Ruta:[/] {', '.join(r.get('ruta') or []) or 'ninguna'}")
        for sql in r.get("sql_ejecutado", []):
            consola.print(f"[cyan]SQL:[/] {sql}")
        for f in r.get("fuentes", []):
            consola.print(f"[cyan]Fuente:[/] {f['titulo']} — {f['seccion']}")
        if r.get("resultado_datos"):
            consola.print(Panel(r["resultado_datos"], title="Agente de datos", border_style="dim"))
        if r.get("resultado_docs"):
            consola.print(Panel(r["resultado_docs"], title="Agente de documentos", border_style="dim"))
    color = "yellow" if r.get("advertencias") else "green"
    consola.print(Panel(r.get("respuesta", ""), title="Respuesta", border_style=color))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mar", description="RAG multiagente: router + agente de datos + agente de documentos")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("preguntar", help="Hace una pregunta")
    p.add_argument("pregunta")
    p.add_argument("-v", "--verbose", action="store_true", help="Muestra ruta, SQL, fuentes y resultados parciales")
    p.add_argument("--rol", default="analista", help="Rol del usuario (filtra documentos)")

    c = sub.add_parser("chat", help="Modo interactivo")
    c.add_argument("-v", "--verbose", action="store_true")
    c.add_argument("--rol", default="analista")

    d = sub.add_parser("init-db", help="Crea la base SQLite de ejemplo")
    d.add_argument("--clientes", type=int, default=2000)
    d.add_argument("--semilla", type=int, default=42)

    args = parser.parse_args(argv)

    if args.comando == "init-db":
        from multi_agent_rag.config import get_settings
        from multi_agent_rag.database import crear_base

        consola.print(f"Base creada en {crear_base(get_settings().db_path, args.clientes, args.semilla)}")
        return

    from multi_agent_rag.graph import construir_grafo, preguntar

    grafo = construir_grafo()
    if args.comando == "preguntar":
        with consola.status("Pensando..."):
            r = preguntar(args.pregunta, grafo, args.rol)
        _mostrar(r, args.verbose)
        return

    consola.print("Escribí tu pregunta (o 'salir').")
    while True:
        try:
            pregunta = consola.input("[bold]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in {"salir", "exit", "quit"}:
            break
        if pregunta:
            with consola.status("Pensando..."):
                r = preguntar(pregunta, grafo, args.rol)
            _mostrar(r, args.verbose)


if __name__ == "__main__":
    main()
