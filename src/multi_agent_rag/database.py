"""Base de datos de ejemplo (SQLite) y ejecución segura de SQL de solo lectura.

Defensa en profundidad:
1. Validación estática: una sola sentencia SELECT/WITH, sin palabras de escritura,
   solo tablas permitidas y sin columnas sensibles.
2. Conexión abierta en modo solo lectura (`mode=ro`).
3. Autorizador de SQLite que deniega cualquier operación que no sea lectura de las
   tablas y columnas permitidas.
"""

from __future__ import annotations

import random
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

DDL = """
CREATE TABLE clientes (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL,
    documento TEXT NOT NULL,
    email TEXT NOT NULL,
    segmento TEXT NOT NULL,
    fecha_alta DATE NOT NULL,
    fecha_ultimo_movimiento DATE NOT NULL,
    estado_contrato TEXT NOT NULL
);
CREATE TABLE prestamos (
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    producto TEXT NOT NULL,
    monto_otorgado REAL NOT NULL,
    saldo_pendiente REAL NOT NULL,
    fecha_otorgamiento DATE NOT NULL,
    dias_atraso INTEGER NOT NULL,
    estado TEXT NOT NULL
);
CREATE INDEX idx_prestamos_cliente ON prestamos(cliente_id);
"""

RANGOS_MONTO = {
    "personal": (200_000, 5_000_000),
    "prendario": (3_000_000, 25_000_000),
    "hipotecario": (30_000_000, 150_000_000),
}


class ErrorSQL(Exception):
    """La consulta fue rechazada o falló al ejecutarse."""


# --- datos de ejemplo -----------------------------------------------------------------


def crear_base(ruta: Path | str, n_clientes: int = 2000, semilla: int = 42, hoy: date | None = None) -> Path:
    """Crea (o recrea) una base con clientes y préstamos sintéticos y reproducibles."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        ruta.unlink()

    hoy = hoy or date.today()
    rng = random.Random(semilla)
    clientes, prestamos = [], []
    pid = 1
    for cid in range(1, n_clientes + 1):
        segmento = rng.choices(["minorista", "pyme", "corporativo"], weights=[75, 20, 5])[0]
        alta = hoy - timedelta(days=rng.randint(30, 3650))
        dias_sin_mov = rng.randint(0, 90) if rng.random() < 0.6 else rng.randint(91, 720)
        ultimo_mov = max(alta, hoy - timedelta(days=dias_sin_mov))
        estado_contrato = "vigente" if rng.random() < 0.85 else "cerrado"
        clientes.append((
            cid, f"Cliente {cid:05d}", str(rng.randint(20_000_000, 45_000_000)),
            f"cliente{cid:05d}@ejemplo.com", segmento, alta.isoformat(),
            ultimo_mov.isoformat(), estado_contrato,
        ))

        for _ in range(rng.choices([0, 1, 2, 3], weights=[30, 45, 18, 7])[0]):
            producto = rng.choices(["personal", "hipotecario", "prendario"], weights=[65, 15, 20])[0]
            monto = round(rng.uniform(*RANGOS_MONTO[producto]), 2)
            otorgado = alta + timedelta(days=rng.randint(0, (hoy - alta).days))
            if rng.random() < 0.25:
                saldo, atraso, estado = 0.0, 0, "cancelado"
            else:
                saldo = round(monto * rng.uniform(0.1, 1.0), 2)
                r = rng.random()
                atraso = 0 if r < 0.78 else (rng.randint(1, 30) if r < 0.88 else rng.randint(31, 365))
                estado = "vigente"
            prestamos.append((pid, cid, producto, monto, saldo, otorgado.isoformat(), atraso, estado))
            pid += 1

    con = sqlite3.connect(ruta)
    try:
        con.executescript(DDL)
        con.executemany("INSERT INTO clientes VALUES (?,?,?,?,?,?,?,?)", clientes)
        con.executemany("INSERT INTO prestamos VALUES (?,?,?,?,?,?,?,?)", prestamos)
        con.commit()
    finally:
        con.close()
    return ruta


# --- validación y ejecución -----------------------------------------------------------

_PROHIBIDAS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex|truncate|grant)\b",
    re.IGNORECASE,
)


def limpiar_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    return sql.rstrip(";").strip()


def validar_sql(sql: str, tablas_permitidas: set[str], columnas_sensibles: set[str]) -> str:
    """Devuelve la consulta limpia o lanza ErrorSQL explicando el motivo."""
    sql = limpiar_sql(sql)
    if not sql:
        raise ErrorSQL("La consulta está vacía.")
    if ";" in sql:
        raise ErrorSQL("Solo se permite una sentencia.")
    if "--" in sql or "/*" in sql:
        raise ErrorSQL("No se permiten comentarios en la consulta.")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise ErrorSQL("Solo se permiten consultas SELECT.")
    if m := _PROHIBIDAS.search(sql):
        raise ErrorSQL(f"Palabra no permitida en consulta de solo lectura: {m.group(1).upper()}.")

    permitidas = {t.lower() for t in tablas_permitidas}
    ctes = {c.lower() for c in re.findall(r"(?:\bwith|,)\s*(\w+)\s+as\s*\(", sql, re.IGNORECASE)}
    for tabla in re.findall(r"\b(?:from|join)\s+([\w\"]+)", sql, re.IGNORECASE):
        tabla = tabla.strip('"').lower()
        if tabla not in permitidas and tabla not in ctes:
            raise ErrorSQL(f"Tabla no permitida: {tabla}. Permitidas: {', '.join(sorted(permitidas))}.")

    for col in columnas_sensibles:
        if re.search(rf"\b{re.escape(col)}\b", sql, re.IGNORECASE):
            raise ErrorSQL(f"La columna '{col}' es un dato sensible y no puede consultarse.")
    return sql


_SQLITE_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)


def _autorizador(tablas_permitidas: set[str], columnas_sensibles: set[str]):
    permitidas = {t.lower() for t in tablas_permitidas}
    sensibles = {c.lower() for c in columnas_sensibles}
    acciones = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, _SQLITE_FUNCTION, _SQLITE_RECURSIVE}

    def autorizar(accion, arg1, arg2, _db, _origen):
        if accion not in acciones:
            return sqlite3.SQLITE_DENY
        if accion == sqlite3.SQLITE_READ:
            if (arg1 or "").lower() not in permitidas or (arg2 or "").lower() in sensibles:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return autorizar


def ejecutar_sql(
    sql: str,
    ruta_db: Path | str,
    tablas_permitidas: set[str],
    columnas_sensibles: set[str] = frozenset(),
    max_filas: int = 100,
) -> dict[str, Any]:
    sql = validar_sql(sql, tablas_permitidas, columnas_sensibles)
    uri = Path(ruta_db).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        con.set_authorizer(_autorizador(tablas_permitidas, columnas_sensibles))
        cur = con.execute(sql)
        columnas = [d[0] for d in cur.description or []]
        filas = cur.fetchmany(max_filas + 1)
    except sqlite3.Error as e:
        raise ErrorSQL(f"Error de SQLite: {e}") from e
    finally:
        con.close()
    return {
        "sql": sql,
        "columnas": columnas,
        "filas": [list(f) for f in filas[:max_filas]],
        "truncado": len(filas) > max_filas,
    }
