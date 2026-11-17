import pytest

from multi_agent_rag.database import ErrorSQL, ejecutar_sql, validar_sql

TABLAS = {"clientes", "prestamos"}
SENSIBLES = {"nombre", "documento", "email"}


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM clientes",
        "SELECT 1; DROP TABLE clientes",
        "UPDATE clientes SET segmento = 'x'",
        "SELECT * FROM sqlite_master",
        "SELECT email FROM clientes",
        "SELECT id FROM clientes -- comentario",
        "PRAGMA table_info(clientes)",
    ],
)
def test_rechaza_consultas_peligrosas(sql):
    with pytest.raises(ErrorSQL):
        validar_sql(sql, TABLAS, SENSIBLES)


def test_limpia_bloques_de_codigo():
    assert validar_sql("```sql\nSELECT COUNT(*) FROM clientes;\n```", TABLAS, SENSIBLES) == "SELECT COUNT(*) FROM clientes"


def test_ejecuta_consulta_valida(settings):
    r = ejecutar_sql(
        "SELECT COUNT(DISTINCT clientes.id) AS n FROM clientes JOIN prestamos ON clientes.id = prestamos.cliente_id "
        "WHERE clientes.fecha_ultimo_movimiento >= date('now', '-90 days') "
        "AND prestamos.estado = 'vigente' AND prestamos.dias_atraso > 30",
        settings.db_path, TABLAS, SENSIBLES,
    )
    assert r["columnas"] == ["n"]
    assert r["filas"][0][0] > 0


def test_autorizador_bloquea_select_estrella_con_columnas_sensibles(settings):
    # Pasa la validación estática (no nombra columnas) pero el autorizador lo frena.
    with pytest.raises(ErrorSQL):
        ejecutar_sql("SELECT * FROM clientes", settings.db_path, TABLAS, SENSIBLES)


def test_autorizador_bloquea_tablas_no_permitidas_en_join_implicito(settings):
    with pytest.raises(ErrorSQL):
        ejecutar_sql("SELECT COUNT(*) FROM clientes, prestamos", settings.db_path, {"clientes"}, SENSIBLES)
