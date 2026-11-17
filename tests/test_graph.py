"""Pruebas de punta a punta del grafo con LLMs falsos (no requieren API key)."""

from conftest import FakeChat, llamar

from multi_agent_rag.database import ejecutar_sql
from multi_agent_rag.graph import Ruta, construir_grafo, numeros_no_respaldados, preguntar

PREGUNTA = "¿Cuántos clientes activos tienen préstamos en mora, y qué dice la política sobre cómo contactarlos?"

SQL_OK = (
    "SELECT COUNT(DISTINCT clientes.id) AS clientes FROM clientes "
    "JOIN prestamos ON clientes.id = prestamos.cliente_id "
    "WHERE clientes.fecha_ultimo_movimiento >= date('now', '-90 days') "
    "AND prestamos.estado = 'vigente' AND prestamos.dias_atraso > 30"
)


def _conteo(settings) -> int:
    return ejecutar_sql(SQL_OK, settings.db_path, {"clientes", "prestamos"})["filas"][0][0]


def _rapido_ambos(consulta_docs="cómo contactar a clientes en mora"):
    return FakeChat(guion=[
        Ruta(destinos=["datos", "documentos"]),
        llamar("buscar_documentos", consulta=consulta_docs),
        "El primer contacto es por correo electrónico (Fuente: Política de Cobranza v3, sección 4).",
    ])


def _principal(n: int, respuesta_final: str):
    return FakeChat(guion=[
        llamar("buscar_definicion", termino="clientes activos"),
        llamar("ejecutar_sql", "2", sql=SQL_OK),
        f"Resultado: {n}. SQL: {SQL_OK}",
        respuesta_final,
    ])


def test_flujo_completo_con_ambos_agentes_en_paralelo(crear_deps, settings):
    n = _conteo(settings)
    final = f"Hay {n} clientes activos con préstamos en mora. Primer contacto por correo (Fuente: Política de Cobranza v3, sección 4)."
    r = preguntar(PREGUNTA, construir_grafo(crear_deps(_principal(n, final), _rapido_ambos())))

    assert r["ruta"] == ["datos", "documentos"]
    assert r["sql_ejecutado"] == [SQL_OK]
    assert r["fuentes"][0]["fuente"] == "politica_cobranza_v3.md"
    assert any("cliente_activo" in e for e in r["evidencia"])  # salió de buscar_definicion
    assert r["advertencias"] == []
    assert r["respuesta"] == final


def test_numero_inventado_genera_advertencia(crear_deps, settings):
    n = _conteo(settings)
    final = "Hay 987654 clientes activos en mora (Fuente: Política de Cobranza v3, sección 4)."
    r = preguntar(PREGUNTA, construir_grafo(crear_deps(_principal(n, final), _rapido_ambos())))

    assert any("987654" in a for a in r["advertencias"])
    assert "Advertencia" in r["respuesta"]


def test_respuesta_sin_cita_genera_advertencia(crear_deps, settings):
    n = _conteo(settings)
    final = f"Hay {n} clientes activos en mora. Hay que mandarles un mail."
    r = preguntar(PREGUNTA, construir_grafo(crear_deps(_principal(n, final), _rapido_ambos())))

    assert "No se pudo verificar la fuente." in r["advertencias"]


def test_pregunta_fuera_de_alcance_va_directo_a_responder(crear_deps):
    principal = FakeChat(guion=["No tengo suficiente información para responder eso."])
    rapido = FakeChat(guion=[Ruta(destinos=[])])

    r = preguntar("¿Quién ganó el mundial de 1986?", construir_grafo(crear_deps(principal, rapido)))

    assert r["ruta"] == []
    assert principal.llamadas == 1 and rapido.llamadas == 1
    assert "resultado_datos" not in r and "resultado_docs" not in r


def test_agente_de_datos_corrige_sql_rechazado(crear_deps, settings):
    n = _conteo(settings)
    principal = FakeChat(guion=[
        llamar("ejecutar_sql", sql="DELETE FROM clientes"),
        llamar("ejecutar_sql", "2", sql=SQL_OK),
        f"Resultado: {n}",
        f"Hay {n} clientes.",
    ])
    rapido = FakeChat(guion=[Ruta(destinos=["datos"])])

    r = preguntar("¿Cuántos clientes activos tienen préstamos en mora?", construir_grafo(crear_deps(principal, rapido)))

    assert r["sql_ejecutado"] == ["DELETE FROM clientes", SQL_OK]
    assert r["evidencia"][0].startswith("Error:")
    assert r["advertencias"] == []


def test_limite_de_pasos_corta_un_agente_en_bucle(crear_deps):
    def en_bucle(mensajes):
        # El responder arma su prompt con "Pregunta:"; el agente de datos, en cambio, pide tools sin parar.
        if any("Pregunta:" in str(m.content) for m in mensajes):
            return "No tengo suficiente información."
        return llamar("buscar_definicion", str(len(mensajes)), termino="mora")

    principal = FakeChat(por_defecto=en_bucle)
    rapido = FakeChat(guion=[Ruta(destinos=["datos"])])

    r = preguntar("¿Cuánta mora hay?", construir_grafo(crear_deps(principal, rapido)))

    assert r["resultado_datos"] == "No se pudo obtener el dato dentro del límite de pasos."
    assert r["respuesta"] == "No tengo suficiente información."


def test_documentos_filtrados_por_rol(crear_deps):
    principal = FakeChat(guion=["No tengo suficiente información."])
    rapido = FakeChat(guion=[
        Ruta(destinos=["documentos"]),
        llamar("buscar_documentos", consulta="cómo contactar a clientes en mora"),
        "No encontré documentos.",
    ])

    r = preguntar(PREGUNTA, construir_grafo(crear_deps(principal, rapido)), rol="comercial")

    assert "politica_cobranza_v3.md" not in {f["fuente"] for f in r["fuentes"]}


def test_numeros_no_respaldados():
    evidencia = '{"filas": [[1240]]} más de 30 días, últimos 90 días'
    assert numeros_no_respaldados("Hay 1.240 clientes (90 días)", evidencia) == []
    assert numeros_no_respaldados("Hay 1.300 clientes", evidencia) == ["1300"]
