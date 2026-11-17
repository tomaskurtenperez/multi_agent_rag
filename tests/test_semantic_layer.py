def test_gana_la_coincidencia_mas_larga(capa):
    ids = [c.id for c in capa.buscar("¿Cuántos clientes activos tienen préstamos en mora?")]
    assert ids == ["cliente_activo", "prestamo", "mora"]
    assert "cliente" not in ids


def test_ignora_tildes_y_mayusculas(capa):
    assert [c.id for c in capa.buscar("PRESTAMOS HIPOTECARIOS")] == ["prestamo"]


def test_resolver_trae_tablas_relaciones_y_join(capa):
    r = capa.resolver("clientes activos con préstamos en mora", umbral_confianza=80)
    assert set(r["tablas"]) == {"clientes", "prestamos"}
    joins = [x["join"] for x in r["relaciones"] if x.get("join")]
    assert "clientes.id = prestamos.cliente_id" in joins
    assert r["advertencias"] == []


def test_esquema_oculta_columnas_sensibles(capa):
    columnas = capa.esquema(["clientes"])["clientes"]["columnas"]
    assert "email" not in columnas and "documento" not in columnas and "nombre" not in columnas
    assert {"email", "documento", "nombre"} <= capa.columnas_sensibles()


def test_concepto_no_certificado_genera_advertencia(capa):
    r = capa.resolver("clientes con contrato vigente", umbral_confianza=80)
    assert [c["id"] for c in r["conceptos"]] == ["cliente_vigente"]
    assert any("no está certificado" in a for a in r["advertencias"])


def test_catalogo_coincide_con_la_base(capa, settings):
    import sqlite3

    con = sqlite3.connect(settings.db_path)
    for tabla, meta in capa.tablas.items():
        reales = {fila[1] for fila in con.execute(f"PRAGMA table_info({tabla})")}
        assert reales == set(meta["columnas"]), tabla
    con.close()
