from multi_agent_rag.retriever import BM25Retriever, cargar_fragmentos


def test_fragmentos_por_seccion(settings):
    fragmentos = cargar_fragmentos(settings.docs_dir)
    cobranza = [f for f in fragmentos if f.fuente == "politica_cobranza_v3.md"]
    assert cobranza[0].titulo == "Política de Cobranza v3"
    assert "4. Canales y secuencia de contacto" in [f.seccion for f in cobranza]


def test_encuentra_la_seccion_de_contacto(settings):
    r = BM25Retriever(cargar_fragmentos(settings.docs_dir))
    top = r.buscar("¿Cómo hay que contactar a los clientes en mora?", k=3)
    assert top[0].fuente == "politica_cobranza_v3.md"
    assert "Canales" in top[0].seccion


def test_encuentra_tasa_del_prestamo_personal(settings):
    r = BM25Retriever(cargar_fragmentos(settings.docs_dir))
    top = r.buscar("tasa de interés del préstamo personal", k=1)
    assert top[0].seccion == "1. Préstamo personal"


def test_filtra_por_rol(settings):
    r = BM25Retriever(cargar_fragmentos(settings.docs_dir))
    consulta = "¿Cómo hay que contactar a los clientes en mora?"
    assert all(f.fuente != "politica_cobranza_v3.md" for f in r.buscar(consulta, k=5, rol="comercial"))
    assert r.buscar(consulta, k=1, rol="cobranzas")[0].fuente == "politica_cobranza_v3.md"


def test_linea_de_acceso_no_queda_en_el_texto(settings):
    assert all("Acceso:" not in f.texto for f in cargar_fragmentos(settings.docs_dir))
