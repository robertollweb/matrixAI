"""165-C3 — el recibo `anomaly_review` («Filas a revisar») y las cifras con etiquetas aportadas.

Lo que se prueba: que el tipo existe y el verificador lo exige entero (cada sección, y sus reglas: sin etiquetas NO hay cifras de
acierto en ningún sitio), que tocar una cifra rompe la firma, que el texto de límites (medido por el banco del C2) y lo que no
afirma van en los dos idiomas, y que con etiquetas salen AUROC/AP/precisión@k con intervalo y un valor ausente es AUSENTE.
"""
from __future__ import annotations

import base64
import copy
import json
import random

import pytest

from matrixai.estudio import filas_a_revisar as fr
from matrixai.estudio.metricas import Muestra, calcular
from matrixai.pipelines import receipt as R

CLAVE = b"k" * 32
HUELLA = "ab" * 32


def _filas(n: int = 150) -> list[dict]:
    azar = random.Random(7)
    filas = []
    for i in range(1, n + 1):
        filas.append({"id": str(i), "edad": str(azar.randint(20, 60)),
                      "importe": f"{azar.gauss(1200, 150):.2f}".replace(".", ","),
                      "ciudad": azar.choice(["Zaragoza", "Málaga", "Ávila"]), "revisada": "no"})
    for i in (10, 77):
        filas[i]["importe"] = "20000,10"
        filas[i]["revisada"] = "sí"
    return filas


@pytest.fixture(scope="module")
def resultado():
    from matrixai_engines.anomalias import puntuar_filas  # noqa: PLC0415
    return puntuar_filas(_filas(), ["edad", "importe", "ciudad", "id"])


def _recibo(resultado, etiquetas=None):
    return fr.recibo_de_filas_a_revisar(resultado, huella_del_csv=HUELLA, creado_en="2026-11-13T00:00:00Z", nombre="prueba",
                                        etiquetas=etiquetas)


def _etiquetadas(resultado, **kw):
    por_fila = {x["fila"]: x["puntuacion"] for x in resultado["filas_a_revisar"]}
    pun = [por_fila[i + 1] for i in range(150)]
    an = [f["revisada"] == "sí" for f in _filas()]
    return {**fr.cifras_con_etiquetas(pun, an, remuestras=100, **kw), "columna": "revisada", "valor_anomalo": "sí"}


def test_el_tipo_existe_con_sus_secciones_y_no_toca_el_de_los_estudios():
    assert R._SECCIONES_POR_TIPO["anomaly_review"] == ("dataset", "detector", "columns", "stability", "limits", "no_claims", "labels")
    assert R._SECCIONES_POR_TIPO["study_execution"] == ("dataset", "protocolo", "candidatos", "seleccion", "pipeline_final",
                                                        "calibrador_umbral", "entorno")


def test_sin_etiquetas_firma_verifica_y_no_lleva_ninguna_cifra_de_acierto(resultado):
    recibo = _recibo(resultado)
    assert R.problemas_de_esquema(recibo) == []
    assert "metrics" not in recibo and recibo["labels"]["source"] == "none"
    assert R._nombres_de_cifras(recibo) == []
    sobre = R.firmar_recibo(recibo, clave=CLAVE, key_id="t")
    assert R.verificar_recibo(sobre, clave=CLAVE)["ok"] is True
    # una columna del CSV llamada «auroc» o «precision» no es una cifra del recibo
    assert R.problemas_de_esquema(_recibo({**resultado, "columnas_usadas": [{"columna": "precision", "tipo": "numerica"}]})) == []


def test_tocar_una_cifra_rompe_la_firma(resultado):
    sobre = R.firmar_recibo(_recibo(resultado), clave=CLAVE, key_id="t")
    cuerpo = json.loads(base64.b64decode(sobre["payload"]))
    cuerpo["stability"]["mean_overlap"] = 0.99 if cuerpo["stability"]["mean_overlap"] != 0.99 else 0.98
    toc = dict(sobre, payload=base64.b64encode(json.dumps(cuerpo, separators=(",", ":"), ensure_ascii=False).encode()).decode())
    assert R.verificar_recibo(toc, clave=CLAVE)["ok"] is False


@pytest.mark.parametrize("seccion", list(R._SECCIONES_POR_TIPO["anomaly_review"]))
def test_el_verificador_exige_cada_seccion(resultado, seccion):
    recibo = _recibo(resultado)
    del recibo[seccion]
    assert any(repr(seccion) in p for p in R.problemas_de_esquema(recibo))
    with pytest.raises(R.ReciboInvalido):
        R.firmar_recibo(recibo, clave=CLAVE, key_id="t")


def test_el_texto_de_limites_va_en_los_dos_idiomas_con_lo_medido(resultado):
    recibo = _recibo(resultado)
    assert "SEMBRADAS" in recibo["limits"]["es"] and "SEEDED" in recibo["limits"]["en"]
    for idioma in ("es", "en"):
        for tipo in fr.LIMITES_MEDIDOS.values():
            for tasa, v in (tipo or {}).items():
                assert f"{round(100 * v)} %" in recibo["limits"][idioma]
    # aud1 I4 -> C2b (regla 845586e7…, DECIDIDO 14-11): la categoría rara con la siembra NEUTRA (11 % / 29 %), no la del C2 (7 % / 37 %,
    # que caía al final del alfabeto); ya no «en revisión» ni «depende del nombre» (con el orden por frecuencia no depende: lo prueba
    # motores, `test_c2b_renombrar_un_nivel_no_cambia_la_lista_en_un_conjunto_real`); y lo que sí queda: la vacía es un valor del centro.
    assert fr.LIMITES_MEDIDOS["categoria_rara"] == {"1 %": 0.111, "5 %": 0.294}
    assert "categoría rara, el 11 % y el 29 %" in recibo["limits"]["es"] and "rare category, 11 % and 29 %" in recibo["limits"]["en"]
    assert "una celda vacía cuenta como un valor del centro" in recibo["limits"]["es"]
    assert "an empty cell counts as a value in the middle" in recibo["limits"]["en"]
    for idioma in ("es", "en"):
        for viejo in ("7 %", "37 %", "REVISIÓN", "REVIEW", "cómo se llamen", "what the categories are called"):
            assert viejo not in recibo["limits"][idioma], viejo
    for quitado in ({}, {"es": recibo["limits"]["es"]}, {"es": recibo["limits"]["es"], "en": " "}):
        malo = copy.deepcopy(recibo)
        malo["limits"] = quitado if quitado else {"es": "", "en": ""}
        assert any("limits" in p for p in R.problemas_de_esquema(malo)), quitado


def test_lo_que_no_afirma_va_en_los_dos_idiomas_y_coincide_con_el_de_motores(resultado):
    from matrixai_engines.anomalias import NO_AFIRMA as motores  # noqa: PLC0415
    assert len(fr.NO_AFIRMA["es"]) == len(fr.NO_AFIRMA["en"]) == len(motores)
    malo = _recibo(resultado)
    malo["no_claims"]["en"] = []
    assert any("no_claims" in p for p in R.problemas_de_esquema(malo))


@pytest.mark.parametrize("donde", ["metrics", "dataset", "stability", "labels"])
def test_sin_etiquetas_una_cifra_de_acierto_en_cualquier_sitio_se_rechaza(resultado, donde):
    recibo = _recibo(resultado)
    if donde == "metrics":
        recibo["metrics"] = [{"name": "auroc", "value": 0.9}]
    else:
        recibo[donde]["auroc"] = 0.9
    assert any("sin etiquetas" in p for p in R.problemas_de_esquema(recibo))
    with pytest.raises(R.ReciboInvalido):
        R.firmar_recibo(recibo, clave=CLAVE, key_id="t")


def test_con_etiquetas_salen_tres_cifras_con_intervalo_y_el_recibo_las_lleva(resultado):
    e = _etiquetadas(resultado)
    assert [m["name"] for m in e["metrics"]] == list(fr.CIFRAS_CON_ETIQUETAS)
    for m in e["metrics"]:
        assert m["ci_low"] is not None and m["ci_low"] <= m["value"] <= m["ci_high"] and m["level"] == 0.95
    # la misma cuenta que el catálogo del núcleo, no otra
    pun = [x["puntuacion"] for x in resultado["filas_a_revisar"]]
    an = [x["fila"] in (11, 78) for x in resultado["filas_a_revisar"]]
    muestra = Muestra(task="binary_classification", y_true=tuple("1" if a else "0" for a in an), classes=("0", "1"),
                      positive_label="1", scores=tuple(pun))
    assert e["metrics"][0]["value"] == calcular("auroc", muestra).value
    recibo = _recibo(resultado, e)
    assert R.problemas_de_esquema(recibo) == [] and recibo["labels"]["source"] == "provided"
    assert "optimistas" in recibo["labels"]["scope"]["es"] and "optimistic" in recibo["labels"]["scope"]["en"]
    assert R.verificar_recibo(R.firmar_recibo(recibo, clave=CLAVE, key_id="t"), clave=CLAVE)["ok"] is True


def test_precision_en_k_a_mano():
    # puntuaciones 5..1; anómalas: la 1.ª y la 3.ª. k=2: de las 2 primeras, 1 es anómala -> 0,5
    e = fr.cifras_con_etiquetas([5, 4, 3, 2, 1], [True, False, True, False, False], k=2, remuestras=50)
    assert e["k"] == 2 and e["metrics"][2]["value"] == 0.5


def test_una_sola_clase_es_ausente_con_motivo_y_nunca_cero(resultado):
    e = fr.cifras_con_etiquetas([3, 2, 1, 0.5], [False] * 4, remuestras=50)
    auroc = e["metrics"][0]
    assert auroc["value"] is None and auroc["undefined_reason"]["es"] and auroc["ci_low"] is None
    assert e["metrics"][1]["value"] is None
    assert e["metrics"][2]["value"] is None and e["metrics"][2]["undefined_reason"]["es"]   # «0 de k» no es una precisión
    recibo = _recibo(resultado, {**e, "columna": "r", "valor_anomalo": "x"})
    assert R.problemas_de_esquema(recibo) == []     # ausente con motivo es honesto
    recibo["metrics"][0]["undefined_reason"] = None     # ausente SIN motivo: un cero por la puerta de atrás
    assert any("exactamente uno" in p for p in R.problemas_de_esquema(recibo))


def test_con_etiquetas_sin_metrics_o_con_cero_por_ausente_se_rechaza(resultado):
    recibo = _recibo(resultado, _etiquetadas(resultado))
    sin = copy.deepcopy(recibo)
    del sin["metrics"]
    assert any("falta 'metrics'" in p for p in R.problemas_de_esquema(sin))
    fuera = copy.deepcopy(recibo)
    fuera["metrics"][0]["value"] = 1.5
    assert any("entre 0 y 1" in p for p in R.problemas_de_esquema(fuera))


def test_la_huella_de_la_lista_cambia_si_se_toca_un_rango(resultado):
    lista = resultado["filas_a_revisar"][:10]
    otra = copy.deepcopy(lista)
    otra[0]["rango"] = 2
    assert fr.digest_de_la_lista(lista) != fr.digest_de_la_lista(otra)
    assert _recibo(resultado)["output"]["output_digests"]["list"] == fr.digest_de_la_lista(resultado["filas_a_revisar"])


def test_un_tipo_inventado_sigue_sin_verificarse():
    assert any("desconocido" in p for p in R.problemas_de_esquema({"event_type": "anomaly_reviw"}))


@pytest.mark.parametrize("valor", [1.2, -0.1, True, "0,9"])
def test_una_estabilidad_que_no_es_un_solape_se_rechaza(resultado, valor):
    """Supervisor (13-11): quitar el rango 0–1 de la regla 4 pasaba la suite (sabotaje verde)."""
    recibo = _recibo(resultado)
    malo = copy.deepcopy(recibo)
    malo["stability"]["mean_overlap"] = valor
    assert any("stability.mean_overlap" in p for p in R.problemas_de_esquema(malo))
    assert not any("stability" in p for p in R.problemas_de_esquema(recibo))


def test_una_cifra_sin_intervalo_tiene_que_decir_por_que(resultado):
    """Supervisor (13-11): quitar la exigencia del motivo del intervalo ausente pasaba la suite (sabotaje verde)."""
    recibo = _recibo(resultado, _etiquetadas(resultado))
    con_valor = next(m for m in recibo["metrics"] if m.get("value") is not None)
    malo = copy.deepcopy(recibo)
    m = next(x for x in malo["metrics"] if x["name"] == con_valor["name"])
    m["ci_low"] = None
    m.pop("ci_unavailable", None)
    assert any(f"{con_valor['name']}: sin intervalo tiene que decir por qué" in p for p in R.problemas_de_esquema(malo))


# ---------------------------------------------------------------------------------------------------------------------------
# aud1 B1: el recibo verifica con EL VERIFICADOR (`verificar_sobre` y la CLI), no solo su firma
# ---------------------------------------------------------------------------------------------------------------------------

def _hojas(o, ruta=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _hojas(v, ruta + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _hojas(v, ruta + (i,))
    else:
        yield ruta


def _con_payload(sobre, cuerpo):
    return dict(sobre, payload=base64.b64encode(json.dumps(cuerpo, separators=(",", ":"), sort_keys=True).encode()).decode())


@pytest.mark.parametrize("con_etiquetas", [False, True])
def test_b1_el_recibo_pasa_el_verificador_entero_y_deja_de_pasar_al_tocar_cualquier_campo(resultado, con_etiquetas):
    """aud1 B1: `steps[0].entry_hash` salía sin `sha256:` y `verificar_sobre` daba ok=False SIEMPRE («no es un sha256 completo»),
    con la firma buena. Las pruebas solo miraban la firma. Ahora: ok=True sin un solo problema, y cada hoja tocada lo rompe."""
    from matrixai.pipelines.verifier import verificar_sobre  # noqa: PLC0415
    recibo = _recibo(resultado, _etiquetadas(resultado) if con_etiquetas else None)
    assert recibo["steps"][0]["entry_hash"] == recibo["dataset"]["digest"] == f"sha256:{HUELLA}"
    sobre = fr.firmar_recibo_de_filas_a_revisar(recibo, clave=CLAVE, key_id="t")
    v = verificar_sobre(sobre, clave=CLAVE)
    assert v["ok"] is True and v["problems"] == [] and "schema" in v["verified"] and "signature" in v["verified"], v
    cuerpo = json.loads(base64.b64decode(sobre["payload"]))
    tocadas = 0
    for ruta in _hojas(cuerpo):
        c = copy.deepcopy(cuerpo)
        d = c
        for k in ruta[:-1]:
            d = d[k]
        viejo = d[ruta[-1]]
        d[ruta[-1]] = (viejo + "x") if isinstance(viejo, str) else ("x" if viejo is None or isinstance(viejo, bool) else viejo + 1)
        assert verificar_sobre(_con_payload(sobre, c), clave=CLAVE)["ok"] is False, ruta
        tocadas += 1
    assert tocadas > 40


def test_b1_un_entry_hash_sin_prefijo_lo_rechaza_el_verificador(resultado):
    """El control del instrumento: el verificador SÍ caza el defecto de antes (si no, la prueba de arriba no mediría nada)."""
    from matrixai.pipelines.verifier import verificar_sobre  # noqa: PLC0415
    recibo = _recibo(resultado)
    recibo["steps"][0]["entry_hash"] = HUELLA
    v = verificar_sobre(fr.firmar_recibo_de_filas_a_revisar(recibo, clave=CLAVE, key_id="t"), clave=CLAVE)
    assert v["ok"] is False and any("entry_hash" in x for x in v["problems"])


def test_menor2_la_precision_en_k_sin_dos_clases_dice_su_motivo_no_el_del_auroc(resultado):
    c = fr.cifras_con_etiquetas([0.1, 0.5, 0.9], [False, False, False], remuestras=20)
    pk = next(m for m in c["metrics"] if m["name"] == "precision_at_k")
    assert pk["value"] is None and "precisión en las k primeras" in pk["undefined_reason"]["es"]
    assert "precision at k" in pk["undefined_reason"]["en"]
    todas = fr.cifras_con_etiquetas([0.1, 0.5, 0.9], [True, True, True], remuestras=20)
    assert "ninguna normal" in next(m for m in todas["metrics"] if m["name"] == "precision_at_k")["undefined_reason"]["es"]


@pytest.mark.parametrize("clave", fr.CLAVES_DE_MOTIVO)
def test_cada_clave_de_motivo_tiene_texto_en_los_dos_idiomas(clave):
    p = {"sin_dato": 9, "filas": 10, "distintos": 9, "presentes": 10, "por": "tramo", "celdas": 3, "ejemplo": "1.234",
         "textos": {"NA": 2}, "motivo_de_la_lectura": "ambigua", "minimo": 200, "columna": "c", "columnas": ["c"],
         "formato": "%Y-%m-%d", "repetidas": ["c"], "vacias": 1}
    t = fr.texto_de_motivo(clave, p)
    assert t["es"].strip() and t["en"].strip() and t["es"] != t["en"]
    with pytest.raises(KeyError):
        fr.texto_de_motivo("no_existe", {})


# ---------------------------------------------------------------------------------------------------------- auditoría 2 del 165
def test_aud2_i3_el_recibo_declara_la_version_medida_del_detector_y_de_sklearn_y_numpy(resultado):
    import numpy  # noqa: PLC0415
    import sklearn  # noqa: PLC0415
    det = _recibo(resultado)["detector"]
    assert det["engine_version"] == resultado["detector"]["versiones"]["detector"]
    assert det["versions"]["scikit_learn"] == sklearn.__version__ and det["versions"]["numpy"] == numpy.__version__
    assert R.problemas_de_esquema(_recibo(resultado)) == []


@pytest.mark.parametrize("quitar", ["engine_version", "versions.scikit_learn", "versions.numpy", "versions"])
@pytest.mark.parametrize("como", ["falta", "nulo", "vacio"])
def test_aud2_i3_el_verificador_exige_las_versiones(resultado, quitar, como):
    recibo = _recibo(resultado)
    sitio, _, hoja = quitar.partition(".")
    padre, clave = (recibo["detector"], sitio) if not hoja else (recibo["detector"][sitio], hoja)
    if como == "falta":
        del padre[clave]
    else:
        padre[clave] = None if como == "nulo" else ""
    problemas = R.problemas_de_esquema(recibo)
    assert any("detector." in p and "versi" in p for p in problemas), problemas


def test_aud2_los_textos_de_limites_dicen_las_dos_codificaciones_y_los_empates():
    es, en = fr.TEXTO_DE_LIMITES["es"], fr.TEXTO_DE_LIMITES["en"]
    assert "forma anterior de ordenar las categorías" in es and "previous way of ordering categories" in en
    assert "con empates" in es and "with ties" in en


def test_aud2_el_recibo_dice_de_que_es_la_huella():
    assert "sin BOM" in fr._HUELLA_DE["es"] and "without BOM" in fr._HUELLA_DE["en"]


# ---------------------------------------------------------------------------------------------------------- auditoría 3 del 165
def test_aud3_i2_los_limites_dicen_que_formas_de_fecha_reconoce_el_nucleo_y_cuales_no():
    """Lo que el texto de límites cita como fecha reconocida lo reconoce `_detect_date_format` (la regla que usa motores), y lo que cita
    como NO reconocida no lo reconoce: si el núcleo aprende una forma nueva, este texto miente y la prueba se pone roja. (El lado del
    detector, que esas columnas se excluyan o se miren, lo ata `test_aud3_i2_lo_que_los_limites_dicen_de_las_fechas_es_lo_que_hace_
    el_detector` en motores.)"""
    from matrixai.training.dataset_analysis import _detect_date_format  # noqa: PLC0415
    for forma in ("2024-03-15", "2024-03-15 09:00", "15/03/2024", "03/15/2024", "2024/03/15", "15-03-2024"):
        assert _detect_date_format([forma]) is not None, forma
        assert all(forma in fr.TEXTO_DE_LIMITES[i] for i in ("es", "en")), forma
    for forma in ("15/03/2024 09:00", "15/03/24"):
        assert _detect_date_format([forma]) is None, forma
        assert all(forma in fr.TEXTO_DE_LIMITES[i] for i in ("es", "en")), forma
    assert "NO se reconocen" in fr.TEXTO_DE_LIMITES["es"] and "NOT recognised" in fr.TEXTO_DE_LIMITES["en"]


def test_aud3_i2_el_aviso_de_marcas_en_una_fecha_no_dice_numerica():
    t = fr.texto_de_motivo("textos_leidos_como_faltante", {"columna": "alta", "textos": {"NA": 3}, "como": "fecha"})
    assert "sigue siendo una fecha" in t["es"] and "numérica" not in t["es"] and "still a date" in t["en"]
    viejo = fr.texto_de_motivo("textos_leidos_como_faltante", {"columna": "x", "textos": {"NA": 3}})
    assert "sigue siendo numérica" in viejo["es"] and "still numeric" in viejo["en"]


def test_aud3_i1_la_cabecera_invalida_dice_cual_se_repite_y_cuantas_no_tienen_nombre():
    t = fr.texto_de_motivo("cabecera_invalida", {"repetidas": ["importe"], "vacias": 0})
    assert "«importe»" in t["es"] and "«importe»" in t["en"] and "sin nombre" not in t["es"]
    t = fr.texto_de_motivo("cabecera_invalida", {"repetidas": [], "vacias": 2})
    assert "2 sin nombre" in t["es"] and "2 without a name" in t["en"]
