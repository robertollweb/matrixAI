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
            for tasa, v in tipo.items():
                assert f"{round(100 * v)} %" in recibo["limits"][idioma]
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
