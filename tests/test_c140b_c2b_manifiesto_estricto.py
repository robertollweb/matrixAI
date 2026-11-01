# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140b-C2b (núcleo) — el manifiesto, el expediente y la ficha TRIPOD admiten el reparto ESTRICTO (`groups_and_time`).

Antes de este corte `DISENOS_DEL_REPARTO` era `("iid","groups","temporal")` y un estudio por sujeto y por el tiempo reventaba al
exportar (el backend lo esquivaba exportando sin `study.split`: la ficha decía «no disponible»). Ahora el manifiesto valida y
graba lo que el reparto estricto declara de sí mismo (`RepartoEstricto.a_json()`, que el Studio manda como `reparto.estricto`) y
la ficha lo dice: el diseño, el corte, las filas SIN ROL con su porcentaje y el tipo de pliegues; el intervalo, condicional a la
ventana. La ficha NO redacta cifras de cobertura (las decide Roberto).

Los datos son los del reparto REAL del 140b-C1 (`proponer_particion_estricta`), no un diccionario escrito a mano: lo que se
afirma es lo que el núcleo declara de verdad.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_c140b_c1_particion_estricta import _cohorte, _estricta   # noqa: E402

from matrixai.estudio import Horizonte   # noqa: E402
from matrixai.export.expediente_clinico import (   # noqa: E402
    FILAS_DEL_ESTUDIO, FILAS_DEL_ESTUDIO_CON_FILAS_SIN_ROL, FILAS_SIN_ROL, REPARTO_POR_DISENO, Campo, ExpedienteClinico,
    texto_del_valor)
from matrixai.export.manifiesto_de_estudio import (   # noqa: E402
    DISENOS_DEL_REPARTO, escribir_manifiesto_de_estudio, manifiesto_de_estudio)
from matrixai.export.probast import mapa_probast   # noqa: E402
from matrixai.export.reproduce import ReproduceManifestError   # noqa: E402
from matrixai.export.tripod import ficha_tripod   # noqa: E402


# ---------------------------------------------------------------------------------------------------------------------------
# El sobre `reparto` del Studio, hecho como lo hace el Studio (`declaracion_del_reparto` + `declaracion_del_reparto_estricto`)
# ---------------------------------------------------------------------------------------------------------------------------

def _sobre_de(propuesta, *, columna_de_tiempo_como_entrada=None):
    """Lo que el backend manda al manifiesto: el bloque `reparto` del sobre de selección. `estricto` es `a_json()` del núcleo con
    `unidades_de_prueba` como NÚMERO de sujetos (el Studio no manda sus nombres), y los límites con su frase en los dos idiomas."""
    reparto, plan = propuesta.reparto, propuesta.plan
    estricto = dict(reparto.a_json())
    estricto["unidades_de_prueba"] = len(reparto.unidades_de_prueba)
    if columna_de_tiempo_como_entrada is not None:
        estricto["columna_de_tiempo_como_entrada"] = columna_de_tiempo_como_entrada
    gap = plan.gap
    return {
        "diseno": "groups_and_time", "unit_id_field": plan.unit_id_field, "time_column": plan.time_column,
        "gap": {"magnitud": gap.magnitud, "unidad": gap.unidad} if gap is not None else None,
        "gap_origen": "horizonte_declarado" if gap is not None else None,
        "cobertura_fuera_de_pliegue": None,
        "limites": [{"clave": l.clave, "motivo": dict(l.motivo), "medida": dict(l.medida or {})} for l in propuesta.limites],
        "estricto": estricto}


@pytest.fixture(scope="module")
def f2q():
    """Parkinson en pequeño: 30 sujetos × 20 visitas, entrada escalonada, fechas `DD/MM/AAAA`, CSV barajado. Pliegues F2q."""
    filas = _cohorte(n=30, visitas=20, entrada=300, tiempo="dmy")
    p = _estricta(filas, folds=3, gap=Horizonte(magnitud=7, unidad="days"))
    assert p.es_viable and p.reparto.tipo_de_pliegues == "por_sujeto_y_ultimo_tramo"
    return filas, p, _sobre_de(p)


@pytest.fixture(scope="module")
def f1():
    """40 sujetos × 7 visitas: el desarrollo no llega a 300 filas (pliegues solo por sujeto, F1) y la escalera relaja el corte."""
    filas = _cohorte(n=40, visitas=7, entrada=30, tiempo="iso")
    p = _estricta(filas, folds=3)
    assert p.es_viable and p.reparto.tipo_de_pliegues == "por_sujeto"
    assert (p.reparto.fraccion_de_sujetos, p.reparto.cuantil) != (p.reparto.fraccion_de_sujetos_pedida, p.reparto.cuantil_pedido)
    return filas, p, _sobre_de(p)


def _manifiesto(filas, p, sobre, **kw):
    r = p.reparto
    return manifiesto_de_estudio(filas=len(filas), filas_de_desarrollo=len(r.desarrollo), filas_de_prueba_reservada=len(r.prueba),
                                 motor_del_modelo_entregado="sklearn.hgb", reparto_del_estudio=sobre, **kw)


# ---------------------------------------------------------------------------------------------------------------------------
# El manifiesto
# ---------------------------------------------------------------------------------------------------------------------------

def test_el_vocabulario_admite_el_reparto_estricto():
    assert DISENOS_DEL_REPARTO == ("iid", "groups", "temporal", "groups_and_time")


class TestElManifiestoGrabaElRepartoEstricto:

    def test_f2q_graba_el_corte_las_filas_sin_rol_los_pliegues_y_el_intervalo(self, f2q):
        filas, p, sobre = f2q
        r = p.reparto
        split = _manifiesto(filas, p, sobre)["study"]["split"]
        assert split["design"] == "groups_and_time" and (split["unit_column"], split["time_column"]) == ("sujeto", "t")
        assert split["gap"] == {"magnitude": 7, "unit": "days"} and split["gap_origin"] == "declared_horizon"
        s = split["strict"]
        assert (s["cut"], s["cut_readable"], s["gap_seconds"]) == (r.corte, r.corte_legible, 7 * 86400.0)
        assert s["cut_readable"] == "2024-09-29"
        # lo pedido y lo usado, por separado
        assert s["subject_fraction"] == {"requested": 0.3, "used": 0.3} and s["quantile"] == {"requested": 0.7, "used": 0.7}
        assert s["cuts_tried"] == 1
        # las filas de cada rol, y las SIN ROL con su porcentaje
        assert (s["development_rows"], s["test_rows"], s["discarded_rows"]) == (337, 114, 149)
        assert s["development_rows"] + s["test_rows"] + s["discarded_rows"] == len(filas) == 600
        assert s["discarded_percentage"] == 24.8
        # sujetos: números, no nombres
        assert (s["development_units"], s["test_units"]) == (18, 10)
        assert s["fold_type"] == "by_subject_and_last_stretch" and s["folds"] == 3
        assert s["last_stretch_start"] == r.inicio_del_tramo
        assert s["interval"] == {
            "method": "cluster_bootstrap_by_unit", "resampling_design": "groups", "resampling_unit": "unit",
            "conditional_on_the_window": True, "test_units": 10, "test_window": list(r.intervalo.ventana_de_prueba)}
        # y el digest cubre el bloque
        assert _manifiesto(filas, p, sobre)["manifest_sha256"] == _manifiesto(filas, p, sobre)["manifest_sha256"]
        otro = copy.deepcopy(sobre)
        otro["estricto"]["corte_legible"] = "2025-01-01"
        assert _manifiesto(filas, p, otro)["manifest_sha256"] != _manifiesto(filas, p, sobre)["manifest_sha256"]

    def test_f1_graba_pliegues_solo_por_sujeto_sin_tramo_y_lo_pedido_distinto_de_lo_usado(self, f1):
        filas, p, sobre = f1
        s = _manifiesto(filas, p, sobre)["study"]["split"]["strict"]
        assert s["fold_type"] == "by_subject" and s["last_stretch_start"] is None
        assert s["subject_fraction"]["requested"] != s["subject_fraction"]["used"] or s["quantile"]["requested"] != s["quantile"]["used"]
        assert s["quantile"]["requested"] == 0.7 and s["quantile"]["used"] == p.reparto.cuantil
        # los límites del reparto (corte relajado, pliegues por sujeto) viajan con su frase en los dos idiomas
        claves = {l["key"] for l in _manifiesto(filas, p, sobre)["study"]["split"]["limits"]}
        assert {"estricto_corte_relajado", "estricto_pliegues_por_sujeto_por_pocas_filas", "estricto_filas_descartadas"} <= claves

    def test_la_columna_de_tiempo_como_entrada_se_valida_y_se_graba(self, f2q):
        filas, p, _ = f2q
        entrada = {"columna": "t", "entra": False, "motivo": {"es": "no entra", "en": "does not go in"}}
        s = _manifiesto(filas, p, _sobre_de(p, columna_de_tiempo_como_entrada=entrada))["study"]["split"]["strict"]
        assert s["time_column_as_input"] == {"column": "t", "enters": False, "reason": {"es": "no entra", "en": "does not go in"}}
        assert "time_column_as_input" not in _manifiesto(filas, p, _sobre_de(p))["study"]["split"]["strict"]
        for malo in ({"columna": "t", "entra": "no", "motivo": {"es": "a", "en": "b"}},
                     {"columna": "t", "entra": True, "motivo": {"es": "a"}}, {"columna": "t", "entra": True}):
            with pytest.raises(ReproduceManifestError, match="columna_de_tiempo_como_entrada"):
                _manifiesto(filas, p, _sobre_de(p, columna_de_tiempo_como_entrada=malo))

    def test_los_demas_disenos_no_llevan_el_bloque_strict(self):
        for rep in ({"diseno": "iid"}, {"diseno": "groups", "unit_id_field": "p"},
                    {"diseno": "temporal", "time_column": "t", "gap": {"magnitud": 1, "unidad": "days"}}):
            assert "strict" not in manifiesto_de_estudio(reparto_del_estudio=rep)["study"]["split"]


def _con(sobre, **cambios):
    """Una copia del sobre con `estricto` cambiado (una clave `None` en `cambios` la quita)."""
    nuevo = copy.deepcopy(sobre)
    for clave, valor in cambios.items():
        if valor is _QUITAR:
            nuevo["estricto"].pop(clave)
        else:
            nuevo["estricto"][clave] = valor
    return nuevo


_QUITAR = object()


class TestUnRepartoEstrictoMalFormadoSeRechazaConSuMotivo:

    def test_groups_and_time_sin_estricto_no_se_publica(self, f2q):
        _, _, sobre = f2q
        sin = {k: v for k, v in sobre.items() if k != "estricto"}
        with pytest.raises(ReproduceManifestError, match="estricto is required for groups_and_time"):
            manifiesto_de_estudio(reparto_del_estudio=sin)
        with pytest.raises(ReproduceManifestError, match="estricto is required for groups_and_time"):
            manifiesto_de_estudio(reparto_del_estudio={"diseno": "groups_and_time", "unit_id_field": "p", "time_column": "t"})

    @pytest.mark.parametrize("falta", ["unit_id_field", "time_column"])
    def test_groups_and_time_sin_sus_dos_columnas_no_se_publica(self, f2q, falta):
        _, _, sobre = f2q
        sin = {k: v for k, v in sobre.items() if k != falta}
        with pytest.raises(ReproduceManifestError, match=f"{falta} is required for groups_and_time"):
            manifiesto_de_estudio(reparto_del_estudio=sin)

    @pytest.mark.parametrize("diseno", ["iid", "groups", "temporal"])
    def test_estricto_en_otro_diseno_se_rechaza(self, f2q, diseno):
        _, _, sobre = f2q
        with pytest.raises(ReproduceManifestError, match="only belongs to groups_and_time"):
            manifiesto_de_estudio(reparto_del_estudio={**sobre, "diseno": diseno})

    @pytest.mark.parametrize("cambio,mensaje", [
        ({"clave_suelta": 1}, "unknown keys"),
        ({"cortes_probados": _QUITAR}, "lacks keys"),
        ({"modo": "otro"}, "modo must be"),
        ({"corte": float("nan")}, "corte must be a finite number"),
        ({"corte_legible": ""}, "corte_legible"),
        ({"hueco_segundos": -1.0}, "hueco_segundos"),
        ({"fraccion_de_sujetos": 0}, r"fraccion_de_sujetos must be in \(0, 1\]"),
        ({"fraccion_de_sujetos_pedida": 1.5}, "fraccion_de_sujetos_pedida"),
        ({"cuantil": 1.0}, r"cuantil must be in \(0, 1\)"),
        ({"cuantil_pedido": True}, "cuantil_pedido"),
        ({"cortes_probados": 0}, "cortes_probados"),
        ({"filas_de_prueba": 0}, "filas_de_prueba"),
        ({"filas_descartadas": -1}, "filas_descartadas"),
        ({"filas_descartadas": 10}, "porcentaje_descartado .* does not follow from its rows"),
        ({"porcentaje_descartado": 0.0}, "does not follow from its rows"),
        ({"porcentaje_descartado": "24.8"}, "porcentaje_descartado"),
        ({"unidades_de_prueba": ["001", "002"]}, "unidades_de_prueba must be an integer"),
        ({"unidades_de_desarrollo": 0}, "unidades_de_desarrollo"),
        ({"tipo_de_pliegues": "por_el_tiempo"}, "tipo_de_pliegues must be one of"),
        ({"tipo_de_pliegues": "por_sujeto"}, "inicio_del_tramo must be null exactly when"),
        ({"inicio_del_tramo": None}, "inicio_del_tramo must be null exactly when"),
        ({"pliegues": 0}, "pliegues"),
    ])
    def test_cada_campo_mal_formado_se_rechaza_con_su_nombre(self, f2q, cambio, mensaje):
        _, _, sobre = f2q
        with pytest.raises(ReproduceManifestError, match=mensaje):
            manifiesto_de_estudio(reparto_del_estudio=_con(sobre, **cambio))

    @pytest.mark.parametrize("cambio,mensaje", [
        ({"condicional_a_la_ventana": False}, "condicional_a_la_ventana must be true"),
        ({"condicional_a_la_ventana": 1}, "condicional_a_la_ventana must be true"),
        ({"unidades_de_prueba": 11}, "must be the number of test units"),
        ({"ventana_de_prueba": [5.0, 1.0]}, "ventana_de_prueba"),
        ({"ventana_de_prueba": [1.0]}, "ventana_de_prueba"),
        ({"metodo": ""}, "intervalo.metodo"),
        ({"otra": 1}, "exactly"),
    ])
    def test_un_intervalo_mal_formado_se_rechaza(self, f2q, cambio, mensaje):
        filas, p, sobre = f2q
        malo = copy.deepcopy(sobre)
        malo["estricto"]["intervalo"].update(cambio)
        with pytest.raises(ReproduceManifestError, match=mensaje):
            _manifiesto(filas, p, malo)

    def test_el_intervalo_sin_una_clave_se_rechaza(self, f2q):
        filas, p, sobre = f2q
        malo = copy.deepcopy(sobre)
        del malo["estricto"]["intervalo"]["ventana_de_prueba"]
        with pytest.raises(ReproduceManifestError, match="exactly"):
            _manifiesto(filas, p, malo)

    def test_estricto_que_no_es_un_objeto(self, f2q):
        _, _, sobre = f2q
        with pytest.raises(ReproduceManifestError, match="estricto must be an object"):
            manifiesto_de_estudio(reparto_del_estudio={**sobre, "estricto": [1, 2]})

    def test_las_filas_de_desarrollo_o_prueba_que_dice_el_estudio_deben_ser_las_del_reparto(self, f2q):
        filas, p, sobre = f2q
        r = p.reparto
        with pytest.raises(ReproduceManifestError, match="filas_de_desarrollo .* disagrees"):
            manifiesto_de_estudio(filas_de_desarrollo=len(r.desarrollo) + 1, reparto_del_estudio=sobre)
        with pytest.raises(ReproduceManifestError, match="filas_de_prueba_reservada .* disagrees"):
            manifiesto_de_estudio(filas_de_prueba_reservada=len(r.prueba) - 1, reparto_del_estudio=sobre)
        # sin las cifras del estudio no hay nada que cruzar
        assert manifiesto_de_estudio(reparto_del_estudio=sobre)["study"]["split"]["strict"]["discarded_rows"] == 149

    def test_un_reparto_con_otro_diseno_o_sin_diseno_sigue_rechazandose(self):
        for malo in ({"diseno": "otro"}, {"diseno": None}, {}):
            with pytest.raises(ReproduceManifestError):
                manifiesto_de_estudio(reparto_del_estudio=malo)


# ---------------------------------------------------------------------------------------------------------------------------
# Los otros tres diseños salen BIT A BIT iguales que antes
# ---------------------------------------------------------------------------------------------------------------------------

#: sha256 (16 primeros hex) del manifiesto `study`, de la ficha TRIPOD (es y en, sin las líneas del entorno) y del mapa PROBAST
#: (es y en, sin el nombre del directorio temporal) de cada diseño, calculados con el árbol de ANTES de este corte (`ed6eae5`/`bb85fb6`,
#: `git archive` aparte). Una prueba que compara con lo que el código nuevo se devuelve a sí mismo no vería nada.
_HUELLAS_DE_ANTES = {
    "iid": {"study": "91f694daf0b8a611", "fichas": "5635ad39a7b94a03", "probast": "9ba26d33666e106e"},
    "groups": {"study": "e4ae70024fa15948", "fichas": "01cbbb30fdd47968", "probast": "63b0cf5ea17a7cd6"},
    "temporal": {"study": "4eb22a2c7497e172", "fichas": "3372ae28c39382d3", "probast": "2cd04efa7985e68a"},
}
_REPARTOS_DE_SIEMPRE = {
    "iid": {"diseno": "iid", "gap": None, "limites": []},
    "groups": {"diseno": "groups", "unit_id_field": "paciente", "limites": [
        {"clave": "k", "motivo": {"es": "uno", "en": "one"}, "medida": {"n": 1}}]},
    "temporal": {"diseno": "temporal", "time_column": "fecha", "gap": {"magnitud": 7, "unidad": "days"},
                 "gap_origen": "horizonte_declarado",
                 "cobertura_fuera_de_pliegue": {"filas_de_desarrollo": 100, "filas_sin_validacion": 34},
                 "limites": [{"clave": "k", "motivo": {"es": "uno", "en": "one"}, "medida": {"n": 1}}]},
}


def _huella(objeto) -> str:
    return hashlib.sha256(json.dumps(objeto, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


@pytest.mark.parametrize("diseno", ["iid", "groups", "temporal"])
def test_los_otros_tres_disenos_salen_bit_a_bit_como_antes(diseno):
    kw = dict(filas=300, filas_de_desarrollo=200, filas_de_prueba_reservada=100, filas_que_entrenaron_el_modelo=180,
              motor_del_modelo_entregado="sklearn.hgb", reparto_del_estudio=_REPARTOS_DE_SIEMPRE[diseno])
    estudio = manifiesto_de_estudio(**kw)["study"]
    with tempfile.TemporaryDirectory() as t:
        escribir_manifiesto_de_estudio(Path(t), **kw)
        fichas = {i: [l for l in ficha_tripod(Path(t), locale=i).splitlines() if "matrixai " not in l and "python " not in l]
                  for i in ("es", "en")}
        probast = {i: json.dumps({k: v for k, v in mapa_probast(Path(t), locale=i).items() if k != "paquete"},
                                 sort_keys=True, default=str) for i in ("es", "en")}
    assert {"study": _huella(estudio), "fichas": _huella(fichas), "probast": _huella(probast)} == _HUELLAS_DE_ANTES[diseno]


# ---------------------------------------------------------------------------------------------------------------------------
# El expediente y la ficha TRIPOD
# ---------------------------------------------------------------------------------------------------------------------------

def _paquete(filas, p, sobre, tmp_path, **kw) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    r = p.reparto
    escribir_manifiesto_de_estudio(tmp_path, filas=len(filas), filas_de_desarrollo=len(r.desarrollo),
                                   filas_de_prueba_reservada=len(r.prueba), filas_que_entrenaron_el_modelo=len(r.desarrollo) - 40,
                                   motor_del_modelo_entregado="sklearn.hgb", reparto_del_estudio=sobre, **kw)
    return tmp_path


def _linea(ficha: str, rotulo: str) -> str:
    return next(l for l in ficha.splitlines() if l.startswith(f"- **{rotulo}**"))


class TestLaFichaDiceElRepartoEstricto:

    def test_es_dice_el_diseno_el_corte_las_filas_sin_rol_con_su_porcentaje_y_el_tipo_de_pliegues(self, f2q, tmp_path):
        filas, p, sobre = f2q
        ficha = ficha_tripod(_paquete(filas, p, sobre, tmp_path), locale="es")
        reparto = _linea(ficha, REPARTO_POR_DISENO.rotulo["es"])
        assert "por sujeto y por el tiempo (groups_and_time)" in reparto
        assert "sujetos que el modelo no vio y con filas posteriores al corte" in reparto
        assert "(sujeto: «sujeto»; tiempo: «t»)" in reparto
        assert "corte en el tiempo: 2024-09-29" in reparto
        assert "149 (24.8 %) filas sin rol (ni de desarrollo ni de prueba)" in reparto
        assert "pliegues por sujeto y por el último tramo de tiempo del desarrollo" in reparto
        assert "hueco de 7 días" in reparto and "el horizonte declarado del problema" in reparto
        assert "sujetos de prueba: el 30 % de los sujetos, con el corte en el cuantil 0.7 del tiempo" in reparto
        assert ("intervalo condicional a esta ventana de tiempo (remuestreo por sujeto, con 10 sujetos de prueba)" in reparto)
        assert "no cuánto rendirá en otro periodo" in reparto
        # los límites del reparto, con su frase del núcleo, tal cual
        assert "límites declarados: " in reparto and "149" in reparto

    def test_en_lo_dice_en_ingles(self, f2q, tmp_path):
        filas, p, sobre = f2q
        ficha = ficha_tripod(_paquete(filas, p, sobre, tmp_path), locale="en")
        reparto = _linea(ficha, REPARTO_POR_DISENO.rotulo["en"])
        assert "by subject and by time (groups_and_time)" in reparto
        assert "subjects the model did not see and rows after the cut" in reparto
        assert "(subject: “sujeto”; time: “t”)" in reparto
        assert "cut in time: 2024-09-29" in reparto
        assert "149 (24.8%) rows without a role (neither development nor test)" in reparto
        assert "folds by subject and by the last time stretch of development" in reparto
        assert "gap of 7 days" in reparto and "the problem's declared horizon" in reparto
        assert "test subjects: 30% of the subjects, with the cut at the 0.7 quantile of time" in reparto
        assert "interval conditional on this time window (resampling by subject, with 10 test subjects)" in reparto
        assert "not how it will perform in another period" in reparto
        # ni una palabra de la línea en español
        for palabra in ("sujetos de prueba", "filas sin rol", "pliegues", "corte en el tiempo", "intervalo condicional"):
            assert palabra not in reparto

    def test_no_redacta_cifras_de_cobertura_del_intervalo(self, f2q, tmp_path):
        """La cobertura del intervalo la decide Roberto: el texto del intervalo no lleva ningún número salvo los sujetos de prueba."""
        filas, p, sobre = f2q
        for idioma in ("es", "en"):
            ficha = ficha_tripod(_paquete(filas, p, sobre, tmp_path / idioma), locale=idioma)
            reparto = _linea(ficha, REPARTO_POR_DISENO.rotulo[idioma])
            frase = reparto[reparto.index("intervalo condicional" if idioma == "es" else "interval conditional"):]
            frase = frase.split("; ")[0]
            assert re.findall(r"\d+(?:[.,]\d+)?", frase) == ["10"], frase
            for palabra in ("cobertura", "coverage", "95", "nominal"):
                assert palabra not in frase

    def test_f1_dice_pliegues_solo_por_sujeto_y_lo_pedido_y_lo_usado_del_corte(self, f1, tmp_path):
        filas, p, sobre = f1
        r = p.reparto
        es = _linea(ficha_tripod(_paquete(filas, p, sobre, tmp_path / "x"), locale="es"), REPARTO_POR_DISENO.rotulo["es"])
        assert "pliegues solo por sujeto, sin tramo de tiempo (el motivo, en los límites declarados)" in es
        assert "último tramo" not in es.split("límites declarados")[0]
        assert "se pidió el 30 % con el corte en el cuantil 0.7 y se usó el" in es
        assert f"{r.fraccion_de_sujetos * 100:g} %" in es and f"cuantil {r.cuantil:g}" in es
        en = _linea(ficha_tripod(_paquete(filas, p, sobre, tmp_path / "y"), locale="en"), REPARTO_POR_DISENO.rotulo["en"])
        assert "folds by subject only, with no time stretch (the reason is in the declared limits)" in en
        assert "30% with the cut at the 0.7 quantile was requested and" in en

    def test_las_filas_del_conjunto_ya_no_son_desarrollo_mas_prueba_y_la_ficha_lo_dice(self, f2q, tmp_path):
        filas, p, sobre = f2q
        for idioma in ("es", "en"):
            ficha = ficha_tripod(_paquete(filas, p, sobre, tmp_path / idioma), locale=idioma)
            total = _linea(ficha, FILAS_DEL_ESTUDIO_CON_FILAS_SIN_ROL.rotulo[idioma])
            assert ("desarrollo, prueba reservada y filas sin rol" if idioma == "es" else
                    "development, holdout test and rows without a role") in total
            assert FILAS_DEL_ESTUDIO.rotulo[idioma] not in ficha
            sin_rol = _linea(ficha, FILAS_SIN_ROL.rotulo[idioma])
            assert ("149 (24.8 % del conjunto)" if idioma == "es" else "149 (24.8% of the dataset)") in sin_rol
            assert "`reproduce.json#study.split.strict`" in sin_rol
            # y las cuatro cifras cuadran: conjunto = desarrollo + prueba + sin rol
            def cifra(rotulo):
                return int(re.search(r"\*\*: (\d+)", _linea(ficha, rotulo)).group(1))
            dev = cifra("Filas de desarrollo (validación cruzada y selección)" if idioma == "es"
                        else "Development rows (cross-validation and selection)")
            prueba = cifra("Filas de la prueba reservada (apartadas antes; ni eligen ni entrenan)" if idioma == "es"
                           else "Holdout test rows (set aside first; they neither choose nor train)")
            assert cifra(FILAS_DEL_ESTUDIO_CON_FILAS_SIN_ROL.rotulo[idioma]) == dev + prueba + cifra(FILAS_SIN_ROL.rotulo[idioma]) == 600

    def test_es_y_en_no_se_intercambian(self, f2q, tmp_path):
        filas, p, sobre = f2q
        es = ficha_tripod(_paquete(filas, p, sobre, tmp_path / "es"), locale="es")
        en = ficha_tripod(_paquete(filas, p, sobre, tmp_path / "en"), locale="en")
        for rotulo in (REPARTO_POR_DISENO, FILAS_SIN_ROL, FILAS_DEL_ESTUDIO_CON_FILAS_SIN_ROL):
            assert _linea(es, rotulo.rotulo["es"]) != _linea(en, rotulo.rotulo["en"])
            assert not any(l.startswith(f"- **{rotulo.rotulo['es']}**") for l in en.splitlines())
            assert not any(l.startswith(f"- **{rotulo.rotulo['en']}**") for l in es.splitlines())

    def test_un_estudio_sin_reparto_estricto_no_trae_la_linea_de_filas_sin_rol(self, tmp_path):
        escribir_manifiesto_de_estudio(tmp_path, filas=300, filas_de_desarrollo=200, filas_de_prueba_reservada=100,
                                       motor_del_modelo_entregado="sklearn.hgb",
                                       reparto_del_estudio={"diseno": "groups", "unit_id_field": "p"})
        expediente = ExpedienteClinico.desde_paquete(tmp_path)
        assert expediente.reparto_estricto is None and expediente.filas_del_estudio is FILAS_DEL_ESTUDIO
        assert FILAS_SIN_ROL not in expediente.inventario()
        assert "filas sin rol" not in ficha_tripod(tmp_path, locale="es")

    def test_el_expediente_lo_lee_del_manifiesto_y_el_inventario_lleva_sus_lineas(self, f2q, tmp_path):
        filas, p, sobre = f2q
        expediente = ExpedienteClinico.desde_paquete(_paquete(filas, p, sobre, tmp_path))
        assert expediente.reparto_estricto["discarded_rows"] == 149
        assert expediente.filas_del_estudio is FILAS_DEL_ESTUDIO_CON_FILAS_SIN_ROL
        inventario = expediente.inventario()
        assert FILAS_SIN_ROL in inventario and REPARTO_POR_DISENO in inventario
        assert FILAS_DEL_ESTUDIO not in inventario and FILAS_DEL_ESTUDIO_CON_FILAS_SIN_ROL in inventario
        campo = expediente.campo(FILAS_SIN_ROL.ruta)
        assert campo.hay_dato and campo.valor["discarded_percentage"] == 24.8

    def test_probast_enumera_las_filas_sin_rol_como_dato_presente(self, f2q, tmp_path):
        filas, p, sobre = f2q
        mapa = mapa_probast(_paquete(filas, p, sobre, tmp_path), locale="es")
        items = [i for dominio in mapa["por_dominio"].values() for i in dominio]
        sin_rol = next(i for i in items if i["clave"] == "filas_sin_rol")
        assert "149" in str(sin_rol) and "24.8" in str(sin_rol)

    def test_el_texto_con_un_bloque_incompleto_no_inventa(self):
        """Un bloque `strict` sin el corte, o un `groups_and_time` sin `strict`, solo dicen lo que tienen."""
        def t(bloque, idioma="es"):
            return texto_del_valor(REPARTO_POR_DISENO, Campo("reproduce.json#study.split", "medido", valor=bloque), idioma)
        assert t({"design": "groups_and_time"}) == ("por sujeto y por el tiempo (groups_and_time): el examen se hace con sujetos "
                                                    "que el modelo no vio y con filas posteriores al corte")
        assert t({"design": "groups_and_time", "strict": {}}) == t({"design": "groups_and_time"})
        solo_filas = t({"design": "groups_and_time", "strict": {"discarded_rows": 3, "discarded_percentage": 2.5}}, "en")
        assert solo_filas.endswith("; 3 (2.5%) rows without a role (neither development nor test)")
        assert "cut" not in solo_filas.split("groups_and_time")[1].split("rows after the cut")[1]
