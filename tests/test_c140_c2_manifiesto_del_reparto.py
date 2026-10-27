# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""140-C2 (núcleo) — lo que el núcleo ofrece al backend para el reparto por sujeto y por tiempo:

* `PropuestaDeParticion.tiempo_por_id` (solo `temporal`): el instante que el generador leyó de cada fila.
* `manifiesto_de_estudio(reparto_del_estudio=...)` → `study.split`, validado, sin rellenar lo que el estudio no dijo.
* El texto con que la ficha lo escribe, en los dos idiomas.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from matrixai.export.expediente_clinico import REPARTO_POR_DISENO, Campo, texto_del_valor
from matrixai.export.manifiesto_de_estudio import DISENOS_DEL_REPARTO, manifiesto_de_estudio
from matrixai.export.reproduce import ReproduceManifestError
from matrixai.training.particion_por_diseno import proponer_particion


def _filas(n=60):
    inicio = datetime(2024, 1, 1)
    return [{"id": f"F{i}", "fecha": (inicio + timedelta(days=i)).strftime("%d/%m/%Y"), "paciente": f"P{i % 6}",
             "y": "si" if i % 2 else "no"} for i in range(n)]


class TestTiempoPorId:

    def test_temporal_lo_lleva_con_los_segundos_de_cada_fila_de_desarrollo_y_prueba(self):
        filas = _filas()
        p = proponer_particion(filas, plan_id="x", observation_id_field="id", split_type="temporal", seed=0, folds=2,
                               time_column="fecha", gap={"magnitud": 2, "unidad": "days"}, objetivo="y")
        assert p.es_viable
        esperado = {f["id"]: (datetime.strptime(f["fecha"], "%d/%m/%Y") - datetime(1970, 1, 1)).total_seconds() for f in filas}
        roles = {*p.plan.observaciones_del_rol("development"), *p.plan.observaciones_del_rol("test")}
        assert p.tiempo_por_id == {i: esperado[i] for i in roles}

    def test_los_demas_disenos_no_lo_llevan(self):
        filas = _filas()
        for diseno, extra in (("iid", {}), ("groups", {"unit_id_field": "paciente"})):
            p = proponer_particion(filas, plan_id="x", observation_id_field="id", split_type=diseno, seed=0, folds=2,
                                   objetivo="y", **extra)
            assert p.es_viable and p.tiempo_por_id is None

    def test_no_cuenta_en_la_igualdad_de_dos_propuestas(self):
        filas = _filas()
        args = dict(plan_id="x", observation_id_field="id", split_type="temporal", seed=0, folds=2, time_column="fecha",
                    objetivo="y")
        a, b = proponer_particion(filas, **args), proponer_particion(filas, **args)
        assert a == b and "tiempo_por_id" not in repr(a)


class TestManifiestoDelReparto:

    def test_temporal_completo(self):
        m = manifiesto_de_estudio(reparto_del_estudio={
            "diseno": "temporal", "unit_id_field": None, "time_column": "fecha", "gap": {"magnitud": 7, "unidad": "days"},
            "gap_origen": "horizonte_declarado",
            "cobertura_fuera_de_pliegue": {"filas_de_desarrollo": 100, "filas_sin_validacion": 34},
            "limites": [{"clave": "k", "motivo": {"es": "uno", "en": "one"}, "medida": {"n": 1}}]})
        assert m["study"]["split"] == {
            "design": "temporal", "time_column": "fecha", "gap": {"magnitude": 7, "unit": "days"},
            "gap_origin": "declared_horizon",
            "out_of_fold_coverage": {"development_rows": 100, "rows_without_validation": 34},
            "limits": [{"key": "k", "reason": {"es": "uno", "en": "one"}, "measure": {"n": 1}}]}

    def test_iid_es_solo_el_diseno_y_sin_el_dato_no_hay_bloque(self):
        assert manifiesto_de_estudio(reparto_del_estudio={"diseno": "iid", "gap": None, "limites": []})["study"]["split"] == {
            "design": "iid"}
        assert "split" not in manifiesto_de_estudio()["study"]

    def test_cambiar_el_reparto_cambia_el_digest(self):
        a = manifiesto_de_estudio(reparto_del_estudio={"diseno": "iid"})
        b = manifiesto_de_estudio(reparto_del_estudio={"diseno": "groups", "unit_id_field": "paciente"})
        assert a["manifest_sha256"] != b["manifest_sha256"]

    @pytest.mark.parametrize("malo", [
        {"diseno": "groups_and_time"}, {"diseno": "otro"}, {"diseno": "temporal", "gap": {"magnitud": 0, "unidad": "days"}},
        {"diseno": "temporal", "gap": {"magnitud": True, "unidad": "days"}},
        {"diseno": "temporal", "gap": {"magnitud": 3, "unidad": "lustros"}},
        {"diseno": "temporal", "gap": {"magnitud": 3, "unidad": "days"}, "gap_origen": "por_costumbre"},
        {"diseno": "temporal", "cobertura_fuera_de_pliegue": {"filas_de_desarrollo": 5, "filas_sin_validacion": 9}},
        {"diseno": "temporal", "limites": [{"clave": "x", "motivo": {"es": "solo es"}}]},
        {"diseno": "temporal", "limites": "no es una lista"}])
    def test_un_reparto_mal_formado_se_rechaza(self, malo):
        with pytest.raises(ReproduceManifestError):
            manifiesto_de_estudio(reparto_del_estudio=malo)

    def test_el_vocabulario_es_el_que_mide_el_studio(self):
        assert DISENOS_DEL_REPARTO == ("iid", "groups", "temporal")


class TestElTextoDelReparto:

    def _t(self, bloque, idioma):
        return texto_del_valor(REPARTO_POR_DISENO, Campo("reproduce.json#study.split", "medido", valor=bloque), idioma)

    def test_los_tres_disenos_en_los_dos_idiomas(self):
        assert self._t({"design": "iid"}, "es") == "al azar por filas (iid)"
        assert self._t({"design": "iid"}, "en") == "at random by rows (iid)"
        assert "«paciente»" in self._t({"design": "groups", "unit_column": "paciente"}, "es")
        assert "“paciente”" in self._t({"design": "groups", "unit_column": "paciente"}, "en")
        t = {"design": "temporal", "time_column": "fecha", "gap": {"magnitude": 30, "unit": "days"},
             "gap_origin": "requested", "out_of_fold_coverage": {"development_rows": 10, "rows_without_validation": 4}}
        assert "hueco de 30 días" in self._t(t, "es") and "pedido por la persona" in self._t(t, "es")
        assert "gap of 30 days" in self._t(t, "en") and "requested by the person" in self._t(t, "en")
        assert "de 10 filas de desarrollo, 4 no tienen" in self._t(t, "es")

    def test_un_dato_ausente_no_se_escribe_y_un_diseno_desconocido_se_ve_tal_cual(self):
        assert self._t({"design": "temporal"}, "es") == "hacia delante en el tiempo (temporal)"
        assert self._t({"design": "raro"}, "es") == "raro"
