"""LA FICHA TRIPOD+AI (85-C6) — la cuña que aprobó Roberto el 2026-08-24.

TRIPOD+AI (BMJ, 2024) es la lista de 27 ítems que piden las revistas para
publicar un modelo de predicción clínica, y lo medido en los estudios
bibliométricos posteriores es que **la mayoría de los trabajos no la cumple**:
se rellena a mano, al final y de memoria.

Esto no es una función nueva del motor: es **el dato que ya se captura**,
escrito en el formato que un público concreto está obligado a entregar. Lo que
se fija aquí es lo único que la hace valer:

1. **No se inventa una casilla.** Lo que no está, se dice que no está **y por
   qué**.
2. **Un ausente no se imprime como dato.** «60 / None / None» pone tres números
   en fila donde solo hay uno.
3. **El aviso de datos sintéticos va ARRIBA**, no en una nota al pie.
4. **No promete nada regulatorio**, y lo dice en la propia ficha.
5. **Y habla un solo idioma**: los motivos de los huecos también.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.export.tripod import FichaNoDisponible, ficha_tripod

_MANIFIESTO = {
    "schema_version": "1.0",
    "reproducible": True,
    "generation": {
        "mode": "coherent",
        "seeds": {"dataset": 20260824, "split": 42, "init": 7},
        "epochs_declared": 60, "epochs_effective": 60, "epochs_ran": 60,
        "backend": "stdlib", "device": "cpu",
        "field_types": {"edad": "scalar", "reingreso_previo": "scalar"},
        "excluded_identifiers": ["id_paciente"],
    },
    "artifacts": {
        "model": {"sha256": "a" * 64},
        "training": {"sha256": "b" * 64},
        "dataset": {"sha256": "c" * 64, "rows": 400},
    },
    "environment": {"matrixai_version": "1.6.0", "python": {"version": "3.12.3"}},
    "metrics": [{"name": "accuracy", "value": 0.93, "dataset_sha256": "c" * 64}],
}


def _paquete(tmp: Path, *, receta: str | None = "alto: edad > 0.75\nDEFAULT: bajo",
             **cambios) -> Path:
    m = json.loads(json.dumps(_MANIFIESTO))
    m.update(cambios)
    (tmp / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
    if receta is not None:
        (tmp / "data_recipe.txt").write_text(receta, encoding="utf-8")
    return tmp


class LaFichaDiceLoQueHayTest(unittest.TestCase):
    def test_las_cifras_del_run_salen_por_su_nombre(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d)), locale="es")
        self.assertIn("400", ficha)                    # filas
        self.assertIn("c" * 64, ficha)                 # huella del dataset
        self.assertIn("60 / 60 / 60", ficha)           # las tres épocas
        self.assertIn("20260824", ficha)               # semillas
        self.assertIn("id_paciente", ficha)            # columnas excluidas
        self.assertIn("0.93", ficha)                   # la métrica
        # Y SOBRE QUÉ se midió, que es lo que casi nadie publica.
        self.assertIn("cccccccccccc", ficha)

    def test_la_receta_viaja_entera_dentro_de_la_ficha(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d)), locale="es")
        self.assertIn("alto: edad > 0.75", ficha)
        self.assertIn("DEFAULT: bajo", ficha)


class LaFichaNoInventaTest(unittest.TestCase):
    def test_un_hueco_se_dice_Y_POR_QUE(self):
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("mode")
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            ficha = ficha_tripod(Path(d), locale="es")
        self.assertIn("no disponible", ficha)
        self.assertIn("no declara con qué modo se generó", ficha)

    def test_NUNCA_se_imprime_un_None_crudo(self):
        """«60 / None / None» pone tres números en fila donde solo hay uno."""
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"]["epochs_effective"] = None
            m["generation"]["epochs_ran"] = None
            m["environment"] = {}
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            ficha = ficha_tripod(Path(d), locale="es")
        self.assertNotIn("None", ficha)
        self.assertIn("60 / _no disponible_ / _no disponible_", ficha)

    def test_sin_metricas_no_se_pinta_una_tabla_vacia(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d), metrics=[]), locale="es")
        self.assertIn("no publica ninguna métrica", ficha)

    def test_lo_que_la_ficha_NO_puede_rellenar_va_ENUMERADO(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d)), locale="es")
        for tema in ("Datos ausentes", "Calibración", "Equidad por subgrupos",
                     "Financiación", "Conflictos de interés"):
            self.assertIn(tema, ficha)

    def test_un_paquete_sin_manifiesto_no_produce_una_ficha_vacia(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(FichaNoDisponible):
                ficha_tripod(Path(d))


class LaFichaNoPrometeTest(unittest.TestCase):
    def test_dice_que_no_es_un_producto_sanitario(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d)), locale="es")
        self.assertIn("no es un producto sanitario", ficha)

    def test_el_aviso_de_datos_SINTETICOS_va_arriba_del_todo(self):
        """Enterarse en una nota al pie de que los datos eran sintéticos es
        enterarse tarde."""
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d)), locale="es")
        posicion = ficha.index("SINTÉTICOS")
        self.assertLess(posicion, ficha.index("## Datos"))

    def test_sin_receta_no_se_declara_sintetico_lo_que_no_lo_es(self):
        with TemporaryDirectory() as d:
            ficha = ficha_tripod(_paquete(Path(d), receta=None), locale="es")
        self.assertNotIn("SINTÉTICOS", ficha)


class LaFichaHablaUnSoloIdiomaTest(unittest.TestCase):
    def test_en_espanol_no_se_cuela_el_motivo_en_ingles(self):
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("mode")
            m["metrics"] = []
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            ficha = ficha_tripod(Path(d), locale="es")
        for ingles in ("the package", "the run did not", "publishes no"):
            self.assertNotIn(ingles, ficha)

    def test_y_en_ingles_no_se_cuela_el_castellano(self):
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("mode")
            m["metrics"] = []
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            ficha = ficha_tripod(Path(d), locale="en")
        for castellano in (" el ", " la ", " que ", " con ", "disponible"):
            self.assertNotIn(castellano, ficha)


if __name__ == "__main__":
    unittest.main()


class LosPredictoresSalenDelPAQUETESiHaceFaltaTest(unittest.TestCase):
    """La primera ficha de la galería decía «Predictores: no disponible» sobre
    un `.mxai` que declara los cuatro con su rango. El dato estaba **dentro del
    paquete** y no llegaba porque la captura del run —que es la que manda para
    `generation`— todavía no los registra.

    Citar el modelo que viaja al lado no es inventar: es leer el fichero que el
    propio paquete lleva.
    """

    _MXAI = (
        "PROJECT P\n\nVECTOR Input[2]\n  edad: Scalar[18, 100]\n"
        "  reingreso: Scalar\nEND\n\n"
        "NETWORK N\n  INPUT Input\n  LAYER Dense units=2 activation=softmax\n"
        "  OUTPUT predicted_class: ProbabilityMap[alto, bajo]\nEND\n\n"
        "GRAPH\n  Input -> N\nEND\n"
    )

    def test_se_leen_del_modelo_con_su_rango(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("field_types")
            (tmp / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            (tmp / "model.mxai").write_text(self._MXAI, encoding="utf-8")
            ficha = ficha_tripod(tmp, locale="es")
        self.assertIn("edad (18–100)", ficha)
        # Un campo sin rango sale por su nombre, sin inventarle uno.
        self.assertIn("reingreso", ficha)
        self.assertNotIn("reingreso (", ficha)

    def test_manda_lo_que_diga_el_MANIFIESTO_cuando_lo_dice(self):
        """La captura del run es la fuente autorizada; el modelo es el respaldo."""
        with TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "reproduce.json").write_text(json.dumps(_MANIFIESTO), encoding="utf-8")
            (tmp / "model.mxai").write_text(self._MXAI, encoding="utf-8")
            ficha = ficha_tripod(tmp, locale="es")
        self.assertIn("edad, reingreso_previo", ficha)   # las del manifiesto
        self.assertNotIn("edad (18–100)", ficha)

    def test_sin_modelo_en_el_paquete_se_dice_que_no_hay(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("field_types")
            (tmp / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            ficha = ficha_tripod(tmp, locale="es")
        self.assertIn("el run no registró los tipos", ficha)

    def test_un_mxai_ilegible_no_produce_una_lista_a_medias(self):
        with TemporaryDirectory() as d:
            tmp = Path(d)
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("field_types")
            (tmp / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            (tmp / "model.mxai").write_text("esto no es un modelo", encoding="utf-8")
            ficha = ficha_tripod(tmp, locale="es")
        self.assertIn("el run no registró los tipos", ficha)
