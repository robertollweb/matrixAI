"""86-C3 — EL ML-BOM DEL PAQUETE, EN CYCLONEDX.

Un ML-BOM es lo que ya aparece en pliegos de compra, y lo que frena su adopción
es que **casi nadie lo genera sin trabajo manual**. `reproduce.json` ya tiene
casi todos los campos, así que esto es un traductor y no una investigación.

Lo que se fija aquí:

1. **Valida contra el esquema OFICIAL** (`tests/data/cyclonedx-bom-1.6.schema.json`,
   descargado del repositorio de la especificación). Sin esto, «emitimos
   CycloneDX» sería una afirmación sobre nuestro propio JSON.
2. **Es determinista**: mismo paquete, mismo número de serie. Un BOM que cambia
   cada vez que se genera no se puede comparar, ni firmar, ni meter dentro de
   un paquete que promete reproducirse.
3. **Lo que no se sabe NO se rellena** — y se puede enumerar.
4. **Los avisos van donde los busca quien lee un BOM**: datos sintéticos y
   paquete no reproducible son dos cosas distintas y las dos se dicen.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.export.bom import SPEC_VERSION, BomNoDisponible, lo_que_falta, ml_bom

_ESQUEMA = Path(__file__).parent / "data" / "cyclonedx-bom-1.6.schema.json"

_MANIFIESTO = {
    "schema_version": "1.0",
    "reproducible": True,
    "manifest_sha256": "d" * 64,
    "generation": {
        "mode": "coherent",
        "seeds": {"dataset": 20260825, "split": 42, "init": 42},
        "epochs_declared": 60, "epochs_effective": 60, "epochs_ran": 60,
        "backend": "stdlib", "device": "cpu", "warm_start": False,
        "excluded_identifiers": ["id_paciente"],
    },
    "artifacts": {
        "model": {"sha256": "a" * 64},
        "training": {"sha256": "b" * 64},
        "recipe": {"sha256": "e" * 64},
        "dataset": {"sha256": "c" * 64, "rows": 400},
    },
    "environment": {"matrixai_version": "1.6.0"},
    "metrics": [{"name": "accuracy", "value": 0.96875, "split": "validation",
                 "direction": "higher_is_better", "dataset_sha256": "c" * 64}],
}


def _paquete(tmp: Path, **cambios) -> Path:
    m = json.loads(json.dumps(_MANIFIESTO))
    m.update(cambios)
    (tmp / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
    return tmp


class ValidaContraElEsquemaOficialTest(unittest.TestCase):
    @unittest.skipUnless(_ESQUEMA.is_file(), "falta el esquema oficial de CycloneDX")
    def test_cero_errores_de_esquema(self):
        try:
            from jsonschema import Draft7Validator
        except ImportError:  # pragma: no cover
            self.skipTest("jsonschema no está instalado")
        esquema = json.loads(_ESQUEMA.read_text(encoding="utf-8"))
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d)))
        errores = list(Draft7Validator(esquema).iter_errors(bom))
        self.assertEqual(errores, [], [f"{list(e.path)}: {e.message}" for e in errores[:5]])

    def test_declara_la_version_que_dice_emitir(self):
        with TemporaryDirectory() as d:
            self.assertEqual(ml_bom(_paquete(Path(d)))["specVersion"], SPEC_VERSION)


class EsDeterministaTest(unittest.TestCase):
    def test_el_mismo_paquete_da_el_MISMO_numero_de_serie(self):
        with TemporaryDirectory() as d1, TemporaryDirectory() as d2:
            a = ml_bom(_paquete(Path(d1)))
            b = ml_bom(_paquete(Path(d2)))
        self.assertEqual(a["serialNumber"], b["serialNumber"])

    def test_y_un_paquete_DISTINTO_da_otro(self):
        with TemporaryDirectory() as d1, TemporaryDirectory() as d2:
            a = ml_bom(_paquete(Path(d1)))
            b = ml_bom(_paquete(Path(d2), manifest_sha256="f" * 64))
        self.assertNotEqual(a["serialNumber"], b["serialNumber"])


class NoRellenaLoQueNoSabeTest(unittest.TestCase):
    def test_un_ausente_no_se_escribe_como_texto(self):
        """`None` escrito como `"None"` es la peor clase de dato: parece uno."""
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("mode")
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            bom = ml_bom(Path(d))
        volcado = json.dumps(bom)
        self.assertNotIn('"None"', volcado)
        nombres = {p["name"] for p in bom["metadata"]["component"]["properties"]}
        self.assertNotIn("matrixai:generation.mode", nombres)

    def test_y_lo_que_falta_se_puede_ENUMERAR(self):
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["generation"].pop("mode")
            m["metrics"] = []
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            faltan = lo_que_falta(ml_bom(Path(d)))
        self.assertTrue(any("generation.mode" in f for f in faltan), faltan)
        self.assertTrue(any("performance metrics" in f for f in faltan), faltan)

    def test_sin_manifiesto_no_hay_BOM_vacio(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(BomNoDisponible):
                ml_bom(Path(d))


class LosAvisosVanDondeSeBuscanTest(unittest.TestCase):
    def test_los_datos_sinteticos_se_declaran_en_las_consideraciones(self):
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d)))
        limites = bom["metadata"]["component"]["modelCard"]["considerations"]["technicalLimitations"]
        self.assertTrue(any("SYNTHETIC" in x for x in limites), limites)

    def test_un_paquete_NO_reproducible_lo_dice_con_su_motivo(self):
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d), reproducible=False,
                                  reproducible_reason="no run capture travels"))
        limites = bom["metadata"]["component"]["modelCard"]["considerations"]["technicalLimitations"]
        self.assertTrue(any("no run capture travels" in x for x in limites), limites)

    def test_las_metricas_viajan_con_su_dataset(self):
        """Una métrica sin decir sobre qué se midió no es un resultado, es un
        número."""
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d)))
        datos = bom["components"][1]["data"][0]["contents"]["properties"]
        self.assertTrue(any(p["name"] == "matrixai:dataset.sha256" for p in datos))
        metricas = bom["metadata"]["component"]["modelCard"]["quantitativeAnalysis"]["performanceMetrics"]
        self.assertEqual(metricas[0]["type"], "accuracy")


if __name__ == "__main__":
    unittest.main()


# ── LO QUE ENCONTRÓ LA AUDITORÍA EXTERNA (2026-08-25, hallazgo 4) ──────────

class SinSemanticaInventadaTest(unittest.TestCase):
    """«Cero errores de esquema no equivale aquí a un BOM fiel».

    Esto fabricaba un `confidenceInterval` de amplitud CERO —`lowerBound ==
    upperBound == valor`— que **afirma que la métrica es exacta**, y eso no lo
    ha medido nadie. Y el contexto que sí se había recogido (partición, dataset,
    dirección) se perdía: la variable se construía y no se escribía en ninguna
    parte.
    """

    def test_NO_se_inventa_un_intervalo_de_confianza(self):
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d)))
        metricas = bom["metadata"]["component"]["modelCard"]["quantitativeAnalysis"]["performanceMetrics"]
        for m in metricas:
            self.assertNotIn("confidenceInterval", m, m)

    def test_la_particion_va_en_SLICE_que_es_para_eso(self):
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d)))
        metricas = bom["metadata"]["component"]["modelCard"]["quantitativeAnalysis"]["performanceMetrics"]
        self.assertEqual(metricas[0]["slice"], "validation")

    def test_el_digest_y_la_direccion_NO_desaparecen(self):
        """CycloneDX no los admite dentro de la métrica (`additionalProperties:
        false`), así que viajan como propiedad del componente — no se pierden ni
        se disfrazan de otra cosa."""
        with TemporaryDirectory() as d:
            bom = ml_bom(_paquete(Path(d)))
        nombres = {p["name"] for p in bom["metadata"]["component"]["properties"]}
        self.assertIn("matrixai:metric.accuracy.dataset_sha256", nombres)
        self.assertIn("matrixai:metric.accuracy.direction", nombres)

    def test_lo_que_falta_incluye_la_LICENCIA(self):
        with TemporaryDirectory() as d:
            faltan = lo_que_falta(ml_bom(_paquete(Path(d))))
        self.assertTrue(any("licenses" in f for f in faltan), faltan)

    def test_y_la_RECETA_cuando_no_la_hay(self):
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["artifacts"]["recipe"] = None      # el caso de datos reales
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            bom = ml_bom(Path(d))
            faltan = lo_que_falta(bom)
        self.assertTrue(any("recipe" in f for f in faltan), faltan)

    def test_una_receta_a_None_no_revienta(self):
        """Salió validando los TRES casos de la galería y no los dos que tenían
        receta: la clave existe con valor `None`, así que el valor por defecto
        de `.get` nunca entraba."""
        with TemporaryDirectory() as d:
            m = json.loads(json.dumps(_MANIFIESTO))
            m["artifacts"]["recipe"] = None
            (Path(d) / "reproduce.json").write_text(json.dumps(m), encoding="utf-8")
            ml_bom(Path(d))     # no levanta
