"""El registry de ejemplo que se PUBLICA tiene que verificar — medido.

Encontrado el 2026-08-20 al enchufar un ejecutor real al motor del 81:
`matrixai registry verify feature_extractor@v1` sobre el ejemplo que
viaja en el repositorio daba **FAIL, rc=1**, y así desde la 1.0.0. Es la
primera cosa que la documentación del caso invita a comprobar.

La causa no era una manipulación: era el paso 6 del propio
`run_case.py`, que **restauraba re-serializando**
(`write_text(json.dumps(original))`) los `params.json` que había tocado a
propósito para enseñar la detección. Los valores volvían; los BYTES no
—`indent=2` contra el `dumps` por defecto—, y el hash de integridad cubre
el fichero, no el objeto. La demostración de que se detecta una
manipulación dejaba la entrada rota de verdad.

Es la misma regla que el recibo del 81 ya defiende (P23-R-0016: el
payload se verifica sobre los bytes que se firmaron). Aquí costó un
ejemplo público.
"""

import json
import unittest
from pathlib import Path

from matrixai.registry.model_registry import ModelRegistry

EJEMPLO = Path(__file__).resolve().parent.parent / "examples" / "text-routing"


class ElEjemploQueSePublicaVerificaTest(unittest.TestCase):
    def test_las_DOS_entradas_pasan_su_propia_verificacion(self):
        registro = ModelRegistry(EJEMPLO / "registry")
        for nombre in ("feature_extractor", "route_classifier"):
            with self.subTest(entrada=nombre):
                self.assertTrue(registro.verify(nombre, "v1"))

    def test_reserializar_los_params_ROMPE_la_integridad(self):
        """La comprobación que le faltaba a la restauración del paso 6:
        sin esto, «restaurado» y «roto» se ven igual."""
        from matrixai.registry.entry_hash import sha256_bytes

        entrada = EJEMPLO / "registry" / "entries" / "feature_extractor" / "v1"
        crudo = (entrada / "params.json").read_bytes()
        declarado = json.loads((entrada / "manifest.json").read_text())["params_content_hash"]

        self.assertEqual(sha256_bytes(crudo), declarado)
        # Mismos valores, otros bytes: y la integridad ya no cuadra.
        reserializado = json.dumps(json.loads(crudo)).encode("utf-8")
        self.assertEqual(json.loads(reserializado), json.loads(crudo))
        self.assertNotEqual(sha256_bytes(reserializado), declarado)

    def test_el_ejemplo_restaura_los_BYTES_no_el_objeto(self):
        """Que el arreglo siga puesto: el paso 6 escribe lo que leyó."""
        codigo = (EJEMPLO / "run_case.py").read_text(encoding="utf-8")
        self.assertIn("te_params.write_bytes(original_bytes)", codigo)
        self.assertNotIn("te_params.write_text(json.dumps(original))", codigo)


if __name__ == "__main__":
    unittest.main()
