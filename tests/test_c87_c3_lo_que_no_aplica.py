"""87-C3 — LO QUE NO SE PUEDE HACER NO SE DICE COMO «NO SE PIDIÓ».

Medido el 2026-08-25 sobre un paquete sin `.mxtrain`:

    sin --retrain:  training  NOT_RUN  — «retraining was not requested»
    con --retrain:  training  INCOMPARABLE — «the dataset could not be regenerated»

Las dos frases apuntan al sitio equivocado. La primera culpa a quien verifica
—pedirlo no habría servido de nada— y la segunda al dataset, que es el SEGUNDO
obstáculo: sin contrato de entrenamiento no hay con qué entrenar aunque los
datos estuvieran.

Y la diferencia entre los dos estados no es cosmética: `NOT_RUN` es una
**elección de quien verifica** y no cuenta como etapa sin comprobar;
`INCOMPARABLE` es un **límite del paquete** y sí cuenta. Con la frase mal
puesta, un paquete que no se puede reentrenar salía como si estuviera
comprobado del todo.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.export.verify import verify_package


def _paquete(tmp: Path, *, con_contrato: bool) -> Path:
    # LOS DIGESTS SON LOS DE LOS FICHEROS DE VERDAD. Con unos inventados el
    # manifiesto falla por integridad y todas las etapas salen INCOMPARABLE por
    # ESO — el fixture mediría otra cosa. Es la tercera vez que la misma lección
    # aparece hoy: un fixture describe algo que existe.
    from matrixai.export.reproduce import sha256_file

    (tmp / "model.mxai").write_text("PROJECT P\n", encoding="utf-8")
    artefactos = {"model": {"path": "model.mxai", "media_type": "text/mxai",
                            "sha256": sha256_file(tmp / "model.mxai")}}
    if con_contrato:
        (tmp / "model.mxtrain").write_text("MODEL model.mxai\n", encoding="utf-8")
        artefactos["training"] = {"path": "model.mxtrain", "media_type": "text/mxtrain",
                                  "sha256": sha256_file(tmp / "model.mxtrain")}
    manifiesto = {
        "schema_version": "1.0",
        "artifacts": artefactos,
        # `files` es un MAPA `ruta -> sha256` y cubre todo lo que el paquete
        # lleva: con la lista vacía el manifiesto sale INCOMPARABLE porque hay
        # ficheros sin cubrir, y otra vez se mediría otra cosa.
        "files": {f.name: sha256_file(f) for f in sorted(tmp.iterdir())
                  if f.is_file() and f.name != "reproduce.json"},
        "reproducible": False,
        "metrics": [],
    }
    # EL DIGEST SE CALCULA, no se inventa: con uno falso el manifiesto FALLA y
    # todas las etapas salen INCOMPARABLE por integridad — o sea que el fixture
    # mediría otra cosa. Lo enseñó ponerse rojo.
    manifiesto["files_covered"] = len(manifiesto["files"])
    from matrixai.export.reproduce import manifest_digest
    manifiesto["manifest_sha256"] = manifest_digest(manifiesto)
    (tmp / "reproduce.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return tmp


def _training(tmp: Path, **kw) -> dict:
    return verify_package(tmp, **kw)["stages"]["training"]


class SinContratoNoSePuedeReentrenarYSeDICETest(unittest.TestCase):
    def test_lo_dice_aunque_NO_se_haya_pedido(self):
        with TemporaryDirectory() as d:
            etapa = _training(_paquete(Path(d), con_contrato=False))
        self.assertEqual(etapa["status"], "INCOMPARABLE")
        self.assertIn(".mxtrain", etapa["reason"])

    def test_y_lo_mismo_cuando_SÍ_se_pide(self):
        """Antes aquí se culpaba al dataset, que es el segundo obstáculo."""
        with TemporaryDirectory() as d:
            etapa = _training(_paquete(Path(d), con_contrato=False), run_training=True)
        self.assertEqual(etapa["status"], "INCOMPARABLE")
        self.assertIn(".mxtrain", etapa["reason"])

    def test_NUNCA_sale_como_PASS(self):
        for pedido in (False, True):
            with TemporaryDirectory() as d:
                etapa = _training(_paquete(Path(d), con_contrato=False), run_training=pedido)
            self.assertNotEqual(etapa["status"], "PASS")

    def test_y_CUENTA_como_etapa_sin_comprobar(self):
        """Es la consecuencia observable: con `NOT_RUN` el paquete salía como
        si estuviera comprobado del todo."""
        with TemporaryDirectory() as d:
            informe = verify_package(_paquete(Path(d), con_contrato=False))
        self.assertIn("training", informe["unchecked_stages"])
        self.assertFalse(informe["fully_checked"])


class ConContratoSeConservaLaDISTINCIONTest(unittest.TestCase):
    def test_no_pedirlo_sigue_siendo_una_ELECCION_de_quien_verifica(self):
        with TemporaryDirectory() as d:
            etapa = _training(_paquete(Path(d), con_contrato=True))
        self.assertEqual(etapa["status"], "NOT_RUN")
        self.assertIn("nadie pidió reentrenar", etapa["reason"])

    def test_y_esa_eleccion_NO_cuenta_como_etapa_sin_comprobar(self):
        with TemporaryDirectory() as d:
            informe = verify_package(_paquete(Path(d), con_contrato=True))
        self.assertNotIn("training", informe["unchecked_stages"])


if __name__ == "__main__":
    unittest.main()
