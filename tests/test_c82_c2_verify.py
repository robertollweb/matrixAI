"""CONTRATO 82-C2 — `matrixai verify`, en local.

Lee un paquete y contesta **por ETAPAS**, no con un veredicto único: un
«falló» que no dice dónde obliga a adivinar, y las cuatro fallan por
motivos distintos.

| Etapa | Qué comprueba |
|---|---|
| `manifest` | integridad del manifiesto y de cada artefacto por su `sha256` |
| `R1` | el dataset regenerado tiene el sha256 COMPLETO esperado |
| `training` | el entrenamiento llegó a término |
| `R3` | las métricas caen dentro de su tolerancia |

Cada una con `PASS` / `FAIL` / `NOT_RUN` / `INCOMPARABLE`.

**`INCOMPARABLE` no es un fallo**: es «no se puede comparar» —falta el
`.mxtrain`, no hay receta, otro entorno—, y *un fallo por falta de acceso
no es una manipulación* (regla heredada del 81).

**Y `verify` procesa paquetes NO CONFIABLES**: el que lo ejecuta puede
haber descargado el ZIP de cualquier sitio. No se ejecuta nada que venga
dentro del paquete para decidir si el paquete es bueno.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from matrixai.export.verify import ESTADOS, verify_package

_RECETA = "genera 200 filas\n"
_CSV = "a,b\n1,2\n3,4\n"


def _bundle(**cambios) -> Path:
    """Un paquete honesto en disco, con su `reproduce.json` de verdad."""
    from matrixai.export.reproduce import (
        añadir_inventario_de_ficheros,
        write_reproduce_manifest,
    )
    d = Path(tempfile.mkdtemp())
    (d / "model.mxai").write_text("NETWORK N\n  DENSE 4\n")
    (d / "training.mxtrain").write_text(
        Path("examples/celsius_to_kelvin.mxtrain").read_text(encoding="utf-8"))
    (d / "recipe.txt").write_text(_RECETA)
    sha = hashlib.sha256(_CSV.encode()).hexdigest()
    train = (d / "training.mxtrain").read_text(encoding="utf-8")
    captura = {
        "schema_version": "1.2",
        "mxai_sha256": hashlib.sha256((d / "model.mxai").read_bytes()).hexdigest(),
        "mxtrain_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "mxtrain_text": train,
        "recipe_sha256": hashlib.sha256(_RECETA.encode()).hexdigest(),
        "recipe_text": _RECETA,
        "seeds": {"dataset": 42},
        "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
        "dataset_rows": 2, "dataset_rows_used": 2,
        "epochs_effective": 10000, "epochs_ran": 10000,
        "warm_start": False,
        "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
    }
    captura.update(cambios.pop("captura", {}))
    write_reproduce_manifest(
        d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
        dataset_sha256=sha, dataset_rows=2, generation={"seeds": {"dataset": 42}},
        run_provenance=captura, weights_source="trained")
    # Y EL INVENTARIO, como hace el producto (`bundle.py`, justo antes de
    # promover el paquete). Sin él este fixture describía un paquete que
    # el producto ya no construye: desde H5 del refutador (2026-08-20) un
    # paquete sin inventario sale INCOMPARABLE, y con razón —no se puede
    # comprobar la integridad de lo que nadie ha declarado—.
    añadir_inventario_de_ficheros(d)
    return d


class LasCuatroEtapasSiempreEstanTest(unittest.TestCase):
    """Un informe al que le falta una etapa obliga a adivinar si pasó o si
    ni se intentó. Las cuatro salen SIEMPRE, con su estado."""

    def test_estan_las_cuatro_y_con_un_estado_conocido(self):
        r = verify_package(_bundle())
        self.assertEqual(set(r["stages"]), {"manifest", "R1", "training", "R3"})
        for nombre, etapa in r["stages"].items():
            with self.subTest(etapa=nombre):
                self.assertIn(etapa["status"], ESTADOS)
                # Y NUNCA un estado a secas: lo que no es PASS dice por qué.
                if etapa["status"] != "PASS":
                    self.assertTrue(etapa.get("reason"), f"{nombre} sin motivo")

    def test_la_salida_es_json_serializable(self):
        # Es su formato de salida: si no se puede volcar, no hay informe.
        json.dumps(verify_package(_bundle()))


class ElManifiestoSeComprueba(unittest.TestCase):
    def test_un_paquete_honesto_pasa_la_etapa_manifest(self):
        r = verify_package(_bundle())
        self.assertEqual(r["stages"]["manifest"]["status"], "PASS",
                         r["stages"]["manifest"].get("reason"))

    def test_cambiar_una_linea_de_la_receta_da_FAIL_en_manifest(self):
        """El criterio de cierre del contrato, literal: sabotear el paquete
        produce FAIL en `manifest`, no un verde."""
        d = _bundle()
        (d / "recipe.txt").write_text(_RECETA + "y una línea más\n")
        etapa = verify_package(d)["stages"]["manifest"]
        self.assertEqual(etapa["status"], "FAIL")
        # Y dice QUÉ artefacto, no «algo no cuadra».
        self.assertIn("recipe", json.dumps(etapa))

    def test_tocar_el_manifiesto_tambien_se_ve(self):
        d = _bundle()
        m = json.loads((d / "reproduce.json").read_text())
        m["reproducible"] = True
        m["claim"] = "todo perfecto"
        (d / "reproduce.json").write_text(json.dumps(m))
        self.assertEqual(verify_package(d)["stages"]["manifest"]["status"], "FAIL")

    def test_sin_reproduce_json_no_hay_nada_que_verificar(self):
        d = Path(tempfile.mkdtemp())
        r = verify_package(d)
        self.assertEqual(r["stages"]["manifest"]["status"], "INCOMPARABLE")
        # No es FAIL: no hay manipulación, hay ausencia.
        self.assertNotEqual(r["exit_code"], 0)


class LoQueNoSePuedeCompararNoEsUnFallo(unittest.TestCase):
    """Regla heredada del 81: un fallo por falta de acceso no es una
    manipulación. Confundirlos acusaría a paquetes honestos."""

    def test_sin_receta_R1_es_INCOMPARABLE_no_FAIL(self):
        d = Path(tempfile.mkdtemp())
        from matrixai.export.reproduce import write_reproduce_manifest
        (d / "model.mxai").write_text("NETWORK N\n  DENSE 4\n")
        write_reproduce_manifest(d, weights_source="trained")
        self.assertEqual(verify_package(d)["stages"]["R1"]["status"], "INCOMPARABLE")

    def test_R3_es_NOT_RUN_mientras_falten_sus_datos(self):
        """El contrato lo exige: hasta que el producto capture split,
        evaluador, agregación, dirección y tolerancia, `verify` DEBE
        devolver NOT_RUN en R3 — no un PASS sobre una comparación que no
        ha hecho."""
        etapa = verify_package(_bundle())["stages"]["R3"]
        self.assertEqual(etapa["status"], "NOT_RUN")
        self.assertTrue(etapa["reason"])

    def test_reentrenar_no_se_hace_sin_pedirlo(self):
        """Reentrenar cuesta minutos u horas: se ofrece, no se impone. Y
        no haberlo hecho se DICE, en vez de dar un verde vacío."""
        etapa = verify_package(_bundle())["stages"]["training"]
        self.assertEqual(etapa["status"], "NOT_RUN")


class ElCodigoDeSalidaEsParaGuionesTest(unittest.TestCase):
    def test_todo_bien_o_incomparable_no_es_lo_mismo_que_manipulado(self):
        limpio = verify_package(_bundle())["exit_code"]
        d = _bundle()
        (d / "recipe.txt").write_text("otra cosa\n")
        saboteado = verify_package(d)["exit_code"]
        self.assertNotEqual(limpio, saboteado)
        # Un FAIL de integridad tiene su propio código: un guion puede
        # distinguir «no se pudo comprobar» de «alguien lo tocó».
        self.assertEqual(saboteado, 2)


if __name__ == "__main__":
    unittest.main()


class PorElComandoNoSoloPorLaFuncionTest(unittest.TestCase):
    """Probar la función no es probar el producto: quien usa esto escribe
    `matrixai verify`, y lo que le importa es lo que ve y el código con el
    que sale."""

    def _correr(self, *args: str) -> tuple[int, str]:
        # `main()` lee de `sys.argv` (no acepta lista), así que se
        # sustituye: es la misma puerta por la que entra un usuario.
        import contextlib
        import io
        import sys as _sys
        from matrixai.cli import main
        salida = io.StringIO()
        viejo = _sys.argv
        _sys.argv = ["matrixai", "verify", *args]
        try:
            with contextlib.redirect_stdout(salida):
                codigo = main()
        finally:
            _sys.argv = viejo
        return codigo, salida.getvalue()

    def test_un_paquete_honesto_NO_sale_acusado(self):
        """AUDITORÍA EXTERNA (2026-08-20): esto exigía **código 0** y el
        paquete tenía `R1 INCOMPARABLE`. O sea que «no se pudo comprobar»
        salía con el código de «nada falló», y quien encadenara
        `verify && desplegar` trataría un paquete no verificado como
        verificado.

        Se conserva la intención —a un paquete honesto NO se le acusa— con
        el criterio corregido: el 2 es la acusación, y ése no aparece. El 3
        dice «no se pudo comprobar del todo», que es la verdad."""
        codigo, texto = self._correr(str(_bundle()))
        self.assertNotEqual(codigo, 2, "un paquete honesto no puede salir acusado")
        self.assertIn(codigo, (0, 3))
        # Las cuatro etapas a la vista, no solo un veredicto.
        for etapa in ("manifest", "R1", "training", "R3"):
            self.assertIn(etapa, texto)

    def test_y_si_algo_NO_se_pudo_comprobar_sale_con_3(self):
        """El código que distingue «nada falló» de «no se pudo mirar»."""
        from matrixai.export.verify import verify_package

        r = verify_package(_bundle())
        sin_comprobar = [k for k, e in r["stages"].items()
                         if e["status"] == "INCOMPARABLE"]
        if sin_comprobar:
            self.assertEqual(r["exit_code"], 3)
            self.assertFalse(r["fully_checked"])
            self.assertEqual(sorted(r["unchecked_stages"]), sorted(sin_comprobar))
        else:
            self.assertEqual(r["exit_code"], 0)
            self.assertTrue(r["fully_checked"])

    def test_saboteado_sale_con_2_y_NOMBRA_el_artefacto(self):
        d = _bundle()
        (d / "recipe.txt").write_text("otra receta\n")
        codigo, texto = self._correr(str(d))
        self.assertEqual(codigo, 2)
        self.assertIn("FAIL", texto)
        self.assertIn("recipe", texto)

    def test_el_json_es_json_y_conserva_el_codigo(self):
        d = _bundle()
        (d / "recipe.txt").write_text("otra receta\n")
        codigo, texto = self._correr(str(d), "--json")
        self.assertEqual(codigo, 2)
        informe = json.loads(texto)
        self.assertEqual(informe["stages"]["manifest"]["status"], "FAIL")


class R1RegeneraDeVerdadTest(unittest.TestCase):
    """La etapa central del contrato, medida contra un dataset REAL.

    Los casos de arriba comprueban la forma del informe; éste comprueba
    que R1 hace lo que dice: rehacer el dataset con la receta y la semilla
    del paquete y comparar su sha256 COMPLETO.
    """

    def _paquete_regenerable(self, *, romper_el_digest: bool = False) -> Path:
        from matrixai.export.reproduce import write_reproduce_manifest
        from matrixai.playground import _generate_synthetic_dataset
        from matrixai.training.dataset_project import generate_project_from_dataset

        filas = "".join(f"{c},{c + 273.15}\n" for c in range(30))
        proj = generate_project_from_dataset(
            "centigrados,prediccionKelvin\n" + filas, "prediccionKelvin",
            column_type_overrides={"centigrados": "number"},
            column_range_overrides={"centigrados": (0.0, 29.0)})

        # El CSV que el paquete promete: se genera AQUÍ con los mismos
        # parámetros que luego se declaran, para que R1 tenga algo real
        # contra lo que comparar.
        gen = _generate_synthetic_dataset(
            proj["mxai"], proj["training_text"], 30, 7, "coherent",
            field_ranges_override=proj.get("field_ranges"))
        csv = gen["csv_text"]
        sha = hashlib.sha256(csv.encode("utf-8")).hexdigest()
        if romper_el_digest:
            sha = hashlib.sha256((csv + "otra fila\n").encode("utf-8")).hexdigest()

        from matrixai.export.reproduce import (
            _epochs_from_training,
            añadir_inventario_de_ficheros,
            write_reproduce_manifest,
        )
        _epocas = _epochs_from_training(proj["training_text"])

        d = Path(tempfile.mkdtemp())
        (d / "model.mxai").write_text(proj["mxai"], encoding="utf-8")
        (d / "training.mxtrain").write_text(proj["training_text"], encoding="utf-8")
        (d / "recipe.txt").write_text("regenerable\n", encoding="utf-8")
        captura = {
            "schema_version": "1.2",
            "mxai_sha256": hashlib.sha256(proj["mxai"].encode()).hexdigest(),
            "mxtrain_sha256": hashlib.sha256(proj["training_text"].encode()).hexdigest(),
            "mxtrain_text": proj["training_text"],
            "recipe_sha256": hashlib.sha256(b"regenerable\n").hexdigest(),
            "recipe_text": "regenerable\n",
            "seeds": {"dataset": 7},
            "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
            "dataset_rows": 30, "dataset_rows_used": 30,
            # Las que declara el `.mxtrain` DEL PROYECTO, leídas de él: con
            # otra cifra el manifiesto declara conflicto en
            # `generation.epochs` y R1 sale INCOMPARABLE por un motivo que
            # no es el que este caso mide (me pasó, y el manifiesto tenía
            # razón).
            "epochs_effective": _epocas, "epochs_ran": _epocas, "warm_start": False,
            "mode": "coherent",
            "field_ranges": proj.get("field_ranges") or {},
            "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
        }
        write_reproduce_manifest(
            d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
            dataset_sha256=sha, dataset_rows=30,
            generation={"seeds": {"dataset": 7}}, run_provenance=captura,
            weights_source="trained")
        # El inventario, como el producto. Ver `_bundle` arriba.
        añadir_inventario_de_ficheros(d)
        return d

    def test_un_dataset_que_se_regenera_igual_da_PASS(self):
        etapa = verify_package(self._paquete_regenerable())["stages"]["R1"]
        self.assertEqual(etapa["status"], "PASS", etapa.get("reason"))
        self.assertEqual(etapa.get("rows"), 30)

    def test_si_el_digest_declarado_no_es_el_que_sale_es_FAIL(self):
        """Y FAIL de verdad, no INCOMPARABLE: el paquete dijo qué debía
        salir y ha salido otra cosa."""
        informe = verify_package(self._paquete_regenerable(romper_el_digest=True))
        etapa = informe["stages"]["R1"]
        self.assertEqual(etapa["status"], "FAIL", etapa.get("reason"))
        # Los DOS digests a la vista: «no coincide» a secas no deja
        # comprobar nada a quien lo lee.
        self.assertIn("expected", etapa)
        self.assertIn("found", etapa)
        self.assertNotEqual(etapa["expected"], etapa["found"])
        self.assertEqual(informe["exit_code"], 2)

    def test_con_retrain_el_entrenamiento_llega_a_termino(self):
        """`--retrain` entrena con el dataset REGENERADO, no con uno
        traído de fuera: si R1 no pudo rehacerlo, no hay con qué entrenar.
        """
        informe = verify_package(self._paquete_regenerable(), run_training=True)
        etapa = informe["stages"]["training"]
        self.assertIn(etapa["status"], ("PASS", "INCOMPARABLE"), etapa.get("reason"))
        if etapa["status"] == "PASS":
            # Y no promete más de lo que ha mirado: terminar no es igualar
            # las métricas, que es R3.
            self.assertIn("R3", etapa.get("note", ""))
        else:
            # Si no pudo, DICE por qué, y no lo llama fallo del paquete.
            self.assertTrue(etapa["reason"])

    def test_sin_R1_no_hay_con_que_entrenar_y_se_dice(self):
        d = _bundle()  # su receta no regenera nada comparable
        etapa = verify_package(d, run_training=True)["stages"]["training"]
        self.assertEqual(etapa["status"], "INCOMPARABLE")
        self.assertTrue(etapa["reason"])


class R3ComparaLasMetricasTest(unittest.TestCase):
    """La cuarta etapa. Tres estados, y son tres cosas distintas:

    * `NOT_RUN` si no se ha reentrenado — sin valor nuevo no hay nada que
      comparar, e inventarse la comparación es justo lo que no puede pasar.
    * `INCOMPARABLE` si el paquete no publica métricas contrastables. Que
      al PRODUCTO le falte capturar el evaluador o la tolerancia es una
      carencia suya, **no una mentira del paquete**.
    * `PASS`/`FAIL` solo cuando de verdad hay con qué comparar.
    """

    def _con_metricas(self, metricas):
        from matrixai.export.reproduce import write_reproduce_manifest
        d = _bundle()
        (d / "reproduce.json").unlink()
        train = (d / "training.mxtrain").read_text(encoding="utf-8")
        sha = hashlib.sha256(_CSV.encode()).hexdigest()
        write_reproduce_manifest(
            d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
            dataset_sha256=sha, dataset_rows=2, generation={"seeds": {"dataset": 42}},
            metrics=metricas, weights_source="trained",
            run_provenance={
                "schema_version": "1.2",
                "mxai_sha256": hashlib.sha256((d / "model.mxai").read_bytes()).hexdigest(),
                "mxtrain_sha256": hashlib.sha256(train.encode()).hexdigest(),
                "mxtrain_text": train,
                "recipe_sha256": hashlib.sha256(_RECETA.encode()).hexdigest(),
                "recipe_text": _RECETA, "seeds": {"dataset": 42},
                "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
                "dataset_rows": 2, "dataset_rows_used": 2,
                "epochs_effective": 10000, "epochs_ran": 10000, "warm_start": False,
                "recipe_verification": {"verified": True, "code": "regenera_el_dataset"}})
        return d

    def test_sin_reentrenar_es_NOT_RUN_no_un_verde(self):
        etapa = verify_package(_bundle())["stages"]["R3"]
        self.assertEqual(etapa["status"], "NOT_RUN")
        self.assertIn("--retrain", etapa["reason"])

    def test_una_metrica_a_la_que_le_faltan_datos_es_INCOMPARABLE(self):
        """Y se DICE qué le falta, para que quien lo lea sepa qué pedirle
        a quien hizo el paquete, en vez de un «no se puede» a secas."""
        d = self._con_metricas([{"name": "accuracy", "value": 0.9}])
        # `training` no llega a PASS con este paquete, así que se mide la
        # rama directamente sobre el manifiesto.
        from matrixai.export.verify import _verificar_r3
        manifiesto = json.loads((d / "reproduce.json").read_text())
        etapa = _verificar_r3(manifiesto, {"status": "PASS", "metrics": {"accuracy": 0.9}})
        self.assertEqual(etapa["status"], "INCOMPARABLE")
        self.assertTrue(etapa["missing"], "no dice qué le falta a la métrica")

    def test_dentro_de_tolerancia_es_PASS(self):
        from matrixai.export.verify import _verificar_r3
        sha = hashlib.sha256(_CSV.encode()).hexdigest()
        completa = {"name": "accuracy", "value": 0.90, "split": "validation",
                    "dataset_sha256": sha, "direction": "higher_is_better",
                    "tolerance_abs": 0.05}
        d = self._con_metricas([completa])
        manifiesto = json.loads((d / "reproduce.json").read_text())
        etapa = _verificar_r3(manifiesto, {"status": "PASS", "metrics": {"accuracy": 0.93}})
        self.assertEqual(etapa["status"], "PASS", etapa.get("reason"))
        self.assertIn("accuracy", etapa["checked"])

    def test_fuera_de_tolerancia_es_FAIL_y_enseña_los_dos_valores(self):
        from matrixai.export.verify import _verificar_r3
        sha = hashlib.sha256(_CSV.encode()).hexdigest()
        completa = {"name": "accuracy", "value": 0.90, "split": "validation",
                    "dataset_sha256": sha, "direction": "higher_is_better",
                    "tolerance_abs": 0.01}
        d = self._con_metricas([completa])
        manifiesto = json.loads((d / "reproduce.json").read_text())
        etapa = _verificar_r3(manifiesto, {"status": "PASS", "metrics": {"accuracy": 0.40}})
        self.assertEqual(etapa["status"], "FAIL")
        # Los dos valores y la tolerancia: «fuera de rango» no deja ver si
        # se pasó por poco o por un orden de magnitud.
        fallo = etapa["metrics"][0]
        self.assertEqual(fallo["published"], 0.90)
        self.assertEqual(fallo["obtained"], 0.40)
        self.assertIn("tolerance", fallo)


class R3DiceElMOTIVO_DeVerdadTest(unittest.TestCase):
    """Sonda del 82-C2 conduciendo el CLI (2026-08-20) — la mitad cara del
    corte, que el refutador dejó sin refutar por falta de tiempo.

    `R3` decía «use --retrain» **pasara lo que pasara**, así que con
    `--retrain` YA PUESTO seguía mandando a ponerlo: le decía a quien
    verifica que hiciera lo que acababa de hacer. Medido tres veces
    seguidas contra el CLI, siempre igual.

    Lo que falta cuando el reentrenamiento se pidió y no llegó a PASS no
    es el flag: es lo que paró a `training`.
    """

    def test_sin_pedirlo_SI_manda_a_ponerlo(self):
        """El otro lado primero: ahí el mensaje era correcto y tiene que
        seguir siéndolo."""
        from matrixai.export.verify import _verificar_r3

        etapa = _verificar_r3({}, {"status": "NOT_RUN"})
        self.assertEqual(etapa["status"], "NOT_RUN")
        self.assertIn("--retrain", etapa["reason"])

    def test_habiendolo_pedido_NO_manda_a_hacer_lo_ya_hecho(self):
        from matrixai.export.verify import _verificar_r3

        etapa = _verificar_r3({}, {
            "status": "INCOMPARABLE",
            "reason": "the dataset could not be regenerated, so there is nothing to train on"})
        self.assertEqual(etapa["status"], "NOT_RUN")
        self.assertNotIn("--retrain", etapa["reason"])
        # Y dice el motivo DE VERDAD, que es el de `training`.
        self.assertIn("could not be regenerated", etapa["reason"])

    def test_y_un_FAIL_del_entrenamiento_tambien_viaja(self):
        from matrixai.export.verify import _verificar_r3

        etapa = _verificar_r3({}, {"status": "FAIL", "reason": "training blew up"})
        self.assertIn("training blew up", etapa["reason"])
        self.assertNotIn("--retrain", etapa["reason"])
