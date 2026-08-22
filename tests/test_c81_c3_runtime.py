"""CONTRATO 81-C3 — el runtime: resolver por digest y respetar el orden.

Lo que este fichero mide, del §13.2:

- **resolver modelos por digest**, no por nombre: un nombre puede apuntar
  hoy a una cosa y mañana a otra, y entonces la traza no describe lo que
  corrió;
- **ejecutar respetando las dependencias**, y negarse si hay un ciclo;
- **límites de tiempo** por nodo;
- y **verificar el pipeline ANTES de ejecutarlo**, que es lo que impide
  empezar algo que no se puede terminar.
"""

import unittest

from matrixai.pipelines.runtime import (
    CicloEnElPipeline,
    DigestNoCoincide,
    ordenar_por_dependencias,
    resolver_por_digest,
    verificar_antes_de_ejecutar,
)


class ResolverPorDIGESTTest(unittest.TestCase):
    def test_un_digest_que_no_coincide_PARA_la_ejecucion(self):
        """Si el modelo que hay no es el que el pipeline declara, seguir
        produciría una traza que describe otra cosa."""
        registry = {"m@v1": "sha256:" + "a" * 64}
        with self.assertRaises(DigestNoCoincide):
            resolver_por_digest("m@v1", "sha256:" + "b" * 64, registry)

    def test_cuando_coincide_devuelve_el_digest_resuelto(self):
        registry = {"m@v1": "sha256:" + "a" * 64}
        self.assertEqual(resolver_por_digest("m@v1", "sha256:" + "a" * 64, registry),
                         "sha256:" + "a" * 64)

    def test_sin_digest_declarado_NO_se_resuelve_por_nombre(self):
        """Un nombre puede apuntar hoy a una cosa y mañana a otra. Aceptar
        «el que haya» convertiría la traza en una promesa que no se puede
        comprobar después."""
        registry = {"m@v1": "sha256:" + "a" * 64}
        with self.assertRaises(DigestNoCoincide):
            resolver_por_digest("m@v1", None, registry)

    def test_un_modelo_que_no_esta_se_dice(self):
        with self.assertRaises(DigestNoCoincide) as caja:
            resolver_por_digest("no_esta@v1", "sha256:" + "a" * 64, {})
        self.assertIn("no_esta@v1", str(caja.exception))


class ElORDENLoManDanLasDependenciasTest(unittest.TestCase):
    def test_un_nodo_va_despues_de_aquel_del_que_depende(self):
        orden = ordenar_por_dependencias({"b": ["a"], "a": [], "c": ["b"]})
        self.assertEqual(orden, ["a", "b", "c"])

    def test_un_CICLO_se_rechaza_ANTES_de_ejecutar(self):
        """Empezar un pipeline con un ciclo deja trabajo a medias y
        efectos ya ejecutados que no se pueden deshacer."""
        with self.assertRaises(CicloEnElPipeline):
            ordenar_por_dependencias({"a": ["b"], "b": ["a"]})

    def test_una_dependencia_que_NO_existe_se_rechaza(self):
        with self.assertRaises(CicloEnElPipeline) as caja:
            ordenar_por_dependencias({"a": ["fantasma"]})
        self.assertIn("fantasma", str(caja.exception))

    def test_el_orden_es_ESTABLE_entre_ejecuciones(self):
        """Dos ejecuciones del mismo pipeline tienen que dar el mismo
        orden: si no, dos trazas del mismo grafo no se pueden comparar."""
        grafo = {"c": [], "a": [], "b": []}
        self.assertEqual(ordenar_por_dependencias(grafo),
                         ordenar_por_dependencias(dict(grafo)))


class VerificarANTESDeEjecutarTest(unittest.TestCase):
    def _pipeline(self, **cambios):
        base = {
            # El `pipeline_id` es OBLIGATORIO desde la 1ª pasada de
            # auditoría: sin él la raíz de la traza caía a un `"run"`
            # inventado e IGUAL para todas las ejecuciones anónimas.
            "pipeline_id": "p-1",
            "nodes": [
                {"id": "a", "model": "m@v1", "entry_hash": "sha256:" + "a" * 64,
                 "depends_on": []},
            ],
            "timeout_s": 60,
        }
        base.update(cambios)
        return base

    def test_un_pipeline_correcto_pasa(self):
        registry = {"m@v1": "sha256:" + "a" * 64}
        r = verificar_antes_de_ejecutar(self._pipeline(), registry)
        self.assertTrue(r["ok"])
        self.assertEqual(r["order"], ["a"])

    def test_un_digest_que_no_cuadra_lo_PARA(self):
        registry = {"m@v1": "sha256:" + "z" * 64}
        r = verificar_antes_de_ejecutar(self._pipeline(), registry)
        self.assertFalse(r["ok"])
        self.assertTrue(r["problems"])

    def test_un_limite_de_tiempo_que_no_es_un_tiempo_se_rechaza(self):
        """Un `timeout` de cero o negativo no limita: mata antes de
        empezar, o no limita nada. Las dos cosas mienten sobre lo que
        promete el campo."""
        for malo in (0, -1, "60"):
            with self.subTest(t=malo):
                r = verificar_antes_de_ejecutar(
                    self._pipeline(timeout_s=malo), {"m@v1": "sha256:" + "a" * 64})
                self.assertFalse(r["ok"])

    def test_dice_TODOS_los_problemas_no_solo_el_primero(self):
        """Arreglar uno y volver a chocar con el siguiente es hacer
        trabajar a alguien de más."""
        malo = {"nodes": [{"id": "a", "model": "no_esta@v1", "depends_on": ["x"]}],
                "timeout_s": 0}
        r = verificar_antes_de_ejecutar(malo, {})
        self.assertGreaterEqual(len(r["problems"]), 2)


if __name__ == "__main__":
    unittest.main()


class LoQueLaAUDITORIA_EncontroTest(unittest.TestCase):
    """1ª pasada de auditoría del 81 (2026-08-20). Tres agujeros que la
    verificación previa dejaba pasar, y los tres se veían **midiendo**:
    ninguno estaba en el código escrito, estaban en lo que el código NO
    miraba."""

    def _pipeline(self, **cambios):
        base = {
            "pipeline_id": "p-1", "timeout_s": 60,
            "nodes": [{"id": "a", "model": "m@v1",
                       "entry_hash": "sha256:" + "a" * 64}],
        }
        base.update(cambios)
        return base

    _REGISTRY = {"m@v1": "sha256:" + "a" * 64}

    def test_DOS_pasos_con_el_mismo_id_no_pasan(self):
        """Corrían con `ok` y uno de los dos **desaparecía**: el grafo es
        un diccionario y el segundo pisaba al primero. Un nodo declarado
        que nunca corrió y nadie lo dijo."""
        nodo = {"id": "a", "model": "m@v1", "entry_hash": "sha256:" + "a" * 64}
        r = verificar_antes_de_ejecutar(
            self._pipeline(nodes=[dict(nodo), dict(nodo)]), self._REGISTRY)
        self.assertFalse(r["ok"])
        self.assertTrue(any("dos pasos con el id" in p for p in r["problems"]), r["problems"])

    def test_sin_pipeline_id_no_pasa(self):
        """La raíz de la traza caía a un `"run"` inventado, y **todas** las
        ejecuciones anónimas compartían raíz: dos trazas distintas no se
        podían distinguir."""
        pipeline = self._pipeline()
        del pipeline["pipeline_id"]
        r = verificar_antes_de_ejecutar(pipeline, self._REGISTRY)
        self.assertFalse(r["ok"])
        self.assertTrue(any("pipeline_id" in p for p in r["problems"]), r["problems"])

    def test_una_POLITICA_rota_se_caza_ANTES_de_ejecutar_nada(self):
        """Antes solo se descubría al llegar a su nodo, con lo que los
        anteriores ya habían corrido —y sus efectos externos con ellos—.
        La verificación previa existe justamente para eso."""
        r = verificar_antes_de_ejecutar(
            self._pipeline(policy={"schema_version": "9.9"}), self._REGISTRY)
        self.assertFalse(r["ok"])
        self.assertTrue(any("no se puede leer" in p for p in r["problems"]), r["problems"])

    def test_y_una_politica_BUENA_no_estorba(self):
        """Una regla que no dispare por falta de datos no es un problema
        del pipeline: aquí se comprueba que se puede LEER, no que decida."""
        buena = {"schema_version": "1.0", "name": "p", "version": "1",
                 "default": "allow",
                 "rules": [{"id": "R1", "when": {"eq": [{"var": "x"}, 1]},
                            "then": "allow"}]}
        r = verificar_antes_de_ejecutar(self._pipeline(policy=buena), self._REGISTRY)
        self.assertTrue(r["ok"], r["problems"])
