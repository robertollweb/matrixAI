"""CONTRATO 81-C1 — políticas como esquema JSON versionado y cerrado.

Decisión de Roberto (2026-08-20): **no se inventa un lenguaje todavía**.
Un esquema JSON con predicados deterministas, resultado
`allow`/`deny`/`abstain`, precedencia explícita y **fallo cerrado**; sin
red, sin reloj, sin aleatoriedad y sin efectos externos.

Y del propio contrato, literal:

> El resultado **DEBE** identificar la **regla** que lo disparó:
> «`policy_results: passed`» no es evidencia auditable;
> «`require_data_cutoff@1.2.0 → allow (R3)`» sí.
"""

import unittest

from matrixai.pipelines.policy import PoliticaInvalida, evaluar_politica


def _politica(reglas, *, version="1.0.0", nombre="p", por_defecto="deny"):
    return {"schema_version": "1.0", "name": nombre, "version": version,
            "default": por_defecto, "rules": reglas}


class ElResultadoNOMBRALaReglaTest(unittest.TestCase):
    def test_dice_qué_regla_decidió_y_de_qué_política(self):
        r = evaluar_politica(
            _politica([{"id": "R3", "when": {"eq": [{"var": "cutoff"}, "2026-01-01"]},
                        "then": "allow"}], nombre="require_data_cutoff",
                      version="1.2.0"),
            {"cutoff": "2026-01-01"})
        self.assertEqual(r["decision"], "allow")
        self.assertEqual(r["rule_id"], "R3")
        # La evidencia auditable que pide el contrato, entera.
        self.assertEqual(r["explain"], "require_data_cutoff@1.2.0 → allow (R3)")

    def test_cuando_decide_el_DEFECTO_tambien_se_dice(self):
        """«Ninguna regla casó» es un motivo, y esconderlo haría que un
        `deny` por defecto pareciera un `deny` razonado."""
        r = evaluar_politica(_politica([{"id": "R1", "when": {"eq": [{"var": "a"}, 1]},
                                         "then": "allow"}]), {"a": 2})
        self.assertEqual(r["decision"], "deny")
        self.assertIsNone(r["rule_id"])
        self.assertIn("default", r["explain"])


class LaPrecedenciaEsEXPLICITATest(unittest.TestCase):
    def test_manda_la_PRIMERA_regla_que_casa(self):
        reglas = [{"id": "R1", "when": {"eq": [{"var": "a"}, 1]}, "then": "deny"},
                  {"id": "R2", "when": {"eq": [{"var": "a"}, 1]}, "then": "allow"}]
        r = evaluar_politica(_politica(reglas), {"a": 1})
        self.assertEqual(r["rule_id"], "R1")
        self.assertEqual(r["decision"], "deny")


class FalloCERRADOTest(unittest.TestCase):
    def test_un_predicado_que_no_se_entiende_NIEGA_no_ignora(self):
        """Saltarse una regla que no se entiende deja pasar justo lo que
        esa regla existía para parar."""
        r = evaluar_politica(
            _politica([{"id": "R1", "when": {"inventado": [1, 2]}, "then": "allow"}]),
            {})
        self.assertEqual(r["decision"], "deny")
        self.assertIn("R1", r["explain"])

    def test_un_dato_que_falta_NO_se_rellena(self):
        r = evaluar_politica(
            _politica([{"id": "R1", "when": {"eq": [{"var": "ausente"}, None]}, "then": "allow"}]),
            {})
        # Ausente no es `None`: comparar un hueco con null diría que sí.
        self.assertEqual(r["decision"], "deny")

    def test_una_politica_sin_version_de_esquema_se_RECHAZA(self):
        with self.assertRaises(PoliticaInvalida):
            evaluar_politica({"name": "p", "rules": []}, {})

    def test_un_esquema_FUTURO_no_se_interpreta_a_medias(self):
        with self.assertRaises(PoliticaInvalida):
            evaluar_politica({**_politica([]), "schema_version": "9.9"}, {})


class DeterministaTest(unittest.TestCase):
    def test_la_misma_entrada_da_la_misma_salida(self):
        politica = _politica([{"id": "R1", "when": {"gt": [{"var": "n"}, 5]}, "then": "allow"}])
        primero = evaluar_politica(politica, {"n": 10})
        segundo = evaluar_politica(politica, {"n": 10})
        self.assertEqual(primero, segundo)

    def test_ABSTAIN_es_un_resultado_no_un_hueco(self):
        """Abstenerse es decir «esta política no opina», que es distinto
        de permitir y de negar. Sin él, una política que no aplica tendría
        que mentir en una dirección."""
        r = evaluar_politica(
            _politica([{"id": "R1", "when": {"eq": [{"var": "a"}, 1]}, "then": "abstain"}]),
            {"a": 1})
        self.assertEqual(r["decision"], "abstain")
        self.assertEqual(r["rule_id"], "R1")


if __name__ == "__main__":
    unittest.main()
