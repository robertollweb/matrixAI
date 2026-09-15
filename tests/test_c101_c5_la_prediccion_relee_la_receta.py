"""LA PREDICCIÓN RELEE LA RECETA, Y ESO NO LO PROBABA NADIE (2026-09-15).

**El hueco, encontrado por una auditoría independiente el mismo día del
arreglo.** `b2fad46` hizo que un nivel DECLARADO deje de leerse como dato
ausente, y lo cubrió con 19 pruebas. **Ninguna de las 19 importa
`prepare_dataset_from_provenance`** —comprobado con `grep`, da 0— y ése es
justo el camino por el que el motor denso **predice**
(`matrixai-engines/.../motores/densa.py:549`). El sabotaje que lo destapó:
quitar `tokens_de_ausencia` de la procedencia justo antes de re-preparar dejaba
**109 pruebas en verde**.

**Y AL ESCRIBIR ESTO, EL HALLAZGO SE CORRIGIÓ A SÍ MISMO.** La auditoría
concluyó que entrenar con la declaración y predecir con la heurística sería
«resultado equivocado silencioso». **Medido: no lo es.** Al perder la
declaración, la re-preparación no reproduce el dataset con el que se entrenó y
el producto **aborta** —invariante 4 del contrato 62: *mismo crudo + misma
versión de receta + hash preparado distinto → es un bug, se aborta*—.

O sea que el camino de predicción **sí tenía guarda**. Lo que no tenía era una
prueba que lo demostrara, y sin ella nadie sabía si abortaba o callaba: las 109
verdes decían lo mismo en los dos casos. **Ésa es la diferencia entre un
producto correcto y un producto del que se puede afirmar algo** — y es
exactamente por qué un sabotaje verde se investiga aunque el código esté bien.

**Cómo se prueba sin tocar el árbol.** Con la medición de 101-C5 corriendo,
`dataset_project.py` está congelado —cada intento lo re-importa desde disco—,
así que el sabotaje no se puede aplicar en sitio. No hace falta: se expresa
como aserto, quitando la clave de la procedencia que se le pasa.

**Dos asertos míos salieron rojos escribiendo esto, y los dos estaban mal EL
ASERTO**: supuse una clave `dataset_analysis` en la procedencia que no existe
—los faltantes viven en `missing_values.missing_category`— y esperaba que los
dos caminos dieran CSV distintos cuando lo que hacen es abortar. Medir la forma
real costó una orden.
"""
from __future__ import annotations

import unittest

from matrixai.training.dataset_project import (
    DatasetProjectError,
    generate_project_from_dataset,
    prepare_dataset_from_provenance,
)

#: `None` es un NIVEL declarado —«sin revestimiento»—, no un hueco. Es la forma
#: exacta del caso real que paró la medición: `MasVnrType` de
#: `house_prices_nominal`, donde 864 de 1.460 filas valen literalmente `None`.
_FILAS = [
    ("None", "120"), ("Piedra", "210"), ("None", "130"), ("Ladrillo", "180"),
    ("None", "125"), ("Piedra", "205"), ("None", "140"), ("Ladrillo", "175"),
    ("None", "135"), ("Piedra", "215"), ("None", "128"), ("Ladrillo", "185"),
]


def _csv(con_hueco: bool = True) -> str:
    filas = list(_FILAS)
    if con_hueco:
        # Un ausente DE VERDAD: celda vacía, que es lo que el motor escribe.
        filas.append(("", "150"))
    cuerpo = "\n".join(f"{rev},{precio}" for rev, precio in filas)
    return f"revestimiento,precio\n{cuerpo}\n"


class LaRecetaCongeladaLLEGA_A_LaPrediccionTest(unittest.TestCase):
    def test_la_receta_guarda_la_declaracion(self):
        """La mitad fácil, y hace falta: sin esto, lo de abajo no significa
        nada — dos caminos pueden coincidir porque no hay nada que leer."""
        res = generate_project_from_dataset(
            _csv(), target_column="precio", tokens_de_ausencia={""})
        spec = res["provenance"]["preparation_spec"]
        self.assertIn("tokens_de_ausencia", spec)
        self.assertEqual(spec["tokens_de_ausencia"], [""])

    def test_sin_declarar_la_receta_OMITE_la_clave_y_no_la_pone_vacia(self):
        """AUSENTE no es VACÍA. Una receta nacida sin declaración tiene que
        poder releerse con la heurística de entonces; escribir `[]` afirmaría
        que su autor dijo «aquí no falta nada», que es otra cosa."""
        res = generate_project_from_dataset(_csv(), target_column="precio")
        self.assertNotIn("tokens_de_ausencia",
                         res["provenance"]["preparation_spec"])

    def test_re_preparar_reproduce_el_MISMO_csv_byte_a_byte(self):
        """Ida y vuelta por el camino real de la predicción."""
        crudo = _csv()
        res = generate_project_from_dataset(
            crudo, target_column="precio", tokens_de_ausencia={""})
        re_prep = prepare_dataset_from_provenance(crudo, res["provenance"])
        self.assertEqual(re_prep.csv_text, res["csv_text"])

    def test_si_la_receta_PIERDE_la_declaracion_la_prediccion_SE_NIEGA(self):
        """**El sabotaje S10, escrito como aserto — y con el final corregido.**

        La auditoría que encontró este hueco concluyó que entrenar con la
        declaración y predecir con la heurística sería «resultado equivocado
        silencioso». **Medido, no lo es**: al perder la declaración, la
        re-preparación no reproduce el dataset con el que se entrenó y el
        producto **aborta** con un fallo de reproducibilidad (invariante 4 del
        contrato 62: *mismo crudo + misma versión de receta + hash preparado
        distinto → es un bug, se aborta*).

        O sea que el camino de predicción **sí tenía guarda**; lo que no tenía
        era una prueba que lo demostrara — y sin ella, nadie sabía si abortaba
        o callaba. Ésa es la diferencia entre un producto correcto y un
        producto del que se puede afirmar algo.

        Si algún día esa guarda se debilitara, esto se pone rojo.
        """
        crudo = _csv()
        res = generate_project_from_dataset(
            crudo, target_column="precio", tokens_de_ausencia={""})

        sin_declaracion = {
            **res["provenance"],
            "preparation_spec": {k: v
                                 for k, v in res["provenance"]["preparation_spec"].items()
                                 if k != "tokens_de_ausencia"},
        }
        with self.assertRaises(DatasetProjectError) as e:
            prepare_dataset_from_provenance(
                crudo, sin_declaracion, allow_incompatible_spec=True)
        self.assertIn("no reproduce", str(e.exception),
                      "se niega, pero por otro motivo: comprobar cuál antes de "
                      "darlo por bueno")

    def test_con_la_declaracion_el_nivel_None_NO_cuenta_como_ausente(self):
        """La mitad que da el significado: no basta con que difieran, hay que
        decir en qué dirección. Con la declaración, `None` es una categoría más
        y el único hueco es la celda vacía."""
        crudo = _csv()
        res = generate_project_from_dataset(
            crudo, target_column="precio", tokens_de_ausencia={""})
        faltantes = res["provenance"]["missing_values"]["missing_category"]
        self.assertEqual(
            faltantes["revestimiento"]["cells"], 1,
            "el único ausente es la celda vacía: las doce filas con el nivel "
            "`None` NO son huecos. Si esto dice 13, la declaración se perdió "
            "por el camino y el modelo entrenaría sobre otro dataset.")
        self.assertEqual(faltantes["revestimiento"]["category"], "__faltante__")


if __name__ == "__main__":
    unittest.main()
