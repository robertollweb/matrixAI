"""101-C0 — LA PARTICIÓN SE HONRA, Y LO DE ANTES NO SE MUEVE.

QUÉ FALLABA, medido el 2026-09-04 con motores reales. El generador escribe
`SPLIT train=0.8 validation=0.2 seed=42` y **ese `seed=42` no manda**: el
entrenador denso ignora `training.dataset.split` por completo salvo
`mode=temporal`, corta por un 0,8 fijo en el orden en que llegan las filas y no
baraja. Dos entrenamientos con semillas distintas daban la misma partición.

Y hay algo peor que la semilla: **no había conjunto de prueba**. La misma
partición de validación elegía la mejor época *y* publicaba la métrica. Un
número elegido sobre los datos con los que se eligió no estima nada — es el
defecto que el 100 documenta y por el que el 101 existe.

CÓMO SE ARREGLA SIN ROMPER NADA. Lo que manda es una **versión de protocolo
declarada en el propio `.mxtrain`** (`SPLIT … protocol=2`). Sin ella, todo se
comporta y se serializa **exactamente como antes** — que es lo que permite que
`matrixai verify --retrain` siga reproduciendo un proyecto de hace meses. Con
ella, se honra lo declarado: semilla, modo y un tercer tramo que no toca nadie.

La pieza más importante de este fichero es el CANARIO: cambiar solo el objetivo
de las filas de prueba no puede cambiar el modelo. Si lo cambia, es que la
prueba está entrando en el entrenamiento por algún sitio.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matrixai.training.parser import (
    MatrixAITrainingParseError,
    parse_training_file,
    parse_training_text,
)
from matrixai.training.particion import (
    PROTOCOLO_SEPARACION,
    particion_declarada,
    particion_legada,
    particion_para,
    particion_temporal,
)


def _mxtrain(split: str, modelo: str = "m.mxai", csv: str = "d.csv") -> str:
    return f"""MODEL {modelo}

DATASET D
  SOURCE csv("{csv}")
  INPUT Input FROM COLUMNS [a, b]
  TARGET y: Label[no, si]
  {split}
END

LOSS L
  TYPE cross_entropy
  PREDICTION Net
  TARGET y
END

OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END

RUN
  EPOCHS 3
END
"""


class LoDeAntesNoSeMueveTest(unittest.TestCase):
    """La mitad del corte es que un proyecto antiguo siga dando lo mismo."""

    def test_la_particion_legada_es_LA_MISMA_formula_que_habia(self):
        """Equivalencia con la fórmula que estaba escrita dentro del entrenador.

        Se comprueba en TODOS los tamaños de 0 a 500, no en tres a mano: el
        borde (`n <= 1`, el `max(1, …)`) es justo donde una extracción se
        desvía, y un puñado de casos elegidos por mí lo pasaría igual.
        """
        for n in range(0, 501):
            with self.subTest(n=n):
                viejo = max(1, int(n * 0.8)) if n > 1 else n
                p = particion_legada(n)
                self.assertEqual(p.train, tuple(range(viejo)))
                self.assertEqual(p.validation, tuple(range(viejo, n)))
                self.assertEqual(p.test, ())

    def test_la_temporal_tambien_es_la_MISMA_del_57_C3(self):
        for n in range(0, 501):
            for ratio in (0.5, 0.7, 0.8):
                with self.subTest(n=n, ratio=ratio):
                    viejo = (max(1, min(n - 1, int(n * ratio))) if n > 1 else n)
                    p = particion_temporal(n, ratio)
                    self.assertEqual(p.train, tuple(range(viejo)))
                    self.assertEqual(p.validation, tuple(range(viejo, n)))

    def test_un_mxtrain_de_antes_se_serializa_IGUAL(self):
        """Sin `test` ni `protocol`, el diccionario no gana ni una clave.

        Mismo criterio que usó el 57-C3 con `mode`: lo nuevo se declara, y lo
        que no lo declara no se entera — ni siquiera en su serialización, que
        es lo que viaja al manifiesto y de lo que cuelgan los digests.
        """
        spec = parse_training_text(_mxtrain("SPLIT train=0.8 validation=0.2 seed=42"))
        self.assertEqual(spec.dataset.split.to_dict(),
                         {"train": 0.8, "validation": 0.2, "seed": 42})

    def test_y_sin_protocolo_el_entrenador_parte_como_siempre(self):
        spec = parse_training_text(_mxtrain("SPLIT train=0.6 validation=0.4 seed=7"))
        p = particion_para(100, spec.dataset.split)
        self.assertEqual(p.protocolo, "legado")
        self.assertEqual(p.train, tuple(range(80)))   # el 0,8 fijo, no el 0,6
        self.assertEqual(p.test, ())


class LoQueSE_DECLARA_SE_HONRATest(unittest.TestCase):

    def test_con_protocolo_la_semilla_MANDA(self):
        """Sin esto, `seed=42` era decorativo: dos semillas, una partición."""
        a = particion_declarada(100, train=0.6, validation=0.2, test=0.2, seed=1)
        b = particion_declarada(100, train=0.6, validation=0.2, test=0.2, seed=2)
        self.assertNotEqual(a.train, b.train)
        self.assertEqual(sorted(a.train + a.validation + a.test), list(range(100)))

    def test_la_misma_semilla_reproduce_los_MISMOS_INDICES(self):
        """Criterio de cierre: «el mismo manifiesto reproduce índices»."""
        a = particion_declarada(250, train=0.6, validation=0.2, test=0.2, seed=99)
        b = particion_declarada(250, train=0.6, validation=0.2, test=0.2, seed=99)
        self.assertEqual((a.train, a.validation, a.test), (b.train, b.validation, b.test))

    def test_los_tres_tramos_no_comparten_NI_UNA_fila(self):
        """Se comprueban CONJUNTOS, no hashes: dos listas distintas pueden
        traer las mismas filas, y comparar hashes lo daría por bueno."""
        for n in (7, 50, 501, 1000):
            for seed in (None, 3):
                with self.subTest(n=n, seed=seed):
                    p = particion_declarada(n, train=0.6, validation=0.2,
                                            test=0.2, seed=seed)
                    self.assertTrue(p.sin_solape())
                    self.assertEqual(len(p.train) + len(p.validation) + len(p.test), n)

    def test_temporal_con_prueba_NO_baraja_y_deja_el_futuro_al_final(self):
        p = particion_declarada(100, train=0.6, validation=0.2, test=0.2,
                                modo="temporal")
        self.assertEqual(p.train, tuple(range(60)))
        self.assertEqual(p.validation, tuple(range(60, 80)))
        self.assertEqual(p.test, tuple(range(80, 100)))

    def test_sin_semilla_no_se_baraja_con_una_inventada(self):
        """Barajar con una semilla que nadie escribió haría el resultado
        irreproducible, y el manifiesto no podría declarar con cuál se hizo."""
        p = particion_declarada(50, train=0.6, validation=0.2, test=0.2, seed=None)
        self.assertEqual(p.train, tuple(range(30)))
        self.assertIsNone(p.semilla)

    def test_la_particion_DICE_con_que_protocolo_se_hizo(self):
        """Un número medido sobre una partición legada y otro sobre una con
        prueba reservada no son comparables; quien lea el manifiesto tiene que
        poder saber cuál es cuál."""
        self.assertEqual(particion_legada(10).como_dict()["protocol"], "legado")
        d = particion_declarada(10, train=0.6, validation=0.2, test=0.2,
                                seed=5).como_dict()
        self.assertEqual(d["protocol"], PROTOCOLO_SEPARACION)
        self.assertEqual(d["seed"], 5)
        self.assertIn("n_test", d)


class UnTramoQueNadieHonraNoSeEscribeTest(unittest.TestCase):
    """Lo que no se puede cumplir se rechaza, no se acepta y se ignora."""

    def test_test_sin_protocolo_se_RECHAZA(self):
        with self.assertRaises(MatrixAITrainingParseError) as e:
            parse_training_text(_mxtrain("SPLIT train=0.6 validation=0.2 test=0.2"))
        self.assertIn("se quedaría escrito sin", str(e.exception))

    def test_los_tres_tramos_tienen_que_sumar_uno(self):
        with self.assertRaises(MatrixAITrainingParseError) as e:
            parse_training_text(
                _mxtrain("SPLIT train=0.6 validation=0.2 test=0.3 protocol=2"))
        self.assertIn("debe sumar 1.0", str(e.exception))
        self.assertIn("test=0.3", str(e.exception))

    def test_un_protocolo_que_no_existe_se_dice(self):
        with self.assertRaises(MatrixAITrainingParseError) as e:
            parse_training_text(_mxtrain("SPLIT train=0.8 validation=0.2 protocol=9"))
        self.assertIn("no existe", str(e.exception))

    def test_y_dos_tramos_con_protocolo_siguen_valiendo(self):
        """El protocolo no obliga a tener prueba: obliga a honrar lo escrito."""
        spec = parse_training_text(
            _mxtrain("SPLIT train=0.7 validation=0.3 seed=4 protocol=2"))
        p = particion_para(100, spec.dataset.split)
        self.assertEqual(p.protocolo, PROTOCOLO_SEPARACION)
        self.assertEqual(p.test, ())
        self.assertEqual(len(p.train), 70)


class ElCanarioDeLaPruebaTest(unittest.TestCase):
    """**Cambiar solo el objetivo de las filas de PRUEBA no cambia el modelo.**

    Es el criterio de cierre del corte y la única forma de comprobar de verdad
    que la prueba no entra en el entrenamiento: si el modelo cambia, está
    entrando por algún sitio —el corte, la elección de época, un rango
    aprendido— y ninguna otra prueba lo vería.
    """

    def _entrenar(self, objetivos_de_prueba: str) -> tuple[str, int]:
        """Entrena con el MISMO desarrollo y distinto objetivo en la prueba."""
        import hashlib
        import json

        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        from matrixai.training.parser import parse_training_file

        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "model.mxai").write_text("""
PROJECT Canario

VECTOR V[2]
  x1: Scalar
  x2: Scalar
END

NETWORK Net
  INPUT V
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END

GRAPH
  V -> Net
END
""", encoding="utf-8")
            filas = ["x1,x2,y"]
            for i in range(40):                       # desarrollo: NO cambia
                filas.append(f"{i},{40 - i},{i % 5}")
            for i, y in enumerate(objetivos_de_prueba):    # prueba: lo que cambia
                filas.append(f"{100 + i},{i},{y}")
            (d / "data.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
            (d / "train.mxtrain").write_text("""MODEL model.mxai

DATASET TrainData
  SOURCE csv("data.csv")
  INPUT V FROM COLUMNS [x1, x2]
  TARGET y: Scalar
  SPLIT train=0.6 validation=0.2 test=0.2 protocol=2
END

LOSS NetLoss
  TYPE mse
  PREDICTION y
  TARGET y
END

OPTIMIZER NetOpt
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END

RUN
  EPOCHS 3
END
""", encoding="utf-8")
            spec = parse_training_file(d / "train.mxtrain")
            res = DenseSupervisedTrainer().train(spec, output_dir=str(d / "out"),
                                                 base_path=d)
            # Se compara el ARTEFACTO escrito, que es lo que se publica y lo
            # que viaja al paquete: si los pesos cambiasen, cambiaría aquí.
            pesos = sorted((d / "out").glob("*.json"))
            firma = hashlib.sha256()
            for f in pesos:
                if f.name.endswith("trace.json"):
                    continue          # la traza lleva tiempos: cambia siempre
                firma.update(f.name.encode())
                firma.update(json.dumps(json.loads(f.read_text(encoding="utf-8")),
                                        sort_keys=True).encode())
            self.assertTrue(pesos, "el entrenador no escribió ningún artefacto")
            return firma.hexdigest(), res.best_epoch

    def test_cambiar_el_objetivo_de_la_PRUEBA_no_cambia_el_modelo(self):
        a, epoca_a = self._entrenar("1111111111")
        b, epoca_b = self._entrenar("9999999999")
        self.assertEqual(a, b, "el modelo cambió al tocar SOLO las filas de prueba: "
                               "la prueba está entrando en el entrenamiento")
        self.assertEqual(epoca_a, epoca_b, "la época elegida cambió al tocar SOLO "
                                           "las filas de prueba")


class TestElCaminoTorchTambienHonraElProtocolo:
    """Lo mismo que el canario stdlib, pero por el camino torch.

    `playground.py` tiene SU PROPIA copia de la decisión de partición para el
    entrenador torch (no pasa por `DenseSupervisedTrainer`), y un sabotaje que
    la desconectaba **no ponía nada rojo** en el resto de este fichero: el
    canario de arriba solo entrena por stdlib. Sin esta clase, el camino torch
    quedaría "arreglado" sin ninguna prueba que lo sostenga — el mismo defecto
    de cobertura que ya costó una vez en el 102-C0.

    Reutiliza el patrón de espía de `test_biblioteca_c3_split_temporal.py`
    (`TestDenseTorchStudioPathHonorsTemporalSplit`): capturar
    `validation_examples` tal como se le pasan a `train_dense_network_torch`.
    """

    _MXAI = """PROJECT DT
VECTOR In[2]
  x1: Scalar
  x2: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=4 activation=relu
  LAYER Dense units=1 activation=linear
  OUTPUT y: Scalar
END
GRAPH
  In -> Net
END
"""

    @staticmethod
    def _csv() -> str:
        # x1 = índice de fila: identifica EXACTAMENTE qué filas terminan en
        # cada tramo mirando los x1 vistos, sin depender del contenido.
        filas = ["x1,x2,y"] + [f"{i},0,{i}" for i in range(10)]
        return "\n".join(filas) + "\n"

    @classmethod
    def _mxtrain(cls, split_line: str) -> str:
        return f"""MODEL DT.mxai
DATASET DS
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [x1, x2]
  TARGET y: Scalar
  {split_line}
END
LOSS L
  TYPE mse
  PREDICTION Net
  TARGET y
END
OPTIMIZER O
  TYPE sgd
  LEARNING_RATE 0.01
  UPDATE Net.*
END
RUN
  EPOCHS 2
END
"""

    def _entrenar_y_capturar(self, split_line: str, monkeypatch):
        import matrixai.training.dense_torch_trainer as dtt
        from matrixai.playground import _run_playground_dense_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
        capturado: dict = {}
        real = dtt.train_dense_network_torch

        def _espia(*args, **kwargs):
            capturado["train_examples"] = args[2] if len(args) > 2 else kwargs.get("examples")
            capturado["validation_examples"] = kwargs.get("validation_examples")
            return real(*args, **kwargs)

        monkeypatch.setattr(dtt, "train_dense_network_torch", _espia)
        r = _run_playground_dense_training(self._MXAI, self._mxtrain(split_line),
                                           self._csv(), epochs_override=2)
        assert r["ok"], r.get("error")
        return capturado

    def test_sin_protocolo_sigue_siendo_el_080_FIJO_de_siempre(self, monkeypatch):
        """Nada declarado → el 0,8 fijo secuencial de antes de C3, byte a
        byte: MISMO criterio que ya probaba `test_mode_random_matches_old_
        hardcoded_80_20_ignoring_custom_ratio` en el fichero del 57-C3."""
        c = self._entrenar_y_capturar("SPLIT train=0.6 validation=0.4", monkeypatch)
        val_x1 = sorted(int(x[0]) for x, _y in c["validation_examples"])
        assert val_x1 == [8, 9]

    def test_CON_protocolo_2_la_semilla_1_da_la_particion_DECLARADA(self, monkeypatch):
        """Es el defecto que este corte arregla, medido por este camino
        exacto: `seed=` no mandaba nada en el entrenador torch tampoco.

        Cada semilla en SU PROPIO test —con `monkeypatch` fresco de pytest—
        y no dos entrenamientos en el mismo test compartiendo el espía: al
        no deshacerse el `setattr` entre dos llamadas del MISMO test, el
        segundo espía envuelve al primero y el resultado deja de ser fiable.
        Medido: reproducirlo así daba `[2, 8]` para las DOS semillas, aunque
        `particion_declarada` —la función pura, sin entrenador de por
        medio— ya daba `(0, 4)` y `(2, 8)` por separado. Era el arnés, no el
        entrenador; separar los tests lo confirma comparando cada uno contra
        la MISMA función pura que usa `dense_trainer.py`.
        """
        from matrixai.training.particion import particion_declarada
        c = self._entrenar_y_capturar(
            "SPLIT train=0.6 validation=0.2 test=0.2 seed=1 protocol=2", monkeypatch)
        val = sorted(int(x[0]) for x, _y in c["validation_examples"])
        esperado = sorted(particion_declarada(10, train=0.6, validation=0.2,
                                              test=0.2, seed=1).validation)
        assert val == esperado

    def test_CON_protocolo_2_la_semilla_2_da_OTRA_particion(self, monkeypatch):
        from matrixai.training.particion import particion_declarada
        c = self._entrenar_y_capturar(
            "SPLIT train=0.6 validation=0.2 test=0.2 seed=2 protocol=2", monkeypatch)
        val = sorted(int(x[0]) for x, _y in c["validation_examples"])
        esperado = sorted(particion_declarada(10, train=0.6, validation=0.2,
                                              test=0.2, seed=2).validation)
        assert val == esperado
        # Y sigue siendo el criterio central: no la misma que la semilla 1.
        assert val != sorted(particion_declarada(
            10, train=0.6, validation=0.2, test=0.2, seed=1).validation)

    def test_CON_protocolo_2_el_tramo_de_TEST_no_entra_en_train_ni_en_val(self, monkeypatch):
        c = self._entrenar_y_capturar(
            "SPLIT train=0.6 validation=0.2 test=0.2 seed=7 protocol=2", monkeypatch)
        train_x1 = {int(x[0]) for x, _y in c["train_examples"]}
        val_x1 = {int(x[0]) for x, _y in c["validation_examples"]}
        assert len(train_x1) == 6
        assert len(val_x1) == 2
        assert train_x1.isdisjoint(val_x1)
        # Las dos filas que faltan (10 - 6 - 2 = 2) son el tramo de prueba:
        # se apartaron, no se perdieron ni se colaron en train o val.
        assert len(train_x1 | val_x1) == 8


if __name__ == "__main__":
    unittest.main()
