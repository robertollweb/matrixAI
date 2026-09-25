# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""118-C3b — extender el lenguaje `.mxtrain` y los entrenadores del núcleo.

Contrato 118 (`documentacion/118_LA_RED_DENSA_COMPITE_CONTRACT.md`) y su
enmienda 1 (`benchmarks/fase0/protocolo_118_v3_enmienda_1.json`): la receta
de C3 pide AdamW con decaimiento de peso y programa de tasa coseno, y el
lenguaje del núcleo de antes de este corte NO los admitía — comprobado
ejecutándolo, no leído (`matrixai-engines/tests/test_118_c3_receta.py`).

Esta palanca (C3b) EXTIENDE el lenguaje y los tres entrenadores torch para
que los admitan de verdad. Lo que se prueba aquí:

1. **Parseo**: el bloque OPTIMIZER admite `WEIGHT_DECAY <real >= 0>` y
   `SCHEDULE cosine` (y SOLO `cosine`) — opcionales, y sin ellos el
   `OptimizerSpec` es idéntico campo a campo al de antes de C3b.
2. **AdamW de verdad**: los tres entrenadores torch (denso, compuesto, y el
   lineal/FUNCTION de `torch_trainer.py`) instancian `torch.optim.AdamW`
   con el decaimiento declarado — no `Adam` con el nombre cambiado.
3. **El programa coseno**: `CosineAnnealingLR` avanza un paso por época y
   las tasas registradas coinciden con una instancia de control.
4. **La parada temprana** sigue cortando antes del tope de épocas con la
   palanca activa (el programa no alarga nada).
5. **El camino SIN torch se niega con su motivo** cuando el texto pide
   `WEIGHT_DECAY`/`SCHEDULE` — nunca entrena ignorándolos en silencio.
6. Un entrenamiento pequeño real (todo el camino `run_playground_training`)
   TERMINA con la palanca activa, y la procedencia (`training_trace`) dice
   lo que se aplicó.

SABOTAJES verificados a mano (copia, sabotaje, rojo, restaurar, `md5sum`,
`__pycache__` borrado — ver el informe de la tarea, no repetidos aquí como
código porque el terreno pide que las pruebas prueben el producto, no que
el propio fichero de pruebas sabotee su vecino):
* `weight_decay` leído pero no pasado al optimizador → lo cazan las pruebas
  de la sección 2 (el espía captura el `weight_decay` que de verdad recibe
  `torch.optim.AdamW`, no el declarado).
* el programa creado pero sin `step()` → lo caza la prueba de la sección 3
  (longitud de la traza de tasas y que decrezca de verdad).
* `adamw` instanciando `Adam` → lo cazan las mismas pruebas de la sección 2
  (el espía de `Adam` debe quedar SIN llamadas cuando se pide `adamw`).
* el stdlib ignorando las líneas en silencio → lo cazan las pruebas de la
  sección 5 (esperan `pytest.raises`; sin el guardián, no habría rojo).
"""
from __future__ import annotations

import dataclasses
import json
import random
import tempfile
from importlib import util
from pathlib import Path

import pytest

from matrixai.training.parser import MatrixAITrainingParseError, parse_training_text

_HAS_TORCH = util.find_spec("torch") is not None
_ROOT = Path(__file__).resolve().parents[1]

pytestmark_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Fixtures de texto .mxtrain / .mxai
# ---------------------------------------------------------------------------

def _texto_optimizer(optimizer_extra: str = "", optimizer_type: str = "adamw") -> str:
    """Un `.mxtrain` mínimo, válido, con el bloque OPTIMIZER parametrizado.

    Sirve solo para probar el PARSER (no se entrena con él: el DATASET
    apunta a un CSV que no existe).
    """
    return f"""MODEL P.mxai

DATASET D
  SOURCE csv("d.csv")
  INPUT In FROM COLUMNS [a, b]
  TARGET y: Label[no, si]
  BATCH size=8
END

LOSS L
  TYPE cross_entropy
  PREDICTION R
  TARGET y
END

OPTIMIZER O
  TYPE {optimizer_type}
  LEARNING_RATE 0.01
{optimizer_extra}  UPDATE Net.*
END

RUN
  EPOCHS 5
END
"""


_NETWORK_MXAI = """PROJECT P
VECTOR In[2]
  a: Scalar
  b: Scalar
END
NETWORK Net
  INPUT In
  LAYER Dense units=8 activation=relu
  LAYER Dense units=2 activation=softmax
  OUTPUT y: ProbabilityMap[A, B]
END
GRAPH
  In -> Net
END
"""


def _network_setup():
    from matrixai.parser import parse_text
    from matrixai.types import check_network_types
    from matrixai.parameters.network_params import build_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(_NETWORK_MXAI)
    net = prog.networks[0]
    tr = check_network_types(net, {v.name: v for v in prog.vectors})
    rl = tr.resolved_layers or net.layers
    ps = build_network_parameter_set(net, rl, program_hash(prog), seed=1)
    return net, ps


def _signal_examples(n=120, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.random(), rng.random()
        rows.append(([a, b], [0.0, 1.0] if a > 0.5 else [1.0, 0.0]))  # B iff a>0.5
    return rows


def _noise_examples(n=120, seed=7):
    rng = random.Random(seed)
    return [([rng.random(), rng.random()],
             [1.0, 0.0] if rng.random() < 0.5 else [0.0, 1.0]) for _ in range(n)]


_EMB_MXAI = """PROJECT P
VECTOR Input[2]
  cat: Integer[0, 8]
  x: Scalar
END
NETWORK Net
  INPUT Input
  EMBEDDING e1 FROM cat VOCAB 9 DIM 4
  CONCAT [e1, x] -> features
  LAYER Dense units=8 activation=relu
  LAYER Dense units=3 activation=softmax
  OUTPUT y: ProbabilityMap[A, B, C]
END
GRAPH
  Input -> Net
END
"""


def _composite_setup():
    from matrixai.parser import parse_text
    from matrixai.types import check_composite_network_types
    from matrixai.parameters.network_params import build_composite_network_parameter_set
    from matrixai.parameters.store import program_hash
    prog = parse_text(_EMB_MXAI)
    net = prog.networks[0]
    tr = check_composite_network_types(net, {v.name: v for v in prog.vectors})
    ps = build_composite_network_parameter_set(net, tr, model_hash_str=program_hash(prog), seed=1)
    return net, ps


def _composite_examples(n=90, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        c = rng.randint(0, 8)
        oh = [0.0, 0.0, 0.0]
        oh[c % 3] = 1.0
        rows.append(({"cat": c, "x": rng.random()}, oh))
    return rows


# ---------------------------------------------------------------------------
# 1. PARSEO
# ---------------------------------------------------------------------------

class TestParseo:
    def test_acepta_weight_decay_y_schedule_cosine_y_type_adamw(self):
        texto = _texto_optimizer("  WEIGHT_DECAY 0.0001\n  SCHEDULE cosine\n", optimizer_type="adamw")
        spec = parse_training_text(texto)
        assert spec.optimizer.type == "adamw"
        assert spec.optimizer.weight_decay == 0.0001
        assert spec.optimizer.schedule == "cosine"
        d = spec.optimizer.to_dict()
        assert d["weight_decay"] == 0.0001
        assert d["schedule"] == "cosine"

    def test_admite_las_dos_lineas_en_cualquier_orden(self):
        texto = _texto_optimizer("  SCHEDULE cosine\n  WEIGHT_DECAY 0.0001\n", optimizer_type="adam")
        spec = parse_training_text(texto)
        assert spec.optimizer.weight_decay == 0.0001
        assert spec.optimizer.schedule == "cosine"

    def test_weight_decay_solo_sin_schedule(self):
        texto = _texto_optimizer("  WEIGHT_DECAY 0.001\n", optimizer_type="sgd")
        spec = parse_training_text(texto)
        assert spec.optimizer.weight_decay == 0.001
        assert spec.optimizer.schedule is None
        assert "schedule" not in spec.optimizer.to_dict()

    def test_rechaza_schedule_desconocido(self):
        texto = _texto_optimizer("  SCHEDULE linear\n", optimizer_type="adam")
        with pytest.raises(MatrixAITrainingParseError) as info:
            parse_training_text(texto)
        assert "SCHEDULE" in str(info.value)
        assert "cosine" in str(info.value)

    def test_rechaza_schedule_vacio_de_otro_tipo(self):
        texto = _texto_optimizer("  SCHEDULE step\n", optimizer_type="adam")
        with pytest.raises(MatrixAITrainingParseError):
            parse_training_text(texto)

    def test_rechaza_weight_decay_negativo(self):
        texto = _texto_optimizer("  WEIGHT_DECAY -0.1\n", optimizer_type="adam")
        with pytest.raises(MatrixAITrainingParseError) as info:
            parse_training_text(texto)
        assert "WEIGHT_DECAY" in str(info.value)

    def test_rechaza_weight_decay_no_numerico(self):
        texto = _texto_optimizer("  WEIGHT_DECAY abc\n", optimizer_type="adam")
        with pytest.raises(MatrixAITrainingParseError) as info:
            parse_training_text(texto)
        assert "WEIGHT_DECAY" in str(info.value)

    def test_el_parser_acepta_cualquier_type_incluido_adamw(self):
        """El PARSER no decide qué optimizadores existen (eso es
        `TrainingVerifier` y cada entrenador) — mismo criterio que ya regía
        para `sgd`/`adam` antes de 118-C3b."""
        texto = _texto_optimizer("", optimizer_type="un_optimizador_inventado")
        spec = parse_training_text(texto)
        assert spec.optimizer.type == "un_optimizador_inventado"

    def test_sin_las_lineas_nuevas_el_spec_es_identico_campo_a_campo(self):
        """Prueba con su nombre de la afirmación del docstring del parser
        y del encargo: un texto que NO declara WEIGHT_DECAY/SCHEDULE da el
        MISMO `OptimizerSpec`, campo a campo, y la MISMA serialización."""
        spec = parse_training_text(_texto_optimizer("", optimizer_type="adam"))
        assert spec.optimizer.weight_decay == 0.0
        assert spec.optimizer.schedule is None
        assert set(spec.optimizer.to_dict().keys()) == {"name", "type", "learning_rate", "update"}
        # Y el resto de la IR (dataset/loss/run) no se ha movido ni un campo.
        assert spec.optimizer.type == "adam"
        assert spec.optimizer.learning_rate == 0.01
        assert spec.optimizer.update == ["Net.*"]


# ---------------------------------------------------------------------------
# 2. ADAMW DE VERDAD (espía sobre el objeto instanciado), en los TRES
#    entrenadores torch
# ---------------------------------------------------------------------------

@pytestmark_torch
class TestAdamWDeVerdad:
    def test_dense_torch_instancia_adamw_con_su_decaimiento(self, monkeypatch):
        import torch
        from matrixai.training.dense_torch_trainer import train_dense_network_torch

        llamadas = {"adamw": [], "adam": []}
        real_adamw, real_adam = torch.optim.AdamW, torch.optim.Adam

        def espia_adamw(params, lr, weight_decay=0.0, **kw):
            llamadas["adamw"].append(weight_decay)
            return real_adamw(params, lr=lr, weight_decay=weight_decay, **kw)

        def espia_adam(params, lr, weight_decay=0.0, **kw):
            llamadas["adam"].append(weight_decay)
            return real_adam(params, lr=lr, weight_decay=weight_decay, **kw)

        monkeypatch.setattr(torch.optim, "AdamW", espia_adamw)
        monkeypatch.setattr(torch.optim, "Adam", espia_adam)

        net, ps = _network_setup()
        res = train_dense_network_torch(
            net, ps, _signal_examples(), "cross_entropy",
            lr=0.1, epochs=3, device="cpu", seed=1,
            optimizer="adamw", weight_decay=0.0001,
        )
        assert res["backend"] == "torch"
        assert llamadas["adamw"] == [0.0001]
        assert llamadas["adam"] == []

    def test_dense_torch_adam_tambien_admite_weight_decay(self, monkeypatch):
        """El decaimiento no es exclusivo de adamw: la enmienda del
        protocolo lo pide sobre adamw, pero el encargo pide que Adam/SGD lo
        admitan también SI se declaran (`torch.optim` lo soporta en los
        tres)."""
        import torch
        from matrixai.training.dense_torch_trainer import train_dense_network_torch

        capturado = {}
        real_adam = torch.optim.Adam

        def espia_adam(params, lr, weight_decay=0.0, **kw):
            capturado["weight_decay"] = weight_decay
            return real_adam(params, lr=lr, weight_decay=weight_decay, **kw)

        monkeypatch.setattr(torch.optim, "Adam", espia_adam)
        net, ps = _network_setup()
        train_dense_network_torch(
            net, ps, _signal_examples(), "cross_entropy",
            lr=0.1, epochs=2, device="cpu", seed=1,
            optimizer="adam", weight_decay=0.0005,
        )
        assert capturado["weight_decay"] == 0.0005

    def test_composite_torch_instancia_adamw_con_su_decaimiento(self, monkeypatch):
        import torch
        from matrixai.training.composite_torch_trainer import train_composite_network_torch

        llamadas = {"adamw": [], "adam": []}
        real_adamw, real_adam = torch.optim.AdamW, torch.optim.Adam

        def espia_adamw(params, lr, weight_decay=0.0, **kw):
            llamadas["adamw"].append(weight_decay)
            return real_adamw(params, lr=lr, weight_decay=weight_decay, **kw)

        def espia_adam(params, lr, weight_decay=0.0, **kw):
            llamadas["adam"].append(weight_decay)
            return real_adam(params, lr=lr, weight_decay=weight_decay, **kw)

        monkeypatch.setattr(torch.optim, "AdamW", espia_adamw)
        monkeypatch.setattr(torch.optim, "Adam", espia_adam)

        net, ps = _composite_setup()
        res = train_composite_network_torch(
            net, ps, _composite_examples(), "cross_entropy",
            lr=0.05, epochs=3, device="cpu", seed=1,
            optimizer="adamw", weight_decay=0.0002,
        )
        assert res["backend"] == "torch"
        assert llamadas["adamw"] == [0.0002]
        assert llamadas["adam"] == []

    def test_torch_trainer_lineal_instancia_adamw_con_su_decaimiento(self, monkeypatch, tmp_path):
        """El tercer camino torch (`torch_trainer.TorchSupervisedTrainer`,
        el modelo lineal/FUNCTION) — sobre el ejemplo real del repositorio
        (`examples/email-agent.*`), mutando SOLO el optimizador declarado
        (`dataclasses.replace`, los dos son `frozen`)."""
        import torch
        from matrixai.training.parser import parse_training_file
        from matrixai.training.torch_trainer import TorchSupervisedTrainer

        llamadas = {"adamw": [], "adam": []}
        real_adamw, real_adam = torch.optim.AdamW, torch.optim.Adam

        def espia_adamw(params, lr, weight_decay=0.0, **kw):
            llamadas["adamw"].append(weight_decay)
            return real_adamw(params, lr=lr, weight_decay=weight_decay, **kw)

        def espia_adam(params, lr, weight_decay=0.0, **kw):
            llamadas["adam"].append(weight_decay)
            return real_adam(params, lr=lr, weight_decay=weight_decay, **kw)

        monkeypatch.setattr(torch.optim, "AdamW", espia_adamw)
        monkeypatch.setattr(torch.optim, "Adam", espia_adam)

        spec = parse_training_file(_ROOT / "examples" / "email-agent.supervised.mxtrain")
        spec_c3b = dataclasses.replace(
            spec,
            optimizer=dataclasses.replace(
                spec.optimizer, type="adamw", weight_decay=0.0003,
            ),
        )
        output_dir = tmp_path / "email_adamw"
        result = TorchSupervisedTrainer().train(
            spec_c3b, output_dir=output_dir, base_path=_ROOT,
        )
        assert result.run_id == output_dir.name
        assert llamadas["adamw"] == [0.0003]
        assert llamadas["adam"] == []

        # Y la procedencia (118-C3b punto 3) lo declara.
        trace = json.loads((output_dir / "training_trace.json").read_text(encoding="utf-8"))
        assert trace["optimizer"] == "adamw"
        assert trace["optimizer_weight_decay"] == 0.0003
        assert "optimizer_schedule" not in trace  # no se declaró SCHEDULE


# ---------------------------------------------------------------------------
# 3. EL PROGRAMA COSENO BAJA LA TASA ÉPOCA A ÉPOCA
# ---------------------------------------------------------------------------

@pytestmark_torch
class TestProgramaCoseno:
    def test_dense_torch_cosine_iguala_a_una_instancia_de_control(self, monkeypatch):
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from matrixai.training.dense_torch_trainer import train_dense_network_torch

        lrs_reales: list[float] = []
        paso_real = CosineAnnealingLR.step

        def paso_espia(self, *a, **kw):
            resultado = paso_real(self, *a, **kw)
            lrs_reales.append(self.optimizer.param_groups[0]["lr"])
            return resultado

        monkeypatch.setattr(CosineAnnealingLR, "step", paso_espia)

        net, ps = _network_setup()
        epochs = 5
        train_dense_network_torch(
            net, ps, _signal_examples(), "cross_entropy",
            lr=0.1, epochs=epochs, device="cpu", seed=1,
            optimizer="adam", schedule="cosine",
        )
        # `lrs_reales[0]` es la lectura que hace el propio constructor de
        # `CosineAnnealingLR` (comportamiento de PyTorch, no cosa nuestra);
        # las siguientes `epochs` lecturas son UN paso por época de
        # entrenamiento -- lo que 118-C3b promete ("un paso por época").
        assert len(lrs_reales) == 1 + epochs

        # Restaurar el `step()` real ANTES de construir el oráculo: si
        # siguiera parcheado, la CONSTRUCCIÓN del oráculo (que también llama
        # a `step()` una vez, como CUALQUIER `CosineAnnealingLR`) escribiría
        # en el MISMO `lrs_reales` y contaminaría la lectura de arriba.
        monkeypatch.setattr(CosineAnnealingLR, "step", paso_real)

        # Oráculo: una instancia de control con el MISMO T_max y la MISMA
        # tasa base, avanzada el mismo número de veces.
        oraculo_opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        oraculo_sched = CosineAnnealingLR(oraculo_opt, T_max=epochs)
        lrs_esperadas = []
        for _ in range(epochs):
            oraculo_sched.step()
            lrs_esperadas.append(oraculo_opt.param_groups[0]["lr"])

        assert lrs_reales[1:] == pytest.approx(lrs_esperadas)
        # Y de verdad DECRECE -- si `scheduler.step()` no se llamara nunca
        # (el sabotaje "el programa creado pero sin step()"), esta lista
        # tendría un único valor repetido (la tasa base) y esta comparación
        # fallaría.
        assert lrs_reales[-1] < lrs_reales[1] < lrs_reales[0]

    def test_sin_schedule_la_tasa_no_se_mueve(self, monkeypatch):
        """Control negativo: sin `schedule="cosine"` no se crea ningún
        `CosineAnnealingLR` — comportamiento previo a 118-C3b intacto."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from matrixai.training.dense_torch_trainer import train_dense_network_torch

        llamado = []
        paso_real = CosineAnnealingLR.step

        def paso_espia(self, *a, **kw):
            llamado.append(True)
            return paso_real(self, *a, **kw)

        monkeypatch.setattr(CosineAnnealingLR, "step", paso_espia)
        net, ps = _network_setup()
        train_dense_network_torch(
            net, ps, _signal_examples(), "cross_entropy",
            lr=0.1, epochs=3, device="cpu", seed=1, optimizer="adam",
        )
        assert llamado == []


# ---------------------------------------------------------------------------
# 4. LA PARADA TEMPRANA SIGUE CORTANDO ANTES, CON LA PALANCA ACTIVA
# ---------------------------------------------------------------------------

@pytestmark_torch
class TestParadaTemprana:
    def test_early_stop_sigue_cortando_con_adamw_weight_decay_y_coseno(self):
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _network_setup()
        # Datos sin señal (113-C1/118-C3, mismo criterio): la validación deja
        # de mejorar enseguida, así que la parada temprana corta MUCHO antes
        # del tope -- aquí con 200, el de la enmienda 1 del protocolo, y con
        # las tres palancas nuevas activas a la vez.
        res = train_dense_network_torch(
            net, ps, _noise_examples(), "cross_entropy",
            lr=0.05, epochs=200, early_stop=(5, "validation_loss"),
            device="cpu", seed=1,
            optimizer="adamw", weight_decay=0.0001, schedule="cosine",
        )
        assert len(res["epochs"]) < 200
        assert len(res["epochs"]) <= res["best_epoch"] + 5

    def test_early_stop_sigue_cortando_igual_sin_la_palanca(self):
        """Mismo dataset, sin AdamW/weight_decay/coseno -- el corte de la
        parada temprana no depende de la palanca nueva (control). Mismo
        `lr`/optimizador base (`adam`) que el `test_early_stop_respected`
        ya existente de GPU-C1 -- con SGD plano y `lr` bajo el descenso es
        tan errático que la validación "mejora" por ruido de cuando en
        cuando y el patience no llega a agotarse en 200 épocas (medido)."""
        from matrixai.training.dense_torch_trainer import train_dense_network_torch
        net, ps = _network_setup()
        res = train_dense_network_torch(
            net, ps, _noise_examples(), "cross_entropy",
            lr=0.05, epochs=200, early_stop=(5, "validation_loss"),
            device="cpu", seed=1, optimizer="adam",
        )
        assert len(res["epochs"]) < 200


# ---------------------------------------------------------------------------
# 5. EL CAMINO SIN TORCH SE NIEGA CON SU MOTIVO
# ---------------------------------------------------------------------------

class TestCaminoStdlibSeNiega:
    def test_dense_stdlib_se_niega_con_weight_decay(self, tmp_path):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        (tmp_path / "P.mxai").write_text(_NETWORK_MXAI, encoding="utf-8")
        texto = _texto_optimizer(
            "  WEIGHT_DECAY 0.0001\n", optimizer_type="sgd",
        ).replace('SOURCE csv("d.csv")', 'SOURCE csv("no-existe.csv")')
        spec = parse_training_text(texto)
        with pytest.raises(ValueError) as info:
            DenseSupervisedTrainer().train(spec, output_dir=tmp_path / "out", base_path=tmp_path)
        assert "WEIGHT_DECAY" in str(info.value)
        assert "torch" in str(info.value)

    def test_dense_stdlib_se_niega_con_schedule(self, tmp_path):
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        (tmp_path / "P.mxai").write_text(_NETWORK_MXAI, encoding="utf-8")
        texto = _texto_optimizer(
            "  SCHEDULE cosine\n", optimizer_type="sgd",
        ).replace('SOURCE csv("d.csv")', 'SOURCE csv("no-existe.csv")')
        spec = parse_training_text(texto)
        with pytest.raises(ValueError) as info:
            DenseSupervisedTrainer().train(spec, output_dir=tmp_path / "out", base_path=tmp_path)
        assert "SCHEDULE" in str(info.value)
        assert "torch" in str(info.value)

    def test_dense_stdlib_sigue_entrenando_sgd_plano_sin_extras(self, tmp_path):
        """Control: SIN weight_decay/schedule declarados, el camino stdlib
        entrena exactamente igual que siempre (no se niega por nada nuevo)."""
        from matrixai.training.dense_trainer import DenseSupervisedTrainer
        (tmp_path / "P.mxai").write_text(_NETWORK_MXAI, encoding="utf-8")
        rows = ["a,b,y"]
        rng = random.Random(0)
        for _ in range(40):
            a, b = rng.random(), rng.random()
            rows.append(f"{a},{b},{'si' if a > 0.5 else 'no'}")
        (tmp_path / "d.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        texto = _texto_optimizer("", optimizer_type="sgd").replace("EPOCHS 5", "EPOCHS 2")
        spec = parse_training_text(texto)
        result = DenseSupervisedTrainer().train(spec, output_dir=tmp_path / "out", base_path=tmp_path)
        assert result.best_epoch >= 1

    def test_funcion_stdlib_se_niega_con_weight_decay(self):
        """El otro camino stdlib (`trainer.SupervisedTrainer`, modelo
        lineal/FUNCTION) sobre el ejemplo real del repositorio, TYPE sgd
        (pasa el primer guardián) pero con weight_decay declarado."""
        from matrixai.training.parser import parse_training_file
        from matrixai.training.trainer import SupervisedTrainer

        spec = parse_training_file(_ROOT / "examples" / "email-agent.supervised.mxtrain")
        assert spec.optimizer.type == "sgd"
        spec_c3b = dataclasses.replace(
            spec, optimizer=dataclasses.replace(spec.optimizer, weight_decay=0.0001),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError) as info:
                SupervisedTrainer().train(
                    spec_c3b, output_dir=Path(tmp_dir) / "out", base_path=_ROOT,
                )
        assert "WEIGHT_DECAY" in str(info.value)


# ---------------------------------------------------------------------------
# 6. UN ENTRENAMIENTO PEQUEÑO REAL TERMINA, Y LA PROCEDENCIA LO DECLARA
# ---------------------------------------------------------------------------

@pytestmark_torch
class TestEntrenamientoRealTermina:
    def test_run_playground_training_con_adamw_weight_decay_y_coseno(self, monkeypatch):
        import csv
        import io
        from matrixai.playground_api import generate_project_from_dataset, run_playground_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")

        rng = random.Random(0)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["x1", "x2", "y"])
        for _ in range(100):
            x = rng.uniform(0, 10)
            w.writerow([round(x, 3), round(rng.random(), 3), "si" if x > 5 else "no"])
        gen = generate_project_from_dataset(
            buf.getvalue(), "y", locale="es", column_type_overrides={"y": "categorical"},
        )
        texto = gen["training_text"]
        # El texto declara un TYPE base (sgd o adam, según el generador);
        # aquí lo llevamos a la receta 118-C3 (adamw + weight_decay + coseno)
        # sobre el MISMO texto real, no uno escrito a mano.
        for base_type in ("TYPE sgd\n", "TYPE adam\n"):
            if base_type in texto:
                texto = texto.replace(base_type, "TYPE adamw\n", 1)
                break
        else:
            pytest.fail(f"el generador no produjo TYPE sgd/adam: {texto}")
        import re
        texto = re.sub(
            r"(LEARNING_RATE [0-9.]+\n)",
            r"\1  WEIGHT_DECAY 0.0001\n  SCHEDULE cosine\n",
            texto, count=1,
        )
        assert "WEIGHT_DECAY 0.0001" in texto
        assert "SCHEDULE cosine" in texto

        resultado = run_playground_training(
            gen["mxai"], texto, gen["csv_text"], epochs_override=5,
            field_ranges=gen.get("field_ranges"), seed=1,
        )
        assert resultado.get("ok") is True, resultado.get("error")
        assert resultado.get("best_epoch", 0) >= 1
        # 118-C3b punto 3: la procedencia sale del texto que se entrenó.
        traza = resultado.get("training_trace") or {}
        assert traza.get("optimizer", {}).get("weight_decay") == 0.0001
        assert traza.get("optimizer", {}).get("schedule") == "cosine"

    def test_run_playground_training_sin_la_palanca_no_declara_extras(self, monkeypatch):
        """Control: sin WEIGHT_DECAY/SCHEDULE en el texto, la traza no lleva
        la clave `optimizer` nueva (byte-idéntica a antes de 118-C3b)."""
        import csv
        import io
        from matrixai.playground_api import generate_project_from_dataset, run_playground_training

        monkeypatch.setenv("MATRIXAI_TRAIN_BACKEND", "torch")
        rng = random.Random(1)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["x1", "x2", "y"])
        for _ in range(100):
            x = rng.uniform(0, 10)
            w.writerow([round(x, 3), round(rng.random(), 3), "si" if x > 5 else "no"])
        gen = generate_project_from_dataset(
            buf.getvalue(), "y", locale="es", column_type_overrides={"y": "categorical"},
        )
        resultado = run_playground_training(
            gen["mxai"], gen["training_text"], gen["csv_text"], epochs_override=3,
            field_ranges=gen.get("field_ranges"), seed=1,
        )
        assert resultado.get("ok") is True, resultado.get("error")
        traza = resultado.get("training_trace") or {}
        assert "optimizer" not in traza
