#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Demo/validación GPU del playground MatrixAI para Colab (sin Docker).

Autónomo: se añade su propio directorio al sys.path, así `import matrixai` funciona
desde la fuente extraída sin pip ni estado del notebook. Ejecutar con:

    !python /content/matrixai_src/tools/colab_gpu_demo.py --width 4096 --depth 16 --rows 60000 --epochs 180

Genera un dataset (M10), entrena y evalúa (M14/M15) por el backend torch (CUDA si hay
GPU), e imprime backend, métricas, tiempos, VRAM EN VIVO por época y VRAM pico. Pensado
para exigir a la GPU varios minutos (entrenamiento full-batch sobre red ancha+profunda).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# El paquete vive en la raíz del tar extraído: <este_dir>/.. = matrixai_src/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def build_specs(width: int, depth: int, n_feats: int, classes: list[str]):
    feats = [f"f{i}" for i in range(n_feats)]
    vec = "\n".join(f"  {f}: Scalar" for f in feats)
    layers = "\n".join(f"  LAYER Dense units={width} activation=relu" for _ in range(depth))
    mxai = (
        f"PROJECT P\nVECTOR In[{len(feats)}]\n{vec}\nEND\n"
        f"NETWORK Net\n  INPUT In\n{layers}\n"
        f"  LAYER Dense units={len(classes)} activation=softmax\n"
        f"  OUTPUT y: ProbabilityMap[{', '.join(classes)}]\nEND\n"
        f"GRAPH\n  In -> Net\nEND\n"
    )
    cols = "[" + ", ".join(feats) + "]"
    mxtrain = (
        f"MODEL P.mxai\nDATASET D\n  SOURCE csv(\"d.csv\")\n"
        f"  INPUT In FROM COLUMNS {cols}\n  TARGET y: Label[{', '.join(classes)}]\n"
        f"  SPLIT train=0.8 validation=0.2 seed=42\n  BATCH size=128\nEND\n"
        f"LOSS L\n  TYPE cross_entropy\n  PREDICTION Net\n  TARGET y\nEND\n"
        f"OPTIMIZER O\n  TYPE sgd\n  LEARNING_RATE 0.1\n  UPDATE Net.*\nEND\n"
    )
    return mxai, mxtrain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096, help="ancho de cada capa oculta")
    ap.add_argument("--depth", type=int, default=16, help="nº de capas ocultas")
    ap.add_argument("--rows", type=int, default=60000, help="filas del dataset sintético")
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--features", type=int, default=24)
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "stdlib"])
    ap.add_argument("--log-every", type=int, default=10, help="cada cuántas épocas imprimir VRAM/tiempo")
    args = ap.parse_args()

    os.environ["MATRIXAI_TRAIN_BACKEND"] = args.backend

    import matrixai  # noqa: F401
    print(f"matrixai OK: {matrixai.__file__}")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"torch {torch.__version__} | CUDA: {has_cuda}"
              + (f" | GPU: {torch.cuda.get_device_name(0)}" if has_cuda else ""))
    except Exception as exc:  # noqa: BLE001
        torch, has_cuda = None, False
        print(f"torch no disponible: {exc}")

    from matrixai.playground import _generate_synthetic_dataset, _run_playground_dense_training

    classes = ["OK", "DESGASTE", "DESALINEACION", "SOBRECARGA", "FUGA", "FALLO"]
    mxai, mxtrain = build_specs(args.width, args.depth, args.features, classes)
    n_params = args.features * args.width + (args.depth - 1) * args.width * args.width + args.width * len(classes)
    print(f"\nConfig: width={args.width} depth={args.depth} rows={args.rows} "
          f"epochs={args.epochs} backend={args.backend}")
    print(f"~{n_params/1e6:.1f}M parámetros | entrenamiento full-batch (todas las filas en una matriz)\n")

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    print("[M10] generando dataset (coherent)...", flush=True)
    gen = _generate_synthetic_dataset(mxai, mxtrain, rows=args.rows, seed=42, mode="coherent")
    if not gen.get("ok"):
        print("ERROR generación:", gen.get("error"))
        return 1
    print(f"[M10] {gen['rows']} filas en {time.time()-t0:.1f}s", flush=True)

    # Callback en vivo: VRAM y ritmo por época durante el run (varios minutos).
    t_train = time.time()
    last = {"t": t_train}

    def _live(entry: dict) -> None:
        ep = entry.get("epoch", 0)
        if ep % args.log_every != 0:
            return
        now = time.time()
        per_ep = (now - last["t"]) / max(1, args.log_every)
        last["t"] = now
        msg = (f"  ep {ep:>4}  train_loss={entry.get('train_loss')}  "
               f"val_loss={entry.get('validation_loss')}  {per_ep:.2f}s/ep")
        if has_cuda:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            msg += f"  VRAM {alloc:.1f}G alloc / {reserved:.1f}G reserved"
        print(msg, flush=True)

    print("[M14/M15] entrenando + evaluando (mira nvidia-smi en otra celda)...", flush=True)
    r = _run_playground_dense_training(mxai, mxtrain, gen["csv_text"],
                                       epochs_override=args.epochs, epoch_callback=_live)
    if not r.get("ok"):
        print("ERROR entrenamiento:", r.get("error"))
        return 1
    dt = time.time() - t_train
    print(f"\n[M14] backend={r['backend']}  eval_backend={r.get('evaluation_backend')}  "
          f"acc={r['accuracy']}  macro_f1={r['macro_f1']}  en {dt:.1f}s")
    if r.get("evaluation_warning"):
        print("  ⚠️ eval warning:", r["evaluation_warning"])
    if has_cuda:
        print(f"VRAM PICO: {torch.cuda.max_memory_allocated()/1e9:.2f} GB de "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")

    ok_gpu = (r["backend"] == "cuda")
    print("\n" + ("✅ GPU end-to-end OK (CUDA)" if ok_gpu else f"ℹ️ backend={r['backend']} (sin CUDA)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
