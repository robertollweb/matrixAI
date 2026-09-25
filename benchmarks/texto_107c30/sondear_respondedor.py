# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — SOLO la sonda de coste del respondedor local de la condición
(3) («un respondedor local de preguntas atómicas»). Esto NO implementa la
condición (3) del estudio (eso es C3.1, "el texto como columnas", con su
propio bucle de preguntas DENTRO de la partición de desarrollo): mide cuánto
cuesta, en esta máquina, contestar UNA pregunta atómica sí/no sobre UN texto,
con el candidato MÁS PEQUEÑO del contrato.

**Candidato**: `Qwen2.5-0.5B-Instruct` (Apache-2.0). El contrato deja
"pendiente... el modelo concreto del respondedor local y su coste en esta
máquina" entre tres candidatos sin medir (Qwen2.5-0.5B-Instruct,
SmolLM2-1.7B-Instruct, Phi-3.5-mini-instruct); este corte mide el más barato
de los tres, como pide el encargo.

**Camino de ejecución, y por qué éste y no otro** (medido en esta sesión,
antes de escribir este fichero):
  - `transformers`+`torch`: NO instalados y el contrato ya avisa que la
    imagen CPU no los lleva ("necesitarían un camino GGUF/onnxruntime").
  - `llama-cpp-python` (camino GGUF): **no se puede construir aquí** -- su
    instalación compila C/C++ con CMake y esta máquina no tiene compilador
    (`gcc`/`cc` ausentes, `pip install` falla en la fase de build de la
    rueda). Sin `sudo` no se puede instalar uno (ver
    `project_sandbox_no_browser.md` de la memoria de Roberto).
  - `onnxruntime-genai` (camino ONNX declarado en el propio contrato): SÍ se
    pudo instalar (`pip install --user --break-system-packages
    onnxruntime-genai`, rueda prefabricada, sin compilar nada). **Esto
    actualizó `onnxruntime` de 1.26.0 a 1.30.0** en el `site-packages` de
    usuario -- se declara aquí porque es un cambio de entorno real, del
    mismo tipo que la casa pide declarar (`generar-terceros.py`,
    "instalar una dependencia deja coja la tabla de terceros").

**El modelo en sí**: no hay una conversión ONNX-GenAI oficial de Qwen en el
repositorio de Qwen; se usa `xiaoyao9184/Qwen2.5-0.5B-Instruct-onnx-genai`
(variante `cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4`), una conversión
de terceros del MISMO `Qwen/Qwen2.5-0.5B-Instruct` (sin reentrenar: convierte
pesos, no los cambia), así que hereda su licencia Apache-2.0 -- pero es una
conversión de un tercero, no de Qwen ni de Microsoft, y se declara así.

Si `onnxruntime-genai` no se puede importar, o el modelo no se puede
descargar, este script **lo dice y para** (sin escribir un resultado a
medias): eso es justo lo que pide el encargo.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent
TOPE_TOTAL_S = 15 * 60  # 15 minutos, el tope del encargo
HILOS = 4
N_TEXTOS = 20
MAX_TOKENS_NUEVOS = 8  # una respuesta atómica sí/no no necesita más
TRUNCADO_TEXTO_CARACTERES = 600  # la pregunta es sobre EL TEXTO, no sobre la novela entera

MODELO_REPO = "xiaoyao9184/Qwen2.5-0.5B-Instruct-onnx-genai"
MODELO_SUBCARPETA = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
MODELO_LOCAL_POR_OMISION = (
    Path.home() / ".cache" / "matrixai" / "text_responders"
    / "qwen2.5-0.5b-instruct-onnx-genai-cpu-int4"
)

SALIDA_POR_OMISION = RAIZ / "resultado_sonda_respondedor.json"

os.environ.setdefault("OMP_NUM_THREADS", str(HILOS))


def _parar(motivo: str) -> "int":
    print(f"SONDA PARADA: {motivo}", file=sys.stderr)
    return 1


def _asegurar_modelo(destino: Path, *, tope_s: float) -> tuple[bool, str]:
    """Descarga el modelo si falta. Devuelve (ok, motivo_si_no)."""
    marcador = destino / "genai_config.json"
    if marcador.is_file():
        return True, ""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "falta 'huggingface_hub' y el modelo no está descargado; no se instala nada más"
    print(f"descargando {MODELO_REPO}/{MODELO_SUBCARPETA} a {destino}...", file=sys.stderr)
    t0 = time.time()
    try:
        tmp_local = destino.parent / (destino.name + "__descarga")
        snapshot_download(repo_id=MODELO_REPO, allow_patterns=[f"{MODELO_SUBCARPETA}/*"],
                          local_dir=str(tmp_local))
        if time.time() - t0 > tope_s:
            return False, f"la descarga tardó más del tope declarado para esta sonda ({tope_s:.0f}s)"
        origen = tmp_local / MODELO_SUBCARPETA
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            import shutil
            shutil.rmtree(destino)
        origen.rename(destino)
        import shutil
        shutil.rmtree(tmp_local, ignore_errors=True)
    except Exception as e:  # noqa: BLE001 -- se declara el motivo, no se re-lanza
        return False, f"descarga fallida: {e!r}"
    return True, ""


def _fijar_hilos_en_config(destino: Path, *, hilos: int) -> None:
    """`intra_op_num_threads`/`inter_op_num_threads` DENTRO de la copia local
    del `genai_config.json` -- es lo que de verdad limita los hilos de
    onnxruntime-genai (el `OMP_NUM_THREADS` de arriba es un cinturón extra,
    no sustituye a esto: onnxruntime no siempre respeta variables de
    entorno de OpenMP con su propio pool de hilos)."""
    ruta = destino / "genai_config.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    so = datos["model"]["decoder"].setdefault("session_options", {})
    if so.get("intra_op_num_threads") == hilos and so.get("inter_op_num_threads") == 1:
        return
    so["intra_op_num_threads"] = hilos
    so["inter_op_num_threads"] = 1
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def _cargar_textos_de_tarea_a(n: int) -> list[str]:
    ruta = RAIZ / "tareas" / "tarea_a.csv"
    if not ruta.is_file():
        raise SystemExit(
            f"falta {ruta}: corre preparar_tareas.py primero (esta sonda usa textos reales de la tarea A)"
        )
    textos = []
    with ruta.open("r", encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            textos.append(fila["texto"][:TRUNCADO_TEXTO_CARACTERES])
            if len(textos) >= n:
                break
    if len(textos) < n:
        raise SystemExit(f"tarea A solo tiene {len(textos)} filas, se pedían {n}")
    return textos


PREGUNTA_ATOMICA = (
    'Lee el siguiente texto y responde SOLO con la palabra "si" o la palabra "no", sin explicación.\n'
    "Pregunta: ¿El texto menciona algún lugar geográfico (una ciudad, región o país)?\n"
    "Texto: {texto}"
)


def medir(*, n_textos: int, hilos: int, tope_total_s: float, modelo_local: Path) -> dict[str, Any]:
    import onnxruntime_genai as og  # noqa: PLC0415 -- se importa aquí para poder fallar limpio antes

    t_inicio_total = time.perf_counter()
    textos = _cargar_textos_de_tarea_a(n_textos)

    t0 = time.perf_counter()
    model = og.Model(str(modelo_local))
    tokenizer = og.Tokenizer(model)
    t_carga_s = time.perf_counter() - t0

    respuestas: list[dict[str, Any]] = []
    parada_por_tope = False
    for i, texto in enumerate(textos):
        if time.perf_counter() - t_inicio_total > tope_total_s:
            parada_por_tope = True
            break
        mensajes = json.dumps([{"role": "user", "content": PREGUNTA_ATOMICA.format(texto=texto)}])
        prompt = tokenizer.apply_chat_template(messages=mensajes)
        tokens_prompt = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=int(len(tokens_prompt)) + MAX_TOKENS_NUEVOS,
                                  do_sample=False, temperature=1.0)
        generador = og.Generator(model, params)

        t_r0 = time.perf_counter()
        generador.append_tokens(tokens_prompt)
        tokens_generados: list[int] = []
        while not generador.is_done() and len(tokens_generados) < MAX_TOKENS_NUEVOS:
            generador.generate_next_token()
            tokens_generados.append(int(generador.get_next_tokens()[0]))
        t_r1 = time.perf_counter()

        texto_respuesta = tokenizer.decode(tokens_generados) if tokens_generados else ""
        respuestas.append({
            "indice": i,
            "n_tokens_prompt": int(len(tokens_prompt)),
            "n_tokens_generados": len(tokens_generados),
            "latencia_s": t_r1 - t_r0,
            "respuesta_bruta": texto_respuesta,
            "ru_maxrss_kb_acumulado": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })

    duracion_total_s = time.perf_counter() - t_inicio_total
    latencias = [r["latencia_s"] for r in respuestas]
    latencias_ordenadas = sorted(latencias)

    def percentil(p: float) -> float | None:
        if not latencias_ordenadas:
            return None
        idx = min(len(latencias_ordenadas) - 1, int(round(p * (len(latencias_ordenadas) - 1))))
        return latencias_ordenadas[idx]

    return {
        "modelo": {
            "candidato": "Qwen2.5-0.5B-Instruct",
            "licencia": "Apache-2.0",
            "conversion_onnx_genai_de": MODELO_REPO,
            "subcarpeta": MODELO_SUBCARPETA,
            "camino_de_ejecucion": "onnxruntime-genai (CPU, int4 RTN block-32)",
            "ruta_local": str(modelo_local),
        },
        "hilos_declarados": hilos,
        "n_textos_pedidos": n_textos,
        "n_textos_respondidos": len(respuestas),
        "parada_por_tope_total": parada_por_tope,
        "tope_total_s": tope_total_s,
        "tiempo_carga_modelo_s": t_carga_s,
        "duracion_total_s": duracion_total_s,
        "latencia_por_respuesta_s": {
            "media": (sum(latencias) / len(latencias)) if latencias else None,
            "mediana": percentil(0.5),
            "p95": percentil(0.95),
            "minima": min(latencias) if latencias else None,
            "maxima": max(latencias) if latencias else None,
        },
        "memoria_pico_kb": max((r["ru_maxrss_kb_acumulado"] for r in respuestas), default=None),
        "memoria_pico_mb": (max((r["ru_maxrss_kb_acumulado"] for r in respuestas), default=0) / 1024.0)
        if respuestas else None,
        "respuestas": respuestas,
        "pregunta_atomica_plantilla": PREGUNTA_ATOMICA,
        "truncado_texto_caracteres": TRUNCADO_TEXTO_CARACTERES,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-textos", type=int, default=N_TEXTOS)
    ap.add_argument("--hilos", type=int, default=HILOS)
    ap.add_argument("--tope-total-s", type=float, default=TOPE_TOTAL_S)
    ap.add_argument("--modelo-local", type=Path, default=MODELO_LOCAL_POR_OMISION)
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_OMISION)
    ns = ap.parse_args()

    t_inicio = time.perf_counter()
    try:
        import onnxruntime_genai  # noqa: F401
    except ImportError:
        return _parar(
            "'onnxruntime-genai' no está instalado. Se instaló en esta sesión con "
            "'pip install --user --break-system-packages onnxruntime-genai' (rueda prefabricada, "
            "sin compilar); si en otra máquina no está, esa es la orden. "
            "'llama-cpp-python' (el otro camino GGUF) NO se pudo instalar aquí: compila C/C++ con "
            "CMake y esta máquina no tiene compilador (gcc/cc ausentes, sin sudo)."
        )

    ok, motivo = _asegurar_modelo(ns.modelo_local, tope_s=ns.tope_total_s)
    if not ok:
        return _parar(f"no se pudo obtener el modelo en {ns.modelo_local}: {motivo}")
    _fijar_hilos_en_config(ns.modelo_local, hilos=ns.hilos)

    tope_restante = ns.tope_total_s - (time.perf_counter() - t_inicio)
    if tope_restante <= 0:
        return _parar("el tope total se agotó preparando el modelo, antes de medir ninguna respuesta")

    resultado = medir(n_textos=ns.n_textos, hilos=ns.hilos, tope_total_s=tope_restante,
                      modelo_local=ns.modelo_local)
    resultado["tiempo_total_del_proceso_s"] = time.perf_counter() - t_inicio

    ns.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in resultado.items() if k != "respuestas"},
                     indent=2, ensure_ascii=False))
    print(f"\nescrito en {ns.salida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
