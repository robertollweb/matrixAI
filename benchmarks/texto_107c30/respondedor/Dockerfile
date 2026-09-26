# 107-C3.0, condición (3) — la imagen del respondedor local (Qwen2.5-0.5B-Instruct vía
# onnxruntime-genai) MÁS el mismo software con el que `medir_c30.py` mide (0), (1) y (2)
# (lightgbm, scikit-learn, tokenizers): la condición (3) tiene que correr con el MISMO
# lightgbm y las MISMAS particiones que las demás, no una réplica aproximada aparte.
#
# Versiones FIJADAS (26-09, leídas de PyPI con `pip index versions`, sin instalar nada en
# el host): lightgbm/scikit-learn/numpy/pandas/tokenizers son las YA instaladas y usadas
# para medir (0)/(1)/(2) en esta máquina (mismo software en las dos condiciones);
# onnxruntime/onnxruntime-genai son la pareja más reciente disponible el 26-09 -- SIN
# comprobar aquí que `pip` las resuelve juntas sin conflicto (esta imagen NO se construye
# en esta sesión, "no instales nada en el host, no hagas docker build" -- lo hace Roberto
# con `preparar.sh`, que falla alto y claro si el resolutor de pip no está de acuerdo).
#
# NO construir esto con `docker build` desde una sesión de agente: hay una release en
# curso y la máquina va justa (encargo de 107-C3, 26-09). Lo construye Roberto.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==2.4.4 \
    pandas==3.0.5 \
    scikit-learn==1.8.0 \
    lightgbm==4.7.0 \
    tokenizers==0.23.2 \
    onnxruntime==1.30.0 \
    onnxruntime-genai==0.17.0
