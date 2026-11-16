# 116-C5 — la imagen para medir el ACIERTO de TabICL v2 en una GPU de verdad (el portátil de Roberto). Se construye UNA vez
# y con red; la medición corre SIN red (`--network none`). Mismo molde que 116-C4 (`../tabicl_116c4_gpu/Dockerfile`).
FROM python:3.12-slim

# torch con CUDA 12.4 y FIJADO a la del paquete GPU del Studio (2.6.0), como 116-C4. Para construir sin GPU (la sonda del servidor
# usa esta misma imagen: torch ve «sin GPU» y el guion se niega salvo --permitir-cpu):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t medicion-tabicl-acierto-gpu:cpu .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --retries 10 --timeout 60 "torch==2.6.0" --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir --retries 10 --timeout 60 "tabicl==2.2.0" numpy pandas scipy scikit-learn

# Los pesos ANTES que el resto: retocar un guion no vuelve a bajar 215 MB. Verificados por sha256: si no casan, no se construye.
COPY bajar_pesos.py /kit/
RUN python3 /kit/bajar_pesos.py /pesos && pip freeze > /kit/versiones_de_la_imagen.txt

# El núcleo y los motores (código del árbol, 16 MB), los ARFF empaquetados (con sha256) y el guion.
COPY codigo /kit/codigo
COPY datos /kit/datos
COPY conjuntos.json REGLA.md medir_acierto.py /kit/
ENV HF_HUB_OFFLINE=1 MATRIXAI_TABICL_PESOS_DIR=/pesos PYTHONDONTWRITEBYTECODE=1
WORKDIR /kit
CMD ["python3", "medir_acierto.py", "--salida", "/salida/resultado_116c5_acierto_gpu.json"]
