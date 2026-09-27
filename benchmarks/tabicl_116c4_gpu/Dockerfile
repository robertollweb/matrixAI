# 116-C4 — la imagen para medir TabICL v2 en una GPU de verdad (el PC de Roberto; decisión del
# 27-09: primero en el paquete GPU, medido antes de meterlo en el estudio). Se construye UNA vez y
# con red; la medición corre SIN red (`--network none`).
FROM python:3.12-slim

# torch con CUDA 12.4: el MISMO índice que la imagen GPU del Studio (`studio-backend/Dockerfile.gpu`),
# para medir lo que llevaría el paquete. Para probar el kit en una máquina sin GPU:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t medicion-tabicl-gpu:cpu .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir "tabicl==2.2.0" numpy pandas scipy scikit-learn

# Los pesos ANTES que los guiones de medir: retocar un guion no vuelve a bajar 215 MB.
# Verificados por sha256 (los de 116-C0 y del adaptador): si no casan, la imagen NO se construye.
COPY bajar_pesos.py /kit/
RUN python3 /kit/bajar_pesos.py /pesos && pip freeze > /kit/versiones_de_la_imagen.txt

COPY medir_punto.py medir_todo.py /kit/
ENV HF_HUB_OFFLINE=1
WORKDIR /kit
CMD ["python3", "medir_todo.py", "--salida", "/salida/resultado_116c4_gpu.json"]
