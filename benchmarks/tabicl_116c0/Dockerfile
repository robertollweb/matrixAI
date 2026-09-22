# 116-C0 — la imagen de la medición de TabICL. Versiones FIJADAS (las del piloto
# del 112): lo medido tiene que poder repetirse con lo mismo.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "tabicl==2.2.0" numpy pandas scipy scikit-learn
