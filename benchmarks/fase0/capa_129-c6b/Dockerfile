# MatrixAI Studio CPU local layer.
# Built on the user's machine the first time start.sh / start.bat runs.
# The base image comes from image/matrixai-studio-v2.12.tar; this layer only adds
# the PyTorch CPU wheel, so the downloadable package stays small (the ZIP does NOT
# carry torch: Roberto, 2026-10-03, «mantener el modelo de descarga para no saturar
# el servidor con descargas demasiado grandes»; the GPU package does the same).
FROM matrixai-studio:129-c6b

LABEL org.opencontainers.image.title="MatrixAI Studio CPU Local Layer"
LABEL org.opencontainers.image.description="MatrixAI Studio base image plus PyTorch 2.6.0 (CPU) wheels"

# MISMA receta que Dockerfile.gpu, con el índice de CPU. Por qué cada cosa:
#
# * `--extra-index-url`, NO `--index-url`: este último SUSTITUYE a PyPI y las
#   dependencias de torch (sympy, networkx, jinja2, fsspec…) viven en PyPI. Con
#   `--extra-index-url` los dos índices se buscan a la vez.
# * `torch==2.6.0+cpu`, con el sufijo: el sufijo `+cpu` solo existe en el índice
#   de PyTorch. Con `torch==2.6.0` a secas, PyPI (que está disponible) ganaría y
#   traería la rueda de Linux con CUDA y sus ~2,9 GB de dependencias nvidia-*:
#   justo lo que este paquete NO debe instalar.
# * Versión fijada (2.6.0, la misma que verifica el paquete GPU, la más nueva con
#   rueda para cp312): un paquete publicado tiene que construir IGUAL siempre.
# * `--retries 10 --timeout 60`: pip trae 15 s de plazo de lectura y un microcorte
#   tumba la descarga; sin caché no se reanuda. Es gratis con buena red.
# * `--no-cache-dir`: la caché de pip acabaría dentro de la capa.
RUN python3 -m pip install --no-cache-dir \
    --retries 10 --timeout 60 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0+cpu"
