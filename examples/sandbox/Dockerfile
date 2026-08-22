# Imagen para `matrixai replay` — contrato 81-C5.
#
# Reproducir un paquete exige un matrixai que sepa `verify` (82-C2). La
# imagen del producto puede ser anterior, y entonces `replay` se niega
# ANTES de dar un veredicto: su `rc=2` de argparse seria indistinguible
# del `rc=2` de «alguien toco el paquete».
#
# Se construye sobre la imagen del Studio, que ya trae Python y las
# dependencias, y se le monta encima ESTE arbol:
#
#   docker build -f examples/sandbox/Dockerfile -t matrixai-sandbox:local .
#   MATRIXAI_SANDBOX_IMAGE=matrixai-sandbox:local python3 -m matrixai replay <paquete>
#
# Cambia la etiqueta base por la que tengas (`docker images | grep matrixai`).
ARG BASE=matrixai-studio:v2.0
FROM ${BASE}
COPY matrixai /opt/matrixai_src/matrixai
ENV PYTHONPATH=/opt/matrixai_src
