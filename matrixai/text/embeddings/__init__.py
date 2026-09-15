# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — catálogo MEDIDO de proveedores de embeddings de texto.

Dos familias (107-C1): embeddings **estáticos** (una tabla: un vector por
token, sin transformer) y **transformers de frase pequeños exportados a
ONNX**. Los pesos NO viajan en la imagen (102-C5): cada proveedor es
descargable, con su licencia aceptada y cada fichero fijado por su digest.

Lo que este paquete declara a mano es la LISTA DE CANDIDATOS (qué repo, qué
fichero, qué revisión). Todo número —tamaño, dimensión, tiempo por 1.000
textos, cobertura de idioma— sale de `medicion.py` y vive en el artefacto
`catalogo_medido.json`: si un número de esa tabla se puede teclear a mano,
está mal hecha.
"""
