#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
#
# Arma (o COMPRUEBA) el paquete que se sube a Colab para medir la Fase 0.
#
# POR QUÉ EXISTE, y es un fallo medido, no una precaución.
# ----------------------------------------------------------------------------
# El paquete se montó A MANO el 2026-09-12. Al día siguiente se repararon dos
# motores —el lineal pasó de que lo mataran a los 639 s a tardar 69, y el denso
# dejó de heredar el tope de 50 MB del producto— y el paquete **siguió llevando
# las copias viejas**. Relanzar la medición en Colab habría perdido
# `KDDCup09_appetency` DOS veces, por las dos razones ya arregladas, y la nota
# del día anterior habría dicho que estaba resuelto.
#
# Un apaño que caduca sin avisar es peor que no ponerlo, porque su comentario
# sigue sonando razonable. Por eso esto tiene modo `--comprobar`: no basta con
# poder reconstruirlo, hace falta poder PREGUNTAR si está viejo.
#
#   empaquetar_colab.sh --comprobar   -> 0 si está al día; 1 y la lista si no
#   empaquetar_colab.sh --construir   -> lo rehace desde los árboles vivos
#
# Lo que NO viaja, a propósito: los ARFF (254 MB — se bajan de OpenML y se
# verifica su sha256 contra el protocolo) y los 8 datasets sellados, que C5
# guarda justamente para no haberlos mirado nunca.
set -euo pipefail

CORE=/home/deployer/matrixAI
ENGINES=/home/deployer/matrixai-engines
DESTINO=/home/deployer/fase0_colab
TARBALL=/home/deployer/fase0_colab.tar.gz
FASE0="$CORE/benchmarks/fase0"

modo="${1:---comprobar}"

# Pares «origen -> ruta dentro del paquete». El paquete replica los dos
# árboles, no una selección: elegir ficheros a mano es cómo se quedó viejo.
copiar_arbol() {  # $1 origen  $2 destino
  rsync -a --delete \
        --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
        "$1"/ "$2"/
}

case "$modo" in
  --construir)
    mkdir -p "$DESTINO/core" "$DESTINO/engines/src"
    copiar_arbol "$CORE/matrixai"                    "$DESTINO/core/matrixai"
    copiar_arbol "$ENGINES/src/matrixai_engines"     "$DESTINO/engines/src/matrixai_engines"
    cp "$FASE0/protocolo_exploratorio.json" "$DESTINO/"
    [ -f "$FASE0/seleccion_40_final.json" ] && cp "$FASE0/seleccion_40_final.json" "$DESTINO/"
    # `pasada_colab.py` y el notebook son del paquete, no de los árboles: se
    # editan aquí y no se pisan.
    tar -czf "$TARBALL" -C "$(dirname "$DESTINO")" \
        --exclude '__pycache__' --exclude '*.pyc' "$(basename "$DESTINO")"
    echo "✓ paquete reconstruido: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
    ;;

  --comprobar)
    viejos=0
    while IFS= read -r -d '' copia; do
      rel="${copia#$DESTINO/core/matrixai/}"
      orig="$CORE/matrixai/$rel"
      if [ ! -f "$orig" ]; then echo "SOBRA   core/matrixai/$rel"; viejos=1; continue; fi
      cmp -s "$copia" "$orig" || { echo "VIEJO   core/matrixai/$rel"; viejos=1; }
    done < <(find "$DESTINO/core/matrixai" -name '*.py' -not -path '*__pycache__*' -print0)

    while IFS= read -r -d '' copia; do
      rel="${copia#$DESTINO/engines/src/matrixai_engines/}"
      orig="$ENGINES/src/matrixai_engines/$rel"
      if [ ! -f "$orig" ]; then echo "SOBRA   engines/$rel"; viejos=1; continue; fi
      cmp -s "$copia" "$orig" || { echo "VIEJO   engines/$rel"; viejos=1; }
    done < <(find "$DESTINO/engines/src/matrixai_engines" -name '*.py' -not -path '*__pycache__*' -print0)

    # Y al revés: un fichero NUEVO en el árbol que el paquete no tenga es tan
    # grave como uno viejo. El motor lineal disperso habría entrado por aquí.
    while IFS= read -r -d '' orig; do
      rel="${orig#$CORE/matrixai/}"
      [ -f "$DESTINO/core/matrixai/$rel" ] || { echo "FALTA   core/matrixai/$rel"; viejos=1; }
    done < <(find "$CORE/matrixai" -name '*.py' -not -path '*__pycache__*' -print0)

    while IFS= read -r -d '' orig; do
      rel="${orig#$ENGINES/src/matrixai_engines/}"
      [ -f "$DESTINO/engines/src/matrixai_engines/$rel" ] || { echo "FALTA   engines/$rel"; viejos=1; }
    done < <(find "$ENGINES/src/matrixai_engines" -name '*.py' -not -path '*__pycache__*' -print0)

    # LOS JSON, que es donde este comprobador tenia un HUECO.
    #
    # El 2026-09-13 se re-firmo el protocolo (cambia su digest) y
    # `--comprobar` dijo «al dia»: solo miraba ficheros `.py`. Una pasada en
    # Colab habria medido contra el protocolo VIEJO mientras esto afirmaba que
    # estaba todo bien — o sea, el mismo fallo que este script existe para
    # impedir, cometido por el script.
    #
    # `seleccion_40_final.json` es opcional (solo hace falta para re-descargar
    # los ARFF por `file_id`); el protocolo NO lo es.
    for j in protocolo_exploratorio.json seleccion_40_final.json; do
      [ -f "$FASE0/$j" ] || continue
      if [ ! -f "$DESTINO/$j" ]; then echo "FALTA   $j"; viejos=1; continue; fi
      cmp -s "$DESTINO/$j" "$FASE0/$j" || { echo "VIEJO   $j"; viejos=1; }
    done

    if [ "$viejos" -eq 0 ]; then
      echo "✓ el paquete de Colab está al día con los dos árboles"
    else
      echo
      echo "✗ el paquete NO está al día. Relanzar la medición así mediría"
      echo "  código que ya no existe. Se rehace con: $0 --construir"
      exit 1
    fi
    ;;

  *) echo "uso: $0 [--comprobar|--construir]" >&2; exit 2;;
esac
