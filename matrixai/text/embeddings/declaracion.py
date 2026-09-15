# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C2 — la declaración del componente de terceros, COMPUESTA desde lo medido.

Cuando el producto usa un modelo ajeno, el expediente lo dice; y lo dice con
números que salen del catálogo medido de 107-C1
(`catalogo_medido.json`), nunca tecleados. Esta pieza es la que compone ese
diccionario: el mismo que consumen el manifiesto, el BOM y la `inference_spec`
(`matrixai.export.terceros`), y el mismo que ya produce
`matrixai_engines.embeddings.ProveedorDeTexto.declaracion()` para el proveedor
que habla español.

**Lo único que NO sale del catálogo es la longitud efectiva**, y eso es el
invariante 6 del 107, reescrito el 2026-09-14 porque se midió falso: hacen
falta TRES cosas y no dos —el digest de los pesos, el del tokenizer **y el
truncado**—, y la tercera **se fija explícitamente en el código de quien
ejecuta el proveedor**, nunca se hereda del valor por omisión de una
dependencia. Por eso aquí se RECIBE, con su procedencia, y se COMPRUEBA contra
el catálogo: si el número fijado no es el que se usó al medir, lo medido
describe otro vector y el componente no se declara.

**Y declara lo que NO se sabe** (invariante 4, y 90 §4.3): `lo_que_no_se_sabe`
no es adorno, es la mitad del expediente. Un componente que solo enumera lo
que se conoce se lee como si lo conociera todo.

Este módulo no importa `numpy` ni `onnxruntime` ni ejecuta nada: lee el
artefacto. El núcleo se instala con `dependencies = []` (102, invariante 1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from matrixai.text.embeddings import cobertura
from matrixai.text.embeddings.catalogo import por_id

__all__ = [
    "ARTEFACTO",
    "DeclaracionImposible",
    "FRASE_DE_OPACIDAD",
    "componente_de_terceros",
]

ARTEFACTO = cobertura.ARTEFACTO

#: Lo que se dice de un embedding preentrenado, literal (107, invariante 4).
#: Vive aquí y no en cada llamante porque **dos sitios declarando lo mismo
#: acaban divergiendo**; el proveedor del núcleo la repite en su `describe()`
#: y una prueba comprueba que son la misma frase.
FRASE_DE_OPACIDAD = (
    "pesos de terceros no auditados; lo que el vector representa no se puede "
    "explicar desde aquí, solo declarar de dónde viene"
)

#: Los orígenes admitidos de la longitud efectiva. Mismo vocabulario cerrado
#: que `matrixai_engines.embeddings.LongitudFijada.origen` y que
#: `matrixai.export.terceros.ORIGENES_DE_LA_LONGITUD`.
_ORIGENES = ("libreria_del_proveedor", "paquete")

_CAMPOS_DE_LA_LONGITUD = ("tokens", "origen", "fuente", "por_que", "fijada_en")


class DeclaracionImposible(Exception):
    """No se puede declarar este componente, y se dice por qué.

    Nunca se rellena con un valor por omisión: un expediente a medias sobre un
    componente opaco es peor que no tenerlo, porque quien lo lea creerá que lo
    que falta no existía.
    """


def _cargar(artefacto: Path | None) -> dict[str, Any]:
    ruta = Path(artefacto) if artefacto is not None else ARTEFACTO
    if not ruta.is_file():
        raise DeclaracionImposible(
            f"no hay catálogo medido en {ruta}: sin él no hay ningún número que "
            "declarar, y este componente no se escribe a mano"
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def _fila(datos: dict[str, Any], proveedor_id: str) -> dict[str, Any]:
    fila = (datos.get("proveedores") or {}).get(proveedor_id)
    if fila is None:
        raise DeclaracionImposible(
            f"{proveedor_id} no está en el catálogo medido: no se declara un "
            "componente del que no se ha medido nada"
        )
    return fila


def _medicion(fila: dict[str, Any], proveedor_id: str) -> dict[str, Any]:
    medicion = fila.get("medicion")
    if not medicion or not (medicion.get("describe") or {}):
        motivo = (fila.get("candidato") or {}).get("no_descargado_porque") or ""
        raise DeclaracionImposible(
            f"{proveedor_id} está fijado pero nunca se ha ejecutado"
            + (f": {motivo}" if motivo else ".")
            + " Su dimensión y su cobertura de idioma no están medidas, así que un "
            "paquete no puede declararlo como el componente que produce sus vectores."
        )
    return medicion


def _fichero(fila: dict[str, Any], ruta: str, proveedor_id: str, que: str) -> dict[str, Any]:
    for f in fila.get("ficheros") or []:
        if f.get("ruta") == ruta:
            return dict(f)
    raise DeclaracionImposible(
        f"{proveedor_id}: el catálogo no fija ningún digest para {ruta!r} ({que}). "
        "El tokenizer es parte del modelo (107, invariante 6): declarar el digest de "
        "los pesos y callar el suyo sería media verdad."
    )


def _comprobar_la_longitud(
    longitud: Mapping[str, Any], describe: Mapping[str, Any], proveedor_id: str,
) -> list[str]:
    """La tercera cosa del invariante 6, contrastada contra lo medido.

    Devuelve contra qué se comprobó. Son DOS testigos y no uno:

    1. el número fijado tiene que ser **el que se usó al medir** — si no, la
       dimensión, los tiempos y la AUC del catálogo describen otro vector y
       este componente los estaría atribuyendo a este truncado;
    2. su PROCEDENCIA tiene que cuadrar con el paquete: `origen="paquete"`
       obliga a que alguno de sus ficheros declare ese número, y
       `origen="libreria_del_proveedor"` obliga a lo contrario —es el caso
       medido de `potion-base-8M`, cuyo 512 no aparece en ningún fichero suyo
       (los dos que hay dicen 1.000.000, o sea «no truncar»)—.
    """
    faltan = [c for c in _CAMPOS_DE_LA_LONGITUD if not str(longitud.get(c) or "").strip()]
    if faltan:
        raise DeclaracionImposible(
            f"{proveedor_id}: la longitud efectiva llega sin {faltan}. Se fija en el "
            "código de quien ejecuta el proveedor y viaja con su fuente; un número sin "
            "procedencia caduca el día que cambie la librería y nada lo dice."
        )
    if longitud["origen"] not in _ORIGENES:
        raise DeclaracionImposible(
            f"{proveedor_id}: origen de la longitud {longitud['origen']!r} desconocido; "
            f"se admiten {list(_ORIGENES)}"
        )
    tokens = longitud["tokens"]
    if type(tokens) is not int or tokens <= 0:
        raise DeclaracionImposible(
            f"{proveedor_id}: longitud efectiva {tokens!r} — tiene que ser un entero positivo")

    medido = describe.get("max_tokens")
    if medido != tokens:
        raise DeclaracionImposible(
            f"{proveedor_id}: se fija una longitud de {tokens} tokens y el catálogo se "
            f"midió con {medido!r}. Lo medido —dimensión, tiempos, AUC— describe el "
            "vector del OTRO truncado; o se vuelve a medir o no se declara."
        )

    declaradas = {
        e.get("valor") for e in (describe.get("max_tokens_declarados_por_el_paquete") or [])
        if isinstance(e, dict)
    }
    if longitud["origen"] == "paquete" and tokens not in declaradas:
        raise DeclaracionImposible(
            f"{proveedor_id}: la longitud dice venir del paquete y ninguno de sus "
            f"ficheros declara {tokens} (declaran {sorted(v for v in declaradas if v is not None)})"
        )
    if longitud["origen"] == "libreria_del_proveedor" and tokens in declaradas:
        raise DeclaracionImposible(
            f"{proveedor_id}: la longitud dice venir del valor por omisión de una "
            f"librería, y el propio paquete declara {tokens}. Si el paquete lo dice, el "
            "origen es el paquete — y la diferencia importa: un número heredado de una "
            "dependencia cambia cuando cambie la dependencia."
        )
    return [
        f"catalogo_medido.json:proveedores.{proveedor_id}.medicion.describe.max_tokens",
        f"catalogo_medido.json:proveedores.{proveedor_id}.medicion.describe."
        f"max_tokens_declarados_por_el_paquete",
    ]


def _lo_que_no_se_sabe(
    proveedor_id: str,
    fila: Mapping[str, Any],
    medicion: Mapping[str, Any],
    idiomas: Mapping[str, Any],
    dependencias: Any,
) -> list[str]:
    """La mitad del expediente que nadie escribe (invariante 4).

    Cada línea sale de un hecho del catálogo, no de una plantilla: si mañana se
    mide el idioma que falta, la línea desaparece sola.
    """
    dicho: list[str] = [FRASE_DE_OPACIDAD]
    dicho.append(
        "no se ha auditado con qué datos se entrenaron estos pesos ni qué contienen: "
        "el catálogo fija su origen, su revisión y su licencia, no su procedencia de datos"
    )
    medidos = sorted(idiomas)
    no_cubiertos = sorted(k for k, v in idiomas.items() if v.get("estado") != "cubierto")
    if no_cubiertos:
        dicho.append(
            "idiomas medidos y NO cubiertos por este proveedor: " + ", ".join(no_cubiertos)
            + " — un texto en uno de ellos da vector, y ese vector no significa gran cosa"
        )
    dicho.append(
        "de cualquier idioma que no sea " + ", ".join(medidos) + " no se puede decir nada: "
        "no se ha medido, y «no medido» no es «no cubierto»"
    )
    dicho.append(
        "lo que este proveedor aporta en una tarea real NO está medido: la AUC de "
        "coherencia dice que distingue temas dentro de un idioma, no que mejore una "
        "decisión sobre las columnas tabulares de nadie (eso es 107-C0)"
    )
    entorno = medicion.get("entorno") or {}
    if entorno.get("maquina"):
        dicho.append(
            f"los tiempos del catálogo son de {entorno['maquina']} con "
            f"{medicion.get('hilos')} hilo(s) y la carga que declara cada fila; no se extrapolan"
        )
    if (medicion.get("describe") or {}).get("tokenizador_de_fuera_del_nucleo"):
        dicho.append(
            "este proveedor se midió con un tokenizador que el núcleo no implementa "
            "(dependencia externa): un número que solo se obtiene instalando algo más "
            "no es el mismo número que daría el núcleo pelado"
        )
    if dependencias is None:
        dicho.append(
            "esta declaración no dice con qué versiones de qué bibliotecas se ejecutó el "
            "proveedor: quien la compuso no las pasó"
        )
    return dicho


def componente_de_terceros(
    proveedor_id: str,
    *,
    longitud: Mapping[str, Any],
    ejecutado_por: str,
    dependencias: Mapping[str, Any] | None = None,
    idiomas: Sequence[str] = (),
    artefacto: Path | None = None,
) -> dict[str, Any]:
    """El componente `embedding_provider` de este proveedor, entero.

    `longitud` es la tercera cosa del invariante 6 y llega de fuera **a
    propósito**: la fija el código que ejecuta el proveedor
    (`matrixai_engines.embeddings.LONGITUDES_FIJADAS` para el multilingüe y el
    estático de cartera), con `tokens`, `origen`, `fuente`, `por_que` y
    `fijada_en`. Aquí se comprueba contra el catálogo, no se inventa.

    `ejecutado_por` no tiene valor por omisión: un `"matrixai-core"` por
    defecto sería mentira el día que lo ejecute otro, y la declaración dice
    QUIÉN corre unos pesos ajenos.

    `idiomas` añade idiomas al veredicto además de los que el catálogo midió:
    preguntar por uno no medido no devuelve un hueco, devuelve `no_medido` con
    su motivo.
    """
    if not str(ejecutado_por or "").strip():
        raise DeclaracionImposible(
            "sin `ejecutado_por` no hay declaración: unos pesos de terceros los corre "
            "alguien concreto, y el expediente dice quién")

    datos = _cargar(artefacto)
    fila = _fila(datos, proveedor_id)
    medicion = _medicion(fila, proveedor_id)
    describe = medicion.get("describe") or {}
    candidato = por_id(proveedor_id)

    licencia = dict(fila.get("licencia") or {})
    if not licencia:
        raise DeclaracionImposible(f"{proveedor_id}: el catálogo no trae su licencia")
    if licencia.get("compatible") is not True:
        raise DeclaracionImposible(
            f"{proveedor_id}: licencia {licencia.get('spdx')!r} incompatible "
            f"({licencia.get('motivo_si_no')}). Descargar y aceptar no elimina "
            "restricciones (107, invariante 5): no entra por bueno que sea."
        )
    aceptacion = fila.get("aceptacion_de_licencia")
    if not aceptacion:
        raise DeclaracionImposible(
            f"{proveedor_id}: nadie ha aceptado su licencia en esta máquina. Aceptar no "
            "elimina restricciones, pero no haber aceptado nada deja al paquete sin decir "
            "de dónde salió el permiso para traerse estos pesos."
        )

    pesos = _fichero(fila, describe.get("modelo") or candidato.modelo, proveedor_id, "los pesos")
    tokenizer = _fichero(fila, "tokenizer.json", proveedor_id, "el tokenizer")
    comprobada_contra = _comprobar_la_longitud(longitud, describe, proveedor_id)

    idiomas_medidos: dict[str, Any] = {}
    pedidos = list((medicion.get("coherencia_por_documento") or {}).keys()) + list(idiomas)
    for idioma in sorted(dict.fromkeys(pedidos)):
        v = cobertura.veredicto(proveedor_id, idioma, artefacto=artefacto)
        idiomas_medidos[idioma] = {
            "estado": v.estado,
            "auc_coherencia_por_documento": v.auc,
            "tokens_por_palabra": v.tokens_por_palabra,
            "frase": v.frase,
            "umbral_cubierto": cobertura.UMBRAL_CUBIERTO,
            "azar": 0.5,
        }
    if not idiomas_medidos:
        raise DeclaracionImposible(
            f"{proveedor_id}: el catálogo no mide su cobertura en ningún idioma, y el "
            "invariante 7 pide declararla por proveedor")

    componente: dict[str, Any] = {
        "componente": "embedding_provider",
        "ejecutado_por": str(ejecutado_por),
        "id": proveedor_id,
        "familia": fila.get("familia"),
        "repo": fila.get("repo"),
        "revision": fila.get("revision"),
        "licencia": licencia,
        "aceptacion_de_licencia": dict(aceptacion),
        "digest_de_los_pesos": pesos,
        "digest_del_tokenizer": tokenizer,
        "ficheros_fijados": [dict(f) for f in (fila.get("ficheros") or [])],
        "dimension": medicion.get("dimension"),
        "pooling": describe.get("pooling"),
        "normaliza": describe.get("normaliza"),
        "compuesto_segun": describe.get("leido_de"),
        "tokenizador": {
            "familia": describe.get("tokenizador"),
            "de_fuera_del_nucleo": describe.get("tokenizador_de_fuera_del_nucleo"),
            "tokens_especiales": describe.get("tokens_especiales"),
        },
        "longitud_maxima": {
            "tokens": longitud["tokens"],
            "origen": longitud["origen"],
            "fuente": longitud["fuente"],
            "por_que": longitud["por_que"],
            "fijada_en": longitud["fijada_en"],
            "comprobada_contra": comprobada_contra,
            "que_declara_el_paquete": list(
                describe.get("max_tokens_declarados_por_el_paquete") or []),
        },
        "idiomas_medidos": idiomas_medidos,
        "idiomas_segun_su_autor": {
            "cita": candidato.idiomas_segun_su_autor,
            "fuente": candidato.fuente_de_esa_cita,
            "advertencia": "es una CITA de la ficha del autor, no una medida; lo "
            "comprobado está en `idiomas_medidos`",
        },
        "opaco": True,
        "opacidad": FRASE_DE_OPACIDAD,
        "datos_de_ajuste": "ninguno",
        "pesos_en_la_imagen": False,
        "medido_en": dict(medicion.get("entorno") or {}),
        "catalogo_medido_el": datos.get("escrito_el"),
    }
    if dependencias is not None:
        componente["dependencias"] = {k: dict(v) for k, v in dependencias.items()}
    componente["lo_que_no_se_sabe"] = _lo_que_no_se_sabe(
        proveedor_id, fila, medicion, idiomas_medidos, dependencias)

    # LA SALIDA SE COMPRUEBA CONTRA EL MISMO VALIDADOR QUE USA EL PAQUETE. No
    # es paranoia: esta pieza compone desde un artefacto que puede cambiar, y
    # un componente a medias que llega hasta el manifiesto y allí se rechaza es
    # un fallo que aparece tres capas más lejos de donde se puede arreglar.
    from matrixai.export.terceros import exigir_completo

    exigir_completo(componente)
    return componente
