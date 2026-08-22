"""CONTRATO 81-C6 — el pipeline del caso, corriendo por el motor del 81.

La arquitectura del §16.3, nodo a nodo:

    tabulares → preproceso+encoder ┐
                                   ├→ fusión → clasificador → política → salida
    nota → texto+encoder+POOL ─────┘

**No se inventa maquinaria**: orquesta `pipelines/engine.py`, decide con
`pipelines/policy.py` y las reglas del caso están en `readmission.py`. Lo
que aporta este módulo es el CABLEADO — que es, otra vez, donde este
proyecto encuentra los huecos.

Dos decisiones que valen su comentario:

* **cada nodo tiene identidad y digest**, aunque no sea un modelo
  entrenado. El digest sale de su versión **y de las políticas que
  aplica**: si mañana alguien sube el tope de longitud del texto, el
  digest cambia y el recibo lo enseña. Un nodo sin identidad haría que el
  recibo dijera «se ejecutó algo» sin decir el qué;
* **la nota clínica NO sale de su nodo.** El nodo de texto devuelve
  rasgos y banderas, nunca el texto: así el dato sensible no puede
  llegar a la traza ni al recibo ni por accidente (§13.3), en vez de
  confiar en que nadie lo copie.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from matrixai.pipelines.engine import ejecutar_pipeline, emitir_recibo
from matrixai.reference.readmission import (
    ADVERTENCIA,
    PERFIL_DE_PRIVACIDAD,
    POLITICAS,
    campos_con_fuga_temporal,
    codificar_tabular,
    codificar_texto,
    decidir,
    fusionar,
    huella_sin_datos,
    preparar_texto,
)

__all__ = [
    "TALLA_TABULAR",
    "VOCABULARIO",
    "clasificador_del_registry",
    "construir_pipeline",
    "ejecutores",
    "evaluar_caso",
    "identidad_de_nodo",
    "registry_del_caso",
]

#: El vocabulario clínico del caso, **cerrado y declarado**. Sale del
#: dominio del ejemplo, no de los datos: un vocabulario deducido del
#: dataset cambiaría al regenerarlo y el modelo dejaría de encajar.
VOCABULARIO: tuple[str, ...] = (
    "disnea", "edema", "fiebre", "dolor", "caida", "confusion",
    "oxigeno", "diuretico", "insulina", "reingreso", "soledad",
)

#: Las 9 columnas que produce el encoder tabular (3 numéricas + 4 one-hot
#: + la opcional y su bandera de ausencia). Fija a propósito: es la mitad
#: del contrato de la fusión.
TALLA_TABULAR = 9


def identidad_de_nodo(version: str, politicas: dict[str, Any]) -> str:
    """El digest de un nodo: su versión Y las políticas que aplica.

    Que las políticas entren no es adorno. Si alguien sube
    `text_max_bytes` de 512 a 4096, el nodo hace algo distinto con la
    misma versión — y un recibo que declarara el mismo digest estaría
    describiendo una ejecución que no ocurrió así.
    """
    material = json.dumps({"version": version, "policies": politicas},
                          sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _politicas_de(*claves: str) -> dict[str, Any]:
    return {c: POLITICAS[c] for c in claves}

#: Qué política aplica cada nodo. Explícito para que el digest de cada uno
#: cambie SOLO cuando cambia algo suyo: si todos dependieran de todas, un
#: ajuste del umbral de confianza movería el digest del tokenizador y
#: nadie sabría por qué.
_NODOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("tabular", "tabular_encoder_version",
     ("missing_values", "unknown_category", "tabular_encoder_version")),
    ("texto", "text_encoder_version",
     ("text_length", "empty_text", "text_encoding", "text_max_bytes",
      "tokenizer", "text_encoder_version")),
    ("fusion", "fusion_version", ("fusion", "fusion_version")),
)


def registry_del_caso(clasificador_digest: str) -> dict[str, str]:
    """El mapa `modelo → digest` que el motor exige.

    El clasificador trae el suyo de fuera —es una entrada real del
    registry de P21— y los demás lo calculan de su versión y políticas.
    """
    mapa = {nombre: identidad_de_nodo(POLITICAS[clave], _politicas_de(*pols))
            for nombre, clave, pols in _NODOS}
    mapa["clasificador"] = clasificador_digest
    return mapa


def construir_pipeline(clasificador_digest: str, *, timeout_s: int = 120) -> dict[str, Any]:
    """El grafo del §16.3, con cada nodo atado a su digest."""
    mapa = registry_del_caso(clasificador_digest)
    return {
        "pipeline_id": "readmission-30d",
        "version": "1.0",
        "timeout_s": timeout_s,
        "nodes": [
            {"id": "tabular", "model": "tabular", "entry_hash": mapa["tabular"],
             "kind": "tabular"},
            {"id": "texto", "model": "texto", "entry_hash": mapa["texto"],
             "kind": "texto"},
            {"id": "fusion", "model": "fusion", "entry_hash": mapa["fusion"],
             "kind": "fusion", "depends_on": ["tabular", "texto"]},
            {"id": "clasificador", "model": "clasificador",
             "entry_hash": mapa["clasificador"], "kind": "clasificador",
             "depends_on": ["fusion"]},
        ],
    }


def ejecutores(
    registro: dict[str, Any],
    nota: Any,
    clasificador: Callable[[list[float]], dict[str, float]],
) -> dict[str, Callable[..., Any]]:
    """Los cuatro ejecutores del caso.

    El registro y la nota se cierran aquí dentro a propósito: si viajaran
    como «entrada» de los nodos, el dato clínico acabaría en el
    transporte y de ahí en el digest de entrada de la traza.
    """

    def _tabular(*, entradas, nodo, contexto):
        return codificar_tabular(registro)

    def _texto(*, entradas, nodo, contexto):
        preparado = preparar_texto(nota)
        codificado = codificar_texto(preparado, VOCABULARIO)
        # Del nodo de texto sale TODO menos el texto.
        return {
            **codificado,
            "bytes": preparado["bytes"],
            "truncated": preparado["truncated"],
            "truncated_at": preparado["truncated_at"],
            "empty": preparado["empty"],
            "text_digest": preparado["digest"],
            "tokenizer": preparado["tokenizer"],
        }

    def _fusion(*, entradas, nodo, contexto):
        return fusionar(entradas["tabular"], entradas["texto"],
                        tallas_esperadas=(TALLA_TABULAR, len(VOCABULARIO)))

    def _clasificador(*, entradas, nodo, contexto):
        fusionado = entradas["fusion"]
        return {"probabilities": clasificador(fusionado["vector"]),
                "fusion": fusionado}

    return {"tabular": _tabular, "texto": _texto, "fusion": _fusion,
            "clasificador": _clasificador}


def clasificador_del_registry(
    registry: Any, nombre: str, version: str,
    etiquetas: tuple[str, ...] = ("no_reingreso", "reingreso"),
) -> tuple[Callable[[list[float]], dict[str, float]], str, str]:
    """El clasificador REAL de P21, y su digest.

    Devuelve las dos cosas juntas a propósito: el digest que se declara en
    el pipeline tiene que ser el de la entrada que de verdad va a
    ejecutar. Pedirlos por separado permitiría declarar uno y correr
    otro, y el recibo describiría un modelo que no fue.

    **Y se comprueba la integridad antes de usarlo**: un modelo tocado no
    decide sobre nadie, y menos aquí.
    """
    from matrixai.parser import parse_text
    from matrixai.parameters import ParameterSet
    from matrixai.registry.model_registry import VerificationError
    from matrixai.runtime import MatrixAIRuntime

    try:
        intacto = registry.verify(nombre, version)
    except VerificationError as exc:
        raise ValueError(
            f"{nombre}@{version} no pasa su verificación de integridad ({exc}): "
            "un modelo alterado no decide sobre un caso clínico, ni de "
            "demostración") from exc
    if not intacto:
        raise ValueError(f"{nombre}@{version} no está íntegro")

    entrada = registry.get(nombre, version)
    directorio = Path(registry.layout.entry_dir(nombre, version))
    programa = parse_text((directorio / "model.mxai").read_text(encoding="utf-8"))
    parametros = ParameterSet.from_dict(
        json.loads((directorio / "params.json").read_text(encoding="utf-8"))
    ).runtime_parameters()

    vector_esperado = entrada.input_type.get("name") if isinstance(entrada.input_type, dict) else None
    talla = entrada.input_type.get("size") if isinstance(entrada.input_type, dict) else None
    campos = [c.name for v in programa.vectors if v.name == vector_esperado for c in [v]][:0]
    campos = next((list(v.fields) for v in programa.vectors if v.name == vector_esperado), [])

    def clasificar(vector: list[float]) -> dict[str, float]:
        if talla is not None and len(vector) != talla:
            # Antes de ejecutar: el modelo declara su talla y ejecutar con
            # otra daría una predicción sobre un vector que no es el suyo.
            raise ValueError(
                f"{nombre}@{version} espera {talla} entradas y recibe "
                f"{len(vector)}")
        entrada_dict = {vector_esperado: dict(zip(campos, vector))}
        salida = MatrixAIRuntime().run(programa, entrada_dict, parameters=parametros)
        crudo = salida["state"].get("riesgo") or salida["state"].get(
            next(iter(salida["state"])))
        valores = list(crudo) if isinstance(crudo, list) else [float(crudo)]
        if len(valores) != len(etiquetas):
            raise ValueError(
                f"el modelo produce {len(valores)} salidas y el caso declara "
                f"{len(etiquetas)} clases: nombrarlas a medias inventaría una")
        return {etiqueta: float(v) for etiqueta, v in zip(etiquetas, valores)}

    # La COORDENADA, además del digest: `nombre@version` es lo que un
    # humano puede ir a buscar al registry, y el digest lo que se
    # comprueba. Sin la primera, el recibo dice «corrió algo con este
    # hash» y hay que salir a buscar cuál era.
    return clasificar, entrada.entry_hash, f"{nombre}@{version}"


def evaluar_caso(
    registro: dict[str, Any],
    nota: Any,
    *,
    momento_de_decision: str,
    clasificador: Callable[[list[float]], dict[str, float]],
    clasificador_digest: str,
    clasificador_version: str = "registry",
    receipt_id: str = "readmission-1",
    locale: str = "es",
) -> dict[str, Any]:
    """Un caso, de punta a punta: pipeline, política, salida y recibo.

    **La advertencia del §16.2 viaja SIEMPRE**, incluso cuando el pipeline
    se rompe: si solo apareciera en el camino bueno, el único sitio donde
    alguien lee un resultado sin ella sería justo el raro.
    """
    pipeline = construir_pipeline(clasificador_digest)
    salida = ejecutar_pipeline(
        pipeline,
        registry=registry_del_caso(clasificador_digest),
        ejecutores=ejecutores(registro, nota, clasificador),
        contexto={"momento_de_decision": momento_de_decision},
    )

    base = {
        "pipeline": salida["report"],
        "trace": salida["trace"],
        "privacy": dict(PERFIL_DE_PRIVACIDAD),
        "disclaimer": ADVERTENCIA.get(locale, ADVERTENCIA["es"]),
        "decision_moment": momento_de_decision,
    }

    if salida["report"]["status"] != "ok":
        # Un pipeline que no terminó NO produce una decisión: devolver una
        # clase «por si acaso» sería exactamente lo que este contrato
        # existe para impedir.
        return {**base, "outcome": "not_run",
                "reason": salida["report"]["reason"]}

    fusionado = salida["values"]["clasificador"]["value"]["fusion"]
    tabular = salida["values"]["tabular"]["value"]
    texto = salida["values"]["texto"]["value"]
    fuga = campos_con_fuga_temporal(registro, momento_de_decision)

    decision = decidir(
        salida["values"]["clasificador"]["value"]["probabilities"],
        contexto={
            "datos_completos": tabular["complete"],
            "categoria_conocida": tabular["known_categories"],
            "sin_fuga_temporal": not fuga,
            "hay_senal_de_texto": fusionado["text_signal_present"],
        },
        locale=locale,
    )

    recibo = emitir_recibo(salida, pipeline=pipeline, receipt_id=receipt_id,
                           purpose="readmission-30d (demostración)")
    # Lo que el recibo tiene que poder decir del caso, y NADA del paciente:
    # versiones, huellas y por qué se decidió lo que se decidió.
    recibo["checks"]["policy_results"].append(decision["policy"])
    recibo["subject"]["disclaimer"] = base["disclaimer"]
    # `models` YA lo pone `emitir_recibo` desde la 2ª pasada de auditoría:
    # aquí se rellenaba a mano, y eso era exactamente la señal de que el
    # emisor daba de menos. Lo único que este caso añade es la VERSIÓN
    # declarada de cada nodo, que el motor no puede saber.
    versiones = {nombre: POLITICAS[clave] for nombre, clave, _ in _NODOS}
    # Y la del clasificador, que el motor no puede saber: es su entrada
    # del registry, no un `"declared-in-pipeline"` que no identifica nada.
    versiones["clasificador"] = clasificador_version
    for modelo in recibo["models"]:
        if modelo["model_id"] in versiones:
            modelo["version"] = versiones[modelo["model_id"]]

    # Y el `input`: el motor no ve el dato del caso —solo lo que produjo
    # cada nodo—, así que la huella la pone quien SÍ lo tiene. Se
    # completa, no se reemplaza: `entry_nodes` y `raw_data_included` los
    # sabe el motor y son suyos.
    recibo["input"].update({
        "payload_digest": huella_sin_datos(json.dumps(registro, sort_keys=True, default=str)),
        "text_digest": texto["text_digest"],
        "data_cutoff": momento_de_decision,
    })

    return {
        **base,
        **decision,
        "temporal_leakage": fuga,
        "text": {k: texto[k] for k in ("bytes", "truncated", "truncated_at",
                                       "empty", "tokenizer")},
        "tabular": {k: tabular[k] for k in ("missing_required", "missing_optional",
                                            "out_of_range", "unknown_categories")},
        "receipt": recibo,
    }
