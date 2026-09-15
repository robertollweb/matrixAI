# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""El ML-BOM de un paquete, en CycloneDX — contrato 86-C3.

POR QUÉ. Un ML-BOM (la lista de materiales de un modelo: qué modelo es, con qué
datos se hizo, en qué entorno y con qué resultados) es lo que ya aparece en
pliegos de compra, y lo que frena su adopción es que **casi nadie lo genera sin
trabajo manual**. Aquí no hay trabajo manual: `reproduce.json` ya tiene casi
todos los campos, así que esto es un TRADUCTOR, no una investigación.

DOS DECISIONES QUE NO SON DE COMODIDAD:

* **Es determinista.** Ni `uuid4()` ni la hora de ahora: el número de serie se
  deriva del propio digest del manifiesto. Un BOM que cambia cada vez que se
  genera no se puede comparar, ni firmar, ni meter en un paquete que promete
  reproducirse — y este producto entero va de eso.
* **Lo que no se sabe NO se rellena.** Un campo inventado en un BOM es peor que
  un campo ausente: quien lo lea creerá que lo sabe. Lo ausente se puede ver
  con `matrixai bom --missing`.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from matrixai.export.terceros import (
    CLAVE_EN_EL_MANIFIESTO as _CLAVE_PROVEEDOR,
    componente_para_el_bom,
    referencia_en_el_bom,
    validar as _validar_proveedor,
)

__all__ = ["BomNoDisponible", "SPEC_VERSION", "ml_bom", "lo_que_falta"]

#: La versión del esquema contra la que se emite y se VALIDA. La 1.5 fue la
#: primera con soporte de modelos y datos; la 1.7 es la vigente. Se emite 1.6
#: porque es la que hoy ingiere más herramienta instalada — y se declara aquí
#: para que cambiarla sea una decisión y no un descuido.
SPEC_VERSION = "1.6"

#: El espacio de nombres del que se derivan los números de serie. Es NUESTRO:
#: uno inventado en un dominio ajeno haría que dos productos distintos
#: generaran el mismo urn para cosas distintas.
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://matrixaistudio.org/mlbom")


class BomNoDisponible(RuntimeError):
    """No hay paquete del que sacar el BOM."""


def _propiedad(nombre: str, valor: Any) -> dict[str, str] | None:
    """Una propiedad de CycloneDX, o nada si no hay valor.

    `None`, `""` y `[]` NO se convierten en `"None"`: un ausente escrito como
    texto es la peor clase de dato, porque parece uno.
    """
    if valor in (None, "", [], {}):
        return None
    if isinstance(valor, (dict, list)):
        valor = json.dumps(valor, sort_keys=True, ensure_ascii=False)
    return {"name": nombre, "value": str(valor)}


def ml_bom(bundle_dir: str | Path) -> dict[str, Any]:
    """El ML-BOM del paquete, listo para validar contra el esquema oficial."""
    bundle = Path(bundle_dir)
    manifiesto = bundle / "reproduce.json"
    if not manifiesto.is_file():
        raise BomNoDisponible(
            f"{bundle}: no reproduce.json — un ML-BOM se escribe desde lo que el "
            f"run registró, y este paquete no registra nada")
    try:
        m = json.loads(manifiesto.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BomNoDisponible(f"{manifiesto}: {exc}") from exc

    gen = m.get("generation") or {}
    art = m.get("artifacts") or {}
    env = m.get("environment") or {}
    digest_manifiesto = str(m.get("manifest_sha256") or "")

    # El número de serie SALE DEL CONTENIDO: mismo paquete, mismo urn.
    serial = uuid.uuid5(_NAMESPACE, digest_manifiesto or str(bundle.name))

    propiedades = [
        _propiedad("matrixai:generation.mode", gen.get("mode")),
        _propiedad("matrixai:generation.seeds", gen.get("seeds")),
        _propiedad("matrixai:epochs.declared", gen.get("epochs_declared")),
        _propiedad("matrixai:epochs.effective", gen.get("epochs_effective")),
        _propiedad("matrixai:epochs.ran", gen.get("epochs_ran")),
        _propiedad("matrixai:backend", gen.get("backend")),
        _propiedad("matrixai:device", gen.get("device")),
        _propiedad("matrixai:warm_start", gen.get("warm_start")),
        _propiedad("matrixai:excluded_identifiers", gen.get("excluded_identifiers")),
        _propiedad("matrixai:reproducible", m.get("reproducible")),
        _propiedad("matrixai:manifest_sha256", digest_manifiesto or None),
        _propiedad("matrixai:recipe.sha256", (art.get("recipe") or {}).get("sha256")),
        _propiedad("matrixai:training_contract.sha256", (art.get("training") or {}).get("sha256")),
    ]

    # EL COMPONENTE DE TERCEROS (107-C2). Si el paquete declara un proveedor de
    # embeddings, el BOM lo enumera con su licencia; si lo declara a medias, NO
    # se enumera a medias —un componente sin licencia se lee como si estuviera
    # declarado— y lo que sale es qué le falta, por su nombre.
    proveedor = m.get(_CLAVE_PROVEEDOR)
    faltan_del_proveedor = _validar_proveedor(proveedor) if proveedor is not None else []
    if proveedor is not None:
        if faltan_del_proveedor:
            propiedades.append(_propiedad("matrixai:embedding_provider.missing",
                                          ", ".join(faltan_del_proveedor)))
        else:
            propiedades.append(_propiedad("matrixai:embedding_provider",
                                          str(proveedor.get("id"))))


    # LAS MÉTRICAS, con lo que hace falta para poder contrastarlas. Una métrica
    # sin decir sobre qué se midió no es un resultado, es un número.
    metricas = []
    for metrica in m.get("metrics") or []:
        if not isinstance(metrica, dict) or metrica.get("value") is None:
            continue
        entrada: dict[str, Any] = {
            "type": str(metrica.get("name")),
            "value": str(metrica.get("value")),
        }
        # NADA DE INTERVALOS DE CONFIANZA INVENTADOS (auditoría externa del
        # 2026-08-25, hallazgo 4). Esto fabricaba un `confidenceInterval` de
        # amplitud CERO —`lowerBound == upperBound == valor`— que afirma que la
        # métrica es exacta, y eso **no lo ha medido nadie**. Y encima el
        # contexto que se había recogido (partición, dataset, dirección) se
        # perdía: la variable se construía y no se escribía en ninguna parte.
        #
        # `performanceMetric` de CycloneDX 1.6 solo admite `type`, `value`,
        # `slice` y `confidenceInterval` (`additionalProperties: false`,
        # comprobado contra el esquema), así que:
        #   · la PARTICIÓN va en `slice`, que es literalmente para eso;
        #   · y lo que no cabe —el digest del dataset y la dirección— viaja como
        #     PROPIEDAD del componente, que es la vía estándar, en vez de
        #     desaparecer o de disfrazarse de otra cosa.
        if metrica.get("split"):
            entrada["slice"] = str(metrica["split"])
        metricas.append(entrada)
        nombre_metrica = str(metrica.get("name"))
        for clave, valor_extra in (("dataset_sha256", metrica.get("dataset_sha256")),
                                   ("direction", metrica.get("direction"))):
            propiedad = _propiedad(f"matrixai:metric.{nombre_metrica}.{clave}", valor_extra)
            if propiedad is not None:
                propiedades.append(propiedad)

    propiedades = [p for p in propiedades if p is not None]

    modelo: dict[str, Any] = {
        "type": "machine-learning-model",
        "bom-ref": f"model:{(art.get('model') or {}).get('sha256', 'unknown')[:16]}",
        "name": bundle.name,
    }
    hash_modelo = (art.get("model") or {}).get("sha256")
    if hash_modelo:
        modelo["hashes"] = [{"alg": "SHA-256", "content": hash_modelo}]
    if propiedades:
        modelo["properties"] = propiedades
    ficha: dict[str, Any] = {}
    if metricas:
        ficha["quantitativeAnalysis"] = {"performanceMetrics": metricas}
    consideraciones = _consideraciones(m, gen)
    if consideraciones:
        ficha["considerations"] = consideraciones
    if ficha:
        modelo["modelCard"] = ficha

    componentes: list[dict[str, Any]] = [modelo]

    dataset = art.get("dataset") or {}
    if dataset.get("sha256"):
        componente_datos: dict[str, Any] = {
            "type": "data",
            "bom-ref": f"data:{str(dataset['sha256'])[:16]}",
            "name": "training-dataset",
            "data": [{
                "type": "dataset",
                "name": "training-dataset",
                # De dónde salen estos datos, dicho con todas las letras: si son
                # sintéticos, quien lea el BOM tiene que saberlo aquí y no
                # deducirlo de una propiedad suelta.
                "contents": {"properties": [
                    p for p in [
                        _propiedad("matrixai:dataset.sha256", dataset.get("sha256")),
                        _propiedad("matrixai:dataset.rows", dataset.get("rows")),
                        _propiedad("matrixai:dataset.origin",
                                   "synthetic" if (art.get("recipe") or {}).get("sha256")
                                   else None),
                    ] if p is not None]},
            }],
        }
        componentes.append(componente_datos)

    # EL PROVEEDOR DE EMBEDDINGS, enumerado como lo que es: otro
    # `machine-learning-model` del BOM, con su licencia y sus enlaces al origen.
    # Y ENLAZADO al modelo por `dependencies`, que es la vía que CycloneDX tiene
    # para decir «este modelo no funciona sin aquel»: una lista de componentes
    # sueltos no dice quién usa a quién, y quien lea el BOM tiene que poder ver
    # que sin esos pesos ajenos el paquete no predice.
    dependencias_del_bom: list[dict[str, Any]] = []
    if proveedor is not None and not faltan_del_proveedor:
        componente_proveedor = componente_para_el_bom(proveedor)
        componentes.append(componente_proveedor)
        dependencias_del_bom.append({
            "ref": modelo["bom-ref"],
            "dependsOn": [referencia_en_el_bom(proveedor)],
        })
        dependencias_del_bom.append({"ref": referencia_en_el_bom(proveedor), "dependsOn": []})

    bom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC_VERSION,
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "tools": {"components": [{
                "type": "application", "name": "matrixai",
                "version": str(env.get("matrixai_version") or "unknown"),
            }]},
            "component": dict(modelo),
        },
        "components": componentes,
    }
    if dependencias_del_bom:
        bom["dependencies"] = dependencias_del_bom
    return bom


def _consideraciones(m: dict[str, Any], gen: dict[str, Any]) -> dict[str, Any]:
    """Lo que hay que saber ANTES de usar este modelo.

    Un paquete no reproducible y uno con datos sintéticos son dos avisos
    distintos, y los dos van aquí porque es donde los busca quien lee un BOM.
    """
    limitaciones: list[str] = []
    if not m.get("reproducible"):
        motivo = str(m.get("reproducible_reason") or "").strip()
        limitaciones.append(
            "This package does not declare itself reproducible" + (f": {motivo}" if motivo else ""))
    # `or {}` y no `.get("recipe", {})`: cuando NO hay receta la clave existe
    # con valor `None` —el caso de lluvia, con datos reales—, así que el valor
    # por defecto nunca entra y esto reventaba con un `AttributeError`. Salió
    # al validar los tres casos de la galería, no los dos que tenían receta.
    if ((m.get("artifacts") or {}).get("recipe") or {}).get("sha256"):
        limitaciones.append(
            "The training data is SYNTHETIC, generated by the recipe shipped in "
            "this package. A model trained this way is not validated on real data")
    if gen.get("mode") == "random":
        limitaciones.append(
            "The dataset was generated in random mode: the target does not depend "
            "on the inputs, so the model cannot have learned anything")
    # 107-C2: un componente de terceros declarado A MEDIAS es un aviso, no un
    # detalle de formato. Quien lea el BOM tiene que enterarse de que el paquete
    # dice usar unos pesos ajenos y no dice cuáles.
    proveedor = m.get(_CLAVE_PROVEEDOR)
    if proveedor is not None:
        faltan = _validar_proveedor(proveedor)
        if faltan:
            limitaciones.append(
                "This package declares a third-party embedding provider whose "
                f"declaration is incomplete (missing: {', '.join(faltan)}): what "
                "produced the model's text columns cannot be identified from here")
    return {"technicalLimitations": limitaciones} if limitaciones else {}


def lo_que_falta(bom: dict[str, Any]) -> list[str]:
    """Lo que este BOM NO puede decir, y por qué. Se imprime con `--missing`.

    Un BOM con huecos explicados es útil; uno con huecos mudos se lee como si
    no hubiera nada que decir.
    """
    faltan: list[str] = []
    componente = (bom.get("metadata") or {}).get("component") or {}
    ficha = componente.get("modelCard") or {}
    if not ficha.get("quantitativeAnalysis"):
        faltan.append("performance metrics: the package publishes none")
    if not ficha.get("considerations"):
        faltan.append("considerations: nothing to warn about was recorded")
    nombres = {p.get("name") for p in componente.get("properties") or []}
    for clave, motivo in (
        ("matrixai:generation.mode", "the package does not declare its generation mode"),
        ("matrixai:generation.seeds", "the run did not record its seeds"),
        ("matrixai:device", "the run did not record which machine it ran on"),
        ("matrixai:recipe.sha256", "this model has no data recipe: its data cannot be regenerated"),
    ):
        if clave not in nombres:
            faltan.append(f"{clave}: {motivo}")
    # LA LICENCIA, que un BOM de compras busca y aquí no hay (auditoría externa
    # del 2026-08-25). No se inventa una: se dice que no consta.
    if not componente.get("licenses"):
        faltan.append(
            "licenses: the package does not declare a license for the model, and one "
            "is not invented here")
    # 107-C2. Se dice QUÉ le falta al componente de terceros, no «hay un
    # problema»: el que lee esto es quien tiene que ir a arreglarlo.
    for propiedad in componente.get("properties") or []:
        if propiedad.get("name") == "matrixai:embedding_provider.missing":
            faltan.append(
                "third-party embedding provider: the package declares one but its "
                f"declaration is incomplete ({propiedad.get('value')}), so it is not "
                "enumerated as a component — half a component reads like a whole one")
    # Y DÓNDE SE FUE EL CONTEXTO DE LAS MÉTRICAS: `performanceMetric` de
    # CycloneDX no admite más campos que `type`, `value`, `slice` y
    # `confidenceInterval`, así que el digest del dataset y la dirección viajan
    # como propiedades. Quien lea el BOM tiene que saber dónde mirar.
    metricas = ((ficha.get("quantitativeAnalysis") or {}).get("performanceMetrics") or [])
    if metricas and not any(str(n or "").startswith("matrixai:metric.") for n in nombres):
        faltan.append(
            "metric context: the dataset digest and direction of each metric could not "
            "be attached to the metric itself (CycloneDX does not allow it) and are not "
            "in the component properties either")
    return faltan
