# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""CONTRATO 111-C1 — lo que la GitHub Action ejecuta de verdad.

`action.yml` es tres pasos de YAML que nadie puede probar fuera de
GitHub. Todo lo que DECIDE algo vive aquí, en Python, para que la suite
lo pruebe y se pueda sabotear: qué alcances quedaron sin realizar, qué
pone el resumen, con qué código sale el trabajo y si hay que instalar
`matrixai-engines`. El YAML solo pasa entradas y recoge el resumen.

**LO QUE ESTE MÓDULO EXISTE PARA IMPEDIR.** Un tick verde cuando la
verificación no pudo comprobar tres de las cuatro evidencias convierte
«no se midió» en «está bien» a la vista de un equipo entero, y en CI
nadie abre el registro de un trabajo verde. Así que:

* nada puede poner verde un `FAIL` —no hay entrada que lo permita—;
* **esta acción nunca es más verde que `verify`**: si `verify` sale con
  2 o 3, el trabajo sale en rojo pase lo que pase. `require` solo puede
  APRETAR (exigir alcances que `verify` deja en `NOT_RUN`), nunca
  aflojar;
* los alcances no realizados se listan SIEMPRE en el resumen con su
  motivo literal y se anotan con `::warning`, aunque el trabajo acabe
  verde por no estar exigidos.

**LOS CUATRO ALCANCES SON LOS DE `verify`, CON SUS NOMBRES.** `manifest`,
`R1`, `training` y `R3` (82-C2). No se renombran a las cinco evidencias
del 106-C3 (`integridad`, `metrics_recomputed`, `inference_repeated`,
`training_repeated`, `selection_recomputed`) porque **no son el mismo
conjunto** y bautizar una cosa con el nombre de otra es exactamente la
«segunda semántica» que el 111 prohíbe en su invariante 6. Hoy las cinco
del 106-C3 no tienen línea de órdenes —`verificar_estudio_completo()` es
una función de Python, sin `console_scripts`— y por eso esta acción no
las puede enseñar: lo dice el resumen en vez de callarlo.

**`matrixai-engines` SOLO SI EL PAQUETE LO PIDE.** Instalarlo siempre
sería una línea menos y haría que un paquete que declara no necesitarlo
se verificara igual, tapando justo el fallo que `verify` existe para
encontrar. Se mira dónde un paquete PUEDE pedirlo de verdad (ver
`peticion_de_engines`) y se declara qué se miró.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = [
    "ALCANCES", "ESTADOS_REALIZADOS", "REQUERIDOS_POR_DEFECTO", "SALIDAS",
    "Alcance", "PeticionDeEngines", "Veredicto",
    "VALORES_VERDADEROS", "VALORES_FALSOS",
    "alcances_del_informe", "decidir", "peticion_de_engines",
    "peticion_de_engines_de_la_ruta",
    "anotaciones", "resumen_markdown", "main", "parser",
]

#: Los cuatro alcances de `matrixai verify`, en su orden y con su nombre.
ALCANCES = ("manifest", "R1", "training", "R3")

#: Un alcance está REALIZADO cuando se intentó y hubo veredicto. `NOT_RUN`
#: («no se intentó») e `INCOMPARABLE` («no se pudo») no son veredictos: son
#: ausencia de veredicto, por motivos distintos, y ninguno es un aprobado.
ESTADOS_REALIZADOS = ("PASS", "FAIL")

#: Lo que se exige por defecto, y el porqué —que es la decisión de este
#: corte y no una omisión—:
#:
#: * `manifest` y `R1` son los dos alcances que una máquina ajena SÍ puede
#:   realizar, y los dos que cazan una manipulación: integridad de cada
#:   artefacto por su `sha256`, y el dataset regenerado con el digest
#:   exacto que el manifiesto declara.
#: * `training` y `R3` necesitan `--retrain`. Y `R3`, además, solo sabe
#:   un alcance de tolerancia, `same_environment_same_seed`, que compara
#:   el digest del entorno ENTERO (versión de Python con su compilador,
#:   plataforma, numpy/onnx/onnxruntime/torch). Un runner de GitHub no lo
#:   iguala nunca para un paquete construido en otro sitio: MEDIDO el
#:   2026-09-15 sobre un paquete íntegro, `--retrain` da `R3
#:   INCOMPARABLE` y código 3. Exigirlos por defecto pondría en rojo a
#:   todos los paquetes honestos, y un rojo que no es un fallo enseña a
#:   poner `continue-on-error` — y entonces no se comprueba nada.
#:
#: No exigido NO es escondido: los dos siguen saliendo en el resumen y en
#: las anotaciones. Quien quiera el trabajo en rojo por ellos escribe
#: `require: all` en su flujo, que es una decisión suya y por escrito.
REQUERIDOS_POR_DEFECTO = ("manifest", "R1")

#: Códigos de salida de la ACCIÓN. Los dos de en medio son los mismos que
#: `matrixai verify` (82-C2) y significan lo mismo, a propósito: un guion
#: que ya sabía leerlos no aprende un segundo idioma.
SALIDAS = {
    "ok": 0,                # todo lo exigido se realizó y nada falló
    "error_de_uso": 1,      # la acción no pudo ni empezar (ruta, entrada mala)
    "fallo": 2,             # algo se comprobó y estaba mal
    "sin_realizar": 3,      # algo exigido no se pudo (o no se quiso) comprobar
}

#: Ficheros del paquete donde un paquete PUEDE pedir `matrixai-engines`,
#: medidos el 2026-09-15 sobre los paquetes reales de esta máquina, no
#: supuestos:
#:
#: * `requirements.txt` — el que llevan los paquetes de `export-bundle`
#:   (los tres de la galería lo traen: `numpy`, `onnxruntime`).
#: * `dependencias.json` — el del paquete de estudio (106-C1), con
#:   `{"engine":…, "dependencias":[…]}`; su `predict.py` dice por escrito
#:   que depende de `matrixai-engines` «declarado en `dependencias.json`
#:   del propio paquete».
#:
#: NO se mira `reproduce.json/environment/packages`: esa lista está
#: CERRADA en el core a `("numpy","onnx","onnxruntime","torch")` —los
#: paquetes cuya versión cambia el resultado—, así que `matrixai-engines`
#: no puede aparecer ahí ni aunque estuviera instalado. Buscarlo ahí sería
#: una sonda que nunca puede dar positivo.
#:
#: Tampoco se mira dentro de `space/`: desde que el Space pasó a ser
#: ESTÁTICO (revisión 2026-09-16, `matrixai/export/space.py`) ya no lleva
#: ningún fichero de dependencias propio —no hay Python que instalar en
#: una página que solo corre ONNX Runtime Web en el navegador—, así que
#: no hay nada ahí que este escaneo pudiera encontrar de todas formas.
FUENTES_DE_DEPENDENCIAS = ("requirements.txt", "dependencias.json")

_NOMBRE_ENGINES = "matrixai-engines"


def _normalizar_distribucion(nombre: str) -> str:
    """Nombre de distribución normalizado (PEP 503).

    `matrixai_engines`, `MatrixAI.Engines` y `matrixai-engines` son el
    mismo paquete para pip, y un `==` crudo diría que no lo pide.
    """
    return re.sub(r"[-_.]+", "-", nombre).lower()


@dataclass(frozen=True)
class Alcance:
    """Un alcance de `verify` tal y como lo devolvió, sin reinterpretar."""

    nombre: str
    estado: str
    motivo: str | None = None
    exigido: bool = False
    #: ¿Es uno de los cuatro de `ALCANCES`? Una etapa que `verify` traiga y
    #: esta acción no conozca NO desaparece —se enseña marcada—, porque un
    #: dibujo afirma por omisión: una fila que falta se lee como que no
    #: había nada que decir, y el recuento seguiría diciendo «4 of 4».
    conocido: bool = True

    @property
    def realizado(self) -> bool:
        return self.estado in ESTADOS_REALIZADOS


@dataclass(frozen=True)
class PeticionDeEngines:
    """¿El paquete pide `matrixai-engines`? Y sobre todo: ¿dónde se miró?

    `mirado_en` viaja SIEMPRE, también cuando la respuesta es «no». Un
    «no lo pide» sin decir dónde se ha mirado no se puede auditar.
    """

    pedido: bool
    mirado_en: tuple[str, ...] = ()
    declarado_en: str | None = None
    problema: str | None = None


@dataclass(frozen=True)
class Veredicto:
    alcances: tuple[Alcance, ...]
    codigo_de_verify: int
    codigo: int
    fallidos: tuple[str, ...] = ()
    sin_realizar: tuple[str, ...] = ()
    exigidos_sin_realizar: tuple[str, ...] = ()
    nota: str | None = None

    @property
    def titulo(self) -> str:
        """El titular del resumen, DEDUCIDO DEL CÓDIGO y no al revés.

        Esto miraba solo `fallidos` y `exigidos_sin_realizar`, así que un
        paquete sin `reproduce.json` verificado con `require: none`
        salía titulado «PARTIAL» —que se lee como «verde con matices»—
        mientras el trabajo se iba en rojo con un 3. MEDIDO el
        2026-09-15: 0 de 4 alcances realizados bajo un titular
        tranquilizador. El titular y el semáforo del trabajo tienen que
        decir lo mismo.
        """
        if self.fallidos:
            return "FAILED"
        if self.codigo != SALIDAS["ok"]:
            return "NOT VERIFIED"
        if self.sin_realizar:
            return "PARTIAL"
        return "VERIFIED"


def alcances_del_informe(
    informe: dict[str, Any], requeridos: Sequence[str] = REQUERIDOS_POR_DEFECTO,
) -> tuple[Alcance, ...]:
    """Los cuatro alcances del informe de `verify`, en su orden.

    Un alcance que el informe NO traiga no se da por bueno ni se omite:
    sale como `MISSING`, que no es un estado realizado. Un `verify` de
    otra versión que dejara de emitir una etapa haría desaparecer una
    fila del resumen, y una fila que falta se lee como que no había nada
    que decir.
    """
    etapas = informe.get("stages")
    if not isinstance(etapas, dict):
        etapas = {}
    exigidos = set(requeridos)
    salida: list[Alcance] = []
    for nombre in ALCANCES:
        etapa = etapas.get(nombre)
        if not isinstance(etapa, dict):
            salida.append(Alcance(nombre, "MISSING",
                                  "verify did not report this scope",
                                  nombre in exigidos))
            continue
        salida.append(Alcance(nombre, str(etapa.get("status") or "MISSING"),
                              etapa.get("reason") or None, nombre in exigidos))

    # Y al revés: una etapa que el informe TRAE y esta acción no conoce
    # tampoco se cae. `verify` puede crecer una quinta —la lista que manda
    # es la suya, no la de aquí—, y sin esto desaparecía del resumen
    # mientras el recuento seguía diciendo «4 of 4»: exactamente el «no se
    # midió» que se lee como «no había nada que mirar».
    for nombre, etapa in etapas.items():
        if nombre in ALCANCES:
            continue
        nombre = str(nombre)
        if isinstance(etapa, dict):
            salida.append(Alcance(nombre, str(etapa.get("status") or "MISSING"),
                                  etapa.get("reason") or None,
                                  nombre in exigidos, conocido=False))
        else:
            salida.append(Alcance(nombre, "MISSING",
                                  "verify reported this scope in a shape this "
                                  "action does not understand",
                                  nombre in exigidos, conocido=False))
    return tuple(salida)


def decidir(informe: dict[str, Any], codigo_de_verify: int,
            requeridos: Sequence[str] = REQUERIDOS_POR_DEFECTO) -> Veredicto:
    """El veredicto de la ACCIÓN a partir del informe y del código de `verify`.

    **La regla que sostiene todo lo demás: nunca más verde que `verify`.**
    Si `verify` salió con algo distinto de 0, esta acción sale en rojo
    aunque `require` no exija nada. `require` puede APRETAR —exigir un
    alcance que `verify` dejó en `NOT_RUN`, que para él es una elección
    de quien verifica y no un problema del paquete— y nada más. Sin esta
    regla, `require: none` convertiría un paquete imposible de comprobar
    en un tick verde, que es justo lo que este corte existe para impedir.
    """
    alcances = alcances_del_informe(informe, requeridos)
    fallidos = tuple(a.nombre for a in alcances if a.estado == "FAIL")
    sin_realizar = tuple(a.nombre for a in alcances if not a.realizado)
    exigidos_sin_realizar = tuple(a.nombre for a in alcances
                                  if a.exigido and not a.realizado)

    nota = None
    if fallidos:
        codigo = SALIDAS["fallo"]
    elif exigidos_sin_realizar:
        codigo = SALIDAS["sin_realizar"]
    elif codigo_de_verify != SALIDAS["ok"]:
        # `verify` vio algo que esta acción no exigía. No se rebaja.
        codigo = (codigo_de_verify if codigo_de_verify in (SALIDAS["fallo"],
                                                           SALIDAS["sin_realizar"])
                  else SALIDAS["sin_realizar"])
        nota = (f"`matrixai verify` exited {codigo_de_verify}; this action never "
                f"reports greener than verify, so the job fails too.")
    else:
        codigo = SALIDAS["ok"]
    return Veredicto(alcances=alcances, codigo_de_verify=codigo_de_verify,
                     codigo=codigo, fallidos=fallidos, sin_realizar=sin_realizar,
                     exigidos_sin_realizar=exigidos_sin_realizar, nota=nota)


# ---------------------------------------------------------------------------
# ¿El paquete pide `matrixai-engines`?
# ---------------------------------------------------------------------------

def _nombres_de_requirements(texto: str) -> set[str]:
    """Nombres de distribución de un `requirements.txt`, normalizados.

    No es un analizador de PEP 508 completo y no hace falta que lo sea:
    lo único que se decide con esto es si aparece UN nombre concreto.
    Se quitan comentarios, opciones (`-r`, `--index-url`), extras,
    marcadores y especificadores de versión.
    """
    nombres: set[str] = set()
    for linea in texto.splitlines():
        linea = linea.split("#", 1)[0].strip()
        if not linea or linea.startswith("-"):
            continue
        linea = linea.split(";", 1)[0].strip()          # marcador de entorno
        linea = re.split(r"\s*@\s*", linea, maxsplit=1)[0].strip()  # URL directa
        linea = re.split(r"[\[<>=!~ ]", linea, maxsplit=1)[0].strip()
        if linea:
            nombres.add(_normalizar_distribucion(linea))
    return nombres


def _texto_del_paquete(paquete: Path, nombre: str) -> str | None:
    """El contenido de un fichero del paquete, sea directorio o ZIP."""
    if paquete.is_dir():
        ruta = paquete / nombre
        try:
            return ruta.read_text(encoding="utf-8") if ruta.is_file() else None
        except OSError:
            return None
    return None


def peticion_de_engines(paquete: Path) -> PeticionDeEngines:
    """¿Este paquete pide `matrixai-engines`, y en qué fichero lo dice?

    Se mira solo donde un paquete PUEDE pedirlo (`FUENTES_DE_DEPENDENCIAS`)
    y se devuelve dónde se ha mirado, también cuando la respuesta es «no».
    """
    mirado: list[str] = []
    for nombre in FUENTES_DE_DEPENDENCIAS:
        texto = _texto_del_paquete(paquete, nombre)
        if texto is None:
            continue
        mirado.append(nombre)
        if nombre.endswith(".json"):
            try:
                datos = json.loads(texto)
            except ValueError:
                continue
            declaradas = datos.get("dependencias") if isinstance(datos, dict) else None
            nombres = {_normalizar_distribucion(str(d))
                       for d in (declaradas or []) if isinstance(d, str)}
        else:
            nombres = _nombres_de_requirements(texto)
        if _NOMBRE_ENGINES in nombres:
            return PeticionDeEngines(True, tuple(mirado), nombre)
    return PeticionDeEngines(False, tuple(mirado))


def _raiz_del_zip(paquete: Path, destino: Path) -> Path:
    """Descomprime el ZIP reusando la guarda del core y da su raíz.

    Se importa `_extraer_paquete` en vez de repetir aquí la regla de la
    carpeta única y el rechazo del `zip-slip`: dos sitios declarando lo
    mismo acaban divergiendo, y este es de los que no conviene que
    diverjan.
    """
    from matrixai.export.verify import _extraer_paquete

    return _extraer_paquete(paquete, destino)


def peticion_de_engines_de_la_ruta(ruta: Path) -> PeticionDeEngines:
    """`peticion_de_engines` sobre una ruta que puede ser un ZIP."""
    if ruta.is_dir():
        return peticion_de_engines(ruta)
    if ruta.is_file() and zipfile.is_zipfile(ruta):
        with tempfile.TemporaryDirectory(prefix="matrixai-ci-") as tmp:
            try:
                return peticion_de_engines(_raiz_del_zip(ruta, Path(tmp)))
            except Exception as exc:  # noqa: BLE001 - el motivo se declara
                return PeticionDeEngines(False, (), None,
                                         f"the archive could not be opened: {exc}")
    return PeticionDeEngines(False, (), None,
                             "the package is neither a directory nor a zip file")


# ---------------------------------------------------------------------------
# Resumen y anotaciones
# ---------------------------------------------------------------------------

_LEYENDA = (
    "`PASS`/`FAIL` are verdicts. `NOT_RUN` (not attempted) and "
    "`INCOMPARABLE` (could not be compared) are **not** verdicts and never "
    "count as a pass."
)

_NOTA_106 = (
    "Scopes come from `matrixai verify` (contract 82-C2): `manifest`, `R1`, "
    "`training`, `R3`. The five separate evidences of 106-C3 "
    "(`integridad`, `metrics_recomputed`, `inference_repeated`, "
    "`training_repeated`, `selection_recomputed`) have no command line yet, "
    "so this action cannot report them."
)


def resumen_markdown(veredicto: Veredicto, *, paquete: str, orden: Sequence[str],
                     engines: PeticionDeEngines,
                     engines_instalado: str | None = None,
                     modo_engines: str = "auto") -> str:
    """El resumen que se pega en `$GITHUB_STEP_SUMMARY`.

    Los alcances no realizados salen DOS veces a propósito: en la tabla,
    donde se leen junto a los demás, y en una lista aparte con su motivo
    literal, que es lo que hay que leer cuando el trabajo sale verde.
    """
    realizados = sum(1 for a in veredicto.alcances if a.realizado)
    lineas = [f"## matrixai verify — {veredicto.titulo}", ""]
    lineas.append(f"**{realizados} of {len(veredicto.alcances)} scopes were actually "
                  f"checked.**")
    if veredicto.fallidos:
        lineas.append(f"**Failed:** {', '.join('`%s`' % f for f in veredicto.fallidos)}.")
    if veredicto.sin_realizar:
        lineas.append("**NOT PERFORMED:** "
                      + ", ".join("`%s`" % s for s in veredicto.sin_realizar)
                      + ". A scope that was not performed is not a scope that passed.")
    if veredicto.nota:
        lineas.append(veredicto.nota)
    lineas += ["", f"* Package: `{paquete}`",
               f"* Command: `{' '.join(orden)}`",
               f"* `matrixai verify` exit code: `{veredicto.codigo_de_verify}` · "
               f"action exit code: `{veredicto.codigo}`", ""]

    lineas += ["| Scope | Status | Performed | Required | Reason |",
               "|---|---|---|---|---|"]
    for a in veredicto.alcances:
        motivo = (a.motivo or "").replace("|", "\\|")
        nombre = f"`{a.nombre}`" if a.conocido else f"`{a.nombre}` (unknown)"
        lineas.append(f"| {nombre} | `{a.estado}` | "
                      f"{'yes' if a.realizado else '**NO**'} | "
                      f"{'yes' if a.exigido else 'no'} | {motivo} |")
    lineas += ["", _LEYENDA, ""]

    if veredicto.sin_realizar:
        lineas.append("### Scopes NOT performed")
        for a in veredicto.alcances:
            if a.realizado:
                continue
            exigido = "required" if a.exigido else "not required by `require`"
            lineas.append(f"* `{a.nombre}` — `{a.estado}` ({exigido}) — "
                          f"{a.motivo or 'no reason given'}")
        lineas.append("")

    desconocidas = [a for a in veredicto.alcances if not a.conocido]
    if desconocidas:
        lineas.append("### Scopes this action does not know")
        lineas.append(
            "`matrixai verify` reported these, and they are **not** among the "
            "four this action knows ("
            + ", ".join(f"`{n}`" for n in ALCANCES)
            + "). They are listed and counted instead of dropped: a scope that "
              "vanishes from the summary reads as a scope that had nothing to "
              f"say, and the count above would still have said "
              f"{len(ALCANCES)} of {len(ALCANCES)}.")
        for a in desconocidas:
            lineas.append(f"* `{a.nombre}` — `{a.estado}` — "
                          f"{a.motivo or 'no reason given'}")
        lineas.append("")

    # Lo que PASÓ primero, y el porqué después. Esto decía «Not requested by
    # the package… It was **not** installed» mirando solo al paquete, así que
    # con `engines: matrixai-engines==0.3` —una petición EXPLÍCITA de quien
    # escribe el flujo— el resumen negaba una instalación que sí había pasado.
    modo = (modo_engines or "").strip()
    lineas.append("### matrixai-engines")
    if engines_instalado:
        porque = (f"the package asks for it in `{engines.declarado_en}`"
                  if engines.pedido and modo.lower() == "auto"
                  else f"the workflow asked for it explicitly (`engines: {modo}`)")
        lineas.append(f"**Installed** from `{engines_instalado}` — {porque}.")
    elif modo.lower() == "skip":
        lineas.append("**Not installed** — the workflow set `engines: skip`."
                      + (f" Note the package DOES ask for it in "
                         f"`{engines.declarado_en}`: verify may fail for that "
                         f"reason alone." if engines.pedido else ""))
    elif engines.problema:
        lineas.append(f"**Not installed** — it could not be determined whether "
                      f"the package asks for it: {engines.problema}.")
    elif engines.pedido:
        lineas.append(f"**NOT installed** although the package asks for it in "
                      f"`{engines.declarado_en}` — see the step log.")
    else:
        mirado = ", ".join(f"`{m}`" for m in engines.mirado_en) or "no dependency file"
        lineas.append(f"**Not installed** — not requested by the package (looked "
                      f"in: {mirado}). Installing it anyway would let a package "
                      f"that declares it does not need it verify all the same, "
                      f"hiding the very failure `verify` exists to find.")
    lineas += ["", _NOTA_106, ""]
    return "\n".join(lineas)


def anotaciones(veredicto: Veredicto) -> list[str]:
    """Las anotaciones `::error`/`::warning` de GitHub, una por alcance.

    Un alcance no realizado que NO estaba exigido deja el trabajo verde,
    y por eso necesita anotación: las anotaciones salen arriba del todo
    en la página del trabajo, verde o rojo. Sin esto, «verde» y «verde
    con dos evidencias sin medir» se verían igual.
    """
    salida: list[str] = []
    for a in veredicto.alcances:
        motivo = (a.motivo or "no reason given").replace("\n", " ")
        if a.estado == "FAIL":
            salida.append(f"::error title=matrixai verify: {a.nombre} FAILED::{motivo}")
        elif not a.realizado:
            nivel = "error" if a.exigido else "warning"
            cola = "" if a.exigido else " (not required by `require`)"
            salida.append(f"::{nivel} title=matrixai verify: {a.nombre} "
                          f"NOT PERFORMED ({a.estado}){cola}::{motivo}")
    return salida


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------

def _requeridos(texto: str) -> tuple[str, ...]:
    """Analiza `require`. Un nombre desconocido es un error, no un aviso.

    Fallar cerrado: `require: manifets` (con la errata) dejaría de exigir
    justo lo que se quería exigir, y el trabajo saldría verde por un dedo.
    """
    bruto = [t for t in re.split(r"[,\s]+", texto.strip()) if t]
    if not bruto:
        return REQUERIDOS_POR_DEFECTO
    if len(bruto) == 1 and bruto[0].lower() == "all":
        return ALCANCES
    if len(bruto) == 1 and bruto[0].lower() == "none":
        return ()
    por_nombre = {a.lower(): a for a in ALCANCES}
    salida: list[str] = []
    for t in bruto:
        if t.lower() in ("all", "none"):
            raise ValueError(f"'{t}' cannot be combined with scope names")
        if t.lower() not in por_nombre:
            raise ValueError(f"unknown scope {t!r}; known scopes are "
                             f"{', '.join(ALCANCES)} (or 'all' / 'none')")
        salida.append(por_nombre[t.lower()])
    return tuple(dict.fromkeys(salida))


def _instalar_engines(requisito: str) -> tuple[bool, str]:
    """Instala `matrixai-engines` con pip y dice qué pasó (no qué se pidió)."""
    orden = [sys.executable, "-m", "pip", "install", requisito]
    proceso = subprocess.run(orden, capture_output=True, text=True)
    salida = (proceso.stdout or "") + (proceso.stderr or "")
    return proceso.returncode == 0, salida.strip()


#: Lo que `--retrain` entiende, y NADA MÁS. Cerrada a propósito y con las
#: dos listas enteras en el mensaje de error.
#:
#: Esto era `in ("true","1","yes")`, de forma que cualquier otro valor
#: significaba «no reentrenar», EN SILENCIO. MEDIDO el 2026-09-15: con
#: `--retrain si` el informe salía diciendo «retraining was not requested
#: (use --retrain)» —le decía a quien verifica que no lo había pedido y le
#: sugería la bandera que acababa de poner—, y `training` y `R3` quedaban
#: sin medir bajo un trabajo que podía acabar verde. Se falla CERRADO, como
#: `--require`: un dedo no puede apagar dos evidencias sin que se vea.
VALORES_VERDADEROS = ("true", "1", "yes", "on")
VALORES_FALSOS = ("false", "0", "no", "off")

#: Cuánto `stderr` de `verify` se guarda cuando no hubo informe que subir.
#: Suficiente para ver la traza y no tanto como para que el artefacto sea
#: el registro entero.
_TOPE_STDERR = 4000


def _booleano(texto: str) -> bool:
    """Un sí/no de una entrada de flujo, FALLANDO CERRADO."""
    t = str(texto).strip().lower()
    if t in VALORES_VERDADEROS:
        return True
    if t in VALORES_FALSOS:
        return False
    raise ValueError(
        f"{texto!r} is not a yes/no value; use one of "
        f"{', '.join(VALORES_VERDADEROS)} (yes) or "
        f"{', '.join(VALORES_FALSOS)} (no). It was NOT read as 'no': an input "
        f"this action does not understand is rejected instead of guessed, "
        f"because guessing 'no' here silently skips two of the four scopes")


def _escribir(fichero_env: str, texto: str) -> None:
    destino = os.environ.get(fichero_env)
    if not destino:
        return
    with open(destino, "a", encoding="utf-8") as fh:
        fh.write(texto + "\n")


def _escribir_informe(ruta: str, texto: str) -> None:
    """Deja el informe en `ruta`. No poder escribirlo NO cambia el veredicto.

    Y se DICE con un `::warning` en vez de callarlo: el trabajo puede ser
    verde de verdad y el artefacto haber quedado vacío, y quien se lo
    descargue tiene que poder saber por qué sin repetir la verificación.
    """
    if not ruta:
        return
    try:
        Path(ruta).write_text(texto, encoding="utf-8")
    except OSError as exc:
        print(f"::warning title=matrixai verify: report not written::{ruta}: {exc}")


def _registro_de_la_accion(motivo: str, detalle: str, codigo: int,
                           **extra: Any) -> str:
    """El informe que se sube cuando `verify` NO llegó a hablar.

    NO se le da forma de informe de `verify` —no trae `stages`— a
    propósito: un fichero que se parece a un informe y no lo es se lee
    como un informe. Dice de quién es, qué pasó y con qué código se salió.
    """
    return json.dumps({"matrixai_verify_action": {
        "verify_ran": False, "outcome": motivo, "detail": detalle,
        "exit_code": codigo, **extra}}, indent=2, ensure_ascii=False) + "\n"


def _resumen_de_error(titular: str, detalle: str, codigo: int) -> str:
    """El resumen de un retorno TEMPRANO: los cuatro alcances sin realizar.

    Los cuatro retornos tempranos de este módulo no escribían NI informe NI
    resumen NI salidas. Quien se encontraba el trabajo rojo por una entrada
    mala se bajaba un artefacto que no existía y una página de trabajo sin
    una palabra de por qué — y el `action.yml` promete el informe SIEMPRE.
    """
    return "\n".join([
        "## matrixai verify — NOT VERIFIED", "",
        f"**0 of {len(ALCANCES)} scopes were actually checked.** "
        f"`matrixai verify` did not produce a report.", "",
        f"**{titular}:** {detalle}", "",
        "**NOT PERFORMED:** " + ", ".join(f"`{a}`" for a in ALCANCES)
        + ". A scope that was not performed is not a scope that passed.", "",
        f"* action exit code: `{codigo}`", "",
        _LEYENDA, "", _NOTA_106, ""])


def _terminar(codigo: int, *, report: str, informe: str, resumen: str,
              titulo: str, sin_realizar: Sequence[str] = ()) -> int:
    """EL ÚNICO sitio por el que se sale, para que no haya salidas mudas.

    Informe, resumen y las tres salidas se escriben aquí y en ningún otro
    lado: mientras cada retorno escribía lo suyo, cuatro de ellos no
    escribían nada y nadie lo notaba porque el trabajo ya salía rojo.
    """
    print(resumen)
    _escribir_informe(report, informe)
    _escribir("GITHUB_STEP_SUMMARY", resumen)
    _escribir("GITHUB_OUTPUT", f"verdict={titulo}")
    _escribir("GITHUB_OUTPUT", f"exit-code={codigo}")
    _escribir("GITHUB_OUTPUT", f"unperformed-scopes={','.join(sin_realizar)}")
    return codigo


def parser() -> argparse.ArgumentParser:
    """La línea de órdenes, aparte de `main` Y PÚBLICA a propósito.

    `action.yml` declara un defecto por entrada y este parser declara otro
    para la misma bandera. *Dos sitios declarando lo mismo acaban
    divergiendo*, y aquí divergir no se nota: el que manda es el del YAML
    —que es el que nadie puede ejecutar fuera de GitHub— y quien lea el
    módulo creerá otra cosa. Sacándolo de dentro de `main`, la suite puede
    comparar los cuatro defectos uno a uno en vez de fiarse del que se
    acordó de mirar.
    """
    p = argparse.ArgumentParser(prog="matrixai-verify-action", add_help=True)
    p.add_argument("--package", required=True)
    p.add_argument("--require", default=",".join(REQUERIDOS_POR_DEFECTO))
    p.add_argument("--retrain", default="false")
    p.add_argument("--locale", default="en")
    p.add_argument("--engines", default="auto",
                   help="'auto' (install matrixai-engines only if the package "
                        "asks for it), 'skip', or a pip requirement to install")
    p.add_argument("--report", default="", help="Where to write verify's JSON report")
    return p


def main(argv: Sequence[str] | None = None, *,
         ejecutar: Callable[[list[str]], tuple[int, str, str]] | None = None,
         instalar: Callable[[str], tuple[bool, str]] | None = None) -> int:
    """Lo que la acción ejecuta. Devuelve el código de salida del trabajo."""
    args = parser().parse_args(list(argv) if argv is not None else None)
    report = args.report

    def _no_pudo_empezar(titular: str, motivo: str, detalle: str) -> int:
        """Rojo, con el motivo arriba del todo Y en el informe que se sube."""
        codigo = SALIDAS["error_de_uso"]
        print(f"::error title=matrixai verify: {titular}::{detalle}")
        return _terminar(codigo, report=report,
                         informe=_registro_de_la_accion(motivo, detalle, codigo),
                         resumen=_resumen_de_error(titular, detalle, codigo),
                         titulo="NOT VERIFIED", sin_realizar=ALCANCES)

    try:
        requeridos = _requeridos(args.require)
    except ValueError as exc:
        return _no_pudo_empezar("bad input", "bad-require", f"require: {exc}")

    try:
        reentrenar = _booleano(args.retrain)
    except ValueError as exc:
        return _no_pudo_empezar("bad input", "bad-retrain", f"retrain: {exc}")

    ruta = Path(args.package)
    if not ruta.exists():
        # `verify` sobre una ruta inexistente contesta «the package carries no
        # reproduce.json», que es verdad y ENGAÑA: suena a paquete incompleto
        # cuando lo que pasa es que ahí no hay nada. MEDIDO el 2026-09-15.
        return _no_pudo_empezar(
            "package not found", "package-not-found",
            f"{args.package} does not exist on the runner")

    modo = args.engines.strip()
    if not modo:
        return _no_pudo_empezar(
            "bad input", "bad-engines",
            "engines: an empty value is not understood; use 'auto', 'skip', or "
            "a pip requirement string")

    engines = peticion_de_engines_de_la_ruta(ruta)
    engines_instalado: str | None = None
    requisito: str | None = None
    if modo.lower() == "skip":
        requisito = None
    elif modo.lower() == "auto":
        # `matrixai-engines` SOLO si el paquete lo pide: instalarlo siempre
        # haría que un paquete que declara no necesitarlo se verificara
        # igual, tapando justo el fallo que `verify` existe para encontrar.
        if engines.pedido:
            requisito = _NOMBRE_ENGINES
    else:
        # Una cadena explícita ES la petición, y viene de quien escribe el
        # flujo. Esto estaba DOCUMENTADO en `action.yml` y no hacía nada:
        # solo se miraba dentro del `elif engines.pedido`, así que
        # `engines: matrixai-engines==0.3` sobre un paquete que no lo
        # declara no instalaba nada y nadie lo decía.
        requisito = modo

    if requisito is not None:
        ok, log = (instalar or _instalar_engines)(requisito)
        print(log)
        if not ok:
            # NO se sigue como si nada: alguien dijo que hacía falta.
            quien = (f"the package declares matrixai-engines in "
                     f"{engines.declarado_en}" if engines.pedido
                     else f"the workflow asked for `engines: {modo}`")
            return _no_pudo_empezar(
                "matrixai-engines", "engines-install-failed",
                f"{quien} and it could not be installed from '{requisito}'")
        engines_instalado = requisito

    orden = [sys.executable, "-m", "matrixai", "verify", str(ruta), "--json",
             "--locale", args.locale]
    if reentrenar:
        orden.append("--retrain")

    if ejecutar is None:
        proceso = subprocess.run(orden, capture_output=True, text=True)
        codigo, salida, error = proceso.returncode, proceso.stdout, proceso.stderr
    else:
        codigo, salida, error = ejecutar(orden)

    # Lo que se sube es la salida de `verify` TAL CUAL. Era un
    # `json.dumps(indent=2)` —reordenado y reindentado— mientras el YAML lo
    # anunciaba como «el informe crudo»: media verdad tranquilizadora, y
    # además borraba justo lo que hay que mirar cuando el informe no se
    # puede leer.
    crudo = salida if salida.strip() else _registro_de_la_accion(
        "verify-wrote-nothing",
        "`matrixai verify` exited without writing anything to stdout",
        SALIDAS["error_de_uso"], verify_exit_code=codigo,
        verify_stderr=(error or "")[:_TOPE_STDERR])

    try:
        informe = json.loads(salida)
        if not isinstance(informe, dict):
            raise ValueError("the report is not a JSON object")
    except ValueError as exc:
        if error:
            print(error)
        codigo_de_uso = SALIDAS["error_de_uso"]
        detalle = (f"`matrixai verify` exited {codigo} and its output could not "
                   f"be read as a JSON report: {exc}")
        print(f"::error title=matrixai verify: unreadable report::{detalle}")
        return _terminar(codigo_de_uso, report=report, informe=crudo,
                         resumen=_resumen_de_error("unreadable report", detalle,
                                                   codigo_de_uso),
                         titulo="NOT VERIFIED", sin_realizar=ALCANCES)

    veredicto = decidir(informe, codigo, requeridos)
    for anotacion in anotaciones(veredicto):
        print(anotacion)
    resumen = resumen_markdown(veredicto, paquete=args.package, orden=orden,
                               engines=engines, engines_instalado=engines_instalado,
                               modo_engines=modo)
    return _terminar(veredicto.codigo, report=report, informe=crudo,
                     resumen=resumen, titulo=veredicto.titulo,
                     sin_realizar=veredicto.sin_realizar)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
