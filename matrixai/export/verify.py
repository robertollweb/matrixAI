"""CONTRATO 82-C2 — verificar un paquete reproducible, POR ETAPAS.

Un veredicto único («falló») obliga a adivinar dónde. Las cuatro etapas
fallan por motivos distintos y se informan por separado:

| Etapa | Qué comprueba |
|---|---|
| `manifest` | integridad del manifiesto y de cada artefacto por su `sha256` |
| `R1` | el dataset regenerado tiene el sha256 COMPLETO esperado |
| `training` | el entrenamiento llegó a término |
| `R3` | las métricas caen dentro de su tolerancia |

**`INCOMPARABLE` no es un fallo.** Es «no se puede comparar»: falta la
receta, falta la semilla, el `.mxtrain` no viaja. *Un fallo por falta de
acceso no es una manipulación* — regla heredada del 81, y confundirlas
acusaría a paquetes honestos de estar manipulados.

**`NOT_RUN` tampoco.** Es «no se ha intentado», y se dice en vez de dar
un verde vacío: reentrenar cuesta minutos u horas y no se impone.

**Este comando procesa paquetes NO CONFIABLES.** Quien lo ejecuta puede
haber descargado el ZIP de cualquier sitio, así que **nada que venga
dentro del paquete se ejecuta** para decidir si el paquete es bueno: se
leen ficheros y se comparan digests. El `.mxtrain` y la receta son datos
aquí, no programas.

**85-C2b — los motivos van en el idioma que se pida.** Este informe ya se
pinta en la interfaz por fases, y *lo que redacta el core se traduce en el
core, no al pintarlo*. Las frases viven en `verify_textos.py`, en los dos
idiomas; aquí solo se eligen. Lo que NO cambia de idioma: los cuatro
`status`, los nombres de etapa y lo que un motivo interpola —una ruta, una
huella, un número—, porque son valores y hay quien los lee por programa.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from matrixai.export.reproduce import (
    REPRODUCE_MANIFEST_FILENAME as REPRODUCE_FILENAME,
    manifest_digest,
    sha256_file,
)
from matrixai.export.verify_textos import IDIOMA_POR_DEFECTO, motivo

__all__ = ["ESTADOS", "SALIDAS", "verify_package"]

#: Los cuatro estados de una etapa. `PASS`/`FAIL` son un juicio;
#: `NOT_RUN`/`INCOMPARABLE` dicen que NO lo hay, por motivos distintos.
ESTADOS = ("PASS", "FAIL", "NOT_RUN", "INCOMPARABLE")

#: Códigos de salida, para que un guion pueda distinguir los casos sin
#: leer el JSON. «No se pudo comprobar» y «alguien lo tocó» no son lo
#: mismo y no comparten código.
SALIDAS = {
    "ok": 0,            # nada falló
    "fail": 2,          # al menos una etapa FAIL: discrepancia o manipulación
    "incomparable": 3,  # no se pudo comprobar (falta el manifiesto)
}

_ORDEN = ("manifest", "R1", "training", "R3")

#: Las versiones de manifiesto que ESTE verificador sabe leer. Cerrada, y
#: por el mismo motivo que en el recibo: lo que no se conoce no se
#: interpreta a medias.
_ESQUEMAS_CONOCIDOS = ("1.0", "1.1", "1.2")


def _etapa(status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    assert status in ESTADOS, status
    out: dict[str, Any] = {"status": status}
    if reason:
        out["reason"] = reason
    out.update(extra)
    return out


def _leer_manifiesto(
    bundle: Path, *, locale: str = IDIOMA_POR_DEFECTO,
) -> tuple[dict[str, Any] | None, str | None]:
    ruta = bundle / REPRODUCE_FILENAME
    if not ruta.is_file():
        return None, motivo("sin_reproduce", locale, fichero=REPRODUCE_FILENAME)
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # El mensaje de la excepción va CITADO: lo escribe la biblioteca
        # estándar, en inglés, y no es prosa de este verificador.
        return None, motivo("reproduce_ilegible", locale,
                            fichero=REPRODUCE_FILENAME, error=exc)
    if not isinstance(datos, dict):
        return None, motivo("reproduce_no_es_objeto", locale,
                            fichero=REPRODUCE_FILENAME)
    return datos, None


def _comprobar_inventario(
    bundle: Path, manifiesto: dict[str, Any], *, locale: str = IDIOMA_POR_DEFECTO,
) -> tuple[list[dict[str, str]], list[str]]:
    """Comprueba TODO lo que viaja: los digests y **lo que sobra**.

    Devuelve `(rotos, sin_cubrir)`. Lo segundo importa tanto como lo
    primero: un fichero que está en el paquete y no en el manifiesto es
    un fichero que nadie ha mirado, y `PASS` sobre eso es afirmar por
    omisión — justo lo que dejó pasar un `predict.py` sustituido.
    """
    declarados = manifiesto.get("files")
    presentes: dict[str, Path] = {}
    for fichero in sorted(bundle.rglob("*")):
        if not fichero.is_file() or fichero.is_symlink():
            continue
        relativa = fichero.relative_to(bundle).as_posix()
        if relativa == "reproduce.json":
            # Su integridad la cubre `manifest_sha256`, y no puede
            # llevar su propio digest dentro.
            continue
        presentes[relativa] = fichero

    if not isinstance(declarados, dict):
        # Un paquete anterior al inventario. NO se da por bueno: se dice
        # qué no se ha podido comprobar, que son todos.
        return [], sorted(presentes)

    rotos: list[dict[str, str]] = []
    for relativa, esperado in sorted(declarados.items()):
        fichero = bundle / relativa
        problema = _ruta_fuera_del_paquete(bundle, relativa, locale=locale)
        if problema is not None:
            rotos.append({"artifact": relativa, "path": relativa, "problem": problema})
            continue
        if not fichero.is_file() or fichero.is_symlink():
            rotos.append({"artifact": relativa, "path": relativa,
                          "problem": motivo("p_declarado_pero_ausente", locale)})
            continue
        real = sha256_file(fichero)
        if real != esperado:
            rotos.append({"artifact": relativa, "path": relativa,
                          "problem": motivo("p_sha256_no_cuadra", locale),
                          "expected": esperado, "found": real})
    # Lo que SOBRA: está en el paquete y el manifiesto no lo nombra.
    sin_cubrir = sorted(set(presentes) - set(declarados))
    return rotos, sin_cubrir


def _ruta_fuera_del_paquete(
    bundle: Path, ruta: str, *, locale: str = IDIOMA_POR_DEFECTO,
) -> str | None:
    """El motivo por el que una ruta declarada NO está dentro, o `None`.

    Un paquete describe **sus** artefactos. Una ruta absoluta o con `..`
    no describe nada suyo: elige un fichero de la máquina de quien
    verifica, y el informe acabaría llamando «modelo» a lo que hubiera
    ahí. Se comprueba **antes** de abrir el fichero — mirar primero y
    preguntar después ya sería haberlo leído.

    Los enlaces simbólicos también: un enlace dentro del paquete que
    apunta fuera es la misma fuga con otra ropa, y por eso se resuelve la
    ruta real antes de comparar.
    """
    if not ruta or ruta.strip() != ruta:
        return motivo("p_ruta_vacia", locale)
    candidata = Path(ruta)
    if candidata.is_absolute():
        return motivo("p_ruta_absoluta", locale)
    if ".." in candidata.parts:
        return motivo("p_ruta_escapa", locale)
    try:
        raiz = bundle.resolve(strict=False)
        destino = (bundle / candidata).resolve(strict=False)
    except OSError as exc:  # pragma: no cover — rutas imposibles del sistema
        return motivo("p_ruta_irresoluble", locale, error=exc)
    if raiz != destino and raiz not in destino.parents:
        # Cubre el enlace simbólico que apunta fuera: la ruta escrita
        # parece de dentro y el fichero real no lo es.
        return motivo("p_ruta_fuera", locale)
    return None


def _verificar_manifiesto(
    bundle: Path, manifiesto: dict[str, Any], *, locale: str = IDIOMA_POR_DEFECTO,
) -> dict[str, Any]:
    """Integridad del manifiesto Y de cada artefacto que declara.

    Las dos cosas, y no solo la primera: un `manifest_sha256` correcto
    sobre unos artefactos cambiados diría que todo está bien.
    """
    declarado = manifiesto.get("manifest_sha256")
    if not isinstance(declarado, str) or not declarado:
        return _etapa("FAIL", motivo("m_sin_digest", locale))
    if declarado != manifest_digest(manifiesto):
        return _etapa("FAIL", motivo("m_digest_no_cuadra", locale),
                      field="manifest_sha256")

    artefactos = manifiesto.get("artifacts")
    if not isinstance(artefactos, dict):
        return _etapa("FAIL", motivo("m_sin_artefactos", locale))

    # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: la versión de esquema
    # no se miraba. Medido: un manifiesto con `schema_version: "999.0"`
    # salía `manifest PASS`. Interpretar a medias un formato que no se
    # conoce es peor que no leerlo.
    version = manifiesto.get("schema_version")
    if version not in _ESQUEMAS_CONOCIDOS:
        return _etapa("INCOMPARABLE",
                      motivo("m_esquema_desconocido", locale, version=repr(version),
                             conocidos=list(_ESQUEMAS_CONOCIDOS)))

    rotos: list[dict[str, str]] = []
    comprobados: list[str] = []
    for nombre, art in artefactos.items():
        if art is None:
            # `None` es una AUSENCIA DECLARADA, no un artefacto roto: el
            # paquete dice «esto no viaja aquí» —el dataset, por ejemplo—
            # y eso es legítimo. Tratarlo como malformado, que fue el
            # primer intento de este arreglo, acusaba a todos los paquetes
            # honestos: arreglar un sesgo puede crear el contrario, y hay
            # que probar los dos lados.
            continue
        if not isinstance(art, dict):
            # AUDITORÍA EXTERNA [BLOQUEANTE]: esto era un `continue` mudo.
            # Un artefacto que el manifiesto declara y el verificador no
            # mira sale igual que uno comprobado, y el informe dice PASS.
            # Media limpieza es peor que ninguna.
            rotos.append({"artifact": nombre, "path": "?",
                          "problem": motivo("p_entrada_invalida", locale)})
            continue
        ruta, esperado = art.get("path"), art.get("sha256")
        # Un digest DECLARADO SIN fichero es legítimo: el paquete dice
        # «el dataset tenía este hash» sin llevarlo dentro, y R1 lo usa
        # para comparar lo que regenere. No hay nada que abrir, así que
        # no hay nada que comprobar aquí.
        #
        # (Segundo intento de este arreglo. El primero lo rechazaba y
        # acusaba a todo paquete honesto que no lleva el dataset. Probar
        # «los dos lados» con UN ejemplo honesto no basta: hay que probar
        # con el abanico, y el mío no tenía este caso.)
        if not isinstance(ruta, str) and isinstance(esperado, str):
            continue
        if not isinstance(ruta, str) or not isinstance(esperado, str):
            # Lo que sí se rechaza: un fichero que VIAJA en el paquete y
            # no trae con qué comprobarlo. Saltárselo en silencio lo
            # dejaba pasar con `PASS`, que es media limpieza.
            rotos.append({"artifact": nombre, "path": str(ruta),
                          "problem": motivo(
                              "p_sin_sha256" if isinstance(ruta, str)
                              else "p_sin_ruta_ni_sha256", locale)})
            continue

        # AUDITORÍA 1ª pasada (2026-08-20) [BLOQUEANTE]: un paquete podía
        # declarar `../..//etc/passwd` —o una ruta absoluta— y `verify`
        # leía el fichero de FUERA y lo daba por bueno con `PASS`. O sea:
        # un paquete ajeno elegía qué fichero de tu máquina se abría, y el
        # informe llamaba «modelo» a lo que hubiera dentro.
        #
        # No basta con que el manifiesto esté firmado consigo mismo: quien
        # fabrica el paquete calcula ese `manifest_sha256` sin esfuerzo. Lo
        # que hay que comprobar es que el artefacto **está dentro**.
        problema = _ruta_fuera_del_paquete(bundle, ruta, locale=locale)
        if problema is not None:
            rotos.append({"artifact": nombre, "path": ruta, "problem": problema})
            continue

        fichero = bundle / ruta
        if fichero.is_symlink():
            # AUDITORÍA EXTERNA: un enlace INTERNO también sobra. El
            # digest cubre los bytes del destino, no la indirección: el
            # mismo manifiesto puede describir dos ficheros distintos
            # según a dónde apunte el enlace mañana.
            rotos.append({"artifact": nombre, "path": ruta,
                          "problem": motivo("p_enlace_simbolico", locale)})
            continue
        if not fichero.is_file():
            rotos.append({"artifact": nombre, "path": ruta,
                          "problem": motivo("p_ausente", locale)})
            continue
        real = sha256_file(fichero)
        if real != esperado:
            rotos.append({"artifact": nombre, "path": ruta,
                          "problem": motivo("p_sha256_no_cuadra", locale),
                          "expected": esperado, "found": real})
        else:
            comprobados.append(nombre)

    if rotos:
        # Se nombra QUÉ artefacto: «algo no cuadra» obliga a abrir el ZIP
        # y comparar a mano los cinco ficheros.
        return _etapa("FAIL", motivo("m_artefacto_no_cuadra", locale),
                      artifacts=rotos)

    # REFUTACIÓN (2026-08-20) [BLOQUEANTE]: hasta aquí se comprobaba lo que
    # el manifiesto DECLARA —cuatro artefactos— y el paquete LLEVA catorce
    # ficheros. Sustituyendo `model.onnx`, `predict.py` y los pesos, esto
    # contestaba `PASS`. Y el Space ejecuta ese `predict.py` justo después.
    rotos_del_inventario, sin_cubrir = _comprobar_inventario(
        bundle, manifiesto, locale=locale)
    if rotos_del_inventario:
        return _etapa("FAIL", motivo("m_fichero_no_cuadra", locale),
                      artifacts=rotos_del_inventario)
    if sin_cubrir:
        # NO es `FAIL`: el paquete no miente, es que no cubre esos
        # ficheros — y decir `PASS` sería afirmar sobre lo que no se ha
        # mirado. Los paquetes anteriores a este inventario caen aquí.
        return _etapa("INCOMPARABLE", motivo("m_ficheros_sin_cubrir", locale),
                      uncovered_files=sin_cubrir, checked=comprobados)
    return _etapa("PASS", checked=comprobados,
                  files_checked=len(manifiesto.get("files") or {}))


def _verificar_r1(
    bundle: Path, manifiesto: dict[str, Any], *, locale: str = IDIOMA_POR_DEFECTO,
) -> dict[str, Any]:
    """Regenerar el dataset desde la receta y comparar su sha256 COMPLETO.

    Lo que impide compararlo NO es un fallo del paquete: sin receta, sin
    semilla o sin el digest esperado no hay nada contra lo que comparar, y
    decir `FAIL` acusaría a un modelo entrenado con datos propios —un caso
    honesto y frecuente— de estar manipulado.
    """
    verificable = manifiesto.get("verifiable")
    r1 = verificable.get("r1") if isinstance(verificable, dict) else None
    if isinstance(r1, dict) and r1.get("possible") is False:
        # El propio manifiesto ya dice que no se puede, y por qué. No se
        # repite el análisis: se cita, que para eso lo redactó quien
        # empaquetó.
        # Lo que redactó quien empaquetó se CITA entre comillas y no se
        # reescribe: es un dato del paquete, como una huella o una ruta, y
        # traducirlo aquí sería poner en su boca algo que no dijo.
        suyo = r1.get("reason")
        return _etapa("INCOMPARABLE",
                      motivo("r1_no_posible_segun_el_paquete", locale, cita=suyo)
                      if suyo else motivo("r1_no_posible", locale),
                      missing=r1.get("missing") or [])

    generacion = manifiesto.get("generation")
    semilla = None
    if isinstance(generacion, dict):
        semillas = generacion.get("seeds")
        if isinstance(semillas, dict):
            semilla = semillas.get("dataset")
    artefactos = manifiesto.get("artifacts")
    receta = artefactos.get("recipe") if isinstance(artefactos, dict) else None
    dataset = artefactos.get("dataset") if isinstance(artefactos, dict) else None
    esperado = dataset.get("sha256") if isinstance(dataset, dict) else None

    faltan = [n for n, v in (("recipe", receta), ("seed", semilla),
                             ("dataset_sha256", esperado)) if not v]
    if faltan:
        return _etapa("INCOMPARABLE",
                      motivo("r1_faltan_datos", locale, faltan=", ".join(faltan)),
                      missing=faltan)

    # TODO lo que decidió aquel CSV viaja en el manifiesto: `mode`,
    # `field_ranges`, `field_types`, `field_categories` y los
    # identificadores excluidos. (Mi primera versión decía que no y
    # devolvía NOT_RUN por eso: era falso, y lo era porque lo supuse en
    # vez de mirar el `generation` que produce el manifiesto.)
    modelo = artefactos.get("model") if isinstance(artefactos, dict) else None
    if not isinstance(modelo, dict) or not modelo.get("path"):
        return _etapa("INCOMPARABLE", motivo("r1_sin_modelo", locale),
                      missing=["model"])
    entrenamiento = artefactos.get("training") if isinstance(artefactos, dict) else None
    if not isinstance(entrenamiento, dict) or not entrenamiento.get("path"):
        # El generador exige el `.mxtrain`: sin él no hay R1, y eso es una
        # ausencia, no una manipulación.
        return _etapa("INCOMPARABLE", motivo("r1_sin_mxtrain", locale),
                      missing=["training"])

    filas = dataset.get("rows") if isinstance(dataset, dict) else None
    if not isinstance(filas, int) or filas <= 0:
        return _etapa("INCOMPARABLE", motivo("r1_sin_filas", locale),
                      missing=["dataset_rows"])

    try:
        mxai_text = (bundle / str(modelo["path"])).read_text(encoding="utf-8")
        training_text = (bundle / str(entrenamiento["path"])).read_text(encoding="utf-8")
        recipe_text = (bundle / str(receta["path"])).read_text(encoding="utf-8")
    except OSError as exc:
        return _etapa("INCOMPARABLE",
                      motivo("r1_artefactos_ilegibles", locale, error=exc))

    # EL MODO NO SE INVENTA. Mi primera versión ponía `"deterministic"`
    # de respaldo —que además ni siquiera es un modo válido: son `random`
    # y `coherent`—, y regenerar con un modo distinto del que se usó da
    # otro dataset y por tanto un FAIL falso, acusando a un paquete
    # honesto. Sin modo declarado, no hay con qué comparar.
    modo = generacion.get("mode") if isinstance(generacion, dict) else None
    if not modo:
        return _etapa("INCOMPARABLE", motivo("r1_sin_modo", locale),
                      missing=["mode"])
    rangos = generacion.get("field_ranges") if isinstance(generacion, dict) else None
    tipos = generacion.get("field_types") if isinstance(generacion, dict) else None
    categorias = generacion.get("field_categories") if isinstance(generacion, dict) else None
    excluidos = generacion.get("excluded_identifiers") if isinstance(generacion, dict) else None

    try:
        # Esto NO ejecuta nada que venga en el paquete: corre el generador
        # DEL CORE tomando la receta como DATO. La receta es un lenguaje
        # acotado que el core interpreta, no un programa que se lanza.
        from matrixai.playground import _generate_synthetic_dataset
        generado = _generate_synthetic_dataset(
            mxai_text, training_text, filas, int(semilla), str(modo),
            recipe_text=recipe_text,
            field_ranges_override={k: (float(v[0]), float(v[1]))
                                   for k, v in (rangos or {}).items()
                                   if isinstance(v, (list, tuple)) and len(v) == 2} or None,
            field_types=tipos or None,
            field_categories=categorias or None,
            field_identifiers=list(excluidos) if excluidos else None,
        )
    except Exception as exc:  # noqa: BLE001 — el generador puede negarse por mil motivos
        # No poder regenerar NO es una manipulación del paquete: se dice
        # qué pasó y se deja en INCOMPARABLE.
        return _etapa("INCOMPARABLE",
                      motivo("r1_no_regenerable", locale, error=exc))

    # `csv_text`, no `csv`: es la clave que el generador devuelve de
    # verdad (medida, después de suponer la otra y comerme un KeyError).
    csv = generado.get("csv_text") if isinstance(generado, dict) else None
    if not isinstance(csv, str) or csv == "":
        return _etapa("INCOMPARABLE", motivo("r1_sin_csv", locale))

    import hashlib
    obtenido = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    if obtenido == esperado:
        return _etapa("PASS", rows=filas, sha256=obtenido, csv_text=csv)
    # AQUÍ sí es FAIL: el paquete dijo qué debía salir y ha salido otra
    # cosa. Se enseñan los DOS digests: «no coincide» a secas no deja
    # comprobar nada a quien lo lea.
    return _etapa("FAIL", motivo("r1_digest_distinto", locale),
                  expected=esperado, found=obtenido)


def _ruta_del_dataset(fuente: Any) -> Path:
    """Dónde escribir el dataset regenerado para que el contrato lo encuentre.

    La ruta la escribe el `.mxtrain` **del paquete**, así que no se obedece a
    ciegas: una absoluta o con `..` elegiría un fichero de la máquina de quien
    verifica, y verificar no escribe fuera de su taller. En ese caso se cae al
    nombre a secas y el entrenamiento dirá lo que tenga que decir; lo que no se
    hace es escribir donde el paquete mande.
    """
    if not fuente:
        return Path("dataset.csv")
    candidata = Path(str(fuente))
    if candidata.is_absolute() or ".." in candidata.parts:
        return Path(candidata.name)
    return candidata


def _reparto_del_dataset(fuente: str, csv: str) -> dict[str, str]:
    """Qué fichero —o ficheros— hay que escribir para reentrenar el mismo run.

    EL CSV ENTERO NO ES EL DATASET DE ENTRENAMIENTO cuando el paquete se hizo
    con `generate-dataset`: ese comando parte las filas en train (80 %) y eval
    (20 %) y el `.mxtrain` cita **el de train**. Esto escribía el csv completo
    ahí, así que reentrenaba con 300 filas un run hecho con 240 — y R3 daba
    `FAIL` por una diferencia que había puesto el verificador.

    Medido el 2026-08-26 con el paquete kelvin de la galería: publicado
    `mae 6.505213034913027e-17`, `verify --retrain` daba `6.499430623326438e-17`,
    y rehacerlo a mano con `generate-dataset` + `train` devolvía el publicado
    hasta el último dígito. La tolerancia de esa métrica está medida en 0,0: la
    diferencia no era ruido.

    Con un CSV que **no** sigue esa convención no hay nada que suponer y se
    escribe entero, como antes.
    """
    from matrixai.training.particion_sintetica import (  # noqa: PLC0415
        corte_train_eval, nombres_del_dataset)

    nombres = nombres_del_dataset(fuente)
    if nombres is None:
        return {fuente: csv}
    lineas = csv.splitlines(keepends=True)
    if len(lineas) < 3:
        # Cabecera y una fila: no hay dos partes que hacer, y partirlo dejaría
        # un fichero vacío que el entrenador leería como un dataset sin filas.
        return {fuente: csv}
    cabecera, filas = lineas[0], lineas[1:]
    corte = corte_train_eval(len(filas))
    ruta_train, ruta_eval = nombres
    return {ruta_train: cabecera + "".join(filas[:corte]),
            ruta_eval: cabecera + "".join(filas[corte:])}


def _verificar_training(
    bundle: Path, manifiesto: dict[str, Any], r1: dict[str, Any],
    *, locale: str = IDIOMA_POR_DEFECTO,
) -> dict[str, Any]:
    """¿El entrenamiento llega a término con lo que el paquete lleva?

    Se entrena con el dataset REGENERADO, no con uno traído de fuera: si
    R1 no pudo rehacerlo, no hay con qué entrenar y esto es
    `INCOMPARABLE`, no un fallo del paquete.

    Lo que se comprueba es que **termina**, no que dé las mismas métricas
    —eso es R3 y necesita datos que el paquete todavía no lleva—.
    Confundirlos daría un `PASS` que promete más de lo que ha mirado.
    """
    if r1.get("status") != "PASS":
        return _etapa("INCOMPARABLE", motivo("t_sin_dataset", locale))

    csv = r1.get("csv_text")
    if not isinstance(csv, str) or not csv:
        return _etapa("INCOMPARABLE", motivo("t_dataset_no_guardado", locale))

    artefactos = manifiesto.get("artifacts") or {}
    entrenamiento = artefactos.get("training") if isinstance(artefactos, dict) else None
    modelo = artefactos.get("model") if isinstance(artefactos, dict) else None
    if not isinstance(entrenamiento, dict) or not isinstance(modelo, dict):
        return _etapa("INCOMPARABLE", motivo("t_sin_mxtrain", locale))

    import shutil
    import tempfile

    taller = Path(tempfile.mkdtemp(prefix="matrixai-verify-"))
    try:
        # Se trabaja sobre una COPIA, en un directorio aparte: verificar no
        # escribe en el paquete de nadie.
        origen_train = bundle / str(entrenamiento["path"])
        ruta_train = taller / origen_train.name
        shutil.copy2(origen_train, ruta_train)

        from matrixai.training.parser import parse_training_file
        spec = parse_training_file(ruta_train)

        # El `.mxtrain` cita su MODEL por ruta, igual que su CSV, y esa
        # ruta es la de la máquina donde se entrenó. El modelo del paquete
        # se copia con el nombre que el contrato busca — si no, el
        # entrenador responde «MODEL not found» y esto saldría
        # INCOMPARABLE por un motivo que no es del paquete (medido).
        destino_modelo = taller / Path(str(getattr(spec, "model", "") or "model.mxai")).name
        destino_modelo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / str(modelo["path"]), destino_modelo)

        # El `.mxtrain` cita su CSV por RUTA, y esa ruta no existe fuera de
        # la máquina donde se entrenó. Se escribe el dataset regenerado
        # justo donde el contrato lo busca — **con su carpeta**.
        #
        # AQUÍ ESTABA EL DEFECTO (2ª auditoría externa del 2026-08-25,
        # hallazgo 2 residual). Esto escribía `taller/<nombre>` con
        # `Path(fuente).name`, o sea TIRANDO la carpeta, mientras el contrato
        # seguía diciendo `datos/…csv`. Con un contrato que cita el CSV dentro
        # de una carpeta —los que escribe `generate-dataset`, que es la mayoría
        # de los paquetes reales— el entrenador respondía «DATASET source not
        # found» y la etapa salía INCOMPARABLE **por culpa del verificador**.
        #
        # No se notó porque se probó desde el directorio donde se había
        # construido el paquete: allí `datos/` existía en el disco y el
        # entrenamiento tiraba de ÉL. Medido el 2026-08-26 desde un ZIP recién
        # descomprimido —que es lo que hace quien se lo descarga—: el paquete
        # kelvin de la galería daba `training INCOMPARABLE` y la página
        # anunciaba `training PASS`.
        fuente = getattr(getattr(spec, "dataset", None), "source", None)
        destino = taller / _ruta_del_dataset(fuente)
        destino.parent.mkdir(parents=True, exist_ok=True)
        for relativa, texto in _reparto_del_dataset(str(fuente or ""), csv).items():
            escrito = taller / _ruta_del_dataset(relativa)
            escrito.parent.mkdir(parents=True, exist_ok=True)
            escrito.write_text(texto, encoding="utf-8")

        # EL ENTRENADOR SE ELIGE COMO LO ELIGE EL CLI, no a dedo: una red
        # densa con `SupervisedTrainer` responde «P4 supervised trainer
        # requires labels» y esto saldría INCOMPARABLE por culpa del
        # verificador, no del paquete (medido 2026-08-20). Mismo criterio
        # que `_cmd_train`, y por eso mismo: dos sitios eligiendo distinto
        # acabarían discrepando.
        from matrixai.parser.parser import parse_file as _parse_mxai
        redes = getattr(_parse_mxai(str(destino_modelo)), "networks", []) or []
        if redes and getattr(redes[0], "transformer_blocks", []):
            from matrixai.training.transformer_trainer import TransformerSupervisedTrainer
            entrenador: Any = TransformerSupervisedTrainer()
        elif redes:
            from matrixai.training.dense_trainer import DenseSupervisedTrainer
            entrenador = DenseSupervisedTrainer()
        else:
            from matrixai.training.trainer import SupervisedTrainer
            entrenador = SupervisedTrainer()
        # LA CONFIGURACIÓN DEL RUN SE APLICA **ANTES** DE ENTRENAR, y ése era
        # el segundo defecto: esto se calculaba DESPUÉS de `train(...)`, así
        # que mutar `spec.epochs` no podía cambiar un entrenamiento que ya
        # había corrido, por mucho que el comentario dijera «se aplica de
        # verdad».
        #
        # Si el entrenador admite semilla se le pasa por su nombre —se mira su
        # firma en vez de suponerlo: el de los modelos FUNCTION no la tiene, y
        # llamarlo con `seed=` reventaría—.
        import inspect  # noqa: PLC0415
        admite_semilla = "seed" in inspect.signature(entrenador.train).parameters
        # EN QUÉ MÁQUINA CORRE ESTE REENTRENAMIENTO, cuando se puede saber.
        # Los dos entrenadores de la biblioteca estándar son CPU por
        # construcción —no hay torch por medio—, así que ahí es un HECHO. Con
        # torch depende del entorno y no se afirma: `None` significa «no lo sé»
        # y deja `device` sin aplicar, como estaba.
        _maquina = "cpu" if entrenador.__class__.__name__ in (
            "SupervisedTrainer", "DenseSupervisedTrainer") else None
        # ¿NO ADMITE SEMILLA ES UNA LIMITACIÓN, O ES QUE NO APLICA? Lo declara
        # el entrenador: los FUNCTION arrancan siempre de los mismos valores
        # (medido: `W1 = [0.05]`, `b1 = 0.0`, dos veces seguidas). Suponerlo
        # aquí sería adivinar por él.
        _determinista = bool(getattr(entrenador, "inicializacion_determinista", False))
        aplicadas, sin_aplicar, del_run, spec = _configuracion_del_run(
            manifiesto, spec, admite_semilla=admite_semilla, maquina=_maquina,
            inicializacion_determinista=_determinista)
        resultado = entrenador.train(
            spec, output_dir=str(taller / "out"), base_path=taller,
            training_path=ruta_train, **del_run)
        # LAS MÉTRICAS QUE EL ENTRENADOR DEJA EN DISCO, leídas ANTES de borrar
        # el taller. El camino FUNCTION no las devuelve en su `TrainingRunResult`
        # —ahí van `accuracy` y las pérdidas— pero escribe un `metrics.json` con
        # `r2` y `mae`, que son justo las que ese tipo de paquete publica. Sin
        # esto, R3 decía «ninguna de las métricas comparables la reportó el
        # reentrenamiento» sobre un fichero que las tenía delante.
        _del_disco = _metricas_del_taller(taller / "out")
    except Exception as exc:  # noqa: BLE001 — el entrenador se niega por mil motivos
        # Que no se pueda entrenar AQUÍ no prueba que el paquete mienta:
        # otro entorno, otro backend, una dependencia que falta.
        return _etapa("INCOMPARABLE", motivo("t_no_arranca", locale, error=exc))
    finally:
        shutil.rmtree(taller, ignore_errors=True)

    # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: el reentrenamiento NO
    # aplicaba la configuración que la captura declara —semilla efectiva,
    # épocas, motor, dispositivo, opciones deterministas—. Entrenar con
    # otra configuración y luego comparar métricas contra las publicadas
    # produce un `FAIL` **falso**, acusando a un paquete honesto: es
    # exactamente el mismo error que ya costó el «modo NO se inventa» de
    # R1.
    #
    # Lo que se puede aplicar, se aplica; lo que no, **se declara**, y R3
    # lo mira antes de comparar. Callarlo dejaría a R3 comparando peras
    # con manzanas y llamándolo veredicto.
    epocas = getattr(resultado, "best_epoch", None)
    # LOS NOMBRES SON LOS DEL RESULTADO, medidos: `TrainingRunResult` no
    # tiene un `metrics` —lo supuse y salía siempre vacío—, tiene
    # `validation_metrics` (las que declara el `.mxtrain`, por su nombre),
    # `accuracy` y las pérdidas.
    medidas: dict[str, Any] = dict(_del_disco)
    validacion = getattr(resultado, "validation_metrics", None)
    if isinstance(validacion, dict):
        medidas.update({str(k): v for k, v in validacion.items()})
    for campo in ("accuracy", "best_validation_loss",
                  "final_train_loss", "final_validation_loss"):
        valor = getattr(resultado, campo, None)
        if isinstance(valor, (int, float)):
            medidas.setdefault(campo, valor)
    return _etapa("PASS", best_epoch=epocas, metrics=medidas,
                  note=motivo("t_nota", locale),
                  applied_from_capture=aplicadas,
                  not_applied_from_capture=sin_aplicar)


def _metricas_del_taller(salida: Path) -> dict[str, Any]:
    """Las métricas que el entrenador dejó EN DISCO al reentrenar.

    El camino FUNCTION no las devuelve en su `TrainingRunResult` —ahí van
    `accuracy` y las pérdidas— pero escribe un `metrics.json` con `r2` y `mae`,
    que son justo las que publica un paquete de regresión. Sin leerlo, R3 decía
    «ninguna de las métricas comparables la reportó el reentrenamiento» con el
    fichero delante.

    Solo números: lo que no lo sea no se compara, y meterlo aquí obligaría a
    quien compara a distinguirlo después.
    """
    fichero = Path(salida) / "metrics.json"
    if not fichero.is_file():
        return {}
    try:
        crudo = json.loads(fichero.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(crudo, dict):
        return {}
    return {str(k): v for k, v in crudo.items() if isinstance(v, (int, float))}


def _verificar_r3(
    manifiesto: dict[str, Any], training: dict[str, Any],
    *, locale: str = IDIOMA_POR_DEFECTO,
) -> dict[str, Any]:
    """¿Las métricas del reentrenamiento caen dentro de su tolerancia?

    Tres estados, y son tres cosas distintas:

    * **`NOT_RUN`** si no se ha reentrenado. Sin un valor nuevo no hay
      nada que comparar con el publicado, y decir otra cosa sería
      inventarse una comparación.
    * **`INCOMPARABLE`** si el paquete no publica ninguna métrica que se
      pueda contrastar. El manifiesto ya dice por métrica qué le falta
      (`comparable`, `incomplete`), así que **se cita en vez de repetir el
      análisis**. Que al producto todavía le falte capturar el evaluador o
      la tolerancia es una carencia del PRODUCTO, no una mentira del
      paquete: `INCOMPARABLE`, no `FAIL`.
    * **`PASS`/`FAIL`** solo cuando de verdad hay con qué comparar.
    """
    if training.get("status") != "PASS":
        # EL MOTIVO DE VERDAD, no siempre el mismo.
        #
        # Esto decía «use --retrain» pasara lo que pasara, así que con
        # `--retrain` YA PUESTO seguía mandando a ponerlo: le decía a
        # quien verifica que hiciera lo que acababa de hacer. Medido
        # conduciendo el CLI (sonda del 82-C2, 2026-08-20), tres pasadas
        # con `--retrain` y las tres con el mismo mensaje.
        #
        # Cuando el reentrenamiento SÍ se pidió y no llegó a PASS, lo que
        # falta no es el flag: es lo que paró a `training`, y ése es el
        # motivo que hay que enseñar. *Arreglar el mensaje no es arreglar
        # el fallo* — salvo cuando el mensaje ES el fallo, como aquí.
        if training.get("status") == "NOT_RUN":
            return _etapa("NOT_RUN", motivo("r3_sin_reentrenamiento", locale))
        # El motivo de `training` YA viene redactado en este idioma: se
        # arrastra tal cual, no se vuelve a traducir.
        porque = training.get("reason") or motivo("r3_training_no_paso", locale)
        return _etapa("NOT_RUN",
                      motivo("r3_reentrenamiento_incompleto", locale, porque=porque))

    metricas = manifiesto.get("metrics")
    if not isinstance(metricas, list) or not metricas:
        return _etapa("INCOMPARABLE", motivo("r3_sin_metricas", locale))

    comparables = [m for m in metricas if isinstance(m, dict) and m.get("comparable")]
    if not comparables:
        # Lo que falta lo dice el manifiesto, métrica a métrica. Se
        # reproduce esa lista para que quien lo lea sepa QUÉ pedirle al
        # productor del paquete, en vez de «no se puede».
        falta: list[str] = []
        for m in metricas:
            if isinstance(m, dict):
                falta.extend(str(x) for x in (m.get("incomplete") or []))
        return _etapa("INCOMPARABLE", motivo("r3_metricas_incompletas", locale),
                      missing=sorted(set(falta)))

    obtenidas = training.get("metrics")
    if not isinstance(obtenidas, dict) or not obtenidas:
        return _etapa("INCOMPARABLE",
                      motivo("r3_reentrenamiento_sin_metricas", locale))

    # LA TOLERANCIA SOLO VALE DONDE SE MIDIÓ. Si el paquete declara una de
    # alcance «mismo entorno» y aquí el entorno es OTRO, esa tolerancia no
    # se midió para esta máquina: comparar contra ella daría `FAIL` a un
    # paquete honesto por estar verificándolo en otro sitio. Un fallo por
    # falta de acceso no es una manipulación, y esto es lo mismo.
    fuera_de_alcance = _fuera_del_alcance_de_la_tolerancia(
        manifiesto, comparables, locale=locale)
    if fuera_de_alcance:
        return _etapa("INCOMPARABLE", fuera_de_alcance)

    # Y si el reentrenamiento NO pudo aplicar parte de la configuración
    # capturada, las métricas de ahora no describen el mismo run: comparar
    # daría un FAIL que acusa al paquete de algo que hizo el verificador.
    sin_aplicar = training.get("not_applied_from_capture") or []
    if sin_aplicar:
        return _etapa("INCOMPARABLE",
                      motivo("r3_configuracion_sin_aplicar", locale,
                             claves=", ".join(sin_aplicar)),
                      not_applied=sin_aplicar)

    fuera: list[dict[str, Any]] = []
    dentro: list[str] = []
    for m in comparables:
        nombre = str(m.get("name") or "")
        publicado, ahora = m.get("value"), obtenidas.get(nombre)
        if not isinstance(publicado, (int, float)) or not isinstance(ahora, (int, float)):
            continue
        abs_ = m.get("tolerance_abs")
        rel = m.get("tolerance_rel")
        diferencia = abs(float(ahora) - float(publicado))
        limite = float(abs_) if isinstance(abs_, (int, float)) else 0.0
        if isinstance(rel, (int, float)):
            limite = max(limite, abs(float(publicado)) * float(rel))
        if diferencia <= limite:
            dentro.append(nombre)
        else:
            fuera.append({"metric": nombre, "published": publicado,
                          "obtained": ahora, "difference": diferencia,
                          "tolerance": limite})

    if not dentro and not fuera:
        return _etapa("INCOMPARABLE", motivo("r3_ninguna_comparable", locale))
    if fuera:
        # Con los DOS valores y la tolerancia: «fuera de rango» a secas no
        # deja ver si se pasó por poco o por un orden de magnitud.
        return _etapa("FAIL", motivo("r3_fuera_de_tolerancia", locale),
                      metrics=fuera)
    return _etapa("PASS", checked=dentro)


#: Alcances de tolerancia que este verificador sabe interpretar. Uno que no
#: reconozca NO se da por bueno: fallo cerrado, como todo lo demás aquí.
_ALCANCES = ("same_environment_same_seed",)


#: Lo que la captura declara y que, si NO se puede aplicar al reentrenar,
#: hace que las métricas no sean comparables. No es toda la captura: el
#: nombre del fichero o la fecha no cambian el resultado.
_CONFIG_QUE_CAMBIA_EL_RESULTADO = (
    "seeds", "epochs_effective", "engine", "device", "deterministic_options",
    "warm_start",
)


def _spec_con_epocas(spec: Any, epocas: Any) -> Any:
    """Otro `spec` con las épocas del run, o `None` si no se puede.

    `TrainingSpec` y `RunSpec` son `frozen`, así que «aplicar» aquí es
    construir, no asignar. Se devuelve `None` cuando el contrato no declara
    `RUN` o el valor no es un entero: entonces no se ha aplicado nada y quien
    llama tiene que DECIRLO, no fingir que sí.
    """
    import dataclasses  # noqa: PLC0415

    run = getattr(spec, "run", None)
    if run is None or not dataclasses.is_dataclass(run):
        return None
    try:
        n = int(epocas)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    try:
        return dataclasses.replace(spec, run=dataclasses.replace(run, epochs=n))
    except (TypeError, ValueError):
        return None


def _configuracion_del_run(
    manifiesto: dict[str, Any], spec: Any, *, admite_semilla: bool = False,
    maquina: str | None = None, inicializacion_determinista: bool = False,
) -> tuple[dict[str, Any], list[str], dict[str, Any], Any]:
    """Qué se pudo aplicar de la captura al reentrenar, y qué no.

    Devuelve las dos listas porque las dos importan: lo aplicado explica
    por qué el resultado debería parecerse, y **lo no aplicado explica por
    qué podría no hacerlo** — y sin eso R3 acusaría a un paquete honesto
    de haber cambiado.
    """
    # DE DÓNDE SE LEE, y esto era el primero de los tres defectos (medido el
    # 2026-08-24): esta función leía `manifiesto["run_provenance"]`, **que no
    # existe en el manifiesto**. La raíz tiene `provenance`, y dentro
    # `provenance.run_capture` es solo `{present, schema_version, sha256}` — un
    # resumen, no la captura. Así que devolvía `({}, [])` SIEMPRE, y un
    # «no aplicado» vacío junto a un «aplicado» vacío se lee como «se aplicó
    # todo». Afirmaba por omisión.
    #
    # Los parámetros efectivos del run SÍ están publicados, y están en
    # `generation`: es el bloque que el manifiesto compone desde la captura.
    generacion = manifiesto.get("generation")
    if not isinstance(generacion, dict):
        return {}, [], {}, spec

    aplicadas: dict[str, Any] = {}
    sin_aplicar: list[str] = []
    para_el_entrenador: dict[str, Any] = {}
    for clave in _CONFIG_QUE_CAMBIA_EL_RESULTADO:
        valor = generacion.get(clave)
        if valor is None:
            continue
        if clave == "epochs_effective":
            # LAS ÉPOCAS VIVEN EN `spec.run.epochs`, Y LOS DOS SON `frozen`.
            # El código anterior hacía `spec.epochs = …` sobre un atributo que
            # NO EXISTE —`hasattr` daba False y la rama no entraba nunca—, así
            # que las épocas del run no se aplicaban jamás. Con dataclasses
            # congeladas la forma de aplicarlas es construir otro spec.
            nuevas = _spec_con_epocas(spec, valor)
            if nuevas is not None:
                spec = nuevas
                aplicadas[clave] = int(valor)
                continue
        if clave == "seeds" and isinstance(valor, dict):
            # LA SEMILLA DE INICIALIZACIÓN, y era el tercer defecto: esto
            # buscaba `seeds["training"]`, una clave que el manifiesto no
            # publica. Las que hay son `dataset`, `split` e `init`, y la que
            # cambia el resultado de reentrenar es **`init`**: de qué pesos
            # arranca. `dataset` ya la usó R1 para regenerar y `split` viaja
            # dentro del propio `.mxtrain`.
            semilla = valor.get("init")
            if isinstance(semilla, int) and admite_semilla:
                # No se toca el `spec`: la semilla de inicialización es un
                # argumento del ENTRENADOR (`train(..., seed=)`), no del
                # contrato. Se devuelve para que la aplique quien entrena.
                para_el_entrenador["seed"] = semilla
                aplicadas["seeds.init"] = semilla
                continue
            if isinstance(semilla, int) and not admite_semilla and inicializacion_determinista:
                # NO APLICA, que no es lo mismo que NO SE PUDO. Este entrenador
                # arranca SIEMPRE de los mismos valores, así que el
                # reentrenamiento reproduce esa inicialización aunque nadie le
                # pase la semilla. Contarlo como «sin aplicar» dejaba a R3 sin
                # veredicto **para siempre** en un paquete que sí se reproduce.
                aplicadas["seeds.init"] = "no aplica: inicialización determinista"
                continue
            if isinstance(semilla, int) and not admite_semilla:
                # Este entrenador no admite semilla —los modelos FUNCTION no
                # la reciben—, así que se DICE en vez de callarlo.
                sin_aplicar.append("seeds.init")
                continue
        if clave == "device" and maquina is not None:
            # NO SE IMPONE, SE COMPARA. Forzar el entorno de quien verifica
            # sería prometer algo que no se controla; comprobar que coincide es
            # un hecho. Y si NO coincide, sigue sin aplicarse — que es la
            # verdad: sus métricas no describen la misma ejecución.
            if str(valor).strip().lower() == maquina:
                aplicadas[clave] = maquina
                continue
            sin_aplicar.append(clave)
            continue
        if clave == "warm_start" and valor is False:
            # `false` significa «este run arrancó de la inicialización», que es
            # EXACTAMENTE lo que hace reentrenar aquí. Contarlo como «no
            # aplicado» dejaba a R3 en INCOMPARABLE por una condición que sí se
            # cumple — y un INCOMPARABLE de más también miente, en la otra
            # dirección.
            aplicadas[clave] = False
            continue
        # Lo que este verificador no sabe imponer se DICE. `engine` y
        # `device` son del entorno de quien verifica, no del `.mxtrain`:
        # forzarlos aquí sería prometer un entorno que no controlamos.
        sin_aplicar.append(clave)
    return aplicadas, sin_aplicar, para_el_entrenador, spec


def _fuera_del_alcance_de_la_tolerancia(
    manifiesto: dict[str, Any], comparables: list[dict[str, Any]],
    *, locale: str = IDIOMA_POR_DEFECTO,
) -> str | None:
    """El motivo por el que la tolerancia no aplica aquí, o `None`.

    Sin esto, un paquete medido en una máquina y verificado en otra saldría
    `FAIL` por diferencias que su tolerancia nunca prometió cubrir — y el
    informe diría «al menos una métrica se sale de su tolerancia», que se
    lee como que alguien tocó algo.
    """
    alcances = {str(m.get("tolerance_scope") or "") for m in comparables}
    alcances.discard("")
    if not alcances:
        return None

    desconocidos = sorted(a for a in alcances if a not in _ALCANCES)
    if desconocidos:
        return motivo("tol_alcance_desconocido", locale,
                      alcances=", ".join(desconocidos))

    if "same_environment_same_seed" not in alcances:
        return None

    declarado = (manifiesto.get("environment") or {}).get("environment_sha256")
    if not isinstance(declarado, str) or not declarado:
        return motivo("tol_sin_digest_de_entorno", locale)

    from matrixai.export.reproduce import build_environment
    actual = build_environment().get("environment_sha256")
    if actual == declarado:
        return None
    # Los DOS digests: «otro entorno» a secas no deja ver en qué se diferencia.
    return motivo("tol_otro_entorno", locale,
                  declarado=declarado[:16], actual=str(actual)[:16])


class PaqueteZipRechazado(ValueError):
    """El ZIP no se abre, y se dice por qué en vez de mirar dentro igualmente."""


def _extraer_paquete(zip_path: Path, destino: Path) -> Path:
    """Descomprime el ZIP en `destino` y devuelve la raíz del paquete.

    **Un ZIP viene de fuera.** Una entrada con `..` o con ruta absoluta escribe
    donde el archivo mande —el zip-slip de siempre—, así que se rechaza el
    archivo ENTERO en vez de saltarse esa entrada: un paquete que intenta eso no
    es un paquete al que se le vaya a dar un informe.

    Si todo cuelga de una sola carpeta —lo que escribe `export-bundle`—, la raíz
    es esa carpeta; si no, es `destino`.
    """
    with zipfile.ZipFile(zip_path) as z:
        nombres = z.namelist()
        for nombre in nombres:
            candidata = Path(nombre)
            if candidata.is_absolute() or ".." in candidata.parts or nombre.startswith("/"):
                raise PaqueteZipRechazado(
                    f"the archive contains an entry that escapes it ({nombre!r}): "
                    "extracting it would write outside the package, so it is not "
                    "unpacked at all")
        z.extractall(destino)
    cimas = {Path(n).parts[0] for n in nombres if Path(n).parts}
    if len(cimas) == 1:
        unica = destino / next(iter(cimas))
        if unica.is_dir():
            return unica
    return destino


def verify_package(bundle_dir: str | Path, *, run_training: bool = False,
                   locale: str = IDIOMA_POR_DEFECTO) -> dict[str, Any]:
    """Verifica un paquete y devuelve el informe por etapas.

    `run_training=False` por defecto a propósito: reentrenar cuesta
    minutos u horas y no se impone a quien solo quería comprobar la
    integridad. No haberlo hecho se DICE (`NOT_RUN`), que no es lo mismo
    que haberlo hecho y que saliera bien.

    `locale` manda sobre TODO lo que este informe redacta —los motivos de
    cada etapa, la nota del entrenamiento y el `problem` de cada artefacto
    roto—, y sobre nada más: los `status`, los nombres de etapa y lo que
    un motivo interpola son valores y no cambian de idioma (85-C2b).
    Español por defecto, como el resto del core.
    """
    bundle = Path(bundle_dir)
    # UN ZIP ES LO QUE LA GENTE SE DESCARGA. Esto solo sabía abrir directorios y
    # un `.zip` caía en «el paquete no lleva reproduce.json» — que es falso: lo
    # lleva dentro, y el verificador nunca lo abrió. Medido el 2026-08-26 sobre
    # los tres paquetes de la galería: los tres decían eso, y los tres lo traían.
    if bundle.is_file() and zipfile.is_zipfile(bundle):
        with tempfile.TemporaryDirectory(prefix="matrixai-zip-") as tmp:
            try:
                extraido = _extraer_paquete(bundle, Path(tmp))
            except PaqueteZipRechazado as exc:
                return {"ok": False, "fully_checked": False,
                        "unchecked_stages": ["manifest", "R1", "training", "R3"],
                        "stages": {n: _etapa("INCOMPARABLE", str(exc))
                                   for n in ("manifest", "R1", "training", "R3")},
                        "exit_code": SALIDAS["incomparable"]}
            return verify_package(extraido, run_training=run_training, locale=locale)

    manifiesto, problema = _leer_manifiesto(bundle, locale=locale)

    if manifiesto is None:
        sin_manifiesto = motivo("sin_manifiesto_que_comparar", locale)
        etapas = {
            "manifest": _etapa("INCOMPARABLE", problema),
            "R1": _etapa("INCOMPARABLE", sin_manifiesto),
            "training": _etapa("INCOMPARABLE", sin_manifiesto),
            "R3": _etapa("INCOMPARABLE", sin_manifiesto),
        }
        # La MISMA forma que la salida normal: al añadir `fully_checked`
        # y `unchecked_stages` este camino se quedó sin ellas, y quien
        # leyera la respuesta se encontraba dos formas para lo mismo —
        # un `KeyError` justo en el caso raro.
        return {"ok": False, "fully_checked": False,
                "unchecked_stages": sorted(etapas),
                "stages": etapas, "exit_code": SALIDAS["incomparable"]}

    etapas: dict[str, dict[str, Any]] = {}
    etapas["manifest"] = _verificar_manifiesto(bundle, manifiesto, locale=locale)

    if etapas["manifest"]["status"] != "PASS":
        # Sin integridad, comparar lo demás no dice nada: los artefactos
        # que se compararían no son los que el manifiesto describe.
        sin_integridad = motivo("sin_integridad", locale)
        for etapa in ("R1", "training", "R3"):
            etapas[etapa] = _etapa("INCOMPARABLE", sin_integridad)
    else:
        etapas["R1"] = _verificar_r1(bundle, manifiesto, locale=locale)
        # LO QUE NO SE PUEDE HACER NO SE DICE COMO «NO SE PIDIÓ» (87-C3).
        #
        # Medido el 2026-08-25 sobre un paquete sin `.mxtrain`: sin `--retrain`
        # el informe decía «retraining was not requested», que apunta al
        # usuario cuando el obstáculo es del paquete — pedirlo no habría
        # servido de nada. Y CON `--retrain` culpaba al dataset, que es el
        # SEGUNDO obstáculo: sin contrato no hay con qué entrenar aunque los
        # datos estuvieran.
        #
        # `NOT_RUN` es una elección de quien verifica; `INCOMPARABLE` es un
        # límite del paquete. Aquí es lo segundo, y por eso cuenta como etapa
        # sin comprobar.
        _artefactos = manifiesto.get("artifacts")
        _contrato = (_artefactos or {}).get("training") if isinstance(_artefactos, dict) else None
        if not _contrato:
            etapas["training"] = _etapa("INCOMPARABLE", motivo("t_sin_mxtrain", locale),
                                        missing=["training"])
        else:
            etapas["training"] = (
                _etapa("NOT_RUN", motivo("t_no_pedido", locale))
                if not run_training else
                _verificar_training(bundle, manifiesto, etapas["R1"], locale=locale)
            )
        etapas["R3"] = _verificar_r3(manifiesto, etapas["training"], locale=locale)

    ordenadas = {n: etapas[n] for n in _ORDEN}
    # AUDITORÍA EXTERNA (2026-08-20) [ALTO]: solo se miraba `FAIL`, así
    # que un paquete con una etapa `INCOMPARABLE` salía con **código 0 y
    # `ok: true`** — el código de «nada falló». Y no es lo mismo: nada
    # falló Y algo no se pudo comprobar. Quien encadene `verify && desplegar`
    # estaría tratando un paquete no verificado como verificado, que es la
    # media verdad tranquilizadora de siempre.
    #
    # `NOT_RUN` NO cuenta: significa que nadie lo pidió (`--retrain`), y eso
    # es una elección de quien verifica, no una limitación del paquete.
    hay_fallo = any(e["status"] == "FAIL" for e in ordenadas.values())
    sin_comprobar = sorted(k for k, e in ordenadas.items()
                           if e["status"] == "INCOMPARABLE")
    if hay_fallo:
        codigo = SALIDAS["fail"]
    elif sin_comprobar:
        codigo = SALIDAS["incomparable"]
    else:
        codigo = SALIDAS["ok"]
    return {
        # `ok` sigue queriendo decir «nada falló» —es lo que significaba y
        # cambiarlo rompería a quien lo lea—, pero ya no viaja solo:
        # `fully_checked` dice si además se pudo comprobar TODO.
        "ok": not hay_fallo,
        "fully_checked": not sin_comprobar and not hay_fallo,
        "unchecked_stages": sin_comprobar,
        "stages": ordenadas,
        "exit_code": codigo,
    }
