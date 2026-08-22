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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matrixai.export.reproduce import (
    REPRODUCE_MANIFEST_FILENAME as REPRODUCE_FILENAME,
    manifest_digest,
    sha256_file,
)

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


def _leer_manifiesto(bundle: Path) -> tuple[dict[str, Any] | None, str | None]:
    ruta = bundle / REPRODUCE_FILENAME
    if not ruta.is_file():
        return None, f"the package carries no {REPRODUCE_FILENAME}"
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{REPRODUCE_FILENAME} is not readable JSON: {exc}"
    if not isinstance(datos, dict):
        return None, f"{REPRODUCE_FILENAME} is not an object"
    return datos, None


def _comprobar_inventario(
    bundle: Path, manifiesto: dict[str, Any]
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
        problema = _ruta_fuera_del_paquete(bundle, relativa)
        if problema is not None:
            rotos.append({"artifact": relativa, "path": relativa, "problem": problema})
            continue
        if not fichero.is_file() or fichero.is_symlink():
            rotos.append({"artifact": relativa, "path": relativa,
                          "problem": "declared in the manifest but missing "
                                     "from the package"})
            continue
        real = sha256_file(fichero)
        if real != esperado:
            rotos.append({"artifact": relativa, "path": relativa,
                          "problem": "sha256 mismatch", "expected": esperado,
                          "found": real})
    # Lo que SOBRA: está en el paquete y el manifiesto no lo nombra.
    sin_cubrir = sorted(set(presentes) - set(declarados))
    return rotos, sin_cubrir


def _ruta_fuera_del_paquete(bundle: Path, ruta: str) -> str | None:
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
        return "path is empty or padded with spaces"
    candidata = Path(ruta)
    if candidata.is_absolute():
        return "path is absolute, and a package only describes its own files"
    if ".." in candidata.parts:
        return "path escapes the package with '..'"
    try:
        raiz = bundle.resolve(strict=False)
        destino = (bundle / candidata).resolve(strict=False)
    except OSError as exc:  # pragma: no cover — rutas imposibles del sistema
        return f"path cannot be resolved: {exc}"
    if raiz != destino and raiz not in destino.parents:
        # Cubre el enlace simbólico que apunta fuera: la ruta escrita
        # parece de dentro y el fichero real no lo es.
        return "path resolves outside the package (symlink?)"
    return None


def _verificar_manifiesto(bundle: Path, manifiesto: dict[str, Any]) -> dict[str, Any]:
    """Integridad del manifiesto Y de cada artefacto que declara.

    Las dos cosas, y no solo la primera: un `manifest_sha256` correcto
    sobre unos artefactos cambiados diría que todo está bien.
    """
    declarado = manifiesto.get("manifest_sha256")
    if not isinstance(declarado, str) or not declarado:
        return _etapa("FAIL", "the manifest declares no manifest_sha256")
    if declarado != manifest_digest(manifiesto):
        return _etapa("FAIL", "the manifest does not match its own manifest_sha256",
                      field="manifest_sha256")

    artefactos = manifiesto.get("artifacts")
    if not isinstance(artefactos, dict):
        return _etapa("FAIL", "the manifest declares no artifacts")

    # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: la versión de esquema
    # no se miraba. Medido: un manifiesto con `schema_version: "999.0"`
    # salía `manifest PASS`. Interpretar a medias un formato que no se
    # conoce es peor que no leerlo.
    version = manifiesto.get("schema_version")
    if version not in _ESQUEMAS_CONOCIDOS:
        return _etapa("INCOMPARABLE",
                      f"the manifest declares schema_version {version!r} and this "
                      f"verifier reads {list(_ESQUEMAS_CONOCIDOS)}: a format it "
                      "does not know is not interpreted halfway")

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
                          "problem": "the artifact entry is neither an object nor "
                                     "a declared absence (null)"})
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
                          "problem": ("the artifact ships a file with no sha256 to "
                                      "check it against" if isinstance(ruta, str)
                                      else "the artifact declares neither a path "
                                           "nor a sha256")})
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
        problema = _ruta_fuera_del_paquete(bundle, ruta)
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
                          "problem": "the artifact is a symlink; an integrity "
                                     "artifact must be a regular file"})
            continue
        if not fichero.is_file():
            rotos.append({"artifact": nombre, "path": ruta, "problem": "missing"})
            continue
        real = sha256_file(fichero)
        if real != esperado:
            rotos.append({"artifact": nombre, "path": ruta, "problem": "sha256 mismatch",
                          "expected": esperado, "found": real})
        else:
            comprobados.append(nombre)

    if rotos:
        # Se nombra QUÉ artefacto: «algo no cuadra» obliga a abrir el ZIP
        # y comparar a mano los cinco ficheros.
        return _etapa("FAIL", "at least one artifact does not match its declared sha256",
                      artifacts=rotos)

    # REFUTACIÓN (2026-08-20) [BLOQUEANTE]: hasta aquí se comprobaba lo que
    # el manifiesto DECLARA —cuatro artefactos— y el paquete LLEVA catorce
    # ficheros. Sustituyendo `model.onnx`, `predict.py` y los pesos, esto
    # contestaba `PASS`. Y el Space ejecuta ese `predict.py` justo después.
    rotos_del_inventario, sin_cubrir = _comprobar_inventario(bundle, manifiesto)
    if rotos_del_inventario:
        return _etapa("FAIL",
                      "at least one file in the package does not match its declared sha256",
                      artifacts=rotos_del_inventario)
    if sin_cubrir:
        # NO es `FAIL`: el paquete no miente, es que no cubre esos
        # ficheros — y decir `PASS` sería afirmar sobre lo que no se ha
        # mirado. Los paquetes anteriores a este inventario caen aquí.
        return _etapa("INCOMPARABLE",
                      "the package ships files the manifest does not cover, so "
                      "their integrity cannot be checked",
                      uncovered_files=sin_cubrir, checked=comprobados)
    return _etapa("PASS", checked=comprobados,
                  files_checked=len(manifiesto.get("files") or {}))


def _verificar_r1(bundle: Path, manifiesto: dict[str, Any]) -> dict[str, Any]:
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
        return _etapa("INCOMPARABLE", str(r1.get("reason") or "R1 is not possible for this package"),
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
                      "cannot regenerate the dataset and compare its sha256: "
                      + ", ".join(faltan) + " unknown",
                      missing=faltan)

    # TODO lo que decidió aquel CSV viaja en el manifiesto: `mode`,
    # `field_ranges`, `field_types`, `field_categories` y los
    # identificadores excluidos. (Mi primera versión decía que no y
    # devolvía NOT_RUN por eso: era falso, y lo era porque lo supuse en
    # vez de mirar el `generation` que produce el manifiesto.)
    modelo = artefactos.get("model") if isinstance(artefactos, dict) else None
    if not isinstance(modelo, dict) or not modelo.get("path"):
        return _etapa("INCOMPARABLE", "the package carries no model to regenerate from",
                      missing=["model"])
    entrenamiento = artefactos.get("training") if isinstance(artefactos, dict) else None
    if not isinstance(entrenamiento, dict) or not entrenamiento.get("path"):
        # El generador exige el `.mxtrain`: sin él no hay R1, y eso es una
        # ausencia, no una manipulación.
        return _etapa("INCOMPARABLE",
                      "the package carries no .mxtrain, which the generator requires",
                      missing=["training"])

    filas = dataset.get("rows") if isinstance(dataset, dict) else None
    if not isinstance(filas, int) or filas <= 0:
        return _etapa("INCOMPARABLE", "the package does not declare how many rows to generate",
                      missing=["dataset_rows"])

    try:
        mxai_text = (bundle / str(modelo["path"])).read_text(encoding="utf-8")
        training_text = (bundle / str(entrenamiento["path"])).read_text(encoding="utf-8")
        recipe_text = (bundle / str(receta["path"])).read_text(encoding="utf-8")
    except OSError as exc:
        return _etapa("INCOMPARABLE", f"the package artifacts cannot be read: {exc}")

    # EL MODO NO SE INVENTA. Mi primera versión ponía `"deterministic"`
    # de respaldo —que además ni siquiera es un modo válido: son `random`
    # y `coherent`—, y regenerar con un modo distinto del que se usó da
    # otro dataset y por tanto un FAIL falso, acusando a un paquete
    # honesto. Sin modo declarado, no hay con qué comparar.
    modo = generacion.get("mode") if isinstance(generacion, dict) else None
    if not modo:
        return _etapa("INCOMPARABLE",
                      "the package does not declare which generation mode produced "
                      "its dataset, and guessing one would rebuild a different one",
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
        return _etapa("INCOMPARABLE", f"the dataset could not be regenerated: {exc}")

    # `csv_text`, no `csv`: es la clave que el generador devuelve de
    # verdad (medida, después de suponer la otra y comerme un KeyError).
    csv = generado.get("csv_text") if isinstance(generado, dict) else None
    if not isinstance(csv, str) or csv == "":
        return _etapa("INCOMPARABLE", "the generator returned no CSV")

    import hashlib
    obtenido = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    if obtenido == esperado:
        return _etapa("PASS", rows=filas, sha256=obtenido, csv_text=csv)
    # AQUÍ sí es FAIL: el paquete dijo qué debía salir y ha salido otra
    # cosa. Se enseñan los DOS digests: «no coincide» a secas no deja
    # comprobar nada a quien lo lea.
    return _etapa("FAIL",
                  "the regenerated dataset does not have the sha256 the package declares",
                  expected=esperado, found=obtenido)


def _verificar_training(
    bundle: Path, manifiesto: dict[str, Any], r1: dict[str, Any],
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
        return _etapa("INCOMPARABLE",
                      "the dataset could not be regenerated, so there is nothing to train on")

    csv = r1.get("csv_text")
    if not isinstance(csv, str) or not csv:
        return _etapa("INCOMPARABLE", "the regenerated dataset was not kept for training")

    artefactos = manifiesto.get("artifacts") or {}
    entrenamiento = artefactos.get("training") if isinstance(artefactos, dict) else None
    modelo = artefactos.get("model") if isinstance(artefactos, dict) else None
    if not isinstance(entrenamiento, dict) or not isinstance(modelo, dict):
        return _etapa("INCOMPARABLE", "the package carries no .mxtrain to retrain from")

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
        # justo donde el contrato lo busca.
        fuente = getattr(getattr(spec, "dataset", None), "source", None)
        destino = taller / (Path(str(fuente)).name if fuente else "dataset.csv")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(csv, encoding="utf-8")

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
        resultado = entrenador.train(
            spec, output_dir=str(taller / "out"), base_path=taller,
            training_path=ruta_train)
    except Exception as exc:  # noqa: BLE001 — el entrenador se niega por mil motivos
        # Que no se pueda entrenar AQUÍ no prueba que el paquete mienta:
        # otro entorno, otro backend, una dependencia que falta.
        return _etapa("INCOMPARABLE", f"retraining could not run: {exc}")
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
    aplicadas, sin_aplicar = _configuracion_del_run(manifiesto, spec)

    epocas = getattr(resultado, "best_epoch", None)
    # LOS NOMBRES SON LOS DEL RESULTADO, medidos: `TrainingRunResult` no
    # tiene un `metrics` —lo supuse y salía siempre vacío—, tiene
    # `validation_metrics` (las que declara el `.mxtrain`, por su nombre),
    # `accuracy` y las pérdidas.
    medidas: dict[str, Any] = {}
    validacion = getattr(resultado, "validation_metrics", None)
    if isinstance(validacion, dict):
        medidas.update({str(k): v for k, v in validacion.items()})
    for campo in ("accuracy", "best_validation_loss",
                  "final_train_loss", "final_validation_loss"):
        valor = getattr(resultado, campo, None)
        if isinstance(valor, (int, float)):
            medidas.setdefault(campo, valor)
    return _etapa("PASS", best_epoch=epocas, metrics=medidas,
                  note="training completed; matching the published metrics is R3",
                  applied_from_capture=aplicadas,
                  not_applied_from_capture=sin_aplicar)


def _verificar_r3(manifiesto: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
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
            return _etapa("NOT_RUN",
                          "metrics can only be contrasted against a fresh training "
                          "run (use --retrain)")
        porque = training.get("reason") or "training did not pass"
        return _etapa("NOT_RUN",
                      f"the retraining run did not complete, so there is no fresh "
                      f"value to contrast: {porque}")

    metricas = manifiesto.get("metrics")
    if not isinstance(metricas, list) or not metricas:
        return _etapa("INCOMPARABLE", "the package publishes no metrics to contrast")

    comparables = [m for m in metricas if isinstance(m, dict) and m.get("comparable")]
    if not comparables:
        # Lo que falta lo dice el manifiesto, métrica a métrica. Se
        # reproduce esa lista para que quien lo lea sepa QUÉ pedirle al
        # productor del paquete, en vez de «no se puede».
        falta: list[str] = []
        for m in metricas:
            if isinstance(m, dict):
                falta.extend(str(x) for x in (m.get("incomplete") or []))
        return _etapa("INCOMPARABLE",
                      "no published metric carries what a comparison needs",
                      missing=sorted(set(falta)))

    obtenidas = training.get("metrics")
    if not isinstance(obtenidas, dict) or not obtenidas:
        return _etapa("INCOMPARABLE",
                      "the retraining did not report metrics to contrast")

    # LA TOLERANCIA SOLO VALE DONDE SE MIDIÓ. Si el paquete declara una de
    # alcance «mismo entorno» y aquí el entorno es OTRO, esa tolerancia no
    # se midió para esta máquina: comparar contra ella daría `FAIL` a un
    # paquete honesto por estar verificándolo en otro sitio. Un fallo por
    # falta de acceso no es una manipulación, y esto es lo mismo.
    fuera_de_alcance = _fuera_del_alcance_de_la_tolerancia(manifiesto, comparables)
    if fuera_de_alcance:
        return _etapa("INCOMPARABLE", fuera_de_alcance)

    # Y si el reentrenamiento NO pudo aplicar parte de la configuración
    # capturada, las métricas de ahora no describen el mismo run: comparar
    # daría un FAIL que acusa al paquete de algo que hizo el verificador.
    sin_aplicar = training.get("not_applied_from_capture") or []
    if sin_aplicar:
        return _etapa("INCOMPARABLE",
                      "the retraining could not apply part of the captured "
                      f"configuration ({', '.join(sin_aplicar)}), so its metrics "
                      "do not describe the same run and a difference would not "
                      "prove the package wrong",
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
        return _etapa("INCOMPARABLE",
                      "none of the comparable metrics was reported by the retraining")
    if fuera:
        # Con los DOS valores y la tolerancia: «fuera de rango» a secas no
        # deja ver si se pasó por poco o por un orden de magnitud.
        return _etapa("FAIL", "at least one metric falls outside its tolerance",
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


def _configuracion_del_run(
    manifiesto: dict[str, Any], spec: Any
) -> tuple[dict[str, Any], list[str]]:
    """Qué se pudo aplicar de la captura al reentrenar, y qué no.

    Devuelve las dos listas porque las dos importan: lo aplicado explica
    por qué el resultado debería parecerse, y **lo no aplicado explica por
    qué podría no hacerlo** — y sin eso R3 acusaría a un paquete honesto
    de haber cambiado.
    """
    captura = manifiesto.get("run_provenance")
    if not isinstance(captura, dict):
        return {}, []

    aplicadas: dict[str, Any] = {}
    sin_aplicar: list[str] = []
    for clave in _CONFIG_QUE_CAMBIA_EL_RESULTADO:
        valor = captura.get(clave)
        if valor is None:
            continue
        if clave == "epochs_effective" and hasattr(spec, "epochs"):
            # Se aplica de verdad: el `.mxtrain` del paquete puede llevar
            # otras épocas que las que el run usó.
            try:
                spec.epochs = int(valor)
                aplicadas[clave] = int(valor)
                continue
            except (TypeError, ValueError, AttributeError):
                pass
        if clave == "seeds" and isinstance(valor, dict) and hasattr(spec, "seed"):
            semilla = valor.get("training")
            if isinstance(semilla, int):
                try:
                    spec.seed = semilla
                    aplicadas["seeds.training"] = semilla
                    continue
                except AttributeError:
                    pass
        # Lo que este verificador no sabe imponer se DICE. `engine` y
        # `device` son del entorno de quien verifica, no del `.mxtrain`:
        # forzarlos aquí sería prometer un entorno que no controlamos.
        sin_aplicar.append(clave)
    return aplicadas, sin_aplicar


def _fuera_del_alcance_de_la_tolerancia(
    manifiesto: dict[str, Any], comparables: list[dict[str, Any]]
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
        return (f"the package declares tolerance scopes this verifier does not "
                f"know ({', '.join(desconocidos)}), so it cannot tell whether "
                "they apply here")

    if "same_environment_same_seed" not in alcances:
        return None

    declarado = (manifiesto.get("environment") or {}).get("environment_sha256")
    if not isinstance(declarado, str) or not declarado:
        return ("the package declares an environment-scoped tolerance but no "
                "environment digest, so there is no way to tell whether this "
                "is the environment it was measured in")

    from matrixai.export.reproduce import build_environment
    actual = build_environment().get("environment_sha256")
    if actual == declarado:
        return None
    # Los DOS digests: «otro entorno» a secas no deja ver en qué se diferencia.
    return (f"the tolerance was measured for the package's own environment "
            f"({declarado[:16]}…) and this one is different ({str(actual)[:16]}…), "
            "so a difference here would not prove the package wrong")


def verify_package(bundle_dir: str | Path, *, run_training: bool = False) -> dict[str, Any]:
    """Verifica un paquete y devuelve el informe por etapas.

    `run_training=False` por defecto a propósito: reentrenar cuesta
    minutos u horas y no se impone a quien solo quería comprobar la
    integridad. No haberlo hecho se DICE (`NOT_RUN`), que no es lo mismo
    que haberlo hecho y que saliera bien.
    """
    bundle = Path(bundle_dir)
    manifiesto, problema = _leer_manifiesto(bundle)

    if manifiesto is None:
        etapas = {
            "manifest": _etapa("INCOMPARABLE", problema),
            "R1": _etapa("INCOMPARABLE", "no manifest to compare against"),
            "training": _etapa("INCOMPARABLE", "no manifest to compare against"),
            "R3": _etapa("INCOMPARABLE", "no manifest to compare against"),
        }
        # La MISMA forma que la salida normal: al añadir `fully_checked`
        # y `unchecked_stages` este camino se quedó sin ellas, y quien
        # leyera la respuesta se encontraba dos formas para lo mismo —
        # un `KeyError` justo en el caso raro.
        return {"ok": False, "fully_checked": False,
                "unchecked_stages": sorted(etapas),
                "stages": etapas, "exit_code": SALIDAS["incomparable"]}

    etapas: dict[str, dict[str, Any]] = {}
    etapas["manifest"] = _verificar_manifiesto(bundle, manifiesto)

    if etapas["manifest"]["status"] != "PASS":
        # Sin integridad, comparar lo demás no dice nada: los artefactos
        # que se compararían no son los que el manifiesto describe.
        motivo = "the package integrity check did not pass, so nothing else can be trusted"
        for etapa in ("R1", "training", "R3"):
            etapas[etapa] = _etapa("INCOMPARABLE", motivo)
    else:
        etapas["R1"] = _verificar_r1(bundle, manifiesto)
        etapas["training"] = (
            _etapa("NOT_RUN", "retraining was not requested (use --retrain)")
            if not run_training else
            _verificar_training(bundle, manifiesto, etapas["R1"])
        )
        etapas["R3"] = _verificar_r3(manifiesto, etapas["training"])

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
