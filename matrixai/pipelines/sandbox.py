"""`Replay & Verify` en entorno aislado — contrato 81-C5.

Dos reglas de la revisión, y la segunda es la que de verdad protege:

> **P23-R-0022.** Se declaran los backends de aislamiento soportados y su
> versión mínima. «Se ejecuta aislado» sin decir con qué no se puede
> auditar: quien lo lea no sabe qué está confiando.
>
> **P23-R-0023.** Cuando ninguno está disponible, **se falla cerrado**:
> no se ejecuta y se dice por qué. **No** se degrada en silencio a
> ejecución sin aislamiento — sería prometer un sandbox y correr sin él,
> que es peor que no ofrecerlo.

Por eso **no hay bandera** para saltárselo. Una salida de emergencia que
nadie vigila acaba siendo el camino normal, y este módulo existe
justamente para que reproducir un paquete ajeno no sea un riesgo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["BACKENDS_SOPORTADOS", "comparar_con_referencia",
           "entorno_de_reproduccion",
           "inventario_de_dependencias", "inventario_dentro_de", "recibo_de_reproduccion", "IMAGEN_POR_DEFECTO", "SinAislamiento",
           "argv_de_comprobacion_previa", "argv_de_reproduccion",
           "describir_aislamiento", "imagen_del_sandbox",
           "replay_and_verify"]

#: La imagen donde se reproduce. Es la del PROPIO producto: reproducir un
#: paquete de MatrixAI en una imagen genérica exigiría instalar algo, y
#: dentro del sandbox **no hay red** —a propósito—, así que no habría con
#: qué. Se puede cambiar con `MATRIXAI_SANDBOX_IMAGE`.
IMAGEN_POR_DEFECTO = "matrixai-studio:v2.0"

#: Los backends soportados, con su versión mínima (P23-R-0022). Docker va
#: primero porque es el que esta casa ya usa y tiene medido: la suite de
#: navegador corre ahí con `--memory` y `--cpus`.
BACKENDS_SOPORTADOS: tuple[dict[str, str], ...] = (
    {"name": "docker", "min_version": "20.10", "probe": "docker"},
    {"name": "podman", "min_version": "4.0", "probe": "podman"},
    {"name": "bubblewrap", "min_version": "0.5", "probe": "bwrap"},
)

#: Los límites que se aplican SIEMPRE. Se declaran aquí y viajan en el
#: informe: decir «aislado» sin decir con qué límites es medio dato, y un
#: sandbox sin tope de memoria no protege de lo que más pasa.
LIMITES = {
    "network": "disabled",     # por defecto, y no hay forma de pedir red
    "cpu": "2",
    "memory": "2g",
    "timeout_s": 900,
    "filesystem": "read-only except the package copy",
}


class SinAislamiento(RuntimeError):
    """No hay con qué aislar, así que no se ejecuta nada."""


def _version_de(programa: str) -> str | None:
    """La versión del backend, o `None` si no está.

    Aparte para poder sustituirlo en las pruebas: comprobar el fallo
    cerrado no puede depender de que la máquina de turno tenga o no
    Docker instalado.
    """
    if shutil.which(programa) is None:
        return None
    try:
        salida = subprocess.run([programa, "--version"], capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if salida.returncode != 0:
        return None
    for pieza in salida.stdout.split():
        if pieza[:1].isdigit():
            return pieza.strip(",")
    return None


def describir_aislamiento() -> dict[str, Any]:
    """Qué backend hay, y cuáles se buscaron.

    Los buscados viajan a propósito: «no hay sandbox» a secas no dice qué
    instalar, y quien lo lea tendría que ir al código a averiguarlo.
    """
    comprobados: list[dict[str, Any]] = []
    elegido: dict[str, Any] | None = None
    for backend in BACKENDS_SOPORTADOS:
        version = _version_de(backend["probe"])
        comprobados.append({"name": backend["name"], "min_version": backend["min_version"],
                            "found": version})
        if version and elegido is None:
            elegido = {"backend": backend["name"], "version": version}
    return {"available": elegido is not None, "chosen": elegido, "checked": comprobados}


def imagen_del_sandbox() -> str:
    """La imagen declarada, o la del producto."""
    return (os.environ.get("MATRIXAI_SANDBOX_IMAGE") or "").strip() or IMAGEN_POR_DEFECTO


def _hay_imagen(programa: str, imagen: str) -> bool:
    """¿Está la imagen AQUÍ? No se descarga sola.

    Dejar que `docker run` la baje metería una descarga de red en el
    camino de un `replay` que promete no tener red, y encima traería
    bytes que nadie ha mirado. Si no está, se dice y no se ejecuta.
    """
    try:
        salida = subprocess.run([programa, "image", "inspect", imagen],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return salida.returncode == 0


def _argv_base(programa: str, imagen: str, paquete: str | None) -> list[str]:
    """La parte del comando que NO cambia: los límites y el aislamiento."""
    montaje: list[str] = []
    if paquete is not None:
        montaje = ["-v", f"{Path(paquete).resolve()}:/pkg:ro"]
    return [
        programa, "run", "--rm",
        "--network=none",
        # Reproducir es correr un paquete que viene de cualquier sitio: sin
        # esto, un `setuid` dentro del contenedor puede subir privilegios
        # aunque se entre con un uid sin ellos. Es barato y cierra la puerta.
        # (Sugerencia de la sesión que escribió el resto del C5.)
        "--security-opt", "no-new-privileges",
        f"--cpus={LIMITES['cpu']}",
        f"--memory={LIMITES['memory']}",
        f"--memory-swap={LIMITES['memory']}",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=512m",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        *montaje,
        "--entrypoint", "python3",
        imagen,
    ]


def argv_de_comprobacion_previa(programa: str, imagen: str) -> list[str]:
    """Le pregunta a la imagen si SABE verificar, antes de creerle un veredicto.

    Hace falta, y no es celo: `matrixai verify` devuelve **2** cuando
    alguien tocó el paquete, y `argparse` devuelve **2** cuando el
    subcomando no existe. Son el mismo número. Medido el 2026-08-20
    contra `matrixai-studio:v2.0`, cuyo `matrixai` es anterior al 82-C2:
    la reproducción salía con `rc=2` y se leía como **«el paquete está
    manipulado»** cuando lo que pasaba era que la imagen no sabe
    comprobarlo.

    Acusar a un paquete honesto porque el verificador es viejo es
    exactamente el fallo que este contrato existe para no cometer.
    """
    return [*_argv_base(programa, imagen, None),
            "-m", "matrixai", "verify", "--help"]


def argv_de_reproduccion(programa: str, paquete: str, imagen: str,
                        *, reentrenar: bool = True) -> list[str]:
    """El comando EXACTO que se ejecuta, aparte para poder auditarlo.

    Viaja después en el informe: decir «se reprodujo aislado» sin decir
    con qué argumentos es una afirmación que nadie puede comprobar, y el
    argv es la única prueba de que los límites declarados en `LIMITES`
    son los que de verdad se aplicaron.

    Lo que no es negociable y por qué:

    * **`--network=none`**: un replay que puede llamar a casa no es un
      replay, es una ejecución de código ajeno con salida a internet.
    * **`--memory-swap` igual a `--memory`**: sin él el contenedor se
      derrama al swap y el tope deja de topar — medido en esta casa, y
      costó el servidor dos veces.
    * **`--user` con el uid de quien llama**: sin él el contenedor
      escribe como ROOT lo que toque, y lo siguiente que pasa es un
      `EACCES` que no dice nada. También costó dos veces.
    * **`--read-only` y el paquete montado `:ro`**: reproducir es leer y
      comprobar; si el paquete pudiera cambiar durante su propia
      verificación, el veredicto no diría nada de los bytes que llegaron.
    """
    # AUDITORÍA EXTERNA (2026-08-20) [BLOQUEANTE]: se ejecutaba `verify`
    # SIN `--retrain`, así que `training` y `R3` quedaban `NOT_RUN` — y el
    # §15.6 pide que «R3 pueda demostrarse dentro de las tolerancias
    # declaradas». Reproducir sin reentrenar no reproduce: comprueba
    # huellas y se queda ahí.
    #
    # `reentrenar` es un parámetro y no una constante porque reentrenar
    # cuesta: quien solo quiera comprobar integridad no tiene por qué
    # pagarlo. Pero el DEFECTO es reproducir de verdad, y el informe dice
    # cuál de las dos cosas se hizo.
    argv = [*_argv_base(programa, imagen, paquete),
            "-m", "matrixai", "verify", "/pkg", "--json"]
    if reentrenar:
        argv.append("--retrain")
    return argv


def _ejecutar_en(backend: dict[str, Any], paquete: str, *,
                 reentrenar: bool = True) -> dict[str, Any]:
    """Corre la reproducción dentro del backend elegido.

    Lo que se ejecuta dentro es `matrixai verify`, que es el comando del
    contrato 82: reproducir un paquete es comprobar lo que el propio
    paquete dice de sí mismo, y tener DOS comprobaciones distintas sería
    dos sitios diciendo lo mismo.

    Aparte de `replay_and_verify` para que el fallo cerrado se pueda
    probar sin un contenedor de verdad — y para que se vea que quien
    ejecuta NO es la función pública.
    """
    programa = str(backend.get("probe") or backend.get("backend") or "")
    if programa not in ("docker", "podman"):
        # `bubblewrap` aísla, pero no monta imágenes: correr aquí el
        # `matrixai` del host sería reproducir con la instalación de quien
        # pregunta, que es justo lo que un replay tiene que evitar.
        raise SinAislamiento(
            f"el backend {programa!r} no puede reproducir un paquete: hace "
            "falta uno con imágenes (docker o podman), porque reproducir con "
            "el MatrixAI del anfitrión no reproduce nada")

    imagen = imagen_del_sandbox()
    if not _hay_imagen(programa, imagen):
        raise SinAislamiento(
            f"no está la imagen {imagen!r} en esta máquina, y no se descarga "
            "sola: dentro del sandbox no hay red a propósito, y bajar bytes "
            "que nadie ha mirado sería peor.\n"
            # EL COMANDO, no «constrúyela». Es la misma regla que este
            # módulo ya aplica a los backends —«no hay sandbox» a secas no
            # dice qué instalar— y aquí se estaba incumpliendo: lo vio la
            # 3ª pasada de auditoría conduciendo el producto.
            "  docker build -f examples/sandbox/Dockerfile "
            "-t matrixai-sandbox:local .\n"
            "  MATRIXAI_SANDBOX_IMAGE=matrixai-sandbox:local matrixai replay <paquete>")

    previa = argv_de_comprobacion_previa(programa, imagen)
    try:
        sabe = subprocess.run(previa, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "timed_out": False, "returncode": None,
                "command": previa,
                "reason": f"no se pudo preguntar a la imagen si sabe verificar: {exc}"}
    if sabe.returncode != 0:
        # NO se ejecuta la reproducción: su `rc=2` sería indistinguible del
        # de un paquete manipulado, y colgarle eso a un paquete honesto es
        # peor que no comprobar nada.
        return {"ok": False, "timed_out": False, "returncode": None,
                "command": previa,
                "reason": (
                    f"el matrixai de la imagen {imagen!r} no tiene el comando "
                    "`verify`, así que no puede comprobar este paquete. No se "
                    "reproduce igualmente: `verify` devuelve 2 cuando alguien "
                    "tocó el paquete y argparse devuelve 2 cuando el subcomando "
                    "no existe — el mismo número, y acusar a un paquete honesto "
                    "porque el verificador es viejo sería peor que no mirar.\n"
                    "Construye una imagen con este árbol dentro:\n"
                    "  docker build -f examples/sandbox/Dockerfile "
                    "-t matrixai-sandbox:local .\n"
                    "  MATRIXAI_SANDBOX_IMAGE=matrixai-sandbox:local "
                    "matrixai replay <paquete>"),
                "stderr": sabe.stderr}

    argv = argv_de_reproduccion(programa, paquete, imagen, reentrenar=reentrenar)
    try:
        salida = subprocess.run(argv, capture_output=True, text=True,
                                timeout=int(LIMITES["timeout_s"]))
    except subprocess.TimeoutExpired:
        # Agotar el tiempo NO es que el paquete esté mal: es que no se
        # pudo comprobar. Colapsarlo en «falla» acusaría a un paquete
        # honesto que solo tarda.
        return {"ok": False, "timed_out": True, "returncode": None,
                "command": argv,
                "reason": f"la reproducción pasó de {LIMITES['timeout_s']}s "
                          "y se cortó: no se pudo comprobar, que no es lo "
                          "mismo que haber fallado"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "timed_out": False, "returncode": None,
                "command": argv,
                "reason": f"no se pudo lanzar el contenedor: {exc}"}

    # Los códigos de `matrixai verify` (82-C2) significan cosas distintas y
    # NO se colapsan: 0 nada falló, 2 alguien lo tocó o hay discrepancia,
    # 3 no se pudo comprobar. Un booleano perdería «no se pudo».
    return {
        "ok": salida.returncode == 0,
        "timed_out": False,
        "returncode": salida.returncode,
        "command": argv,
        "stdout": salida.stdout,
        "stderr": salida.stderr,
    }


#: EL INVENTARIO, ESCRITO UNA SOLA VEZ.
#:
#: Se guarda como PROGRAMA porque tiene que poder ejecutarse **dentro**
#: del contenedor, que es donde ocurre la reproducción. Tenerlo además
#: como función normal aquí serían dos implementaciones de lo mismo, y
#: dos sitios declarando lo mismo acaban divergiendo: el día que cambie
#: cómo se lee una licencia, el inventario de dentro y el de fuera
#: dirían cosas distintas sobre el mismo paquete.
_PROGRAMA_INVENTARIO = r"""
import json, sys
from importlib import metadata

paquetes = []
for dist in metadata.distributions():
    try:
        nombre = dist.metadata["Name"]
    except (KeyError, TypeError):
        continue
    if not nombre:
        continue
    licencia = dist.metadata.get("License")
    if not licencia or licencia == "UNKNOWN":
        # Muchos paquetes la declaran solo en los clasificadores.
        clasificadores = dist.metadata.get_all("Classifier") or []
        licencia = next(
            (c.split("::")[-1].strip() for c in clasificadores
             if c.startswith("License ::")), None)
    paquetes.append({"name": nombre, "version": dist.version,
                     "license": licencia or None})
paquetes.sort(key=lambda p: p["name"].lower())
sys.stdout.write(json.dumps({
    "packages": paquetes,
    "count": len(paquetes),
    "without_declared_license": [p["name"] for p in paquetes if p["license"] is None],
    "python_version": sys.version.split()[0],
}))
"""


def inventario_de_dependencias() -> dict[str, Any]:
    """Qué hay instalado AQUÍ y con qué licencia.

    OJO A DÓNDE MIRA: este es el inventario de **este** proceso. Para el
    recibo NO sirve —lo que hay que inventariar es el entorno donde se
    reprodujo, que es el contenedor—, y usarlo allí fue justo el defecto
    H3 del refutador (2026-08-20): el recibo traía 109 paquetes del
    anfitrión (`Twisted`, `boto3`, `bcc`…) presentándolos como el entorno
    de reproducción, cuando la imagen tenía 27 y ninguno coincidía. Dos
    imágenes distintas daban inventarios **idénticos**, porque ninguna de
    las dos se estaba mirando. Para el recibo, `inventario_dentro_de`.

    **Lo que no se sabe se dice.** Una licencia que los metadatos no
    declaran sale como `null`, no como «desconocida-pero-seguro-que-vale»:
    inventar una licencia es peor que no tenerla, porque alguien la usaría
    para decidir.
    """
    # El MISMO programa que corre dentro del contenedor, aquí en
    # memoria: se le recoge la salida en vez de reescribirlo.
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(_PROGRAMA_INVENTARIO, "<inventario>", "exec"), {})  # noqa: S102
    return json.loads(buffer.getvalue())


def inventario_dentro_de(programa: str, imagen: str) -> dict[str, Any] | None:
    """El inventario del ENTORNO DONDE SE REPRODUJO. `None` si no se pudo.

    `None` es «no se pudo preguntar», y quien lo reciba tiene que
    **decirlo**, no rellenarlo con el del anfitrión: un inventario que
    describe otra máquina es peor que no tener inventario, porque §15.7
    pide auditar «sustitución de una dependencia» y una dependencia
    sustituida dentro de la imagen no aparecería por ningún lado.
    """
    try:
        salida = subprocess.run(
            [*_argv_base(programa, imagen, None), "-c", _PROGRAMA_INVENTARIO],
            capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    if salida.returncode != 0:
        return None
    try:
        datos = json.loads(salida.stdout or "")
    except ValueError:
        return None
    return datos if isinstance(datos, dict) else None


class _SinMedir:
    """«Todavía no se ha medido», que no es lo mismo que «no se pudo».

    `None` ya significa «se preguntó dentro del contenedor y no
    contestó». Sin un tercer valor, quien pasara `None` para decir «mídelo
    tú» estaría declarando un fallo que no ha ocurrido.
    """


_SIN_MEDIR = _SinMedir()


def _dependencias_declaradas(inventario: dict[str, Any] | None) -> dict[str, Any]:
    """El bloque `dependencies` del informe, diciendo QUÉ se midió.

    Un inventario ausente no es un inventario vacío: `packages: []` se
    lee como «no hay dependencias», que es una afirmación, y falsa. Aquí
    la ausencia viaja como ausencia, con su motivo.
    """
    if inventario is None:
        return {
            "measured": False,
            "measured_in": None,
            "reason": ("no se pudo inventariar dentro del contenedor; "
                       "el inventario del anfitrión describiría otra máquina"),
            "packages": None,
            "count": None,
            "without_declared_license": None,
        }
    return {"measured": True, "measured_in": "sandbox", **inventario}


def _identidad_de_imagen(programa: str, imagen: str) -> str | None:
    """El id de CONTENIDO de la imagen, que `docker tag` no cambia.

    H4 del refutador (2026-08-20): la huella de entorno llevaba dentro la
    **etiqueta**, así que `docker tag matrixai-sandbox:local …:clon`
    fabricaba «dos entornos» con la misma imagen — y §15.6 pide reproducir
    «en al menos dos entornos de referencia aprobados», que así se
    satisfacía con un renombrado.
    """
    try:
        salida = subprocess.run(
            [programa, "image", "inspect", "--format", "{{.Id}}", imagen],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if salida.returncode != 0:
        return None
    return (salida.stdout or "").strip() or None


def _version_dentro_de(programa: str, imagen: str) -> str | None:
    """La versión de matrixai que hay DENTRO de la imagen, o `None`.

    `None` es «no se pudo preguntar», y se distingue de una versión: una
    huella de entorno que rellenara esto con un valor por defecto haría
    parecer iguales dos entornos que no lo son.
    """
    try:
        salida = subprocess.run(
            [*_argv_base(programa, imagen, None),
             "-c", "import matrixai,sys; sys.stdout.write("
                   "getattr(matrixai,'__version__','?'))"],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if salida.returncode != 0:
        return None
    return (salida.stdout or "").strip() or None


def entorno_de_reproduccion(imagen: str, backend: dict[str, Any],
                            inventario: dict[str, Any] | None | _SinMedir = _SIN_MEDIR,
                            ) -> dict[str, Any]:
    """El entorno EFECTIVO, con su digest — para poder decir «otro».

    §15.6 pide reproducir «en al menos dos entornos de referencia». Sin
    una huella del entorno no se puede afirmar que dos reproducciones
    fueran en entornos distintos, ni que una tercera repitiera el mismo.
    """
    import hashlib
    import json as _json

    sonda = str(backend.get("probe") or "docker")
    # El inventario se MIDE UNA VEZ. Quien ya lo tenga lo pasa: arrancar
    # el contenedor otra vez para volver a preguntar lo mismo no solo
    # cuesta, es que las dos respuestas podrían no coincidir y entonces
    # la huella no describiría el inventario que se publica.
    if isinstance(inventario, _SinMedir):
        inventario = inventario_dentro_de(sonda, imagen)

    # LO QUE HAY DENTRO ENTRA EN LA HUELLA. LA ETIQUETA, NO.
    #
    # H4 del refutador (2026-08-20): esto llevaba `image` dentro del
    # sha256, así que `docker tag …:local …:clon` daba
    # `same_environment: False` sobre LA MISMA IMAGEN — y §15.6, que pide
    # «al menos dos entornos de referencia aprobados», se satisfacía con
    # un renombrado. Yo ya había declarado ese defecto corregido: añadí
    # `matrixai_version` y dejé `image` donde estaba, que es media
    # limpieza, y media limpieza es peor que ninguna porque el comentario
    # que la acompañaba seguía sonando razonable.
    #
    # Ahora la huella es de CONTENIDO: el id de la imagen (que `docker
    # tag` no cambia), la versión de matrixai medida dentro, el Python de
    # dentro y el digest del inventario de dentro — una dependencia
    # sustituida cambia la huella, que es lo que pide §15.7.
    dentro = {
        "image_id": _identidad_de_imagen(sonda, imagen),
        "matrixai_version": _version_dentro_de(sonda, imagen),
        "python_version": (inventario or {}).get("python_version"),
        "dependencies_sha256": (
            hashlib.sha256(_json.dumps((inventario or {}).get("packages"),
                                       sort_keys=True).encode("utf-8")).hexdigest()
            if inventario is not None else None),
        "backend": backend.get("backend") or backend.get("name"),
        "backend_version": backend.get("version"),
        "limits": dict(LIMITES),
    }
    entorno = {
        # La etiqueta se CONSERVA —hace falta para saber qué se invocó—,
        # pero fuera del sha256: es un nombre, no el entorno.
        "image": imagen,
        "image_is_not_part_of_the_fingerprint": True,
        **dentro,
    }
    entorno["environment_sha256"] = hashlib.sha256(
        _json.dumps(dentro, sort_keys=True).encode("utf-8")).hexdigest()
    # Y SI NO SE PUDO MIRAR DENTRO, SE DICE. Una huella construida sobre
    # cuatro `null` es idéntica para dos entornos cualesquiera: presentarla
    # sin avisar afirmaría que son el mismo.
    entorno["measured_inside"] = inventario is not None and dentro["image_id"] is not None
    return entorno


def recibo_de_reproduccion(informe: dict[str, Any], paquete: str,
                           receipt_id: str) -> dict[str, Any]:
    """El recibo de lo que se reprodujo (§15.6).

    **Se emite pase lo que pase**, incluso cuando no se pudo comprobar:
    un recibo que solo existe cuando todo va bien no sirve para archivar
    lo que pasó, que es justo para lo que existe. Lo que cambia es lo que
    DICE: `outcome` distingue las tres cosas que el CLI ya distingue —
    cuadró, alguien lo tocó, no se pudo comprobar.
    """
    import hashlib

    resultado = informe.get("result") or {}
    codigo = resultado.get("returncode")
    if not informe.get("checked"):
        resultado_final = "not_checked"
    elif codigo == 0:
        resultado_final = "reproduced"
    elif codigo == 3:
        # `3` es «no se pudo comprobar del todo», NO «alguien lo tocó».
        # Colapsarlo en `mismatch` —que es lo que hacía este recibo— acusa
        # a un paquete honesto de una discrepancia que nadie ha medido.
        # Lo vi comparando dos entornos: los dos con las MISMAS etapas y
        # uno decía `reproduced` y el otro `mismatch`.
        resultado_final = "not_fully_checked"
    else:
        resultado_final = "mismatch"

    # La huella del paquete: sin ella el recibo no dice DE QUÉ habla.
    huellas: list[str] = []
    ilegibles: list[str] = []
    raiz = Path(paquete)
    for fichero in sorted(raiz.rglob("*")):
        if fichero.is_symlink():
            # H7 del refutador (2026-08-20): esto era un `continue` seco,
            # ANTES del bloque que cuenta lo ilegible, así que un enlace
            # colado en el paquete no entraba en `unreadable` y
            # `covers_whole_package` seguía diciendo `true` sobre un
            # paquete que el digest no cubría — con el comentario de
            # abajo afirmando lo contrario tres líneas más allá.
            #
            # No se sigue el enlace (apuntaría fuera del paquete, y
            # `/etc/passwd` no es contenido del paquete), pero SÍ se
            # cuenta: es una entrada del directorio que el digest no
            # cubre, y eso es exactamente lo que `unreadable` significa.
            ilegibles.append(
                f"{fichero.relative_to(raiz)}: enlace simbólico, no se sigue "
                f"(apunta a {os.readlink(fichero)!r})")
            continue
        if not fichero.is_file():
            continue
        try:
            huellas.append(
                f"{fichero.relative_to(raiz)}:"
                f"{hashlib.sha256(fichero.read_bytes()).hexdigest()}")
        except OSError as exc:
            # Un fichero que no se puede leer NO se traga: el digest
            # dejaría de cubrir el paquete entero y el recibo estaría
            # afirmando sobre menos de lo que dice. Se cuenta y se dice.
            ilegibles.append(f"{fichero.relative_to(raiz)}: {exc.strerror}")
    return {
        "schema_version": "1.0",
        "receipt_id": receipt_id,
        "event_type": "replay",
        # Un recibo sin fecha no se puede ordenar ni caducar (§14.2). Lo
        # pedía el esquema y este recibo no lo llevaba: salió al exigir
        # las secciones de §14.2 en los dos extremos (H2).
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subject": {"purpose": "package reproduction (81-C5)"},
        "package": {
            "path": str(raiz),
            "content_digest": "sha256:" + hashlib.sha256(
                "\n".join(huellas).encode("utf-8")).hexdigest(),
            "files": len(huellas),
            # Si esto no está vacío, el digest de arriba NO cubre el
            # paquete entero, y quien lo lea tiene que saberlo.
            "unreadable": ilegibles,
            "covers_whole_package": not ilegibles,
        },
        "environment": informe.get("environment"),
        "steps": [{"id": "replay", "status": resultado_final,
                   "returncode": resultado.get("returncode"),
                   "retrained": informe.get("retrained")}],
        "output": {"outcome": resultado_final,
                   "stages": _etapas_del_informe(resultado)},
        "dependencies": informe.get("dependencies"),
        "evidence": {"command": resultado.get("command")},
    }


def _etapas_del_informe(resultado: dict[str, Any]) -> dict[str, Any] | None:
    """Las cuatro etapas del `verify` de dentro, si las hubo."""
    import json as _json

    salida = resultado.get("stdout")
    if not isinstance(salida, str) or not salida.strip():
        return None
    try:
        datos = _json.loads(salida)
    except ValueError:
        return None
    etapas = datos.get("stages") if isinstance(datos, dict) else None
    if not isinstance(etapas, dict):
        return None
    return {k: v.get("status") for k, v in etapas.items() if isinstance(v, dict)}


def comparar_con_referencia(nuevo: dict[str, Any], referencia: dict[str, Any]) -> dict[str, Any]:
    """Compara un recibo de reproducción con el de referencia (§15.6).

    Lo que se compara y por qué:

    * **el paquete**: si el `content_digest` no coincide, no se están
      reproduciendo la misma cosa y lo demás da igual — se dice y se para;
    * **el resultado**: `reproduced` contra `mismatch` es la diferencia
      que importa;
    * **las etapas**, una a una: «algo cambió» sin decir cuál obliga a
      abrir los dos recibos al lado;
    * **el entorno**: si son el mismo, dos reproducciones iguales prueban
      menos de lo que parece. **Y eso se DICE**, porque es justo la media
      verdad que §15.6 pide evitar al exigir dos entornos.
    """
    def _campo(recibo: Any, *ruta: str) -> Any:
        actual: Any = recibo
        for paso in ruta:
            if not isinstance(actual, dict):
                return None
            actual = actual.get(paso)
        return actual

    paquete_nuevo = _campo(nuevo, "package", "content_digest")
    paquete_ref = _campo(referencia, "package", "content_digest")
    if paquete_nuevo != paquete_ref:
        return {
            "comparable": False,
            "reason": ("los dos recibos describen paquetes distintos "
                       f"({str(paquete_ref)[:20]}… frente a {str(paquete_nuevo)[:20]}…): "
                       "no se está reproduciendo la misma cosa"),
        }

    etapas_n = _campo(nuevo, "output", "stages") or {}
    etapas_r = _campo(referencia, "output", "stages") or {}
    difieren = sorted(k for k in set(etapas_n) | set(etapas_r)
                      if etapas_n.get(k) != etapas_r.get(k))
    entorno_n = _campo(nuevo, "environment", "environment_sha256")
    entorno_r = _campo(referencia, "environment", "environment_sha256")
    mismo_entorno = entorno_n == entorno_r

    return {
        "comparable": True,
        "same_package": True,
        "same_outcome": _campo(nuevo, "output", "outcome") == _campo(referencia, "output", "outcome"),
        "outcome": {"reference": _campo(referencia, "output", "outcome"),
                    "new": _campo(nuevo, "output", "outcome")},
        # Las etapas que difieren, NOMBRADAS.
        "stages_differing": {k: {"reference": etapas_r.get(k), "new": etapas_n.get(k)}
                             for k in difieren},
        "same_environment": mismo_entorno,
        # EN QUÉ se diferencian, no solo que se diferencian: «otro
        # entorno» a secas manda a comparar los dos recibos a mano.
        # EN QUÉ se diferencian, no solo que se diferencian: «otro
        # entorno» a secas manda a comparar los dos recibos a mano. La
        # ETIQUETA de la imagen no está en la lista a propósito (H4):
        # cambiarla no cambia el entorno, y nombrarla como diferencia era
        # lo que hacía pasar `docker tag` por un segundo entorno.
        "environment_differences": sorted(
            campo for campo in ("image_id", "backend", "backend_version",
                                "matrixai_version", "python_version",
                                "dependencies_sha256")
            if _campo(nuevo, "environment", campo)
            != _campo(referencia, "environment", campo)),
        # El aviso que hace honesta la comparación.
        "note": _nota_de_la_comparacion(
            mismo_entorno,
            _campo(nuevo, "output", "outcome"),
            _campo(referencia, "output", "outcome"),
            bool(_campo(nuevo, "environment", "measured_inside"))
            and bool(_campo(referencia, "environment", "measured_inside"))),
    }


#: Los desenlaces que NO son una reproducción medida. Coincidir en uno de
#: ellos no sostiene nada: son dos veces «no se pudo comprobar».
_SIN_MEDICION = frozenset({"not_fully_checked", "not_checked", "error"})


def _nota_de_la_comparacion(mismo_entorno: bool, desenlace_nuevo: Any,
                            desenlace_ref: Any, medido_dentro: bool) -> str:
    """El aviso que hace honesta la comparación — y que MIRA EL RESULTADO.

    H4 del refutador, segunda mitad (2026-08-20): esta nota dependía solo
    de `same_environment`, así que con los dos desenlaces en
    `not_fully_checked` seguía imprimiendo «coincidir aquí **sí sostiene
    reproducibilidad**». Dos «no se pudo comprobar» que coinciden no
    sostienen nada, y decir que sí es la media verdad tranquilizadora que
    §15.6 existe para no dar.
    """
    if not medido_dentro:
        return ("no se pudo mirar DENTRO de alguno de los dos entornos, así "
                "que la huella no los distingue: esta comparación no sostiene "
                "ni repetibilidad ni reproducibilidad")
    if desenlace_nuevo in _SIN_MEDICION or desenlace_ref in _SIN_MEDICION:
        return (f"al menos una de las dos reproducciones no llegó a comprobarse "
                f"({desenlace_ref} / {desenlace_nuevo}): coincidir aquí es "
                "coincidir en no haber medido, y no sostiene reproducibilidad")
    if mismo_entorno:
        return ("las dos reproducciones fueron en el MISMO entorno, así que "
                "coincidir prueba repetibilidad, no reproducibilidad: §15.6 pide "
                "al menos dos entornos de referencia")
    return ("reproducciones en entornos DISTINTOS y las dos comprobadas: "
            "coincidir aquí sí sostiene reproducibilidad")


def replay_and_verify(paquete: str, *, reentrenar: bool = True,
                      receipt_id: str | None = None) -> dict[str, Any]:
    """Reproduce y verifica un paquete DENTRO de un entorno aislado.

    Si no hay con qué aislar, **no se ejecuta**: se levanta
    `SinAislamiento` diciendo qué falta. No hay parámetro para saltárselo
    y no lo va a haber.
    """
    aislamiento = describir_aislamiento()
    if not aislamiento["available"]:
        opciones = ", ".join(
            f"{b['name']} >= {b['min_version']}" for b in BACKENDS_SOPORTADOS)
        raise SinAislamiento(
            "no se puede reproducir sin aislamiento: no hay ningún backend "
            f"disponible. Instala uno de estos y vuelve a intentarlo: {opciones}. "
            "No se ejecuta igualmente a propósito — prometer un sandbox y "
            "correr sin él es peor que no ofrecerlo.")

    elegido = aislamiento["chosen"]
    # El backend elegido viaja con su `probe`, que es el programa que se
    # invoca: sin él, `_ejecutar_en` tendría que volver a deducirlo del
    # nombre, y deducir dos veces lo mismo es cómo acaban discrepando.
    sonda = next((b["probe"] for b in BACKENDS_SOPORTADOS
                  if b["name"] == elegido["backend"]), elegido["backend"])
    resultado = _ejecutar_en({**elegido, "probe": sonda}, paquete,
                             reentrenar=reentrenar)
    # UNA sola pasada al contenedor para el inventario: la usan la huella
    # del entorno y el bloque `dependencies` del informe, y así los dos
    # hablan exactamente del mismo entorno.
    inventario_medido = inventario_dentro_de(sonda, imagen_del_sandbox())
    informe = {
        "ok": bool(resultado.get("ok")),
        # «No se pudo comprobar» ARRIBA, no enterrado en el resultado: un
        # `ok: false` a secas se lee como «el paquete está mal».
        "checked": resultado.get("returncode") is not None,
        "package": paquete,
        # El aislamiento se DECLARA en el informe: quien lea esto tiene
        # que poder auditar bajo qué condiciones se reprodujo.
        "isolation": {**elegido, "limits": dict(LIMITES),
                      "image": imagen_del_sandbox()},
        # QUÉ se hizo, no solo que se hizo: sin reentrenar, `training` y
        # `R3` quedan sin ejecutar y el informe no puede pasar por una
        # reproducción completa.
        "retrained": bool(reentrenar),
        # EL ENTORNO con su huella (§15.6): sin ella no se puede afirmar
        # que dos reproducciones fueran en entornos distintos.
        "environment": entorno_de_reproduccion(
            imagen_del_sandbox(), {**elegido, "probe": sonda}, inventario_medido),
        # Y CON QUÉ se reprodujo: «reproducible» sin decir con qué
        # dependencias es media promesa. Se mide DENTRO del contenedor
        # (H3): el inventario del anfitrión describe otra máquina, y
        # ponerlo aquí era afirmar sobre un entorno que nadie había
        # mirado. `None` = no se pudo, y se declara como tal en vez de
        # rellenarlo con el de fuera.
        "dependencies": _dependencias_declaradas(inventario_medido),
        "result": resultado,
    }
    # El RECIBO, siempre: uno que solo existe cuando todo va bien no sirve
    # para archivar lo que pasó, que es para lo que existe.
    informe["receipt"] = recibo_de_reproduccion(
        informe, paquete, receipt_id or f"replay-{informe['environment']['environment_sha256'][:12]}")
    return informe
