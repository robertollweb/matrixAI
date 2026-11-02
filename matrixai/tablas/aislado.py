# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""calamine en un PROCESO HIJO con techo de memoria y de tiempo (150-C1, auditoría B2).

Por qué: un fichero de 1–6 KB hace que calamine pida 181 GB o 549 GB de golpe; Rust no lanza una
excepción cuando una reserva falla, ABORTA el proceso (`memory allocation of … bytes failed`,
SIGABRT), y eso no se puede capturar desde Python. Medido el 2026-11-02 con los casos del auditor
(`xls_aborta.xls`, y un `.xlsx` de 2 KB cuyo `sharedStrings` declara `uniqueCount="1000000000"`).
En el proceso del Studio eso tumbaría el servidor entero.

Así que la lectura con calamine se hace aquí, en un hijo:

* con `RLIMIT_AS` (memoria virtual) a la medida de lo que la hoja necesita (`topes.techo_del_lector`),
  `RLIMIT_CORE = 0` (un aborto no deja un volcado de cientos de MB) y `RLIMIT_CPU` de respaldo;
* con un tope de tiempo de pared: si se pasa, se le MATA y se le espera (sin huérfanos);
* el hijo es `python -I -c <fuente>` (sin variables de entorno ni `site` del usuario), con las rutas
  EXACTAS de calamine que vería el padre, y NO importa `matrixai`: solo la stdlib y calamine;
* devuelve los valores con `pickle`, y el padre los lee con un `Unpickler` que solo admite las
  clases de `datetime` (un hijo comprometido no puede pedir que el padre ejecute nada).

El padre traduce cada final a una `TablaRechazada` con su código: `lector_sin_memoria`,
`lector_tiempo`, `lector_abortado`, `protegido`, `lector_fallo`.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time

from .errores import TablaRechazada, rechazar

__all__ = ["leer_hoja", "empezar_a_leer", "Lectura", "listar_hojas", "calamine_disponible",
           "ULTIMO_HIJO"]

#: Lo último que midió un hijo (para las mediciones y las pruebas): segundos, VmHWM, código.
ULTIMO_HIJO: dict = {}

# La fuente del hijo. Solo stdlib + calamine; NO importa matrixai (arranca en ~50 ms).
_FUENTE_DEL_HIJO = r'''
import sys, json, pickle
args = json.loads(sys.argv[1])
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args["memoria"]:
        resource.setrlimit(resource.RLIMIT_AS, (args["memoria"], args["memoria"]))
    if args["cpu"]:
        resource.setrlimit(resource.RLIMIT_CPU, (args["cpu"], args["cpu"] + 5))
except Exception:
    pass
sys.path[:0] = args["rutas"]
salida = sys.stdout.buffer
def hwm():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmHWM:"):
                return int(l.split()[1]) * 1024
    except Exception:
        return None
def responder(obj):
    obj["hwm"] = hwm()
    pickle.dump(obj, salida, protocol=4)
    salida.flush()
try:
    datos = sys.stdin.buffer.read()
    import io
    import python_calamine as pc
    try:
        wb = pc.load_workbook(io.BytesIO(datos))
        if args["op"] == "hojas":
            responder({"ok": True, "hojas": [(m.name, str(m.visible).rsplit(".", 1)[-1])
                                              for m in wb.sheets_metadata]})
        else:
            h = wb.get_sheet_by_name(args["hoja"])
            inicio = h.start
            if inicio is None:
                responder({"ok": True, "inicio": None, "filas": [], "combinadas": []})
            else:
                filas = h.to_python(skip_empty_area=True)
                ancho = max((len(f) for f in filas), default=0)
                # Sin las celdas vacías del final de cada fila: una celda suelta en Z1048576 son
                # un millón de filas de 26 vacías que no hace falta copiar (el ancho va aparte).
                for i, f in enumerate(filas):
                    if f and f[-1] == "":
                        if f.count("") == len(f):
                            filas[i] = []
                        else:
                            while f[-1] == "":
                                f.pop()
                comb = [(tuple(a), tuple(b)) for a, b in (h.merged_cell_ranges or [])]
                responder({"ok": True, "inicio": tuple(inicio), "filas": filas, "ancho": ancho,
                           "combinadas": comb})
    except pc.PasswordError:
        responder({"ok": False, "codigo": "protegido"})
except MemoryError:
    try:
        responder({"ok": False, "codigo": "lector_sin_memoria"})
    except BaseException:
        sys.exit(70)
except BaseException as e:
    responder({"ok": False, "codigo": "lector_fallo", "tipo": type(e).__name__,
               "mensaje": str(e)[:300]})
'''

_PERMITIDAS = {("datetime", "datetime"), ("datetime", "date"), ("datetime", "time"),
               ("datetime", "timedelta")}


class _UnpicklerDeValores(pickle.Unpickler):
    """Solo listas, tuplas, textos, números, booleanos, `None` y fechas: nada que ejecutar."""

    def find_class(self, modulo, nombre):
        if (modulo, nombre) in _PERMITIDAS:
            import datetime
            return getattr(datetime, nombre)
        raise pickle.UnpicklingError(f"clase no permitida: {modulo}.{nombre}")


def _rutas_de_calamine() -> list[str] | None:
    """Los directorios desde los que el padre importaría `python_calamine` (sin importarlo)."""
    try:
        spec = importlib.util.find_spec("python_calamine")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    paquete = os.path.dirname(os.path.abspath(spec.origin))
    rutas = [os.path.dirname(paquete)]
    rutas += [os.path.abspath(p) if p else os.getcwd() for p in sys.path]
    return rutas


def calamine_disponible() -> bool:
    return _rutas_de_calamine() is not None


def _nombre_de_senal(n: int) -> str:
    try:
        return signal.Signals(n).name
    except (ValueError, AttributeError):
        return f"señal {n}"


class Lectura:
    """Una lectura en curso en su proceso hijo. `esperar()` da el resultado (o lanza la
    `TablaRechazada` que toque); `cancelar()` mata al hijo y lo espera (sin huérfanos). Mientras
    tanto el padre puede hacer otra cosa (la geometría): la comunicación va en un hilo."""

    def __init__(self, datos: bytes, args: dict, *, memoria: int | None, segundos: float):
        rutas = _rutas_de_calamine()
        if rutas is None:
            raise rechazar("falta_lector")
        if not sys.executable:
            raise rechazar("lector_fallo", tipo="no se encuentra el intérprete de Python")
        self.memoria, self.segundos = memoria, segundos
        args = dict(args, rutas=rutas, memoria=int(memoria) if memoria else 0,
                    cpu=int(segundos) + 5 if segundos else 0)
        self.t0 = time.monotonic()
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-I", "-c", _FUENTE_DEL_HIJO, json.dumps(args)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                close_fds=True)
        except OSError as e:
            raise rechazar("lector_fallo",
                           tipo=f"no se pudo arrancar el lector ({type(e).__name__})") from None
        self._salida = None
        self._fallo = None
        self._hilo = threading.Thread(target=self._comunicar, args=(datos,), daemon=True)
        self._hilo.start()

    def _comunicar(self, datos: bytes) -> None:
        try:
            self._salida = self.proc.communicate(datos, timeout=self.segundos)
        except subprocess.TimeoutExpired:
            self._matar()
            self._fallo = "lector_tiempo"
        except BaseException as e:      # el hilo nunca deja al hijo vivo
            self._matar()
            self._fallo = e

    def _matar(self) -> None:
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.communicate(timeout=30)
        except Exception:
            pass

    def cancelar(self) -> None:
        self._matar()
        self._hilo.join(timeout=60)

    def esperar(self) -> dict:
        try:
            self._hilo.join()
        except BaseException:
            self.cancelar()
            raise
        if self._fallo == "lector_tiempo":
            ULTIMO_HIJO.clear()
            ULTIMO_HIJO.update(segundos=time.monotonic() - self.t0, codigo="lector_tiempo", hwm=None)
            raise rechazar("lector_tiempo", segundos=_segundos_legibles(self.segundos))
        if isinstance(self._fallo, BaseException):
            raise rechazar("lector_fallo", tipo=type(self._fallo).__name__)
        out, err = self._salida
        return _interpretar(out, err, self.proc.returncode, time.monotonic() - self.t0,
                            self.memoria, self.segundos)


def _ejecutar(datos: bytes, args: dict, *, memoria: int | None, segundos: float) -> dict:
    return Lectura(datos, args, memoria=memoria, segundos=segundos).esperar()


def _interpretar(out: bytes, err: bytes, rc: int | None, dt: float, memoria: int | None,
                 segundos: float) -> dict:
    resultado = None
    if out:
        try:
            resultado = _UnpicklerDeValores(io.BytesIO(out)).load()
        except Exception:
            resultado = None
    cola = (err or b"")[-4000:].decode("utf-8", "replace")
    ULTIMO_HIJO.clear()
    ULTIMO_HIJO.update(segundos=dt, rc=rc, hwm=(resultado or {}).get("hwm") if isinstance(resultado, dict) else None)
    if isinstance(resultado, dict) and rc == 0:
        if resultado.get("ok"):
            ULTIMO_HIJO["codigo"] = "ok"
            return resultado
        codigo = resultado.get("codigo")
        ULTIMO_HIJO["codigo"] = codigo
        if codigo == "protegido":
            raise rechazar("protegido")
        mensaje = str(resultado.get("mensaje") or "").lower()
        hwm = resultado.get("hwm") or 0
        if codigo == "lector_sin_memoria" or (
                resultado.get("tipo") == "PanicException"
                and (any(m in mensaje for m in ("alloc", "capacity", "memory"))
                     or "MemoryError" in cola or "memory allocation of" in cola
                     or (memoria and hwm >= 0.9 * memoria))):
            # Rust no siempre aborta al quedarse sin memoria: a veces entra en pánico y pyo3 lo da
            # como `PanicException` («capacity overflow», o «PyObject pointer is null» tras un
            # MemoryError de Python; medido con `celda_perdida.xlsx` y un techo de 1 GiB). Es lo
            # mismo: no cabe.
            ULTIMO_HIJO["codigo"] = "lector_sin_memoria"
            raise rechazar("lector_sin_memoria", techo=_techo_legible(memoria))
        raise rechazar("lector_fallo", tipo=str(resultado.get("tipo") or "error"))
    # El hijo no terminó por su pie: abortó (Rust), lo mató una señal, o se quedó sin memoria.
    sin_memoria = ("memory allocation of" in cola or "MemoryError" in cola
                   or "out of memory" in cola.lower() or rc == 70)
    if rc is not None and rc < 0 and -rc == getattr(signal, "SIGXCPU", -1):
        ULTIMO_HIJO["codigo"] = "lector_tiempo"
        raise rechazar("lector_tiempo", segundos=_segundos_legibles(segundos))
    if sin_memoria:
        ULTIMO_HIJO["codigo"] = "lector_sin_memoria"
        raise rechazar("lector_sin_memoria", techo=_techo_legible(memoria))
    ULTIMO_HIJO["codigo"] = "lector_abortado"
    como = _nombre_de_senal(-rc) if rc is not None and rc < 0 else f"código {rc}"
    if rc is not None and rc < 0 and -rc in _SENALES_DE_FUERA:
        # SIGKILL/SIGTERM/SIGINT/SIGHUP no las da un fichero dañado: las manda otro proceso o el
        # sistema (el OOM del kernel, un `kill`, el apagado). Decir «fichero dañado» era falso.
        raise rechazar("lector_abortado", variante="desde_fuera", como=como)
    raise rechazar("lector_abortado", como=como)


_SENALES_DE_FUERA = {getattr(signal, n) for n in ("SIGKILL", "SIGTERM", "SIGINT", "SIGHUP")
                     if hasattr(signal, n)}


def _segundos_legibles(s) -> str:
    """`120` → «120»; `0.5` → «0,5» (con `int()` un tope por debajo de 1 s decía «más de 0 s»)."""
    s = float(s or 0)
    if s >= 1 and s.is_integer():
        return str(int(s))
    return f"{s:.3g}".replace(".", ",")


def _techo_legible(memoria: int | None) -> str:
    from .. import limits
    return limits.human_bytes(memoria) if memoria else "sin techo"


def listar_hojas(datos: bytes, *, memoria: int | None, segundos: float) -> list[tuple[str, str]]:
    """`[(nombre, "Visible"|"Hidden"|"VeryHidden")]` de un libro (para el `.xls`)."""
    return _ejecutar(datos, {"op": "hojas"}, memoria=memoria, segundos=segundos)["hojas"]


def empezar_a_leer(datos: bytes, hoja: str, *, memoria: int | None, segundos: float) -> Lectura:
    """Como `leer_hoja`, pero sin esperar: devuelve la `Lectura` en curso."""
    return Lectura(datos, {"op": "leer", "hoja": hoja}, memoria=memoria, segundos=segundos)


def leer_hoja(datos: bytes, hoja: str, *, memoria: int | None, segundos: float) -> dict:
    """`{inicio: (fila0, col0) | None, filas: [[valor…]], combinadas: [((f0, c0), (f1, c1))]}`
    (todo 0-based, como calamine)."""
    return _ejecutar(datos, {"op": "leer", "hoja": hoja}, memoria=memoria, segundos=segundos)


__all__ += ["TablaRechazada"]
