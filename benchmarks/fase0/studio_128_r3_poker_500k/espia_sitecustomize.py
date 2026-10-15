"""Espía de SOLO LECTURA de la medida (2) del 128-R3 (cola nocturna). Va en PYTHONPATH: se carga en CADA proceso, también en
los hijos `spawn` (un intento de la búsqueda, el reajuste final). Activo solo con ESPIA_R3_COLA_LOG. No cambia nada de lo que
se ejecuta: llama a lo original con los mismos argumentos y devuelve lo mismo.

  * `Motor.fit`: el presupuesto que recibe (techo de pared), las filas, cómo terminó y su `entrenamiento_efectivo`, con el
    nombre del proceso (`reajuste-final` es el hijo del reajuste) y el `__file__` de los tres paquetes.
  * `estudio_job.producir_perfil_clinico` (lo llama el reajuste con la muestra de TEST del campeón y la de la REFERENCIA, las
    mismas filas): la métrica de selección de las dos en test, que el sobre no guarda. Es la cifra del criterio (d)."""
import json, os, sys, time

try:
    import apport_python_hook  # noqa: F401
    apport_python_hook.install()
except Exception:
    pass

_LOG = os.environ.get("ESPIA_R3_COLA_LOG")


def _escribir(registro):
    linea = (json.dumps(registro, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    fd = os.open(_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, linea)
    finally:
        os.close(fd)


def _proceso():
    import multiprocessing
    return {"pid": os.getpid(), "proceso": multiprocessing.current_process().name,
            "modulos": {m: getattr(sys.modules.get(m), "__file__", None)
                        for m in ("matrixai", "matrixai_engines", "matrixai_studio")}}


def _parchear_motor(modulo):
    Motor = getattr(modulo, "Motor", None)
    if Motor is None or getattr(Motor.fit, "_espia_r3_cola", False):
        return
    original = Motor.fit

    def fit(self, train, validation, spec, budget, *args, **kwargs):
        t0 = time.time()
        base = {**_proceso(), "que": "fit", "candidato": kwargs.get("candidate"),
                "clase": type(self).__module__ + "." + type(self).__name__, "wall_seconds": budget.wall_seconds,
                "train_n": getattr(train, "n", None), "validation_n": getattr(validation, "n", None), "t0": t0}
        fit_result, fitted = original(self, train, validation, spec, budget, *args, **kwargs)
        try:
            pred = fitted.spec.predictor if fitted is not None else None
            efectivo = pred.get("entrenamiento_efectivo") if isinstance(pred, dict) else None
        except Exception as exc:  # noqa: BLE001
            efectivo = repr(exc)
        _escribir({**base, "estado": getattr(fit_result, "state", None), "segundos": time.time() - t0,
                   "entrenamiento_efectivo": efectivo})
        return fit_result, fitted

    fit._espia_r3_cola = True
    Motor.fit = fit


def _parchear_estudio_job(modulo):
    original = getattr(modulo, "producir_perfil_clinico", None)
    if original is None or getattr(original, "_espia_r3_cola", False):
        return

    def producir_perfil_clinico(*args, **kwargs):
        try:
            from matrixai.estudio.metricas import evaluar
            metrica = kwargs.get("metrica_de_comparacion")
            cifras = {}
            for nombre in ("muestra_test", "muestra_test_baseline"):
                muestra = kwargs.get(nombre)
                if muestra is not None and metrica:
                    informe = evaluar(muestra, [metrica])
                    cifras[nombre] = next((m.value for m in informe.metrics if m.metric_id == metrica), None)
                else:
                    cifras[nombre] = None
            _escribir({**_proceso(), "que": "test", "metrica": metrica, **cifras,
                       "baseline_comparado": kwargs.get("baseline_comparado"),
                       "sin_comparacion_con_el_baseline": kwargs.get("sin_comparacion_con_el_baseline")})
        except Exception as exc:  # noqa: BLE001 -- el espía nunca tumba nada
            _escribir({**_proceso(), "que": "test", "error": repr(exc)})
        return original(*args, **kwargs)

    producir_perfil_clinico._espia_r3_cola = True
    modulo.producir_perfil_clinico = producir_perfil_clinico


_PARCHES = {"matrixai_engines.motor": _parchear_motor, "matrixai_studio.estudio_job": _parchear_estudio_job}

if _LOG:
    import importlib.abc

    class _Espia(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _PARCHES:
                return None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(fullname, path, target)
                if spec is not None:
                    cargar = spec.loader.exec_module

                    def exec_module(modulo, _cargar=cargar, _parche=_PARCHES[fullname]):
                        _cargar(modulo)
                        _parche(modulo)
                    spec.loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Espia())
