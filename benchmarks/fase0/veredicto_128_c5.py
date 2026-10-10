#!/usr/bin/env python3
"""128 · C5 — el VEREDICTO de «la GPU en el paquete GPU»: el brazo B (TabM con `MATRIXAI_TABM_DISPOSITIVO=auto` y
`--gpus all`) contra el brazo A (`cpu`), medidos en el MISMO PC de Roberto. Un solo sitio: lo usa el medidor del kit
(`kit_128_c5/medir_c5.py`) para cortar, y quien lea el resultado.

    python3 veredicto_128_c5.py resultado_128_c5.json [--suelo X]
    python3 veredicto_128_c5.py resultado_128_c5.json --evaluacion-nueva evaluacion_c5p.json   # R9.8: fichero NUEVO

LA REGLA (C5′, la de Roberto del 09-10 13:40, «banda de 1 punto + tiempo»; contrato 128, «REGLA NUEVA DE LA GPU»).
Sustituye a la E2 («ninguno baja más que el suelo de ruido y alguno mejora»), que con un suelo de 0 contaba como baja
hasta −0,08 puntos. Por conjunto, d = cifra del test de B (GPU) − la de A (CPU), en puntos; suelo = el MAYOR de los dos
brazos; `m1 = max(1, suelo)` y `m2 = max(2, suelo)`:

- **igual**: |d| < m1 · **sube**: d ≥ m1 · **baja tolerada**: −m2 < d ≤ −m1 (no bloquea por sí sola; cuenta en contra)
  · **empeora**: d ≤ −m2, o deja de terminar (bloquea siempre).
- **mejora de tiempo**: el estudio de B tarda la MITAD o menos que el de A (pared) y el conjunto queda «igual».
- **ENTRA si** ninguno empeora y (suben + mejoras de tiempo) SUPERAN a las bajas toleradas (la regla 4 del 120, con el
  tiempo como mejora). Si no, el dispositivo se queda en `cpu` y el código no se retira (la señal lo permite).
- **El corte** (regla 9): en el primer conjunto que empeora o deja de terminar, o cuando las bajas toleradas ya no se
  pueden compensar ni con TODOS los que faltan.
- Una caída a CPU DECLARADA no cuenta como GPU: ni su subida ni su mejora de tiempo.

- **La cifra de un conjunto** es la del campeón en el TEST (`seleccion.evaluacion_final`), en puntos ×100: la misma que
  C3′ (`veredicto_120_c3p.cifra_de_test`, importada, no copiada). Los dos brazos tienen los mismos candidatos, así que
  no hay «el máximo sobre más candidatos» que suba por construcción; y es lo que recibe quien usa el modelo.
- **El suelo de ruido** es el de CADA brazo: los 4 conjuntos de control se miden dos veces en cada uno, y la mayor
  |diferencia| de la cifra entre las dos (0 si son idénticas) es su suelo. Se usa el MAYOR de los dos: la cifra de
  TabM depende del reloj (plazo de pared), así que el suelo de la CPU tampoco es cero, y en la GPU sumará el orden
  de las sumas de cuBLAS. Un campeón que cambia al repetir NO es un error (es ruido, y se declara); uno que completa
  en una repetición y no en la otra SÍ: el instrumento no es repetible, `Incomparable`.
- «Deja de completar» y «empieza a completar» cuentan aparte (la primera siempre impide entrar; la segunda cuenta como
  mejora).
- **Presencia**: el brazo B tiene que haber entrenado TabM en la GPU en algún conjunto (el dispositivo lo declara el
  motor) y el A en ninguno; si no, no se midió lo registrado y NO entra, diga lo que diga la tabla. Y (128, I1 de su
  auditoría) donde TabM es el CAMPEÓN del brazo B, su modelo ENTREGADO —el del reajuste final, el que recibe quien usa
  el Studio— tiene que haberse entrenado en la GPU o declarar su caída a la CPU: una búsqueda en GPU con un entregado
  de CPU callado (el B1) no es «la GPU».
- **Una caída a la CPU DECLARADA en el brazo B no es la GPU** (128, N3 de su 2.ª auditoría): si el modelo ENTREGADO de un
  conjunto declara `cayo_a_cpu`, ese conjunto se entrenó en la CPU: su mejora NO cuenta como mejora de la GPU (se dice
  «cayó a CPU: no cuenta como GPU»), y todas esas caídas se LISTAN en `veredicto.caidas_a_cpu_en_b`.
- **Presencia por DECLARACIÓN, no por VRAM** (128-C5′, H18 de la revisión del 10-10): que TabM entrenó en la GPU lo
  prueba el dispositivo que declaran SUS intentos (`dispositivos_de_tabm`, solo los del motor TabM) o su modelo
  ENTREGADO; la VRAM de más que se mide durante el estudio la puede haber puesto otro componente u otro proceso:
  se ENSEÑA (`vram_auxiliar_en_b`) como evidencia auxiliar y no basta.
- **El instrumento y el producto, por separado** (R9.7): `instrumento_midio_bien` (los dos brazos son comparables,
  midieron lo registrado, hay referencia A para cada conjunto y la pasada está completa o la cortó la regla),
  `la_regla_se_cumple` (la banda de 1 punto + tiempo) y `producto_entra` (las dos). `entra` es el mismo valor que
  `producto_entra`, conservado por los lectores de antes. Un fallo del instrumento NO es «la GPU no entra».
- **Qué se compara con qué** (R9.3/R9.4, H16): los conjuntos son los del BANCO por IDENTIDAD (`BANCO`), no por recuento;
  y los dos brazos tienen que ser compatibles en imagen, código del medidor y del veredicto (sha256), manifiesto y
  CSV de los conjuntos, memoria y CPUs (`incompatibilidades_de_huellas`): si no, `Incomparable` con el motivo. Una
  referencia A ausente es `sin_referencia` («no medido»), no «A no completó».
- **Un número no finito no es una cifra** (H15): una cifra de test NaN/±inf, un suelo negativo o no finito, o una
  pared negativa son `Incomparable`; una pared no finita, cero o ausente no cuenta como mejora de tiempo.
- **RAM y tiempo, sobre los conjuntos COMUNES** (H17): `resumen_emparejado`; los que solo tiene un brazo, aparte.
- **Una evaluación NUEVA de un resultado ya medido** (R9.8): `evaluacion_nueva` evalúa con esta regla un
  `resultado_128_c5.json` anterior sin tocarlo, y cita su sha256 y su veredicto original.
- **Lo que la comparación INCLUYE y no está medido** (128, N4): desde la reparación del B1 la densa ve CUDA en el
  paquete GPU, y eso cuesta RAM de host y tiempo. En el kit lo paga el brazo B y no el A: una diferencia de memoria,
  un OOM o más pared del B puede venir de ahí y no de TabM. No se pudo medir sin una GPU (`COSTE_DE_VER_CUDA`); el JSON
  recoge la RAM pico de cada brazo (`veredicto.ram_pico_mb_por_brazo`) para verlo.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

EXACTO = 1e-9
#: C5′: la banda de «igual»/«sube» (puntos) y la bajada que ya es «empeora»; el suelo de ruido las SUBE si es mayor.
BANDA_MINIMA = 1.0
EMPEORA_MINIMA = 2.0
#: C5′: B tarda esta fracción de la pared de A o menos → mejora de tiempo (si el resultado queda «igual»).
FRACCION_DE_TIEMPO = 0.5
AQUI = Path(__file__).resolve().parent
#: 128-C5′ (H16): los conjuntos del banco POR NOMBRE, en el orden del manifiesto del kit
#: (`kit_128_c5/datos/manifiesto.json`; una prueba lo compara). Completar no es contar 13: es que estén ESTOS.
BANCO = ("dresses-sales", "climate-model-simulation-crashes", "us_crime", "pc1", "micro-mass", "mfeat-factors",
         "wilt", "pendigits", "Internet-Advertisements", "house_16H", "KDDCup09_appetency", "APSFailure",
         "Allstate_Claims_Severity")
#: los de control (los que se miden dos veces para el suelo de ruido): los cuatro primeros.
CONTROL = BANCO[:4]
#: 128-C5′ (H16): lo que tiene que ser IGUAL en la huella de los dos brazos (el entorno aparte: ver abajo).
CAMPOS_DE_LA_HUELLA_IGUALES = ("imagen_id", "sha256_medir_c5", "sha256_veredicto", "sha256_manifiesto", "memoria",
                               "cpus")
#: lo ÚNICO del entorno que distingue a los dos brazos (la señal del dispositivo y que el contenedor vea la GPU).
ENTORNO_QUE_DEFINE_EL_BRAZO = ("MATRIXAI_TABM_DISPOSITIVO", "gpus")
#: aviso (NO regla): con un suelo así de alto la regla de arriba no distingue casi nada.
SUELO_QUE_AVISA = 2.0


def _cargar(nombre: str):
    """El veredicto de C3′, por ruta (sin tocar `sys.path`). Está al lado (repo) o al lado del kit."""
    for carpeta in (AQUI, AQUI.parent):
        ruta = carpeta / f"{nombre}.py"
        if ruta.is_file():
            spec = importlib.util.spec_from_file_location(nombre, ruta)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError(f"{nombre}.py no está junto a veredicto_128_c5.py")


_v3 = _cargar("veredicto_120_c3p")
Incomparable = _v3.Incomparable
completa = _v3.completa
cifra_de_test = _v3.cifra_de_test
MOTOR_TABM = _v3.MOTOR_TABM


def _numero_finito(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validar_suelo(suelo, de_quien: str = "el suelo de ruido") -> float:
    """128-C5′ (H15): un suelo negativo o no finito no es un suelo (con −1 la banda de «igual» se volvía de 1 punto
    igualmente y con NaN todo quedaba «igual»): instrumento roto, `Incomparable`."""
    if not _numero_finito(suelo) or suelo < 0:
        raise Incomparable(f"{de_quien} no es válido ({suelo!r}): tiene que ser un número finito ≥ 0")
    return float(suelo)


def suelo_de_ruido(primera: dict[str, dict], segunda: dict[str, dict]) -> float:
    """La mayor |diferencia| de la cifra de test entre las dos pasadas de los mismos conjuntos; 0.0 si son idénticas.
    A diferencia de C3′, un campeón que cambia al repetir es ruido (se declara en `campeones_que_cambian`), no un
    error. `Incomparable` si los conjuntos no son los mismos o uno completa en una pasada y no en la otra."""
    if set(primera) != set(segunda) or not primera:
        raise Incomparable(f"el control de repetición no tiene los mismos conjuntos: {sorted(primera)} / {sorted(segunda)}")
    peor = 0.0
    for n in sorted(primera):
        a, b = primera[n], segunda[n]
        if completa(a) != completa(b):
            raise Incomparable(f"{n}: completa en una repetición y no en la otra ({a.get('estado')} / {b.get('estado')})")
        if not completa(a):
            continue
        d = abs(cifra_de_test(a) - cifra_de_test(b))
        peor = max(peor, d if d >= EXACTO else 0.0)
    return peor


def campeones_que_cambian(primera: dict[str, dict], segunda: dict[str, dict]) -> list[str]:
    return sorted(n for n in primera if completa(primera[n]) and completa(segunda.get(n))
                  and primera[n].get("campeon") != segunda[n].get("campeon"))


#: VRAM de más (MB) sobre la de antes del estudio que se SEÑALA como «posible uso de la GPU». 128-C5′ (H18): ya NO
#: prueba nada —otro componente o proceso puede haberla puesto—; solo se enseña (`vram_auxiliar`).
VRAM_QUE_DELATA_LA_GPU_MB = 200.0


def entreno_tabm_en_gpu(rec: dict | None) -> bool:
    """¿Entrenó TabM en la GPU en este conjunto? SOLO lo que el motor DECLARÓ (128-C5′, H18): el dispositivo de su
    modelo entregado o un `device`/`dispositivo` = `cuda` en alguno de SUS intentos (`dispositivos_de_tabm`, que el
    medidor saca de las entradas del motor TabM del estado). La VRAM no cuenta (`vram_auxiliar`), y tampoco
    `dispositivos_vistos` —que cuenta las declaraciones de TODO el estado, no de TabM—."""
    if not rec:
        return False
    entregado = rec.get("entrenamiento_del_entregado") or {}
    if entregado.get("dispositivo") == "cuda":
        return True
    return _cuenta_de(rec.get("dispositivos_de_tabm"), "cuda") > 0


def _cuenta_de(cuentas, clave: str) -> int:
    v = (cuentas or {}).get(clave, 0) if isinstance(cuentas, dict) else 0
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0


def declara_cuda_en_algun_sitio(rec: dict | None) -> bool:
    """Para acusar al brazo A (CPU) basta CUALQUIER declaración de `cuda` en su estado, atribuida a TabM o no: la
    referencia no puede haber tocado la GPU. (Al revés que para probar la GPU en B: ahí hace falta atribuirla.)"""
    return entreno_tabm_en_gpu(rec) or _cuenta_de((rec or {}).get("dispositivos_vistos"), "cuda") > 0


def vram_auxiliar(rec: dict | None) -> float | None:
    """La VRAM de más medida durante el estudio (MB) si llega al umbral que sugiere uso de la GPU; si no, `None`.
    EVIDENCIA AUXILIAR: se enseña, no prueba (H18)."""
    extra = ((rec or {}).get("recursos") or {}).get("vram_extra_mb")
    return float(extra) if _numero_finito(extra) and extra >= VRAM_QUE_DELATA_LA_GPU_MB else None


def entregado_de_tabm_en_cpu_sin_decirlo(rec: dict | None) -> bool:
    """128 (I1): ¿el campeón es TabM y su modelo ENTREGADO NO consta en la GPU NI declara su caída a la CPU? Es lo que
    pasaba con el B1 (la búsqueda en la GPU, el reajuste final en la CPU sin decirlo). Con el campeón en otro motor no
    hay entregado de TabM que mirar: `False`."""
    if not rec or not completa(rec) or rec.get("campeon") != MOTOR_TABM:
        return False
    entregado = rec.get("entrenamiento_del_entregado") or {}
    return not (entregado.get("dispositivo") == "cuda" or entregado.get("cayo_a_cpu") is True)


def cayo_a_cpu_el_entregado(rec: dict | None) -> bool:
    """128 (N3): ¿el modelo ENTREGADO de este conjunto declara que se entrenó en la CPU por una caída?"""
    return bool(rec and (rec.get("entrenamiento_del_entregado") or {}).get("cayo_a_cpu") is True)


def caidas_a_cpu(conjuntos: dict[str, dict]) -> list[dict]:
    """128 (N3): las caídas a CPU DECLARADAS por el modelo entregado de cada conjunto, con su motivo."""
    return [{"nombre": n, "campeon": r.get("campeon"),
             "por_que": (r.get("entrenamiento_del_entregado") or {}).get("por_que_cayo")
             or (r.get("entrenamiento_del_entregado") or {}).get("caida_a_cpu")}
            for n, r in sorted(conjuntos.items()) if cayo_a_cpu_el_entregado(r)]


#: 128 (N4): lo que la comparación INCLUYE y no se ha medido. NO es una regla: se dice junto al veredicto.
COSTE_DE_VER_CUDA = ("el brazo B lleva de más el coste de que la densa vea CUDA en el paquete GPU (RAM de host y "
                     "tiempo; NO medido): una diferencia de memoria o un OOM del B puede venir de ahí y no de TabM")


def presencia(conjuntos_a: dict[str, dict], conjuntos_b: dict[str, dict]) -> str | None:
    """El motivo para PARAR (o para no entrar) o None."""
    en_a = sorted(n for n, r in conjuntos_a.items() if completa(r) and declara_cuda_en_algun_sitio(r))
    if en_a:
        return f"el brazo A (CPU) entrenó TabM en la GPU en {en_a}: no es la referencia registrada"
    callados = sorted(n for n, r in conjuntos_b.items() if entregado_de_tabm_en_cpu_sin_decirlo(r))
    if callados:
        return (f"el brazo B entregó TabM entrenado en la CPU sin declarar la caída en {callados}: lo que se compara "
                "no es la GPU (el modelo entregado no la usó)")
    if not any(completa(r) and entreno_tabm_en_gpu(r) for r in conjuntos_b.values()):
        aux = sorted(n for n, r in conjuntos_b.items() if vram_auxiliar(r) is not None)
        nota = (f" (hay VRAM de más en {aux} como evidencia auxiliar, y no basta: la prueba es que TabM DECLARE el "
                "dispositivo en sus intentos o en el entregado)") if aux else ""
        return "el brazo B no entrenó TabM en la GPU en ningún conjunto (ninguna declaración de dispositivo): " \
               "no se midió la GPU" + nota
    return None


def clase_por_diferencia(d: float, suelo: float) -> str:
    """C5′: `sube` | `igual` | `baja_tolerada` | `empeora` según d = B − A (puntos) y el suelo. Los bordes son
    del lado que dice el contrato: sube con d ≥ m1; baja tolerada con d ≤ −m1 (y > −m2); empeora con d ≤ −m2.
    128-C5′ (H15): una diferencia no finita o un suelo negativo/no finito es `Incomparable`, no «igual»."""
    validar_suelo(suelo)
    if not _numero_finito(d):
        raise Incomparable(f"la diferencia entre brazos no es un número finito ({d!r}): una cifra del test no finita")
    m1, m2 = max(BANDA_MINIMA, suelo), max(EMPEORA_MINIMA, suelo)
    if d <= -m2 + EXACTO:
        return "empeora"
    if d <= -m1 + EXACTO:
        return "baja_tolerada"
    if d >= m1 - EXACTO:
        return "sube"
    return "igual"


def mejora_de_tiempo(pared_a, pared_b) -> bool:
    """C5′: ¿el estudio de B tardó la mitad o menos que el de A? Sin las dos paredes medidas, FINITAS y > 0, no
    (128-C5′, H15: un tiempo de 0 s, o −1, o NaN, ya contaba como «mucho más rápido»)."""
    return (_numero_finito(pared_a) and _numero_finito(pared_b) and pared_a > 0 and pared_b > 0
            and pared_b <= FRACCION_DE_TIEMPO * pared_a + EXACTO)


def _csv_distinto(a: dict, b: dict) -> str | None:
    ca, cb = a.get("csv"), b.get("csv")
    if isinstance(ca, dict) and isinstance(cb, dict):
        for k in ("sha256", "filas", "columnas", "bytes"):
            if ca.get(k) != cb.get(k):
                return f"{k} {ca.get(k)!r} en A y {cb.get(k)!r} en B"
    return None


def comparar(a: dict | None, b: dict, suelo: float) -> dict:
    """Un conjunto: brazo A (CPU) frente a B (GPU). `clase` ∈ sube | igual | mejora_de_tiempo | baja_tolerada |
    empeora | deja_de_completar | empieza_a_completar | ninguno_completa | sin_referencia. `mejora_de_tiempo` es un
    «igual» cuyo estudio de B tardó la mitad o menos. `sin_referencia` (128-C5′, H16): el brazo A NO TIENE ese
    conjunto (`a is None`): «no medido», que no es «A midió y no completó» (`empieza_a_completar`) — impide entrar y
    lo dice."""
    nombre = b.get("nombre") or (a or {}).get("nombre")
    ca, cb = completa(a), completa(b)
    base = {"nombre": nombre, "campeon_a": (a or {}).get("campeon"), "campeon_b": b.get("campeon"),
            "pared_a_s": (a or {}).get("pared_estudio_s"), "pared_b_s": b.get("pared_estudio_s"),
            "tabm_en_gpu_en_b": entreno_tabm_en_gpu(b),
            "vram_auxiliar_en_b_mb": vram_auxiliar(b),
            "cayo_a_cpu_en_b": cayo_a_cpu_el_entregado(b)}
    for brazo, pared in (("A", base["pared_a_s"]), ("B", base["pared_b_s"])):
        if isinstance(pared, (int, float)) and not isinstance(pared, bool) and pared < 0:
            raise Incomparable(f"{nombre}: el tiempo de pared del brazo {brazo} es negativo ({pared}): instrumento roto")
    if a is None:
        return {**base, "clase": "sin_referencia", "a": None, "b": cifra_de_test(b) if cb else None,
                "diferencia": None}
    distinto = _csv_distinto(a, b)
    if distinto:
        raise Incomparable(f"{nombre}: los dos brazos no midieron el mismo CSV ({distinto})")
    if ca and cb:
        if a.get("tarea") != b.get("tarea"):
            raise Incomparable(f"{nombre}: tarea {a.get('tarea')!r} en A y {b.get('tarea')!r} en B")
        xa, xb = cifra_de_test(a), cifra_de_test(b)
        d = xb - xa
        clase = clase_por_diferencia(d, suelo)
        if clase == "igual" and mejora_de_tiempo(base["pared_a_s"], base["pared_b_s"]):
            clase = "mejora_de_tiempo"
        return {**base, "clase": clase, "a": xa, "b": xb, "diferencia": d}
    clase = "deja_de_completar" if ca else "empieza_a_completar" if cb else "ninguno_completa"
    return {**base, "clase": clase, "a": cifra_de_test(a) if ca else None,
            "b": cifra_de_test(b) if cb else None, "diferencia": None}


def _es_mejora(c: dict) -> bool:
    """C5′: sube, mejora de tiempo o empieza a completar; la de un entregado que cayó a la CPU NO cuenta (N3)."""
    return c["clase"] in ("sube", "mejora_de_tiempo", "empieza_a_completar") and not c.get("cayo_a_cpu_en_b")


def corta(comparaciones: list[dict], esperados: int | None = None) -> str | None:
    """El corte (regla 9) tras el último conjunto medido del brazo B: el último empeora o deja de completar, o las
    bajas toleradas ya no se pueden compensar ni con TODOS los que faltan (`esperados` = cuántos tiene la pasada;
    sin él solo se mira lo primero)."""
    ult = comparaciones[-1]
    if ult["clase"] == "empeora":
        return f"{ult['nombre']} empeora {ult['diferencia']:.2f} puntos (la banda de empeorar)"
    if ult["clase"] == "deja_de_completar":
        return f"{ult['nombre']} deja de completar"
    if ult["clase"] == "sin_referencia":
        return f"{ult['nombre']}: el brazo A no tiene ese conjunto (no medido): no hay con qué comparar"
    if esperados is not None:
        bajas = sum(1 for c in comparaciones if c["clase"] == "baja_tolerada")
        mejoras = sum(1 for c in comparaciones if _es_mejora(c))
        faltan = max(0, esperados - len(comparaciones))
        if bajas and bajas >= mejoras + faltan:
            return (f"{bajas} bajas toleradas y {mejoras} mejoras: ni con los {faltan} conjuntos que faltan se "
                    "compensan")
    return None


def veredicto(comparaciones: list[dict], suelo: float, *, esperados: int, presencia_motivo: str | None = None,
              faltan: list[str] | None = None, incompleta_por_la_regla: bool | None = None) -> dict:
    """La regla C5′. `esperados` = cuántos conjuntos debía haber (13): con menos (corte o pasada parcial), NO entra.
    128-C5′ (R9.7): el resultado separa `instrumento_midio_bien` (comparables, lo registrado, con referencia y
    completo) de `la_regla_se_cumple` y de `producto_entra` (las dos); cada uno con sus motivos.
    `faltan` = los NOMBRES del banco que no se midieron (por identidad, no por recuento); una pasada incompleta es
    del PRODUCTO solo si la regla de corte la habría cortado ahí (`incompleta_por_la_regla`); si no, es del instrumento."""
    validar_suelo(suelo)
    cl = Counter(c["clase"] for c in comparaciones)
    candidatas = [c for c in comparaciones if c["clase"] in ("sube", "mejora_de_tiempo", "empieza_a_completar")]
    # N3: la mejora de un conjunto cuyo entregado cayó a la CPU no es de la GPU
    suben = sum(1 for c in candidatas if c["clase"] != "mejora_de_tiempo" and _es_mejora(c))
    tiempo = sum(1 for c in candidatas if c["clase"] == "mejora_de_tiempo" and _es_mejora(c))
    mejoras_caidas = [c["nombre"] for c in candidatas if c.get("cayo_a_cpu_en_b")]
    bajas = cl["baja_tolerada"]
    sin_referencia = [c["nombre"] for c in comparaciones if c["clase"] == "sin_referencia"]
    del_instrumento, de_la_regla = [], []
    if presencia_motivo:
        del_instrumento.append(presencia_motivo)
    if sin_referencia:
        del_instrumento.append(f"sin referencia del brazo A (conjuntos NO medidos, que no es «A no completó»): "
                               f"{sin_referencia}")
    if cl["empeora"]:
        de_la_regla.append(f"empeoran (bajan {max(EMPEORA_MINIMA, suelo):g} puntos o más): "
                           f"{[c['nombre'] for c in comparaciones if c['clase'] == 'empeora']}")
    if cl["deja_de_completar"]:
        de_la_regla.append(f"dejan de completar: "
                           f"{[c['nombre'] for c in comparaciones if c['clase'] == 'deja_de_completar']}")
    if suben + tiempo <= bajas:
        detalle = (f" (solo {mejoras_caidas}, que cayó a CPU: no cuenta como GPU)" if mejoras_caidas else "")
        de_la_regla.append(f"las mejoras ({suben} suben + {tiempo} de tiempo) no superan a las bajas toleradas "
                           f"({bajas}: {[c['nombre'] for c in comparaciones if c['clase'] == 'baja_tolerada']})"
                           + detalle)
    if len(comparaciones) < esperados or faltan:
        sin_medir = f" (faltan {faltan})" if faltan else ""
        (de_la_regla if incompleta_por_la_regla else del_instrumento).append(
            f"solo {len(comparaciones)} de {esperados} conjuntos medidos{sin_medir}"
            + ("" if incompleta_por_la_regla else
               " y la pasada no la cortó la regla C5′ en ese punto: está incompleta, no es un resultado de la regla"))
    instrumento_ok, regla_ok = not del_instrumento, not de_la_regla
    return {"regla": "C5′ (banda de 1 punto + tiempo): ninguno empeora ni deja de terminar y las mejoras "
                     "(suben + mejoras de tiempo) superan a las bajas toleradas",
            "suelo_de_ruido": suelo, "suelo_alto": suelo > SUELO_QUE_AVISA,
            "banda": {"igual_o_sube_desde": max(BANDA_MINIMA, suelo), "empeora_desde": max(EMPEORA_MINIMA, suelo),
                      "mejora_de_tiempo_si_pared_b_le_a_x": FRACCION_DE_TIEMPO},
            "conjuntos": [c["nombre"] for c in comparaciones], "esperados": esperados,
            "faltan": list(faltan or []),
            "suben": suben, "mejoras_de_tiempo": tiempo, "bajas_toleradas": bajas,
            "mejoras_que_cayeron_a_cpu": mejoras_caidas,
            "coste_no_medido_en_b": COSTE_DE_VER_CUDA, "empeoran": cl["empeora"],
            "dejan_de_completar": cl["deja_de_completar"], "sin_referencia": sin_referencia,
            "iguales": cl["igual"], "ninguno_completa": cl["ninguno_completa"],
            "tabm_campeon_en_b": [c["nombre"] for c in comparaciones if c["campeon_b"] == MOTOR_TABM],
            # R9.7: el instrumento y el producto, por separado y con claves que no se confunden
            "instrumento_midio_bien": instrumento_ok, "motivos_del_instrumento": del_instrumento,
            "la_regla_se_cumple": regla_ok, "motivos_de_la_regla": de_la_regla,
            "producto_entra": instrumento_ok and regla_ok,
            "entra": instrumento_ok and regla_ok,              # el mismo valor que `producto_entra` (lectores de antes)
            "motivos_para_no_entrar": del_instrumento + de_la_regla}


def incompatibilidades_de_huellas(huella_a, huella_b) -> list[str]:
    """128-C5′ (H16): por qué los dos brazos NO son comparables, o `[]`. La huella del brazo (`medir_c5`) trae la
    imagen (`imagen_id`), el código del medidor y del veredicto (sha256), el manifiesto de los CSV y los recursos
    (memoria, CPUs): tienen que ser IGUALES en los dos, y constar (lo que falta no se da por igual). El entorno
    solo puede diferir en lo que define el brazo (`ENTORNO_QUE_DEFINE_EL_BRAZO`)."""
    motivos = []
    for brazo, h in (("A", huella_a), ("B", huella_b)):
        if not isinstance(h, dict) or not h:
            motivos.append(f"el brazo {brazo} no trae su huella: no se puede comprobar que se midió lo mismo")
    if motivos:
        return motivos
    for campo in CAMPOS_DE_LA_HUELLA_IGUALES:
        va, vb = huella_a.get(campo), huella_b.get(campo)
        if va in (None, "") or vb in (None, ""):
            quien = " ni ".join(k for k, v in (("A", va), ("B", vb)) if v in (None, ""))
            motivos.append(f"{campo}: no consta en el brazo {quien}")
        elif va != vb:
            motivos.append(f"{campo}: A = {va!r} frente a B = {vb!r}")
    ea, eb = (huella_a.get("entorno") or {}), (huella_b.get("entorno") or {})
    if isinstance(ea, dict) and isinstance(eb, dict):
        sobrantes = {k: (ea.get(k), eb.get(k)) for k in sorted(set(ea) | set(eb))
                     if k not in ENTORNO_QUE_DEFINE_EL_BRAZO and ea.get(k) != eb.get(k)}
        if sobrantes:
            motivos.append(f"el entorno difiere en más que la señal del dispositivo: {sobrantes}")
    return motivos


def _exigir_identidad_de_conjuntos(a: dict, b: dict) -> None:
    """128-C5′ (H16): los conjuntos de los dos brazos son los del BANCO, por nombre; uno inesperado es `Incomparable`
    (antes se contaban registros: 13 cualesquiera completaban la pasada)."""
    for brazo, d in (("A", a), ("B", b)):
        raros = sorted(set(d.get("conjuntos") or {}) - set(BANCO))
        if raros:
            raise Incomparable(f"el brazo {brazo} trae conjuntos que no son del banco: {raros}")
        control = set(d.get("repeticion_de_control") or {})
        if control and control != set(CONTROL):
            raise Incomparable(f"la repetición de control del brazo {brazo} no es la de los 4 conjuntos de control: "
                               f"{sorted(control)} (esperados {sorted(CONTROL)})")


def resumen_emparejado(conjuntos_a: dict[str, dict], conjuntos_b: dict[str, dict]) -> dict:
    """128-C5′ (H17): la RAM pico y la pared de cada brazo, resumidas SOLO sobre los conjuntos que tienen los dos
    (emparejados), conjunto a conjunto. Un máximo de grupos distintos (13 frente a 4) puede enseñar un «ahorro» que
    es solo que el brazo tiene más conjuntos. Los que solo tiene un brazo, o a los que a un brazo le falta el dato,
    van APARTE (`no_emparejados`), con su dato, para verlos sin mezclarlos."""
    def dato(rec, campo):
        v = (rec.get("recursos") or {}).get("ram_pico_mb") if campo == "ram" else rec.get("pared_estudio_s")
        return float(v) if _numero_finito(v) and (campo == "ram" or v > 0) else None
    orden = {n: i for i, n in enumerate(BANCO)}
    comunes = sorted(set(conjuntos_a) & set(conjuntos_b), key=lambda n: (orden.get(n, len(orden)), n))
    ram = {n: {"A": dato(conjuntos_a[n], "ram"), "B": dato(conjuntos_b[n], "ram")} for n in comunes}
    pared = {n: {"A": dato(conjuntos_a[n], "pared"), "B": dato(conjuntos_b[n], "pared")} for n in comunes}
    con_ram = [n for n in comunes if ram[n]["A"] is not None and ram[n]["B"] is not None]
    con_pared = [n for n in comunes if pared[n]["A"] is not None and pared[n]["B"] is not None]

    def solo(propios, otros):
        return {n: {"ram_pico_mb": dato(r, "ram"), "pared_s": dato(r, "pared")}
                for n, r in sorted(propios.items()) if n not in otros}
    return {"conjuntos_comunes": comunes,
            "ram_pico_mb": {k: (round(max(ram[n][k] for n in con_ram), 1) if con_ram else None) for k in "AB"},
            "ram_pico_mb_por_conjunto": {n: ram[n] for n in con_ram},
            "pared_total_s": {k: (round(sum(pared[n][k] for n in con_pared), 1) if con_pared else None) for k in "AB"},
            "pared_s_por_conjunto": {n: pared[n] for n in con_pared},
            "no_emparejados": {"solo_en_A": solo(conjuntos_a, conjuntos_b), "solo_en_B": solo(conjuntos_b, conjuntos_a),
                               "sin_ram_en_un_brazo": [n for n in comunes if n not in con_ram],
                               "sin_pared_en_un_brazo": [n for n in comunes if n not in con_pared]}}


def evaluar(doc: dict, suelo_dado: float | None = None) -> dict:
    """Todo el veredicto de un `resultado_128_c5.json`: el suelo (el mayor de los dos brazos, o `suelo_dado`), las
    comparaciones y la regla. `Incomparable` si los brazos no son compatibles (imagen, código, manifiesto, CSV,
    recursos), si los conjuntos no son los del banco o si una cifra, un suelo o un tiempo no valen (128-C5′)."""
    a, b = doc["brazos"]["A"], doc["brazos"]["B"]
    _exigir_identidad_de_conjuntos(a, b)
    incompatibles = incompatibilidades_de_huellas(a.get("huella"), b.get("huella"))
    if incompatibles:
        raise Incomparable("los dos brazos no son comparables: " + "; ".join(incompatibles))
    esperados_doc = doc.get("esperados")
    if esperados_doc is not None and esperados_doc != len(BANCO):
        raise Incomparable(f"el resultado dice esperar {esperados_doc} conjuntos y el banco tiene {len(BANCO)}")
    suelos = {}
    for k, brazo in (("A", a), ("B", b)):
        if brazo.get("repeticion_de_control"):
            sin_principal = sorted(set(brazo["repeticion_de_control"]) - set(brazo["conjuntos"]))
            if sin_principal:
                raise Incomparable(f"el brazo {k} repitió {sin_principal} y no trae su medida principal: "
                                   "no hay con qué comparar la repetición")
            suelos[k] = suelo_de_ruido({n: brazo["conjuntos"][n] for n in brazo["repeticion_de_control"]},
                                       brazo["repeticion_de_control"])
    if suelo_dado is not None:
        suelo = validar_suelo(suelo_dado, "el suelo dado")
    elif len(suelos) == 2:
        suelo = max(suelos.values())
    else:
        raise Incomparable("falta la repetición de control de algún brazo: no hay suelo de ruido (usa --suelo con el "
                           "del resultado principal si es una pasada parcial)")
    validar_suelo(suelo)
    comps = []
    for n, rec in b["conjuntos"].items():
        ref = a["conjuntos"].get(n)
        if ref is not None and (ref.get("csv") is None or rec.get("csv") is None):
            raise Incomparable(f"{n}: no consta el CSV medido en algún brazo: no se puede comprobar que es el mismo")
        comps.append(comparar(ref, rec, suelo))
    medidos = {c["nombre"] for c in comps}
    faltan = [n for n in BANCO if n not in medidos]
    corte = (b.get("corte") or {})
    # una pasada cortada es del PRODUCTO si ESTA regla la habría cortado en ese punto; si la cortó otra (la E2), no
    por_la_regla = bool(faltan) and corte.get("tipo") == "regla_9" and corta(comps, esperados=len(BANCO)) is not None
    v = veredicto(comps, suelo, esperados=len(BANCO),
                  presencia_motivo=presencia(a["conjuntos"], b["conjuntos"]),
                  faltan=faltan, incompleta_por_la_regla=por_la_regla)
    v["suelos_por_brazo"] = suelos
    v["caidas_a_cpu_en_b"] = caidas_a_cpu(b["conjuntos"])           # con campeón y motivo (N3)
    emparejado = resumen_emparejado(a["conjuntos"], b["conjuntos"])
    v["resumen_emparejado"] = emparejado                            # H17: RAM y pared sobre los conjuntos comunes
    v["ram_pico_mb_por_brazo"] = emparejado["ram_pico_mb"]          # N4, ahora sobre los comunes (antes: de grupos distintos)
    v["vram_auxiliar_en_b"] = {n: vram_auxiliar(r) for n, r in sorted(b["conjuntos"].items())
                               if vram_auxiliar(r) is not None}     # H18: auxiliar, no prueba
    v["campeones_que_cambian_al_repetir"] = {
        k: campeones_que_cambian({n: brazo["conjuntos"][n] for n in brazo["repeticion_de_control"]},
                                 brazo["repeticion_de_control"])
        for k, brazo in (("A", a), ("B", b)) if brazo.get("repeticion_de_control")}
    return {"comparaciones": comps, "veredicto": v}


def _sha256_de(ruta) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def evaluacion_nueva(doc_original: dict, *, origen: dict, suelo_dado: float | None = None) -> dict:
    """128-C5′ (R9.8): evalúa con ESTA regla un `resultado_128_c5.json` ya medido (con la E2, por ejemplo) como una
    evaluación NUEVA que CITA al original (su fichero, su sha256 y lo que decía): el original no se toca ni se
    reescribe. `origen` = {"fichero": nombre, "sha256": hex}. Si los brazos no son comparables, la evaluación
    lo dice (el instrumento no midió bien) en vez de fallar."""
    original = doc_original.get("veredicto") or {}
    cita = {**origen, "regla_del_original": original.get("regla"), "entra_segun_el_original": original.get("entra"),
            "motivos_segun_el_original": original.get("motivos_para_no_entrar")}
    cabecera = {"tipo": "evaluacion_nueva_de_un_resultado_ya_medido", "regla": "C5′ (banda de 1 punto + tiempo)",
                "evaluado": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256_veredicto_128_c5": _sha256_de(__file__),
                "original": cita}
    try:
        return {"evaluacion": cabecera, **evaluar(doc_original, suelo_dado)}
    except Incomparable as e:
        return {"evaluacion": cabecera, "comparaciones": [],
                "veredicto": sin_veredicto_por_incomparable(str(e))}


def sin_veredicto_por_incomparable(motivo: str) -> dict:
    """El veredicto cuando los brazos no se pueden comparar: el INSTRUMENTO no midió bien, y no hay regla que aplicar
    (`la_regla_se_cumple` es `None`, no «falso»: no se aplicó)."""
    return {"instrumento_midio_bien": False, "motivos_del_instrumento": [f"incomparable: {motivo}"],
            "la_regla_se_cumple": None, "motivos_de_la_regla": [], "producto_entra": False, "entra": False,
            "motivos_para_no_entrar": [f"incomparable: {motivo}"]}


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 2
    ruta_original = Path(argv[1])
    doc = json.loads(ruta_original.read_text())
    suelo = float(argv[argv.index("--suelo") + 1]) if "--suelo" in argv else None
    salida = Path(argv[argv.index("--evaluacion-nueva") + 1]) if "--evaluacion-nueva" in argv else None
    if salida is not None:
        # R9.8: la evaluación es un fichero NUEVO; el original no se sobrescribe (ni por la misma ruta ni por otra
        # que ya exista).
        if salida.exists() or salida.resolve() == ruta_original.resolve():
            print(f"NO ESCRIBO: {salida} ya existe o es el resultado original; la evaluación nueva va a un fichero nuevo")
            return 2
    try:
        if salida is not None:
            r = evaluacion_nueva(doc, origen={"fichero": ruta_original.name, "sha256": _sha256_de(ruta_original)},
                                 suelo_dado=suelo)
            salida.write_text(json.dumps(r, ensure_ascii=False, indent=1))
            print(f"Evaluación nueva escrita en {salida} (cita a {ruta_original.name}, sin tocarlo)")
        else:
            r = evaluar(doc, suelo)
    except Incomparable as e:
        print("INCOMPARABLE:", e); return 3
    for c in r["comparaciones"]:
        dif = "" if c["diferencia"] is None else f"{c['diferencia']:+.2f}"
        marca = "  [cayó a CPU: no cuenta como GPU]" if c.get("cayo_a_cpu_en_b") else ""
        print(f"{c['nombre']:34} {c['clase']:20} {dif:>7}  {c['campeon_a']} -> {c['campeon_b']}{marca}")
    print(json.dumps(r["veredicto"], ensure_ascii=False, indent=1))
    for k in r["veredicto"].get("caidas_a_cpu_en_b", []):
        print(f"CAYÓ A CPU en el brazo B: {k['nombre']} (campeón {k['campeon']}, {k['por_que']}): no cuenta como GPU")
    print("NO MEDIDO, incluido en la comparación:", COSTE_DE_VER_CUDA)
    if (doc["brazos"]["B"] or {}).get("corte"):
        print("CORTADA:", doc["brazos"]["B"]["corte"])
    v = r["veredicto"]
    print("INSTRUMENTO:", "midió bien" if v["instrumento_midio_bien"] else "NO midió bien: " + "; ".join(v["motivos_del_instrumento"]))
    print("LA GPU ENTRA" if v["producto_entra"] else "LA GPU NO ENTRA (el dispositivo se queda en cpu)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
