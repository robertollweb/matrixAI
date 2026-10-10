#!/usr/bin/env python3
"""129 · C0 — MAPAS, solo lectura: por qué para TabM, qué columnas le llegan y qué cola ve su parada temprana.

Tres mapas de los 13 conjuntos no sellados del banco (los del 120/128), SIN cambiar nada del producto:

1. **Por qué para TabM** (`tabla_por_que`), de los `*_estados/` YA guardados del 128 (nada se recalcula): por conjunto y
   pliegue, en cada pasada medida, si el intento terminó por **paciencia**, por **plazo**, no dio **candidato** o lo mató el
   **techo externo**, con las épocas y la mejor época (el mismo dato que lee `reloj_c4_128.py`).
2. **Qué le llega a TabM** (`columnas`): con la preparación REAL del estudio (la del Studio —`estudio_job` + núcleo— y
   luego `redes/preparacion_tabm.ajustar_preparacion` de los motores, llamadas, no copiadas), cuántas columnas son numéricas,
   cuántas binarias o constantes EN TRAIN (el `train` de TabM, el 85 % del `entrena` del pliegue) y cuántos indicadores
   `{columna}__faltante`.
3. **La cola 85/15** (`colas`): con la partición real del estudio (semilla 0, 5 pliegues), por conjunto y pliegue, las clases
   (o la media y la dispersión, en regresión) de la cola que `_reparto_de_entrena` aparta para la parada temprana frente a las
   del `entrena` entero; si los pliegues comparten cola; y, si el conjunto declara grupos, si la cola parte una unidad.

Cómo se obtienen 2 y 3 sin copiar nada: un subproceso por conjunto llama a `estudio_job.iniciar_estudio` con el MISMO cuerpo
que el HTTP del 120/128 y sustituye EN MEMORIA `ejecutar_estudio` por una captura (como `sonda_128_c0.py`): el estudio hace
su confirmación, su partición y arma sus intentos; la captura pide a `intentos._construir(...)` el intento de TabM de cada
pliegue (la preparación del Studio, la compartida por los motores) y le aplica `ajustar_preparacion`. Nadie entrena. La guarda de
memoria de TabM se calla en memoria (como `TABM_FORZADO` del 128-C3b) para que TabM esté en los intentos también en los grandes.

    PYTHONPATH=<unión>/matrixAI:<unión>/matrixai-engines/src:<unión>/matrixaistudio/studio-backend \\
      OMP_NUM_THREADS=2 python3 mapas_129_c0.py --leer             # solo el mapa 1 (lee JSON, sin cálculo)
    ... python3 mapas_129_c0.py                                      # los tres (un subproceso por conjunto)
    ... python3 mapas_129_c0.py --solo dresses-sales,micro-mass      # algunos

Reglas ESCRITAS ANTES de mirar (se pueden cambiar solo con una enmienda, no a la vista del resultado):
- Cola «representativa» (clasificación): ninguna clase del `entrena` con esperado ≥ 1 en la cola AUSENTE de ella, y ningún
  z hipergeométrico por clase por encima de `UMBRAL_Z` (3,0).
- Cola «representativa» (regresión): |z| de la media de la cola frente a la del `entrena` ≤ `UMBRAL_Z` (con corrección de
  población finita) y la razón de desviaciones dentro de [`RAZON_STD_MIN`, `RAZON_STD_MAX`].
- R6 (C4 del contrato): las «afectadas» son las numéricas binarias o constantes en train; el corte se descarta por «nada que
  ganar» si son menos del `UMBRAL_R6` (10 %) de las columnas, salvo en los anchos.

Solo LEE los repos de la unión del 128 (no escribe en ellos ni deja `__pycache__`). Escribe `mapas_129_c0.json` junto al guion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NUCLEO = AQUI.parents[1]
HOME = Path.home()
SALIDA = AQUI / "mapas_129_c0.json"
AUXILIAR = HOME / ".agentes" / "129-c0"
UNION = HOME / "worktrees" / "f1-128-union"
UNION_PYTHONPATH = ":".join([str(UNION / "matrixAI"), str(UNION / "matrixai-engines" / "src"),
                             str(UNION / "matrixaistudio" / "studio-backend")])

UMBRAL_Z = 3.0
RAZON_STD_MIN, RAZON_STD_MAX = 0.75, 1.0 / 0.75
UMBRAL_R6 = 0.10
SOLAPE_MISMA_COLA = 0.9              # dos pliegues «tienen la misma cola» si comparten al menos esta fracción de la menor
FRACCION_DE_LA_COLA = 0.85          # la que declara `_reparto_de_entrena`; se COMPRUEBA contra su fuente (ver `_la_fuente_dice_85`)
ANCHOS = ("micro-mass", "Internet-Advertisements")
GRANDES = ("KDDCup09_appetency", "APSFailure", "Allstate_Claims_Severity")   # grupo D: columnas solo en el pliegue 0

#: Las pasadas con TabM del 128: (rótulo, directorio dentro de `benchmarks/fase0/`). La primera es la referencia sellada.
PASADA_DE_REFERENCIA = "referencia_v212"


# ============================================================================================ el mapa 1: por qué para TabM
_RX_TABM = re.compile(r"^(?P<motor>[^-]*tabm[^-]*)-.*-p(?P<pliegue>\d+)-r(?P<rep>\d+)$")


def por_que_de_un_intento(entrada: dict | None) -> dict:
    """Por qué terminó (o no) un intento de TabM. `por_que`:

    - `paciencia`: `parado_por_paciencia` (terminó de aprender); `plazo`: `parado_por_plazo` (lo cortó su reloj de entrenamiento);
    - `maximo_de_epocas`: terminó sin ninguna de las dos (se agotó el máximo de épocas);
    - `techo_externo`: el padre lo mató a los 90 s (el intento no declaró nada);
    - `sin_candidato`: el motor declaró un fallo (p. ej. el plazo se acabó antes de la primera época) o el intento no existe.
    """
    if entrada is None:
        return {"por_que": "sin_candidato", "detalle": "no hay intento de TabM para este pliegue"}
    if entrada.get("estado") != "completed":
        motivo = entrada.get("motivo") or {}
        es = (motivo.get("es") if isinstance(motivo, dict) else str(motivo)) or ""
        en = (motivo.get("en") if isinstance(motivo, dict) else "") or ""
        fuera = "desde fuera" in es or "from the outside" in en
        return {"por_que": "techo_externo" if fuera else "sin_candidato", "estado": entrada.get("estado"),
                "detalle": es[:160]}
    ent = (entrada.get("config_efectiva") or {}).get("entrenamiento_efectivo")
    if not isinstance(ent, dict):
        return {"por_que": "sin_candidato", "estado": "completed", "detalle": "completado sin entrenamiento_efectivo"}
    plazo, paciencia = bool(ent.get("parado_por_plazo")), bool(ent.get("parado_por_paciencia"))
    if plazo and paciencia:                    # no debería ocurrir: se dice en vez de elegir en silencio
        por_que = "plazo_y_paciencia"
    elif plazo:
        por_que = "plazo"
    elif paciencia:
        por_que = "paciencia"
    else:
        por_que = "maximo_de_epocas"
    ejecutadas, mejor = ent.get("epocas_ejecutadas"), ent.get("mejor_epoca")
    return {"por_que": por_que, "epocas": ejecutadas, "epocas_completas": ent.get("epocas_completas"),
            "mejor_epoca": mejor, "mejor_es_la_ultima": (mejor == ejecutadas) if mejor is not None else None,
            "ultima_epoca_parcial": bool(ent.get("ultima_epoca_parcial")),
            "plazo_antes_de_la_primera_epoca_completa": bool(ent.get("plazo_antes_de_la_primera_epoca_completa")),
            "plazo_de_entrenamiento_s": ent.get("plazo_de_entrenamiento_segundos"),
            "wall_seconds": entrada.get("wall_seconds")}


def intentos_de_tabm(estudio: dict) -> dict[int, dict]:
    """`{pliegue: entrada}` de los intentos de TabM (repetición 0) de un estudio guardado (`leaderboard`)."""
    out: dict[int, dict] = {}
    for e in estudio.get("leaderboard") or []:
        m = _RX_TABM.match(e.get("candidate") or "")
        if m and int(m.group("rep")) == 0:
            out[int(m.group("pliegue"))] = e
    return out


def ausencia_de_tabm(estudio: dict) -> str:
    """Por qué un estudio guardado NO trae intentos de TabM (cuando no los trae): `estudio_sin_resultado` (el sondeo falló, no
    hay leaderboard), `fuera_por_memoria` (la guarda del 120-C3s lo sacó: `motores_fuera_por_memoria`) o `tabm_no_competia`
    (una pasada que no lo llevaba, p. ej. un control solo de árboles)."""
    if not estudio.get("leaderboard"):
        return "estudio_sin_resultado"
    fuera = estudio.get("motores_fuera_por_memoria") or {}
    if any("tabm" in str(k) for k in fuera):
        return "fuera_por_memoria"
    return "tabm_no_competia"


def tabla_de_una_pasada(directorio: Path, *, pliegues: int = 5) -> dict[str, dict[int, dict]]:
    """`{conjunto: {pliegue: por_que_de_un_intento}}` de los `*.json` de UN directorio de estados. Un estudio sin ningún
    intento de TabM da `sin_candidato` con su `ausencia` (ver `ausencia_de_tabm`), no un cero."""
    res: dict[str, dict[int, dict]] = {}
    for f in sorted(directorio.glob("*.json")):
        est = json.loads(f.read_text())
        t = intentos_de_tabm(est)
        ausencia = None if t else ausencia_de_tabm(est)
        res[f.stem] = {p: (por_que_de_un_intento(t.get(p)) if t else
                           {"por_que": "sin_candidato", "ausencia": ausencia,
                            "detalle": ((est.get("motores_fuera_por_memoria") or {}).get("matrixai.dense.tabm_cpu") or {}
                                        ).get("es", "")[:160]})
                       for p in range(pliegues)}
    return res


def resumir_pasadas(pasadas: dict[str, dict[int, dict]]) -> dict:
    """Resumen de un conjunto a través de TODAS sus pasadas: cuántos intentos por causa, rango de épocas y de mejor época,
    cuántas veces la mejor época fue la última, y si la causa cambia entre pasadas del mismo pliegue."""
    causas: dict[str, int] = {}
    epocas, mejores, ultima, n_con_mejor = [], [], 0, 0
    por_pliegue_causas: dict[int, set] = {}
    sin_tabm: list[str] = []
    for nombre_pasada, tabla in pasadas.items():
        if tabla and all(i.get("ausencia") == "tabm_no_competia" for i in tabla.values()):
            sin_tabm.append(nombre_pasada)          # una pasada que no llevaba TabM no es un «sin candidato» de TabM
            continue
        for p, info in tabla.items():
            causas[info["por_que"]] = causas.get(info["por_que"], 0) + 1
            por_pliegue_causas.setdefault(p, set()).add(info["por_que"])
            if info.get("epocas") is not None:
                epocas.append(info["epocas"])
            if info.get("mejor_epoca") is not None:
                mejores.append(info["mejor_epoca"])
                n_con_mejor += 1
                ultima += 1 if info.get("mejor_es_la_ultima") else 0
    return {"intentos": sum(causas.values()), "por_causa": dict(sorted(causas.items())),
            "epocas": [min(epocas), max(epocas)] if epocas else None,
            "mejor_epoca": [min(mejores), max(mejores)] if mejores else None,
            "mejor_es_la_ultima": f"{ultima} de {n_con_mejor}" if n_con_mejor else None,
            "pliegues_con_causa_distinta_entre_pasadas": sorted(p for p, s in por_pliegue_causas.items() if len(s) > 1),
            "pasadas_sin_tabm_no_contadas": sorted(sin_tabm)}


def pasadas_con_tabm(base: Path) -> dict[str, Path]:
    """Los directorios de estados con TabM de las 12+ pasadas del 128 (se leen todos los que existen)."""
    d: dict[str, Path] = {}
    ref, c4 = base / "referencia_128_pcpu_v212_estados", base / "c4_128_pcpu_estados"
    d[PASADA_DE_REFERENCIA] = ref
    d["integracion_c4"] = c4
    for i in range(1, 6):
        d[f"control_tabm_v212_{i}"] = base / f"control_tabm_v212_{i}_estados"
        d[f"control_tabm_c4_{i}"] = base / f"control_tabm_c4_{i}_estados"
    for i in (1, 2):
        d[f"control_pendigits_v212_{i}"] = base / f"control_pendigits_v212_{i}_estados"
        d[f"control_pendigits_c4_{i}"] = base / f"control_pendigits_c4_{i}_estados"
    d["repeticion_de_control_v212"] = ref / "repeticion_de_control"
    d["repeticion_de_control_c4"] = c4 / "repeticion_de_control"
    return {k: v for k, v in d.items() if v.is_dir()}


def resumen_por_fases(ruta: Path) -> dict:
    """De `tabm_por_fases_128_r4.json` (la sonda por fase del 128-R4): por conjunto y corrida (plazo del producto / plazo amplio),
    cómo paró TabM, épocas, segundos por época (entrenar + validar) y de preparación. Con esto se ve cuánto necesitan para parar
    por paciencia house_16H y mfeat-factors y qué pasa con los grandes con 1.800 s."""
    datos = json.loads(ruta.read_text())
    out: dict[str, dict] = {}
    for clave, v in (datos.get("conjuntos") or {}).items():
        nombre = clave.split(":")[0]
        res_por_rotulo = {r["rotulo"]: r for r in v.get("resumen_de_corridas") or []}
        corridas = []
        for c in (v.get("medida") or {}).get("corridas") or []:
            ee = c.get("entrenamiento_efectivo")
            info = por_que_de_un_intento({"estado": c.get("estado"), "motivo": c.get("motivo"),
                                          "config_efectiva": {"entrenamiento_efectivo": ee}} if ee else
                                         {"estado": c.get("estado"), "motivo": c.get("motivo")})
            r = res_por_rotulo.get(c["rotulo"]) or {}
            ep = r.get("epocas") or {}
            partes = r.get("partes_del_ajuste") or {}
            corridas.append({"rotulo": c["rotulo"], "plazo_s": c.get("plazo_s"), "estado": c.get("estado"),
                             "por_que": info["por_que"], "epocas": info.get("epocas"), "mejor_epoca": info.get("mejor_epoca"),
                             "pared_s": r.get("pared_s"), "ajustar_s": partes.get("ajustar_s"),
                             "preparar_s": partes.get("preparar_s"), "entrenar_mediana_por_epoca_s": ep.get("entrenar_mediana_s"),
                             "validar_mediana_por_epoca_s": ep.get("validar_mediana_s"), "filas": c.get("filas")})
        out[nombre] = {"corridas": corridas, "csv": v.get("csv")}
    return out


def mapa_por_que(base: Path) -> dict:
    """El mapa 1 entero: por conjunto, la tabla de la referencia (5 pliegues), la de cada pasada y el resumen."""
    todas = {nombre: tabla_de_una_pasada(ruta) for nombre, ruta in pasadas_con_tabm(base).items()}
    conjuntos = sorted({c for t in todas.values() for c in t})
    out: dict[str, dict] = {}
    for c in conjuntos:
        pas = {nombre: t[c] for nombre, t in todas.items() if c in t}
        out[c] = {"referencia_v212": pas.get(PASADA_DE_REFERENCIA),
                  "pasadas": {n: {str(p): i for p, i in tab.items()} for n, tab in pas.items()},
                  "resumen_de_todas_las_pasadas": resumir_pasadas(pas),
                  "resumen_de_la_referencia": resumir_pasadas({PASADA_DE_REFERENCIA: pas[PASADA_DE_REFERENCIA]})
                  if PASADA_DE_REFERENCIA in pas else None}
        if out[c]["referencia_v212"] is not None:
            out[c]["referencia_v212"] = {str(p): i for p, i in out[c]["referencia_v212"].items()}
    ruta_fases = base / "tabm_por_fases_128_r4.json"
    return {"pasadas_leidas": {n: str(r.relative_to(base)) for n, r in pasadas_con_tabm(base).items()},
            "conjuntos": out,
            "tabm_por_fases_128_r4": resumen_por_fases(ruta_fases) if ruta_fases.exists() else None}


# ======================================================================================== el mapa 2: columnas que llegan a TabM
def clasificar_columnas(nombres, matriz, sufijo_faltante: str = "__faltante") -> dict:
    """Recuento de las columnas NUMÉRICAS de TabM por cuántos valores distintos tienen en la matriz de TRAIN (la que TabM
    ve: tras rellenar con la mediana). `constante`: 1 valor; `binaria`: exactamente 2. Los indicadores se cuentan aparte."""
    import numpy as np  # noqa: PLC0415

    matriz = np.asarray(matriz)
    if matriz.ndim != 2 or matriz.shape[1] != len(nombres):
        raise ValueError(f"matriz {matriz.shape} no cuadra con {len(nombres)} nombres")
    cuenta = {"numericas": len(nombres), "constantes": 0, "binarias": 0, "binarias_01": 0, "faltante": 0,
              "faltante_constantes": 0, "faltante_binarias": 0, "no_faltante_constantes": 0, "no_faltante_binarias": 0}
    afectadas: list[str] = []
    for j, nombre in enumerate(nombres):
        valores = np.unique(matriz[:, j])
        distintos = len(valores)
        es_indicador = str(nombre).endswith(sufijo_faltante)
        cuenta["faltante"] += 1 if es_indicador else 0
        if distintos <= 1:
            cuenta["constantes"] += 1
            cuenta["faltante_constantes" if es_indicador else "no_faltante_constantes"] += 1
            afectadas.append(str(nombre))
        elif distintos == 2:
            cuenta["binarias"] += 1
            cuenta["binarias_01"] += 1 if set(valores.tolist()) == {0.0, 1.0} else 0
            cuenta["faltante_binarias" if es_indicador else "no_faltante_binarias"] += 1
            afectadas.append(str(nombre))
    cuenta["afectadas_por_r6"] = cuenta["constantes"] + cuenta["binarias"]
    cuenta["proporcion_afectadas"] = round(cuenta["afectadas_por_r6"] / len(nombres), 4) if len(nombres) else None
    cuenta["_afectadas"] = afectadas
    return cuenta


def comparar_con_el_estado(nombres_numericas, nombres_categoricas, config_efectiva: dict | None) -> dict:
    """CONTROL del instrumento: las columnas que este mapa dice que llegan a TabM frente a las que el propio intento guardó
    (`config_efectiva.columnas_numericas` / `columnas_categoricas` de su estado). Si el estado las omitió por tamaño (un dict
    con `omitido`), no es comparable y se dice; si son listas, tienen que ser el MISMO conjunto de nombres."""
    ce = config_efectiva or {}
    num, cat = ce.get("columnas_numericas"), ce.get("columnas_categoricas")
    if not isinstance(num, list) or not isinstance(cat, list):
        return {"comparable": False, "motivo": "el estado no guardó las listas de columnas (omitidas por tamaño o ausentes)"}
    return {"comparable": True, "numericas_coinciden": sorted(num) == sorted(nombres_numericas),
            "categoricas_coinciden": sorted(cat) == sorted(nombres_categoricas),
            "en_el_estado": [len(num), len(cat)], "en_el_mapa": [len(nombres_numericas), len(nombres_categoricas)]}


def columnas_de_tabm(train_features, predictores, *, d_embedding: int, semilla: int) -> dict:
    """LO MISMO que hace el motor al entrenar (`densa_tabm._ajustar_con_torch`): `ajustar_preparacion` sobre las `features` de
    train y `predictors` del spec; después se cuenta sobre la matriz numérica que `_matriz_numerica` arma con las medianas."""
    from matrixai_engines.redes import preparacion_tabm as pt  # noqa: PLC0415

    prep = pt.ajustar_preparacion(train_features, predictores, d_embedding=d_embedding, semilla=semilla)
    num = tuple(prep.columnas_numericas)
    cuenta = {"numericas": 0, "categoricas": len(prep.columnas_categoricas), "predictores": len(predictores),
              "_nombres_numericas": sorted(num), "_nombres_categoricas": sorted(prep.columnas_categoricas),
              "columnas_de_entrada": len(num) + len(prep.columnas_categoricas), "filas_train": len(train_features)}
    if num:
        matriz = pt._matriz_numerica(train_features, num, prep.medianas)
        cuenta.update(clasificar_columnas(num, matriz))
    else:
        cuenta.update(numericas=0, constantes=0, binarias=0, faltante=0, afectadas_por_r6=0, proporcion_afectadas=None,
                      _afectadas=[])
    en_entrada = cuenta["columnas_de_entrada"]
    cuenta["proporcion_afectadas_de_la_entrada"] = (round(cuenta["afectadas_por_r6"] / en_entrada, 4)
                                                    if en_entrada else None)
    return cuenta


# ============================================================================================== el mapa 3: la cola 85/15
def _hipergeometrica(n_pob: int, k_pob: int, n_muestra: int) -> tuple[float, float]:
    esperado = n_muestra * k_pob / n_pob
    varianza = (n_muestra * (k_pob / n_pob) * (1 - k_pob / n_pob) * (n_pob - n_muestra) / (n_pob - 1)
                if n_pob > 1 else 0.0)
    return esperado, varianza


def diagnostico_de_cola_clasificacion(etiquetas_entrena, etiquetas_cola) -> dict:
    """Clases del `entrena` entero frente a las de la cola. Esperado = n_clase × |cola| / |entrena|. z hipergeométrico por
    clase (muestreo sin reposición). `ausentes_graves`: ausentes de la cola con esperado ≥ 1."""
    from collections import Counter  # noqa: PLC0415

    ent, cola = Counter(etiquetas_entrena), Counter(etiquetas_cola)
    n_ent, n_cola = sum(ent.values()), sum(cola.values())
    por_clase, ausentes, z_max, peor = {}, [], 0.0, None
    for clase in sorted(ent, key=str):
        esperado, var = _hipergeometrica(n_ent, ent[clase], n_cola)
        obs = cola.get(clase, 0)
        z = (obs - esperado) / math.sqrt(var) if var > 0 else 0.0
        por_clase[str(clase)] = {"entrena": ent[clase], "cola": obs, "esperado": round(esperado, 2), "z": round(z, 2)}
        if obs == 0:
            ausentes.append({"clase": str(clase), "entrena": ent[clase], "esperado": round(esperado, 2)})
        if abs(z) > abs(z_max):
            z_max, peor = z, str(clase)
    graves = [a for a in ausentes if a["esperado"] >= 1.0]
    # la cola puede traer una clase que el `entrena` no tiene: imposible si es su sufijo, se dice si ocurriera
    nuevas = sorted(str(c) for c in cola if c not in ent)
    return {"tipo": "clasificacion", "n_entrena": n_ent, "n_cola": n_cola, "clases_entrena": len(ent),
            "clases_en_cola": sum(1 for c in ent if cola.get(c, 0) > 0), "ausentes": ausentes,
            "ausentes_graves": graves, "clases_nuevas_en_la_cola": nuevas, "z_max": round(z_max, 2), "clase_del_z_max": peor,
            "minimo_en_cola": min(cola.values()) if cola else 0,
            "minimo_en_entrena": min(ent.values()) if ent else 0, "por_clase": por_clase,
            "representativa": (not graves) and abs(z_max) <= UMBRAL_Z and not nuevas}


def diagnostico_de_cola_regresion(y_entrena, y_cola) -> dict:
    """Media y dispersión del objetivo en la cola frente al `entrena` (z con corrección de población finita)."""
    n_ent, n_cola = len(y_entrena), len(y_cola)
    m_ent = sum(y_entrena) / n_ent
    s_ent = math.sqrt(sum((v - m_ent) ** 2 for v in y_entrena) / n_ent)
    m_cola = sum(y_cola) / n_cola
    s_cola = math.sqrt(sum((v - m_cola) ** 2 for v in y_cola) / n_cola)
    error = s_ent * math.sqrt((1.0 / n_cola) * (1.0 - n_cola / n_ent)) if n_cola < n_ent else 0.0
    z = (m_cola - m_ent) / error if error > 0 else 0.0
    razon = (s_cola / s_ent) if s_ent > 0 else 1.0
    return {"tipo": "regresion", "n_entrena": n_ent, "n_cola": n_cola, "media_entrena": m_ent, "media_cola": m_cola,
            "std_entrena": s_ent, "std_cola": s_cola, "z_media": round(z, 2), "razon_std": round(razon, 3),
            "representativa": abs(z) <= UMBRAL_Z and RAZON_STD_MIN <= razon <= RAZON_STD_MAX}


def unidades_partidas(unidad_de_ajuste, unidad_de_cola) -> dict:
    """En `groups`: unidades que tienen filas EN el ajuste y EN la cola a la vez (el reparto posicional las parte)."""
    partidas = sorted(set(unidad_de_ajuste) & set(unidad_de_cola), key=str)
    return {"unidades_en_el_ajuste": len(set(unidad_de_ajuste)), "unidades_en_la_cola": len(set(unidad_de_cola)),
            "unidades_partidas": len(partidas), "ejemplos": [str(u) for u in partidas[:5]], "parte_una_unidad": bool(partidas)}


def solape_de_colas(colas: dict[int, list]) -> dict:
    """¿Comparten cola los pliegues? Por cada par, filas en común / filas de la menor; y los grupos de pliegues con la misma
    cola (≥ `SOLAPE_MISMA_COLA` de la menor) -- si lo hacen, la parada temprana de esos pliegues ve las MISMAS filas."""
    conj = {p: set(v) for p, v in colas.items()}
    pares = {}
    ps = sorted(conj)
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            menor = min(len(conj[a]), len(conj[b])) or 1
            pares[f"{a}-{b}"] = round(len(conj[a] & conj[b]) / menor, 3)
    grupos: list[list[int]] = []
    for p in ps:
        for g in grupos:
            if all(len(conj[p] & conj[q]) / (min(len(conj[p]), len(conj[q])) or 1) >= SOLAPE_MISMA_COLA for q in g):
                g.append(p)
                break
        else:
            grupos.append([p])
    return {"solape_por_par": pares, "grupos_con_la_misma_cola": grupos,
            "pliegues_con_cola_propia": sum(1 for g in grupos if len(g) == 1)}


def decidir_sobre_h20(colas_por_conjunto: dict[str, dict]) -> dict:
    """La decisión que pide §4.1: ¿la cola es representativa en los 13? `colas_por_conjunto[c]["pliegues"][p]` lleva
    `diagnostico` (con `representativa`) y, si el conjunto declara grupos, `unidades`. Sin defecto visible → H20 queda en
    «corrección de método sin efecto medible»; si no, se nombra qué conjuntos y pliegues."""
    con_defecto: dict[str, dict] = {}
    n_pliegues = 0
    for c, d in colas_por_conjunto.items():
        for p, info in d.get("pliegues", {}).items():
            n_pliegues += 1
            diag = info.get("diagnostico") or {}
            motivos = []
            if diag.get("ausentes_graves"):
                motivos.append(f"{len(diag['ausentes_graves'])} clase(s) con esperado ≥ 1 ausente(s) de la cola")
            elif diag.get("representativa") is False:
                if diag.get("tipo") == "clasificacion":
                    motivos.append(f"z máximo por clase {diag['z_max']} (> {UMBRAL_Z}) en la clase {diag['clase_del_z_max']}")
                else:
                    motivos.append(f"z de la media {diag['z_media']}, razón de dispersión {diag['razon_std']}")
            if (info.get("unidades") or {}).get("parte_una_unidad"):
                motivos.append(f"la cola parte {info['unidades']['unidades_partidas']} unidad(es)")
            if motivos:
                con_defecto.setdefault(c, {})[p] = motivos
    return {"pliegues_mirados": n_pliegues, "conjuntos_con_defecto": con_defecto,
            "la_cola_es_representativa_en_todos": not con_defecto,
            "veredicto": ("la cola 85/15 es representativa en todos los pliegues: H20 queda en «corrección de método sin "
                          "efecto medible» (pregunta 4 del contrato)") if not con_defecto else
            ("hay colas NO representativas en: " + ", ".join(sorted(con_defecto)) + " -- C3 tiene algo que arreglar que se ve")}


def conjuntos_sin_nada_que_ganar_en_r6(columnas_por_conjunto: dict[str, dict]) -> dict:
    """Regla del C4: ¿menos del 10 % de las columnas de entrada son binarias o constantes? (salvo los anchos, que se miden aparte)."""
    bajo, alto = [], []
    for c, d in columnas_por_conjunto.items():
        prop = d.get("proporcion_afectadas_de_la_entrada")
        if prop is None:
            continue
        (bajo if prop < UMBRAL_R6 else alto).append(c)
    return {"umbral": UMBRAL_R6, "por_debajo_del_umbral": sorted(bajo), "por_encima_del_umbral": sorted(alto),
            "anchos": [c for c in ANCHOS if c in columnas_por_conjunto]}


def control_del_pliegue_0(nombre: str, fold0: dict, base: Path) -> dict:
    """Compara el pliegue 0 de este mapa con el intento de TabM del pliegue 0 de la REFERENCIA v2.12 (mismos datos, misma
    partición: semilla 0)."""
    ruta = base / "referencia_128_pcpu_v212_estados" / f"{nombre}.json"
    if not ruta.exists():
        return {"comparable": False, "motivo": "sin estado de la referencia"}
    e = intentos_de_tabm(json.loads(ruta.read_text())).get(0)
    if e is None:
        return {"comparable": False, "motivo": "la referencia no tiene intento de TabM en el pliegue 0"}
    return comparar_con_el_estado(fold0["_nombres_numericas"], fold0["_nombres_categoricas"], e.get("config_efectiva"))


# ====================================================================================================== el trabajo por conjunto
def cuerpo_del_estudio(gen, c: dict) -> dict:
    """El MISMO cuerpo que `referencia_120_r0.estudiar` manda al HTTP (csv crudo, objetivo, tarea, entradas, clase positiva)."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("referencia_120_r0_129c0", AQUI / "referencia_120_r0.py")
    r0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r0)
    csv_bytes, cols, excl, nfilas = r0.csv_de(gen, c["data_id"])
    cuerpo = {"csv_text": csv_bytes.decode("utf-8"), "objetivo": c["objetivo"], "tarea": c["tarea"],
              "entradas": [x for x in cols if x != c["objetivo"]], "unidad_de_observacion": "una fila"}
    if c["tarea"] == "binary_classification" and c.get("clase_positiva") is not None:
        cuerpo["clase_positiva"] = c["clase_positiva"]
    return cuerpo


def _la_fuente_dice_85(ej) -> dict:
    """Localiza el corte 85/15 EN LA FUENTE (línea y texto), para citarlo y para que el mapa falle si cambia."""
    import inspect  # noqa: PLC0415

    lineas, inicio = inspect.getsourcelines(ej._reparto_de_entrena)
    corte = next(((inicio + i, l.strip()) for i, l in enumerate(lineas) if "0.85" in l and "corte" in l), None)
    return {"funcion": "_reparto_de_entrena", "archivo": inspect.getsourcefile(ej._reparto_de_entrena),
            "linea_de_la_funcion": inicio, "linea_del_corte": corte[0] if corte else None,
            "texto_del_corte": corte[1] if corte else None, "es_85_posicional": bool(corte)}


def trabajo_de_un_conjunto(nombre: str, c: dict, salida: Path) -> int:
    """Corre en un SUBPROCESO por conjunto. Escribe `salida` (JSON) con las columnas y las colas de ese conjunto."""
    import importlib.util  # noqa: PLC0415

    import matrixai_studio.estudio_job as ej  # noqa: PLC0415
    from matrixai_engines.motores import densa_tabm  # noqa: PLC0415

    R: dict = {"conjunto": nombre, "modulos": {"estudio_job": ej.__file__, "densa_tabm": densa_tabm.__file__},
               "inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "colas": {}, "columnas": {}}
    for m in R["modulos"].values():
        if not str(m).startswith(str(UNION)):
            R["error"] = f"importa {m}, no la unión del 128"
            salida.write_text(json.dumps(R, indent=1, ensure_ascii=False))
            return 4
    spec = importlib.util.spec_from_file_location("generar_ejemplos_medidos_129c0",
                                                  HOME / "matrixaistudio/studio-backend/scripts/generar_ejemplos_medidos.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    cuerpo = cuerpo_del_estudio(gen, c)
    R["cuerpo"] = {k: (v if k != "csv_text" else f"<{len(v)} caracteres>") for k, v in cuerpo.items() if k != "entradas"}
    R["cuerpo"]["n_entradas"] = len(cuerpo["entradas"])
    R["corte_85_15"] = _la_fuente_dice_85(ej)
    pliegues_columnas = 1 if nombre in GRANDES else None            # None = todos
    de_verdad = ej._memoria_de_tabm_que_no_cabe
    ej._memoria_de_tabm_que_no_cabe = lambda filas, columnas: None   # como TABM_FORZADO: TabM está en los intentos del grande
    hecho = {"capturado": False}
    d_emb = densa_tabm.D_EMBEDDING
    semilla = densa_tabm.SEMILLA_DEL_RUIDO_DE_CUANTILES

    def captura(estudio, motores, intentos, *, fase=None, directorio_checkpoint=None, cancelado=lambda: False,
                aislar_por_proceso=False, **_):
        if hecho["capturado"]:
            return ej.Leaderboard(estudio_id=estudio.estudio_id)
        hecho["capturado"] = True
        try:
            propuesta, problema, por_id = intentos._propuesta, intentos._problema, intentos._por_id
            objetivo = problema.target
            claves = list(intentos._claves)
            n_pliegues = len({k[1] for k in claves if k[0] == 0})
            R["tarea"] = problema.task
            R["split_type"] = cuerpo.get("split_type", "iid")
            R["unit_id_field"] = cuerpo.get("unit_id_field")
            motores_en_los_intentos = sorted({k[2] for k in claves})
            R["motores"] = motores_en_los_intentos
            try:
                R["la_guarda_de_memoria_de_verdad_habria_dicho"] = de_verdad(len(por_id), len(problema.predictors))
            except Exception as exc:   # noqa: BLE001 -- es informativa
                R["la_guarda_de_memoria_de_verdad_habria_dicho"] = f"{type(exc).__name__}: {exc}"
            R["n_pliegues"] = n_pliegues
            R["limites_de_la_particion"] = [getattr(l, "clave", str(l)) for l in (propuesta.limites or ())]
            colas_ids: dict[int, list] = {}
            for i in range(n_pliegues):
                pliegue = propuesta.pliegues.pliegue_de(repeticion=0, pliegue=i)
                ajuste, cola = ej._reparto_de_entrena(pliegue)
                colas_ids[i] = list(cola)
                ent = list(pliegue.entrena)
                if problema.task == "regression":
                    diag = diagnostico_de_cola_regresion([por_id[x][objetivo] for x in ent],
                                                         [por_id[x][objetivo] for x in cola])
                else:
                    diag = diagnostico_de_cola_clasificacion([por_id[x][objetivo] for x in ent],
                                                             [por_id[x][objetivo] for x in cola])
                info = {"n_entrena": len(ent), "n_ajuste": len(ajuste), "n_cola": len(cola), "n_valida": len(pliegue.valida),
                        "diagnostico": diag}
                if cuerpo.get("unit_id_field"):
                    campo = cuerpo["unit_id_field"]
                    info["unidades"] = unidades_partidas([por_id[x][campo] for x in ajuste], [por_id[x][campo] for x in cola])
                R["colas"][str(i)] = info
            R["solape_de_colas"] = solape_de_colas(colas_ids)
            tabm_id = ej._MOTOR_TABM
            if tabm_id not in motores_en_los_intentos:
                R["error"] = f"TabM no está entre los motores de los intentos: {motores_en_los_intentos}"
                return ej.Leaderboard(estudio_id=estudio.estudio_id)
            for i in range(n_pliegues if pliegues_columnas is None else pliegues_columnas):
                d = intentos._construir(0, i, tabm_id)
                train = d.train if getattr(d, "en_fichero", None) is None else d.en_fichero.cargar()[0]
                cuenta = columnas_de_tabm(train.features, d.spec.predictors, d_embedding=d_emb, semilla=semilla)
                cuenta["columnas_tras_preparar_del_studio"] = len(train.features[0]) if train.features else 0
                cuenta["pliegue"] = i
                R["columnas"][str(i)] = cuenta
                del d, train
        except BaseException as exc:   # noqa: BLE001 -- el mapa que falla lo DICE; no se queda a medias en silencio
            import traceback  # noqa: PLC0415
            R["error"] = f"{type(exc).__name__}: {exc}"
            R["traza"] = traceback.format_exc()[-1500:]
        finally:
            if hasattr(intentos, "soltar"):
                intentos.soltar()
        return ej.Leaderboard(estudio_id=estudio.estudio_id)

    ej.ejecutar_estudio = captura
    ej._reajustar_en_un_proceso_aparte = lambda *a, **k: None
    directorio = AUXILIAR / "estudios_tmp" / nombre
    directorio.mkdir(parents=True, exist_ok=True)
    r = ej.iniciar_estudio(cuerpo, directorio_de_estudios=directorio)
    R["respuesta"] = {k: v for k, v in (r.get("estudio") or r).items() if k in ("estado", "job_id", "clave", "error")}
    job = ej._jobs.get((r.get("estudio") or {}).get("job_id"))
    if job is not None:
        job["hilo"].join()
    R["fin"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not hecho["capturado"] and "error" not in R:
        R["error"] = "la captura no llegó a ejecutarse: el estudio no pasó a la competición"
    salida.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str))
    return 0 if "error" not in R else 3


# ============================================================================================================== procedencia
def sha256_de(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"<git falló: {r.stderr.strip()[:100]}>"


def procedencia(base: Path, bancos: list[dict]) -> dict:
    repos = {"union_matrixAI": UNION / "matrixAI", "union_matrixai-engines": UNION / "matrixai-engines",
             "union_matrixaistudio": UNION / "matrixaistudio", "nucleo_principal": NUCLEO}
    leidos = {
        "estudio_job.py": UNION / "matrixaistudio/studio-backend/matrixai_studio/estudio_job.py",
        "particion_por_diseno.py": UNION / "matrixAI/matrixai/training/particion_por_diseno.py",
        "preparacion.py (núcleo)": UNION / "matrixAI/matrixai/training/preparacion.py",
        "preparacion_tabm.py": UNION / "matrixai-engines/src/matrixai_engines/redes/preparacion_tabm.py",
        "densa_tabm.py": UNION / "matrixai-engines/src/matrixai_engines/motores/densa_tabm.py",
        "referencia_120_r0.py": AQUI / "referencia_120_r0.py", "protocolo_120.json": AQUI / "protocolo_120.json",
        "reloj_c4_128.py": AQUI / "reloj_c4_128.py", "tabm_por_fases_128_r4.json": AQUI / "tabm_por_fases_128_r4.json",
        "generar_ejemplos_medidos.py": HOME / "matrixaistudio/studio-backend/scripts/generar_ejemplos_medidos.py",
        "este guion": Path(__file__).resolve()}
    estados: dict[str, str] = {}
    for _, ruta in pasadas_con_tabm(base).items():
        for f in sorted(ruta.glob("*.json")):
            estados[str(f.relative_to(base))] = sha256_de(f)
    arff = {}
    ruta_arff = HOME / "fase0_openml_datos" / "arff"
    for c in bancos:
        f = ruta_arff / f"{c['data_id']}.arff"
        if f.exists():
            arff[f"{c['nombre']} ({c['data_id']}.arff)"] = sha256_de(f)
    return {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "commits": {k: {"HEAD": _git(r, "rev-parse", "HEAD")[:12], "sucio": bool(_git(r, "status", "--porcelain"))}
                        for k, r in repos.items()},
            "sha256_de_lo_leido": {k: sha256_de(p) for k, p in leidos.items() if p.exists()},
            "sha256_de_los_estados": estados, "sha256_de_los_arff": arff,
            "constantes": {"UMBRAL_Z": UMBRAL_Z, "RAZON_STD": [RAZON_STD_MIN, RAZON_STD_MAX], "UMBRAL_R6": UMBRAL_R6}}


# ==================================================================================================================== main
def _banco() -> list[dict]:
    return json.loads((AQUI / "protocolo_120.json").read_text())["conjuntos"]["banco"]


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leer", action="store_true", help="solo el mapa 1 (JSON ya guardados)")
    ap.add_argument("--solo", default=None, help="conjuntos separados por comas")
    ap.add_argument("--memoria", default=None, help="techo de memoria del subproceso de cada conjunto (p. ej. 5G), vía systemd-run")
    ap.add_argument("--worker", default=None, help="(interno) corre UN conjunto y escribe su parte")
    ap.add_argument("--salida", default=str(SALIDA))
    a = ap.parse_args(argv)
    banco = _banco()
    if a.worker:
        c = next(x for x in banco if x["nombre"] == a.worker)
        return trabajo_de_un_conjunto(a.worker, c, Path(a.salida))
    elegidos = [c for c in banco if a.solo is None or c["nombre"] in a.solo.split(",")]
    if a.solo and len(elegidos) != len(a.solo.split(",")):
        raise SystemExit(f"--solo: hay nombres que no son del banco ({[c['nombre'] for c in banco]})")
    res: dict = {"contrato": "129-C0", "procedencia": procedencia(AQUI, banco),
                 "tabla_por_que": mapa_por_que(AQUI)}
    if not a.leer:
        AUXILIAR.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=UNION_PYTHONPATH, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", MATRIXAI_ESTUDIO_SIN_DENSA="1", MATRIXAI_ESTUDIOS_DIR=str(AUXILIAR / "estudios_tmp"))
        env.pop("MATRIXAI_ESTUDIO_SIN_TABM", None)
        columnas, colas, detalle = {}, {}, {}
        for c in elegidos:
            espera = 0
            while float(open("/proc/loadavg").read().split()[0]) > 6:       # regla del encargo: con carga > 6 se espera
                if espera >= 1800:
                    raise SystemExit("carga > 6 durante 30 min seguidos: no se calcula; reintentar luego")
                time.sleep(30)
                espera += 30
            parte = AUXILIAR / f"parte_{c['nombre']}.json"
            parte.unlink(missing_ok=True)
            t0 = time.time()
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", c["nombre"], "--salida", str(parte)]
            if a.memoria:    # techo del subproceso (sin swap en esta máquina): si se pasa lo mata el cgroup, no el servidor
                cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.memoria}", "-p", "MemorySwapMax=0"] + cmd
            rc = subprocess.run(cmd, env=env, cwd=str(AUXILIAR)).returncode
            if not parte.exists():
                raise SystemExit(f"{c['nombre']}: el subproceso (rc {rc}) no dejó parte")
            p = json.loads(parte.read_text())
            p["pared_s"] = round(time.time() - t0, 1)
            p["rc"] = rc
            detalle[c["nombre"]] = {k: p.get(k) for k in ("rc", "pared_s", "error", "traza", "tarea", "split_type", "unit_id_field",
                                                           "motores", "n_pliegues", "limites_de_la_particion", "corte_85_15",
                                                           "modulos", "respuesta", "solape_de_colas",
                                                           "la_guarda_de_memoria_de_verdad_habria_dicho")}
            columnas[c["nombre"]] = p.get("columnas") or {}
            fold0 = columnas[c["nombre"]].get("0") if isinstance(columnas[c["nombre"]], dict) else None
            if fold0 and "_nombres_numericas" in fold0:
                detalle[c["nombre"]]["control_contra_el_estado_de_la_referencia"] = control_del_pliegue_0(
                    c["nombre"], fold0, AQUI)
            for f in columnas[c["nombre"]].values():
                if isinstance(f, dict):
                    f.pop("_nombres_numericas", None)
                    f.pop("_nombres_categoricas", None)
            colas[c["nombre"]] = {"pliegues": p.get("colas") or {}, "solape": p.get("solape_de_colas")}
            print(f"{c['nombre']:36} rc {rc}  {p['pared_s']:7.1f} s  {p.get('error', 'ok')}", flush=True)
        res["detalle_por_conjunto"] = detalle
        res["columnas"] = columnas
        res["colas"] = colas
        res["decision_h20"] = decidir_sobre_h20(colas)
        res["r6"] = conjuntos_sin_nada_que_ganar_en_r6(
            {c: (d.get("0") or {}) for c, d in columnas.items() if isinstance(d, dict)})
        res["parcial"] = a.solo is not None
    Path(a.salida).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print("escrito", a.salida)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
