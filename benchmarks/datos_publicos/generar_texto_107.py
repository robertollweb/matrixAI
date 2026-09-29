# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — LOS DATOS PÚBLICOS DE LA MEDICIÓN DE TEXTO, GENERADOS DESDE LOS REGISTROS
SELLADOS de `benchmarks/texto_107c30/`.

Mismo patrón que `generar.py` (117-C1, Fase 0): **ninguna cifra se escribe a mano**. Este
guion lee los JSON sellados de `benchmarks/texto_107c30/` -- NUNCA el CSV de `tareas/`, que
es un enlace sin versionar a datos locales -- y compone `texto_107_publico.json`. Una
prueba (`tests/test_107_c30_datos_publicos_texto.py`) lo regenera y lo compara BYTE A BYTE
con el publicado.

    python3 benchmarks/datos_publicos/generar_texto_107.py            # comprueba el publicado
    python3 benchmarks/datos_publicos/generar_texto_107.py --escribir # lo regenera

**Las cuatro tareas, las SEIS condiciones.** A (noticias falsas, auroc), B (casos
clínicos, auroc), C (contratos PLACSP, `rmse` -- MENOS es mejor), D (BOE, auroc). Seis
condiciones por tarea: (0) sin texto, (1) embedding, (2) TF-IDF, (3) respondedor local
(Qwen2.5-0.5B), (4) Jev fabricando columnas, (5) Jev preguntado DIRECTAMENTE, sin entrenar
(29-09). Ni todas las tareas tienen las seis medidas, y eso se declara con una CLAVE de
motivo, nunca con un hueco silencioso:

  - (3) NUNCA se midió para C ni D: `no_medida_por_coste` (el propio registro sellado de
    (3) ya lo dice, y aquí se comprueba contra ese texto, no se repite a mano).
  - (4) NUNCA se manda a B: son casos clínicos, decisión de Roberto -- por eso
    `medir_c30_jev.py` se niega con su propio motivo si alguien lo intenta.
    `nunca_con_jev_datos_clinicos`.
  - (4) nunca estuvo en el alcance de D (solo A y C, "públicas y no clínicas", declarado en
    la cabecera de `medir_c30_jev.py`): `fuera_del_registro`.
  - (4) de C se recoge en cuanto `resultado_c30_jev_C.json` existe (ya medido, 29-09); mientras
    no exista, se declara `pendiente` en vez de un hueco silencioso -- el mecanismo, no un
    estado fijo de una fecha concreta.
  - (5) («Jev directo, sin entrenar», registro sellado del 29-09) solo corre en A y D --
    públicas y binarias: `noul` es una probabilidad 0-1, no un número continuo, así que C
    (regresión) nunca lo recibe: `regresion_sin_numero_continuo`. B, igual que en (4): nunca
    datos clínicos, `nunca_con_jev_datos_clinicos`.

**Lo que se COPIA y lo que se CALCULA.** El `metrica_puntual` de cada condición, y el
veredicto/diferencia/intervalo de cada comparación frente a (0): copiados tal cual de la
fuente. Las diferencias DESCRITAS -- (4) frente a (2), (4) frente a (3), (3) frente a (2) --
se calculan aquí como resta simple de dos `metrica_puntual` ya copiados; donde la fuente ya
escribió esa misma resta (4 vs 2 en el registro de (4), 3 vs 2 en el de (3)), se COMPRUEBA
que coincide antes de publicar (nunca un segundo número). (4) frente a (3) no está en
ningún registro -- ninguna fuente compara Jev con el respondedor local directamente -- así
que se calcula sin nada que comprobar, siempre a partir de los dos `metrica_puntual` que sí
son de la fuente.

**El acuerdo pregunta a pregunta (tarea A).** Las respuestas de Jev y del respondedor local
se guardan en cachés separadas, con esquemas de clave DISTINTOS: la del respondedor local
lleva el prefijo `"A\\x1f"` (tarea + separador de unidad) y la de Jev no. Se compara sobre
las claves COMUNES tras quitar ese prefijo -- comprobado aquí que las 7715 de una encajan
con las 7715 de la otra tras quitarlo, no supuesto.

Salida determinista (claves ordenadas, sin ninguna fecha de "ahora" propia: las fechas que
aparecen son las que cada fuente ya llevaba escritas) para que la comparación byte a byte
de la prueba sea estable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

AQUI = Path(__file__).resolve().parent
TEXTO_107C30 = AQUI.parent / "texto_107c30"

RUTA_RESULTADO_C30 = TEXTO_107C30 / "resultado_c30.json"
RUTA_RESUMEN_TAREAS = TEXTO_107C30 / "resumen_tareas.json"
RUTA_MARCADOR_NUMBER_A = TEXTO_107C30 / "marcador_number_A.json"
#: LAS PASADAS DE JEV Y SU COSTE (29-09, `registrar_pasadas_jev.py`). Existe porque el resultado de una
#: tarea guarda los segundos de SU ejecución: la C necesitó cuatro (corte de red, saldo, tope de la cola y
#: el resto desde la caché), y su resultado dice 1.122 s, que es solo la última. Con más de una pasada, lo
#: que se publica es el total de pared y la lista, nunca `segundos_preguntando` como si fuera el de todo.
RUTA_PASADAS_JEV = TEXTO_107C30 / "pasadas_c30_jev.json"

#: condición (5), «Jev directo, sin entrenar» (registro sellado 29-09, `medir_c30_jev_directo.py`):
#: UN fichero para las dos tareas candidatas (A y D), como escribe el propio medidor.
RUTA_JEV_DIRECTO = TEXTO_107C30 / "resultado_c30_jev_directo.json"

#: condición (3), respondedor local: solo A y B están medidas (los ficheros existen).
RUTAS_RESPONDEDOR = {
    "A": TEXTO_107C30 / "resultado_c30_respondedor_a.json",
    "B": TEXTO_107C30 / "resultado_c30_respondedor_b.json",
}
RUTAS_RESPUESTAS_RESPONDEDOR = {
    "A": TEXTO_107C30 / "respuestas_c30_respondedor_a.json",
    "B": TEXTO_107C30 / "respuestas_c30_respondedor_b.json",
}

#: condición (4), Jev: SOLO A y C son candidatas -- lo dice la propia cabecera de
#: `medir_c30_jev.py` ("Solo las tareas A ... y C ... -- públicas y no clínicas"). B nunca
#: se manda (dato clínico); D no está en el registro en absoluto.
TAREAS_CANDIDATAS_A_JEV = ("A", "C")

MOTIVO_NO_MEDIDA_POR_COSTE = "no_medida_por_coste"
MOTIVO_NUNCA_CON_JEV_DATOS_CLINICOS = "nunca_con_jev_datos_clinicos"
MOTIVO_FUERA_DEL_REGISTRO = "fuera_del_registro"
MOTIVO_PENDIENTE = "pendiente"
#: (5) en C: Jev (`noul`) da una probabilidad 0-1, nunca un número continuo -- C es regresión.
MOTIVO_REGRESION_SIN_NUMERO_CONTINUO = "regresion_sin_numero_continuo"

MOTIVO_4_POR_TAREA_NO_CANDIDATA = {
    "B": MOTIVO_NUNCA_CON_JEV_DATOS_CLINICOS,
    "D": MOTIVO_FUERA_DEL_REGISTRO,
}

#: (5), «Jev directo, sin entrenar»: SOLO A (noticias falsas) y D (BOE, vigencia) -- públicas
#: y binarias (registro sellado 29-09, cabecera de `medir_c30_jev_directo.py`).
TAREAS_CANDIDATAS_A_JEV_DIRECTO = ("A", "D")
MOTIVO_5_POR_TAREA_NO_CANDIDATA = {
    "B": MOTIVO_NUNCA_CON_JEV_DATOS_CLINICOS,
    "C": MOTIVO_REGRESION_SIN_NUMERO_CONTINUO,
}

TAREAS = ("A", "B", "C", "D")

#: auroc: mayor es mejor (probabilidad de ordenar bien un par positivo/negativo).
#: rmse: menor es mejor (error). Cualquier otra métrica es una sorpresa: se para.
DIRECCION_METRICA = {"auroc": True, "rmse": False}

#: Solo redondeo de coma flotante: la diferencia es una resta de dos números ya copiados.
TOLERANCIA = 1e-9

#: La clave del respondedor local en `respuestas_c30_respondedor_<t>.json` es
#: "<TAREA>\x1f<row_id>\x1f<termino>"; la de Jev en `respuestas_c30_jev_<T>.json` es
#: "<row_id>\x1f<termino>", sin ese prefijo. Medido, no supuesto (las dos cachés de A).
SEPARADOR_CLAVE_RESPUESTAS = "\x1f"


class DatosQueNoCuadran(RuntimeError):
    """Lo recalculado aquí no coincide con lo que escribió el registro sellado: no se publica."""


def _sha256_de(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _cargar(ruta: Path) -> dict[str, Any]:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _procedencia_de(ruta: Path, datos: dict[str, Any] | None) -> dict[str, Any]:
    """sha256 del fichero (calculado aquí) + `script_sha256`/`generado` que YA traía."""
    entrada: dict[str, Any] = {"sha256": _sha256_de(ruta)}
    if datos is not None:
        if "script_sha256" in datos:
            entrada["script_sha256"] = datos["script_sha256"]
        if "generado" in datos:
            entrada["generado"] = datos["generado"]
    return entrada


def _condicion_3(tarea: str) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Devuelve (bloque de condición 3 para publicar, JSON completo de la fuente o None,
    bloque de esa tarea dentro de esa fuente o None)."""
    if tarea in RUTAS_RESPONDEDOR:
        datos = _cargar(RUTAS_RESPONDEDOR[tarea])
        bloque_tarea = datos["tareas"][tarea]
        if bloque_tarea.get("omitida"):
            raise DatosQueNoCuadran(f"{tarea}: se esperaba condición 3 medida en "
                                     f"{RUTAS_RESPONDEDOR[tarea].name} y está omitida")
        condicion = bloque_tarea["condiciones"]["3_respondedor"]
        return {"metrica_puntual": condicion["metrica_puntual"]}, datos, bloque_tarea

    # No medida: comprobar que AL MENOS una de las dos fuentes que sí existen la declara
    # omitida por coste (nunca se repite el motivo a mano sin comprobarlo contra la fuente).
    for ruta in RUTAS_RESPONDEDOR.values():
        datos_ref = _cargar(ruta)
        bloque = datos_ref["tareas"].get(tarea)
        if bloque and bloque.get("omitida"):
            if "no medida por coste" not in bloque.get("motivo", ""):
                raise DatosQueNoCuadran(f"{tarea}: motivo de omisión de la condición 3 en "
                                         f"{ruta.name} no es el esperado: {bloque.get('motivo')!r}")
            return {"omitida": True, "motivo": MOTIVO_NO_MEDIDA_POR_COSTE}, None, None
    raise DatosQueNoCuadran(f"{tarea}: la condición 3 no está medida NI declarada omitida "
                             f"en ninguna fuente disponible")


def _condicion_4(tarea: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Devuelve (bloque de condición 4 para publicar, JSON completo de la fuente o None)."""
    if tarea not in TAREAS_CANDIDATAS_A_JEV:
        return {"omitida": True, "motivo": MOTIVO_4_POR_TAREA_NO_CANDIDATA[tarea]}, None
    ruta = TEXTO_107C30 / f"resultado_c30_jev_{tarea}.json"
    if not ruta.exists():
        # Sigue midiéndose (caso de C mientras este corte se escribe): declarado, no un
        # hueco. El día que el fichero aparezca, esta misma rama lo recoge sin tocar nada.
        return {"omitida": True, "motivo": MOTIVO_PENDIENTE}, None
    datos = _cargar(ruta)
    if datos.get("tarea") != tarea:
        raise DatosQueNoCuadran(f"{ruta.name} declara tarea={datos.get('tarea')!r}, se esperaba {tarea!r}")
    condicion = datos["condiciones"]["4_jev"]
    return {"metrica_puntual": condicion["metrica_puntual"]}, datos


def _condicion_5(tarea: str, datos_jev_directo: dict[str, Any] | None
                  ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Devuelve (bloque de condición 5 para publicar, bloque de esa tarea dentro de
    `resultado_c30_jev_directo.json`, o None si la tarea no es candidata a (5))."""
    if tarea not in TAREAS_CANDIDATAS_A_JEV_DIRECTO:
        return {"omitida": True, "motivo": MOTIVO_5_POR_TAREA_NO_CANDIDATA[tarea]}, None
    if datos_jev_directo is None:
        raise DatosQueNoCuadran(f"{tarea}: falta {RUTA_JEV_DIRECTO.name} con la condición 5 medida")
    bloque_tarea = datos_jev_directo.get("tareas", {}).get(tarea)
    if bloque_tarea is None:
        raise DatosQueNoCuadran(f"{tarea}: {RUTA_JEV_DIRECTO.name} no trae esta tarea")
    if bloque_tarea.get("condicion") != "5_jev_directo":
        raise DatosQueNoCuadran(f"{tarea}: {RUTA_JEV_DIRECTO.name} declara condicion="
                                 f"{bloque_tarea.get('condicion')!r}, se esperaba '5_jev_directo'")
    condicion = bloque_tarea["condiciones"]["5_jev_directo"]
    return {"metrica_puntual": condicion["metrica_puntual"]}, bloque_tarea


def _bloque_jev_directo(bloque_tarea: dict[str, Any], modelo: str) -> dict[str, Any]:
    """El bloque de estadísticas de (5) para UNA tarea: la pregunta LITERAL (del registro
    sellado) y los faltantes, contados y declarados -- nunca rellenados."""
    respuestas = bloque_tarea["respuestas"]
    if respuestas["n_con_respuesta"] + respuestas["n_faltante"] != respuestas["n_total"]:
        raise DatosQueNoCuadran(
            f"{bloque_tarea.get('tarea')!r}: n_con_respuesta({respuestas['n_con_respuesta']}) + "
            f"n_faltante({respuestas['n_faltante']}) != n_total({respuestas['n_total']}) en la condición 5")
    return {
        "modelo": modelo,
        "pregunta": bloque_tarea["pregunta"],
        "n_total": respuestas["n_total"],
        "n_con_respuesta": respuestas["n_con_respuesta"],
        "n_faltante": respuestas["n_faltante"],
        "proporcion_faltante": respuestas["proporcion_faltante"],
        "segundos_preguntando": respuestas["segundos"],
    }


def _punto(condiciones: dict[str, Any], clave: str) -> float | None:
    return condiciones[clave].get("metrica_puntual")


def _diferencia_descrita(m_x: float, m_y: float, fuente_bloque: dict[str, Any] | None,
                          clave_diferencia_en_fuente: str | None) -> dict[str, Any]:
    diferencia = m_x - m_y
    resultado: dict[str, Any] = {"diferencia": diferencia}
    if fuente_bloque is not None:
        declarada = fuente_bloque[clave_diferencia_en_fuente]
        if abs(declarada - diferencia) > TOLERANCIA:
            raise DatosQueNoCuadran(
                f"la diferencia recalculada ({diferencia}) no coincide con la que la fuente "
                f"ya escribió en {clave_diferencia_en_fuente!r} ({declarada})")
        if "nota" in fuente_bloque:
            resultado["nota"] = fuente_bloque["nota"]  # copiada, no redactada aquí
    return resultado


def _jev_de_las_respuestas(tarea: str, respuestas: dict[str, Any]) -> dict[str, int]:
    """Quita el prefijo `"<tarea>\\x1f"` de las claves del respondedor local, para que
    encajen con las de Jev (que no lo llevan). Comprobado: TODAS tienen que llevarlo."""
    prefijo = f"{tarea}{SEPARADOR_CLAVE_RESPUESTAS}"
    sin_prefijo = {}
    for clave, valor in respuestas.items():
        if not clave.startswith(prefijo):
            raise DatosQueNoCuadran(f"respuestas del respondedor local de {tarea}: clave sin "
                                     f"el prefijo esperado {prefijo!r}: {clave!r}")
        sin_prefijo[clave[len(prefijo):]] = valor
    return sin_prefijo


def _bloque_preguntas(modelo: str, tiempos_preguntas: dict[str, Any], n_peticiones: int) -> dict[str, Any]:
    n_si, n_no, n_faltante = tiempos_preguntas["n_si"], tiempos_preguntas["n_no"], tiempos_preguntas["n_faltante"]
    n_respuestas = tiempos_preguntas["n_respuestas"]
    if n_si + n_no + n_faltante != n_respuestas:
        raise DatosQueNoCuadran(f"n_si({n_si}) + n_no({n_no}) + n_faltante({n_faltante}) != "
                                 f"n_respuestas({n_respuestas}) para {modelo!r}")
    return {
        "modelo": modelo,
        "n_peticiones_al_modelo": n_peticiones,
        "n_respuestas": n_respuestas,
        "n_faltante": n_faltante,
        "proporcion_faltante": tiempos_preguntas["proporcion_faltante"],
        "segundos_preguntando": tiempos_preguntas["segundos"],
        "proporcion_si": n_si / n_respuestas,
    }


def _componer_tarea(tarea: str, c30: dict[str, Any], resumen: dict[str, Any],
                     marcador_a: dict[str, Any] | None,
                     pasadas_jev: dict[str, Any] | None,
                     datos_jev_directo: dict[str, Any] | None,
                     ficheros_leidos: dict[Path, dict[str, Any] | None]) -> dict[str, Any]:
    bloque_c30 = c30["tareas"][tarea]
    bloque_resumen = resumen["tareas"][tarea]
    metrica = bloque_c30["metrica"]
    if metrica not in DIRECCION_METRICA:
        raise DatosQueNoCuadran(f"{tarea}: métrica desconocida {metrica!r}")

    filas_por_particion = {"train": bloque_c30["n_train"], "dev": bloque_c30["n_dev"], "test": bloque_c30["n_test"]}
    if filas_por_particion != bloque_resumen["n_por_particion"]:
        raise DatosQueNoCuadran(f"{tarea}: filas por partición no coinciden entre "
                                 f"resultado_c30.json y resumen_tareas.json")

    condiciones: dict[str, Any] = {}
    for clave in ("0_sin_texto", "1_embedding", "2_tfidf_lineal"):
        condiciones[clave] = {"metrica_puntual": bloque_c30["condiciones"][clave]["metrica_puntual"]}

    condicion_3, datos_resp, bloque_resp_tarea = _condicion_3(tarea)
    condiciones["3_respondedor"] = condicion_3
    if datos_resp is not None:
        ficheros_leidos[RUTAS_RESPONDEDOR[tarea]] = datos_resp

    condicion_4, datos_jev = _condicion_4(tarea)
    condiciones["4_jev"] = condicion_4
    if datos_jev is not None:
        ficheros_leidos[TEXTO_107C30 / f"resultado_c30_jev_{tarea}.json"] = datos_jev

    condicion_5, bloque_5_tarea = _condicion_5(tarea, datos_jev_directo)
    condiciones["5_jev_directo"] = condicion_5

    # -- (5) recalcula (0) para su propio emparejamiento: comprobado contra la (0) ya
    #    publicada de resultado_c30.json, no supuesto -- mismo patrón que el resto de esta
    #    función, que nunca deja que dos fuentes se contradigan en silencio.
    if bloque_5_tarea is not None:
        m0_c30 = bloque_c30["condiciones"]["0_sin_texto"]["metrica_puntual"]
        m0_recalc = bloque_5_tarea["condiciones"]["0_sin_texto_recalculada"]["metrica_puntual_muestra_completa"]
        if abs(m0_recalc - m0_c30) > TOLERANCIA:
            raise DatosQueNoCuadran(
                f"{tarea}: (0) recalculada en {RUTA_JEV_DIRECTO.name} ({m0_recalc}) no "
                f"coincide con la publicada en resultado_c30.json ({m0_c30})")

    # -- comparaciones frente a (0), copiadas de la fuente que midió cada condición --------
    comparaciones: dict[str, Any] = {}
    for clave, n in (("1_embedding", "1"), ("2_tfidf_lineal", "2")):
        comparaciones[clave] = dict(bloque_c30["comparaciones"][f"{n}_vs_0"])
    if "metrica_puntual" in condicion_3:
        comparaciones["3_respondedor"] = dict(bloque_resp_tarea["comparaciones"]["3_vs_0"])
    if "metrica_puntual" in condicion_4:
        comparaciones["4_jev"] = dict(datos_jev["comparaciones"]["4_vs_0"])
    if "metrica_puntual" in condicion_5:
        comparaciones["5_jev_directo"] = dict(bloque_5_tarea["comparaciones"]["5_vs_0"])

    # -- diferencias DESCRITAS, sin veredicto -----------------------------------------------
    m1 = _punto(condiciones, "1_embedding")
    m2 = _punto(condiciones, "2_tfidf_lineal")
    m3 = _punto(condiciones, "3_respondedor")
    m4 = _punto(condiciones, "4_jev")
    m5 = _punto(condiciones, "5_jev_directo")
    diferencias_descritas: dict[str, Any] = {}
    if m3 is not None:
        fuente = bloque_resp_tarea["comparaciones"].get("3_vs_2_diferencia_descrita")
        diferencias_descritas["3_respondedor_menos_2_tfidf_lineal"] = _diferencia_descrita(
            m3, m2, fuente, "diferencia_3_menos_2")
    if m4 is not None:
        fuente = datos_jev["comparaciones"].get("4_vs_2_diferencia_descrita")
        diferencias_descritas["4_jev_menos_2_tfidf_lineal"] = _diferencia_descrita(
            m4, m2, fuente, "diferencia_4_menos_2")
    if m3 is not None and m4 is not None:
        # Ninguna fuente compara Jev con el respondedor local directamente: nada que
        # comprobar, se calcula entero a partir de los dos `metrica_puntual` ya copiados.
        diferencias_descritas["4_jev_menos_3_respondedor"] = _diferencia_descrita(m4, m3, None, None)
    if m5 is not None:
        fuente_1 = bloque_5_tarea["comparaciones"].get("5_vs_1_diferencia_descrita")
        diferencias_descritas["5_jev_directo_menos_1_embedding"] = _diferencia_descrita(
            m5, m1, fuente_1, "diferencia_5_menos_1")
        fuente_2 = bloque_5_tarea["comparaciones"].get("5_vs_2_diferencia_descrita")
        diferencias_descritas["5_jev_directo_menos_2_tfidf_lineal"] = _diferencia_descrita(
            m5, m2, fuente_2, "diferencia_5_menos_2")
        if m4 is not None:
            fuente_4 = bloque_5_tarea["comparaciones"].get("5_vs_4_diferencia_descrita")
            diferencias_descritas["5_jev_directo_menos_4_jev"] = _diferencia_descrita(
                m5, m4, fuente_4, "diferencia_5_menos_4")

    # -- términos de las preguntas: de la condición que exista (3 o 4), comprobados entre
    #    sí si las dos existen (la tarea A los tiene por los dos caminos) ------------------
    terminos = None
    if bloque_resp_tarea is not None and "terminos" in bloque_resp_tarea:
        terminos = bloque_resp_tarea["terminos"]
    if datos_jev is not None and "terminos" in datos_jev:
        if terminos is not None and terminos != datos_jev["terminos"]:
            raise DatosQueNoCuadran(f"{tarea}: los términos de la condición 3 y los de la "
                                     f"condición 4 no coinciden")
        terminos = datos_jev["terminos"]

    resultado: dict[str, Any] = {
        "nombre": bloque_resumen["nombre"],
        "metrica": metrica,
        "mayor_es_mejor": DIRECCION_METRICA[metrica],
        "filas_por_particion": filas_por_particion,
        "condiciones": condiciones,
        "comparaciones_frente_a_0": comparaciones,
        "diferencias_descritas": diferencias_descritas,
    }
    if terminos is not None:
        resultado["terminos"] = terminos

    # -- bloque `jev`: solo para tareas donde Jev se midió de verdad ------------------------
    if datos_jev is not None:
        tiempos_jev = datos_jev["condiciones"]["4_jev"]["tiempos"]["preguntas_al_modelo"]
        bloque_jev = _bloque_preguntas(datos_jev["modelo"], tiempos_jev, tiempos_jev["n_peticiones_maximo"])

        if tarea in RUTAS_RESPUESTAS_RESPONDEDOR and datos_resp is not None:
            tiempos_resp = bloque_resp_tarea["condiciones"]["3_respondedor"]["tiempos"]["preguntas_al_modelo"]
            # el respondedor local no hace batching: una petición por pregunta.
            bloque_jev["respondedor_local"] = _bloque_preguntas(
                datos_resp["modelo"]["candidato"], tiempos_resp, tiempos_resp["n_respuestas"])

            ruta_resp_jev = TEXTO_107C30 / f"respuestas_c30_jev_{tarea}.json"
            ruta_resp_local = RUTAS_RESPUESTAS_RESPONDEDOR[tarea]
            respuestas_jev = _cargar(ruta_resp_jev)
            respuestas_local = _jev_de_las_respuestas(tarea, _cargar(ruta_resp_local))
            ficheros_leidos[ruta_resp_jev] = None
            ficheros_leidos[ruta_resp_local] = None

            comunes = set(respuestas_jev) & set(respuestas_local)
            if not comunes:
                raise DatosQueNoCuadran(f"{tarea}: ninguna clave común entre las respuestas "
                                         f"de Jev y las del respondedor local")
            n_coincidencias = sum(1 for k in comunes
                                   if respuestas_jev[k]["valor"] == respuestas_local[k]["valor"])
            bloque_jev["acuerdo_con_respondedor_local"] = {
                "n_comparadas": len(comunes),
                "n_coincidencias": n_coincidencias,
                "proporcion": n_coincidencias / len(comunes),
            }
        if pasadas_jev is not None and tarea in pasadas_jev["por_tarea"]:
            registro = pasadas_jev["por_tarea"][tarea]
            lista = [{"nombre": p["nombre"], "desenlace": p["desenlace"],
                      "segundos_de_pared": p["segundos_de_pared"],
                      **({"causa": p["causa"]} if p.get("causa") else {})} for p in registro["pasadas"]]
            total = sum(p["segundos_de_pared"] for p in lista)
            if abs(total - registro["segundos_de_pared_total"]) > 1e-6:
                raise DatosQueNoCuadran(f"{tarea}: el total de pasadas no es la suma de sus pasadas")
            if not any(p["desenlace"] == "completada" for p in lista):
                raise DatosQueNoCuadran(f"{tarea}: ninguna pasada completada y hay resultado")
            bloque_jev["pasadas"] = lista
            bloque_jev["segundos_de_pared_total"] = total
            if len(lista) > 1:
                # Los segundos del resultado son los de la ÚLTIMA pasada: fuera, y se dice por qué.
                del bloque_jev["segundos_preguntando"]
                bloque_jev["reanudada_desde_la_cache"] = True
        resultado["jev"] = bloque_jev

    # -- bloque `jev_directo`: condición (5), solo para A y D --------------------------------
    if bloque_5_tarea is not None:
        resultado["jev_directo"] = _bloque_jev_directo(bloque_5_tarea, datos_jev_directo["modelo"])

    if tarea == "A":
        if marcador_a is None:
            raise DatosQueNoCuadran("falta marcador_number_A.json para la tarea A")
        if marcador_a.get("tarea") != "A":
            raise DatosQueNoCuadran(f"marcador_number_A.json declara tarea={marcador_a.get('tarea')!r}")
        resultado["artefactos"] = {"marcador_number": dict(marcador_a)}

    return resultado


def generar() -> str:
    c30 = _cargar(RUTA_RESULTADO_C30)
    resumen = _cargar(RUTA_RESUMEN_TAREAS)
    marcador_a = _cargar(RUTA_MARCADOR_NUMBER_A) if RUTA_MARCADOR_NUMBER_A.exists() else None
    pasadas_jev = _cargar(RUTA_PASADAS_JEV) if RUTA_PASADAS_JEV.exists() else None
    datos_jev_directo = _cargar(RUTA_JEV_DIRECTO) if RUTA_JEV_DIRECTO.exists() else None

    ficheros_leidos: dict[Path, dict[str, Any] | None] = {
        RUTA_RESULTADO_C30: c30,
        RUTA_RESUMEN_TAREAS: resumen,
    }
    if marcador_a is not None:
        ficheros_leidos[RUTA_MARCADOR_NUMBER_A] = marcador_a
    if pasadas_jev is not None:
        ficheros_leidos[RUTA_PASADAS_JEV] = pasadas_jev
    if datos_jev_directo is not None:
        ficheros_leidos[RUTA_JEV_DIRECTO] = datos_jev_directo

    tareas = {}
    for tarea in TAREAS:
        tareas[tarea] = _componer_tarea(tarea, c30, resumen, marcador_a if tarea == "A" else None,
                                        pasadas_jev, datos_jev_directo, ficheros_leidos)

    procedencia = {ruta.name: _procedencia_de(ruta, datos)
                   for ruta, datos in sorted(ficheros_leidos.items(), key=lambda kv: kv[0].name)}
    procedencia["generar_texto_107_script_sha256"] = _sha256_de(Path(__file__))

    salida = {"formato": "107-C30.v1", "tareas": tareas, "procedencia": procedencia}
    if pasadas_jev is not None:
        salida["coste_jev"] = {"uso_de_la_clave_usd": pasadas_jev["coste"]["uso_de_la_clave_usd"],
                               "cubre": "A_C_y_llamadas_de_prueba"}
    return serializar(salida)


def serializar(datos: dict[str, Any]) -> str:
    """UNA forma de escribirlo, para que la comparación byte a byte signifique algo."""
    return json.dumps(datos, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


RUTA_PUBLICADO = AQUI / "texto_107_publico.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escribir", action="store_true", help="regenera el JSON publicado")
    args = parser.parse_args(argv)
    texto = generar()
    if args.escribir:
        RUTA_PUBLICADO.write_text(texto, encoding="utf-8")
        print(f"escrito {RUTA_PUBLICADO} ({len(texto.encode('utf-8'))} bytes)")
        return 0
    publicado = RUTA_PUBLICADO.read_text(encoding="utf-8") if RUTA_PUBLICADO.exists() else None
    if publicado != texto:
        print("el JSON publicado NO es el que sale de los registros: regenerarlo con "
              "--escribir (y mirar por qué cambió)", file=sys.stderr)
        return 1
    print("el JSON publicado es el que sale de los registros")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
