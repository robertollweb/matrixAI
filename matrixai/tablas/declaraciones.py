# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Lo que la conversión HIZO, dicho en los dos idiomas (150-C1, invariante 3).

«Nada se quita, se renombra, se rellena ni se convierte sin quedar declarado (qué, dónde y
cuántas veces)»: cada declaración es `{tipo, columna, cuantas, detalle, texto: {es, en}}`. El
texto lo redacta el núcleo (invariante 9); la interfaz solo lo enseña.
"""
from __future__ import annotations

__all__ = ["declarar", "TIPOS"]


def _lista(v, n: int = 10) -> str:
    v = list(v or [])
    s = ", ".join(str(x) for x in v[:n])
    return s + (" …" if len(v) > n else "")


def _tipos(v) -> str:
    """`{"numero": 2, "texto": 1}` → «numero: 2, texto: 1»; una lista → sus elementos."""
    if isinstance(v, dict):
        return _lista(f"{k}: {n}" for k, n in v.items())
    return _lista(v)


def _col(d: dict) -> str:
    return d.get("columna") or ""


# tipo → (función que da el texto en español, función que da el texto en inglés)
TIPOS: dict[str, tuple] = {
    "columnas_vacias_quitadas": (
        lambda d: f"Se quitaron {d['cuantas']} columnas totalmente vacías ({_lista(d['detalle'].get('letras'))}).",
        lambda d: f"{d['cuantas']} completely empty columns were removed ({_lista(d['detalle'].get('letras'))})."),
    "nombre_repetido_renombrado": (
        lambda d: f"La columna {d['detalle'].get('letra')} se llamaba «{d['detalle'].get('original')}», como otra: ahora es «{_col(d)}».",
        lambda d: f"Column {d['detalle'].get('letra')} was called «{d['detalle'].get('original')}», like another one: it is now «{_col(d)}»."),
    "nombre_vacio_con_datos": (
        lambda d: f"La columna {d['detalle'].get('letra')} tiene datos pero no nombre: ahora es «{_col(d)}».",
        lambda d: f"Column {d['detalle'].get('letra')} has data but no name: it is now «{_col(d)}»."),
    "nombre_normalizado": (
        lambda d: f"El nombre «{d['detalle'].get('original')}» se escribió «{_col(d)}» (espacios sobrantes quitados).",
        lambda d: f"The name «{d['detalle'].get('original')}» was written as «{_col(d)}» (extra spaces removed)."),
    "cabecera_compuesta": (
        lambda d: f"La cabecera de dos filas se unió en un solo nombre: «{_col(d)}» (grupo «{d['detalle'].get('grupo')}», subcolumna «{d['detalle'].get('sub')}»).",
        lambda d: f"The two-row header was joined into a single name: «{_col(d)}» (group «{d['detalle'].get('grupo')}», sub-column «{d['detalle'].get('sub')}»)."),
    "filas_previas_omitidas": (
        lambda d: f"Las {d['cuantas']} filas por encima de la cabecera no entran en la tabla ({_lista(d['detalle'].get('textos'), 3)}).",
        lambda d: f"The {d['cuantas']} rows above the header are not part of the table ({_lista(d['detalle'].get('textos'), 3)})."),
    "filas_excluidas": (
        lambda d: f"Se quitaron {d['cuantas']} filas que elegiste (filas {_lista(d['detalle'].get('filas'))}).",
        lambda d: f"{d['cuantas']} rows you chose were removed (rows {_lista(d['detalle'].get('filas'))})."),
    "filas_vacias_quitadas": (
        lambda d: f"Se quitaron {d['cuantas']} filas totalmente vacías (filas {_lista(d['detalle'].get('filas'))}).",
        lambda d: f"{d['cuantas']} completely empty rows were removed (rows {_lista(d['detalle'].get('filas'))})."),
    "celdas_combinadas_vacias": (
        lambda d: f"En «{_col(d)}», {d['cuantas']} celdas cubiertas por una combinación quedan vacías (rangos {_lista(d['detalle'].get('rangos'))}).",
        lambda d: f"In «{_col(d)}», {d['cuantas']} cells covered by a merge are left empty (ranges {_lista(d['detalle'].get('rangos'))})."),
    "combinadas_rellenadas": (
        lambda d: f"En «{_col(d)}», {d['cuantas']} celdas cubiertas por una combinación se rellenaron con el valor de la celda combinada (rangos {_lista(d['detalle'].get('rangos'))}).",
        lambda d: f"In «{_col(d)}», {d['cuantas']} cells covered by a merge were filled with the merged cell's value (ranges {_lista(d['detalle'].get('rangos'))})."),
    "errores_de_celda": (
        lambda d: f"En «{_col(d)}», {d['cuantas']} celdas con error de Excel ({_lista(d['detalle'].get('tipos'))}) quedan vacías.",
        lambda d: f"In «{_col(d)}», {d['cuantas']} cells with an Excel error ({_lista(d['detalle'].get('tipos'))}) are left empty."),
    "no_finito": (
        lambda d: f"En «{_col(d)}», {d['cuantas']} valores no finitos (NaN o infinito) quedan vacíos.",
        lambda d: f"In «{_col(d)}», {d['cuantas']} non-finite values (NaN or infinity) are left empty."),
    "fechas_a_medianoche": (
        lambda d: f"En «{_col(d)}», {d['cuantas']} fechas sin hora se escribieron a las 00:00:00 porque otras celdas de la columna llevan hora.",
        lambda d: f"In «{_col(d)}», {d['cuantas']} dates without a time were written at 00:00:00 because other cells in the column have a time."),
    "fecha_truncada": (
        lambda d: f"En «{_col(d)}», {d['cuantas']} fechas u horas llevaban fracciones de segundo: se truncaron a segundos.",
        lambda d: f"In «{_col(d)}», {d['cuantas']} dates or times had fractions of a second: they were truncated to seconds."),
    "hora_del_dia": (
        lambda d: f"«{_col(d)}» es una hora del día: {d['cuantas']} valores escritos como HH:MM:SS.",
        lambda d: f"«{_col(d)}» is a time of day: {d['cuantas']} values written as HH:MM:SS."),
    "duracion": (
        lambda d: f"«{_col(d)}» es una duración: {d['cuantas']} valores escritos como HH:MM:SS (las horas pueden pasar de 24).",
        lambda d: f"«{_col(d)}» is a duration: {d['cuantas']} values written as HH:MM:SS (hours can exceed 24)."),
    "formato_porcentaje": (
        lambda d: f"«{_col(d)}» tiene formato de porcentaje en {d['cuantas']} celdas: se guarda el valor (0,153 es 15,3 %), sin el símbolo.",
        lambda d: f"«{_col(d)}» has percentage format in {d['cuantas']} cells: the stored value is kept (0.153 is 15.3 %), without the symbol."),
    "columna_mixta": (
        lambda d: f"«{_col(d)}» mezcla tipos de valor ({_tipos(d['detalle'].get('tipos'))}).",
        lambda d: f"«{_col(d)}» mixes value types ({_tipos(d['detalle'].get('tipos'))})."),
    "posible_fecha_en_serie": (
        lambda d: f"«{_col(d)}» son números entre 20.000 y 60.000 sin formato de fecha: ¿son fechas? No se han convertido.",
        lambda d: f"«{_col(d)}» holds numbers between 20,000 and 60,000 with no date format: are they dates? They were not converted."),
    "filas_ocultas": (
        lambda d: f"{d['cuantas']} filas de la tabla están ocultas en Excel y se han incluido igualmente (filas {_lista(d['detalle'].get('filas'))}).",
        lambda d: f"{d['cuantas']} rows of the table are hidden in Excel and were included anyway (rows {_lista(d['detalle'].get('filas'))})."),
    "columnas_ocultas": (
        lambda d: f"{d['cuantas']} columnas de la tabla están ocultas en Excel y se han incluido igualmente ({_lista(d['detalle'].get('letras'))}).",
        lambda d: f"{d['cuantas']} columns of the table are hidden in Excel and were included anyway ({_lista(d['detalle'].get('letras'))})."),
    "hoja_oculta": (
        lambda d: f"La hoja «{d['detalle'].get('hoja')}» está oculta en el libro.",
        lambda d: f"Sheet «{d['detalle'].get('hoja')}» is hidden in the workbook."),
    "macros_ignoradas": (
        lambda d: "El libro lleva macros (.xlsm): se ignoran, no se ejecuta nada.",
        lambda d: "The workbook contains macros (.xlsm): they are ignored, nothing is run."),
    "no_comprobable_en_xls": (
        lambda d: f"En un .xls antiguo no se puede comprobar: {_lista(d['detalle'].get('que'))}.",
        lambda d: f"In an old .xls file this cannot be checked: {_lista(d['detalle'].get('que'))}."),
    "hay_celdas_combinadas": (
        lambda d: f"La hoja tiene {d['cuantas']} rangos de celdas combinadas ({_lista(d['detalle'].get('rangos'))}): las celdas cubiertas llegan vacías.",
        lambda d: f"The sheet has {d['cuantas']} merged cell ranges ({_lista(d['detalle'].get('rangos'))}): the covered cells arrive empty."),
    "formulas_sin_valor": (
        lambda d: f"{d['cuantas']} celdas tienen fórmula pero ningún valor guardado ({_lista(d['detalle'].get('celdas'))}): esta hoja no se podrá convertir hasta que se guarde en Excel.",
        lambda d: f"{d['cuantas']} cells have a formula but no saved value ({_lista(d['detalle'].get('celdas'))}): this sheet cannot be converted until it is saved in Excel."),
    "posible_cabecera_de_dos_filas": (
        lambda d: f"La fila {d['detalle'].get('fila')} y la siguiente parecen una cabecera de dos filas (grupo y subcolumna).",
        lambda d: f"Row {d['detalle'].get('fila')} and the next one look like a two-row header (group and sub-column)."),
    "nombres_repetidos": (
        lambda d: f"La cabecera propuesta repite o deja sin nombre {d['cuantas']} columnas ({_lista(d['detalle'].get('nombres'))}): se renombrarán y se declarará.",
        lambda d: f"The proposed header repeats or leaves unnamed {d['cuantas']} columns ({_lista(d['detalle'].get('nombres'))}): they will be renamed and declared."),
    "hoja_sin_datos": (
        lambda d: "La hoja no tiene datos.",
        lambda d: "The sheet has no data."),
}


def declarar(tipo: str, *, columna: str | None = None, cuantas: int = 1, **detalle) -> dict:
    d = {"tipo": tipo, "columna": columna, "cuantas": int(cuantas), "detalle": detalle}
    es, en = TIPOS[tipo]
    d["texto"] = {"es": es(d), "en": en(d)}
    return d
