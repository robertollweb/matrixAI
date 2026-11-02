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


def _p(n: int, uno: str, varios: str) -> str:
    """Singular o plural según `n`: `uno` tal cual si n == 1; si no, `varios` con `{n}` puesto.
    NINGÚN texto dice «1 celdas» ni «1 cells» (`test_ningun_texto_dice_1_celdas…` los recorre todos)."""
    return uno if n == 1 else varios.replace("{n}", str(n))


def _col(d: dict) -> str:
    return d.get("columna") or ""


_MOTIVOS_FECHA = {
    "anterior_a_la_epoca": ("anterior al primer día de Excel; Excel enseña ####", "before Excel's first day; Excel shows ####"),
    "29_02_1900": ("el 29-02-1900, un día que no existió", "29-02-1900, a day that never existed"),
    "posterior_a_9999": ("posterior a 9999-12-31; Excel enseña ####", "after 9999-12-31; Excel shows ####"),
}


def _motivos_fecha(d: dict, idioma: str) -> str:
    i = 0 if idioma == "es" else 1
    return ", ".join(f"{_MOTIVOS_FECHA.get(k, (k, k))[i]}: {n}"
                     for k, n in sorted((d["detalle"].get("motivos") or {}).items()))


# tipo → (función que da el texto en español, función que da el texto en inglés)
TIPOS: dict[str, tuple] = {
    "columnas_vacias_quitadas": (
        lambda d: _p(d['cuantas'], f"Se quitó 1 columna totalmente vacía ({_lista(d['detalle'].get('letras'))}).",
                     "Se quitaron {n} columnas totalmente vacías (" + _lista(d['detalle'].get('letras')) + ")."),
        lambda d: _p(d['cuantas'], f"1 completely empty column was removed ({_lista(d['detalle'].get('letras'))}).",
                     "{n} completely empty columns were removed (" + _lista(d['detalle'].get('letras')) + ").")),
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
        lambda d: _p(d['cuantas'], f"La fila por encima de la cabecera no entra en la tabla ({_lista(d['detalle'].get('textos'), 3)}).",
                     "Las {n} filas por encima de la cabecera no entran en la tabla (" + _lista(d['detalle'].get('textos'), 3) + ")."),
        lambda d: _p(d['cuantas'], f"The row above the header is not part of the table ({_lista(d['detalle'].get('textos'), 3)}).",
                     "The {n} rows above the header are not part of the table (" + _lista(d['detalle'].get('textos'), 3) + ").")),
    "filas_excluidas": (
        lambda d: _p(d['cuantas'], f"Se quitó 1 fila que elegiste (fila {_lista(d['detalle'].get('filas'))}).",
                     "Se quitaron {n} filas que elegiste (filas " + _lista(d['detalle'].get('filas')) + ")."),
        lambda d: _p(d['cuantas'], f"1 row you chose was removed (row {_lista(d['detalle'].get('filas'))}).",
                     "{n} rows you chose were removed (rows " + _lista(d['detalle'].get('filas')) + ").")),
    "filas_vacias_quitadas": (
        lambda d: _p(d['cuantas'], f"Se quitó 1 fila totalmente vacía (fila {_lista(d['detalle'].get('filas'))}).",
                     "Se quitaron {n} filas totalmente vacías (filas " + _lista(d['detalle'].get('filas')) + ")."),
        lambda d: _p(d['cuantas'], f"1 completely empty row was removed (row {_lista(d['detalle'].get('filas'))}).",
                     "{n} completely empty rows were removed (rows " + _lista(d['detalle'].get('filas')) + ").")),
    "celdas_combinadas_vacias": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 celda cubierta por una combinación queda vacía (rango {_lista(d['detalle'].get('rangos'))}).",
                     f"En «{_col(d)}», " + "{n} celdas cubiertas por una combinación quedan vacías (rangos " + _lista(d['detalle'].get('rangos')) + ")."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 cell covered by a merge is left empty (range {_lista(d['detalle'].get('rangos'))}).",
                     f"In «{_col(d)}», " + "{n} cells covered by a merge are left empty (ranges " + _lista(d['detalle'].get('rangos')) + ").")),
    "combinadas_rellenadas": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 celda cubierta por una combinación se rellenó con el valor de la celda combinada (rango {_lista(d['detalle'].get('rangos'))}).",
                     f"En «{_col(d)}», " + "{n} celdas cubiertas por una combinación se rellenaron con el valor de la celda combinada (rangos " + _lista(d['detalle'].get('rangos')) + ")."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 cell covered by a merge was filled with the merged cell's value (range {_lista(d['detalle'].get('rangos'))}).",
                     f"In «{_col(d)}», " + "{n} cells covered by a merge were filled with the merged cell's value (ranges " + _lista(d['detalle'].get('rangos')) + ").")),
    "errores_de_celda": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 celda con error de Excel ({_lista(d['detalle'].get('tipos'))}) queda vacía.",
                     f"En «{_col(d)}», " + "{n} celdas con error de Excel (" + _lista(d['detalle'].get('tipos')) + ") quedan vacías."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 cell with an Excel error ({_lista(d['detalle'].get('tipos'))}) is left empty.",
                     f"In «{_col(d)}», " + "{n} cells with an Excel error (" + _lista(d['detalle'].get('tipos')) + ") are left empty.")),
    "no_finito": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 valor no finito (NaN o infinito) queda vacío.",
                     f"En «{_col(d)}», " + "{n} valores no finitos (NaN o infinito) quedan vacíos."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 non-finite value (NaN or infinity) is left empty.",
                     f"In «{_col(d)}», " + "{n} non-finite values (NaN or infinity) are left empty.")),
    "fechas_a_medianoche": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 fecha sin hora se escribió a las 00:00:00 porque otras celdas de la columna llevan hora.",
                     f"En «{_col(d)}», " + "{n} fechas sin hora se escribieron a las 00:00:00 porque otras celdas de la columna llevan hora."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 date without a time was written at 00:00:00 because other cells in the column have a time.",
                     f"In «{_col(d)}», " + "{n} dates without a time were written at 00:00:00 because other cells in the column have a time.")),
    "fecha_truncada": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 fecha u hora llevaba fracciones de segundo: se truncó a segundos.",
                     f"En «{_col(d)}», " + "{n} fechas u horas llevaban fracciones de segundo: se truncaron a segundos."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 date or time had fractions of a second: it was truncated to seconds.",
                     f"In «{_col(d)}», " + "{n} dates or times had fractions of a second: they were truncated to seconds.")),
    "hora_del_dia": (
        lambda d: _p(d['cuantas'], f"«{_col(d)}» es una hora del día: 1 valor escrito como HH:MM:SS.",
                     f"«{_col(d)}» es una hora del día: " + "{n} valores escritos como HH:MM:SS."),
        lambda d: _p(d['cuantas'], f"«{_col(d)}» is a time of day: 1 value written as HH:MM:SS.",
                     f"«{_col(d)}» is a time of day: " + "{n} values written as HH:MM:SS.")),
    "duracion": (
        lambda d: _p(d['cuantas'], f"«{_col(d)}» es una duración: 1 valor escrito como HH:MM:SS (las horas pueden pasar de 24).",
                     f"«{_col(d)}» es una duración: " + "{n} valores escritos como HH:MM:SS (las horas pueden pasar de 24)."),
        lambda d: _p(d['cuantas'], f"«{_col(d)}» is a duration: 1 value written as HH:MM:SS (hours can exceed 24).",
                     f"«{_col(d)}» is a duration: " + "{n} values written as HH:MM:SS (hours can exceed 24).")),
    "formato_porcentaje": (
        lambda d: _p(d['cuantas'], f"«{_col(d)}» tiene formato de porcentaje en 1 celda: se guarda el valor (0,153 es 15,3 %), sin el símbolo.",
                     f"«{_col(d)}» tiene formato de porcentaje en " + "{n} celdas: se guarda el valor (0,153 es 15,3 %), sin el símbolo."),
        lambda d: _p(d['cuantas'], f"«{_col(d)}» has percentage format in 1 cell: the stored value is kept (0.153 is 15.3 %), without the symbol.",
                     f"«{_col(d)}» has percentage format in " + "{n} cells: the stored value is kept (0.153 is 15.3 %), without the symbol.")),
    "formato_con_ceros": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 celda tiene un formato con ceros a la izquierda ({d['detalle'].get('codigo')}): Excel la enseña distinta (p. ej. {d['detalle'].get('ejemplo')} se ve {d['detalle'].get('visible')}) y el CSV lleva el valor guardado ({d['detalle'].get('ejemplo')}). Si es un código (postal, de cliente…), guarda la columna como texto en Excel.",
                     f"En «{_col(d)}», " + "{n} celdas tienen un formato con ceros a la izquierda (" + str(d['detalle'].get('codigo')) + f"): Excel las enseña distintas (p. ej. {d['detalle'].get('ejemplo')} se ve {d['detalle'].get('visible')}) y el CSV lleva el valor guardado ({d['detalle'].get('ejemplo')}). Si es un código (postal, de cliente…), guarda la columna como texto en Excel."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 cell has a leading-zeros format ({d['detalle'].get('codigo')}): Excel shows it differently (e.g. {d['detalle'].get('ejemplo')} shows as {d['detalle'].get('visible')}) and the CSV has the stored value ({d['detalle'].get('ejemplo')}). If it is a code (postal, customer…), store the column as text in Excel.",
                     f"In «{_col(d)}», " + "{n} cells have a leading-zeros format (" + str(d['detalle'].get('codigo')) + f"): Excel shows them differently (e.g. {d['detalle'].get('ejemplo')} shows as {d['detalle'].get('visible')}) and the CSV has the stored value ({d['detalle'].get('ejemplo')}). If it is a code (postal, customer…), store the column as text in Excel.")),
    "entero_no_exacto": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 número entero tiene más de 15 cifras: Excel no lo guarda exacto, así que el CSV lleva el valor del fichero ({_lista(d['detalle'].get('ejemplos'), 3)}), que puede no ser el que se tecleó. Si es un identificador, guárdalo como texto en Excel.",
                     f"En «{_col(d)}», " + "{n} números enteros tienen más de 15 cifras: Excel no los guarda exactos, así que el CSV lleva el valor del fichero (" + _lista(d['detalle'].get('ejemplos'), 3) + "), que puede no ser el que se tecleó. Si son identificadores, guárdalos como texto en Excel."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 whole number has more than 15 digits: Excel does not store it exactly, so the CSV has the file's value ({_lista(d['detalle'].get('ejemplos'), 3)}), which may not be what was typed. If it is an identifier, store it as text in Excel.",
                     f"In «{_col(d)}», " + "{n} whole numbers have more than 15 digits: Excel does not store them exactly, so the CSV has the file's values (" + _lista(d['detalle'].get('ejemplos'), 3) + "), which may not be what was typed. If they are identifiers, store them as text in Excel.")),
    "fecha_no_valida": (
        lambda d: _p(d['cuantas'], f"En «{_col(d)}», 1 celda con formato de fecha guarda un número que no es una fecha ({_motivos_fecha(d, 'es')}; {_lista(d['detalle'].get('celdas'))}): queda vacía.",
                     f"En «{_col(d)}», " + "{n} celdas con formato de fecha guardan un número que no es una fecha (" + _motivos_fecha(d, 'es') + "; " + _lista(d['detalle'].get('celdas')) + "): quedan vacías."),
        lambda d: _p(d['cuantas'], f"In «{_col(d)}», 1 cell with a date format holds a number that is not a date ({_motivos_fecha(d, 'en')}; {_lista(d['detalle'].get('celdas'))}): it is left empty.",
                     f"In «{_col(d)}», " + "{n} cells with a date format hold a number that is not a date (" + _motivos_fecha(d, 'en') + "; " + _lista(d['detalle'].get('celdas')) + "): they are left empty.")),
    "columna_mixta": (
        lambda d: f"«{_col(d)}» mezcla tipos de valor ({_tipos(d['detalle'].get('tipos'))}).",
        lambda d: f"«{_col(d)}» mixes value types ({_tipos(d['detalle'].get('tipos'))})."),
    "posible_fecha_en_serie": (
        lambda d: f"«{_col(d)}» son números entre 20.000 y 60.000 sin formato de fecha: ¿son fechas? No se han convertido.",
        lambda d: f"«{_col(d)}» holds numbers between 20,000 and 60,000 with no date format: are they dates? They were not converted."),
    "filas_ocultas": (
        lambda d: _p(d['cuantas'], f"1 fila de la tabla está oculta en Excel y se ha incluido igualmente (fila {_lista(d['detalle'].get('filas'))}).",
                     "{n} filas de la tabla están ocultas en Excel y se han incluido igualmente (filas " + _lista(d['detalle'].get('filas')) + ")."),
        lambda d: _p(d['cuantas'], f"1 row of the table is hidden in Excel and was included anyway (row {_lista(d['detalle'].get('filas'))}).",
                     "{n} rows of the table are hidden in Excel and were included anyway (rows " + _lista(d['detalle'].get('filas')) + ").")),
    "columnas_ocultas": (
        lambda d: _p(d['cuantas'], f"1 columna de la tabla está oculta en Excel y se ha incluido igualmente ({_lista(d['detalle'].get('letras'))}).",
                     "{n} columnas de la tabla están ocultas en Excel y se han incluido igualmente (" + _lista(d['detalle'].get('letras')) + ")."),
        lambda d: _p(d['cuantas'], f"1 column of the table is hidden in Excel and was included anyway ({_lista(d['detalle'].get('letras'))}).",
                     "{n} columns of the table are hidden in Excel and were included anyway (" + _lista(d['detalle'].get('letras')) + ").")),
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
        lambda d: _p(d['cuantas'], f"La hoja tiene 1 rango de celdas combinadas ({_lista(d['detalle'].get('rangos'))}): las celdas cubiertas llegan vacías.",
                     "La hoja tiene {n} rangos de celdas combinadas (" + _lista(d['detalle'].get('rangos')) + "): las celdas cubiertas llegan vacías."),
        lambda d: _p(d['cuantas'], f"The sheet has 1 merged cell range ({_lista(d['detalle'].get('rangos'))}): the covered cells arrive empty.",
                     "The sheet has {n} merged cell ranges (" + _lista(d['detalle'].get('rangos')) + "): the covered cells arrive empty.")),
    "formulas_sin_valor": (
        lambda d: _p(d['cuantas'], f"1 celda tiene fórmula pero ningún valor guardado ({_lista(d['detalle'].get('celdas'))}): esta hoja no se podrá convertir hasta que se guarde en Excel.",
                     "{n} celdas tienen fórmula pero ningún valor guardado (" + _lista(d['detalle'].get('celdas')) + "): esta hoja no se podrá convertir hasta que se guarde en Excel."),
        lambda d: _p(d['cuantas'], f"1 cell has a formula but no saved value ({_lista(d['detalle'].get('celdas'))}): this sheet cannot be converted until it is saved in Excel.",
                     "{n} cells have a formula but no saved value (" + _lista(d['detalle'].get('celdas')) + "): this sheet cannot be converted until it is saved in Excel.")),
    "formulas_a_recalcular": (     # I-R1 de 150-C1: `<calcPr fullCalcOnLoad="1"/>` (XlsxWriter)
        lambda d: _p(d['cuantas'], f"1 celda tiene fórmula ({_lista(d['detalle'].get('celdas'))}) y el libro pide recalcular al abrirlo: su valor guardado no es el resultado y esta hoja no se podrá convertir hasta que se abra y se guarde en Excel o LibreOffice.",
                     "{n} celdas tienen fórmula (" + _lista(d['detalle'].get('celdas')) + ") y el libro pide recalcular al abrirlo: su valor guardado no es el resultado y esta hoja no se podrá convertir hasta que se abra y se guarde en Excel o LibreOffice."),
        lambda d: _p(d['cuantas'], f"1 cell has a formula ({_lista(d['detalle'].get('celdas'))}) and the workbook asks to be recalculated when opened: its saved value is not the result and this sheet cannot be converted until it is opened and saved in Excel or LibreOffice.",
                     "{n} cells have a formula (" + _lista(d['detalle'].get('celdas')) + ") and the workbook asks to be recalculated when opened: their saved values are not the result and this sheet cannot be converted until it is opened and saved in Excel or LibreOffice.")),
    "posible_cabecera_de_dos_filas": (
        lambda d: f"La fila {d['detalle'].get('fila')} y la siguiente parecen una cabecera de dos filas (grupo y subcolumna).",
        lambda d: f"Row {d['detalle'].get('fila')} and the next one look like a two-row header (group and sub-column)."),
    "nombres_repetidos": (
        lambda d: _p(d['cuantas'], f"La cabecera propuesta repite o deja sin nombre 1 columna ({_lista(d['detalle'].get('nombres'))}): se renombrará y se declarará.",
                     "La cabecera propuesta repite o deja sin nombre {n} columnas (" + _lista(d['detalle'].get('nombres')) + "): se renombrarán y se declarará."),
        lambda d: _p(d['cuantas'], f"The proposed header repeats or leaves unnamed 1 column ({_lista(d['detalle'].get('nombres'))}): it will be renamed and declared.",
                     "The proposed header repeats or leaves unnamed {n} columns (" + _lista(d['detalle'].get('nombres')) + "): they will be renamed and declared.")),
    "hoja_sin_datos": (
        lambda d: "La hoja no tiene datos.",
        lambda d: "The sheet has no data."),
}


def declarar(tipo: str, *, columna: str | None = None, cuantas: int = 1, **detalle) -> dict:
    d = {"tipo": tipo, "columna": columna, "cuantas": int(cuantas), "detalle": detalle}
    es, en = TIPOS[tipo]
    d["texto"] = {"es": es(d), "en": en(d)}
    return d
