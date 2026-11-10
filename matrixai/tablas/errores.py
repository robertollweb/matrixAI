# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""Por qué se rechaza una tabla — **bilingüe en el núcleo** (150-C1, invariante 9).

«Lo que redacta el core se traduce en el core»: la interfaz no compone motivos, los recibe
escritos en español y en inglés. Ninguna excepción de una biblioteca (`BadZipFile`,
`CalamineError`, `ImportError`…) llega a la persona: todas se convierten aquí en una
`TablaRechazada` con un código estable (para el código que decide) y un motivo (para quien lee).
"""
from __future__ import annotations

__all__ = ["TablaRechazada", "rechazar", "CODIGOS", "VARIANTES"]

#: código → (motivo en español, motivo en inglés). Las llaves `{x}` salen de `detalle`.
CODIGOS: dict[str, tuple[str, str]] = {
    "vacio": (
        "El fichero está vacío (0 bytes).",
        "The file is empty (0 bytes)."),
    "html_no_excel": (
        "Esto es una página web (HTML) con extensión de Excel, no un libro de Excel. Ábrelo en "
        "Excel y guárdalo como «Libro de Excel (.xlsx)».",
        "This is a web page (HTML) with an Excel extension, not an Excel workbook. Open it in "
        "Excel and save it as «Excel Workbook (.xlsx)»."),
    "texto_no_excel": (
        "Esto es un fichero de texto (parece un CSV), no un libro de Excel, aunque lleve la "
        "extensión {extension}. Si es un CSV, súbelo con su extensión .csv.",
        "This is a text file (it looks like a CSV), not an Excel workbook, even though it has "
        "the {extension} extension. If it is a CSV, upload it with its .csv extension."),
    "zip_danado": (
        "El fichero está dañado o incompleto (no se puede leer como un libro de Excel). Puede "
        "que la descarga se cortara: vuelve a descargarlo o guárdalo de nuevo desde Excel.",
        "The file is damaged or incomplete (it cannot be read as an Excel workbook). The "
        "download may have been cut off: download it again or save it again from Excel."),
    "no_es_excel": (
        "El fichero es un archivo comprimido, pero no contiene un libro de Excel.",
        "The file is a compressed archive, but it does not contain an Excel workbook."),
    "protegido": (
        "El libro está protegido con contraseña y no se puede leer. Quita la contraseña en "
        "Excel (Archivo > Información > Proteger libro) y vuelve a subirlo.",
        "The workbook is password-protected and cannot be read. Remove the password in Excel "
        "(File > Info > Protect Workbook) and upload it again."),
    "formato_no_soportado": (
        "El formato {formato} no está soportado. Se abren .xlsx, .xlsm y .xls.",
        "The {formato} format is not supported. .xlsx, .xlsm and .xls files can be opened."),
    # 150-C8 (Parquet): el lector es `pyarrow`, extra opcional. Sin él, Parquet dice esto y Excel sigue funcionando.
    "falta_pyarrow": (
        "Falta pyarrow, el lector de Parquet: instálalo con «pip install pyarrow» (o con pip install "
        "\"matrixai-core[parquet]\"). Los libros de Excel y los CSV se abren igual sin él.",
        "pyarrow, the Parquet reader, is missing: install it with «pip install pyarrow» (or with pip install "
        "\"matrixai-core[parquet]\"). Excel workbooks and CSV files open the same without it."),
    "pyarrow_no_carga": (
        "pyarrow está instalado pero no se ha podido cargar ({tipo}). Puede que falte memoria o que la instalación esté dañada: "
        "reinstálalo con «pip install --force-reinstall pyarrow». Los libros de Excel y los CSV se abren igual.",
        "pyarrow is installed but could not be loaded ({tipo}). The machine may be short of memory or the installation may be "
        "damaged: reinstall it with «pip install --force-reinstall pyarrow». Excel workbooks and CSV files open the same."),
    "parquet_danado": (
        "El fichero Parquet está dañado o incompleto (no se puede leer: {tipo}). Puede que la descarga se "
        "cortara: vuelve a descargarlo o a exportarlo.",
        "The Parquet file is damaged or incomplete (it cannot be read: {tipo}). The download may have been "
        "cut off: download or export it again."),
    "parquet_cifrado": (
        "El fichero Parquet está cifrado y no se puede leer aquí. Expórtalo sin cifrar desde donde salió.",
        "The Parquet file is encrypted and cannot be read here. Export it unencrypted from wherever it came from."),
    "parquet_no_legible": (
        "El fichero Parquet usa algo que este lector no admite ({tipo}). Vuelve a exportarlo con compresión "
        "estándar (snappy, zstd o sin comprimir).",
        "The Parquet file uses something this reader does not support ({tipo}). Export it again with standard "
        "compression (snappy, zstd or uncompressed)."),
    "parquet_vacio": (
        "El fichero Parquet no tiene filas o no tiene columnas.",
        "The Parquet file has no rows or no columns."),
    "parquet_sin_columnas_convertibles": (
        "Ninguna de las {n_columnas} del fichero Parquet se puede convertir a CSV: todas son listas, estructuras, "
        "mapas, binarios u otros tipos que un CSV no puede guardar.",
        "None of the {n_columnas} in the Parquet file can be converted to CSV: they are all lists, structs, "
        "maps, binaries or other types a CSV cannot hold."),
    "formato_desconocido": (
        "No se reconoce el formato del fichero (empieza por bytes que no son de un Excel). "
        "Se abren .xlsx, .xlsm, .xls, .parquet y .csv.",
        "The file format is not recognised (it does not start like an Excel file). .xlsx, "
        ".xlsm, .xls, .parquet and .csv files can be opened."),
    "bomba": (
        "El libro descomprimido ocuparía {descomprimido} (el fichero pesa {original}) y el "
        "tope es {tope}. No se abre: podría agotar la memoria de este equipo.",
        "The workbook would take {descomprimido} once uncompressed (the file is {original}) "
        "and the limit is {tope}. It is not opened: it could exhaust this machine's memory."),
    # «al menos»: la geometría deja de recorrer la hoja en cuanto pasa del tope (no hace falta
    # leer el resto para saber que no cabe), así que el número es una cota, no el total.
    "demasiadas_filas": (
        "La hoja «{hoja}» tiene al menos {n_filas} y el tope es {tope}.",
        "Sheet «{hoja}» has at least {n_filas} and the limit is {tope}."),
    "demasiadas_columnas": (
        "La hoja «{hoja}» tiene al menos {n_columnas} y el tope es {tope}.",
        "Sheet «{hoja}» has at least {n_columnas} and the limit is {tope}."),
    "memoria_insuficiente": (
        "Leer la hoja «{hoja}» necesita unos {necesario} de memoria y este equipo tiene "
        "{disponible} disponibles. No es un tope que se pueda cambiar: copia los datos a un "
        "libro más pequeño.",
        "Reading sheet «{hoja}» needs about {necesario} of memory and this machine has "
        "{disponible} available. It is not a limit that can be changed: copy the data to a "
        "smaller workbook."),
    "hoja_no_encontrada": (
        "No existe la hoja «{hoja}». Hojas del libro: {hojas}.",
        "There is no sheet «{hoja}». Sheets in the workbook: {hojas}."),
    "hoja_requerida": (
        "El libro tiene más de una hoja visible ({hojas}): elige cuál se abre. Las hojas no "
        "se unen.",
        "The workbook has more than one visible sheet ({hojas}): choose which one to open. "
        "Sheets are never merged."),
    "hoja_vacia": (
        "La hoja «{hoja}» no tiene datos.",
        "Sheet «{hoja}» has no data."),
    "sin_hojas": (
        "El libro no tiene ninguna hoja de datos.",
        "The workbook has no data sheets."),
    "cabecera_invalida": (
        "La fila de cabecera {fila} no es válida para la hoja «{hoja}» (tiene datos en las "
        "filas {primera} a {ultima}).",
        "Header row {fila} is not valid for sheet «{hoja}» (it has data in rows {primera} "
        "to {ultima})."),
    "parametro_invalido": (
        "Parámetro no válido: {parametro}.",
        "Invalid parameter: {parametro}."),
    "sin_filas_de_datos": (
        "Debajo de la cabecera no queda ninguna fila de datos (cabecera en la fila {fila}).",
        "No data rows are left below the header (header in row {fila})."),
    "formula_sin_valor": (
        "Hay {n_celdas} con una fórmula pero ningún valor guardado (por ejemplo {celdas}). "
        "El libro lo escribió otra herramienta sin calcular las fórmulas y aquí no se calculan "
        "nunca: ábrelo en Excel, guárdalo y vuelve a subirlo.",
        "{there} {n_celdas} with a formula but no saved value (for example {celdas}). The "
        "workbook was written by another tool without calculating the formulas, and they are "
        "never calculated here: open it in Excel, save it and upload it again."),
    # (la variante «recalcular» de `formula_sin_valor` va en `VARIANTES`, con el mismo código)
    "falta_lector": (
        "Falta el lector de Excel. Instálalo con: pip install \"matrixai-core[tablas]\".",
        "The Excel reader is missing. Install it with: pip install \"matrixai-core[tablas]\"."),
    "demasiados_textos": (
        "El libro guarda {tamano} de textos (la tabla de textos compartidos) y el tope es {tope}. "
        "Leerla obligaría a cargarla entera en memoria: copia los datos a un libro nuevo.",
        "The workbook stores {tamano} of text (the shared strings table) and the limit is {tope}. "
        "Reading it would mean loading all of it into memory: copy the data to a new workbook."),
    "lector_sin_memoria": (
        "El lector de Excel se quedó sin memoria leyendo el libro (techo: {techo}) y se ha "
        "parado. El fichero pide mucha más memoria de la que su contenido justifica: puede estar "
        "dañado o fabricado a propósito. Ábrelo en Excel y guárdalo de nuevo.",
        "The Excel reader ran out of memory reading the workbook (limit: {techo}) and was "
        "stopped. The file asks for far more memory than its content justifies: it may be "
        "damaged or crafted on purpose. Open it in Excel and save it again."),
    "lector_tiempo": (
        "El lector de Excel tardó más de {segundos} s en leer el libro y se ha parado. Si el "
        "libro es muy grande, copia solo la hoja que necesitas a un libro nuevo.",
        "The Excel reader took more than {segundos} s to read the workbook and was stopped. If "
        "the workbook is very large, copy just the sheet you need to a new workbook."),
    "lector_abortado": (
        "El lector de Excel se detuvo de golpe leyendo el libro ({como}). El fichero puede estar "
        "dañado: ábrelo en Excel y guárdalo de nuevo.",
        "The Excel reader stopped abruptly reading the workbook ({como}). The file may be "
        "damaged: open it in Excel and save it again."),
    "error_interno": (
        "No se ha podido convertir por un fallo del propio conversor ({tipo}), no del fichero. "
        "Avisa con el fichero si puedes.",
        "It could not be converted because of a fault in the converter itself ({tipo}), not in "
        "the file. Please report it with the file if you can."),
    "pdf_no_excel": (
        "Esto es un PDF, no un libro de Excel, aunque lleve la extensión {extension}. Las tablas "
        "de un PDF no se leen: exporta los datos a Excel o a CSV desde donde salió el PDF.",
        "This is a PDF, not an Excel workbook, even though it has the {extension} extension. "
        "Tables inside a PDF are not read: export the data to Excel or CSV from wherever the "
        "PDF came from."),
    "xml_no_excel": (
        "Esto es un fichero XML, no un libro de Excel, aunque lleve la extensión {extension}. "
        "Ábrelo en Excel y guárdalo como «Libro de Excel (.xlsx)».",
        "This is an XML file, not an Excel workbook, even though it has the {extension} "
        "extension. Open it in Excel and save it as «Excel Workbook (.xlsx)»."),
    "lector_fallo": (
        "No se ha podido leer el libro ({tipo}). Puede estar dañado: ábrelo en Excel y "
        "guárdalo de nuevo.",
        "The workbook could not be read ({tipo}). It may be damaged: open it in Excel and "
        "save it again."),
}


class TablaRechazada(Exception):
    """Un fichero que NO se convierte, con su motivo en los dos idiomas.

    `codigo` es estable (lo usa el código que decide); `motivo_es`/`motivo_en` los lee la
    persona; `detalle` lleva los números y nombres que el motivo cita.
    """

    def __init__(self, codigo: str, motivo_es: str, motivo_en: str, detalle: dict | None = None):
        super().__init__(motivo_es)
        self.codigo = codigo
        self.motivo_es = motivo_es
        self.motivo_en = motivo_en
        self.detalle = dict(detalle or {})

    def motivo(self, idioma: str = "es") -> str:
        return self.motivo_en if str(idioma).lower().startswith("en") else self.motivo_es

    def como_dict(self) -> dict:
        return {"error_kind": "tabla_rechazada", "codigo": self.codigo,
                "motivo_es": self.motivo_es, "motivo_en": self.motivo_en,
                "detalle": self.detalle}


def _plural(n, es_uno: str, es_varios: str, en_uno: str, en_varios: str) -> tuple[str, str]:
    return ((f"1 {es_uno}" if n == 1 else f"{n} {es_varios}"),
            (f"1 {en_uno}" if n == 1 else f"{n} {en_varios}"))


#: (código, variante) → (es, en): el MISMO código estable con otro motivo para quien lee.
VARIANTES: dict[tuple[str, str], tuple[str, str]] = {
    # 150-C8 (Parquet): las mismas guardas, dichas con las palabras de un fichero de columnas (sin «hoja»).
    ("bomba", "parquet"): (
        "Los datos del fichero Parquet ocuparían {descomprimido} descomprimidos (el fichero pesa {original}) y el "
        "tope es {tope}. No se abre: podría agotar la memoria de este equipo.",
        "The Parquet file's data would take {descomprimido} once uncompressed (the file is {original}) and the "
        "limit is {tope}. It is not opened: it could exhaust this machine's memory."),
    ("demasiadas_filas", "parquet"): (
        "El fichero Parquet tiene {n_filas} y el tope es {tope}.",
        "The Parquet file has {n_filas} and the limit is {tope}."),
    ("demasiadas_columnas", "parquet"): (
        "El fichero Parquet tiene {n_columnas} y el tope es {tope}.",
        "The Parquet file has {n_columnas} and the limit is {tope}."),
    ("memoria_insuficiente", "parquet"): (
        "Convertir el fichero Parquet necesita unos {necesario} de memoria y este equipo tiene {disponible} "
        "disponibles. No es un tope que se pueda cambiar: exporta menos filas o columnas.",
        "Converting the Parquet file needs about {necesario} of memory and this machine has {disponible} "
        "available. It is not a limit that can be changed: export fewer rows or columns."),
    ("memoria_insuficiente", "parquet_techo_del_proceso"): (
        "Convertir el fichero Parquet necesita que el proceso que lo convierte use unos {necesario} de memoria, y "
        "solo puede usar {disponible} más (su techo es {techo}). No indica que el fichero esté dañado: exporta menos "
        "filas o columnas.",
        "Converting the Parquet file needs the process converting it to use about {necesario} of memory, and it can "
        "only use {disponible} more (its limit is {techo}). It does not mean the file is damaged: export fewer rows "
        "or columns."),
    ("lector_sin_memoria", "parquet"): (
        "El lector de Parquet se quedó sin memoria leyendo el fichero y se ha parado. O el fichero es demasiado "
        "grande para este equipo, o pide mucha más memoria de la que su contenido justifica (dañado o fabricado a "
        "propósito). Exporta menos filas o columnas.",
        "The Parquet reader ran out of memory reading the file and was stopped. Either the file is too large for "
        "this machine, or it asks for far more memory than its content justifies (damaged or crafted on purpose). "
        "Export fewer rows or columns."),
    # I-R1 de 150-C1 (decisión del supervisor, revocable por Roberto): XlsxWriter guarda cada
    # fórmula con un valor 0 de relleno y `<calcPr fullCalcOnLoad="1"/>`; el 0 NO es el
    # resultado, así que se rechaza igual que una fórmula sin valor.
    ("formula_sin_valor", "recalcular"): (
        "Hay {n_celdas} con una fórmula (por ejemplo {celdas}) y el libro pide recalcular todas "
        "sus fórmulas al abrirlo: el valor guardado no es el resultado (quien lo escribió dejó un "
        "relleno, normalmente 0) y aquí no se calculan nunca. Ábrelo en Excel o LibreOffice, "
        "guárdalo y vuelve a subirlo.",
        "{there} {n_celdas} with a formula (for example {celdas}) and the workbook asks for all "
        "its formulas to be recalculated when it is opened: the saved value is not the result "
        "(the tool that wrote it left a placeholder, usually 0) and they are never calculated "
        "here. Open it in Excel or LibreOffice, save it and upload it again."),
    # I1 de la auditoría de 150-C3: el proceso que convierte tiene un techo propio (`RLIMIT_AS`) por debajo de lo que el
    # equipo tiene libre; lo que no cabe es el libro en ESE techo, y se dice con las dos cifras.
    ("memoria_insuficiente", "techo_del_proceso"): (
        "Leer la hoja «{hoja}» necesita que el proceso que convierte el libro use unos {necesario} de memoria, y solo "
        "puede usar {disponible} más (su techo es {techo}). No indica que el fichero esté dañado: copia los datos a un "
        "libro más pequeño.",
        "Reading sheet «{hoja}» needs the process converting the workbook to use about {necesario} of memory, and it "
        "can only use {disponible} more (its limit is {techo}). It does not mean the file is damaged: copy the data to "
        "a smaller workbook."),
    # M5 de la auditoría de 150-C3: sin nombre (o sin punto) no hay extensión que citar, y «(sin extensión)» se colaba
    # en el texto inglés.
    ("pdf_no_excel", "sin_extension"): (
        "Esto es un PDF, no un libro de Excel. Las tablas de un PDF no se leen: exporta los datos a Excel o a CSV "
        "desde donde salió el PDF.",
        "This is a PDF, not an Excel workbook. Tables inside a PDF are not read: export the data to Excel or CSV from "
        "wherever the PDF came from."),
    ("xml_no_excel", "sin_extension"): (
        "Esto es un fichero XML, no un libro de Excel. Ábrelo en Excel y guárdalo como «Libro de Excel (.xlsx)».",
        "This is an XML file, not an Excel workbook. Open it in Excel and save it as «Excel Workbook (.xlsx)»."),
    ("texto_no_excel", "sin_extension"): (
        "Esto es un fichero de texto (parece un CSV), no un libro de Excel. Si es un CSV, súbelo con su extensión "
        ".csv.",
        "This is a text file (it looks like a CSV), not an Excel workbook. If it is a CSV, upload it with its .csv "
        "extension."),
    ("lector_abortado", "desde_fuera"): (
        "El lector de Excel se detuvo leyendo el libro porque lo paró otro proceso o el sistema "
        "({como}; por ejemplo, la máquina se quedó sin memoria). No indica que el fichero esté "
        "dañado: vuelve a intentarlo.",
        "The Excel reader was stopped while reading the workbook by another process or by the "
        "system ({como}; for example, the machine ran out of memory). It does not mean the file "
        "is damaged: try again."),
}


def rechazar(codigo: str, *, variante: str | None = None, **detalle) -> TablaRechazada:
    """Construye la excepción con el motivo del catálogo (`raise rechazar("vacio")`); con
    `variante`, el motivo de `VARIANTES[(codigo, variante)]` y el mismo código.

    Los recuentos se escriben con singular/plural en cada idioma (`n_filas`, `n_columnas`,
    `n_celdas`): ningún motivo dice «1 filas».

    Un valor que se escribe distinto en inglés va con el sufijo `_en` (`parametro=…, parametro_en=…`): el motivo
    inglés usa ese y el español el otro (M4 de la auditoría de 150-C3: «Invalid parameter: … debe ser 1 o 2»)."""
    es = {k: _a_texto(v, "es") for k, v in detalle.items()}
    en = {k: _a_texto(v, "en") for k, v in detalle.items()}
    for clave in [k for k in en if k.endswith("_en")]:
        en[clave[:-3]] = en[clave]
    for clave, (a, b, c, d) in {"filas": ("fila", "filas", "row", "rows"),
                                "columnas": ("columna", "columnas", "column", "columns"),
                                "cuantas": ("celda", "celdas", "cell", "cells")}.items():
        if isinstance(detalle.get(clave), int):
            n = detalle[clave]
            nombre = "n_celdas" if clave == "cuantas" else f"n_{clave}"
            es[nombre], en[nombre] = _plural(n, a, b, c, d)
            if clave == "cuantas":
                en["there"] = "There is" if n == 1 else "There are"
    msg_es, msg_en = CODIGOS[codigo] if variante is None else VARIANTES[(codigo, variante)]
    return TablaRechazada(codigo, msg_es.format(**es), msg_en.format(**en), detalle)


#: Cuántos elementos de una lista caben en el TEXTO de un motivo. El resto se cuenta («y 39.990 más»): una lista que la
#: persona manda (`filas_a_excluir`: decenas de miles de filas) no se enumera entera en una frase que va a una pantalla
#: (menor 2 de la re-auditoría de 150-C3). El dato completo, si hace falta, vive en el `detalle` de la excepción.
MAX_ELEMENTOS_EN_UN_MOTIVO = 10


def lista_para_texto(valores, idioma: str = "es") -> str:
    """`1, 2, 3` o, pasados `MAX_ELEMENTOS_EN_UN_MOTIVO`, `1, 2, … y N más` / `… and N more` (sin separador de miles:
    el mismo número en los dos idiomas)."""
    valores = list(valores)
    if len(valores) <= MAX_ELEMENTOS_EN_UN_MOTIVO:
        return ", ".join(str(x) for x in valores)
    resto = len(valores) - MAX_ELEMENTOS_EN_UN_MOTIVO
    primeros = ", ".join(str(x) for x in valores[:MAX_ELEMENTOS_EN_UN_MOTIVO])
    return f"{primeros} y {resto} más" if idioma == "es" else f"{primeros}, and {resto} more"


def _a_texto(v, idioma: str = "es") -> str:
    if isinstance(v, (list, tuple)):
        return lista_para_texto(v, idioma)
    return str(v)
