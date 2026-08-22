"""CONTRATO 81-C6 — riesgo de reingreso a 30 días: las REGLAS del caso.

**Lo que este caso NO es**, y va primero porque es lo que más importa
(§16.2): es **demostrativo**, **no está clínicamente validado**, **no
sirve para decisiones asistenciales** y corre sobre **datos sintéticos**.
Es un sistema de apoyo, y ni siquiera eso todavía. Esta frase viaja en la
salida de cada ejecución —no en un anexo— porque un caso de reingreso
hospitalario que no la lleve **donde se lee el resultado** se leerá como
una herramienta clínica, y esa es la peor afirmación por omisión que este
contrato puede permitir.

Aquí vive lo que el C6 aporta de nuevo: las **políticas declaradas** y
las comprobaciones que las hacen cumplirse. La maquinaria —motor, recibo,
sandbox, paquete reproducible— ya está en los cortes anteriores y no se
vuelve a escribir.

Las reglas, y cada una responde a una forma concreta de mentir:

* **un valor ausente no es un cero.** Rellenar una variable clínica que
  falta con la media convierte «no lo sabemos» en «es normal», que es
  justo el dato que cambiaría la decisión;
* **una categoría desconocida no se parece a ninguna.** Mapearla a la más
  frecuente produce una predicción sobre un paciente que no es el que se
  describió;
* **un texto que se trunca lo DICE.** Truncar en silencio deja fuera el
  final de la nota —donde suele estar lo agudo— y el resultado se lee
  como si hubiera visto la nota entera;
* **una nota vacía no es una nota normal.** Es ausencia de señal, y se
  declara como tal;
* **un byte que no es UTF-8 no se reemplaza.** Sustituirlo por `?` cambia
  el texto sobre el que se decide sin que nadie lo sepa;
* **un dato posterior al momento de la decisión no entra.** Es fuga
  temporal, y un modelo que la sufre parece mucho mejor de lo que es;
* **una fusión de dimensiones que no encajan no se recorta.** Ajustarla
  produciría un vector que no describe a nadie;
* **por debajo del umbral de confianza NO se predice: se abstiene.** Una
  clase con confianza de moneda al aire se lee igual que una segura.

**Y la línea que separa RECHAZAR de ABSTENERSE**, porque confundirlas fue
el primer error de este módulo y se corrigió antes de escribir la primera
prueba:

* se **RECHAZA** (excepción) lo que está mal formado o es inseguro de
  procesar: un texto que no es UTF-8, una fusión de dimensiones que no
  encajan, un número que no es un número. Ahí no hay nada que decidir;
* se **ABSTIENE** (política) lo que está bien formado y **no basta** para
  decidir: falta una variable obligatoria, la categoría no existe, hay
  fuga temporal o la confianza es baja. Ahí sí hay algo que decir, y es
  «no me pronuncio», con la regla que lo decidió al lado.

Meter las abstenciones en excepciones tenía dos consecuencias, las dos
malas: el pipeline **se caía** donde debía abstenerse —y caerse no es lo
mismo que no pronunciarse—, y las reglas R1, R2 y R3 de la política
**no podían dispararse nunca**, que es una política que parece bien
escrita y no decide jamás.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

__all__ = [
    "ADVERTENCIA",
    "CAMPOS_SENSIBLES",
    "CATEGORIAS",
    "CATEGORIAS_CONOCIDAS",
    "PERFIL_DE_PRIVACIDAD",
    "POLITICA_DE_ABSTENCION",
    "POLITICAS",
    "DimensionesIncompatibles",
    "TextoNoUTF8",
    "ValorNoNumerico",
    "campos_con_fuga_temporal",
    "analisis_de_confianza",
    "cobertura_por_clase",
    "codificar_tabular",
    "codificar_texto",
    "decidir",
    "fusionar",
    "huella_sin_datos",
    "preparar_texto",
]

#: LA ADVERTENCIA, en los dos idiomas y **en la salida**, no en un anexo.
#: §16.6: «el caso no se presenta como validación clínica».
ADVERTENCIA = {
    "es": ("DEMOSTRACIÓN. Datos sintéticos. NO validado clínicamente y NO "
           "apto para decisiones asistenciales: es un sistema de apoyo, y "
           "no sustituye al profesional."),
    "en": ("DEMONSTRATION. Synthetic data. NOT clinically validated and NOT "
           "fit for care decisions: it is a support system, and it does not "
           "replace the professional."),
}

#: Las variables tabulares del caso. `required=True` significa que **sin
#: ella no se decide**: no hay imputación que valga para una variable que
#: cambia el resultado.
VARIABLES_TABULARES: tuple[dict[str, Any], ...] = (
    {"name": "edad", "kind": "number", "range": (0.0, 120.0), "required": True},
    {"name": "ingresos_previos_12m", "kind": "number", "range": (0.0, 20.0), "required": True},
    {"name": "dias_de_estancia", "kind": "number", "range": (0.0, 60.0), "required": True},
    {"name": "servicio", "kind": "category", "required": True},
    {"name": "alta_voluntaria", "kind": "number", "range": (0.0, 1.0), "required": False},
)

#: Las categorías QUE EXISTEN. Cerrada a propósito: una categoría que no
#: está aquí no se parece a ninguna de éstas — se abstiene.
CATEGORIAS: dict[str, tuple[str, ...]] = {
    "servicio": ("medicina_interna", "cardiologia", "neumologia", "cirugia"),
}
CATEGORIAS_CONOCIDAS = CATEGORIAS  # nombre largo, para quien lea el informe

#: Lo que NUNCA entra en la traza ni en el recibo (§13.3 del contrato).
#: Se declara aquí, en un solo sitio, y `huella_sin_datos` es lo único que
#: sale de estos campos.
CAMPOS_SENSIBLES = ("nota_clinica", "nhc", "nombre", "apellidos", "fecha_nacimiento")

PERFIL_DE_PRIVACIDAD = {
    "profile": "local-only",
    "network": "disabled",
    "trace_contains": ["identificadores", "versiones", "digests", "tiempos",
                       "estados", "resultados de políticas"],
    "trace_never_contains": list(CAMPOS_SENSIBLES),
    # Un digest a secas NO protege un campo de pocos valores posibles
    # (§13.3): el servicio tiene cuatro, y probar los cuatro lo revierte.
    # Por eso el servicio viaja como categoría declarada y no como huella
    # «anonimizada», que sería una promesa falsa.
    "low_cardinality_note": (
        "los campos de pocos valores posibles no se protegen con un hash "
        "simple: aquí no se pretende anonimizarlos, se declaran"),
}

#: LAS POLÍTICAS, declaradas. Que estén escritas en un sitio que viaja con
#: el recibo es lo que permite auditar por qué se decidió lo que se
#: decidió — y no volver a discutirlas cada vez.
POLITICAS = {
    "missing_values": "required_missing_abstains",
    "unknown_category": "abstains",
    "text_length": "truncate_and_declare",
    "empty_text": "declared_as_absent_signal",
    "text_encoding": "utf8_strict_reject",
    "temporal": "reject_after_decision_moment",
    "fusion": "reject_dimension_mismatch",
    "confidence": "abstain_below_threshold",
    # El umbral, escrito: un umbral que vive en el código y no en la
    # política es un umbral que nadie puede auditar.
    "confidence_threshold": 0.60,
    "text_max_bytes": 512,
    "tokenizer": "byte_v1",
    "tabular_encoder_version": "readmission-tabular-1.0",
    "text_encoder_version": "readmission-text-1.0",
    "fusion_version": "readmission-fusion-1.0",
}

#: La política de abstención en el esquema del 81-C1, para que la evalúe
#: `policy.py` y su decisión entre en la traza CON la regla que decidió.
POLITICA_DE_ABSTENCION: dict[str, Any] = {
    "schema_version": "1.0",
    "name": "readmission_abstention",
    "version": "1.0",
    # Fallo cerrado: lo que no case NIEGA. En un caso clínico, «no opino»
    # nunca puede convertirse en «adelante».
    "default": "deny",
    "rules": [
        {"id": "R1-datos-incompletos", "when": {"eq": [{"var": "datos_completos"}, False]},
         "then": "deny"},
        {"id": "R2-categoria-desconocida", "when": {"eq": [{"var": "categoria_conocida"}, False]},
         "then": "deny"},
        {"id": "R3-fuga-temporal", "when": {"eq": [{"var": "sin_fuga_temporal"}, False]},
         "then": "deny"},
        {"id": "R4-baja-confianza", "when": {"lt": [{"var": "confianza"}, 0.60]},
         "then": "deny"},
        {"id": "R5-todo-en-orden", "when": {"ge": [{"var": "confianza"}, 0.60]},
         "then": "allow"},
    ],
}


class ValorNoNumerico(ValueError):
    """Un número que no es un número: eso es un dato mal recogido.

    Se RECHAZA en vez de tratarlo como ausente, porque pasarlo por
    ausente escondería el error de recogida detrás de una abstención
    perfectamente razonable.
    """


class TextoNoUTF8(ValueError):
    """El texto no es UTF-8 válido y no se arregla por nuestra cuenta."""


class DimensionesIncompatibles(ValueError):
    """Los dos lados de la fusión no encajan."""


def huella_sin_datos(valor: Any) -> str:
    """La huella de un dato sensible — **lo ÚNICO que sale de él**.

    No es anonimización y no se vende como tal: para un campo de pocos
    valores posibles, probar los valores revierte el hash (§13.3). Aquí
    sirve para poder decir «es el mismo texto» sin llevar el texto.
    """
    crudo = valor if isinstance(valor, bytes) else str(valor).encode("utf-8")
    return "sha256:" + hashlib.sha256(crudo).hexdigest()


def campos_con_fuga_temporal(
    registro: dict[str, Any], momento_de_decision: str
) -> list[str]:
    """Los campos POSTERIORES al momento en que se decide, o lista vacía.

    Devuelve en vez de levantar porque lo usan **dos** caminos con
    consecuencias distintas y una sola regla: al decidir sobre un caso, la
    fuga lleva a **abstención** (con su regla R3 nombrada); al construir
    el dataset, a **rechazo**. Dos funciones parecidas acabarían
    discrepando, y la que se quedara atrás sería la que deja pasar.

    Un modelo entrenado o servido con datos del futuro parece mucho mejor
    de lo que es, y el error no se ve en las métricas: se ve el día que
    se usa de verdad.

    **Sin momento declarado, TODO es sospechoso.** Devolver «no hay fuga»
    porque nadie dijo contra qué comparar sería fallo abierto.
    """
    if not momento_de_decision:
        return ["<sin momento de decisión declarado>"]
    return sorted(
        clave for clave, valor in registro.items()
        if clave.endswith("_at") and isinstance(valor, str) and valor > momento_de_decision
    )


def codificar_tabular(registro: dict[str, Any]) -> dict[str, Any]:
    """Las variables tabulares, normalizadas y CON lo que falta declarado.

    **Nada se imputa.** Una variable obligatoria que falta levanta
    `CampoAusente`; una opcional que falta viaja como `None` y se declara
    en `missing`, nunca como `0.0` — un cero es un dato y la ausencia no.
    """
    valores: dict[str, float | None] = {}
    faltan_obligatorias: list[str] = []
    faltan_opcionales: list[str] = []
    fuera_de_rango: list[str] = []
    desconocidas: list[str] = []
    for var in VARIABLES_TABULARES:
        nombre = str(var["name"])
        crudo = registro.get(nombre)
        if crudo is None or crudo == "":
            # NO se imputa, y tampoco se levanta: falta un dato, y de eso
            # se ABSTIENE la política — caerse aquí no es lo mismo que no
            # pronunciarse, y encima dejaría R1 sin poder dispararse.
            (faltan_obligatorias if var["required"] else faltan_opcionales).append(nombre)
            if var["kind"] == "category":
                for opcion in CATEGORIAS.get(nombre, ()):
                    valores[f"{nombre}_{opcion}"] = None
            else:
                valores[nombre] = None
                # LA AUSENCIA, DENTRO DEL VECTOR. Un modelo de entrada fija
                # no puede representar «no lo sé» dejando el hueco: o el
                # vector encoge —y entonces no encaja— o se rellena con un
                # cero que se lee como una medida.
                #
                # Se hace lo único honesto: el hueco va a 0.0 **y al lado
                # va una bandera que dice que eso NO es una medición**. Así
                # el modelo puede aprender que faltar significa algo, en vez
                # de aprender que faltar es ser normal.
                if not var["required"]:
                    valores[f"{nombre}__ausente"] = 1.0
            continue

        if var["kind"] == "category":
            conocidas = CATEGORIAS.get(nombre, ())
            if str(crudo) not in conocidas:
                # Tampoco se mapea «a la más parecida»: predeciría sobre un
                # paciente que no es éste. Se declara y se abstiene (R2).
                desconocidas.append(f"{nombre}={crudo}")
                for opcion in conocidas:
                    valores[f"{nombre}_{opcion}"] = None
                continue
            # One-hot: una categoría no es un número ordenado, y darle uno
            # haría que «cirugía» estuviera más lejos de «cardiología» que
            # de «neumología» por el orden en que se escribieron.
            for opcion in conocidas:
                valores[f"{nombre}_{opcion}"] = 1.0 if str(crudo) == opcion else 0.0
            continue

        try:
            numero = float(crudo)
        except (TypeError, ValueError):
            # ESTO sí se rechaza: no es incertidumbre clínica, es un dato
            # mal recogido, y pasarlo por ausente lo escondería detrás de
            # una abstención perfectamente razonable.
            raise ValorNoNumerico(
                f"{nombre}={crudo!r} no es un número: tratarlo como si "
                "faltara escondería un dato mal recogido detrás de una "
                "abstención") from None
        minimo, maximo = var["range"]
        # Se recorta al rango DECLARADO y se dice cuando pasa: un valor
        # fuera de rango suele ser un error de recogida, y colarlo
        # normalizado a 1.3 mueve la predicción sin que nadie lo vea.
        recortado = min(max(numero, minimo), maximo)
        if recortado != numero:
            fuera_de_rango.append(f"{nombre}={numero}")
        valores[nombre] = (recortado - minimo) / (maximo - minimo) if maximo > minimo else 0.0
        if not var["required"]:
            # La bandera va SIEMPRE, presente o no: un vector cuya talla
            # depende de lo que falte no encaja con el modelo, y ésa es
            # justo la avería que la fusión existe para no dejar pasar.
            valores[f"{nombre}__ausente"] = 0.0

    return {
        "values": valores,
        # Las tres listas SEPARADAS: una obligatoria que falta abstiene,
        # una opcional no, y un valor fuera de rango es otra cosa —
        # juntarlas en un solo `missing` hacía que un campo opcional
        # ausente abstuviera, que es lo contrario de «opcional».
        "missing_required": faltan_obligatorias,
        "missing_optional": faltan_opcionales,
        "out_of_range": fuera_de_rango,
        "unknown_categories": desconocidas,
        "complete": not faltan_obligatorias,
        "known_categories": not desconocidas,
        "encoder_version": POLITICAS["tabular_encoder_version"],
    }


def preparar_texto(nota: Any) -> dict[str, Any]:
    """La nota clínica, con su política de longitud y codificación DICHAS.

    * **UTF-8 estricto**: un byte inválido no se reemplaza por `?`. Eso
      cambiaría el texto sobre el que se decide sin que nadie lo sepa.
    * **Truncar se DECLARA**: el final de una nota es donde suele estar lo
      agudo, y recortarlo en silencio deja un resultado que se lee como si
      hubiera visto la nota entera.
    * **Vacía no es normal**: es ausencia de señal, y así se declara.
    """
    if isinstance(nota, bytes):
        try:
            nota = nota.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TextoNoUTF8(
                f"la nota no es UTF-8 válido ({exc}): no se reemplazan los "
                "bytes que no se entienden, porque eso cambiaría el texto "
                "sobre el que se decide") from None
    if nota is None:
        nota = ""
    if not isinstance(nota, str):
        raise TextoNoUTF8(f"una nota clínica es texto, y ha llegado {type(nota).__name__}")

    crudo = nota.encode("utf-8")
    tope = int(POLITICAS["text_max_bytes"])
    truncado = len(crudo) > tope
    return {
        "text": nota,
        "bytes": len(crudo),
        "truncated": truncado,
        "truncated_at": tope if truncado else None,
        # «Vacía» es un estado declarado, no un texto de ceros.
        "empty": len(nota.strip()) == 0,
        "digest": huella_sin_datos(nota),
        "tokenizer": POLITICAS["tokenizer"],
        "max_bytes": tope,
    }


def codificar_texto(preparado: dict[str, Any], vocabulario: tuple[str, ...]) -> dict[str, Any]:
    """Bolsa de palabras sobre un vocabulario DECLARADO.

    Se usa la misma forma que el caso de routing de este repositorio: un
    vocabulario cerrado y presencia binaria. Lo que aporta el C6 es que
    una nota **vacía** no produce un vector de ceros que se lea como «sin
    hallazgos»: produce un vector de ceros **declarado como ausencia**.
    """
    if preparado["empty"]:
        return {
            "features": {p: 0.0 for p in vocabulario},
            # La diferencia entera del corte está en esta clave.
            "signal_present": False,
            "encoder_version": POLITICAS["text_encoder_version"],
        }
    texto = preparado["text"].encode("utf-8")[: int(POLITICAS["text_max_bytes"])]
    palabras = set(re.findall(r"[a-záéíóúñü]+", texto.decode("utf-8", "ignore").lower()))
    return {
        "features": {p: (1.0 if p in palabras else 0.0) for p in vocabulario},
        "signal_present": True,
        "encoder_version": POLITICAS["text_encoder_version"],
    }


def fusionar(tabular: dict[str, Any], texto: dict[str, Any], *,
             tallas_esperadas: tuple[int, int] | None = None) -> dict[str, Any]:
    """Concatena las dos modalidades, **comprobando que encajan**.

    Recortar o rellenar para que cuadren produciría un vector que no
    describe a nadie, y la predicción saldría igual de convincente.
    """
    # `None` → 0.0 **y ya está declarado** en `missing_required` /
    # `unknown_categories`, que es lo que abstiene. Filtrar los `None`
    # —el primer intento— encogía el vector cuando faltaba algo, y
    # entonces la fusión se quejaba de dimensiones por un motivo que no
    # era el suyo: el fallo estaría dos pasos antes de donde se ve.
    izquierda = [0.0 if v is None else v for v in tabular["values"].values()]
    derecha = list(texto["features"].values())
    if tallas_esperadas is not None:
        esperada_izq, esperada_der = tallas_esperadas
        if len(izquierda) != esperada_izq or len(derecha) != esperada_der:
            raise DimensionesIncompatibles(
                f"la fusión esperaba {esperada_izq}+{esperada_der} y recibe "
                f"{len(izquierda)}+{len(derecha)}: ajustarla produciría un "
                "vector que no describe a nadie")
    return {
        "vector": izquierda + derecha,
        "tabular_size": len(izquierda),
        "text_size": len(derecha),
        # Que la señal de texto esté o no viaja HASTA el final: una fusión
        # que no lo dijera dejaría al clasificador tratando una nota vacía
        # como una nota sin hallazgos.
        "text_signal_present": texto["signal_present"],
        "fusion_version": POLITICAS["fusion_version"],
    }


def decidir(probabilidades: dict[str, float], *, contexto: dict[str, Any],
            locale: str = "es") -> dict[str, Any]:
    """La salida del caso: una clase, o una ABSTENCIÓN — nunca a medias.

    La decisión la toma la política del 81-C1 (`policy.py`), no un `if`
    aquí: así entra en la traza **con la regla que decidió**, y quien
    audite ve por qué se abstuvo y no solo que se abstuvo.
    """
    from matrixai.pipelines.policy import evaluar_politica

    if not probabilidades:
        raise ValueError("no hay probabilidades sobre las que decidir")
    clase = max(probabilidades, key=lambda k: probabilidades[k])
    confianza = float(probabilidades[clase])

    resultado = evaluar_politica(POLITICA_DE_ABSTENCION, {**contexto, "confianza": confianza})
    permitido = resultado["decision"] == "allow"
    return {
        # Cuando se abstiene NO viaja una clase: que estuviera ahí, aunque
        # fuera «informativa», es lo que alguien acabaría leyendo.
        "outcome": "prediction" if permitido else "abstained",
        "class": clase if permitido else None,
        "confidence": confianza,
        "threshold": float(POLITICAS["confidence_threshold"]),
        "policy": resultado,
        "text_signal_present": contexto.get("hay_senal_de_texto"),
        # LA ADVERTENCIA, donde se lee el resultado.
        "disclaimer": ADVERTENCIA.get(locale, ADVERTENCIA["es"]),
    }


def analisis_de_confianza(
    predicciones: list[tuple[str, float, str]], *, cubos: int = 10
) -> dict[str, Any]:
    """La CALIBRACIÓN del modelo (§16.4), medida — no prometida.

    `predicciones` son tripletas `(clase_predicha, confianza, clase_real)`.
    Devuelve la tabla de fiabilidad por cubo de confianza y el **ECE**
    (error de calibración esperado): cuánto se separa, de media, lo que el
    modelo dice de lo que acierta.

    Para qué importa aquí: el caso se abstiene por debajo de un umbral de
    confianza, y **ese umbral solo significa algo si la confianza está
    calibrada**. Un modelo que dice 0,9 y acierta 0,6 hace que el umbral
    deje pasar justo lo que existía para parar.

    **Y el aviso que va con el número, siempre**: esto se mide sobre datos
    de la MISMA distribución. No dice nada de un caso que el modelo no ha
    visto nunca —ahí puede estar igual de seguro y equivocado—, y por eso
    la calibración **no** es una defensa contra lo que queda fuera de
    distribución. Confundir las dos cosas sería exactamente la media
    verdad tranquilizadora que este contrato persigue.
    """
    if type(cubos) is not int or cubos < 2:
        raise ValueError(f"cubos debe ser un entero >= 2, y es {cubos!r}")
    if not predicciones:
        # Una tabla vacía con un ECE de 0.0 se leería como calibración
        # perfecta, que es lo contrario de «no se ha medido».
        return {"measured": False, "reason": "no hay predicciones que medir",
                "bins": [], "ece": None, "n": 0,
                "scope": "in-distribution only"}

    cubetas: dict[int, list[int]] = {}
    for _, confianza, _ in predicciones:
        if not 0.0 <= float(confianza) <= 1.0:
            raise ValueError(
                f"una confianza fuera de [0,1] ({confianza!r}) no es una "
                "probabilidad, y calibrar sobre ella no significaría nada")
    for predicha, confianza, real in predicciones:
        indice = min(int(float(confianza) * cubos), cubos - 1)
        acumulado = cubetas.setdefault(indice, [0, 0, 0.0])
        acumulado[0] += 1
        acumulado[1] += int(predicha == real)
        acumulado[2] += float(confianza)

    total = len(predicciones)
    tabla: list[dict[str, Any]] = []
    ece = 0.0
    for indice in sorted(cubetas):
        n, aciertos, suma = cubetas[indice]
        acierto = aciertos / n
        confianza_media = suma / n
        ece += (n / total) * abs(acierto - confianza_media)
        tabla.append({
            "from": indice / cubos, "to": (indice + 1) / cubos, "n": n,
            "accuracy": acierto, "mean_confidence": confianza_media,
            # El signo importa: por encima de cero el modelo se queda
            # corto, por debajo se pasa de seguro — y pasarse es el que
            # rompe el umbral de abstención.
            "gap": acierto - confianza_media,
        })
    return {
        "measured": True, "n": total, "bins": tabla, "ece": ece,
        "scope": "in-distribution only",
        "caveat": (
            "medido sobre datos de la misma distribución: no dice nada de un "
            "caso que el modelo no ha visto, donde puede estar igual de "
            "seguro y equivocado. La calibración NO es una defensa contra lo "
            "que queda fuera de distribución."),
    }


def cobertura_por_clase(
    predicciones: list[tuple[str, float, str]], *, umbral: float | None = None
) -> dict[str, Any]:
    """A QUIÉN deja sin respuesta la abstención, clase por clase.

    Existe por un número medido en este mismo caso el 2026-08-20, y es el
    hallazgo más importante del corte:

        se abstiene en el **39 %** del dataset entero
        …pero en el **89,5 %** de los reingresos reales
        …y en el **16,3 %** de los no-reingresos
        y cuando habla, acierta el **94,7 %**

    Ese 94,7 % es un número estupendo conseguido **callándose justo donde
    importa**. Un sistema que solo se pronuncia sobre los pacientes que no
    van a reingresar es inútil para lo que existe, y el porcentaje global
    de abstención —un 39 % que suena prudente— lo esconde entero.

    Por eso la cobertura se mide **por clase** y no en global: una
    abstención repartida por igual es prudencia, y una concentrada en una
    clase es un sesgo con buena prensa.
    """
    if umbral is None:
        umbral = float(POLITICAS["confidence_threshold"])
    if not predicciones:
        return {"measured": False, "reason": "no hay predicciones que medir",
                "by_class": {}, "abstention_rate": None,
                "accuracy_when_answering": None}

    por_clase: dict[str, dict[str, int]] = {}
    abstenidas = respondidas = aciertos = 0
    for predicha, confianza, real in predicciones:
        acumulado = por_clase.setdefault(real, {"n": 0, "abstained": 0, "correct": 0})
        acumulado["n"] += 1
        if float(confianza) < umbral:
            acumulado["abstained"] += 1
            abstenidas += 1
        else:
            respondidas += 1
            aciertos += int(predicha == real)
            acumulado["correct"] += int(predicha == real)

    detalle = {
        clase: {
            **datos,
            "abstention_rate": datos["abstained"] / datos["n"],
            # La cobertura es lo contrario de la abstención, y se enseña
            # también: «cubre el 10 %» se entiende peor que «se calla el
            # 90 %», y las dos son el mismo dato.
            "coverage": 1.0 - (datos["abstained"] / datos["n"]),
        }
        for clase, datos in por_clase.items()
    }
    tasas = [d["abstention_rate"] for d in detalle.values()]
    return {
        "measured": True,
        "threshold": umbral,
        "n": len(predicciones),
        "abstention_rate": abstenidas / len(predicciones),
        "accuracy_when_answering": (aciertos / respondidas) if respondidas else None,
        "by_class": detalle,
        # La diferencia entre clases, con nombre: es lo que hay que mirar.
        "worst_covered_class": max(detalle, key=lambda c: detalle[c]["abstention_rate"]),
        "spread": max(tasas) - min(tasas),
        "caveat": (
            "una tasa de abstención GLOBAL esconde a quién deja sin "
            "respuesta: repartida por igual es prudencia, concentrada en una "
            "clase es un sesgo con buena prensa. Y la exactitud «cuando "
            "responde» sube justamente por callarse los casos difíciles."),
    }
