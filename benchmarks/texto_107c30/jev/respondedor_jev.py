# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C3.0 — condición (4): el ADAPTADOR de bajo nivel para **Jev**
(TypeSafe, `typesafe/jev-1.13-20260917`) como respondedor de preguntas
atómicas, por la Decisions API alfa de OpenRouter. Es la MISMA idea que la
condición (3) (`medir_c30_respondedor.py`, el respondedor local
Qwen2.5-0.5B): una pregunta sí/no por término, sobre el texto de una fila.

**Ver `jev/LEEME.md`** para el formato exacto de la petición/respuesta, con
las URL de origen, y para lo que NO se pudo determinar (el endpoint exacto
tuvo tres lecturas distintas entre fuentes; se deja `base_url`
CONFIGURABLE a propósito).

**Este módulo nunca hace una petición de verdad en las pruebas**
(`tests/test_107_c30_jev.py`): un servidor HTTP falso local imita la
Decisions API. Fuera de las pruebas, tampoco hace ninguna a menos que se le
dé una clave real (`cargar_clave_api`) -- sin ella, se NIEGA.

**Tarea B (casos clínicos) PROHIBIDA aquí, no solo en el medidor**
(`verificar_tarea_permitida`): esta comprobación vive en el adaptador para
que ningún camino que reutilice este módulo -- ni siquiera uno futuro fuera
de `medir_c30_jev.py` -- pueda mandarle datos clínicos a un servicio de
terceros por error. Es la misma idea que "el hueco está en el cableado, no
en el API": la negativa no puede depender de que el llamante se acuerde de
comprobarlo.

**La clave (`OPENROUTER_API_KEY`) nunca se escribe en registros, resultados
ni excepciones.** Se lee UNA vez de `~/.config/matrixai/jev.env` y no se
guarda en ningún atributo de este módulo más que como valor de paso en la
pila de llamadas; `redactar()` es la red de seguridad para cualquier texto
ajeno (cuerpo de una respuesta de error) que en teoría no debería traerla
pero, por si acaso, se sanea antes de construir un mensaje de excepción.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

RAIZ = Path(__file__).resolve().parent  # benchmarks/texto_107c30/jev/

# ---------------------------------------------------------------------------
# lo que fija esta condición (ver LEEME.md para las fuentes de cada número)
# ---------------------------------------------------------------------------

#: Versión FIJADA del modelo -- nunca un alias móvil como "jev-latest", que
#: cambiaría de versión sin que este código se entere. La misma que mide
#: PriorBench y la que declara el encargo.
MODELO_JEV = "typesafe/jev-1.13-20260917"

#: La Decisions API alfa de OpenRouter (LEEME.md, "El endpoint" -- discrepancia
#: entre fuentes, sin resolver sin una petición real). Configurable a
#: propósito: `--base-url` / el parámetro `base_url` de cada función, para
#: que las pruebas apunten a un servidor falso local y para corregirlo sin
#: tocar código el día que haya clave.
BASE_URL_POR_OMISION = "https://openrouter.ai/api/alpha/decisions"

#: Dónde vive la clave y qué variable lee -- nunca en el repo, nunca como
#: argumento de línea de órdenes (acabaría en `ps`).
FICHERO_CLAVE_POR_OMISION = Path.home() / ".config" / "matrixai" / "jev.env"
VAR_CLAVE = "OPENROUTER_API_KEY"

#: Precio citado en LEEME.md (OpenRouter, `typesafe/jev-1.13`): la salida es
#: gratis, se paga por token de ENTRADA.
PRECIO_USD_POR_MILLON_TOKENS_ENTRADA = 0.042
PRECIO_USD_POR_MILLON_TOKENS_SALIDA = 0.0
LIMITE_CONTEXTO_TOKENS = 32_000

#: Supuestos de `--estimar`, NO medidos -- ver LEEME.md "Supuestos de tokens".
SUPUESTO_TOKENS_POR_CARACTER = 0.28
SUPUESTO_TOKENS_FIJOS_POR_PREGUNTA = 70

#: Sin techo oficial de peticiones/minuto publicado por OpenRouter para
#: modelos de pago (LEEME.md); el punto de operación práctico MEDIDO por
#: PriorBench a 8 peticiones concurrentes (10,41/s) -- un supuesto
#: declarado para `--estimar`, no un límite real.
PETICIONES_POR_SEGUNDO_ASUMIDAS = 10.41

TOPE_REINTENTOS_POR_OMISION = 5
ESPERA_BASE_S_POR_OMISION = 1.0
TIMEOUT_S_POR_OMISION = 30.0

#: Las tareas A (noticias falsas) y C (contratos PLACSP) son públicas; la B
#: (casos clínicos) NUNCA pasa por un servicio de terceros. D no está en el
#: encargo de esta condición.
TAREAS_ADMITIDAS_JEV: tuple[str, ...] = ("A", "C")
TAREAS_PROHIBIDAS_JEV: dict[str, str] = {"B": "datos clínicos: nunca con Jev"}


def verificar_tarea_permitida(nombre: str) -> None:
    """Se niega, con su motivo, a cualquier tarea que no sea A o C. B tiene
    un motivo propio y explícito (no un genérico "no admitida"): es la
    comprobación con nombre para el sabotaje "admite B"."""
    if nombre in TAREAS_PROHIBIDAS_JEV:
        raise SystemExit(f"tarea {nombre} PROHIBIDA para Jev: {TAREAS_PROHIBIDAS_JEV[nombre]}")
    if nombre not in TAREAS_ADMITIDAS_JEV:
        raise SystemExit(
            f"tarea {nombre} no admitida para la condición (4) (Jev): solo {TAREAS_ADMITIDAS_JEV}"
        )


# ---------------------------------------------------------------------------
# la clave: SOLO desde el fichero declarado, nunca impresa ni guardada
# ---------------------------------------------------------------------------

def _parsear_env(contenido: str) -> dict[str, str]:
    """Un `.env` mínimo: `VAR=valor` por línea, `#` de comentario, comillas
    simples/dobles opcionales alrededor del valor. No usa ninguna librería
    de terceros para algo tan pequeño."""
    valores: dict[str, str] = {}
    for linea in contenido.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        valores[clave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def cargar_clave_api(ruta: Path = FICHERO_CLAVE_POR_OMISION, *, var: str = VAR_CLAVE) -> str:
    """Lee `var` de `ruta`. Se NIEGA con un motivo claro -- qué fichero, qué
    variable -- si el fichero no existe o la variable falta o está vacía.
    La clave que devuelve no se imprime ni se guarda en ningún atributo de
    este módulo; quien la reciba tiene la misma obligación."""
    if not ruta.is_file():
        raise SystemExit(
            f"falta la clave de OpenRouter para Jev: no existe {ruta}. "
            f"Crea ese fichero con una línea '{var}=<tu clave>' antes de medir la condición (4)."
        )
    clave = _parsear_env(ruta.read_text(encoding="utf-8")).get(var, "").strip()
    if not clave:
        raise SystemExit(
            f"falta la clave de OpenRouter para Jev: {ruta} existe pero no define '{var}' "
            "(o está vacía)."
        )
    return clave


def redactar(texto: str, clave: str) -> str:
    """Sustituye cualquier aparición LITERAL de `clave` por un marcador. Se
    aplica a TODO texto ajeno (cuerpo de una respuesta HTTP de error) antes
    de meterlo en un mensaje de excepción -- que en teoría no debería traer
    la clave (nunca va en el cuerpo, solo en la cabecera `Authorization`),
    pero es la red de seguridad, no la única barrera."""
    if not clave:
        return texto
    return texto.replace(clave, "***CLAVE-REDACTADA***")


# ---------------------------------------------------------------------------
# errores: reintentable (429/5xx) vs fatal (el resto)
# ---------------------------------------------------------------------------

class ErrorJev(Exception):
    """Cualquier error al hablar con Jev/OpenRouter. El mensaje nunca lleva
    la clave (ver `redactar`, aplicada en todo punto donde se construye)."""


class ErrorJevReintentable(ErrorJev):
    """429 (límite de tasa) o 5xx (fallo del proveedor/pasarela): tiene
    sentido reintentar con espera."""


class ErrorJevFatal(ErrorJev):
    """4xx que no es 429 (401 sin clave válida, 400 mal formada, 402 sin
    crédito, 403...): reintentar no lo arregla."""


# ---------------------------------------------------------------------------
# la petición HTTP (un solo intento; `preguntar` añade el reintento)
# ---------------------------------------------------------------------------

def _peticion_decisions(*, base_url: str, modelo: str, estado: str,
                        preguntas: dict[str, dict[str, Any]], clave: str,
                        timeout_s: float) -> dict[str, Any]:
    cuerpo = json.dumps({"model": modelo, "state": estado, "questions": preguntas}).encode("utf-8")
    peticion = urllib.request.Request(
        base_url, data=cuerpo, method="POST",
        headers={"Authorization": f"Bearer {clave}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(peticion, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        cuerpo_error = redactar(e.read().decode("utf-8", errors="replace"), clave)
        mensaje = redactar(f"HTTP {e.code} de {base_url}: {cuerpo_error}", clave)
        if e.code == 429 or e.code >= 500:
            raise ErrorJevReintentable(mensaje) from None
        raise ErrorJevFatal(mensaje) from None
    except urllib.error.URLError as e:
        raise ErrorJevReintentable(redactar(f"error de red hablando con {base_url}: {e}", clave)) from None
    except json.JSONDecodeError as e:
        raise ErrorJevFatal(redactar(f"respuesta no-JSON de {base_url}: {e}", clave)) from None


def preguntar(*, estado: str, preguntas: dict[str, dict[str, Any]], clave: str,
             modelo: str = MODELO_JEV, base_url: str = BASE_URL_POR_OMISION,
             timeout_s: float = TIMEOUT_S_POR_OMISION,
             tope_reintentos: int = TOPE_REINTENTOS_POR_OMISION,
             espera_base_s: float = ESPERA_BASE_S_POR_OMISION,
             dormir: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Una petición con una o más preguntas `noul` sobre el MISMO `estado`
    (el batching nativo de Jev -- LEEME.md, "nunca encadenar llamadas": una
    fila con 5 términos manda UNA petición con 5 preguntas, no 5
    peticiones). Reintenta con espera exponencial (`espera_base_s * 2**n`)
    ante 429/5xx, hasta `tope_reintentos` veces; agotado el tope, se rinde
    (propaga `ErrorJevReintentable`). Un error NO reintentable (401, 400,
    402...) se propaga en el PRIMER intento, sin esperar nada."""
    intento = 0
    while True:
        try:
            return _peticion_decisions(base_url=base_url, modelo=modelo, estado=estado,
                                       preguntas=preguntas, clave=clave, timeout_s=timeout_s)
        except ErrorJevReintentable:
            intento += 1
            if intento > tope_reintentos:
                raise
            dormir(espera_base_s * (2 ** (intento - 1)))


# ---------------------------------------------------------------------------
# análisis de la respuesta `noul`: sí -> 1, no -> 0, lo demás -> faltante
# ---------------------------------------------------------------------------

def analizar_respuesta_noul(respuesta_pregunta: Any) -> int | None:
    """Convierte la respuesta de UNA pregunta `noul` a 1 (sí) / 0 (no) / None
    (faltante). `noul` es un flotante 0..1 ("0 significa no y 1 significa
    sí", LEEME.md); se umbraliza en 0,5, el propio punto medio de la
    documentación. Faltante CONTADO -- no descartado en silencio -- cuando
    la respuesta no tiene la forma esperada: ausente, `type` distinto de
    `"noul"`, valor no numérico (incluido `bool`, que en Python es subclase
    de `int` y colaría como 0/1 sin esta comprobación explícita), o fuera de
    `[0, 1]`."""
    if not isinstance(respuesta_pregunta, dict):
        return None
    if respuesta_pregunta.get("type") != "noul":
        return None
    valor = respuesta_pregunta.get("noul")
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    valor = float(valor)
    if not (0.0 <= valor <= 1.0):
        return None
    return 1 if valor >= 0.5 else 0


# ---------------------------------------------------------------------------
# caché por (row_id, término) -- un fichero por tarea, ver medir_c30_jev.py
# ---------------------------------------------------------------------------

def _escribir_json_atomico(ruta: Path, datos: Any) -> None:
    """`os.replace`, nunca `write_text` a secas -- el mismo patrón que
    `medir_c30_respondedor._escribir_json_atomico` (el `JSONDecodeError` de
    `estudio_status` en producción fue justo esto: un lector a medio
    escribir viendo un JSON roto). Se repite aquí, sin importar ese módulo,
    para que este adaptador HTTP no arrastre lightgbm/sklearn/matrixai_engines
    solo por escribir una caché."""
    tmp = ruta.with_name(ruta.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(datos, indent=None, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, ruta)


class CacheRespuestasJev:
    """Respuestas ya dadas, por (row_id, término). El fichero YA está
    limitado a una tarea (`respuestas_c30_jev_<tarea>.json`, decidido por
    quien construye la ruta), así que la tarea no hace falta en la clave --
    a diferencia de `medir_c30_respondedor.CacheRespondedor`, que comparte un
    único fichero entre A y B. Se escribe de forma ATÓMICA tras cada FILA
    (todas sus preguntas pendientes en una sola petición), no tras cada
    pregunta suelta."""

    def __init__(self, ruta: Path) -> None:
        self.ruta = ruta
        self._datos: dict[str, dict[str, Any]] = {}
        if ruta.is_file():
            try:
                self._datos = json.loads(ruta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._datos = {}

    @staticmethod
    def _clave(row_id: str, termino: str) -> str:
        return f"{row_id}\x1f{termino}"

    def obtener(self, row_id: str, termino: str) -> dict[str, Any] | None:
        return self._datos.get(self._clave(row_id, termino))

    def fijar(self, row_id: str, termino: str, entrada: dict[str, Any]) -> None:
        self._datos[self._clave(row_id, termino)] = entrada

    def persistir(self) -> None:
        _escribir_json_atomico(self.ruta, self._datos)

    def __len__(self) -> int:
        return len(self._datos)


# ---------------------------------------------------------------------------
# una fila: todos sus términos pendientes, en UNA petición, con caché
# ---------------------------------------------------------------------------

def responder_terminos(
    *, tarea: str, row_id: str, texto: str, terminos: list[str],
    texto_pregunta: Callable[[str], str], cache: CacheRespuestasJev, clave: str,
    modelo: str = MODELO_JEV, base_url: str = BASE_URL_POR_OMISION,
    timeout_s: float = TIMEOUT_S_POR_OMISION,
    tope_reintentos: int = TOPE_REINTENTOS_POR_OMISION,
    espera_base_s: float = ESPERA_BASE_S_POR_OMISION,
    dormir: Callable[[float], None] = time.sleep,
) -> dict[str, int | None]:
    """Las respuestas a `terminos` para una fila, reusando la caché para los
    que ya se preguntaron y mandando el RESTO en UNA sola petición a Jev
    (nunca una petición por término). `texto_pregunta(termino)` construye el
    texto de `instructions` -- inyectado, no fijado aquí, para que este
    adaptador no decida la plantilla de pregunta (esa es responsabilidad del
    medidor, `medir_c30_jev.py`, que reutiliza la LITERAL de (3)).

    Se niega, vía `verificar_tarea_permitida`, si `tarea` es B o cualquier
    otra no admitida -- ANTES de tocar la red."""
    verificar_tarea_permitida(tarea)
    pendientes = [t for t in terminos if cache.obtener(row_id, t) is None]
    if pendientes:
        # Los identificadores de la petición son `q0`, `q1`… y NO los términos: un término es
        # una palabra del texto (con tildes, eñes o lo que traiga) y la API no documenta qué
        # admite como identificador. Un 400 por eso se vería en la primera llamada de pago.
        ids = {f"q{i}": t for i, t in enumerate(pendientes)}
        preguntas = {qid: {"type": "noul", "instructions": texto_pregunta(t)} for qid, t in ids.items()}
        datos = preguntar(estado=texto, preguntas=preguntas, clave=clave, modelo=modelo,
                          base_url=base_url, timeout_s=timeout_s, tope_reintentos=tope_reintentos,
                          espera_base_s=espera_base_s, dormir=dormir)
        respuestas = datos.get("answers", {})
        for qid, t in ids.items():
            valor = analizar_respuesta_noul(respuestas.get(qid))
            cache.fijar(row_id, t, {"noul_bruto": respuestas.get(qid), "valor": valor})
        cache.persistir()
    return {t: cache.obtener(row_id, t)["valor"] for t in terminos}
