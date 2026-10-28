# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""109-C3 — el expediente clínico de un paquete: de dónde sale cada frase.

QUÉ RESUELVE. La ficha TRIPOD+AI (`tripod.py`) y los huecos PROBAST+AI
(`probast.py`) leen LO MISMO: el manifiesto del run, el perfil clínico de
109-C2 y lo que el equipo haya declarado. Escribir dos lectores acabaría con
dos verdades —el invariante que más veces ha costado un fallo en este
repositorio—, así que el lector es uno y los dos informes lo usan.

LOS TRES ESTADOS, Y NO HAY UN CUARTO (invariante 6 del 109).

* `MEDIDO` — lo calculó el core y está en un fichero del paquete.
* `DECLARADO` — lo escribió una persona; viaja con quién lo declaró.
* `FALTA` — no está, y se dice **qué campo lo sostendría**.

**No existe «bajo riesgo de sesgo», ni alto, ni poco claro.** Un juicio de
riesgo de sesgo lo emite un revisor humano leyendo el estudio; un programa que
lo imprimiese estaría firmando por él. Aquí no es una convención de estilo: es
`ESTADOS`, cerrado, comprobado al construir cada campo, y
`sin_veredicto_de_riesgo()`, que revienta si el texto renderizado llega a
contener el vocabulario de veredicto del instrumento.

EL ESTADO NO SE PASA A MANO: LO DICE EL FICHERO DEL QUE SALE EL DATO. Un
argumento `estado=` invitaría a que alguien marcase `MEDIDO` algo que escribió
una persona, que es justo la confusión que este corte existe para impedir. El
manifiesto y el perfil los escribe el core (`MEDIDO`); `team_declaration.json`
lo escribe el equipo (`DECLARADO`). La única excepción es cerrada, corta y está
escrita: `_DECLARADOS_EN_EL_PERFIL`, los campos que el perfil **registra** pero
que declaró una persona (los umbrales y quién los declaró, la prevalencia de la
población destino, quién predefinió los subgrupos, la política de ausentes y si
los datos son sintéticos), y `_DECLARADOS_EN_EL_MANIFIESTO`, el modo de un
estudio, que elige quien lo lanza, y los motores que ese modo deja fuera.

NO FABRICAR (invariante 5). Población, uso previsto, criterios de inclusión,
definición y momento del desenlace, umbrales y proceso actual los declara el
equipo. Si no están, el campo sale `FALTA` **con su ruta**, nunca relleno de
oficio y nunca en blanco. Un `Campo` en estado `FALTA` no puede llevar valor, y
uno `MEDIDO` o `DECLARADO` no puede no llevarlo: lo comprueba `__post_init__`.

EL ALCANCE DE VALIDACIÓN SE RE-DERIVA, NO SE COPIA. `PerfilClinico.a_json()`
guarda `alcance`, pero también guarda `evidencia` y `diseno`, que es de donde
sale. Aquí se vuelve a derivar con la misma función del core
(`alcance_de_validacion`) y **se compara con lo guardado**: si no coinciden, el
paquete se editó por fuera y eso se dice en vez de creerle al campo.

Y UNA COSA QUE NO SE COPIA A PROPÓSITO: la prosa de la ficha. Este módulo
publica el token del alcance y **los dos campos de los que se deriva**, que es lo
que se puede afirmar y re-derivar; una frase redactada no se puede comprobar
contra nada.
*(Historia, porque este párrafo daba antes OTRO motivo: la redacción de
`internal_only` de `perfil_clinico.py` afirmaba «no hay separación temporal»
también cuando el diseño SÍ era temporal —6 de las 24 combinaciones, auditoría
del 2026-09-15—. **Se reparó el 2026-09-16** y ya dice que la partición separa
por tiempo aunque la etiqueta de evidencia no la sostenga. El motivo de no
copiar la prosa sigue siendo bueno por sí solo; el defecto que lo ilustraba, ya
no existe.)*

STDLIB PURO, como el resto del paquete.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DECLARADO",
    "ESTADOS",
    "FALTA",
    "MEDIDO",
    "VEREDICTOS_DE_RIESGO",
    "Campo",
    "ExpedienteClinico",
    "ExpedienteNoDisponible",
    "VeredictoDeRiesgo",
    "sin_veredicto_de_riesgo",
]

#: Los tres estados, y no hay un cuarto. Que sea una tupla cerrada y no una
#: cadena libre es lo que hace imposible añadir «bajo riesgo» sin tocar esta
#: línea — y esta línea tiene pruebas con su nombre.
MEDIDO = "medido"
DECLARADO = "declarado"
FALTA = "falta"
ESTADOS = (MEDIDO, DECLARADO, FALTA)

#: Los ficheros del paquete de los que puede salir un campo, y quién los
#: escribe. La clave es el nombre del fichero dentro del paquete; el valor, el
#: estado que le corresponde a TODO lo que salga de él.
FUENTES: dict[str, str] = {
    "reproduce.json": MEDIDO,
    "clinical_profile.json": MEDIDO,
    "team_declaration.json": DECLARADO,
    "data_recipe.txt": MEDIDO,
    "model.mxai": MEDIDO,
}

#: La excepción, cerrada y escrita: campos que el perfil clínico REGISTRA pero
#: que declaró una persona. Coincidencia exacta o prefijo con punto.
_DECLARADOS_EN_EL_PERFIL = (
    "datos_sinteticos",
    "tabla_de_umbrales.declarados_por",
    "tabla_de_umbrales.prevalencia_declarada",
    "segmentos_predefinidos_por",
    "politica_de_faltantes",
)

#: Y la misma excepción en el manifiesto de un ESTUDIO: el MODO lo elige quien lanza el
#: estudio (rápido o completo) y el estudio solo lo REGISTRA; los motores que quedan fuera
#: son lo que ese modo elegido deja fuera, no algo medido (no se midieron). Rotularlos
#: «MEDIDO por el core» —con «no se midieron» en la misma línea— se contradecía
#: (re-auditoría de 138, 26-10). Coincidencia exacta.
_DECLARADOS_EN_EL_MANIFIESTO = (
    "study.mode",
    "study.engines_left_out_by_mode",
)

#: EL VOCABULARIO DE VEREDICTO DEL INSTRUMENTO, en los dos idiomas.
#:
#: Esto NO es un barrido de palabras escogidas a ojo: son las etiquetas con las
#: que PROBAST cierra un dominio, que es exactamente lo que el invariante 6
#: prohíbe que escriba un programa. Decir «riesgo de sesgo» está permitido y
#: hace falta —hay que nombrar la herramienta—; lo que no se puede es
#: CALIFICARLO.
VEREDICTOS_DE_RIESGO: tuple[str, ...] = (
    "bajo riesgo", "riesgo bajo",
    "alto riesgo", "riesgo alto",
    "riesgo moderado", "moderado riesgo",
    "riesgo poco claro", "riesgo no claro", "riesgo incierto",
    "sin riesgo de sesgo", "no hay riesgo de sesgo",
    "baja preocupacion", "alta preocupacion",
    "low risk", "risk is low",
    "high risk", "risk is high",
    "moderate risk", "unclear risk", "risk is unclear",
    "no risk of bias",
    "low concern", "high concern",
)


#: Los motivos que redacta el propio lector, en los dos idiomas. Los del
#: inventario van en `INVENTARIO`; estos son los que no dependen de qué campo se
#: pidió, sino del estado del fichero.
_MOTIVOS: dict[str, dict[str, str]] = {
    "es": {"roto": "no se puede leer", "sin_fichero": "el paquete no trae",
           "sin_campo": "no declara",
           "comparacion_incomparable": "el estudio intentó compararlo con el baseline y no "
                                       "pudo"},
    "en": {"roto": "cannot be read", "sin_fichero": "the package does not ship",
           "sin_campo": "does not declare",
           "comparacion_incomparable": "the study tried to compare it against the baseline "
                                       "and could not"},
}


class ExpedienteNoDisponible(RuntimeError):
    """No hay paquete del que sacar el expediente."""


class VeredictoDeRiesgo(RuntimeError):
    """Un texto de este paquete ha llegado a calificar el riesgo de sesgo.

    No se corrige sobre la marcha ni se tacha la frase: se levanta, porque el
    error no es tipográfico. Alguien ha escrito un juicio que solo puede emitir
    quien revisa el estudio, y borrarlo en silencio dejaría el resto del texto
    sonando como si lo hubiera pensado.
    """


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin nada que no sea letra o número.

    Así `**BAJO** riesgo`, `bajo  riesgo` y `Bajo riesgo.` son la misma cadena:
    el asterisco de Markdown no puede usarse para colar el veredicto.
    """
    plano = unicodedata.normalize("NFKD", texto.lower())
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in plano).split())


def sin_veredicto_de_riesgo(texto: str, *, origen: str) -> str:
    """Devuelve `texto` si no califica el riesgo de sesgo; si lo califica, revienta.

    Es la última puerta antes de que un informe salga de este paquete, y está
    aquí y no en una prueba a propósito: una prueba protege lo que alguien se
    acuerde de probar, y esto tiene que protegerlo aunque la frase entre por una
    traducción nueva dentro de dos años.
    """
    plano = _normalizar(texto)
    for veredicto in VEREDICTOS_DE_RIESGO:
        if veredicto in plano:
            raise VeredictoDeRiesgo(
                f"{origen}: el texto contiene «{veredicto}». Este informe rellena lo "
                f"medido, marca lo declarado y enumera lo que falta, y NUNCA califica "
                f"el riesgo de sesgo: eso lo emite quien revisa el estudio.")
    return texto


@dataclass(frozen=True)
class Campo:
    """Un dato del expediente con la ruta exacta que lo sostiene.

    `ruta` es `fichero.json#camino.punteado`, y no es decoración: el criterio de
    terminado del 109-C3 pide que **cada frase de la ficha trace a un campo**, y
    la forma de cumplirlo es que la frase no se pueda escribir sin él.
    """

    ruta: str
    estado: str
    valor: Any = None
    motivo: str = ""

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS:
            raise ValueError(
                f"estado desconocido: {self.estado!r}. Solo hay tres "
                f"({', '.join(ESTADOS)}) y ninguno califica el riesgo de sesgo.")
        if "#" not in self.ruta:
            raise ValueError(
                f"ruta sin campo: {self.ruta!r}. Se escribe «fichero#campo» para "
                f"que la frase que la use trace a un dato concreto.")
        fichero = self.ruta.split("#", 1)[0]
        if fichero not in FUENTES:
            raise ValueError(
                f"ruta a un fichero que el paquete no tiene: {fichero!r}. "
                f"Los del paquete son {', '.join(sorted(FUENTES))}.")
        # UN HUECO NO LLEVA VALOR Y UN DATO NO ES UN HUECO. Las dos direcciones,
        # porque las dos han sido un fallo: un `FALTA` con valor es un hueco
        # relleno de oficio, y un `MEDIDO` sin valor es una casilla que parece
        # rellena porque tiene rótulo.
        if self.estado == FALTA:
            if self.valor is not None:
                raise ValueError(
                    f"{self.ruta}: un campo que FALTA no puede llevar valor "
                    f"({self.valor!r}). Un hueco relleno es peor que un hueco.")
            if not self.motivo:
                raise ValueError(
                    f"{self.ruta}: un campo que FALTA dice por qué falta.")
        elif self.valor is None:
            raise ValueError(
                f"{self.ruta}: estado {self.estado} sin valor. Si no hay dato, el "
                f"estado es «{FALTA}» y lleva su motivo.")

    @property
    def hay_dato(self) -> bool:
        return self.estado != FALTA

    @property
    def fichero(self) -> str:
        return self.ruta.split("#", 1)[0]

    def a_json(self) -> dict[str, Any]:
        return {"ruta": self.ruta, "estado": self.estado, "valor": self.valor,
                "motivo": self.motivo or None}


class ExpedienteClinico:
    """Lo que un paquete exportado sostiene, con la ruta de cada dato.

    Se construye con `desde_paquete()`. Sin `reproduce.json` no hay expediente:
    un informe sobre un paquete que no registró nada sería un formulario, no un
    informe.
    """

    def __init__(self, bundle: Path, contenido: dict[str, Any]) -> None:
        self.bundle = bundle
        self._contenido = contenido

    # -- construcción --------------------------------------------------------

    @classmethod
    def desde_paquete(cls, bundle_dir: str | Path) -> "ExpedienteClinico":
        bundle = Path(bundle_dir)
        manifiesto = bundle / "reproduce.json"
        if not manifiesto.is_file():
            raise ExpedienteNoDisponible(
                f"{bundle}: no reproduce.json — a reporting record is written from "
                f"what the run recorded, and this package records nothing")
        contenido: dict[str, Any] = {}
        for nombre in ("reproduce.json", "clinical_profile.json",
                       "team_declaration.json"):
            ruta = bundle / nombre
            if not ruta.is_file():
                continue
            try:
                contenido[nombre] = json.loads(ruta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if nombre == "reproduce.json":
                    raise ExpedienteNoDisponible(f"{ruta}: {exc}") from exc
                # UN FICHERO ILEGIBLE NO ES UN FICHERO AUSENTE. Si se tragase el
                # error, el informe diría «no declarado» sobre una declaración
                # que SÍ existe y está rota: media verdad tranquilizadora.
                contenido[nombre] = {"__ilegible__": str(exc)}
        receta = bundle / "data_recipe.txt"
        if receta.is_file():
            contenido["data_recipe.txt"] = receta.read_text(encoding="utf-8").strip()
        return cls(bundle, contenido)

    # -- acceso --------------------------------------------------------------

    def tiene(self, fichero: str) -> bool:
        return fichero in self._contenido

    def ilegible(self, fichero: str) -> str | None:
        datos = self._contenido.get(fichero)
        if isinstance(datos, dict):
            return datos.get("__ilegible__")
        return None

    @property
    def es_de_estudio(self) -> bool:
        """Si el manifiesto es el de un ESTUDIO (lleva el bloque `study` de `manifiesto_de_estudio`)."""
        manifiesto = self._contenido.get("reproduce.json")
        return isinstance(manifiesto, dict) and isinstance(manifiesto.get("study"), dict)

    def inventario(self) -> tuple["Evidencia", ...]:
        """El inventario que corresponde a ESTE paquete: el de siempre, o el de un estudio.

        En un estudio, la línea de las filas lleva su rótulo de estudio y detrás van
        el reparto y el modo; el motivo de que existan está encima de
        `FILAS_DEL_ESTUDIO`. La línea de los motores que dejó fuera el modo solo
        va si el modo los dejó: en el completo no hay ninguno, y «falta» sería un
        hueco inventado."""
        if not self.es_de_estudio:
            return INVENTARIO
        estudio = self._contenido["reproduce.json"]["study"]
        lineas: list[Evidencia] = []
        for evidencia in INVENTARIO:
            if evidencia.clave == "fuente_de_los_datos":
                lineas.append(FUENTE_EN_ESTUDIO)
                continue
            if evidencia.clave == "filas":
                lineas.append(FILAS_DEL_ESTUDIO)
                lineas.extend(self.reparto_del_estudio())
                # 140-C2: CÓMO se repartieron las filas del examen (por filas al azar, por sujeto, por el tiempo), con sus
                # columnas, el hueco y lo que no tiene predicción fuera de pliegue. Solo si el estudio lo declaró: uno anterior
                # al 140 no lo registró y «no disponible» sería un hueco que no sabemos si lo es.
                if self.diseno_del_reparto is not None:
                    lineas.append(REPARTO_POR_DISENO)
                continue
            lineas.append(evidencia)
            if evidencia.clave == "metricas":
                lineas.append(MODO_DEL_ESTUDIO)
                if estudio.get("mode") == "rapido":
                    lineas.append(FUERA_POR_EL_MODO)
                lineas.extend((MOTOR_ENTREGADO, CALIBRACION_Y_UMBRAL))
                # Reparación de la re-auditoría del 139 (I1): las ÉPOCAS de una red entregada con épocas fijas. Solo si el
                # estudio lo dijo (un árbol no tiene épocas: «falta» sería un hueco inventado).
                if self.entrenamiento_de_la_red is not None:
                    lineas.append(ENTRENAMIENTO_DE_LA_RED)
        return tuple(lineas)

    @property
    def entrenamiento_de_la_red(self) -> dict[str, Any] | None:
        """`study.network_training` del manifiesto (139-C4: una red reentrenada con épocas fijas), o `None`."""
        bloque = ((self._contenido.get("reproduce.json") or {}).get("study") or {}).get("network_training")
        return bloque if isinstance(bloque, dict) else None

    @property
    def diseno_del_reparto(self) -> dict[str, Any] | None:
        """`study.split` del manifiesto (140-C2: diseño del reparto, columnas, hueco, cobertura y límites), o `None`."""
        bloque = ((self._contenido.get("reproduce.json") or {}).get("study") or {}).get("split")
        return bloque if isinstance(bloque, dict) else None

    def reparto_del_estudio(self) -> tuple["Evidencia", ...]:
        """Las líneas del reparto de las filas de un estudio: las tres de siempre y, si el estudio contó la cola de
        parada temprana (139-C2), la cola. Tras `filas_que_entrenaron_el_modelo`, que es de donde sale su resta."""
        lineas = list(REPARTO_DEL_ESTUDIO)
        reparto = (self._contenido.get("reproduce.json") or {}).get("study", {}).get("rows")
        if isinstance(reparto, dict) and "early_stopping_tail" in reparto:
            # Reparación de la re-auditoría del 139 (I2): con una red de épocas fijas la cola NO decide cuándo parar (el
            # estudio lo declara: `early_stopping_tail_decides: false`). Mismo dato y ruta, otro rótulo: el que es verdad.
            red = self.entrenamiento_de_la_red
            lineas.append(COLA_DE_UNA_RED_CON_EPOCAS_FIJAS
                          if red is not None and red.get("early_stopping_tail_decides") is False
                          else COLA_DE_PARADA_TEMPRANA)
        return tuple(lineas)

    @property
    def hay_perfil(self) -> bool:
        """Si el paquete trae el perfil clínico de 109-C2."""
        return (self.tiene("clinical_profile.json")
                and self.ilegible("clinical_profile.json") is None)

    def campo(self, ruta: str, *, motivo_si_falta: str = "",
              locale: str = "en") -> Campo:
        """El campo en `fichero#camino.punteado`, con su estado y su ruta.

        El estado **no se pasa**: sale del fichero (`FUENTES`), con la excepción
        cerrada de `_DECLARADOS_EN_EL_PERFIL`.

        `locale` solo se usa para los motivos que este método redacta por su
        cuenta —un fichero roto, un fichero que no está—: media aplicación
        traducida se ve enseguida, y un motivo en castellano dentro de la ficha
        en inglés es exactamente eso.
        """
        idioma = "es" if str(locale or "en").strip().lower() == "es" else "en"
        fichero, camino = ruta.split("#", 1)
        if fichero not in FUENTES:
            raise ValueError(f"fichero desconocido en la ruta: {fichero!r}")
        roto = self.ilegible(fichero)
        if roto is not None:
            return Campo(ruta, FALTA,
                         motivo=f"{fichero} {_MOTIVOS[idioma]['roto']}: {roto}")
        if fichero not in self._contenido:
            return Campo(ruta, FALTA,
                         motivo=(motivo_si_falta
                                 or f"{_MOTIVOS[idioma]['sin_fichero']} {fichero}"))
        valor: Any = self._contenido[fichero]
        if camino:
            for paso in camino.split("."):
                if not isinstance(valor, dict) or paso not in valor:
                    valor = None
                    break
                valor = valor[paso]
        if valor is None or valor == [] or valor == {} or valor == "":
            return Campo(ruta, FALTA,
                         motivo=(motivo_si_falta
                                 or f"{fichero} {_MOTIVOS[idioma]['sin_campo']} "
                                    f"{camino}"))
        return Campo(ruta, self._estado_de(fichero, camino), valor=valor)

    def _estado_de(self, fichero: str, camino: str) -> str:
        if fichero == "clinical_profile.json":
            for declarado in _DECLARADOS_EN_EL_PERFIL:
                if camino == declarado or camino.startswith(declarado + "."):
                    return DECLARADO
        if fichero == "reproduce.json" and camino in _DECLARADOS_EN_EL_MANIFIESTO:
            return DECLARADO
        return FUENTES[fichero]

    def resolver(self, locale: str
                 ) -> tuple[tuple[tuple["Evidencia", Campo], ...], tuple[str, ...]]:
        """Todo el inventario resuelto contra ESTE paquete, más los avisos.

        Los dos informes salen de aquí. Que sea una sola pasada es lo que hace
        imposible que la ficha TRIPOD+AI diga que hay calibración y los huecos
        PROBAST+AI digan que falta: leen el mismo campo del mismo fichero.
        """
        idioma = "es" if str(locale or "en").strip().lower() == "es" else "en"
        pares: list[tuple[Evidencia, Campo]] = []
        avisos: list[str] = []
        for evidencia in self.inventario():
            motivo = evidencia.motivo_si_falta[idioma]
            if evidencia.clave == "alcance_de_validacion":
                campo, aviso = self.alcance_de_validacion()
                if aviso:
                    avisos.append(aviso)
                if not campo.hay_dato:
                    campo = Campo(evidencia.ruta, FALTA, motivo=motivo)
            elif evidencia.clave == "comparacion_con_el_baseline":
                campo = self.comparacion_con_el_baseline(motivo, idioma)
            elif evidencia.clave == "politica_de_faltantes":
                campo = self.campo(evidencia.ruta, motivo_si_falta=motivo, locale=idioma)
                # Sin política del equipo, sigue FALTANDO (nadie la declaró); pero si el
                # paquete registra lo que hizo su preparación, se dice eso y no «no consta».
                if not campo.hay_dato and self.ilegible("clinical_profile.json") is None:
                    preparacion = self.ausentes_segun_la_preparacion(idioma)
                    if preparacion:
                        campo = Campo(evidencia.ruta, FALTA, motivo=preparacion)
            else:
                campo = self.campo(evidencia.ruta, motivo_si_falta=motivo,
                                   locale=idioma)
            pares.append((evidencia, campo))
        return tuple(pares), tuple(avisos)

    # -- lo derivado ---------------------------------------------------------

    def ausentes_segun_la_preparacion(self, idioma: str) -> str | None:
        """Lo que la PREPARACIÓN del paquete registra de los valores ausentes, o `None`.

        Un paquete de estudio no trae `clinical_profile.json#politica_de_faltantes` (la
        política que declara un equipo), pero sí `politica_preparacion.json`: por columna,
        qué parte faltaba en las filas con que se ajustó y qué se hace con un ausente. La
        ficha decía «este paquete no declara ninguna política de valores ausentes, así que
        no se puede afirmar qué se hizo con ellos» con ese fichero al lado (re-auditoría de
        138, 26-10: `presión`, 8 % ausente, tratada de forma nativa). Se dice lo que el
        fichero registra, leído tal cual, y nada más: sin él, `None` y el motivo de siempre.
        La regla de qué se hace con un ausente es la de `preparacion.transformar_fila`.
        """
        es = idioma == "es"
        ruta = self.bundle / "politica_preparacion.json"
        if not ruta.is_file():
            return None
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return (f"politica_preparacion.json {_MOTIVOS[idioma]['roto']}: {exc}")
        columnas = datos.get("columnas") if isinstance(datos, dict) else None
        if not isinstance(columnas, list):
            return None
        n = datos.get("filas_de_train_efectivas")
        cabeza = (("el equipo no ha declarado una política de datos ausentes; lo que hace "
                   "con ellos la preparación lo registra el paquete en "
                   "`politica_preparacion.json#columnas`")
                  if es else
                  ("no team has declared a missing-data policy; what the preparation does "
                   "with missing values is recorded by the package in "
                   "`politica_preparacion.json#columnas`"))
        donde = ((f", ajustada en {n} filas de entrenamiento" if es
                  else f", fitted on {n} training rows")
                 if isinstance(n, int) and not isinstance(n, bool) else "")
        partes: list[str] = []
        for col in columnas:
            if not isinstance(col, dict):
                continue
            p = col.get("proporcion_faltante")
            if not isinstance(p, (int, float)) or isinstance(p, bool) or p <= 0:
                continue
            nombre = str(col.get("columna"))
            pct = _numero(round(p * 100, 1))
            pct = (pct.replace(".", ",") if es else pct) + " %"
            if col.get("tipo") == "categorica":
                trato = ("un ausente es una categoría propia" if es
                         else "a missing value is a category of its own")
            elif col.get("admite_nativo") is True:
                trato = ("se deja ausente para el motor, que lo admite" if es
                         else "it is left missing for the engine, which accepts it")
            elif isinstance(col.get("mediana"), (int, float)):
                med = _numero(col["mediana"])
                trato = (f"se rellena con la mediana de entrenamiento ({med})" if es
                         else f"it is filled with the training median ({med})")
            else:
                trato = ("el fichero no dice qué se hace con él" if es
                         else "the file does not say what is done with it")
            partes.append(f"{nombre}: {pct} {'ausente' if es else 'missing'}; {trato}")
        if not partes:
            cola = ("en esas filas no faltaba ningún valor" if es
                    else "no value was missing in those rows")
            return f"{cabeza}{donde}: {cola}"
        return f"{cabeza}{donde}. " + ". ".join(partes)

    def comparacion_con_el_baseline(self, motivo_generico: str, idioma: str) -> Campo:
        """La comparación con el baseline, o POR QUÉ no la hay, dicho con lo que el
        paquete trae y no con un genérico.

        Re-auditoría del 2026-09-18: una comparación `incomparable` (el estudio la
        intentó y las filas no casaban) salía MEDIDA, con «auroc — · incomparable»
        como valor, y un perfil que traía el motivo de su ausencia
        (`sin_comparacion_con_el_baseline`) recibía el «o esto o paquete viejo» de
        siempre. Las dos cosas son lo contrario de declarar lo que pasó: la primera
        cuenta como medido algo que no se pudo medir, la segunda calla un motivo
        que está en el paquete."""
        ruta = "clinical_profile.json#comparacion_con_el_baseline"
        campo = self.campo(ruta, motivo_si_falta=motivo_generico, locale=idioma)
        if campo.hay_dato:
            valor = campo.valor
            if isinstance(valor, dict) and valor.get("veredicto") == "incomparable":
                razon = (valor.get("undefined_reason") or {}).get(idioma) or ""
                texto = _MOTIVOS[idioma]["comparacion_incomparable"]
                return Campo(ruta, FALTA, motivo=f"{texto}: {razon}" if razon else texto)
            return campo
        sin = self.campo("clinical_profile.json#sin_comparacion_con_el_baseline")
        if sin.hay_dato:
            from matrixai.estudio.perfil_clinico import _T  # noqa: PLC0415
            texto = _T[idioma].get(f"sin_comparacion_{sin.valor}")
            if texto:
                return Campo(ruta, FALTA, motivo=texto)
        return campo

    def alcance_de_validacion(self) -> tuple[Campo, str]:
        """El alcance RE-DERIVADO de `evidencia` y `diseno`, y lo que discrepe.

        Devuelve el campo y un aviso: vacío si lo guardado coincide con lo
        derivado, y con lo que dice cada uno si no. No se cree al campo
        `alcance` guardado: dos sitios declarando lo mismo acaban divergiendo, y
        aquí uno de los dos se puede editar a mano después de exportar.
        """
        evidencia = self.campo("clinical_profile.json#evidencia")
        diseno = self.campo("clinical_profile.json#diseno")
        if not (evidencia.hay_dato and diseno.hay_dato):
            falta = evidencia if not evidencia.hay_dato else diseno
            return Campo("clinical_profile.json#evidencia+diseno", FALTA,
                         motivo=falta.motivo), ""
        from matrixai.estudio.perfil_clinico import (  # noqa: PLC0415
            alcance_de_validacion as derivar,
        )

        derivado = derivar(evidencia=str(evidencia.valor), diseno=str(diseno.valor))
        guardado = self._contenido["clinical_profile.json"].get("alcance")
        aviso = ""
        if guardado is not None and guardado != derivado:
            aviso = (f"el paquete guarda «{guardado}» y de evidencia="
                     f"{evidencia.valor} + diseno={diseno.valor} se deriva "
                     f"«{derivado}»")
        return Campo("clinical_profile.json#evidencia+diseno", MEDIDO,
                     valor={"alcance": derivado, "evidencia": evidencia.valor,
                            "diseno": diseno.valor}), aviso


# ---------------------------------------------------------------------------
# El inventario: qué se puede sostener, con la ruta que lo sostiene
# ---------------------------------------------------------------------------

#: Los dominios con los que se organiza lo que sostiene un paquete.
#:
#: PROBAST agrupa sus preguntas señal en participantes, predictores, desenlace
#: y análisis, y añade la aplicabilidad. **La asignación de CADA entrada de este
#: inventario a su dominio es NUESTRA**, no del instrumento: sirve para que
#: quien rellene el formulario oficial encuentre junto lo que le hace falta, y
#: no sustituye a leerlo. Lo dice también la propia ficha.
DOMINIOS = ("participantes", "predictores", "desenlace", "analisis", "aplicabilidad")


@dataclass(frozen=True)
class Evidencia:
    """Una cosa que un informe de reporte pregunta, y dónde vive la respuesta."""

    clave: str
    dominio: str
    ruta: str
    rotulo: dict[str, str]
    motivo_si_falta: dict[str, str]
    resumen: str = "tal_cual"

    def __post_init__(self) -> None:
        if self.dominio not in DOMINIOS:
            raise ValueError(f"dominio desconocido: {self.dominio!r}")
        for mapa, nombre in ((self.rotulo, "rotulo"),
                             (self.motivo_si_falta, "motivo_si_falta")):
            if set(mapa) != {"es", "en"}:
                raise ValueError(
                    f"{self.clave}.{nombre}: hacen falta las dos redacciones (es, en). "
                    f"Media aplicación traducida se ve, y se ve enseguida.")


def _e(clave: str, dominio: str, ruta: str, es: str, en: str,
       falta_es: str, falta_en: str, resumen: str = "tal_cual") -> Evidencia:
    return Evidencia(clave=clave, dominio=dominio, ruta=ruta,
                     rotulo={"es": es, "en": en},
                     motivo_si_falta={"es": falta_es, "en": falta_en},
                     resumen=resumen)


_LO_ESCRIBE_EL_EQUIPO_ES = ("lo declara el equipo; este informe lo pide, no lo "
                            "inventa")
_LO_ESCRIBE_EL_EQUIPO_EN = ("the team declares this; this report asks for it, it "
                            "does not invent it")
_SIN_PERFIL_ES = "el paquete no trae el perfil clínico"
_SIN_PERFIL_EN = "the package ships no clinical profile"

#: EL INVENTARIO. Es la única lista, y los dos informes la leen: la ficha
#: TRIPOD+AI escribe con ella sus secciones y los huecos PROBAST+AI enumeran con
#: ella lo que hay por dominio. Añadir aquí una línea la añade en los dos.
INVENTARIO: tuple[Evidencia, ...] = (
    _e("uso_previsto", "aplicabilidad", "team_declaration.json#uso_previsto",
       "Uso previsto", "Intended use",
       _LO_ESCRIBE_EL_EQUIPO_ES, _LO_ESCRIBE_EL_EQUIPO_EN),
    _e("poblacion", "participantes", "team_declaration.json#poblacion",
       "Población destinataria", "Target population",
       _LO_ESCRIBE_EL_EQUIPO_ES, _LO_ESCRIBE_EL_EQUIPO_EN),
    _e("criterios_de_inclusion", "participantes",
       "team_declaration.json#criterios_de_inclusion",
       "Criterios de inclusión", "Inclusion criteria",
       _LO_ESCRIBE_EL_EQUIPO_ES, _LO_ESCRIBE_EL_EQUIPO_EN, "lista"),
    _e("fuente_de_los_datos", "participantes", "reproduce.json#generation.mode",
       "Fuente de los datos", "Data source",
       "el paquete no declara con qué modo se generó",
       "the package does not declare which generation mode produced it"),
    _e("filas", "participantes", "reproduce.json#artifacts.dataset.rows",
       "Filas que entrenaron", "Rows that trained",
       "el manifiesto no registra el tamaño del dataset",
       "the manifest records no dataset size"),
    _e("huella_del_dataset", "participantes",
       "reproduce.json#artifacts.dataset.sha256",
       "Huella del dataset (sha256)", "Dataset digest (sha256)",
       "el manifiesto no registra la huella del dataset",
       "the manifest records no dataset digest"),
    _e("datos_sinteticos", "participantes",
       "clinical_profile.json#datos_sinteticos",
       "¿Datos sintéticos?", "Synthetic data?",
       _SIN_PERFIL_ES, _SIN_PERFIL_EN, "booleano"),
    _e("predictores", "predictores", "reproduce.json#generation.field_types",
       "Predictores (entradas)", "Predictors (inputs)",
       "el run no registró los tipos de las columnas de entrada",
       "the run did not record the input field types", "lista"),
    _e("columnas_excluidas", "predictores",
       "reproduce.json#generation.excluded_identifiers",
       "Columnas excluidas", "Excluded columns",
       "el run no registró ninguna columna excluida",
       "the run recorded no excluded column", "lista"),
    _e("definicion_del_desenlace", "desenlace",
       "team_declaration.json#definicion_del_desenlace",
       "Definición del desenlace", "Outcome definition",
       _LO_ESCRIBE_EL_EQUIPO_ES, _LO_ESCRIBE_EL_EQUIPO_EN),
    _e("momento_del_desenlace", "desenlace",
       "team_declaration.json#momento_del_desenlace",
       "Momento del desenlace", "Outcome timing",
       _LO_ESCRIBE_EL_EQUIPO_ES, _LO_ESCRIBE_EL_EQUIPO_EN),
    _e("prevalencia_declarada", "desenlace",
       "clinical_profile.json#tabla_de_umbrales.prevalencia_declarada",
       "Prevalencia declarada (población destino)",
       "Declared prevalence (target population)",
       "nadie ha declarado la prevalencia: sin ella no se publican PPV ni NPV",
       "nobody declared prevalence: without it PPV and NPV are not published"),
    _e("prevalencia_observada", "desenlace",
       "clinical_profile.json#tabla_de_umbrales.prevalencia_observada",
       "Prevalencia observada en la muestra", "Prevalence observed in the sample",
       _SIN_PERFIL_ES, _SIN_PERFIL_EN),
    _e("umbrales", "analisis", "clinical_profile.json#tabla_de_umbrales.filas",
       "Umbrales medidos", "Thresholds measured",
       "sin umbrales declarados no hay tabla: no se pone un 0,5 de cortesía",
       "with no declared thresholds there is no table: no 0.5 out of courtesy",
       "umbrales"),
    _e("umbrales_declarados_por", "analisis",
       "clinical_profile.json#tabla_de_umbrales.declarados_por",
       "Umbrales declarados por", "Thresholds declared by",
       _LO_ESCRIBE_EL_EQUIPO_ES, _LO_ESCRIBE_EL_EQUIPO_EN),
    _e("calibracion", "analisis", "clinical_profile.json#calibracion.ece",
       "Calibración (ECE)", "Calibration (ECE)",
       "no se ha medido la calibración; no medirla no la vuelve buena",
       "calibration was not measured; not measuring it does not make it good"),
    _e("curva_de_decision", "analisis",
       "clinical_profile.json#curva_de_decision.puntos",
       "Curva de decisión (beneficio neto)", "Decision curve (net benefit)",
       "no se ha medido la curva de decisión",
       "the decision curve was not measured", "puntos_dca"),
    _e("subgrupos", "analisis", "clinical_profile.json#segmentos",
       "Subgrupos medidos", "Subgroups measured",
       "no se ha medido ningún subgrupo",
       "no subgroup was measured", "segmentos"),
    _e("subgrupos_predefinidos_por", "analisis",
       "clinical_profile.json#segmentos_predefinidos_por",
       "Subgrupos predefinidos por", "Subgroups predefined by",
       "nadie los ha predefinido: sin equipo que los declare de antemano, lo que "
       "se mida es exploratorio",
       "nobody predefined them: with no team declaring them beforehand, whatever "
       "is measured is exploratory"),
    _e("politica_de_faltantes", "analisis",
       "clinical_profile.json#politica_de_faltantes",
       "Política de datos ausentes", "Missing-data policy",
       "no se declara ninguna política de datos ausentes: lo que se hizo con "
       "ellos no consta",
       "no missing-data policy is declared: what was done with them is not on "
       "record"),
    _e("metricas", "analisis", "reproduce.json#metrics",
       "Rendimiento medido", "Measured performance",
       "el paquete no publica ninguna métrica",
       "the package publishes no metric", "metricas"),
    _e("alcance_de_validacion", "analisis",
       "clinical_profile.json#evidencia+diseno",
       "Alcance de la validación", "Validation scope",
       _SIN_PERFIL_ES, _SIN_PERFIL_EN, "alcance"),
    _e("comparacion_con_el_baseline", "analisis",
       "clinical_profile.json#comparacion_con_el_baseline",
       "Comparación con el baseline", "Comparison against the baseline",
       "el perfil clínico no la trae: o el estudio no pudo plantearla —y el "
       "perfil dice por qué en `sin_comparacion_con_el_baseline`— o el paquete "
       "se exportó con una versión que todavía no la escribía",
       "the clinical profile does not ship it: either the study could not pose "
       "it —and the profile says why in `sin_comparacion_con_el_baseline`— or "
       "the package was exported by a version that did not write it yet", "comparacion"),
    _e("proceso_actual", "analisis", "team_declaration.json#proceso_actual",
       "Comparación con el proceso actual", "Comparison against current practice",
       "no se ha declarado el proceso actual con el que comparar: se dice en vez de "
       "suponerlo",
       "no current practice has been declared to compare against: it is said rather "
       "than assumed"),
)

#: Índice por clave, para que nadie tenga que recorrer la tupla a mano.
POR_CLAVE: dict[str, Evidencia] = {e.clave: e for e in INVENTARIO}

# ---------------------------------------------------------------------------
# Lo que cambia cuando el paquete es de un ESTUDIO (reparación de 138-C1..C3)
# ---------------------------------------------------------------------------
#
# Un manifiesto de estudio (`manifiesto_de_estudio`, con su bloque `study`) pone en
# `artifacts.dataset.rows` las filas del CONJUNTO —las de la huella—, y el modelo
# entregado NO entrenó con todas: hubo una prueba reservada que no vio, y dentro del
# desarrollo se ajustó con una parte. «Filas que entrenaron: 360» con 72 reservadas
# era la ficha diciendo lo contrario de lo que pasó (la auditoría de C1–C3). Para un
# estudio, la MISMA ruta lleva otro rótulo, y el reparto va en sus líneas propias,
# leídas de lo que el estudio contó (nunca deducidas aquí). El camino clásico no
# cambia: sin bloque `study`, el inventario de arriba tal cual.

#: El rótulo de `artifacts.dataset.rows` en un estudio: lo que esa cifra ES ahí.
FILAS_DEL_ESTUDIO = _e(
    "filas", "participantes", "reproduce.json#artifacts.dataset.rows",
    "Filas del conjunto de datos (desarrollo y prueba reservada)",
    "Dataset rows (development and holdout test)",
    "el manifiesto no registra el tamaño del dataset",
    "the manifest records no dataset size")

#: La fuente de los datos en un estudio. `generation.mode` es el modo del GENERADOR sintético, y su motivo de siempre («no
#: declara con qué modo se generó») junto a «Modo del estudio: rápido» se leía como una contradicción: un estudio entrena
#: con el CSV de quien lo trae, y lo que falta es de dónde salió ese CSV.
FUENTE_EN_ESTUDIO = _e(
    "fuente_de_los_datos", "participantes", "reproduce.json#generation.mode",
    "Fuente de los datos", "Data source",
    "el estudio trabaja con el CSV de quien lo hizo, y el paquete no declara de dónde salió",
    "the study works on the CSV of whoever ran it, and the package does not declare where it came from")

_SIN_REPARTO_ES = "el estudio no guardó cómo repartió sus filas"
_SIN_REPARTO_EN = "the study did not record how it split its rows"

#: Las líneas que SOLO tiene un estudio, en el orden en que se escriben, justo después
#: de la de las filas del conjunto.
REPARTO_DEL_ESTUDIO: tuple[Evidencia, ...] = (
    _e("filas_de_desarrollo", "participantes", "reproduce.json#study.rows.development",
       "Filas de desarrollo (validación cruzada y selección)",
       "Development rows (cross-validation and selection)",
       _SIN_REPARTO_ES, _SIN_REPARTO_EN),
    _e("filas_de_prueba_reservada", "participantes", "reproduce.json#study.rows.holdout_test",
       "Filas de la prueba reservada (apartadas antes; ni eligen ni entrenan)",
       "Holdout test rows (set aside first; they neither choose nor train)",
       _SIN_REPARTO_ES, _SIN_REPARTO_EN),
    _e("filas_que_entrenaron_el_modelo", "participantes",
       "reproduce.json#study.rows.trained_the_delivered_model",
       "Filas que entrenaron el modelo entregado", "Rows that trained the delivered model",
       _SIN_REPARTO_ES, _SIN_REPARTO_EN),
)

#: La cola con la que el modelo entregado decide cuándo parar (139-C2): filas de DESARROLLO que no entrenan. Solo va en el
#: inventario si el estudio la contó (`study.rows.early_stopping_tail`): un estudio guardado antes del 139 no la trae, y un
#: motor sin parada temprana no tiene cola; en ninguno de los dos «no disponible» sería un hueco real.
COLA_DE_PARADA_TEMPRANA = _e(
    "filas_de_cola_de_parada_temprana", "participantes", "reproduce.json#study.rows.early_stopping_tail",
    "Filas de la cola de parada temprana (del desarrollo; deciden cuándo parar, no entrenan)",
    "Early-stopping tail rows (from development; they decide when to stop, they do not train)",
    _SIN_REPARTO_ES, _SIN_REPARTO_EN)

#: La MISMA cola cuando el entregado es una red reentrenada con épocas fijas (reparación de la re-auditoría del 139, I2): ahí
#: no decide nada —no hay paciencia—, solo se aparta y no entrena. Mismo dato (`clave`, `ruta`), el rótulo que es verdad.
COLA_DE_UNA_RED_CON_EPOCAS_FIJAS = _e(
    "filas_de_cola_de_parada_temprana", "participantes", "reproduce.json#study.rows.early_stopping_tail",
    "Filas de la cola de parada temprana (del desarrollo; se apartan y no entrenan; con las épocas fijadas de la red no "
    "deciden nada)",
    "Early-stopping tail rows (from development; set aside, they do not train; with the network's fixed epochs they "
    "decide nothing)",
    _SIN_REPARTO_ES, _SIN_REPARTO_EN)

#: CÓMO SE REPARTIERON LAS FILAS DEL EXAMEN (140-C2, Ola 2): al azar por filas, por sujeto (`unit_column`) o hacia delante en el
#: tiempo (`time_column`, con su hueco y de dónde sale), y cuántas filas de desarrollo no tienen predicción fuera de pliegue.
#: Solo va en el inventario si el estudio lo declaró (un estudio anterior al 140 no lo registró).
REPARTO_POR_DISENO = _e(
    "reparto_por_diseno", "participantes", "reproduce.json#study.split",
    "Cómo se reparten las filas del examen (diseño, columnas, hueco y cobertura)",
    "How the exam rows are split (design, columns, gap and coverage)",
    "el estudio no registró cómo repartió las filas del examen",
    "the study did not record how it split the exam rows",
    "reparto_por_diseno")

#: LAS ÉPOCAS DE UNA RED ENTREGADA (reparación de la re-auditoría del 139, I1): cuántas se fijaron, de dónde salen y si el
#: plazo cortó el ajuste de selección (entonces el número lo puso ese reloj). Solo en el inventario si el estudio lo dijo.
ENTRENAMIENTO_DE_LA_RED = _e(
    "entrenamiento_de_la_red", "analisis", "reproduce.json#study.network_training",
    "Épocas de la red entregada", "Epochs of the delivered network",
    "el estudio no registró con cuántas épocas se entrenó la red entregada",
    "the study did not record how many epochs the delivered network was trained for",
    "entrenamiento_de_la_red")

#: QUÉ MOTOR ES EL MODELO ENTREGADO (139-C2; la deuda del 138: el expediente no lo decía). Va SIEMPRE en un estudio: un
#: estudio que no lo registró dice «no disponible», que es un hueco real.
MOTOR_ENTREGADO = _e(
    "motor_del_modelo_entregado", "analisis", "reproduce.json#study.delivered_engine",
    "Motor del modelo entregado", "Engine of the delivered model",
    "el estudio no registró qué motor es el modelo entregado",
    "the study did not record which engine is the delivered model")

#: CÓMO SE CALIBRÓ Y SE ELIGIÓ EL UMBRAL del modelo entregado (139-C2). Siempre en un estudio; uno guardado antes del
#: 139-C1 no lo registró y se dice así, SIN suponer el método de antes: lo que consta es lo que consta.
CALIBRACION_Y_UMBRAL = _e(
    "calibracion_y_umbral", "analisis", "reproduce.json#study.calibration_and_threshold",
    "Calibración y umbral del modelo entregado", "Calibration and threshold of the delivered model",
    "el estudio no registró cómo calibró el modelo ni cómo eligió su umbral",
    "the study did not record how it calibrated the model or chose its threshold",
    "calibracion_y_umbral")

#: El modo con que corrió el estudio (138-C1) y, si fue el rápido, quién no compitió por él.
MODO_DEL_ESTUDIO = _e(
    "modo_del_estudio", "analisis", "reproduce.json#study.mode",
    "Modo del estudio", "Study mode",
    "el estudio no registró con qué modo corrió",
    "the study did not record which mode it ran in", "modo")
FUERA_POR_EL_MODO = _e(
    "motores_fuera_por_el_modo", "analisis", "reproduce.json#study.engines_left_out_by_mode",
    "Motores que no compitieron por el modo (no se midieron; no es que puntuaran mal)",
    "Engines that did not compete because of the mode (not measured; they did not score badly)",
    "el estudio no registró qué motores dejó fuera el modo",
    "the study did not record which engines the mode left out", "lista")

#: Cómo se escribe cada modo. El vocabulario es el de `MODOS_DE_ESTUDIO`; uno que no esté
#: aquí se escribe tal cual, no se traduce a otro.
_NOMBRE_DEL_MODO: dict[str, dict[str, str]] = {
    "es": {"rapido": "rápido", "completo": "completo"},
    "en": {"rapido": "quick", "completo": "full"},
}


def _numero(valor: float) -> str:
    """Cuatro decimales como mucho, y sin ceros de relleno.

    `0.17852870321684206` en una ficha no es más preciso, es menos legible: los
    dígitos de más son ruido de coma flotante, no medida. Los cuatro decimales
    son los mismos que usa la ficha del perfil clínico (109-C2) — dos formatos
    distintos para el mismo número acabarían pareciendo dos números.
    """
    texto = f"{valor:.4f}"
    if "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto or "0"


def _cuenta(valor: Any) -> int | None:
    return valor if type(valor) is int and valor >= 0 else None


def _texto_de_la_calibracion(bloque: dict[str, Any], idioma: str) -> str:
    """139-C2 — cómo se calibró el modelo entregado y se eligió su umbral, escrito en el idioma pedido.

    Solo dice lo que el bloque trae: una cuenta ausente no se escribe (ni como cero), y el motivo de haber caído al pliegue
    0 es el que dio el estudio, en este idioma. Un método fuera del vocabulario se enseña TAL CUAL, sin traducirlo a otro."""
    es = idioma == "es"
    metodo = bloque.get("method")
    pliegues = _cuenta(bloque.get("folds"))
    fuera = _cuenta(bloque.get("out_of_fold_rows"))
    para_calibrar = _cuenta(bloque.get("rows_for_calibration"))
    para_umbral = _cuenta(bloque.get("rows_for_threshold"))
    partes: list[str] = []
    if metodo == "fuera_de_pliegue":
        partes.append("con predicciones fuera de pliegue (cada fila puntuada por un modelo que no la vio)" if es
                      else "with out-of-fold predictions (each row scored by a model that had not seen it)")
        if fuera is not None and pliegues is not None:
            partes.append(f"{fuera} filas de desarrollo en {pliegues} pliegues" if es
                          else f"{fuera} development rows across {pliegues} folds")
        elif fuera is not None:
            partes.append(f"{fuera} filas de desarrollo" if es else f"{fuera} development rows")
    elif metodo == "pliegue_0":
        partes.append("método de antes: con la parte de validación del pliegue 0" if es
                      else "the previous method: with the validation part of fold 0")
    else:
        partes.append(str(metodo))
    if para_calibrar is not None:
        partes.append(f"{para_calibrar} para ajustar la recalibración" if es
                      else f"{para_calibrar} to fit the recalibration")
    if para_umbral is not None:
        partes.append(f"{para_umbral} para elegir el umbral y las bandas" if es
                      else f"{para_umbral} to choose the threshold and the bands")
    texto = "; ".join(partes)
    motivo = bloque.get("reason")
    if isinstance(motivo, dict):
        dicho = motivo.get(idioma) or motivo.get("es" if idioma != "es" else "en")
        if dicho:
            texto += f". Motivo: {dicho}" if es else f". Reason: {dicho}"
    # I2 de la auditoría del C2/C3 (D4): en un reparto por el tiempo, el supuesto, la deriva MEDIDA (y lo que no mide) y la nota
    # de la predicción conforme. Solo lo que el bloque trae.
    for clave in ("temporal_assumption", "conformal_without_exchangeability"):
        nota = bloque.get(clave)
        if isinstance(nota, dict) and (nota.get(idioma) or "").strip():
            texto += f". {nota[idioma].rstrip('.')}"
    deriva = bloque.get("drift_between_halves")
    if isinstance(deriva, dict):
        dicho = _texto_de_la_deriva(deriva, idioma)
        if dicho:
            texto += f". {dicho}"
    return texto


def _texto_de_la_deriva(deriva: dict[str, Any], idioma: str) -> str:
    """La deriva medida entre las dos mitades de las validaciones de un reparto `temporal`: QUÉ mide (dentro del desarrollo,
    entre la mitad que calibra y la que elige el umbral) y qué NO mide (del desarrollo al futuro). Vacía si faltan las cifras."""
    es = idioma == "es"
    antes, despues, z, umbral = (deriva.get(k) for k in ("in_the_calibration_half", "in_the_threshold_half", "z", "z_threshold"))
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (antes, despues, z, umbral)):
        return ""
    pct = deriva.get("measure") == "prevalence"

    def cifra(v: float) -> str:
        return f"{100 * v:.1f} %" if pct else _numero(v)
    que = (("la proporción de la clase positiva" if pct else "la media del objetivo") if es
           else ("the share of the positive class" if pct else "the mean of the target"))
    hay = deriva.get("drift") is True
    if es:
        veredicto = (f"pasa del umbral ({_numero(umbral)}): el umbral y la prevalencia se eligieron con el pasado y pueden "
                     "haber derivado" if hay else f"por debajo del umbral ({_numero(umbral)}): sin deriva apreciable entre las "
                     "dos mitades")
        return (f"Deriva medida DENTRO del desarrollo: {que} en la mitad anterior de cada validación (la que calibra) es "
                f"{cifra(antes)} y en la posterior (la que elige el umbral) {cifra(despues)}; z = {_numero(abs(z))}, {veredicto}. "
                "No mide la deriva del desarrollo al futuro (a la prueba reservada ni a lo que venga después)")
    veredicto = (f"above the threshold ({_numero(umbral)}): the threshold and the prevalence were chosen with the past and "
                 "may have drifted" if hay else f"below the threshold ({_numero(umbral)}): no noticeable drift between the "
                 "two halves")
    return (f"Drift measured WITHIN development: {que} in the earlier half of each validation (the one that calibrates) is "
            f"{cifra(antes)} and in the later half (the one that picks the threshold) {cifra(despues)}; z = {_numero(abs(z))}, "
            f"{veredicto}. It does not measure drift from development to the future (to the held-out test or beyond)")


def _texto_del_reparto(bloque: dict[str, Any], idioma: str) -> str:
    """140-C2 — cómo se repartieron las filas del examen, escrito en el idioma pedido. Solo lo que el bloque trae: un hueco o
    una cobertura ausentes no se escriben; el diseño fuera del vocabulario se enseña TAL CUAL, sin traducirlo a otro."""
    es = idioma == "es"
    diseno = bloque.get("design")
    columna_de_sujeto, columna_de_tiempo = bloque.get("unit_column"), bloque.get("time_column")
    if diseno == "iid":
        texto = "al azar por filas (iid)" if es else "at random by rows (iid)"
    elif diseno == "groups":
        texto = ((f"por sujeto (groups): el examen se hace con sujetos que el modelo no vio, columna «{columna_de_sujeto}»"
                  if columna_de_sujeto else "por sujeto (groups)") if es else
                 (f"by subject (groups): the exam uses subjects the model did not see, column “{columna_de_sujeto}”"
                  if columna_de_sujeto else "by subject (groups)"))
    elif diseno == "temporal":
        texto = ((f"hacia delante en el tiempo (temporal): se entrena con el pasado y se mide en el futuro, columna "
                  f"«{columna_de_tiempo}»" if columna_de_tiempo else "hacia delante en el tiempo (temporal)") if es else
                 (f"forward in time (temporal): trained on the past and measured on the future, column "
                  f"“{columna_de_tiempo}”" if columna_de_tiempo else "forward in time (temporal)"))
    else:
        texto = str(diseno)
    gap = bloque.get("gap")
    if isinstance(gap, dict) and gap.get("magnitude") is not None:
        # M5 de la auditoría del C2/C3: «1 día», no «1 días» (y «1 day», no «1 days»).
        uno = gap.get("magnitude") == 1
        unidad = ({"es": {"seconds": "segundos", "minutes": "minutos", "hours": "horas", "days": "días", "weeks": "semanas",
                          "months": "meses", "years": "años"},
                   "en": {}} if not uno else
                  {"es": {"seconds": "segundo", "minutes": "minuto", "hours": "hora", "days": "día", "weeks": "semana",
                          "months": "mes", "years": "año"},
                   "en": {"seconds": "second", "minutes": "minute", "hours": "hour", "days": "day", "weeks": "week",
                          "months": "month", "years": "year"}})["es" if es else "en"].get(gap.get("unit"), gap.get("unit"))
        cuanto = f"{gap['magnitude']:g}" if isinstance(gap["magnitude"], (int, float)) else str(gap["magnitude"])
        origen = {"declared_horizon": ("el horizonte declarado del problema", "the problem's declared horizon"),
                  "requested": ("pedido por la persona", "requested by the person")}.get(bloque.get("gap_origin"))
        texto += (f"; hueco de {cuanto} {unidad} entre lo que entrena y lo que se mide" if es
                  else f"; gap of {cuanto} {unidad} between what trains and what is measured")
        if origen is not None:
            texto += f" ({origen[0] if es else origen[1]})"
    cobertura = bloque.get("out_of_fold_coverage")
    if isinstance(cobertura, dict):
        total, sin = _cuenta(cobertura.get("development_rows")), _cuenta(cobertura.get("rows_without_validation"))
        if total is not None and sin is not None:
            texto += (f"; de {total} filas de desarrollo, {sin} no tienen predicción fuera de pliegue" if es
                      else f"; of {total} development rows, {sin} have no out-of-fold prediction")
    limites = bloque.get("limits")
    if isinstance(limites, list) and limites:
        dichos = [(l.get("reason") or {}).get(idioma) for l in limites if isinstance(l, dict)]
        dichos = [d for d in dichos if d]
        if dichos:
            texto += ("; límites declarados: " if es else "; declared limits: ") + " · ".join(dichos)
    objetivo = bloque.get("target_in_development_and_test")
    if isinstance(objetivo, dict):
        dicho = _texto_del_objetivo_en_la_prueba(objetivo, idioma)
        if dicho:
            texto += f"; {dicho}"
    return texto


def _texto_del_objetivo_en_la_prueba(o: dict[str, Any], idioma: str) -> str:
    """El objetivo en el desarrollo y en la prueba reservada (decisión de Roberto del 28-10): solo informa."""
    es = idioma == "es"
    dev, test = o.get("development") or {}, o.get("test") or {}
    mucho = o.get("changes_a_lot") is True
    if o.get("task") == "classification":
        mayor = o.get("largest_change") or {}
        clase = mayor.get("class")
        a, b = (dev.get("shares") or {}).get(clase), (test.get("shares") or {}).get(clase)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return ""
        base = (f"la clase «{clase}» es el {100 * a:.1f} % del desarrollo y el {100 * b:.1f} % de la prueba reservada" if es
                else f"class “{clase}” is {100 * a:.1f}% of development and {100 * b:.1f}% of the held-out test")
    else:
        a, b, sd = dev.get("mean"), test.get("mean"), dev.get("standard_deviation")
        if not all(isinstance(v, (int, float)) for v in (a, b, sd)):
            return ""
        base = (f"la media del objetivo es {_numero(a)} en el desarrollo y {_numero(b)} en la prueba reservada (desviación "
                f"típica del desarrollo: {_numero(sd)})" if es else
                f"the target mean is {_numero(a)} in development and {_numero(b)} in the held-out test (development standard "
                f"deviation: {_numero(sd)})")
    aviso = ((": CAMBIA MUCHO (solo se informa; no cambia ninguna decisión del estudio)" if mucho
              else " (no cambia mucho)") if es else
             (": IT CHANGES A LOT (reported only; it changes no decision of the study)" if mucho
              else " (it does not change much)"))
    return base + aviso


def _segundos(valor: Any, idioma: str) -> str | None:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    texto = f"{float(valor):g}"
    return texto.replace(".", ",") if idioma == "es" else texto


def _texto_del_entrenamiento_de_la_red(bloque: dict[str, Any], idioma: str) -> str:
    """Reparación de la re-auditoría del 139 (I1) — las épocas de la red entregada, escritas en el idioma pedido. Solo lo que
    el bloque trae: una cifra ausente no se escribe, y «la selección la cortó su plazo» solo con un `true` del estudio."""
    es = idioma == "es"
    fijadas = _cuenta(bloque.get("fixed_epochs"))
    mejor = _cuenta(bloque.get("best_epoch_of_selection"))
    completas = _cuenta(bloque.get("completed_epochs_of_selection"))
    final = _cuenta(bloque.get("epochs_run_in_final"))
    partes: list[str] = []
    if fijadas is not None:
        partes.append(f"{fijadas} épocas fijadas, sin parada temprana" if es
                      else f"{fijadas} fixed epochs, without early stopping")
    if bloque.get("delivers") == "last_epoch":
        partes.append("se entrega la última" if es else "the last one is delivered")
    if mejor is not None and completas is not None:
        partes.append(f"salen del ajuste de selección: su mejor época ({mejor}), acotada a sus épocas completas ({completas})"
                      if es else f"they come from the selection fit: its best epoch ({mejor}), capped at its completed "
                      f"epochs ({completas})")
    elif mejor is not None:
        partes.append(f"salen de la mejor época del ajuste de selección ({mejor})" if es
                      else f"they come from the selection fit's best epoch ({mejor})")
    if bloque.get("selection_stopped_by_time_limit") is True:
        plazo = _segundos(bloque.get("selection_time_limit_seconds"), idioma)
        de = (f" (unos {plazo} s)" if es else f" (about {plazo} s)") if plazo is not None else ""
        partes.append(f"ese ajuste de selección lo cortó su tope de tiempo{de}: el número de épocas lo fijó ese reloj" if es
                      else f"that selection fit was cut short by its time limit{de}: the number of epochs was set by that clock")
    elif bloque.get("selection_stopped_by_time_limit") is False:
        partes.append("ese ajuste de selección no lo cortó su tope de tiempo" if es
                      else "that selection fit was not cut short by its time limit")
    if final is not None:
        partes.append(f"el modelo entregado hizo {final}" if es else f"the delivered model ran {final}")
    if bloque.get("early_stopping_tail_decides") is False:
        partes.append("la cola de parada temprana no decide nada" if es else "the early-stopping tail decides nothing")
    return "; ".join(partes)


def _si_no(valor: Any, locale: str) -> str:
    if locale == "es":
        return "sí" if valor else "no"
    return "yes" if valor else "no"


def texto_del_valor(evidencia: Evidencia, campo: Campo, locale: str) -> str:
    """El valor, escrito para que lo lea una persona y sin inventar nada.

    Un volcado crudo de una lista de diccionarios no es un dato legible, y un
    resumen que se invente un número es peor. Cada forma de resumir cuenta lo
    que hay: cuántos umbrales, en cuántos supera el modelo a las dos
    referencias, qué subgrupos y cuáles son exploratorios.
    """
    valor = campo.valor
    if evidencia.resumen == "alcance" and isinstance(valor, dict):
        # EL ALCANCE, CON LOS DOS CAMPOS DE LOS QUE SALE, y no la frase de la
        # ficha del perfil: el token y sus dos entradas se pueden re-derivar y
        # comprobar; una frase redactada, no. (Hasta el 2026-09-16 este
        # comentario decía además que aquella frase era FALSA para diseños
        # temporales con evidencia débil. Era cierto entonces y se reparó ese
        # día; ver el docstring del módulo.)
        return (f"{valor['alcance']} · evidencia={valor['evidencia']} · "
                f"diseno={valor['diseno']}")
    if evidencia.resumen == "comparacion" and isinstance(valor, dict):
        # Las fichas de la comparación, no una frase: la métrica, la diferencia
        # con su intervalo, el veredicto y contra quién. Un veredicto
        # `incomparable` no tiene diferencia ni intervalo, y no se le inventan.
        ic = valor.get("intervalo") or {}
        punto = valor.get("diferencia_puntual")
        tramo = (f" [{_numero(ic['ci_low'])}, {_numero(ic['ci_high'])}]"
                 if ic.get("ci_low") is not None and ic.get("ci_high") is not None
                 else "")
        return (f"{valor.get('metric_id')} "
                f"{_numero(punto) if punto is not None else '—'}{tramo} · "
                f"{valor.get('veredicto')} · baseline={valor.get('baseline')}")
    if isinstance(valor, float):
        return _numero(valor)
    if evidencia.resumen == "calibracion_y_umbral" and isinstance(valor, dict):
        return _texto_de_la_calibracion(valor, "es" if locale == "es" else "en")
    if evidencia.resumen == "reparto_por_diseno" and isinstance(valor, dict):
        return _texto_del_reparto(valor, "es" if locale == "es" else "en")
    if evidencia.resumen == "entrenamiento_de_la_red" and isinstance(valor, dict):
        return _texto_del_entrenamiento_de_la_red(valor, "es" if locale == "es" else "en")
    if evidencia.resumen == "modo" and isinstance(valor, str):
        return _NOMBRE_DEL_MODO["es" if locale == "es" else "en"].get(valor, valor)
    if evidencia.resumen == "booleano":
        return _si_no(valor, locale)
    if evidencia.resumen == "lista" and isinstance(valor, (list, tuple)):
        return ", ".join(str(v) for v in valor)
    if evidencia.resumen == "lista" and isinstance(valor, dict):
        return ", ".join(str(k) for k in valor)
    if evidencia.resumen == "umbrales" and isinstance(valor, (list, tuple)):
        return ", ".join(f"{f.get('umbral')}" for f in valor
                         if isinstance(f, dict))
    if evidencia.resumen == "puntos_dca" and isinstance(valor, (list, tuple)):
        supera = sum(1 for p in valor
                     if isinstance(p, dict) and p.get("supera_las_referencias"))
        if locale == "es":
            return (f"{len(valor)} umbrales; supera a las dos referencias en "
                    f"{supera}")
        return f"{len(valor)} thresholds; beats both references at {supera}"
    if evidencia.resumen == "segmentos" and isinstance(valor, (list, tuple)):
        piezas = []
        for s in valor:
            if not isinstance(s, dict):
                continue
            marca = ("predefinido" if s.get("predefinido") else "exploratorio") \
                if locale == "es" else \
                ("predefined" if s.get("predefinido") else "exploratory")
            piezas.append(f"{s.get('segmento_id')} ({marca})")
        return ", ".join(piezas)
    if evidencia.resumen == "metricas" and isinstance(valor, (list, tuple)):
        piezas = []
        for m in valor:
            if not isinstance(m, dict):
                continue
            sobre = str(m.get("dataset_sha256") or "")
            cola = f" [{sobre[:12]}…]" if sobre else ""
            piezas.append(f"{m.get('name')}={m.get('value')}{cola}")
        return "; ".join(piezas)
    if isinstance(valor, dict):
        return json.dumps(valor, sort_keys=True, ensure_ascii=False)
    if isinstance(valor, (list, tuple)):
        return ", ".join(str(v) for v in valor)
    return str(valor)


# ---------------------------------------------------------------------------
# Cómo se escribe una línea — una sola vez, para los dos informes
# ---------------------------------------------------------------------------

#: Los rótulos de los tres estados. **Ninguno califica nada**: dicen de dónde
#: viene el dato, no si basta.
ROTULOS_DE_ESTADO: dict[str, dict[str, str]] = {
    "es": {MEDIDO: "MEDIDO por el core",
           DECLARADO: "DECLARADO por una persona",
           FALTA: "falta"},
    "en": {MEDIDO: "MEASURED by the core",
           DECLARADO: "DECLARED by a person",
           FALTA: "missing"},
}

ROTULOS_DE_DOMINIO: dict[str, dict[str, str]] = {
    "es": {"participantes": "Participantes", "predictores": "Predictores",
           "desenlace": "Desenlace", "analisis": "Análisis",
           "aplicabilidad": "Aplicabilidad"},
    "en": {"participantes": "Participants", "predictores": "Predictors",
           "desenlace": "Outcome", "analisis": "Analysis",
           "aplicabilidad": "Applicability"},
}


def rotulo_de_estado(estado: str, locale: str) -> str:
    return ROTULOS_DE_ESTADO["es" if locale == "es" else "en"][estado]


def rotulo_de_dominio(dominio: str, locale: str) -> str:
    return ROTULOS_DE_DOMINIO["es" if locale == "es" else "en"][dominio]


def linea_de_campo(evidencia: Evidencia, campo: Campo, locale: str) -> str:
    """La línea de Markdown de un campo, con su estado y SU RUTA.

    Es la única forma que tienen los dos informes de escribir una línea, y por
    eso la ruta no es opcional: el criterio de terminado del 109-C3 pide que
    cada frase trace a un campo, y aquí no hay forma de escribir una que no lo
    haga. La ruta va aunque el campo falte —sobre todo si falta—: saber QUÉ
    campo lo arreglaría es la mitad útil del hueco.
    """
    idioma = "es" if locale == "es" else "en"
    rotulo = evidencia.rotulo[idioma]
    if campo.estado == FALTA:
        cuerpo = f"_{rotulo_de_estado(FALTA, idioma)}_ — {campo.motivo}"
    else:
        cuerpo = (f"{texto_del_valor(evidencia, campo, idioma)} · "
                  f"{rotulo_de_estado(campo.estado, idioma)}")
    return f"- **{rotulo}**: {cuerpo} · `{campo.ruta}`"
