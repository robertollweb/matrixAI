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
los datos son sintéticos).

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
           "sin_campo": "no declara"},
    "en": {"roto": "cannot be read", "sin_fichero": "the package does not ship",
           "sin_campo": "does not declare"},
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
                f"{origen}: el texto contiene «{veredicto}». El invariante 6 del "
                f"contrato 109 dice que el core rellena lo medido, marca lo "
                f"declarado y enumera lo que falta, y NUNCA califica el riesgo de "
                f"sesgo: eso lo emite quien revisa el estudio.")
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
        for evidencia in INVENTARIO:
            motivo = evidencia.motivo_si_falta[idioma]
            if evidencia.clave == "alcance_de_validacion":
                campo, aviso = self.alcance_de_validacion()
                if aviso:
                    avisos.append(aviso)
                if not campo.hay_dato:
                    campo = Campo(evidencia.ruta, FALTA, motivo=motivo)
            else:
                campo = self.campo(evidencia.ruta, motivo_si_falta=motivo,
                                   locale=idioma)
            pares.append((evidencia, campo))
        return tuple(pares), tuple(avisos)

    # -- lo derivado ---------------------------------------------------------

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
_SIN_PERFIL_ES = "el paquete no trae el perfil clínico (109-C2)"
_SIN_PERFIL_EN = "the package ships no clinical profile (109-C2)"

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
       "es anterior a 109-C3",
       "the clinical profile does not ship it: either the study could not pose "
       "it —and the profile says why in `sin_comparacion_con_el_baseline`— or "
       "the package predates 109-C3", "comparacion"),
    _e("proceso_actual", "analisis", "team_declaration.json#proceso_actual",
       "Comparación con el proceso actual", "Comparison against current practice",
       "invariante 4 del contrato 109: sin proceso actual declarado, se dice",
       "invariant 4 of contract 109: with no current practice declared, it is said"),
)

#: Índice por clave, para que nadie tenga que recorrer la tupla a mano.
POR_CLAVE: dict[str, Evidencia] = {e.clave: e for e in INVENTARIO}


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
