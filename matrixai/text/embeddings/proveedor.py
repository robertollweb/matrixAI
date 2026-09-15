# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — el proveedor: de un texto a un vector, con onnxruntime.

Lo que hace el proveedor es exactamente lo que el paquete descargado dice que
hay que hacer, ni más ni menos. Cómo se agrupa la salida y si se normaliza NO
se escriben aquí: se LEEN de `modules.json` (sentence-transformers) o de
`config.json` (model2vec), que es donde el autor del modelo lo declara. Si un
paquete no lo declara de una forma que este módulo entienda, el proveedor no
se construye — nunca se cae a un comportamiento «por defecto», porque el
defecto equivocado no da un error: da un vector plausible.

Dos formas de grafo, detectadas por sus entradas, no por el nombre del
modelo:

  · **bolsa** (`input_ids` plano + `offsets`): el propio grafo agrupa. Es lo
    que exporta `model2vec` — una tabla con `EmbeddingBag` dentro.
  · **secuencia** (`input_ids` + `attention_mask` [+ `token_type_ids`]):
    devuelve un vector por token y el pooling lo hace este módulo.

`numpy` y `onnxruntime` se importan DENTRO de las funciones: `matrixai-core`
se instala con `dependencies = []` (102, invariante 1) y el catálogo tiene
que poder leerse en una máquina que no los tenga.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from matrixai.text.embeddings.catalogo import Candidato
from matrixai.text.embeddings.descarga import DescargaError, raiz_cache
from matrixai.text.embeddings.wordpiece import WordPiece, desde_tokenizer_json

#: Topes de seguridad para cuando el paquete declara «no truncar»
#: (`model2vec` pone `seq_length: 1000000`). Son DOS y no uno, porque el coste
#: de un texto largo no es el mismo en las dos formas de grafo:
#:
#:  · **bolsa**: el grafo suma vectores de una tabla — coste LINEAL. Truncar
#:    aquí no protege de nada y sí cambia el vector: medido, con un tope de
#:    4096 el coseno contra `model2vec` se cae a 0,695 en un texto de 7.000
#:    palabras. Así que no se trunca, igual que hace su librería.
#:  · **secuencia**: la atención es CUADRÁTICA en la longitud. Ahí un texto
#:    patológico sí se lleva la máquina por delante, y el tope se aplica —
#:    pero en la práctica no llega a usarse nunca, porque un transformer de
#:    frase siempre declara su longitud.
TOPE_BOLSA = 1_000_000
TOPE_SECUENCIA = 4096


class ProveedorError(Exception):
    """El proveedor no se puede construir o no puede cumplir lo que promete."""


class IdiomaNoCubierto(Exception):
    """Se pidió un idioma que el proveedor no cubre. Existe para que quien
    llama tenga que decidir qué hacer: un idioma no cubierto es una
    limitación declarada (107, invariante 7), nunca un vector en silencio."""


@dataclass(frozen=True)
class ComposicionDelVector:
    """Cómo se pasa de los vectores por token a un vector por texto, según lo
    que declara el paquete. `fuente` dice de qué fichero salió cada cosa."""

    pooling: str
    normaliza: bool
    dimension_declarada: int | None
    fuente: str
    #: TODAS las longitudes máximas que declara el paquete, por fichero. En
    #: plural a propósito: `all-MiniLM-L6-v2` declara TRES y no coinciden —
    #: `tokenizer.json` dice 128, `sentence_bert_config.json` 256 y
    #: `tokenizer_config.json` 512. Medido el 2026-09-14.
    longitudes_declaradas: tuple[tuple[str, int], ...] = ()
    #: La que se usa, y por qué.
    longitud_usada: int | None = None
    motivo_de_la_longitud: str = ""


def _longitudes_declaradas(directorio: Path) -> list[tuple[str, int]]:
    """Recoge cuántas longitudes máximas declara el paquete y dónde.

    No se queda con la primera: un paquete que se contradice a sí mismo tiene
    que poder decirlo, porque elegir en silencio produce un vector que no es
    el que la ficha del modelo promete, y eso no se ve.
    """
    encontradas: list[tuple[str, int]] = []
    sbert = directorio / "sentence_bert_config.json"
    if sbert.is_file():
        v = json.loads(sbert.read_text(encoding="utf-8")).get("max_seq_length")
        if isinstance(v, int) and v > 0:
            encontradas.append(("sentence_bert_config.json:max_seq_length", v))
    tok = directorio / "tokenizer.json"
    if tok.is_file():
        trunc = (json.loads(tok.read_text(encoding="utf-8")) or {}).get("truncation") or {}
        v = trunc.get("max_length")
        if isinstance(v, int) and v > 0:
            encontradas.append(("tokenizer.json:truncation.max_length", v))
    cfg = directorio / "config.json"
    if cfg.is_file():
        v = json.loads(cfg.read_text(encoding="utf-8")).get("seq_length")
        if isinstance(v, int) and v > 0:
            encontradas.append(("config.json:seq_length", v))
    tcfg = directorio / "tokenizer_config.json"
    if tcfg.is_file():
        v = json.loads(tcfg.read_text(encoding="utf-8")).get("model_max_length")
        if isinstance(v, int) and 0 < v <= 1_000_000:
            encontradas.append(("tokenizer_config.json:model_max_length", v))
    return encontradas


def elegir_longitud(declaradas: list[tuple[str, int]], *, tope: int) -> tuple[int, str]:
    """Qué longitud se usa cuando el paquete declara varias.

    Manda `sentence_bert_config.json`, porque es la que usa de verdad
    `sentence-transformers` al cargar el modelo — y el vector que promete la
    ficha del proveedor es el que produce ESA implementación. Medido: con la
    de `tokenizer.json` (128) el coseno contra la referencia baja a 0,9990 en
    los textos largos; con 256 vuelve a 1,0.
    """
    if not declaradas:
        return tope, f"el paquete no declara ninguna; se usa el tope de seguridad {tope}"
    por_fichero = dict(declaradas)
    #: Orden de mando, y cada escalón tiene su motivo:
    #:  1. `sentence_bert_config.json` — la que usa `sentence-transformers` al
    #:     cargar el modelo, o sea la que produce el vector que promete la ficha.
    #:  2. `config.json:seq_length` — la de `model2vec`, que vale 1.000.000, es
    #:     decir «no truncar». Va ANTES que `tokenizer_config.json`, porque ese
    #:     trae el 512 heredado del BERT del que se destiló y truncar ahí daría
    #:     otro vector para los textos largos.
    #:  3. `tokenizer.json:truncation` y 4. `tokenizer_config.json`.
    for clave in (
        "sentence_bert_config.json:max_seq_length",
        "config.json:seq_length",
        "tokenizer.json:truncation.max_length",
        "tokenizer_config.json:model_max_length",
    ):
        if clave in por_fichero:
            valor = por_fichero[clave]
            n = len({v for _, v in declaradas})
            extra = (
                f" (el paquete declara {n} longitudes distintas: "
                + ", ".join(f"{k}={v}" for k, v in declaradas)
                + ")"
                if n > 1
                else ""
            )
            if valor > tope:
                return tope, (
                    f"{clave} dice {valor}, que es «no truncar»; se acota al tope de "
                    f"seguridad {tope} para que un texto patológico no se lleve la "
                    f"máquina por delante{extra}"
                )
            return valor, f"la declara {clave}{extra}"
    clave, valor = declaradas[0]
    return valor, f"la declara {clave}"


def leer_composicion(directorio: Path) -> ComposicionDelVector:
    """Lee del paquete cómo se compone el vector, o se niega a adivinarlo."""
    modules = directorio / "modules.json"
    config = directorio / "config.json"

    if modules.is_file():
        try:
            mods = json.loads(modules.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ProveedorError(f"{modules}: no es JSON válido ({e})") from None
        tipos = [str(m.get("type", "")).rsplit(".", 1)[-1] for m in mods]
        normaliza = "Normalize" in tipos
        if "Pooling" in tipos:
            idx = tipos.index("Pooling")
            ruta = directorio / (mods[idx].get("path") or "") / "config.json"
            if not ruta.is_file():
                raise ProveedorError(f"{modules} declara un Pooling en {ruta} y ese fichero no está")
            cfg = json.loads(ruta.read_text(encoding="utf-8"))
            activos = [k for k, v in cfg.items() if k.startswith("pooling_mode_") and v is True]
            if activos != ["pooling_mode_mean_tokens"]:
                raise ProveedorError(
                    f"{ruta}: modo de pooling {activos} — este módulo solo reproduce la media "
                    "de tokens; cualquier otro daría otro vector"
                )
            return ComposicionDelVector(
                pooling="media",
                normaliza=normaliza,
                dimension_declarada=cfg.get("word_embedding_dimension"),
                fuente=f"{modules.name} + {ruta.parent.name}/config.json",
                longitudes_declaradas=tuple(_longitudes_declaradas(directorio)),
            )
        if "StaticEmbedding" in tipos:
            cfg = json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
            # `model2vec` normaliza si su config lo dice; `modules.json` lo
            # repite. Si los dos hablan y se contradicen, se para: no hay un
            # criterio para elegir cuál miente.
            en_config = cfg.get("normalize")
            if en_config is not None and bool(en_config) != normaliza:
                raise ProveedorError(
                    f"{directorio}: config.json dice normalize={en_config!r} y modules.json "
                    f"{'sí' if normaliza else 'no'} trae Normalize — no se elige por cuenta propia"
                )
            return ComposicionDelVector(
                pooling="en_el_grafo",
                normaliza=normaliza,
                dimension_declarada=cfg.get("hidden_dim"),
                fuente=f"{modules.name} + config.json",
                longitudes_declaradas=tuple(_longitudes_declaradas(directorio)),
            )
        raise ProveedorError(f"{modules}: módulos {tipos} — ninguno es un pooling que este módulo reproduzca")

    raise ProveedorError(f"{directorio}: sin modules.json, no hay dónde leer cómo se compone el vector")


class ProveedorDeEmbeddings:
    """Un proveedor cargado y listo. `codificar` es lo único que hace."""

    def __init__(
        self,
        candidato: Candidato,
        directorio: Path,
        sesion: Any,
        tokenizador: WordPiece,
        composicion: ComposicionDelVector,
    ) -> None:
        self.tokenizador_de_fuera = False
        self.candidato = candidato
        self.directorio = directorio
        self._sesion = sesion
        self.tokenizador = tokenizador
        self.composicion = composicion
        self._entradas = [e.name for e in sesion.get_inputs()]
        self._salida = sesion.get_outputs()[0].name
        if "offsets" in self._entradas:
            self.forma = "bolsa"
        elif "attention_mask" in self._entradas:
            self.forma = "secuencia"
        else:
            raise ProveedorError(
                f"{candidato.id}: entradas {self._entradas} — ni bolsa (offsets) ni "
                "secuencia (attention_mask); este módulo no sabe alimentarlo"
            )
        if self.forma == "bolsa" and composicion.pooling != "en_el_grafo":
            raise ProveedorError(
                f"{candidato.id}: el grafo agrupa por su cuenta pero el paquete declara "
                f"pooling {composicion.pooling!r} — se agruparía dos veces"
            )
        forma_salida = sesion.get_outputs()[0].shape
        self.dimension = int(forma_salida[-1])
        if composicion.dimension_declarada is not None and composicion.dimension_declarada != self.dimension:
            raise ProveedorError(
                f"{candidato.id}: el paquete declara dimensión {composicion.dimension_declarada} "
                f"y el grafo devuelve {self.dimension}"
            )

    # ------------------------------------------------------------------ carga
    @classmethod
    def cargar(
        cls,
        candidato: Candidato,
        *,
        directorio: Path | None = None,
        hilos: int = 1,
        verificar_digests: list[Any] | None = None,
        tokenizador: Any | None = None,
    ) -> "ProveedorDeEmbeddings":
        """`tokenizador` inyecta uno de fuera cuando el núcleo no sabe hacerlo.

        Existe para poder MEDIR los proveedores cuyo tokenizador es Unigram de
        SentencePiece, que el núcleo no reproduce en stdlib. Lo que se mide con
        él queda marcado como medido con dependencia externa: un número que
        solo se puede obtener instalando algo más no es el mismo número.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ProveedorError(
                "onnxruntime no está instalado. El núcleo no lo lleva a propósito "
                "(102, invariante 1); es el proveedor de embeddings quien lo necesita."
            ) from None

        if candidato.tokenizador != "wordpiece" and tokenizador is None:
            raise ProveedorError(
                f"{candidato.id}: su tokenizador es {candidato.tokenizador!r}, y este paquete solo "
                "reproduce WordPiece en stdlib. Con la dependencia `tokenizers` sí se puede: es "
                "una decisión declarada, no un olvido."
            )

        d = Path(directorio) if directorio is not None else raiz_cache() / candidato.id
        if not d.is_dir():
            raise ProveedorError(
                f"{candidato.id}: no está descargado en {d}. Los pesos de terceros no van en la "
                "imagen (102-C5): descárgalo antes, aceptando su licencia."
            )
        if verificar_digests:
            from matrixai.text.embeddings.descarga import verificar_fichero

            for f in verificar_digests:
                try:
                    verificar_fichero(d / f.ruta, f)
                except DescargaError as e:
                    raise ProveedorError(f"{candidato.id}: {e}") from None

        composicion = leer_composicion(d)

        opciones = ort.SessionOptions()
        opciones.log_severity_level = 3
        opciones.intra_op_num_threads = hilos
        opciones.inter_op_num_threads = 1
        sesion = ort.InferenceSession(
            str(d / candidato.modelo), opciones, providers=["CPUExecutionProvider"]
        )
        # El tope depende de la FORMA del grafo, así que la sesión se abre
        # antes que el tokenizador: sin saber si el grafo agrupa por su cuenta
        # no se puede decidir si truncar protege de algo.
        es_bolsa = "offsets" in [e.name for e in sesion.get_inputs()]
        if candidato.longitud_efectiva_de_su_libreria is not None:
            # Manda sobre cualquier fichero: es la que produce el vector que la
            # librería del proveedor da de verdad, y el paquete no la dice.
            usada = candidato.longitud_efectiva_de_su_libreria
            motivo = (
                f"no la declara ningún fichero del paquete: es la que aplica su librería "
                f"({candidato.fuente_de_la_longitud_efectiva})"
            )
        else:
            usada, motivo = elegir_longitud(
                list(composicion.longitudes_declaradas), tope=TOPE_BOLSA if es_bolsa else TOPE_SECUENCIA
            )
        composicion = dataclasses.replace(composicion, longitud_usada=usada, motivo_de_la_longitud=motivo)
        if tokenizador is not None:
            tok = tokenizador
        else:
            tok = desde_tokenizer_json(
                d / "tokenizer.json", max_longitud=usada, forzar_longitud=True
            )
        proveedor = cls(candidato, d, sesion, tok, composicion)
        proveedor.tokenizador_de_fuera = tokenizador is not None
        return proveedor

    # --------------------------------------------------------------- codifica
    def codificar(self, textos: Sequence[str], *, lote: int = 32) -> Any:
        """Devuelve una matriz `[len(textos), dimension]` de float32."""
        import numpy as np

        if isinstance(textos, str):
            raise ProveedorError("codificar recibe una secuencia de textos, no un texto suelto")
        salida = np.empty((len(textos), self.dimension), dtype=np.float32)
        for inicio in range(0, len(textos), lote):
            trozo = list(textos[inicio : inicio + lote])
            salida[inicio : inicio + len(trozo)] = self._codificar_lote(trozo)
        return salida

    def _tokenizar(self, textos: Sequence[str]) -> list[list[int]]:
        con = self.candidato.tokens_especiales
        return [self.tokenizador.codificar(t, con_especiales=con).ids for t in textos]

    def _codificar_lote(self, textos: Sequence[str]) -> Any:
        import numpy as np

        ids = self._tokenizar(textos)
        if self.forma == "bolsa":
            # Un texto sin ningún token (cadena vacía, o solo caracteres que
            # el normalizador tira) no tiene vector. NO se rellena con ceros:
            # un valor ausente no es un cero, y un vector de ceros pasaría por
            # un texto legítimo aguas abajo.
            vacios = [i for i, x in enumerate(ids) if not x]
            if vacios:
                raise ProveedorError(
                    f"los textos en las posiciones {vacios[:5]} no dejan ningún token tras "
                    "normalizar: este proveedor no tiene vector para ellos. Fíltralos o "
                    "decláralos ausentes antes de pedirlo."
                )
            plano = np.array([i for x in ids for i in x], dtype=np.int64)
            offsets = np.zeros(len(ids), dtype=np.int64)
            acumulado = 0
            for k, x in enumerate(ids):
                offsets[k] = acumulado
                acumulado += len(x)
            bruto = self._sesion.run([self._salida], {"input_ids": plano, "offsets": offsets})[0]
            vectores = np.asarray(bruto, dtype=np.float32)
        else:
            largo = max(len(x) for x in ids)
            n = len(ids)
            entrada = np.full((n, largo), self.tokenizador.pad_id, dtype=np.int64)
            mascara = np.zeros((n, largo), dtype=np.int64)
            for k, x in enumerate(ids):
                entrada[k, : len(x)] = x
                mascara[k, : len(x)] = 1
            alimento: dict[str, Any] = {"input_ids": entrada, "attention_mask": mascara}
            if "token_type_ids" in self._entradas:
                alimento["token_type_ids"] = np.zeros((n, largo), dtype=np.int64)
            bruto = self._sesion.run([self._salida], alimento)[0]
            estados = np.asarray(bruto, dtype=np.float32)
            peso = mascara.astype(np.float32)[:, :, None]
            vectores = (estados * peso).sum(axis=1) / np.clip(peso.sum(axis=1), 1e-9, None)

        if self.composicion.normaliza:
            normas = np.linalg.norm(vectores, axis=1, keepdims=True)
            vectores = vectores / np.clip(normas, 1e-12, None)
        return vectores.astype(np.float32)

    # --------------------------------------------------------------- describe
    def describe(self) -> dict[str, Any]:
        return {
            "id": self.candidato.id,
            "familia": self.candidato.familia,
            "repo": self.candidato.repo,
            "revision": self.candidato.revision,
            "modelo": self.candidato.modelo,
            "forma_del_grafo": self.forma,
            "entradas": list(self._entradas),
            "dimension": self.dimension,
            "pooling": self.composicion.pooling,
            "normaliza": self.composicion.normaliza,
            "leido_de": self.composicion.fuente,
            "tokenizador": self.candidato.tokenizador,
            "tokenizador_de_fuera_del_nucleo": self.tokenizador_de_fuera,
            "tokens_especiales": self.candidato.tokens_especiales,
            "max_tokens": self.tokenizador.max_longitud,
            "max_tokens_por_que": self.composicion.motivo_de_la_longitud,
            "max_tokens_declarados_por_el_paquete": [
                {"donde": k, "valor": v} for k, v in self.composicion.longitudes_declaradas
            ],
            "opaco": True,
            "opacidad": "pesos de terceros no auditados; lo que el vector representa no se "
            "puede explicar desde aquí, solo declarar de dónde viene",
        }
