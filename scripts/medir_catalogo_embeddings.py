#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""107-C1 — genera el catálogo MEDIDO de proveedores de embeddings.

La tabla se MIDE, no se escribe. Este guión es el único sitio donde nace
`matrixai/text/embeddings/catalogo_medido.json`, y `test_c107_c1_catalogo.py`
usa ESTAS MISMAS funciones para comprobarlo — para que no haya dos formas de
decir lo mismo, que es como las tablas empiezan a mentir.

    python3 scripts/medir_catalogo_embeddings.py --fijar
        Lee del origen, para cada candidato y en SU revisión: tamaño y digest
        de cada fichero, y la licencia, que se traduce a SPDX por tabla
        explícita. No descarga pesos.

    python3 scripts/medir_catalogo_embeddings.py --descargar [ID ...]
        Trae los pesos a la caché local y los verifica contra los digests.
        Exige haber aceptado la licencia (--acepto-la-licencia-de ID).

    python3 scripts/medir_catalogo_embeddings.py --medir [ID ...]
        Ejecuta de verdad: dimensión, tiempo por 1.000 textos con su control
        de linealidad, y cobertura de es y en sobre un corpus paralelo real.

    python3 scripts/medir_catalogo_embeddings.py --comprobar
        ¿Sigue cuadrando el artefacto con los ficheros de la caché?

El corpus de medida es FLORES-200 (dev/devtest), 997 + 1.012 frases
alineadas frase a frase en español y en inglés. Se usa SOLO para medir; no se
redistribuye ni se deriva nada de él, y vive en la caché del usuario, nunca
en el repositorio. Licencia CC-BY-SA-4.0 (fuente en CORPUS_URL).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from matrixai.text.embeddings import medicion  # noqa: E402
from matrixai.text.embeddings.catalogo import CANDIDATOS, por_id  # noqa: E402
from matrixai.text.embeddings.descarga import (  # noqa: E402
    DescargaError,
    FicheroDelProveedor,
    descargar_fichero,
    raiz_cache,
    sha256_de,
    verificar_fichero,
)

ARTEFACTO = RAIZ / "matrixai" / "text" / "embeddings" / "catalogo_medido.json"

CORPUS_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CORPUS_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
CORPUS_BYTES = 25585843
CORPUS_LICENCIA = "CC-BY-SA-4.0"

#: Cuántos textos por idioma. El corte pide 1.000.
N_TEXTOS = 1000


# --------------------------------------------------------------------- corpus
def asegurar_corpus() -> Path:
    """Descarga y verifica FLORES-200 en la caché. Sin él no se mide nada.

    No se reutiliza el descargador del proveedor tal cual porque su allowlist
    es el dominio de Hugging Face: este corpus vive en otro sitio. Lo que sí
    se reutiliza es lo que importa —comprobar el sha256 antes de usar el
    fichero— porque un corpus cambiado cambiaría todos los números de la
    tabla sin que nadie lo notase.
    """
    import urllib.request

    destino = raiz_cache() / "_corpus"
    destino.mkdir(parents=True, exist_ok=True)
    tar = destino / "flores200_dataset.tar.gz"
    if not tar.is_file() or sha256_de(tar) != CORPUS_SHA256:
        parcial = tar.with_suffix(".parcial")
        with urllib.request.urlopen(CORPUS_URL, timeout=300) as r, open(parcial, "wb") as f:
            while True:
                trozo = r.read(1 << 20)
                if not trozo:
                    break
                f.write(trozo)
        obtenido = sha256_de(parcial)
        if obtenido != CORPUS_SHA256:
            parcial.unlink(missing_ok=True)
            raise SystemExit(f"el corpus descargado no cuadra: {obtenido} != {CORPUS_SHA256}")
        parcial.replace(tar)
    extraido = destino / "flores200_dataset"
    if not extraido.is_dir():
        with tarfile.open(tar) as t:
            miembros = [
                m
                for m in t.getmembers()
                if m.isfile()
                and ("spa_Latn" in m.name or "eng_Latn" in m.name or m.name.endswith("metadata_devtest.tsv"))
            ]
            t.extractall(destino, members=miembros)
    return extraido


def cargar_corpus() -> dict[str, object]:
    base = asegurar_corpus()
    es = (base / "devtest" / "spa_Latn.devtest").read_text(encoding="utf-8").splitlines()
    en = (base / "devtest" / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
    meta = (base / "metadata_devtest.tsv").read_text(encoding="utf-8").splitlines()[1:]
    articulos = [linea.split("\t")[0] for linea in meta]
    n = min(N_TEXTOS, len(es), len(en), len(articulos))
    return {"es": es[:n], "en": en[:n], "articulos": articulos[:n], "n": n}


# ----------------------------------------------------------------- artefacto
def leer_artefacto() -> dict:
    if ARTEFACTO.is_file():
        return json.loads(ARTEFACTO.read_text(encoding="utf-8"))
    return {"generado_por": "scripts/medir_catalogo_embeddings.py", "proveedores": {}}


def escribir_artefacto(datos: dict) -> None:
    datos["escrito_el"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ARTEFACTO.write_text(json.dumps(datos, indent=1, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    print(f"escrito {ARTEFACTO.relative_to(RAIZ)}")


def entorno_de_medida() -> dict:
    """Con qué se midió. Un tiempo sin su máquina al lado no es un dato."""
    try:
        modelo = ""
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                modelo = linea.split(":", 1)[1].strip()
                break
    except OSError:
        modelo = ""
    try:
        carga = Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        carga = ""
    import onnxruntime as ort

    return {
        "maquina": platform.platform(),
        "cpu": modelo,
        "nucleos": __import__("os").cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": __import__("numpy").__version__,
        "carga_al_empezar": carga,
        "medido_el": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ------------------------------------------------------------------ acciones
def accion_fijar(ids: list[str]) -> None:
    datos = leer_artefacto()
    for cid in ids:
        c = por_id(cid)
        try:
            bloque = medicion.fijar_candidato(c)
        except medicion.MedicionError as e:
            print(f"FALLA {cid}: {e}")
            continue
        anterior = datos["proveedores"].get(cid, {})
        anterior.update(bloque)
        anterior["candidato"] = {
            "tokenizador": c.tokenizador,
            "tokens_especiales": c.tokens_especiales,
            "idiomas_segun_su_autor": c.idiomas_segun_su_autor,
            "fuente_de_esa_cita": c.fuente_de_esa_cita,
            "motivo": c.motivo,
            "no_descargado_porque": c.no_descargado_porque,
            "notas": list(c.notas),
        }
        datos["proveedores"][cid] = anterior
        lic = bloque["licencia"]
        print(
            f"fijado {cid:48s} {lic['spdx']:12s} pesos={bloque['pesos_bytes']/1e6:8.2f} MB "
            f"total={bloque['descarga_total_bytes']/1e6:8.2f} MB compatible={lic['compatible']}"
        )
    escribir_artefacto(datos)


def accion_descargar(ids: list[str], aceptadas: set[str]) -> None:
    datos = leer_artefacto()
    for cid in ids:
        c = por_id(cid)
        bloque = datos["proveedores"].get(cid)
        if not bloque or "ficheros" not in bloque:
            print(f"FALLA {cid}: fíjalo primero (--fijar)")
            continue
        if not bloque["licencia"]["compatible"]:
            print(f"SALTA {cid}: licencia {bloque['licencia']['spdx']} — {bloque['licencia']['motivo_si_no']}")
            continue
        if cid not in aceptadas:
            print(
                f"SALTA {cid}: los pesos son de terceros bajo {bloque['licencia']['spdx']}. "
                f"Descarga con --acepto-la-licencia-de {cid} si aceptas sus términos "
                f"({bloque['licencia']['fuente']})."
            )
            continue
        destino = raiz_cache() / cid
        print(f"descargando {cid} ({bloque['descarga_total_bytes']/1e6:.2f} MB) a {destino}")
        for ruta, estado in medicion.descargar_candidato(c, bloque, destino).items():
            print(f"   {ruta}: {estado}")
        bloque["descargado"] = True
        bloque["aceptacion_de_licencia"] = {
            "spdx": bloque["licencia"]["spdx"],
            "aceptada_el": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "por": "quien ejecutó este guión con --acepto-la-licencia-de",
        }
    escribir_artefacto(datos)


def tokens_especiales_de(ruta: Path) -> tuple[str, str | None]:
    """El token DESCONOCIDO y el de RELLENO, leídos del `tokenizer.json` que de
    verdad ejecuta `tokenizers`, no de `tokenizer_config.json` ni escritos aquí.

    Estaban escritos a mano (`<unk>`, `<pad>`), que es lo que usa el MiniLM
    multilingüe. MEDIDO el 2026-09-17 con `potion-multilingual-128M`: su
    `tokenizer_config.json` dice `<pad>`/`<unk>` —heredado de XLM-R— y NINGUNO
    de los dos existe en su vocabulario; los reales son `[PAD]` (id 0) y
    `[UNK]` (id 1). Con `<unk>` a mano, la cobertura cuenta los desconocidos
    comparando con un token que nunca sale: la cobertura saldría INFLADA y sin
    un error. Con el MiniLM, leer de aquí da `<unk>` y `<pad>`, los mismos de
    siempre: su medida no se mueve.

    El desconocido es `vocab[unk_id]` del modelo, que es el que el tokenizador
    emite. El relleno es el que declara el bloque `padding`; si no declara
    ninguno, `None`: un modelo estático (forma «bolsa», `input_ids` + `offsets`)
    no rellena, y exigírselo impedía medirlo.
    """
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    modelo = datos.get("model") or {}
    unk_id = modelo.get("unk_id")
    vocab = modelo.get("vocab")
    if unk_id is None or not isinstance(vocab, list) or not 0 <= unk_id < len(vocab):
        raise SystemExit(f"{ruta}: no declara un token desconocido (`model.unk_id`) que exista en su vocabulario")
    desconocido = vocab[unk_id][0]
    relleno = (datos.get("padding") or {}).get("pad_token")
    return desconocido, relleno


class _TokenizadorDeFuera:
    """Envuelve `tokenizers` (Rust) para poder MEDIR los proveedores cuyo
    tokenizador el núcleo no reproduce en stdlib — Unigram de SentencePiece,
    con su `precompiled_charsmap`, que es lo que usan TODOS los multilingües
    pequeños de este catálogo.

    Vive en `scripts/`, no en `matrixai/`: el núcleo se instala con
    `dependencies = []` y esto es un instrumento de medida, no producto. Lo
    que se mida con él queda marcado en el artefacto, porque un número que
    solo se obtiene instalando algo más no es el mismo número.
    """

    def __init__(self, ruta: Path, max_longitud: int) -> None:
        from tokenizers import Tokenizer

        self._t = Tokenizer.from_file(str(ruta))
        self._t.no_padding()
        self._t.enable_truncation(max_length=max_longitud)
        self.max_longitud = max_longitud
        self.unk_token, relleno = tokens_especiales_de(ruta)
        # `None` si el tokenizador no declara relleno: solo lo necesita un grafo
        # transformer, y eso se comprueba al cargar el proveedor, no aquí.
        self.pad_id = self._t.token_to_id(relleno) if relleno is not None else None

    def codificar(self, texto: str, *, con_especiales: bool = True):
        from types import SimpleNamespace

        e = self._t.encode(texto, add_special_tokens=con_especiales)
        return SimpleNamespace(ids=e.ids, tokens=e.tokens)


def accion_medir(ids: list[str], repeticiones: int, hilos: int, tokenizador_externo: bool = False) -> None:
    from matrixai.text.embeddings.proveedor import ProveedorDeEmbeddings, ProveedorError

    corpus = cargar_corpus()
    datos = leer_artefacto()
    datos["corpus_de_medida"] = {
        "nombre": "FLORES-200 devtest (spa_Latn / eng_Latn)",
        "url": CORPUS_URL,
        "sha256": CORPUS_SHA256,
        "bytes": CORPUS_BYTES,
        "licencia": CORPUS_LICENCIA,
        "n_textos_por_idioma": corpus["n"],
        "por_que": "frases reales alineadas una a una en es y en, con el artículo del "
        "que salen — permite medir cobertura de idioma sin etiquetar nada a mano",
    }
    for cid in ids:
        c = por_id(cid)
        bloque = datos["proveedores"].setdefault(cid, {})
        externo = None
        if tokenizador_externo and c.tokenizador != "wordpiece":
            from matrixai.text.embeddings.proveedor import leer_composicion, elegir_longitud, TOPE_SECUENCIA

            comp = leer_composicion(raiz_cache() / cid)
            largo, _ = elegir_longitud(list(comp.longitudes_declaradas), tope=TOPE_SECUENCIA)
            externo = _TokenizadorDeFuera(raiz_cache() / cid / "tokenizer.json", largo)
        try:
            proveedor = ProveedorDeEmbeddings.cargar(
                c,
                hilos=hilos,
                verificar_digests=medicion.ficheros_fijados(bloque) if "ficheros" in bloque else None,
                tokenizador=externo,
            )
        except (ProveedorError, DescargaError) as e:
            bloque["medicion"] = {"no_medido_porque": str(e)}
            print(f"NO MEDIDO {cid}: {e}")
            continue

        if externo is not None and proveedor.forma != "bolsa" and externo.pad_id is None:
            raise SystemExit(f"{cid}: su grafo rellena lotes y su tokenizer.json no declara `padding`")

        import time

        t0 = time.perf_counter()
        ProveedorDeEmbeddings.cargar(c, hilos=hilos, tokenizador=externo)
        carga_s = round(time.perf_counter() - t0, 3)

        print(f"\n== midiendo {cid} (hilos={hilos})")
        entorno = entorno_de_medida()
        describe = proveedor.describe()
        control = medicion.control_de_linealidad(proveedor, corpus["en"])
        tiempos = {
            idioma: medicion.cronometrar(proveedor, corpus[idioma], repeticiones=repeticiones)
            for idioma in ("es", "en")
        }
        tokenizacion = {idioma: medicion.medir_tokenizacion(proveedor, corpus[idioma]) for idioma in ("es", "en")}
        vectores = {idioma: proveedor.codificar(corpus[idioma]) for idioma in ("es", "en")}
        cruzado = medicion.recuperacion_translingue(vectores["es"], vectores["en"])
        coherencia = {
            idioma: medicion.coherencia_por_documento(vectores[idioma], corpus["articulos"])
            for idioma in ("es", "en")
        }
        bloque["medicion"] = {
            # El entorno va DENTRO de cada fila, no una vez para todo el
            # artefacto: cada proveedor se mide cuando se mide, y la máquina
            # no está igual de tranquila en los dos momentos. Un `entorno`
            # global le pondría a los tres primeros la carga del cuarto, y eso
            # es declarar lo que se pidió en vez de lo que pasó.
            "entorno": entorno,
            "describe": describe,
            "dimension": proveedor.dimension,
            "carga_del_modelo_s": carga_s,
            "hilos": hilos,
            "medido_con_tokenizador_de_fuera_del_nucleo": bool(externo),
            "que_significa_eso": (
                "el núcleo NO puede producir estos vectores tal cual: su tokenizador es "
                "Unigram de SentencePiece y hace falta la dependencia `tokenizers`. El "
                "número es real, pero solo se obtiene con esa dependencia instalada."
            )
            if externo
            else "",
            "control_del_instrumento": control,
            "tiempo": tiempos,
            "tokenizacion": tokenizacion,
            "recuperacion_translingue_es_en": cruzado,
            "coherencia_por_documento": coherencia,
            "metodo": {
                "tiempo": f"{repeticiones} repeticiones sobre {corpus['n']} textos reales por idioma, "
                "una pasada de calentamiento tirada, reloj solo alrededor de codificar "
                "(tokenizar + inferir + agrupar); la carga del modelo va aparte",
                "idiomas": "tokens por palabra y % de [UNK] miden lo que el TOKENIZADOR cubre; "
                "la coherencia por documento (AUC) mide si el modelo distingue temas DENTRO "
                "de cada idioma; la recuperación translingüe mide si los dos idiomas "
                "comparten espacio (un modelo monolingüe da ahí casi el azar aunque el "
                "idioma le funcione)",
            },
        }
        try:
            bloque["medicion"]["entorno"]["carga_al_terminar"] = Path("/proc/loadavg").read_text().split()[0]
        except OSError:
            pass
        print(
            f"   dim={proveedor.dimension}  s/1000 es={tiempos['es']['segundos_por_1000_textos']} "
            f"en={tiempos['en']['segundos_por_1000_textos']}  lineal={control['lineal']}"
        )
        print(
            f"   tok/palabra es={tokenizacion['es']['tokens_por_palabra']} en={tokenizacion['en']['tokens_por_palabra']}"
            f"  AUC es={coherencia['es']['auc']} en={coherencia['en']['auc']}"
            f"  P@1 es->en={cruzado['p_at_1_a_b']}"
        )
    escribir_artefacto(datos)


def accion_comprobar() -> int:
    """¿El artefacto sigue cuadrando con lo que hay en la caché?"""
    datos = leer_artefacto()
    fallos = 0
    for cid, bloque in datos.get("proveedores", {}).items():
        if not bloque.get("descargado"):
            print(f"- {cid}: no descargado ({bloque.get('candidato', {}).get('no_descargado_porque') or 'sin motivo'})")
            continue
        d = raiz_cache() / cid
        # «No está en esta caché» y «está y no cuadra» NO son lo mismo, y
        # mezclarlos hace ruido justo donde hay que mirar: lo primero pasa en
        # cuanto alguien apunta MATRIXAI_EMBEDDINGS_HOME a otro sitio; lo
        # segundo significa que unos pesos de terceros cambiaron debajo.
        rotos, ausentes = [], []
        for f in medicion.ficheros_fijados(bloque):
            ruta = d / f.ruta
            if not ruta.is_file():
                ausentes.append(f.ruta)
                continue
            try:
                verificar_fichero(ruta, f)
            except DescargaError as e:
                print(f"ROJO {cid}/{f.ruta}: {e}")
                rotos.append(f.ruta)
        fallos += len(rotos)
        total = len(bloque["ficheros"])
        if not rotos and not ausentes:
            print(f"ok   {cid}: {total} ficheros verificados")
        elif rotos:
            print(f"ROJO {cid}: {len(rotos)} de {total} ficheros NO cuadran")
        else:
            print(f"-    {cid}: no está en esta caché ({len(ausentes)} de {total} ficheros); "
                  f"nada que comprobar en {d}")
    return fallos


def accion_tabla() -> None:
    """La tabla del contrato, generada. Tampoco esa se teclea a mano."""
    from matrixai.text.embeddings import cobertura

    datos = leer_artefacto()
    print("| proveedor | familia | licencia | pesos | dim | s/1.000 es | s/1.000 en | tok/palabra es · en | AUC es · en | es | en |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cid, fila in datos["proveedores"].items():
        m = fila.get("medicion") or {}
        t = m.get("tiempo") or {}
        tk = m.get("tokenizacion") or {}
        co = m.get("coherencia_por_documento") or {}
        def _n(x, s="—"):
            return s if x is None else x
        marca = " ᵈ" if m.get("medido_con_tokenizador_de_fuera_del_nucleo") else ""
        print(
            f"| `{cid}`{marca} | {fila['familia']} | {fila['licencia']['spdx']} | "
            f"{fila['pesos_bytes']/1e6:.1f} MB | {_n(m.get('dimension'))} | "
            f"{_n((t.get('es') or {}).get('segundos_por_1000_textos'))} | "
            f"{_n((t.get('en') or {}).get('segundos_por_1000_textos'))} | "
            f"{_n((tk.get('es') or {}).get('tokens_por_palabra'))} · {_n((tk.get('en') or {}).get('tokens_por_palabra'))} | "
            f"{_n((co.get('es') or {}).get('auc'))} · {_n((co.get('en') or {}).get('auc'))} | "
            f"**{cobertura.veredicto(cid, 'es').estado}** | **{cobertura.veredicto(cid, 'en').estado}** |"
        )
    print()
    print("ᵈ medido con la dependencia `tokenizers`, que el núcleo no lleva.")
    for cid, fila in datos["proveedores"].items():
        m = fila.get("medicion") or {}
        if not m.get("tiempo"):
            motivo = (fila.get("candidato") or {}).get("no_descargado_porque") or m.get("no_medido_porque") or "sin motivo"
            print(f"\n- **`{cid}` no se midió**: {motivo}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fijar", action="store_true")
    p.add_argument("--descargar", action="store_true")
    p.add_argument("--medir", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--tabla", action="store_true", help="escribe la tabla del catálogo en markdown")
    p.add_argument("--acepto-la-licencia-de", action="append", default=[], metavar="ID")
    p.add_argument("--repeticiones", type=int, default=5)
    p.add_argument("--hilos", type=int, default=1)
    p.add_argument(
        "--tokenizador-externo",
        action="store_true",
        help="mide también los proveedores cuyo tokenizador el núcleo no reproduce, "
        "usando la dependencia `tokenizers`; queda marcado en el artefacto",
    )
    p.add_argument("ids", nargs="*", default=[])
    a = p.parse_args()

    ids = a.ids or [c.id for c in CANDIDATOS]
    if a.fijar:
        accion_fijar(ids)
    if a.descargar:
        accion_descargar(ids, set(a.acepto_la_licencia_de))
    if a.medir:
        accion_medir(ids, a.repeticiones, a.hilos, a.tokenizador_externo)
    if a.tabla:
        accion_tabla()
    if a.comprobar:
        return accion_comprobar()
    if not any([a.fijar, a.descargar, a.medir, a.comprobar, a.tabla]):
        p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
