"""150-C4 — el ORIGEN del fichero en el recibo y en el manifiesto de reproducción (núcleo).

Un Excel entra al producto convertido a un CSV (150-C1) y un CSV `;` con decimal español entra con la coma pasada a punto
(150-C2): **lo que se hashea y lo que el modelo vio es el CSV derivado**. Este corte hace que el recibo lo diga: de qué
fichero salió, con qué hoja, la huella del ORIGINAL y cuántas cosas se declararon.

Las INVARIANTES del contrato, cada una con su prueba con nombre (los sabotajes de cada una, en
`~/.agentes/150/c4/sabotajes.md`):

1. el hash que se verifica sigue siendo el del CSV derivado;
2. el del original es informativo y NO entra en R1–R3 (y se dice);
3. un paquete SIN `origen_del_fichero` se lee y se verifica igual que hoy (fixture generado con el código de ANTES);
4. uno CON él lo lleva el verificador sin tropezar;
5. la captura del run es CERRADA: la clave está dada de alta (si no, saldría como «parámetro de generación») y solo la
   lleva la versión 1.3;
6. un CSV sin conversión ni declaración NO lleva `origen_del_fichero`.

CONVENCIÓN DEL FICHERO: funciones `test_*` de pytest (y clases `Test*`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import pytest

from matrixai.export import origen_del_fichero as odf
from matrixai.export import reproduce as rp
from matrixai.export.manifiesto_de_estudio import manifiesto_de_estudio
from matrixai.export.reproduce import ReproduceManifestError
from matrixai.export.tripod import ficha_tripod
from matrixai.export.verify import verify_package

RAIZ = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "tablas"
ANTES = pathlib.Path(__file__).resolve().parent / "fixtures" / "origen_del_fichero" / "paquete_antes"

try:
    import python_calamine  # noqa: F401
    HAY_CALAMINE = True
except ImportError:
    HAY_CALAMINE = False

requiere_calamine = pytest.mark.skipif(
    not HAY_CALAMINE, reason="falta python-calamine (extra `tablas`): pip install 'matrixai-core[tablas]'")

_SHA_ORIGINAL = hashlib.sha256(b"el libro de Excel tal como lo subio la persona").hexdigest()


def _declaracion(tipo: str, cuantas: int = 1, columna: str | None = None) -> dict:
    return {"tipo": tipo, "columna": columna, "cuantas": cuantas, "detalle": {},
            "texto": {"es": f"se hizo {tipo}", "en": f"{tipo} was done"}}


def _origen_a_mano(**cambios) -> dict:
    """Un origen con la forma de `TablaConvertida.origen` (150-C1): tres declaraciones, una hoja y un nombre."""
    origen = {
        "esquema": 1, "formato": "xlsx", "nombre": "ventas.xlsx", "sha256": _SHA_ORIGINAL, "bytes": 4096,
        "hoja": "Pedidos", "fila_de_cabecera": 1, "filas_de_cabecera": 1, "rango": "A1:D10",
        "filas_excluidas": [{"fila": 4, "texto": "Subtotal Ana"}],
        "conversor": {"nombre": "matrixai.tablas", "versiones": {"python-calamine": "0.9.0"}},
        "declaraciones": [_declaracion("filas_excluidas", 1), _declaracion("columnas_vacias_quitadas", 2),
                          _declaracion("nombre_repetido_renombrado", 1, "AC_2")],
    }
    origen.update(cambios)
    return origen


# ── el módulo: validar, componer, redactar ───────────────────────────────────────────────────────

@requiere_calamine
def test_el_origen_REAL_de_una_conversion_de_excel_pasa_el_validador_y_viaja_entero():
    """El validador se prueba contra lo que la conversión de verdad devuelve, no solo contra un diccionario a mano."""
    from matrixai.tablas import convertir
    datos = (FIXTURES / "erp_totales.xlsx").read_bytes()
    tabla = convertir(datos, "erp_totales.xlsx", fila_de_cabecera=1, filas_a_excluir=[4, 7, 9, 10])
    origen = odf.normalizar_origen_del_fichero(tabla.origen)
    assert origen == tabla.origen  # nada se quita ni se renombra por el camino
    assert origen["sha256"] == hashlib.sha256(datos).hexdigest()
    assert odf.numero_de_declaraciones(origen) == len(tabla.origen["declaraciones"]) >= 1
    # y el JSON canónico lo digiere el manifiesto (sin NaN ni objetos raros)
    rp.canonical_json(origen)


@pytest.mark.parametrize("cambio, que", [
    ({"sha256": _SHA_ORIGINAL[:16]}, "huella corta (`data_…`) no es una prueba"),
    ({"sha256": _SHA_ORIGINAL.upper()}, "mayúsculas"),
    ({"sha256": None}, "sin huella del original"),
    ({"formato": ""}, "sin formato"),
    ({"formato": "x" * 40}, "formato que no es un nombre"),
    ({"esquema": 99}, "esquema que este núcleo no sabe leer"),
    ({"esquema": True}, "esquema booleano"),
    ({"bytes": -1}, "bytes negativos"),
    ({"bytes": 1.5}, "bytes decimales"),
    ({"nombre": "a\x00b.xlsx"}, "nombre con un carácter de control"),
    ({"hoja": "x" * 300}, "hoja larguísima"),
    ({"declaraciones": "tres"}, "declaraciones que no son una lista"),
    ({"declaraciones": ["una"]}, "declaración que no es un objeto"),
    ({"declaraciones": [{"cuantas": 1}]}, "declaración sin tipo"),
    ({"declaraciones": [{"tipo": "x", "cuantas": -2}]}, "cuantas negativo"),
])
def test_un_origen_mal_formado_se_rechaza_con_su_motivo(cambio, que):
    with pytest.raises(ReproduceManifestError):
        odf.normalizar_origen_del_fichero(_origen_a_mano(**cambio))


def test_un_origen_que_no_es_un_objeto_o_pesa_demasiado_se_rechaza():
    for malo in ("ventas.xlsx", [1], 3):
        with pytest.raises(ReproduceManifestError):
            odf.normalizar_origen_del_fichero(malo)
    enorme = _origen_a_mano(declaraciones=[_declaracion("nombre_normalizado", 1, f"c{i}") for i in range(12000)])
    with pytest.raises(ReproduceManifestError, match="bytes"):
        odf.normalizar_origen_del_fichero(enorme)


def test_el_origen_normalizado_es_una_copia_y_no_toca_lo_que_llega():
    crudo = _origen_a_mano()
    antes = json.dumps(crudo, sort_keys=True)
    salida = odf.normalizar_origen_del_fichero(crudo)
    salida["hoja"] = "OTRA"
    salida["declaraciones_nuevas"] = True
    assert json.dumps(crudo, sort_keys=True) == antes
    assert crudo["hoja"] == "Pedidos"


def test_las_dos_listas_de_declaraciones_se_cuentan_juntas():
    origen = _origen_a_mano(declaraciones_de_entrada=[_declaracion("decimal_coma_convertido", 120, "precio")])
    assert odf.numero_de_declaraciones(origen) == 4
    assert odf.numero_de_declaraciones(_origen_a_mano(declaraciones=[])) == 0


# ── invariante 6: un CSV sin conversión ni declaración NO lleva origen ───────────────────────────

def test_un_csv_sin_conversion_ni_declaracion_NO_lleva_origen_del_fichero():
    """No se inventa un origen: ni vacío, ni con la huella de sí mismo, ni con `declaraciones: []`."""
    csv = "a,b\n1,2\n3,4\n"
    assert odf.origen_para_el_estudio(None, texto_recibido=csv, declaraciones_de_entrada=[]) is None
    assert odf.origen_para_el_estudio(None, texto_recibido=csv, declaraciones_de_entrada=None) is None
    # y con un nombre de fichero tampoco: el nombre no es una conversión
    assert odf.origen_para_el_estudio(None, texto_recibido=csv, declaraciones_de_entrada=[], nombre="datos.csv") is None


def test_un_csv_de_siempre_no_gana_la_clave_en_el_manifiesto_de_un_estudio():
    manifiesto = manifiesto_de_estudio(filas=40, huella_del_dataset=hashlib.sha256(b"x").hexdigest())
    assert "origen_del_fichero" not in manifiesto["artifacts"]["dataset"]
    assert "origen_del_fichero" not in json.dumps(manifiesto)


def test_un_csv_punto_y_coma_con_decimal_coma_lleva_su_origen_con_la_huella_del_texto_recibido():
    from matrixai.training.data import normalize_csv_text_declarando
    recibido = "fecha;precio;unidades\n2024-01-01;12,5;3\n2024-01-02;7,25;4\n"
    canonico, declaraciones = normalize_csv_text_declarando(recibido)
    assert declaraciones and declaraciones[0]["tipo"] == "decimal_coma_convertido"
    origen = odf.origen_para_el_estudio(None, texto_recibido=recibido, declaraciones_de_entrada=declaraciones,
                                        nombre="ventas.csv")
    assert origen["formato"] == "csv" and origen["nombre"] == "ventas.csv"
    # la huella es la del texto TAL COMO LLEGÓ (con `;` y coma), no la del canónico que se hashea en el recibo
    assert origen["sha256"] == hashlib.sha256(recibido.encode("utf-8")).hexdigest()
    assert origen["sha256"] != hashlib.sha256(canonico.encode("utf-8")).hexdigest()
    assert origen["bytes"] == len(recibido.encode("utf-8")) and origen["huella_de"] == "texto_recibido"
    assert odf.numero_de_declaraciones(origen) == len(declaraciones)
    assert origen["declaraciones"] == declaraciones


def test_un_libro_convertido_con_declaraciones_de_entrada_las_lleva_aparte_y_cuenta_las_dos():
    entrada = [_declaracion("decimal_coma_convertido", 5, "precio")]
    origen = odf.origen_para_el_estudio(_origen_a_mano(), texto_recibido="a,b\n", declaraciones_de_entrada=entrada)
    assert origen["declaraciones_de_entrada"] == entrada
    assert len(origen["declaraciones"]) == 3 and odf.numero_de_declaraciones(origen) == 4
    # sin lo que decir al entrar, el origen del libro pasa tal cual (sin clave vacía)
    solo = odf.origen_para_el_estudio(_origen_a_mano(), texto_recibido="a,b\n", declaraciones_de_entrada=[])
    assert "declaraciones_de_entrada" not in solo


def test_el_origen_para_el_estudio_rechaza_un_origen_mal_formado_en_vez_de_callarlo():
    with pytest.raises(ReproduceManifestError):
        odf.origen_para_el_estudio(_origen_a_mano(sha256="corta"), texto_recibido="a,b\n", declaraciones_de_entrada=[])


def test_el_texto_del_origen_en_espanol_y_en_ingles_dice_lo_que_dice_el_contrato():
    origen = _origen_a_mano()
    assert odf.texto_del_origen(origen, "es").startswith(
        f"Excel `ventas.xlsx`, hoja `Pedidos`, huella del original `{_SHA_ORIGINAL}`, 3 declaraciones")
    assert odf.texto_del_origen(origen, "en").startswith(
        f"Excel `ventas.xlsx`, sheet `Pedidos`, digest of the original `{_SHA_ORIGINAL}`, 3 declarations")
    # singular y cero
    assert ", 1 declaración" in odf.texto_del_origen(_origen_a_mano(declaraciones=[_declaracion("x")]), "es")
    assert ", 1 declaration" in odf.texto_del_origen(_origen_a_mano(declaraciones=[_declaracion("x")]), "en")
    assert ", 0 declaraciones" in odf.texto_del_origen(_origen_a_mano(declaraciones=[]), "es")
    assert "1 declaraciones" not in odf.texto_del_origen(_origen_a_mano(declaraciones=[_declaracion("x")]), "es")


def test_el_texto_del_origen_dice_que_la_huella_del_original_es_informativa_en_los_dos_idiomas():
    es, en = odf.texto_del_origen(_origen_a_mano(), "es"), odf.texto_del_origen(_origen_a_mano(), "en")
    assert "informativo" in es and "no entra en la verificación" in es
    assert "informative" in en and "takes part in no" not in en and "no part in verification" in en


def test_lo_que_el_origen_no_sabe_se_omite_y_no_se_rellena_en_el_texto():
    origen = _origen_a_mano()
    del origen["nombre"], origen["hoja"]
    es = odf.texto_del_origen(origen, "es")
    assert es.startswith(f"Excel, huella del original `{_SHA_ORIGINAL}`, 3 declaraciones")
    assert "None" not in es and "``" not in es
    csv = odf.origen_de_un_csv("a;b\n1,5;x\n", [_declaracion("decimal_coma_convertido", 1, "a")])
    assert odf.texto_del_origen(csv, "es").startswith("CSV, huella del original `")
    assert odf.texto_del_origen(csv, "en").startswith("CSV, digest of the original `")


def test_el_bloque_del_manifiesto_dice_que_es_informativo_y_por_que_en_es_y_en():
    bloque = odf.bloque_del_manifiesto(odf.normalizar_origen_del_fichero(_origen_a_mano()))
    assert bloque["informativo"] is True
    assert set(bloque["nota"]) == {"es", "en"}
    assert "R1–R3" in bloque["nota"]["es"] and "R1–R3" in bloque["nota"]["en"]
    assert "artifacts.dataset.sha256" in bloque["nota"]["es"]


# ── invariante 5: la captura del run es cerrada ──────────────────────────────────────────────────

def _captura(version="1.3", **extra) -> dict:
    mxtrain = "TRAINING t\nEND\n"
    base = {"schema_version": version, "mxai_sha256": hashlib.sha256(b"m").hexdigest(),
            "mxtrain_sha256": hashlib.sha256(mxtrain.encode()).hexdigest(), "mxtrain_text": mxtrain,
            "dataset_sha256_raw": hashlib.sha256(b"csv").hexdigest(), "dataset_rows": 3}
    base.update(extra)
    return base


def test_origen_del_fichero_esta_dado_de_alta_en_las_claves_del_nucleo_de_la_captura():
    """Sin el alta saldría publicado como parámetro de generación (el comentario de `_RUN_PROVENANCE_CORE_KEYS`)."""
    assert "origen_del_fichero" in rp._RUN_PROVENANCE_CORE_KEYS
    assert odf.CLAVE == "origen_del_fichero"


def test_la_clave_dada_de_alta_no_se_cuela_en_generation_ni_se_valida_como_parametro():
    cap = rp._normalize_run_provenance(_captura(origen_del_fichero=_origen_a_mano()))
    fuente_gen = {k: v for k, v in cap.items() if k not in rp._RUN_PROVENANCE_CORE_KEYS}
    assert "origen_del_fichero" not in fuente_gen
    # y el bloque `generation` de un manifiesto que lo lleva no lo publica como parámetro
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "model.mxai").write_text("NETWORK N\n")
        manifiesto = rp.build_reproduce_manifest(
            d, run_provenance=_captura(mxai_sha256=hashlib.sha256(b"NETWORK N\n").hexdigest(),
                                       origen_del_fichero=_origen_a_mano()))
    assert "origen_del_fichero" not in manifiesto["generation"]
    assert manifiesto["artifacts"]["dataset"]["origen_del_fichero"]["sha256"] == _SHA_ORIGINAL


def test_solo_la_captura_1_3_puede_llevar_el_origen_y_una_version_desconocida_se_rechaza():
    # 1.3 con origen: bien, y una captura SIN origen en 1.2 sigue siendo válida
    assert rp._normalize_run_provenance(_captura("1.3", origen_del_fichero=_origen_a_mano()))
    assert rp._normalize_run_provenance(_captura("1.2"))
    assert rp._normalize_run_provenance(_captura("1.3"))  # 1.3 sin origen también se lee
    # 1.0–1.2 con la clave: se RECHAZA (un lector de esas versiones la publicaría como parámetro)
    for v in ("1.0", "1.1", "1.2"):
        with pytest.raises(ReproduceManifestError, match="origen_del_fichero"):
            rp._normalize_run_provenance(_captura(v, origen_del_fichero=_origen_a_mano()))
    # una versión que este núcleo no conoce no se interpreta a medias, lleve o no el origen
    for v in ("1.4", "2.0", None):
        with pytest.raises(ReproduceManifestError, match="schema_version"):
            rp._normalize_run_provenance(_captura(v, origen_del_fichero=_origen_a_mano()))


def test_una_captura_con_un_origen_mal_formado_se_rechaza_y_un_null_explicito_se_quita():
    with pytest.raises(ReproduceManifestError, match="run_provenance.origen_del_fichero"):
        rp._normalize_run_provenance(_captura(origen_del_fichero=_origen_a_mano(sha256="x")))
    cap = rp._normalize_run_provenance(_captura("1.2", origen_del_fichero=None))
    assert "origen_del_fichero" not in cap  # «origen vacío» no existe


def test_la_version_de_la_captura_del_playground_y_la_del_validador_son_la_misma():
    from matrixai import playground
    assert playground.RUN_PROVENANCE_SCHEMA_VERSION_CON_ORIGEN == odf.VERSION_DE_LA_CAPTURA_CON_ORIGEN \
        == rp._RUN_PROVENANCE_SCHEMA_VERSIONS[-1] == "1.3"
    assert playground.RUN_PROVENANCE_SCHEMA_VERSION == rp._RUN_PROVENANCE_SCHEMA_VERSION_SIN_ORIGEN == "1.2"


def test_capturar_run_sin_origen_sigue_escribiendo_1_2_y_con_origen_1_3():
    sin = rp.capturar_run(mxai_text="NETWORK N\n", mxtrain_text="TRAINING t\nEND\n", dataset_csv="a,b\n1,2\n")
    assert sin["schema_version"] == "1.2" and "origen_del_fichero" not in sin
    con = rp.capturar_run(mxai_text="NETWORK N\n", mxtrain_text="TRAINING t\nEND\n", dataset_csv="a,b\n1,2\n",
                          origen_del_fichero=_origen_a_mano())
    assert con["schema_version"] == "1.3" and con["origen_del_fichero"]["hoja"] == "Pedidos"


# ── el paquete clásico (captura 1.3) de punta a punta: manifiesto, verificador, ficha ───────────

def _paquete_regenerable(*, origen=None, romper_el_digest=False, sha_del_origen_correcta=False) -> pathlib.Path:
    """Un paquete honesto cuyo dataset SE REGENERA (R1 puede dar PASS de verdad), con o sin origen.

    `romper_el_digest`: el digest que declara el paquete NO es el del dataset que sale al regenerar.
    `sha_del_origen_correcta`: el sha256 del ORIGEN es el del dataset regenerado (la trampa: si R1 mirase el origen,
    un paquete con el digest roto pasaría)."""
    from matrixai.export.reproduce import (_epochs_from_training, añadir_inventario_de_ficheros,
                                           write_reproduce_manifest)
    from matrixai.playground import _generate_synthetic_dataset
    from matrixai.training.dataset_project import generate_project_from_dataset

    filas = "".join(f"{c},{c + 273.15}\n" for c in range(30))
    proj = generate_project_from_dataset(
        "centigrados,prediccionKelvin\n" + filas, "prediccionKelvin",
        column_type_overrides={"centigrados": "number"}, column_range_overrides={"centigrados": (0.0, 29.0)})
    gen = _generate_synthetic_dataset(proj["mxai"], proj["training_text"], 30, 7, "coherent",
                                      field_ranges_override=proj.get("field_ranges"))
    csv = gen["csv_text"]
    sha_bueno = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    sha = hashlib.sha256((csv + "otra fila\n").encode("utf-8")).hexdigest() if romper_el_digest else sha_bueno
    epocas = _epochs_from_training(proj["training_text"])

    d = pathlib.Path(tempfile.mkdtemp(prefix="c150c4-"))
    (d / "model.mxai").write_text(proj["mxai"], encoding="utf-8")
    (d / "training.mxtrain").write_text(proj["training_text"], encoding="utf-8")
    (d / "recipe.txt").write_text("regenerable\n", encoding="utf-8")
    captura = {
        "schema_version": "1.2",
        "mxai_sha256": hashlib.sha256(proj["mxai"].encode()).hexdigest(),
        "mxtrain_sha256": hashlib.sha256(proj["training_text"].encode()).hexdigest(),
        "mxtrain_text": proj["training_text"],
        "recipe_sha256": hashlib.sha256(b"regenerable\n").hexdigest(), "recipe_text": "regenerable\n",
        "seeds": {"dataset": 7}, "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
        "dataset_rows": 30, "dataset_rows_used": 30,
        "epochs_effective": epocas, "epochs_ran": epocas, "warm_start": False, "mode": "coherent",
        "field_ranges": proj.get("field_ranges") or {},
        "recipe_verification": {"verified": True, "code": "regenera_el_dataset"},
    }
    if origen is not None:
        origen = dict(origen)
        if sha_del_origen_correcta:
            origen["sha256"] = sha_bueno
        captura["schema_version"] = "1.3"
        captura["origen_del_fichero"] = origen
    write_reproduce_manifest(d, training_filename="training.mxtrain", recipe_filename="recipe.txt",
                             dataset_sha256=sha, dataset_rows=30, generation={"seeds": {"dataset": 7}},
                             run_provenance=captura, weights_source="trained")
    añadir_inventario_de_ficheros(d)
    return d


def _manifiesto(d: pathlib.Path) -> dict:
    return json.loads((d / "reproduce.json").read_text(encoding="utf-8"))


def _informe(d: pathlib.Path) -> dict:
    inf = verify_package(d)
    return {"exit": inf["exit_code"], "ok": inf["ok"], "fully": inf["fully_checked"],
            "etapas": {n: (e["status"], e.get("reason")) for n, e in inf["stages"].items()}}


def test_el_manifiesto_publica_el_origen_en_artifacts_dataset_informativo_y_con_la_huella_derivada_al_lado():
    d = _paquete_regenerable(origen=_origen_a_mano())
    m = _manifiesto(d)
    dataset = m["artifacts"]["dataset"]
    bloque = dataset["origen_del_fichero"]
    assert bloque["formato"] == "xlsx" and bloque["hoja"] == "Pedidos" and bloque["sha256"] == _SHA_ORIGINAL
    assert bloque["informativo"] is True and set(bloque["nota"]) == {"es", "en"}
    # INVARIANTE 1: `artifacts.dataset.sha256` es el del CSV DERIVADO (el que entró al entrenamiento), no el del libro
    assert dataset["sha256"] != _SHA_ORIGINAL
    # y el digest del manifiesto lo cubre (tocar el origen a mano lo rompe)
    assert m["manifest_sha256"] == rp.manifest_digest(m)
    m["artifacts"]["dataset"]["origen_del_fichero"]["hoja"] = "Otra"
    assert m["manifest_sha256"] != rp.manifest_digest(m)


def test_el_hash_que_se_verifica_es_el_del_csv_derivado_y_R1_lo_regenera_aunque_haya_origen():
    d = _paquete_regenerable(origen=_origen_a_mano())
    inf = verify_package(d)
    assert inf["stages"]["R1"]["status"] == "PASS", inf["stages"]["R1"]
    m = _manifiesto(d)
    assert inf["stages"]["R1"]["sha256"] == m["artifacts"]["dataset"]["sha256"] != _SHA_ORIGINAL


def test_la_huella_del_original_NO_entra_en_R1_un_digest_roto_no_se_arregla_con_un_origen_que_cuadra():
    """La trampa: el origen lleva el sha256 del dataset que SÍ se regenera, y el digest que declara el paquete está roto.
    R1 debe FALLAR contra el digest del paquete. Si R1 mirase la huella del original, este paquete pasaría."""
    d = _paquete_regenerable(origen=_origen_a_mano(), romper_el_digest=True, sha_del_origen_correcta=True)
    etapa = verify_package(d)["stages"]["R1"]
    assert etapa["status"] == "FAIL", etapa
    assert etapa["expected"] == _manifiesto(d)["artifacts"]["dataset"]["sha256"]
    assert etapa["expected"] != etapa["found"] == _manifiesto(d)["artifacts"]["dataset"]["origen_del_fichero"]["sha256"]


def test_una_cifra_medida_sobre_el_original_es_de_otro_dataset_y_se_declara_conflicto():
    """R3/export: una métrica cuyo `dataset_sha256` es el del ORIGINAL no es del dataset que entrenó (el derivado)."""
    from matrixai.export.reproduce import build_reproduce_manifest
    d = _paquete_regenerable(origen=_origen_a_mano())
    cap = json.loads(json.dumps(rp._normalize_run_provenance(_captura_del_paquete(d))))
    derivado = _manifiesto(d)["artifacts"]["dataset"]["sha256"]
    def conflictos(sha):
        m = build_reproduce_manifest(
            d, training_filename="training.mxtrain", recipe_filename="recipe.txt", run_provenance=cap,
            weights_source="trained",
            metrics=[{"name": "mae", "value": 0.1, "split": "validation", "dataset_sha256": sha}])
        return [c for c in m["conflicts"] if "metrics[0].dataset_sha256" in str(c)]
    assert conflictos(derivado) == []
    assert conflictos(_SHA_ORIGINAL) != []


def _captura_del_paquete(d: pathlib.Path) -> dict:
    """La captura del run que `_paquete_regenerable` metió (reconstruida igual: no se guarda aparte en disco)."""
    m = _manifiesto(d)
    mxtrain = (d / "training.mxtrain").read_text(encoding="utf-8")
    sha = m["artifacts"]["dataset"]["sha256"]
    return {"schema_version": "1.3", "mxai_sha256": hashlib.sha256((d / "model.mxai").read_bytes()).hexdigest(),
            "mxtrain_sha256": hashlib.sha256(mxtrain.encode()).hexdigest(), "mxtrain_text": mxtrain,
            "recipe_sha256": hashlib.sha256(b"regenerable\n").hexdigest(), "recipe_text": "regenerable\n",
            "seeds": {"dataset": 7}, "dataset_sha256_raw": sha, "dataset_sha256_prepared": sha,
            "dataset_rows": 30, "dataset_rows_used": 30, "mode": "coherent",
            "origen_del_fichero": _origen_a_mano()}


def test_un_paquete_CON_origen_lo_lleva_el_verificador_sin_tropezar_y_da_lo_mismo_que_sin_el():
    sin = _informe(_paquete_regenerable())
    con = _informe(_paquete_regenerable(origen=_origen_a_mano()))
    assert con == sin, (sin, con)
    assert con["etapas"]["manifest"][0] == "PASS"
    # también cuando el origen es de un CSV, y con las dos listas de declaraciones
    otro = _origen_a_mano(declaraciones_de_entrada=[_declaracion("decimal_coma_convertido", 9, "precio")])
    assert _informe(_paquete_regenerable(origen=otro)) == sin


def test_un_paquete_CON_origen_se_verifica_tambien_por_la_orden_matrixai_verify():
    d = _paquete_regenerable(origen=_origen_a_mano())
    r = subprocess.run([sys.executable, "-m", "matrixai", "verify", str(d)], capture_output=True, text=True,
                       cwd=RAIZ, timeout=120)
    sin = subprocess.run([sys.executable, "-m", "matrixai", "verify", str(_paquete_regenerable())],
                         capture_output=True, text=True, cwd=RAIZ, timeout=120)
    assert r.returncode == sin.returncode, (r.stdout, r.stderr, sin.stdout)
    assert "Traceback" not in r.stderr and "Traceback" not in r.stdout


def test_un_paquete_SIN_origen_de_antes_del_corte_se_lee_y_se_verifica_igual_que_entonces():
    """INVARIANTE 3 con un paquete generado por el código de ANTES (`0f2c052`, el HEAD del 150-C3), no por el de hoy."""
    esperado = json.loads((ANTES.parent / "esperado_antes.json").read_text(encoding="utf-8"))
    informe = _informe(ANTES)
    assert informe["etapas"] == {k: tuple(v) for k, v in esperado["etapas"].items()}
    assert (informe["exit"], informe["ok"], informe["fully"]) == (esperado["exit"], esperado["ok"], esperado["fully"])
    m = _manifiesto(ANTES)
    assert m["manifest_sha256"] == rp.manifest_digest(m)  # el digest que escribió el código de entonces sigue cuadrando
    assert "origen_del_fichero" not in json.dumps(m)
    # y la ficha de ese paquete sale sin ninguna línea de origen
    assert "Origen" not in ficha_tripod(ANTES, locale="es") and "Origin" not in ficha_tripod(ANTES, locale="en")


def test_una_captura_sin_origen_produce_el_mismo_manifiesto_que_antes_byte_a_byte():
    """El `reproduce.json` de ANTES del corte, recomputado ahora con la MISMA captura: el mismo manifiesto."""
    import shutil
    from matrixai.export.reproduce import añadir_inventario_de_ficheros, write_reproduce_manifest
    esperado = json.loads((ANTES.parent / "esperado_antes.json").read_text(encoding="utf-8"))
    captura = json.loads((ANTES.parent / "captura_antes.json").read_text(encoding="utf-8"))
    assert captura["schema_version"] == "1.2"
    copia = pathlib.Path(tempfile.mkdtemp(prefix="c150c4-antes-")) / "p"
    shutil.copytree(ANTES, copia)
    (copia / "reproduce.json").unlink()
    write_reproduce_manifest(
        copia, training_filename="training.mxtrain", recipe_filename="recipe.txt", dataset_sha256=esperado["dataset_sha256"],
        dataset_rows=30, generation={"seeds": {"dataset": 7}}, run_provenance=captura, weights_source="trained")
    añadir_inventario_de_ficheros(copia)
    nuevo, guardado = _manifiesto(copia), _manifiesto(ANTES)
    # el entorno (versiones de bibliotecas) puede diferir entre máquinas: se compara todo lo demás, y el digest que cubre
    # TODO lo demás con el entorno ya puesto de acuerdo
    for m in (guardado, nuevo):
        m.pop("environment"), m.pop("manifest_sha256")
    assert nuevo == guardado


# ── la ficha TRIPOD+AI y el expediente ───────────────────────────────────────────────────────────

def test_la_ficha_tripod_enseña_el_origen_junto_a_la_huella_del_dataset_en_es_y_en():
    d = _paquete_regenerable(origen=_origen_a_mano())
    derivado = _manifiesto(d)["artifacts"]["dataset"]["sha256"]
    ruta = "`reproduce.json#artifacts.dataset.origen_del_fichero`"
    es = ficha_tripod(d, locale="es").splitlines()
    i = next(n for n, l in enumerate(es) if l.startswith("- **Huella del dataset (sha256)**"))
    assert es[i].startswith(f"- **Huella del dataset (sha256)**: {derivado}")
    assert es[i + 1].startswith(
        f"- **Origen**: Excel `ventas.xlsx`, hoja `Pedidos`, huella del original `{_SHA_ORIGINAL}`, 3 declaraciones")
    assert es[i + 1].endswith(f" · {ruta}") and "informativo" in es[i + 1]
    en = ficha_tripod(d, locale="en").splitlines()
    j = next(n for n, l in enumerate(en) if l.startswith("- **Dataset digest (sha256)**"))
    assert en[j + 1].startswith(
        f"- **Origin**: Excel `ventas.xlsx`, sheet `Pedidos`, digest of the original `{_SHA_ORIGINAL}`, 3 declarations")
    assert "informative" in en[j + 1] and en[j + 1].endswith(f" · {ruta}")


def test_la_ficha_tripod_de_un_csv_sin_origen_no_tiene_linea_de_origen_ni_hueco_inventado():
    d = _paquete_regenerable()
    for locale, marca in (("es", "Origen"), ("en", "Origin")):
        ficha = ficha_tripod(d, locale=locale)
        assert f"**{marca}**" not in ficha and "origen_del_fichero" not in ficha


def test_el_expediente_y_probast_incluyen_el_origen_solo_si_el_paquete_lo_declara():
    from matrixai.export.expediente_clinico import ExpedienteClinico
    from matrixai.export.probast import huecos_probast
    con = ExpedienteClinico.desde_paquete(_paquete_regenerable(origen=_origen_a_mano()))
    sin = ExpedienteClinico.desde_paquete(_paquete_regenerable())
    claves_con = [e.clave for e in con.inventario()]
    claves_sin = [e.clave for e in sin.inventario()]
    assert "origen_del_fichero" in claves_con and "origen_del_fichero" not in claves_sin
    assert claves_con[claves_con.index("huella_del_dataset") + 1] == "origen_del_fichero"
    assert [c for c in claves_con if c != "origen_del_fichero"] == claves_sin
    # un CSV sin origen no recibe un hueco «sin origen» en PROBAST: no es una laguna, es que no hubo conversión
    assert "**Origen**" not in huecos_probast(_paquete_regenerable(), locale="es")
    assert f"- **Origen**: Excel `ventas.xlsx`, hoja `Pedidos`, huella del original `{_SHA_ORIGINAL}`, 3 declaraciones" in \
        huecos_probast(_paquete_regenerable(origen=_origen_a_mano()), locale="es")


def test_el_origen_del_expediente_se_lee_como_dato_del_core_y_el_texto_es_el_mismo_que_el_de_la_ficha():
    from matrixai.export.expediente_clinico import ExpedienteClinico, ORIGEN_DEL_FICHERO, texto_del_valor
    d = _paquete_regenerable(origen=_origen_a_mano())
    campo = ExpedienteClinico.desde_paquete(d).campo(ORIGEN_DEL_FICHERO.ruta)
    assert campo.hay_dato
    for idioma in ("es", "en"):
        assert texto_del_valor(ORIGEN_DEL_FICHERO, campo, idioma) == odf.texto_del_origen(campo.valor, idioma)


# ── el manifiesto de un estudio ──────────────────────────────────────────────────────────────────

def test_el_manifiesto_de_un_estudio_lleva_el_origen_en_artifacts_dataset_y_la_huella_derivada_sigue_siendo_la_del_csv():
    derivado = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    m = manifiesto_de_estudio(filas=40, huella_del_dataset=derivado, origen_del_fichero=_origen_a_mano())
    dataset = m["artifacts"]["dataset"]
    assert dataset["sha256"] == derivado and dataset["rows"] == 40
    assert dataset["origen_del_fichero"]["sha256"] == _SHA_ORIGINAL and dataset["origen_del_fichero"]["informativo"] is True
    assert m["manifest_sha256"] == rp.manifest_digest(m)
    # sin origen, el mismo manifiesto sin la clave
    assert "origen_del_fichero" not in manifiesto_de_estudio(filas=40, huella_del_dataset=derivado)["artifacts"]["dataset"]


def test_el_manifiesto_de_un_estudio_rechaza_un_origen_mal_formado():
    with pytest.raises(ReproduceManifestError):
        manifiesto_de_estudio(filas=40, huella_del_dataset=hashlib.sha256(b"x").hexdigest(),
                              origen_del_fichero=_origen_a_mano(sha256="corta"))


def test_la_ficha_de_un_estudio_con_origen_lo_lleva_tambien():
    """El camino del Studio: un paquete de ESTUDIO (con su bloque `study`) cuyo manifiesto trae el origen."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="c150c4-estudio-"))
    m = manifiesto_de_estudio(filas=40, huella_del_dataset=hashlib.sha256(b"a,b\n1,2\n").hexdigest(), modo="rapido",
                              origen_del_fichero=_origen_a_mano())
    (d / "reproduce.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    ficha = ficha_tripod(d, locale="es")
    assert f"- **Origen**: Excel `ventas.xlsx`, hoja `Pedidos`, huella del original `{_SHA_ORIGINAL}`, 3 declaraciones" in ficha
    sin = pathlib.Path(tempfile.mkdtemp(prefix="c150c4-estudio-"))
    (sin / "reproduce.json").write_text(json.dumps(manifiesto_de_estudio(
        filas=40, huella_del_dataset=hashlib.sha256(b"a,b\n1,2\n").hexdigest(), modo="rapido"), indent=2), encoding="utf-8")
    assert "**Origen**" not in ficha_tripod(sin, locale="es")


# ── el entrenamiento: el parámetro llega a la captura del run ────────────────────────────────────

def _entrenar(**kwargs):
    from matrixai.playground import _get_job_status, _submit_training_job, _training_jobs
    from matrixai.training.dataset_project import generate_project_from_dataset
    filas = "".join(f"{c},{c + 273.15}\n" for c in range(100))
    proj = generate_project_from_dataset(
        "centigrados,prediccionKelvin\n" + filas, "prediccionKelvin",
        column_type_overrides={"centigrados": "number"}, column_range_overrides={"centigrados": (0.0, 99.0)})
    sub = _submit_training_job(proj["mxai"], proj["training_text"], proj["csv_text"], 2,
                               field_ranges=proj.get("field_ranges"), target_range=proj.get("target_range"), **kwargs)
    if not sub.get("ok"):
        return sub, None
    for _ in range(300):
        if _get_job_status(sub["job_id"])["status"] in ("done", "error", "cancelled", "timeout"):
            break
        time.sleep(0.2)
    return sub, _training_jobs[sub["job_id"]]


def test_el_entrenamiento_con_origen_lo_guarda_en_la_captura_1_3_y_la_captura_se_lee():
    sub, job = _entrenar(origen_del_fichero=_origen_a_mano())
    assert sub["ok"], sub
    prov = job["run_provenance"]
    assert prov["schema_version"] == "1.3" and prov["origen_del_fichero"]["sha256"] == _SHA_ORIGINAL
    assert prov["origen_del_fichero"] == odf.normalizar_origen_del_fichero(_origen_a_mano())
    # la huella que ata los pesos es la del CSV que entró (derivado), no la del libro
    assert prov["dataset_sha256_raw"] != _SHA_ORIGINAL and prov["dataset_sha256_prepared"] != _SHA_ORIGINAL
    assert rp._normalize_run_provenance(prov)["origen_del_fichero"]["hoja"] == "Pedidos"


def test_el_entrenamiento_sin_origen_NO_lleva_la_clave_y_la_captura_sigue_siendo_1_2():
    sub, job = _entrenar()
    assert sub["ok"], sub
    assert job["run_provenance"]["schema_version"] == "1.2"
    assert "origen_del_fichero" not in job["run_provenance"]
    assert rp._normalize_run_provenance(job["run_provenance"])


def test_el_entrenamiento_con_un_origen_mal_formado_se_rechaza_antes_de_crear_el_trabajo():
    from matrixai.playground import _training_jobs
    antes = set(_training_jobs)
    sub, job = _entrenar(origen_del_fichero=_origen_a_mano(sha256="corta"))
    assert sub["ok"] is False and sub["clave"] == "origen_del_fichero_no_valido" and job is None
    assert set(_training_jobs) == antes  # no hay trabajo a medias
