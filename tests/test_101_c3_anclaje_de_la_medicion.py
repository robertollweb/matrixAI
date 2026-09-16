# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3/101-C4 — un número medido tiene que poder VOLVER A ATARSE.

Las dos deudas que cierra este fichero, con lo que se midió el 2026-09-12
antes de tocar nada:

* **El JSON de la pasada no se podía anclar y su caché era INERTE.**
  `pasada_exploratoria_101_c3_resultado.json` (medido el 09-07) trae 720
  registros y los 720 con `entorno_digest=None` y `motor_digest=None` —
  aquel fichero se escribió antes de que el script guardara sus digests.
  Pasado por el `_cargar_cache`/`_reusable` de hoy: **REUSABLES 0 DE 720**,
  es decir una re-pasada repite los 74,8 min enteros. Y 0 reusados se lee
  como «no había nada que reusar», que es exactamente la media verdad
  tranquilizadora.

* **El informe de 101-C4 tampoco.** Reajustando lightgbm sobre el mismo
  `adult` el 09-12: el `pipeline_digest` guardado es
  `b888d0b1296e5ef6…` y hoy sale `eeea0cdfd354427e…` — **NO reproduce**.
  El `split_plan_digest` sí (`bdef5eb8ef5610ef…`), así que la deriva está
  en el motor, no en la partición; `matrixai-engines` acumuló 33 commits
  entre el 09-07 y el 09-12. Con el JSON en la mano no se puede decir cuál,
  porque no guardaba ni el commit ni las versiones.

CADA MITAD TIENE SU TEST. «El bloque lleva el commit» lo pasaría una
versión que escribe el commit de un árbol sucio como si estuviera limpio,
que es peor que no escribir nada; por eso al lado va «un árbol sucio se
declara sucio». Y «un árbol sucio no es anclable» lo pasaría una versión
que nunca ancla nada; por eso al lado va «un árbol limpio SÍ es anclable».
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ / "benchmarks" / "fase0"))
sys.path.insert(0, str(_RAIZ.parent / "matrixai-engines" / "src"))

import pasada_exploratoria_101_c3 as pasada  # noqa: E402

_DIR_FASE0 = _RAIZ / "benchmarks" / "fase0"
#: Los tres ficheros de medición commiteados ANTES de que existiera el bloque
#: de procedencia, con la fecha en que se midieron. No se rellenan: ver
#: `test_los_tres_JSON_viejos_siguen_verificando_SU_PROPIO_digest`.
_JSON_MEDIDOS = (
    (_DIR_FASE0 / "pasada_exploratoria_101_c3_resultado.json", "digest_resultados_crudos"),
    (_DIR_FASE0 / "informe_101_c4_caso_adult.json", "digest"),
    (_DIR_FASE0 / "informe_101_c4_caso_sick.json", "digest"),
)


def _repo_de_juguete(raiz: Path) -> None:
    """Un repositorio git de verdad, mínimo y LIMPIO. De juguete el tamaño, no
    el mecanismo: `_estado_del_repositorio` llama a git, y un doble de prueba
    que devolviera texto inventado no probaría que lee bien su salida."""
    raiz.mkdir(parents=True, exist_ok=True)
    def correr(*args):
        subprocess.run(("git", "-C", str(raiz), *args), check=True, capture_output=True)

    correr("init", "-q", "-b", "principal")
    correr("config", "user.email", "prueba@ejemplo")
    correr("config", "user.name", "Prueba")
    (raiz / "motor.py").write_text("# uno\n", encoding="utf-8")
    correr("add", "motor.py")
    correr("commit", "-q", "-m", "inicial")


# --- el commit, y su otra mitad: que un árbol sucio lo diga ----------------

def test_el_bloque_lleva_el_commit_de_LOS_DOS_repositorios():
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})
    assert set(bloque["repositorios"]) == {"matrixAI", "matrixai-engines"}
    for nombre, estado in bloque["repositorios"].items():
        assert estado["commit"] is not None, f"{nombre} sin commit en este árbol"
        assert len(estado["commit"]) == 40, f"{nombre}: {estado['commit']!r} no es un sha completo"


def test_un_arbol_LIMPIO_se_declara_limpio(tmp_path):
    _repo_de_juguete(tmp_path / "limpio")
    estado = pasada._estado_del_repositorio(tmp_path / "limpio")
    assert estado["arbol_sucio"] is False
    assert estado["ficheros_modificados"] == []
    assert estado["ficheros_sin_seguimiento"] == []


def test_un_arbol_SUCIO_se_declara_sucio_y_con_el_fichero_que_lo_ensucia(tmp_path):
    """La mitad sin la que la reparación la pasaría una versión que miente: un
    commit de un árbol sucio publicado como si el árbol estuviera limpio."""
    raiz = tmp_path / "sucio"
    _repo_de_juguete(raiz)
    (raiz / "motor.py").write_text("# DOS, sin commitear\n", encoding="utf-8")
    estado = pasada._estado_del_repositorio(raiz)
    assert estado["arbol_sucio"] is True
    assert estado["ficheros_modificados"] == ["motor.py"]


def test_un_fichero_SIN_SEGUIMIENTO_tambien_ensucia_el_arbol(tmp_path):
    """Un módulo nuevo sin commitear puede tapar a otro y cambiar lo medido:
    no contarlo sería declarar limpio un árbol que no lo está."""
    raiz = tmp_path / "con_extra"
    _repo_de_juguete(raiz)
    (raiz / "motor_nuevo.py").write_text("# nadie lo ha commiteado\n", encoding="utf-8")
    estado = pasada._estado_del_repositorio(raiz)
    assert estado["arbol_sucio"] is True
    assert estado["ficheros_sin_seguimiento"] == ["motor_nuevo.py"]
    assert estado["ficheros_modificados"] == []


def test_sin_repositorio_NO_se_inventa_un_commit(tmp_path):
    """Un valor ausente no es un cero, y aquí tampoco es una cadena de relleno:
    `None` CON su motivo."""
    fuera = tmp_path / "esto_no_es_un_repo"
    fuera.mkdir()
    estado = pasada._estado_del_repositorio(fuera)
    assert estado["commit"] is None
    assert estado["arbol_sucio"] is None
    assert estado["motivo"], "se calló el motivo de no tener commit"


# --- `anclable` es una conclusión, y se prueba en los dos sentidos ---------

def test_un_arbol_SUCIO_hace_la_medicion_NO_anclable(tmp_path, monkeypatch):
    raiz = tmp_path / "sucio"
    _repo_de_juguete(raiz)
    (raiz / "motor.py").write_text("# cambiado y sin commitear\n", encoding="utf-8")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"juguete": raiz})
    bloque = pasada.procedencia_de_la_medicion(digests_de_codigo={}, datos_de_entrada={})
    assert bloque["anclable"] is False
    assert any("SUCIO" in aviso for aviso in bloque["avisos"]), bloque["avisos"]
    assert any(estado["commit"][:12] in aviso
              for aviso in bloque["avisos"]
              for estado in bloque["repositorios"].values()), (
        "el aviso no dice QUÉ commit no sirve")


def test_un_arbol_LIMPIO_con_sus_datos_SI_es_anclable(tmp_path, monkeypatch):
    """La otra mitad: sin esto, la reparación la pasaría una versión que
    declara `anclable=False` siempre y nunca ancla nada."""
    raiz = tmp_path / "limpio"
    _repo_de_juguete(raiz)
    datos = tmp_path / "entrada.arff"
    datos.write_text("@relation r\n", encoding="utf-8")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"juguete": raiz})
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": "abc"}, datos_de_entrada={"uno": datos})
    assert bloque["avisos"] == []
    assert bloque["anclable"] is True


def test_un_git_que_NO_responde_deja_la_medicion_sin_anclar(tmp_path, monkeypatch):
    """El host puede no tener git. Entonces la medición sigue siendo válida,
    pero NO anclable -- y lo dice, en vez de publicar un hueco mudo."""
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"juguete": tmp_path / "no_repo"})
    (tmp_path / "no_repo").mkdir()
    bloque = pasada.procedencia_de_la_medicion(digests_de_codigo={}, datos_de_entrada={})
    assert bloque["anclable"] is False
    assert any("SIN COMMIT" in aviso for aviso in bloque["avisos"]), bloque["avisos"]


# --- versiones, datos y el identificador ----------------------------------

def test_las_versiones_de_bibliotecas_van_EN_el_bloque():
    """`versiones_de_bibliotecas()` (102-C1) ya existía y nadie la llamaba
    desde aquí: el hueco estaba en el CABLEADO, no en el API."""
    bloque = pasada.procedencia_de_la_medicion(digests_de_codigo={}, datos_de_entrada={})
    versiones = bloque["versiones_de_bibliotecas"]
    for clave in ("python", "numpy", "matrixai_core", "platform"):
        assert clave in versiones, f"falta {clave} en las versiones"


def test_el_sha256_de_los_datos_es_el_del_fichero_y_va_ENTERO(tmp_path, monkeypatch):
    datos = tmp_path / "entrada.arff"
    datos.write_bytes(b"@relation lo_que_sea\n@data\n1,2,si\n")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {})
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={"uno": datos})
    esperado = hashlib.sha256(datos.read_bytes()).hexdigest()
    assert bloque["datos_de_entrada"]["uno"]["sha256"] == esperado
    assert len(esperado) == 64, "una huella truncada es cómoda de leer y no es una prueba"
    assert bloque["datos_de_entrada"]["uno"]["bytes"] == datos.stat().st_size


def test_un_dato_de_entrada_AUSENTE_se_declara_y_rompe_el_anclaje(tmp_path, monkeypatch):
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {})
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={"fantasma": tmp_path / "no_esta.arff"})
    assert bloque["datos_de_entrada"]["fantasma"]["sha256"] is None
    assert bloque["datos_de_entrada"]["fantasma"]["motivo"]
    assert bloque["anclable"] is False


def test_el_procedencia_id_CAMBIA_si_cambia_lo_que_describe(tmp_path, monkeypatch):
    """Sin este test, los de arriba los pasaría un bloque bien escrito cuyo
    identificador es una constante."""
    datos = tmp_path / "entrada.arff"
    datos.write_bytes(b"uno")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {})
    primero = pasada.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": "aaa"}, datos_de_entrada={"uno": datos})
    segundo = pasada.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": "bbb"}, datos_de_entrada={"uno": datos})
    assert primero["procedencia_id"] != segundo["procedencia_id"]
    datos.write_bytes(b"dos")
    tercero = pasada.procedencia_de_la_medicion(
        digests_de_codigo={"entorno": "bbb"}, datos_de_entrada={"uno": datos})
    assert tercero["procedencia_id"] != segundo["procedencia_id"], (
        "el id no cambió al cambiar los DATOS de entrada")


# --- los JSON ya escritos: se clasifican, no se rellenan -------------------

def test_un_JSON_sin_bloque_se_declara_SIN_PROCEDENCIA():
    declarada = pasada.procedencia_declarada({"resultados": [], "digest": "lo que sea"})
    assert declarada["estado"] == "sin_procedencia"
    assert declarada["anclable"] is False
    assert declarada["explicacion"] == pasada.SIN_PROCEDENCIA


def test_un_JSON_con_bloque_NO_anclable_no_se_confunde_con_uno_sin_bloque():
    """Los dos estados dicen «no te fíes», pero por motivos distintos: uno no
    sabe nada, el otro sabe exactamente qué falla. Confundirlos sería perder
    justo el motivo."""
    declarada = pasada.procedencia_declarada(
        {"procedencia": {"anclable": False, "avisos": ["matrixAI: ARBOL SUCIO al medir"]}})
    assert declarada["estado"] == "no_anclable"
    assert declarada["anclable"] is False
    assert "ARBOL SUCIO" in declarada["explicacion"]


def test_un_JSON_con_bloque_anclable_se_declara_anclable():
    declarada = pasada.procedencia_declarada({"procedencia": {"anclable": True, "avisos": []}})
    assert declarada["estado"] == "anclable"
    assert declarada["anclable"] is True


@pytest.mark.parametrize("ruta,clave", _JSON_MEDIDOS, ids=lambda v: getattr(v, "name", v))
def test_los_tres_JSON_viejos_siguen_verificando_SU_PROPIO_digest(ruta: Path, clave: str):
    """La decisión, convertida en prueba: los tres ficheros NO se tocan.

    Se midieron el 2026-09-07 (la pasada y "sick") y el 09-09 ("adult"), y
    rellenarles hoy el commit y las versiones de hoy sería fabricar la
    procedencia que justamente les falta. Además su `digest_canonico` es lo
    ÚNICO que hoy los ata, y añadirles una clave —aunque fuera para marcarlos—
    lo rompería: comprobado el 09-12, los tres verifican. La marca va por
    fuera, en `procedencia_declarada()`, que es código y se puede ejecutar.

    Este test sigue valiendo si algún día se relanza la medición: un fichero
    reescrito por el script trae su procedencia y su digest nuevo, y también
    tiene que verificar."""
    if not ruta.exists():
        pytest.skip(f"{ruta.name} no está en este árbol")
    from matrixai.estudio.validacion import digest_canonico
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    guardado = payload.pop(clave)
    assert digest_canonico(payload) == guardado, f"{ruta.name} ya no verifica su propio digest"


@pytest.mark.parametrize("ruta,clave", _JSON_MEDIDOS, ids=lambda v: getattr(v, "name", v))
def test_un_JSON_ya_escrito_declara_procedencia_COMPLETA_o_NINGUNA(ruta: Path, clave: str):
    """Nunca un estado intermedio: media verdad tranquilizadora es peor que
    callar. O el fichero es de antes del 09-12 y se declara `sin_procedencia`,
    o trae el bloque entero con sus cuatro piezas."""
    if not ruta.exists():
        pytest.skip(f"{ruta.name} no está en este árbol")
    payload = json.loads(ruta.read_text(encoding="utf-8"))
    declarada = pasada.procedencia_declarada(payload)
    if declarada["estado"] == "sin_procedencia":
        assert "procedencia" not in payload
        return
    assert "procedencia" in payload, (
        f"{ruta.name}: se declara con procedencia y no trae el bloque")
    bloque = payload["procedencia"]
    for clave_exigida in ("repositorios", "versiones_de_bibliotecas", "digests_de_codigo",
                         "datos_de_entrada", "anclable", "avisos", "procedencia_id"):
        assert clave_exigida in bloque, f"{ruta.name}: procedencia a medias, falta {clave_exigida}"


# --- el caché: la medición del 0 de 720, y su otra mitad -------------------

def test_un_registro_SIN_los_digests_no_se_reusa():
    """El 0 de 720 en una línea: así estaban los 720 registros del 09-07."""
    assert pasada._reusable({"auroc": 0.9}, "ent", "mot") is False
    assert pasada._reusable({"entorno_digest": None, "motor_digest": None}, "ent", "mot") is False
    assert pasada._reusable(None, "ent", "mot") is False


def test_un_registro_CON_los_digests_iguales_SI_se_reusa():
    """La otra mitad: sin esto, la reparación la pasaría un caché que no reusa
    nunca -- correcto en el papel y 74,8 min de castigo en cada pasada."""
    previo = {"entorno_digest": "ent", "motor_digest": "mot"}
    assert pasada._reusable(previo, "ent", "mot") is True
    assert pasada._reusable(previo, "ent", "OTRO") is False
    assert pasada._reusable(previo, "OTRO", "mot") is False


def test_sobre_el_JSON_COMMITEADO_ninguna_reutilizacion_es_silenciosa(tmp_path):
    """La medición real, y en los dos mundos posibles.

    Con el fichero del 09-07 (sin procedencia): REUSABLES 0 DE 720, y por eso
    `main()` lo grita en vez de dejar un «0 reusados» que parece normal. Si
    algún día se relanza, el fichero traerá procedencia y entonces lo que se
    exige es que TODOS sus registros lleven sus digests."""
    ruta = _DIR_FASE0 / "pasada_exploratoria_101_c3_resultado.json"
    if not ruta.exists():
        pytest.skip("el JSON de la pasada no está en este árbol")
    registros, payload = pasada._cargar_cache(ruta)
    entorno = pasada._digest_entorno()
    por_motor = {n: pasada._digest_fichero(f) for n, f in pasada._FICHERO_POR_MOTOR.items()}
    reusables = sum(1 for clave, r in registros.items()
                    if pasada._reusable(r, entorno, por_motor[clave[1]]))
    if pasada.procedencia_declarada(payload)["estado"] == "sin_procedencia":
        assert reusables == 0, (
            "un fichero sin procedencia no puede tener registros reusables: "
            "sus digests no existen")
        assert len(registros) == 720
    else:
        sin_digest = [c for c, r in registros.items() if r.get("entorno_digest") is None]
        assert sin_digest == [], f"registros sin digest en un fichero con procedencia: {sin_digest[:3]}"


# --- las procedencias citadas: un reusado no hereda la de hoy --------------

def test_un_intento_REUSADO_sin_procedencia_no_hereda_la_de_hoy():
    """720 números escritos hoy de los que 700 se midieron hace cinco días no
    son 720 números de hoy. Declarar lo que PASÓ."""
    hoy = {"procedencia_id": "HOY", "anclable": True}
    resultados = [{"procedencia_id": "HOY"}, {"procedencia_id": None}, {"procedencia_id": None}]
    citadas, sin_ninguna = pasada._procedencias_citadas(resultados, hoy, {})
    assert sin_ninguna == 2
    assert set(citadas) == {"HOY"}


def test_una_procedencia_de_OTRA_pasada_viaja_con_los_intentos_que_midio():
    hoy = {"procedencia_id": "HOY", "anclable": True}
    previa = {"procedencia_id": "ANTES", "anclable": True}
    resultados = [{"procedencia_id": "HOY"}, {"procedencia_id": "ANTES"}]
    citadas, sin_ninguna = pasada._procedencias_citadas(resultados, hoy, {"ANTES": previa})
    assert sin_ninguna == 0
    assert citadas == {"HOY": hoy, "ANTES": previa}


def test_una_procedencia_CITADA_y_ausente_se_declara_en_vez_de_callarse():
    """El fichero previo cita un id que no trae: eso es un fichero roto, y
    callarlo dejaría intentos firmados por una procedencia que no existe."""
    hoy = {"procedencia_id": "HOY", "anclable": True}
    citadas, _ = pasada._procedencias_citadas([{"procedencia_id": "PERDIDA"}], hoy, {})
    assert citadas["PERDIDA"]["anclable"] is False
    assert citadas["PERDIDA"]["avisos"]


# --- el CABLEADO: que el bloque llegue al fichero, no que exista la función --

class _PliegueFalso:
    def __init__(self, entrena, valida):
        self.entrena, self.valida = entrena, valida


class _IntentoFalso:
    estado = "completed"
    motivo_del_estado = None
    informe = {"metrics": [{"metric_id": "auroc", "value": 0.9},
                           {"metric_id": "accuracy", "value": 0.8}]}


def _pasada_de_juguete(monkeypatch, salida: Path, motores_falsos) -> dict:
    """`main()` ENTERO, con los cuatro trozos caros sustituidos por dobles.

    No es «probar la función en vez del producto»: lo que se ejerce aquí es el
    camino real de `main()` -- sella la procedencia, la cuelga de cada registro,
    reusa del caché, reúne las procedencias citadas y escribe el JSON. Los
    dobles están solo donde el coste es el problema (cargar 12 ARFF, particionar
    y lanzar 720 subprocesos: 74,8 min medidos el 09-07), nunca donde vive lo
    que se quiere comprobar."""
    from types import SimpleNamespace

    filas = [{"row_id": f"r{i}", "y": "si" if i % 2 else "no"} for i in range(10)]
    por_id = {f["row_id"]: f for f in filas}
    plan = SimpleNamespace(observaciones_del_rol=lambda rol: ["r8", "r9"],
                           digest=lambda: "plan-falso")
    pliegues = SimpleNamespace(pliegue_de=lambda repeticion, pliegue: (
        _PliegueFalso(["r0", "r1", "r2", "r3"], ["r4", "r5"])
        if (repeticion, pliegue) == (0, 0) else None))
    propuesta = SimpleNamespace(plan=plan, pliegues=pliegues, es_viable=True)

    monkeypatch.setattr(pasada, "DATASETS", [(1063, "kc2", "pequeno", "si", "no")])
    monkeypatch.setattr(pasada, "particiones_base",
                        lambda *a, **k: (por_id, propuesta, object(), "y", ("x",)))
    monkeypatch.setattr(pasada, "preparar_para_motor", lambda tr, todas, *a, **k: list(todas))
    monkeypatch.setattr(pasada, "Particion",
                        SimpleNamespace(desde_filas=lambda *a, **k: object()))
    monkeypatch.setattr(pasada, "ejecutar_intento_aislado", lambda *a, **k: _IntentoFalso())
    monkeypatch.setattr(pasada, "MotorBaseline", lambda: motores_falsos[0])
    monkeypatch.setattr(pasada, "MotorLineal", lambda: motores_falsos[1])
    monkeypatch.setattr(pasada, "MotorArbolLightGBM", lambda: motores_falsos[2])
    monkeypatch.setattr(pasada, "MotorDensaPropia", lambda: motores_falsos[3])
    monkeypatch.setattr(sys, "argv", ["pasada", "--salida", str(salida)])
    pasada.main()
    return json.loads(salida.read_text(encoding="utf-8"))


@pytest.fixture
def motores_falsos():
    from types import SimpleNamespace
    return [SimpleNamespace(nombre=nombre) for nombre in pasada._FICHERO_POR_MOTOR]


def test_el_JSON_que_ESCRIBE_la_pasada_lleva_la_procedencia(tmp_path, monkeypatch, motores_falsos):
    """La mitad que faltaba de verdad: una `procedencia_de_la_medicion()`
    perfecta a la que `main()` no llame deja el fichero exactamente igual de
    inanclable que antes. El hueco está en el CABLEADO, no en el API."""
    salida = _pasada_de_juguete(monkeypatch, tmp_path / "salida.json", motores_falsos)
    assert "procedencia" in salida, "main() escribió el JSON sin bloque de procedencia"
    bloque = salida["procedencia"]
    assert set(bloque["repositorios"]) == {"matrixAI", "matrixai-engines"}
    assert bloque["versiones_de_bibliotecas"]["numpy"]
    assert bloque["digests_de_codigo"]["entorno"] == pasada._digest_entorno()
    assert bloque["datos_de_entrada"]["kc2"]["sha256"]
    assert salida["procedencias"][bloque["procedencia_id"]] == bloque
    assert {r["procedencia_id"] for r in salida["resultados"]} == {bloque["procedencia_id"]}
    assert salida["n_intentos_sin_procedencia"] == 0
    # UN registro por motor de la pasada, contados — no un 4 escrito a mano.
    # El 4 se quedó viejo el 2026-09-13, cuando la pasada pasó a correr los
    # SIETE motores del protocolo: una lista copiada es la que acaba
    # divergiendo, y un recuento copiado también.
    assert len(salida["resultados"]) == len(motores_falsos)


def test_el_JSON_escrito_sigue_verificando_su_digest(tmp_path, monkeypatch, motores_falsos):
    """Añadir la procedencia no puede romper lo único que ya ataba el fichero."""
    from matrixai.estudio.validacion import digest_canonico
    salida = _pasada_de_juguete(monkeypatch, tmp_path / "salida.json", motores_falsos)
    guardado = salida.pop("digest_resultados_crudos")
    assert digest_canonico(salida) == guardado
    assert "procedencia" in salida, "la procedencia queda FUERA del digest: se podría editar"


def test_lo_REUSADO_de_un_fichero_sin_procedencia_se_cuenta_aparte(
        tmp_path, monkeypatch, motores_falsos):
    """El caso real del 09-07 en pequeño: un fichero previo cuyos registros no
    traen procedencia. Los digests SÍ coinciden (se los ponemos), así que se
    reusan de verdad -- y aun así el fichero nuevo no los firma como medidos
    hoy."""
    ruta = tmp_path / "salida.json"
    entorno = pasada._digest_entorno()
    previos = [{"dataset": "kc2", "motor": motor.nombre, "repeticion": 0, "pliegue": 0,
                "estado": "completed", "auroc": 0.5,
                "entorno_digest": entorno,
                # El presupuesto de pared entra en la clave del caché desde el
                # 2026-09-13 (sale del protocolo registrado, no de una
                # constante de la pasada, así que cambiarlo no toca ningún
                # fichero de código y el `entorno_digest` no se enteraría).
                # Aquí se pone EL MISMO que la pasada va a aplicar al cubo de
                # `kc2`, porque lo que esta prueba quiere ejercer es el camino
                # en que SÍ se reusa.
                "presupuesto_wall_s": pasada.wall_seconds_del_cubo("pequeno"),
                "motor_digest": pasada._digest_fichero(pasada._FICHERO_POR_MOTOR[motor.nombre])}
               for motor in motores_falsos]
    ruta.write_text(json.dumps({"resultados": previos}), encoding="utf-8")
    salida = _pasada_de_juguete(monkeypatch, ruta, motores_falsos)
    assert salida["n_reusados"] == len(motores_falsos)
    assert salida["n_intentos_sin_procedencia"] == len(motores_falsos)
    assert all(r["procedencia_id"] is None for r in salida["resultados"])
    assert all(r["auroc"] == 0.5 for r in salida["resultados"]), "no se reusó el número viejo"


def test_lo_MEDIDO_hoy_si_firma_con_la_procedencia_de_hoy(
        tmp_path, monkeypatch, motores_falsos):
    """La otra mitad: sin esto, la reparación la pasaría un `main()` que pone
    `procedencia_id=None` en todos los registros y nunca firma nada."""
    ruta = tmp_path / "salida.json"
    previos = [{"dataset": "kc2", "motor": motor.nombre, "repeticion": 0, "pliegue": 0,
                "estado": "completed", "auroc": 0.5,
                "entorno_digest": "DE OTRO CODIGO", "motor_digest": "DE OTRO CODIGO"}
               for motor in motores_falsos]
    ruta.write_text(json.dumps({"resultados": previos}), encoding="utf-8")
    salida = _pasada_de_juguete(monkeypatch, ruta, motores_falsos)
    assert salida["n_reusados"] == 0
    assert salida["n_intentos_sin_procedencia"] == 0
    assert all(r["procedencia_id"] == salida["procedencia"]["procedencia_id"]
              for r in salida["resultados"])


def test_el_informe_de_101_C4_tambien_cablea_la_procedencia_a_SU_salida():
    """101-C4 no se puede ejercer con dobles como la pasada: su `main()` ajusta
    de verdad los cuatro motores sobre `adult` (48.842 filas; solo particionar
    tardó 270,4 s medidos el 2026-09-12), y meterlo en la suite costaría más que
    lo que protege. Se comprueba estáticamente lo único que puede fallar en
    silencio -- que la llamada existe y que la clave llega al diccionario que
    se escribe -- y el límite queda declarado aquí: esto NO ejecuta el informe."""
    import ast
    fuente = (_DIR_FASE0 / "informe_101_c4.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    principal = next(n for n in arbol.body
                     if isinstance(n, ast.FunctionDef) and n.name == "main")
    llamadas = {n.func.id for n in ast.walk(principal)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "procedencia_de_la_medicion" in llamadas, (
        "informe_101_c4.main() no sella procedencia")
    escrituras = [n for n in ast.walk(principal)
                  if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                  and any(isinstance(t, ast.Name) and t.id == "salida" for t in n.targets)]
    assert escrituras, "no se encontró el diccionario `salida` de informe_101_c4"
    claves = {k.value for k in escrituras[0].value.keys if isinstance(k, ast.Constant)}
    assert "procedencia" in claves, "el JSON de 101-C4 se escribe sin procedencia"


# ---------------------------------------------------------------------------
# EL `False` CON MOTIVO — 2026-09-16
#
# `anclable` era un si/no que no distingue «sucio con codigo que corre» de
# «sucio con un JSON de resultados ahi puesto». De las cinco mediciones de Fase
# 0, TRES dicen `anclable: false` por algo que no puede mover un solo numero, y
# UNA lo dice por el propio guion de la pasada, que los mueve todos — y las
# cuatro se leen igual. Un guardia que dice que no casi siempre deja de leerse:
# la misma familia que la nota «este fichero es sensible a la carga», que tapo
# un 500 de producto dos dias.
#
# Estas pruebas fijan las dos mitades: que AFILA (dice cual esta en el grafo) y
# que NO ABLANDA (nadie pasa a `anclable: true` por lo que diga este bloque).
# ---------------------------------------------------------------------------

def test_la_suciedad_dice_QUE_fichero_y_de_que_repositorio(tmp_path, monkeypatch):
    raiz = tmp_path / "sucio"
    _repo_de_juguete(raiz)
    (raiz / "motor.py").write_text("# sin commitear\n", encoding="utf-8")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"uno": raiz})
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})
    ficheros = bloque["suciedad"]["ficheros"]
    assert [(f["repositorio"], f["ruta"], f["tipo"]) for f in ficheros] == [
        ("uno", "motor.py", "modificado")]
    assert bloque["suciedad"]["n_sucios"] == 1


def test_un_fichero_sucio_QUE_ESTE_IMPORTADO_se_declara_en_el_grafo(tmp_path, monkeypatch):
    """La mitad que AFILA, y se mide contra `sys.modules`, no contra una lista.

    Una lista de rutas escrita a mano seria un segundo sitio declarando lo que
    `CLAUDE.md` ya declara — y dos sitios declarando lo mismo acaban
    divergiendo. Aqui el propio guion de la pasada se ensucia a proposito: esta
    importado por este proceso (lo importa este fichero de pruebas), asi que
    tiene que salir demostrado.
    """
    raiz = Path(pasada.__file__).resolve().parents[2]
    relativa = str(Path(pasada.__file__).resolve().relative_to(raiz))
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"core": raiz})
    monkeypatch.setattr(pasada, "_estado_del_repositorio", lambda _r: {
        "commit": "0" * 40, "arbol_sucio": True,
        "ficheros_modificados": [relativa], "ficheros_sin_seguimiento": [],
        "motivo": None})
    s = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})["suciedad"]
    assert s["n_demostrablemente_en_el_grafo"] == 1
    assert s["rutas_demostrablemente_en_el_grafo"] == [relativa]
    assert s["ficheros"][0]["importado_al_sellar"] is True


def test_un_fichero_sucio_que_NADIE_importa_no_sale_en_el_grafo(tmp_path, monkeypatch):
    raiz = tmp_path / "sucio"
    _repo_de_juguete(raiz)
    (raiz / "resultados.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"uno": raiz})
    s = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})["suciedad"]
    assert s["n_sucios"] == 1
    assert s["n_demostrablemente_en_el_grafo"] == 0
    assert s["ficheros"][0]["importado_al_sellar"] is False


def test_el_bloque_AFILA_pero_NO_ABLANDA_el_anclaje(tmp_path, monkeypatch):
    """La mitad que impide que esto se convierta en una coartada.

    Un fichero sucio que nadie importa sigue dejando la medicion NO anclable.
    Sin este aserto, el paso siguiente «obvio» —«si no esta en el grafo, que
    cuente como limpio»— convertiria una COTA INFERIOR en un alta: los intentos
    corren en hijos de `multiprocessing.spawn` que re-importan desde disco y
    pueden cargar mas modulos que el proceso que sella.
    """
    raiz = tmp_path / "sucio"
    _repo_de_juguete(raiz)
    (raiz / "resultados.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"uno": raiz})
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})
    assert bloque["anclable"] is False
    assert bloque["avisos"], "un arbol sucio sigue teniendo su aviso"
    assert bloque["suciedad"]["n_demostrablemente_en_el_grafo"] == 0


def test_el_campo_se_llama_importado_al_sellar_y_NO_esta_fuera_del_grafo():
    """El nombre es la mitad de la honestidad de este bloque.

    `importado_al_sellar: false` significa «no se ha podido demostrar que
    corra», no «esta fuera». Un campo llamado `esta_fuera_del_grafo` afirmaria
    lo segundo, que es justo lo que esta medicion NO puede sostener. Y el propio
    bloque lleva escrito por que.
    """
    bloque = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})
    s = bloque["suciedad"]
    assert "por_que_un_False_aqui_NO_absuelve" in s
    assert "COTA INFERIOR" in s["por_que_un_False_aqui_NO_absuelve"]
    for f in s["ficheros"]:
        assert "importado_al_sellar" in f
        assert "esta_fuera_del_grafo" not in f


def test_la_suciedad_entra_en_el_procedencia_id(tmp_path, monkeypatch):
    """Dos pasadas del mismo commit sucias de formas DISTINTAS no son la misma
    procedencia, y no pueden compartir identificador."""
    raiz = tmp_path / "sucio"
    _repo_de_juguete(raiz)
    monkeypatch.setattr(pasada, "_RUTAS_DE_REPOSITORIO", {"uno": raiz})
    (raiz / "a.py").write_text("# a\n", encoding="utf-8")
    uno = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})["procedencia_id"]
    (raiz / "a.py").unlink()
    (raiz / "b.py").write_text("# b\n", encoding="utf-8")
    otro = pasada.procedencia_de_la_medicion(
        digests_de_codigo={}, datos_de_entrada={})["procedencia_id"]
    assert uno != otro
