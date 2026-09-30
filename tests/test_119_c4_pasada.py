# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""119-C4 — el ARNÉS de la CONFIRMACIÓN (`pasada_119_c4.py`), NO el motor en
sí (sus pruebas viven en `matrixai-engines`) NI el arnés de C3 (ya probado en
`test_119_c3_pasada.py`, cuyos ayudantes — motor falso, `_registro`, `_serie`,
la v2 real — se REUTILIZAN aquí, importados, nunca redefinidos).

Nada aquí entrena nada de verdad: `main()` corre con un MOTOR FALSO
(`ejecutar_intento_aislado` sustituido), y sobre sellados REALES solo se
tocan los DOS más baratos (pc3, banknote-authentication) para probar el
camino entero sin coste. Ningún test aquí llama al motor de verdad sobre un
sellado: donde una prueba comprueba que `--solo` se niega con el motor
«real», el real es un CENTINELA que revienta si se le llama.

Lo que se prueba, por encargo del supervisor (30-09) y de la auditoría del
guion (30-09: B1, I1-I6, M1-M9):

* los 8 sellados se niegan cuando se pide un NO sellado (al revés que C3), y
  la PASADA mide SOLO los sellados (`main()` en modo pasada, con un no
  sellado barato en la lista);
* «UNA SOLA EJECUCIÓN»: se niega sobre un resultado COMPLETO en la `--salida`
  O en CUALQUIERA de las rutas fijas (la de la cola y las dos del árbol), y
  reanuda un PARCIAL reusando lo medido (fallos incluidos, nunca
  reintentados) o PARANDO si algo de lo medido no es reusable;
* `--solo` con el motor real PARA antes de cargar nada;
* el anclaje (sellos de C3, EL C3 medido por digest y por blob, motores y su
  digest, árbol limpio, diff del núcleo y entorno de C3) para AL ARRANCAR,
  antes del primer intento — y admite la ejecución legítima (el commit de
  C4 con el resultado de C3 dentro: el B1 de la auditoría, sobre la historia
  REAL de este repo);
* el X/40 se compone bien (31/32 + y), con casos fabricados donde el
  resultado se sabe a mano y con `main()` de punta a punta en modo pasada;
* los fallos en los sellados cuentan como en la enmienda 2 (fallido,
  AUSENTE o sin métrica pierden el conjunto) y el X/40 lo declara;
* el digest de la caché de C4 cubre su propio guion;
* `--estimar` nunca llama al motor real, no presenta ausencias como ceros, y
  AVISA si la orden de encolado no serviría (HEAD sin el guion, árbol sucio).
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

import pasada_119_c3 as p119  # noqa: E402
import pasada_119_c4 as p4  # noqa: E402
import pasada_114c6_ensamblado as c6  # noqa: E402
import pasada_amplia_101_c5 as c5  # noqa: E402
import protocolo as protocolo_mod  # noqa: E402

# Ayudantes REUTILIZADOS de C3 -- nunca redefinidos aquí.
from test_119_c3_pasada import (  # noqa: E402
    _ESPERADAS_3, _motor_falso, _serie, _v2, _valores_v2, PROTOCOLO_V2,
)

_FASE0 = _RAIZ / "benchmarks" / "fase0"
NUEVO, DENSA = p119.NOMBRE_MOTOR_NUEVO, p119.NOMBRE_DENSA_V2
REGLA = PROTOCOLO_V2.regla_de_cierre


@pytest.fixture(scope="module")
def _datasets():
    c6.preparar_protocolo_v2()
    todos = c5.datasets_de_la_pasada(PROTOCOLO_V2)
    sellados = [d for d in todos if d.sellado]
    return todos, sellados


def _md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. LOS 8 SELLADOS -- un NO sellado se niega (al revés que C3)
# ---------------------------------------------------------------------------

def test_sin_solo_son_los_8_sellados(_datasets):
    todos, sellados = _datasets
    datasets, subconjunto = p4.datasets_de_c4(todos, sellados, solo=None)
    assert len(datasets) == 8
    assert all(d.sellado for d in datasets)
    assert subconjunto is None
    assert {d.nombre for d in datasets} == {
        "kr-vs-kp", "letter", "splice", "pol", "pc3", "banknote-authentication", "Satellite",
        "diamonds"}


def test_solo_un_sellado_se_acepta(_datasets):
    todos, sellados = _datasets
    datasets, subconjunto = p4.datasets_de_c4(todos, sellados, solo="pc3")
    assert [d.nombre for d in datasets] == ["pc3"]
    assert subconjunto == ["pc3"]


def test_solo_un_no_sellado_se_niega(_datasets):
    """Al revés que C3 (`datasets_de_c3`, que niega un sellado): aquí lo que
    se niega es un conjunto que NO está sellado, porque ese es de C3."""
    todos, sellados = _datasets
    with pytest.raises(SystemExit, match="NO SELLADO.*diabetes"):
        p4.datasets_de_c4(todos, sellados, solo="diabetes")


def test_solo_mezcla_sellado_y_no_sellado_tambien_se_niega(_datasets):
    todos, sellados = _datasets
    with pytest.raises(SystemExit, match="NO SELLADO"):
        p4.datasets_de_c4(todos, sellados, solo="pc3,diabetes")


def test_solo_conjunto_desconocido_para(_datasets):
    todos, sellados = _datasets
    with pytest.raises(SystemExit, match="no están en el protocolo"):
        p4.datasets_de_c4(todos, sellados, solo="no-existe-este-conjunto")


def test_cli_se_niega_a_medir_un_no_sellado(tmp_path, monkeypatch):
    """La misma negativa, pero por la línea de órdenes entera (`main()`):
    un `--solo` con un no sellado para ANTES de tocar el motor falso."""
    salida = tmp_path / "solo.json"
    llamadas: list = []
    with pytest.raises(SystemExit, match="NO SELLADO"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(p4, "ejecutar_intento_aislado", _motor_falso(lambda *a: 0.9, llamadas=llamadas))
            p4.main(["--solo", "diabetes", "--salida", str(salida)])
    assert llamadas == []  # el motor falso no se llegó a llamar
    assert not salida.exists()


# ---------------------------------------------------------------------------
# 1b. `--solo` CON EL MOTOR REAL PARA (auditoría del guion, I2)
# ---------------------------------------------------------------------------

def test_el_ejecutor_real_guardado_es_el_de_subproceso():
    """La comparación de `main()` es por IDENTIDAD con lo guardado al
    importar: si lo guardado no fuera el real, `--solo` correría con él."""
    import matrixai_engines.subproceso as subproceso  # noqa: PLC0415

    assert p4._EJECUTAR_INTENTO_AISLADO_REAL is subproceso.ejecutar_intento_aislado


def test_solo_con_el_motor_real_para_antes_de_cargar_nada(tmp_path, monkeypatch):
    """`--solo pc3` con el ejecutor «real»: PARA antes de cargar nada. El
    «real» es aquí un CENTINELA (el mismo objeto en los dos sitios que
    `main()` compara), así que si la guardia faltara la prueba reventaría con
    un AssertionError y NUNCA llamaría al motor de verdad."""
    llamadas: list = []

    def centinela(*a, **kw):
        llamadas.append(kw.get("dataset"))
        raise AssertionError("--solo llegó al ejecutor «real»: la guardia de I2 no está")

    monkeypatch.setattr(p4, "ejecutar_intento_aislado", centinela)
    monkeypatch.setattr(p4, "_EJECUTAR_INTENTO_AISLADO_REAL", centinela)
    salida = tmp_path / "solo.json"
    with pytest.raises(SystemExit, match="--solo NO corre con el motor real"):
        p4.main(["--solo", "pc3", "--salida", str(salida)])
    assert llamadas == []
    assert not salida.exists()


def test_el_docstring_ya_no_propone_solo_sobre_un_sellado():
    texto = " ".join(p4.__doc__.split())
    assert "--solo kr-vs-kp" not in texto
    assert "`--solo` NO corre con el motor real" in texto


# ---------------------------------------------------------------------------
# 2. «UNA SOLA EJECUCIÓN»: se niega sobre un completo EN CUALQUIER RUTA FIJA
# ---------------------------------------------------------------------------

def _payload_pasada(*, parcial) -> dict:
    return {"corte": "119-C4", "tipo_de_ejecucion": "pasada", "parcial": parcial, "resultados": []}


@pytest.fixture
def _rutas_fijas(tmp_path, monkeypatch):
    """Las rutas fijas de «una sola ejecución», en tmp_path: la prueba no
    puede depender de si la confirmación real ya existe en esta máquina."""
    rutas = (tmp_path / "cola" / "pasada_119_c4_resultado.json",
             tmp_path / "arbol" / "pasada_119_c4_resultado.json",
             tmp_path / "arbol" / "resultado_pasada_119_c4.json")
    for r in rutas:
        r.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(p4, "RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION", rutas)
    return rutas


def test_las_rutas_fijas_son_la_cola_y_las_dos_del_arbol():
    assert p4.SALIDA_DE_C4_EN_LA_COLA == Path(
        "/home/deployer/cola-nocturna/resultados/119-c4/pasada_119_c4_resultado.json")
    assert set(p4.RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION) == {
        p4.SALIDA_DE_C4_EN_LA_COLA, _FASE0 / "pasada_119_c4_resultado.json",
        _FASE0 / "resultado_pasada_119_c4.json"}


def test_una_sola_ejecucion_no_hace_nada_si_no_hay_ningun_fichero(tmp_path, _rutas_fijas):
    p4.exigir_una_sola_ejecucion(tmp_path / "no-existe.json")  # no revienta


def test_una_sola_ejecucion_se_niega_sobre_un_resultado_completo(tmp_path, _rutas_fijas):
    ruta = tmp_path / "c4.json"
    ruta.write_text(json.dumps(_payload_pasada(parcial=False)), encoding="utf-8")
    with pytest.raises(SystemExit, match="UNA vez, en los SELLADOS"):
        p4.exigir_una_sola_ejecucion(ruta)


@pytest.mark.parametrize("cual", [0, 1, 2])
def test_una_sola_ejecucion_mira_las_rutas_fijas_aunque_la_salida_sea_otra(tmp_path, _rutas_fijas,
                                                                          cual):
    """I3: con el completo en la ruta de la cola (o en cualquiera de las dos
    del árbol), OTRA `--salida` también para."""
    _rutas_fijas[cual].write_text(json.dumps(_payload_pasada(parcial=False)), encoding="utf-8")
    with pytest.raises(SystemExit, match=str(_rutas_fijas[cual].name)):
        p4.exigir_una_sola_ejecucion(tmp_path / "otra-salida.json")


def test_una_sola_ejecucion_falla_cerrado_si_parcial_no_es_true(tmp_path, _rutas_fijas):
    ruta = tmp_path / "c4.json"
    ruta.write_text(json.dumps(_payload_pasada(parcial=None)), encoding="utf-8")
    with pytest.raises(SystemExit, match="COMPLETO"):
        p4.exigir_una_sola_ejecucion(ruta)


def test_una_sola_ejecucion_reanuda_un_parcial(tmp_path, _rutas_fijas):
    ruta = tmp_path / "c4.json"
    ruta.write_text(json.dumps(_payload_pasada(parcial=True)), encoding="utf-8")
    _rutas_fijas[0].write_text(json.dumps(_payload_pasada(parcial=True)), encoding="utf-8")
    p4.exigir_una_sola_ejecucion(ruta)  # no revienta: es la misma ejecución, cortada


def test_una_solo_de_prueba_en_una_ruta_fija_no_cuenta_como_la_confirmacion(tmp_path,
                                                                             _rutas_fijas):
    payload = dict(_payload_pasada(parcial=False), tipo_de_ejecucion="solo")
    _rutas_fijas[0].write_text(json.dumps(payload), encoding="utf-8")
    p4.exigir_una_sola_ejecucion(tmp_path / "c4.json")  # un --solo no es la confirmación


def test_ruta_de_salida_de_c3_respeta_el_resultado_real_que_se_le_pasa(tmp_path):
    """La parametrización de C3 que C4 usa: un tipo que no es la pasada NO
    escribe en el `ruta_resultado_real` que se le pasa (el de C4), aunque no
    sea el de C3 (auditoría del guion: S21)."""
    real = tmp_path / "real_de_otro.json"
    omision = {"pasada": real, "solo": tmp_path / "s.json", "estimar": tmp_path / "e.json"}
    with pytest.raises(SystemExit, match="no escribe en el resultado real"):
        p119.ruta_de_salida("solo", str(real), salida_por_omision=omision,
                            ruta_resultado_real=real)
    assert p119.ruta_de_salida("pasada", None, salida_por_omision=omision,
                               ruta_resultado_real=real) == real.resolve()


@pytest.mark.parametrize("tipo", ["solo", "estimar"])
def test_ruta_de_salida_c4_no_deja_escribir_una_prueba_en_ninguna_ruta_real(tipo):
    for ruta in p4.RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION:
        with pytest.raises(SystemExit, match="resultado real"):
            p4.ruta_de_salida_c4(tipo, str(ruta))
    assert p4.ruta_de_salida_c4("pasada", str(p4.SALIDA_DE_C4_EN_LA_COLA)) == \
        p4.SALIDA_DE_C4_EN_LA_COLA.resolve()


# ---------------------------------------------------------------------------
# 2b. LO YA MEDIDO SE REUSA (fallos incluidos) O LA PASADA PARA (I4)
# ---------------------------------------------------------------------------

def _previo(**cambios) -> dict:
    return {"dataset": "pc3", "repeticion": 0, "pliegue": 0, "estado": "completed",
            "entorno_digest": "E", "motor_digest": "M", "presupuesto_wall_s": 120.0,
            "datos_sha256": "D", **cambios}


def _decidir(previo):
    return p4.decidir_con_el_previo(previo, entorno_digest="E", motor_digest="M",
                                    wall_seconds=120.0, datos_sha256="D")


def test_sin_previo_se_mide():
    assert _decidir(None) == "medir"


@pytest.mark.parametrize("estado", ["completed", "failed", "cancelled"])
def test_un_previo_del_mismo_codigo_se_reusa_tambien_si_fallo(estado):
    """En C4 un fallido ES la medida: se reusa, NUNCA se reintenta (C3 sí
    reintenta: `_reusable_c3` no reusa un `failed`)."""
    assert _decidir(_previo(estado=estado)) == "reusar"
    if estado != "completed":
        assert p119._reusable_c3(_previo(estado=estado), "E", "M", 120.0,
                                 datos_sha256="D") is False  # la regla de C3, distinta


@pytest.mark.parametrize("cambio", [{"entorno_digest": "OTRO"}, {"motor_digest": "OTRO"},
                                    {"presupuesto_wall_s": 300.0}, {"datos_sha256": "OTROS"}])
@pytest.mark.parametrize("estado", ["completed", "failed"])
def test_un_previo_que_no_es_reusable_para_en_vez_de_re_medir(cambio, estado):
    with pytest.raises(SystemExit, match="UNA vez"):
        _decidir(_previo(estado=estado, **cambio))


def test_al_arrancar_para_si_algun_previo_es_de_otro_codigo():
    previos = [_previo(), _previo(pliegue=1, entorno_digest="OTRO")]
    kwargs = dict(nombres={"pc3"}, entorno_digest="E", motor_digest="M",
                  datos_sha256_por_conjunto={"pc3": "D"})
    with pytest.raises(SystemExit, match="1 de 2 registros previos"):
        p4.exigir_que_lo_previo_sea_reusable(previos, **kwargs)
    assert p4.exigir_que_lo_previo_sea_reusable(previos[:1], **kwargs)["todos_reusables"] is True


# ---------------------------------------------------------------------------
# 3. VERIFICAR LOS SELLOS DE C3, Y EL ANCLAJE AL MISMO MOTOR
# ---------------------------------------------------------------------------

#: Los componentes del digest del entorno de un C3 FABRICADO (el refuerzo sin
#: git compara con esto lo que se le pase como «de hoy»).
_COMPONENTES_FABRICADOS = {"core:benchmarks/fase0/pasada_119_c3.py": "c3-v0",
                           "core:matrixai/estudio/metricas.py": "aaaa",
                           "version:torch": "2.x"}


def _payload_c3_fabricado(*, resultados=None, commit_core="c0ffee", commit_engines="beefea",
                          digest_motor="deadbeef", parcial=False, sucio_core=False,
                          componentes=None, tipo="pasada", subconjunto=False,
                          cumplidos=(31, 32), densa=(11, 32)) -> dict:
    """Sella UNA sola vez (`sellar_la_salida` no es idempotente: llamarla dos
    veces sobre el MISMO dict dejaría un `digest_resultados_crudos` que
    incluye el `digest_resultados_crudos` viejo como campo más, y
    `verificar_los_sellos_de_c3` -- que sí excluye ese campo -- lo vería
    como manipulado. Por eso todo lo que varía es un parámetro de aquí, no
    una mutación posterior con un segundo sellado)."""
    payload = {
        "corte": "119-C3", "tipo_de_ejecucion": tipo, "parcial": parcial,
        "es_subconjunto_de_prueba": subconjunto,
        "resultados": resultados if resultados is not None else [
            {"dataset": "dsA", "motor": NUEVO, "auroc": 0.9}],
        "procedencia": {"repositorios": {
            "matrixAI": {"commit": commit_core, "arbol_sucio": sucio_core},
            "matrixai-engines": {"commit": commit_engines, "arbol_sucio": False}}},
        "digest_de_la_cache": {"motor_nuevo": {"digest": digest_motor},
                               "componentes": dict(componentes if componentes is not None
                                                   else _COMPONENTES_FABRICADOS)},
        "veredicto": {
            "cumplidos_con_el_motor_nuevo": {"cumplidos": cumplidos[0], "datasets": cumplidos[1]},
            "cumplidos_de_la_densa_v2_en_los_mismos_conjuntos": {"cumplidos": densa[0],
                                                                 "datasets": densa[1]}},
    }
    p119.c3.sellar_la_salida(payload)
    return payload


def _procedencia_c4(*, commit_core="c0ffee", commit_engines="beefea") -> dict:
    return {"repositorios": {"matrixAI": {"commit": commit_core, "arbol_sucio": False},
                             "matrixai-engines": {"commit": commit_engines, "arbol_sucio": False}}}


def _poner_c3(monkeypatch, tmp_path, payload, *, fijar_el_digest=True) -> Path:
    """Escribe el C3 fabricado y lo hace pasar por EL C3 medido (el digest
    fijado del guion es el del C3 real; aquí, el del fabricado)."""
    ruta = tmp_path / "resultado_c3.json"
    ruta.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(p4, "RUTA_DEL_RESULTADO_C3", ruta)
    if fijar_el_digest:
        monkeypatch.setattr(p4, "DIGEST_RESULTADOS_CRUDOS_DE_C3", payload["digest_resultados_crudos"])
    return ruta


def _anclar(procedencia, digest_motor="deadbeef", componentes=None):
    return p4.exigir_c3_sellado_y_anclado(
        procedencia, digest_motor,
        componentes_de_c3_ahora=dict(componentes if componentes is not None
                                     else _COMPONENTES_FABRICADOS))


def test_verificar_los_sellos_de_c3_cuadra_recien_sellado():
    payload = _payload_c3_fabricado()
    sellos = p4.verificar_los_sellos_de_c3(payload)
    assert sellos["cuadra"] is True
    assert sellos["problemas"] == []


def test_para_si_los_sellos_de_c3_no_cuadran():
    """Tocar CUALQUIER cosa después de sellar invalida `digest_resultados_
    crudos`; tocar `resultados` invalida TAMBIÉN `digest_solo_de_resultados`."""
    payload = _payload_c3_fabricado()
    payload["procedencia"]["repositorios"]["matrixAI"]["commit"] = "otro-commit-distinto"
    sellos = p4.verificar_los_sellos_de_c3(payload)
    assert sellos["cuadra"] is False
    assert any("digest_resultados_crudos" in p for p in sellos["problemas"])
    assert not any("digest_solo_de_resultados" in p for p in sellos["problemas"])

    payload2 = _payload_c3_fabricado()
    payload2["resultados"][0]["auroc"] = 0.1234  # cambia DENTRO de resultados
    sellos2 = p4.verificar_los_sellos_de_c3(payload2)
    assert sellos2["cuadra"] is False
    assert any("digest_solo_de_resultados" in p for p in sellos2["problemas"])
    assert any("digest_resultados_crudos" in p for p in sellos2["problemas"])


def test_exigir_c3_sellado_y_anclado_pasa_si_todo_cuadra(tmp_path, monkeypatch):
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado())
    anclaje = _anclar(_procedencia_c4())
    assert anclaje["sellos"]["cuadra"] is True
    assert anclaje["payload_c3"]["corte"] == "119-C3"
    assert anclaje["diferencia_del_entorno"]["admitida"] is True


def test_para_si_no_esta_el_resultado_de_c3(tmp_path, monkeypatch):
    monkeypatch.setattr(p4, "RUTA_DEL_RESULTADO_C3", tmp_path / "no-esta.json")
    with pytest.raises(SystemExit, match="no está"):
        _anclar(_procedencia_c4())


def test_para_si_c3_esta_parcial(tmp_path, monkeypatch):
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(parcial=True))
    with pytest.raises(SystemExit, match="a medias"):
        _anclar(_procedencia_c4())


@pytest.mark.parametrize("tipo, subconjunto", [("solo", False), ("pasada", True)])
def test_para_si_c3_no_es_la_pasada_real(tmp_path, monkeypatch, tipo, subconjunto):
    """M4: un C3 de `--solo` (o un subconjunto) no es la pasada que subió."""
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(tipo=tipo, subconjunto=subconjunto))
    with pytest.raises(SystemExit, match="no es la PASADA real de C3"):
        _anclar(_procedencia_c4())


def test_para_si_c3_esta_sellado_pero_no_es_el_c3_medido(tmp_path, monkeypatch):
    """B1: un C3 RE-SELLADO (sellos que cuadran, otro contenido) no se suma:
    el `digest_resultados_crudos` tiene que ser el fijado en el guion."""
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(), fijar_el_digest=False)
    assert p4.DIGEST_RESULTADOS_CRUDOS_DE_C3.startswith("640137070dc30a28")
    with pytest.raises(SystemExit, match="NO es el C3 medido"):
        _anclar(_procedencia_c4())


def test_para_si_un_commit_de_motores_distinto_sigue_parando(tmp_path, monkeypatch):
    """`matrixai-engines` se queda EXACTAMENTE como estaba (defecto de
    diseño del 30-09: solo matrixAI puede ser OTRO commit, nunca motores)."""
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(commit_engines="engines-viejo"))
    with pytest.raises(SystemExit, match="matrixai-engines.*engines-viejo"):
        _anclar(_procedencia_c4(commit_engines="engines-nuevo"))


def test_para_si_el_digest_del_motor_difiere(tmp_path, monkeypatch):
    """Comprobación EXTRA (además de los commits, que pueden no cambiar el
    código si el árbol estaba sucio): el digest del motor que C3 registró
    tiene que ser el mismo que calcula esta ejecución de C4."""
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(digest_motor="digest-de-c3"))
    with pytest.raises(SystemExit, match="digest del motor"):
        _anclar(_procedencia_c4(), digest_motor="digest-de-c4-distinto")


def test_para_si_el_arbol_de_matrixai_esta_sucio(tmp_path, monkeypatch):
    """Sin commit al que fiar el diff: PARA, aunque el commit declarado sea
    el mismo en las dos mediciones."""
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(sucio_core=True))
    with pytest.raises(SystemExit, match="SUCIO"):
        _anclar(_procedencia_c4())


# --- 3a. el refuerzo SIN git: el entorno de C3, componente a componente ------

def test_el_entorno_puede_diferir_solo_en_la_parametrizacion_de_c3(tmp_path, monkeypatch):
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado())
    hoy = dict(_COMPONENTES_FABRICADOS, **{"core:benchmarks/fase0/pasada_119_c3.py": "c3-v1"})
    anclaje = _anclar(_procedencia_c4(), componentes=hoy)
    assert anclaje["diferencia_del_entorno"]["distintos"] == ["core:benchmarks/fase0/pasada_119_c3.py"]


@pytest.mark.parametrize("hoy", [
    dict(_COMPONENTES_FABRICADOS, **{"core:matrixai/estudio/metricas.py": "OTRO"}),
    dict(_COMPONENTES_FABRICADOS, **{"version:torch": "3.0"}),
    {k: v for k, v in _COMPONENTES_FABRICADOS.items() if k != "version:torch"},
    dict(_COMPONENTES_FABRICADOS, **{"core:matrixai/nuevo.py": "x"}),
], ids=["un_modulo", "una_version", "falta_uno", "sobra_uno"])
def test_para_si_el_entorno_de_c3_cambio(tmp_path, monkeypatch, hoy):
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado())
    with pytest.raises(SystemExit, match="entorno: 1 componentes"):
        _anclar(_procedencia_c4(), componentes=hoy)


def test_para_si_c3_no_registra_sus_componentes(tmp_path, monkeypatch):
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(componentes={}))
    with pytest.raises(SystemExit, match="no registra `digest_de_la_cache.componentes`"):
        _anclar(_procedencia_c4())


# ---------------------------------------------------------------------------
# 3b. matrixAI PUEDE ser otro commit -- si el diff cabe en lo admitido
#     (defecto de diseño señalado por el supervisor, 30-09: la cola fija el
#     HEAD COMMITEADO, así que exigir el MISMO commit de matrixAI que C3
#     bloquearía SIEMPRE el único camino legítimo). Con un repo git
#     FABRICADO en tmp_path -- nunca el árbol real -- salvo donde se dice.
# ---------------------------------------------------------------------------

def _repo_git_temporal(tmp_path, nombre="nucleo-fabricado") -> Path:
    raiz = tmp_path / nombre
    raiz.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=raiz, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=raiz, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=raiz, check=True)
    return raiz


def _commit(raiz: Path, ficheros: dict, mensaje: str) -> str:
    for ruta, contenido in ficheros.items():
        p = raiz / ruta
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contenido, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=raiz, check=True)
    subprocess.run(["git", "commit", "-q", "-m", mensaje], cwd=raiz, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True,
                          check=True).stdout.strip()


def _blob(raiz: Path, commit: str, ruta: str) -> str:
    return subprocess.run(["git", "rev-parse", f"{commit}:{ruta}"], cwd=raiz, capture_output=True,
                          text=True, check=True).stdout.strip()


def test_diferencia_de_nucleo_un_commit_que_solo_cambia_lo_admitido_pasa(tmp_path):
    raiz = _repo_git_temporal(tmp_path)
    c0 = _commit(raiz, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "c0")
    c1 = _commit(raiz, {"benchmarks/fase0/pasada_119_c4.py": "nuevo",
                        "benchmarks/fase0/pasada_119_c3.py": "v1 (parametrizado)",
                        "tests/test_119_c4_pasada.py": "un test",
                        "documentacion/119_LA_RED_DENSA_NUEVA_CONTRACT.md": "estado nuevo"}, "c1")
    dif = p4.diferencia_de_nucleo_entre_c3_y_c4(raiz, c0, c1)
    assert dif["admitida"] is True
    assert dif["ficheros_no_admitidos"] == []
    assert set(dif["ficheros_cambiados"]) == {
        "benchmarks/fase0/pasada_119_c4.py", "benchmarks/fase0/pasada_119_c3.py",
        "tests/test_119_c4_pasada.py", "documentacion/119_LA_RED_DENSA_NUEVA_CONTRACT.md"}


def test_diferencia_de_nucleo_un_cambio_en_matrixai_para(tmp_path):
    raiz = _repo_git_temporal(tmp_path)
    c0 = _commit(raiz, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "c0")
    c1 = _commit(raiz, {"matrixai/estudio/metricas.py": "un cambio en el núcleo"}, "c1")
    dif = p4.diferencia_de_nucleo_entre_c3_y_c4(raiz, c0, c1)
    assert dif["admitida"] is False
    assert dif["ficheros_no_admitidos"] == ["matrixai/estudio/metricas.py"]


def test_diferencia_de_nucleo_un_cambio_en_el_json_del_protocolo_para(tmp_path):
    raiz = _repo_git_temporal(tmp_path)
    c0 = _commit(raiz, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "c0")
    c1 = _commit(raiz, {"benchmarks/fase0/protocolo_119_v4.json": "{}"}, "c1")
    dif = p4.diferencia_de_nucleo_entre_c3_y_c4(raiz, c0, c1)
    assert dif["admitida"] is False
    assert dif["ficheros_no_admitidos"] == ["benchmarks/fase0/protocolo_119_v4.json"]


def test_diferencia_de_nucleo_admite_el_resultado_de_c3_solo_con_su_blob(tmp_path, monkeypatch):
    """B1: el resultado de C3 (que el commit de C3 no podía tener) se admite
    SOLO si su blob en el commit de C4 es el sellado; con otro contenido,
    para. Y el prefijo `benchmarks/fase0/` sigue cerrado (prueba de arriba)."""
    raiz = _repo_git_temporal(tmp_path)
    ruta = p4.RUTA_RELATIVA_DEL_RESULTADO_C3
    c0 = _commit(raiz, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "c0")
    c1 = _commit(raiz, {ruta: '{"el": "sellado"}', "benchmarks/fase0/pasada_119_c4.py": "c4"}, "c1")
    monkeypatch.setattr(p4, "BLOB_DEL_RESULTADO_C3_SELLADO", _blob(raiz, c1, ruta))
    dif = p4.diferencia_de_nucleo_entre_c3_y_c4(raiz, c0, c1)
    assert dif["admitida"] is True, dif
    assert dif["resultado_de_c3_en_el_commit_de_c4"]["cuadra"] is True
    c2 = _commit(raiz, {ruta: '{"el": "RE-SELLADO"}'}, "c2")
    dif2 = p4.diferencia_de_nucleo_entre_c3_y_c4(raiz, c0, c2)
    assert dif2["admitida"] is False
    assert dif2["ficheros_no_admitidos"] == [ruta]
    assert dif2["resultado_de_c3_en_el_commit_de_c4"]["cuadra"] is False


def test_b1_sobre_la_historia_real_el_commit_de_c3_al_de_hoy_se_admite():
    """B1, sobre ESTE repo (no fabricado): entre el commit que midió C3
    (`f54caf3`) y el que añadió su resultado (`e078fe6`) cambiaron el JSON de
    C3 y una prueba. Antes de la reparación salía `admitida: False` con
    `[resultado_pasada_119_c3.json]` (medido por la auditoría): paraba la
    ejecución legítima SIEMPRE. Y las dos constantes fijadas son las de
    verdad: el blob en `e078fe6` y el digest del fichero del árbol."""
    payload = p119._leer_json(_FASE0 / "resultado_pasada_119_c3.json")
    commit_c3 = payload["procedencia"]["repositorios"]["matrixAI"]["commit"]
    assert commit_c3.startswith("f54caf3")
    dif = p4.diferencia_de_nucleo_entre_c3_y_c4(_RAIZ, commit_c3, "e078fe6")
    assert dif["ficheros_cambiados"] == [p4.RUTA_RELATIVA_DEL_RESULTADO_C3,
                                         "tests/test_119_c3_pasada.py"]
    assert dif["admitida"] is True, dif["ficheros_no_admitidos"]
    assert _blob(_RAIZ, "e078fe6", p4.RUTA_RELATIVA_DEL_RESULTADO_C3) == \
        p4.BLOB_DEL_RESULTADO_C3_SELLADO
    assert payload["digest_resultados_crudos"] == p4.DIGEST_RESULTADOS_CRUDOS_DE_C3


def test_diferencia_de_nucleo_mismo_commit_no_llama_a_git():
    dif = p4.diferencia_de_nucleo_entre_c3_y_c4(Path("/no/existe/ni/hace/falta"), "abc", "abc")
    assert dif["admitida"] is True
    assert dif["ficheros_cambiados"] == []


def test_diferencia_de_nucleo_para_si_el_commit_de_c3_no_existe(tmp_path):
    raiz = _repo_git_temporal(tmp_path)
    c1 = _commit(raiz, {"a.txt": "x"}, "c1")
    with pytest.raises(SystemExit, match="no existe"):
        p4.diferencia_de_nucleo_entre_c3_y_c4(raiz, "0" * 40, c1)


def test_exigir_c3_sellado_y_anclado_admite_otro_commit_de_nucleo(tmp_path, monkeypatch):
    """Integración: `exigir_c3_sellado_y_anclado` con un commit de matrixAI
    DISTINTO en C3 y en C4, pero el diff entre los dos solo toca el guion de
    C4 -- el camino legítimo que encolar.sh usa de verdad. El diff se mide en
    el repo del que la procedencia lee el commit (`_RUTAS_DE_REPOSITORIO`)."""
    raiz = _repo_git_temporal(tmp_path)
    c0 = _commit(raiz, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "c0")
    c1 = _commit(raiz, {"benchmarks/fase0/pasada_119_c4.py": "nuevo"}, "c1")
    monkeypatch.setitem(p119.c3._RUTAS_DE_REPOSITORIO, "matrixAI", raiz)
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(commit_core=c0))
    anclaje = _anclar(_procedencia_c4(commit_core=c1))
    assert anclaje["diferencia_de_nucleo"]["admitida"] is True
    assert anclaje["diferencia_de_nucleo"]["commit_c3"] == c0
    assert anclaje["diferencia_de_nucleo"]["commit_c4"] == c1


def test_exigir_c3_sellado_y_anclado_para_si_el_otro_commit_toca_matrixai(tmp_path, monkeypatch):
    raiz = _repo_git_temporal(tmp_path)
    c0 = _commit(raiz, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "c0")
    c1 = _commit(raiz, {"matrixai/estudio/metricas.py": "cambio en el núcleo"}, "c1")
    monkeypatch.setitem(p119.c3._RUTAS_DE_REPOSITORIO, "matrixAI", raiz)
    _poner_c3(monkeypatch, tmp_path, _payload_c3_fabricado(commit_core=c0))
    with pytest.raises(SystemExit, match="matrixai/estudio/metricas.py"):
        _anclar(_procedencia_c4(commit_core=c1))


# ---------------------------------------------------------------------------
# 4. LOS FALLOS EN LOS SELLADOS CUENTAN COMO EN LA ENMIENDA 2
# ---------------------------------------------------------------------------

def _cumplidos_sellados(nuevo, v2_otros, v2_densa, *, nombres=("dsA",), esperadas=_ESPERADAS_3):
    return p4.cumplidos_de_los_sellados(
        resultados_v2_todos=v2_otros + v2_densa, resultados_c4=nuevo,
        esperadas_por_conjunto={n: esperadas for n in nombres}, regla=REGLA,
        metrica_por_dataset={n: "auroc" for n in nombres}, nombres_sellados=list(nombres))


def test_cumplidos_de_los_sellados_cuenta_cumplidos_y_no_cumplidos():
    otros = _serie("lightgbm", [0.80, 0.80, 0.80])
    densa = _serie(DENSA, [0.79, 0.79, 0.79])
    nuevo = _serie(NUEVO, [0.80, 0.80, 0.80])
    cumplidos_nuevo, cumplidos_v2 = _cumplidos_sellados(nuevo, otros, densa)
    assert cumplidos_nuevo["cumplidos"] == 1 and cumplidos_nuevo["datasets"] == 1
    assert cumplidos_v2["cumplidos"] == 1


@pytest.mark.parametrize("hueco", ["fallos", "ausentes", "sin_metrica"])
def test_un_intento_sin_medida_del_nuevo_en_un_sellado_pierde_el_conjunto(hueco):
    """Enmienda 2, condición 1: CUALQUIER intento sin medida del motor nuevo
    -- fallido, AUSENTE o completado SIN la métrica de cierre -- pierde el
    conjunto, exactamente como en C3 (aquí sobre un «sellado» fabricado). El
    fallido solo lo ve también la regla de cierre; el ausente y el sin
    métrica solo los cuentan los fallos de la enmienda (S6 de la auditoría)."""
    otros = _serie("lightgbm", [0.80, 0.80, 0.80])
    densa = _serie(DENSA, [0.79, 0.79, 0.79])
    nuevo_con_hueco = _serie(NUEVO, [0.80, 0.80, 0.80], **{hueco: {(0, 1)}})
    cumplidos_nuevo, cumplidos_v2 = _cumplidos_sellados(nuevo_con_hueco, otros, densa)
    assert cumplidos_nuevo["cumplidos"] == 0
    assert cumplidos_nuevo["detalle"][0]["perdido_por_fallo"] is True
    assert cumplidos_v2["cumplidos"] == 1  # la densa v2, sin el hueco, sigue cumpliendo


def test_el_nuevo_compite_en_el_sitio_de_la_densa_v2():
    """El CAMPO: en los sellados, la densa v2 sale y entra el motor nuevo.
    Aquí la densa v2 es la mejor de la v2 (0,90) y el nuevo (0,85) está a 5
    puntos de ella pero por encima de todos los demás: cumple SOLO si la
    densa v2 se sustituye (S7 de la auditoría: sin sustituirla, no cumple)."""
    otros = _serie("lightgbm", [0.80] * 3)
    densa = _serie(DENSA, [0.90] * 3)
    nuevo = _serie(NUEVO, [0.85] * 3)
    cumplidos_nuevo, cumplidos_v2 = _cumplidos_sellados(nuevo, otros, densa)
    assert cumplidos_nuevo["cumplidos"] == 1
    assert cumplidos_nuevo["detalle"][0]["mejor"] == NUEVO
    assert cumplidos_v2["cumplidos"] == 1  # la densa v2, en la v2 SIN tocar, es la mejor


def test_cumplidos_de_los_sellados_con_dos_conjuntos():
    """Un conjunto cumple y el otro no: `datasets` cuenta los DOS."""
    otros_a = _serie("lightgbm", [0.80] * 3, dataset="dsA")
    densa_a = _serie(DENSA, [0.79] * 3, dataset="dsA")
    nuevo_a = _serie(NUEVO, [0.80] * 3, dataset="dsA")  # cumple
    otros_b = _serie("lightgbm", [0.80] * 3, dataset="dsB")
    densa_b = _serie(DENSA, [0.79] * 3, dataset="dsB")
    nuevo_b = _serie(NUEVO, [0.50] * 3, dataset="dsB")  # NO cumple, muy lejos
    cumplidos_nuevo, _ = _cumplidos_sellados(
        nuevo_a + nuevo_b, otros_a + otros_b, densa_a + densa_b, nombres=("dsA", "dsB"))
    assert cumplidos_nuevo["cumplidos"] == 1
    assert cumplidos_nuevo["datasets"] == 2


def test_cumplidos_de_los_sellados_es_la_misma_cuenta_que_la_de_c3():
    """M6: UNA sola implementación. Lo que C4 cuenta en los sellados es lo
    que `veredicto_final` de C3 cuenta con los mismos registros."""
    otros = _serie("lightgbm", [0.80] * 3) + _serie("lightgbm", [0.70] * 3, dataset="dsB")
    densa = _serie(DENSA, [0.90] * 3) + _serie(DENSA, [0.69] * 3, dataset="dsB")
    nuevo = _serie(NUEVO, [0.85] * 3) + _serie(NUEVO, [0.70] * 3, dataset="dsB",
                                               ausentes={(0, 2)})
    nombres = ["dsA", "dsB"]
    de_c4 = _cumplidos_sellados(nuevo, otros, densa, nombres=tuple(nombres))
    de_c3 = p119.veredicto_final(
        resultados_v2=otros + densa, resultados_del_motor_nuevo=nuevo, veredictos_por_conjunto=[],
        esperadas_por_conjunto={n: _ESPERADAS_3 for n in nombres}, regla=REGLA,
        metrica_por_dataset={n: "auroc" for n in nombres}, nombres_de_los_conjuntos=nombres)
    assert de_c4[0] == de_c3["cumplidos_con_el_motor_nuevo"]
    assert de_c4[1] == de_c3["cumplidos_de_la_densa_v2_en_los_mismos_conjuntos"]


# ---------------------------------------------------------------------------
# 5. EL X/40: 31/32 de C3 + y/8 de C4, y la decisión de D2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cumplidos_c4, x_esperado, decision_esperada", [
    (6, 37, "entra_en_la_cartera"),            # 37 >= 32
    (1, 32, "entra_en_la_cartera"),            # exactamente el umbral
    (0, 31, "avance_publicado_sin_cartera"),   # 31: > 27, < 32
])
def test_el_x_de_40_se_compone_bien(cumplidos_c4, x_esperado, decision_esperada):
    cumplidos_c3 = {"cumplidos": 31, "datasets": 32}
    cumplidos_c4_d = {"cumplidos": cumplidos_c4, "datasets": 8}
    densa_v2_c3 = {"cumplidos": 11, "datasets": 32}
    densa_v2_c4 = {"cumplidos": 0, "datasets": 8}
    x40 = p4.x_de_40(cumplidos_c3=cumplidos_c3, cumplidos_c4=cumplidos_c4_d,
                     densa_v2_c3=densa_v2_c3, densa_v2_c4=densa_v2_c4)
    assert x40["x_de_40"] == x_esperado
    assert x40["n_conjuntos"] == 40
    assert x40["reparto"]["c3_no_sellados"]["cumplidos"] == 31
    assert x40["reparto"]["c4_sellados"]["cumplidos"] == cumplidos_c4
    assert x40["decision_segun_d2"]["decision"] == decision_esperada
    assert x40["densa_v2_en_los_mismos_conjuntos"]["cumplidos"] == 11


def test_el_x_de_40_por_debajo_del_techo_se_cierra_como_el_118():
    cumplidos_c3 = {"cumplidos": 10, "datasets": 32}
    cumplidos_c4 = {"cumplidos": 0, "datasets": 8}
    densa_v2 = {"cumplidos": 0, "datasets": 32}
    densa_v2_c4 = {"cumplidos": 0, "datasets": 8}
    x40 = p4.x_de_40(cumplidos_c3=cumplidos_c3, cumplidos_c4=cumplidos_c4,
                     densa_v2_c3=densa_v2, densa_v2_c4=densa_v2_c4)
    assert x40["x_de_40"] == 10
    assert x40["decision_segun_d2"]["decision"] == "se_cierra_como_el_118"


def test_decision_segun_d2_el_techo_es_estimado_no_medido():
    d27 = p4.decision_segun_d2(27, 40)
    d28 = p4.decision_segun_d2(28, 40)
    assert d27["decision"] == "se_cierra_como_el_118"  # 27 NO es > 27
    assert d28["decision"] == "avance_publicado_sin_cartera"
    assert d27["techo_de_los_retoques"]["medido"] is False
    assert d27["umbral_de_la_cartera"]["medido"] is True
    assert "x = 27" in d27["frontera_27"]  # M1: el caso que D2 no define, declarado


def test_el_x_de_40_declara_como_cuenta_los_fallos_en_los_sellados():
    """M7: la enmienda 2 se EXTIENDE a los sellados y en C4 no se reintenta;
    las dos cosas, escritas dentro del X/40."""
    x40 = p4.x_de_40(cumplidos_c3={"cumplidos": 31, "datasets": 32},
                     cumplidos_c4={"cumplidos": 4, "datasets": 8},
                     densa_v2_c3={"cumplidos": 11, "datasets": 32},
                     densa_v2_c4={"cumplidos": 5, "datasets": 8})
    fallos = x40["como_se_cuentan_los_fallos"]
    assert "8 sellados" in fallos["extension_a_los_sellados"]
    assert "NO se reintenta" in fallos["sin_reintentos_en_los_sellados"]
    assert fallos["a_condicion_1"] == p119.COMO_SE_CUENTAN_LOS_FALLOS["a_condicion_1"]


def test_el_x_de_40_real_de_c3_reproduce_31_de_32():
    """No fabricado: el `veredicto` YA sellado de C3 en el árbol, leído tal
    cual, trae 31/32 -- el número que el contrato declara medido."""
    payload_c3 = p119._leer_json(p4.RUTA_DEL_RESULTADO_C3)
    assert payload_c3["veredicto"]["cumplidos_con_el_motor_nuevo"]["cumplidos"] == 31
    assert payload_c3["veredicto"]["cumplidos_con_el_motor_nuevo"]["datasets"] == 32
    sellos = p4.verificar_los_sellos_de_c3(payload_c3)
    assert sellos["cuadra"] is True, sellos["problemas"]


# ---------------------------------------------------------------------------
# 6. --estimar NUNCA LLAMA AL MOTOR REAL, y avisa si su orden no serviría
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def estimacion_c4(tmp_path_factory):
    salida = tmp_path_factory.mktemp("estimar_c4") / "estimacion.json"

    def explota(*a, **kw):
        raise AssertionError("ejecutar_intento_aislado NO debía llamarse durante --estimar")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(p4, "ejecutar_intento_aislado", explota)
        p4.main(["--estimar", "--salida", str(salida)])
    return json.loads(salida.read_text(encoding="utf-8"))


def test_estimar_no_llama_al_motor(estimacion_c4):
    assert estimacion_c4["tipo_de_ejecucion"] == "estimar"
    assert estimacion_c4["estimacion"]["total_horas"] > 0
    assert len(estimacion_c4["medidas_de_c3_usadas"]) > 0


def test_estimar_encola_c4_y_no_c3(estimacion_c4):
    """S22 de la auditoría: la orden es la de C4 -- su trabajo, su guion y su
    salida en la cola --, de día y de noche."""
    encolar = estimacion_c4["para_encolar"]
    esperado = (f"~/encolar.sh 119-c4 ", f"python3 benchmarks/fase0/pasada_119_c4.py --salida "
                                         f"{p4.SALIDA_DE_C4_EN_LA_COLA}")
    for orden in (encolar["de_dia"]["encolar"], encolar["de_noche"]["ordenes_una_por_noche"][0]):
        assert all(e in orden for e in esperado), orden
        assert "pasada_119_c3" not in orden


def test_estimar_memoria_usa_el_suelo_de_8g_ya_probado_por_c3_no_4g(estimacion_c4):
    """Los 387 registros del motor nuevo en resultado_pasada_119_c3.json
    traen `rss_pico_mb` en `null` (medido): sin dato real que escalar, el
    suelo tiene que ser el techo con el que C3 corrió DE VERDAD (8 GiB,
    cola nocturna), no un 4G sin probar nada del motor nuevo. Y el pico
    AUSENTE es `null`, no un 0 que parezca medido (M2)."""
    assert p4.MEMORIA_MINIMA_SUGERIDA_GB == 8
    memoria = estimacion_c4["memoria"]
    assert memoria["memory_max_sugerido"] == "8G"
    assert memoria["suelo_gb"] == 8
    assert "8 GiB" in memoria["de_donde_sale_el_suelo"]
    assert memoria["aviso"] is not None  # sigue diciendo que C3 no trae el dato
    assert memoria["pico_mb_medido_en_c3"] is None


def test_estimar_la_preparacion_sale_de_lo_medido_no_de_ceros(estimacion_c4):
    """M2: la preparación por intento de los 8 sellados sale de los registros
    de C3 (cada uno trae la suya), no de un 0,0 del padre de la continuación."""
    medidas = estimacion_c4["medidas_de_c3_usadas"]
    assert sum(1 for m in medidas.values() if m["preparacion_s"] > 0) >= 30
    por_conjunto = estimacion_c4["estimacion"]["por_conjunto"]
    assert len(por_conjunto) == 8
    assert sum(1 for c in por_conjunto if c["preparacion_s_por_intento"] > 0) >= 6


def test_medidas_desde_c3_un_dato_ausente_es_none():
    payload = {"resultados": [{"dataset": "dsA", "motor": NUEVO, "wall_s": 2.0, "cubo": "pequeno"}],
               "particion_por_dataset": {}, "tiempos_del_padre_por_dataset": {}}
    m = p4.medidas_desde_c3(payload)["dsA"]
    assert m["preparacion_s"] is None and m["carga_s"] is None and m["celdas"] is None
    assert m["wall_s"] == 2.0


def test_estimar_se_niega_si_no_esta_el_resultado_de_c3(tmp_path, monkeypatch):
    monkeypatch.setattr(p4, "RUTA_DEL_RESULTADO_C3", tmp_path / "no-esta.json")
    with pytest.raises(SystemExit, match="no está"):
        p4.main(["--estimar", "--salida", str(tmp_path / "e.json")])


def test_medidas_desde_c3_no_estan_vacias_para_los_datos_reales():
    payload_c3 = p119._leer_json(p4.RUTA_DEL_RESULTADO_C3)
    medidas = p4.medidas_desde_c3(payload_c3)
    assert len(medidas) == 32  # los 32 no sellados que C3 sí midió
    for m in medidas.values():
        assert m["wall_s"] > 0 and m["cubo"] in ("pequeno", "mediano", "grande")


def test_avisos_de_encolado(tmp_path, monkeypatch):
    """M3: la orden de `--estimar` fija el HEAD de ahora; si ese HEAD no trae
    el guion de C4, o un árbol está sucio, la orden NO serviría y se AVISA."""
    nucleo = _repo_git_temporal(tmp_path, "nucleo")
    motores = _repo_git_temporal(tmp_path, "motores")
    _commit(nucleo, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "sin C4")
    _commit(motores, {"README": "m"}, "motores")
    monkeypatch.setitem(p119.c3._RUTAS_DE_REPOSITORIO, "matrixAI", nucleo)
    monkeypatch.setitem(p119.c3._RUTAS_DE_REPOSITORIO, "matrixai-engines", motores)
    avisos = p4.avisos_de_encolado()
    assert len(avisos) == 1 and "NO contiene benchmarks/fase0/pasada_119_c4.py" in avisos[0]
    _commit(nucleo, {p4.GUION_DE_C4: "c4"}, "con C4")
    assert p4.avisos_de_encolado() == []
    (nucleo / p4.GUION_DE_C4).write_text("c4 editado sin commitear", encoding="utf-8")
    (motores / "nuevo.py").write_text("x", encoding="utf-8")
    avisos = p4.avisos_de_encolado()
    assert len(avisos) == 2 and all("SUCIO" in a for a in avisos)


# ---------------------------------------------------------------------------
# 7. EL DIGEST DE LA CACHÉ DE C4 CUBRE SU PROPIO GUION (I6)
# ---------------------------------------------------------------------------

def test_el_guion_de_c4_entra_en_el_digest_de_su_cache():
    etiqueta = "core:benchmarks/fase0/pasada_119_c4.py"
    assert etiqueta in {p119._etiqueta(f) for f in p4.ficheros_que_c4_anade_al_entorno()}
    componentes = p4.componentes_del_digest_del_entorno_c4()
    assert componentes[etiqueta] == p119.c3._digest_fichero(Path(p4.__file__))
    assert etiqueta not in p119.componentes_del_digest_del_entorno()  # el de C3, sin él
    fuera = p4.lo_que_no_cubre_el_digest_c4()[
        "modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest"]
    assert etiqueta not in fuera


# ---------------------------------------------------------------------------
# 8. EL ARNÉS ENTERO, CON UN MOTOR FALSO, SOBRE DOS SELLADOS BARATOS (--solo)
# ---------------------------------------------------------------------------

_ESCENARIO_C4 = "pc3,banknote-authentication"


@pytest.fixture(scope="module")
def pasada_falsa_c4(tmp_path_factory, _v2):
    salida = tmp_path_factory.mktemp("pasada_c4") / "resultado.json"
    densa = {n: _valores_v2(_v2, n, DENSA, "auroc") for n in ("pc3", "banknote-authentication")}

    def valor(dataset, rep, pl):
        delta = {"pc3": 0.10, "banknote-authentication": 0.0001}[dataset]
        return min(1.0, densa[dataset][(rep, pl)] + delta)

    llamadas: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(p4, "ejecutar_intento_aislado",
                   _motor_falso(valor, llamadas=llamadas))
        p4.main(["--solo", _ESCENARIO_C4, "--salida", str(salida)])
    return {"salida": salida, "payload": json.loads(salida.read_text(encoding="utf-8")),
            "llamadas": llamadas}


def test_main_c4_llama_una_vez_por_intento_de_la_particion(pasada_falsa_c4):
    llamadas = pasada_falsa_c4["llamadas"]
    n_pc3 = sum(1 for l in llamadas if l["dataset"] == "pc3")
    n_bank = sum(1 for l in llamadas if l["dataset"] == "banknote-authentication")
    assert n_pc3 == 15 and n_bank == 15  # cubo "pequeno": 3 repeticiones x 5 pliegues


def test_main_c4_declara_es_subconjunto_de_prueba(pasada_falsa_c4):
    payload = pasada_falsa_c4["payload"]
    assert payload["es_subconjunto_de_prueba"] is True
    assert sorted(payload["subconjunto_pedido"]) == sorted(_ESCENARIO_C4.split(","))
    assert payload["corte"] == "119-C4"
    assert payload["parcial"] is False


def test_main_c4_tipo_solo_no_compone_el_x_de_40(pasada_falsa_c4):
    """`--solo` es para probar el guion: nunca cuenta como la confirmación,
    así que no compone (ni exige) el X/40."""
    assert pasada_falsa_c4["payload"]["veredicto_x_de_40"] is None


def test_main_c4_la_arquitectura_de_cada_intento_se_comprueba(pasada_falsa_c4):
    resultados = [r for r in pasada_falsa_c4["payload"]["resultados"] if r["motor"] == NUEVO]
    assert resultados
    for r in resultados:
        assert r["arquitectura"]["k"] == 8
        assert r["arquitectura"]["d_block"] == 256


def test_main_c4_veredicto_por_conjunto_tiene_los_dos_sellados(pasada_falsa_c4):
    nombres = {v["dataset"] for v in pasada_falsa_c4["payload"]["veredicto_por_conjunto"]}
    assert nombres == {"pc3", "banknote-authentication"}


def test_main_c4_el_artefacto_no_se_contradice_sobre_su_digest(pasada_falsa_c4):
    """I6, en el artefacto: el guion de C4 está en los componentes y NO en la
    lista de «fuera del digest», que dice de sí misma que tiene que estar
    vacía (en las pruebas, solo pueden aparecer los módulos de PRUEBA)."""
    payload = pasada_falsa_c4["payload"]
    assert "core:benchmarks/fase0/pasada_119_c4.py" in payload["digest_de_la_cache"]["componentes"]
    fuera = payload["lo_que_no_cubre_el_digest"][
        "modulos_de_los_repos_cargados_por_este_proceso_fuera_del_digest"]
    assert all(m.startswith("core:tests/") for m in fuera), fuera


def test_main_c4_reusa_el_cache_en_una_segunda_pasada(tmp_path, _v2):
    """Repetir la MISMA orden sobre el mismo fichero reusa: el motor falso
    se llama 0 veces la segunda vez."""
    salida = tmp_path / "resultado.json"
    densa = _valores_v2(_v2, "pc3", DENSA, "auroc")

    def valor(dataset, rep, pl):
        return densa[(rep, pl)] + 0.05

    llamadas1: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(p4, "ejecutar_intento_aislado", _motor_falso(valor, llamadas=llamadas1))
        p4.main(["--solo", "pc3", "--salida", str(salida)])
    assert len(llamadas1) == 15

    llamadas2: list = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(p4, "ejecutar_intento_aislado", _motor_falso(valor, llamadas=llamadas2))
        p4.main(["--solo", "pc3", "--salida", str(salida)])
    assert llamadas2 == []
    payload2 = json.loads(salida.read_text(encoding="utf-8"))
    assert payload2["n_reusados"] == 15


# ---------------------------------------------------------------------------
# 9. LA PASADA (sin --solo) DE PUNTA A PUNTA: motor falso + anclaje FABRICADO
#    (auditoría del guion, I5). El anclaje entero es de verdad -- sellos, git,
#    procedencia, componentes del entorno --, sobre DOS repos git fabricados
#    (núcleo: el commit de C3 y, encima, el de C4 con el resultado de C3
#    dentro; motores: uno) y un C3 fabricado y sellado con 31/32 (densa v2
#    11/32). La lista de la pasada se recorta a pc3 + banknote (sellados) +
#    kc2 (NO sellado: la pasada no puede medirlo).
# ---------------------------------------------------------------------------

_SELLADOS_DE_LA_PASADA = ("banknote-authentication", "pc3")
_LISTA_DE_LA_PASADA = ("kc2",) + _SELLADOS_DE_LA_PASADA


def _anclar_con_repos_fabricados(mp, base: Path, *, commit_engines_de_c3: str | None = None
                                 ) -> dict:
    nucleo = _repo_git_temporal(base, "nucleo")
    motores = _repo_git_temporal(base, "motores")
    c0 = _commit(nucleo, {"benchmarks/fase0/pasada_119_c3.py": "v0"}, "el commit que midió C3")
    e0 = _commit(motores, {"README": "motores"}, "motores")
    payload = _payload_c3_fabricado(
        commit_core=c0, commit_engines=commit_engines_de_c3 or e0,
        digest_motor=p119._digest_motor_nuevo(),
        componentes=p119.componentes_del_digest_del_entorno())
    ruta_rel = p4.RUTA_RELATIVA_DEL_RESULTADO_C3
    c1 = _commit(nucleo, {ruta_rel: json.dumps(payload),
                          "benchmarks/fase0/pasada_119_c4.py": "el guion de C4",
                          "tests/test_119_c4_pasada.py": "sus pruebas"}, "el commit de C4")
    mp.setattr(p4, "RUTA_DEL_RESULTADO_C3", nucleo / ruta_rel)
    mp.setattr(p4, "BLOB_DEL_RESULTADO_C3_SELLADO", _blob(nucleo, c1, ruta_rel))
    mp.setattr(p4, "DIGEST_RESULTADOS_CRUDOS_DE_C3", payload["digest_resultados_crudos"])
    mp.setitem(p119.c3._RUTAS_DE_REPOSITORIO, "matrixAI", nucleo)
    mp.setitem(p119.c3._RUTAS_DE_REPOSITORIO, "matrixai-engines", motores)
    rutas = (base / "cola.json", base / "arbol_por_omision.json", base / "arbol_copiado.json")
    mp.setattr(p4, "RUTAS_QUE_MIRA_UNA_SOLA_EJECUCION", rutas)
    original = c5.datasets_de_la_pasada
    mp.setattr(p119.c5, "datasets_de_la_pasada",
               lambda prot: [d for d in original(prot) if d.nombre in _LISTA_DE_LA_PASADA])
    return {"nucleo": nucleo, "c0": c0, "c1": c1, "rutas_fijas": rutas}


def _mejor_de_la_v2(v2: dict) -> dict:
    mejor: dict = {}
    for r in v2["resultados"]:
        if r["dataset"] in _SELLADOS_DE_LA_PASADA and r["estado"] == "completed" \
                and r["motor"] not in ("baseline", DENSA) and r.get("auroc") is not None:
            k = (r["dataset"], r["repeticion"], r["pliegue"])
            mejor[k] = max(r["auroc"], mejor.get(k, -1.0))
    return mejor


def _valor_que_gana(v2: dict):
    """El mejor de la v2 (sin baseline ni densa) + 0,001: el motor nuevo es el
    mejor en cada pliegue, así que CUMPLE en cada sellado sin fallos. La densa
    v2 cumple en banknote y NO en pc3 (4,6 puntos del mejor, medido): 1 de 2."""
    mejor = _mejor_de_la_v2(v2)
    return lambda dataset, rep, pl: mejor.get((dataset, rep, pl), 0.5) + 0.001


def _pasada(mp, v2, salida: Path, *, llamadas: list, fallos=frozenset(), revienta_en=None):
    mp.setattr(p4, "ejecutar_intento_aislado",
               _motor_falso(_valor_que_gana(v2), llamadas=llamadas, fallos=fallos,
                            revienta_en=revienta_en))
    p4.main(["--salida", str(salida)])


@pytest.fixture(scope="module")
def pasada_real_falsa(tmp_path_factory, _v2):
    base = tmp_path_factory.mktemp("pasada_real_c4")
    salida = base / "salida" / "pasada_119_c4_resultado.json"
    llamadas: list = []
    with pytest.MonkeyPatch.context() as mp:
        anclaje = _anclar_con_repos_fabricados(mp, base)
        _pasada(mp, _v2, salida, llamadas=llamadas)
    return {"salida": salida, "payload": json.loads(salida.read_text(encoding="utf-8")),
            "llamadas": llamadas, "anclaje": anclaje}


def test_la_pasada_mide_solo_los_sellados(pasada_real_falsa):
    """S11: con un NO sellado (kc2) en la lista, la pasada mide SOLO los
    sellados, 15 intentos cada uno."""
    llamadas = pasada_real_falsa["llamadas"]
    assert {l["dataset"] for l in llamadas} == set(_SELLADOS_DE_LA_PASADA)
    assert len(llamadas) == 30
    assert {d["nombre"] for d in pasada_real_falsa["payload"]["datasets_declarados"]} == \
        set(_SELLADOS_DE_LA_PASADA)


def test_la_pasada_compone_el_x_de_40_como_31_mas_y(pasada_real_falsa):
    """X = 31 (C3, del resultado sellado) + y (C4) con y = 2 sabido a mano: el
    nuevo es el mejor en los dos sellados. n = 32 + 2 (la lista está
    recortada), así que D2 dice «incompleta». La densa v2 da 11 + 1."""
    payload = pasada_real_falsa["payload"]
    assert payload["tipo_de_ejecucion"] == "pasada" and payload["parcial"] is False
    x40 = payload["veredicto_x_de_40"]
    assert x40 is not None
    assert x40["x_de_40"] == 33 and x40["n_conjuntos"] == 34
    assert x40["reparto"]["c3_no_sellados"]["cumplidos"] == 31
    c4 = x40["reparto"]["c4_sellados"]
    assert c4["cumplidos"] == 2 and c4["datasets"] == 2
    assert {d["dataset"] for d in c4["detalle"]} == set(_SELLADOS_DE_LA_PASADA)
    assert all(d["cumple"] and not d["perdido_por_fallo"] for d in c4["detalle"])
    assert x40["densa_v2_en_los_mismos_conjuntos"]["cumplidos"] == 12
    assert x40["densa_v2_en_los_mismos_conjuntos"]["reparto"]["c4_sellados"]["cumplidos"] == 1
    assert x40["decision_segun_d2"]["decision"] == "incompleta"
    assert x40["n_intentos_reintentados"] == 0


def test_la_pasada_admite_el_commit_de_c4_con_el_resultado_de_c3(pasada_real_falsa):
    """B1 de punta a punta: el commit de C4 (encima del de C3, con el
    resultado de C3 DENTRO) se admite, y el anclaje del entorno también."""
    x40 = pasada_real_falsa["payload"]["veredicto_x_de_40"]
    dif = x40["diferencia_de_nucleo_entre_c3_y_c4"]
    anclaje = pasada_real_falsa["anclaje"]
    assert (dif["commit_c3"], dif["commit_c4"]) == (anclaje["c0"], anclaje["c1"])
    assert p4.RUTA_RELATIVA_DEL_RESULTADO_C3 in dif["ficheros_cambiados"]
    assert dif["admitida"] is True and dif["resultado_de_c3_en_el_commit_de_c4"]["cuadra"] is True
    assert x40["diferencia_del_entorno_entre_c3_y_c4"]["admitida"] is True


def test_la_pasada_se_niega_sobre_su_propio_resultado_completo(tmp_path, monkeypatch, _v2,
                                                              pasada_real_falsa):
    """S9: la misma orden sobre el resultado COMPLETO para, con el anclaje en
    regla (si no, pararía por otra cosa y la prueba no diría nada): 0
    llamadas y el fichero intacto."""
    salida = tmp_path / "completo.json"
    salida.write_bytes(pasada_real_falsa["salida"].read_bytes())
    md5 = _md5(salida)
    _anclar_con_repos_fabricados(monkeypatch, tmp_path)
    llamadas: list = []
    with pytest.raises(SystemExit, match="UNA vez, en los SELLADOS"):
        _pasada(monkeypatch, _v2, salida, llamadas=llamadas)
    assert llamadas == []
    assert _md5(salida) == md5


def test_la_pasada_se_niega_con_otra_salida_si_la_cola_ya_tiene_el_completo(
        tmp_path, monkeypatch, _v2, pasada_real_falsa):
    """I3 de punta a punta: el completo en la ruta de la COLA y la orden con
    OTRA `--salida` también para, sin medir nada."""
    anclaje = _anclar_con_repos_fabricados(monkeypatch, tmp_path)
    cola = anclaje["rutas_fijas"][0]
    cola.write_bytes(pasada_real_falsa["salida"].read_bytes())
    otra = tmp_path / "otra" / "salida.json"
    llamadas: list = []
    with pytest.raises(SystemExit, match="cola.json"):
        _pasada(monkeypatch, _v2, otra, llamadas=llamadas)
    assert llamadas == []
    assert not otra.exists()


def test_la_pasada_comprueba_el_anclaje_al_arrancar(tmp_path, monkeypatch, _v2):
    """I1: con el motor DESANCLADO (C3 midió con otro commit de motores), la
    pasada PARA antes del primer intento -- no con los sellados ya medidos --
    y no escribe nada."""
    _anclar_con_repos_fabricados(monkeypatch, tmp_path, commit_engines_de_c3="otro-commit")
    salida = tmp_path / "c4.json"
    llamadas: list = []
    with pytest.raises(SystemExit, match="matrixai-engines: C3 midió en otro-commit"):
        _pasada(monkeypatch, _v2, salida, llamadas=llamadas)
    assert llamadas == []
    assert not salida.exists()


@pytest.fixture(scope="module")
def parcial_con_un_fallo(tmp_path_factory, _v2):
    """Una pasada que FALLA en (banknote, 0, 0) y la cola corta en (pc3, 0, 1)."""
    base = tmp_path_factory.mktemp("parcial_c4")
    salida = base / "parcial.json"
    llamadas: list = []
    with pytest.MonkeyPatch.context() as mp:
        _anclar_con_repos_fabricados(mp, base)
        with pytest.raises(KeyboardInterrupt):
            _pasada(mp, _v2, salida, llamadas=llamadas,
                    fallos=frozenset({("banknote-authentication", 0, 0)}),
                    revienta_en=("pc3", 0, 1))
    return {"salida": salida, "llamadas": llamadas,
            "payload": json.loads(salida.read_text(encoding="utf-8"))}


def test_un_corte_no_pierde_ningun_intento_ya_ejecutado(parcial_con_un_fallo):
    """M5: el punto de control va tras CADA intento. El corte llega en
    (pc3, 0, 1): los 15 de banknote y (pc3, 0, 0) ya están escritos (con el
    de C3, por repetición o cada 60 s, (pc3, 0, 0) se perdía y se re-medía)."""
    payload = parcial_con_un_fallo["payload"]
    assert payload["parcial"] is True
    assert len(parcial_con_un_fallo["llamadas"]) == 16
    assert len(payload["resultados"]) == 16
    assert ("pc3", 0, 0) in {(r["dataset"], r["repeticion"], r["pliegue"])
                             for r in payload["resultados"]}


def test_al_reanudar_un_fallo_sellado_se_reusa_no_se_reintenta(tmp_path, monkeypatch, _v2,
                                                               parcial_con_un_fallo):
    """I4 a: el fallido (banknote, 0, 0) ES la medida: la reanudación NO lo
    vuelve a llamar, mide solo los 14 que faltan de pc3, y banknote queda
    PERDIDO por el fallo (enmienda 2 en los sellados): X = 31 + 1."""
    salida = tmp_path / "parcial.json"
    salida.write_bytes(parcial_con_un_fallo["salida"].read_bytes())
    _anclar_con_repos_fabricados(monkeypatch, tmp_path)
    llamadas: list = []
    _pasada(monkeypatch, _v2, salida, llamadas=llamadas)
    llamados = {(l["dataset"], l["repeticion"], l["pliegue"]) for l in llamadas}
    assert ("banknote-authentication", 0, 0) not in llamados
    assert ("pc3", 0, 0) not in llamados
    assert len(llamadas) == 14 and {d for d, _, _ in llamados} == {"pc3"}
    payload = json.loads(salida.read_text(encoding="utf-8"))
    fallido = [r for r in payload["resultados"]
               if (r["dataset"], r["repeticion"], r["pliegue"]) == ("banknote-authentication", 0, 0)]
    assert len(fallido) == 1 and fallido[0]["estado"] == "failed" and fallido[0]["reusado"] is True
    assert payload["n_intentos_reintentados"] == 0 and payload["intentos_reintentados"] == []
    x40 = payload["veredicto_x_de_40"]
    assert x40["x_de_40"] == 32
    detalle = {d["dataset"]: d for d in x40["reparto"]["c4_sellados"]["detalle"]}
    assert detalle["banknote-authentication"]["perdido_por_fallo"] is True
    assert detalle["pc3"]["cumple"] is True
    assert x40["n_intentos_reintentados"] == 0


def test_al_reanudar_con_otro_codigo_para_sin_medir(tmp_path, monkeypatch, _v2,
                                                    parcial_con_un_fallo):
    """I4 b: si lo ya medido es de OTRO código (aquí, otro digest del entorno
    de C4), la reanudación PARA antes del primer intento en vez de re-medir
    los 16 sellados en silencio."""
    salida = tmp_path / "parcial.json"
    salida.write_bytes(parcial_con_un_fallo["salida"].read_bytes())
    md5 = _md5(salida)
    _anclar_con_repos_fabricados(monkeypatch, tmp_path)
    original = p4.componentes_del_digest_del_entorno_c4
    monkeypatch.setattr(p4, "componentes_del_digest_del_entorno_c4",
                        lambda: dict(original(), **{"core:benchmarks/fase0/otro.py": "otro"}))
    llamadas: list = []
    with pytest.raises(SystemExit, match="16 de 16 registros previos"):
        _pasada(monkeypatch, _v2, salida, llamadas=llamadas)
    assert llamadas == []
    assert _md5(salida) == md5
