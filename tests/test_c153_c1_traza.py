"""153-C1 — la traza firmada del servicio exportado (Ola 3, idea 19): formato `matrixai.traza/v1`, su comprobador,
`matrixai trace verify` y la paridad con el comprobador de JavaScript.

Decisiones que sostienen estas pruebas (contrato 153): DS3 (formato, firma con separación de dominio, cadena por
`prev`, génesis de 64 ceros), DS7 (vocabulario de 145-D11, nunca «verificado»), DS8–DS11 (clave pública y k_minimo dentro
de cada bloque, histograma del valor en regresión, la hora no se comprueba) y §4 (lo que una firma válida NO demuestra).

Los vectores (`tests/fixtures/traza/vectores_traza_v1.json`) llevan el veredicto ESCRITO A MANO por el generador, no
sacado del comprobador, y los comparten el núcleo y `comprobar_traza.mjs`; por eso aquí no se regeneran para comprobar:
solo se comprueba que regenerarlos da los mismos bytes (Ed25519 es determinista).

Los nombres `test_sabotaje_…` son los que matan, cada uno, un sabotaje del comprobador (ver el §9 del contrato).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import pytest

from matrixai.pipelines import firma_asimetrica as A
from matrixai.pipelines import traza as T
from matrixai.pipelines.canonical import jcs_bytes

RAIZ = Path(__file__).resolve().parents[1]
FIXTURES = RAIZ / "tests" / "fixtures" / "traza"
VECTORES = json.loads((FIXTURES / "vectores_traza_v1.json").read_text(encoding="utf-8"))
CASOS = {c["nombre"]: c for c in VECTORES["casos"]}
CADENA = [c["linea"] for c in VECTORES["cadena_valida"]["bloques"][:3]]  # los 3 bloques de clasificación
CLAVE_A = VECTORES["claves_de_prueba"]["A"]
CLAVE_B = VECTORES["claves_de_prueba"]["B"]
_NODE = shutil.which("node") or ("/usr/bin/node" if Path("/usr/bin/node").exists() else None)


def _comprobar(lineas, ancla=None, esperar_genesis=False):
    return T.comprobar_traza(lineas, ancla, esperar_genesis)


def _caso(nombre):
    c = CASOS[nombre]
    return _comprobar(c["lineas"], c.get("ancla"), c.get("esperar_genesis", False))


def _firmado_con(clave, bloque):
    return T.firmar_bloque(bytes.fromhex(clave["semilla_hex"]), bloque)


def _linea(b):
    return jcs_bytes(b).decode("utf-8")


# ───────────────────────────────────────────────────────────────── los vectores compartidos

@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_vector_compartido(nombre):
    """Cada caso de los vectores da el estado, el motivo y el bloque escritos a mano (y lo demás que el caso declare)."""
    r = _caso(nombre)
    for campo, esperado in CASOS[nombre]["esperado"].items():
        assert r.get(campo) == esperado, f"{nombre}: {campo}: esperado {esperado!r}, obtenido {r.get(campo)!r} ({r.get('detalle')})"


def test_los_vectores_son_los_que_el_generador_produce_y_son_muchos():
    """El JSON no se ha tocado a mano: regenerar da los MISMOS bytes. Y no se quedó en cuatro casos."""
    spec = importlib.util.spec_from_file_location("generar_vectores_traza", FIXTURES / "generar_vectores.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.texto_de_los_vectores() == (FIXTURES / "vectores_traza_v1.json").read_text(encoding="utf-8")
    assert len(CASOS) >= 60 and len(CASOS) == len(VECTORES["casos"])


def test_las_claves_de_prueba_son_de_prueba_y_el_fichero_no_lleva_ninguna_pem():
    texto = (FIXTURES / "vectores_traza_v1.json").read_text(encoding="utf-8")
    assert "PRIVATE KEY" not in texto and "BEGIN" not in texto
    assert "CLAVES DE PRUEBA" in VECTORES["AVISO"]
    for k in (CLAVE_A, CLAVE_B):
        assert A.huella_de_clave_publica(A.cargar_clave_privada(bytes.fromhex(k["semilla_hex"]))) == k["huella"]
    assert VECTORES["prefijo_de_firma"].encode("utf-8") == T.PREFIJO_FIRMA == b"matrixai.traza.firma/v1\n"


def test_el_canonico_el_mensaje_firmado_y_el_hash_de_cada_bloque_son_los_grabados():
    """Lo que el JS y C6 reproducen byte a byte: `canonico_sin_firma`, SHA-256 del mensaje firmado y hash del bloque."""
    for c in VECTORES["cadena_valida"]["bloques"]:
        b = json.loads(c["linea"])
        assert jcs_bytes({k: v for k, v in b.items() if k != "firma"}).decode("utf-8") == c["canonico_sin_firma"]
        assert hashlib.sha256(T.mensaje_firmado(b)).hexdigest() == c["sha256_del_mensaje_firmado"]
        assert T.hash_de_bloque(b) == c["hash_del_bloque"]


def test_la_firma_de_un_bloque_se_reproduce_con_el_escritor_del_nucleo():
    """`firmar_bloque` (Python) produce los MISMOS bytes que los vectores (Ed25519 es determinista)."""
    for c in VECTORES["cadena_valida"]["bloques"]:
        b = json.loads(c["linea"])
        assert _linea(_firmado_con(CLAVE_A, b)) == c["linea"]


# ───────────────────────────────────────────────────────────────── los sabotajes, uno por prueba

def test_sabotaje_editar_un_bloque_da_la_firma_no_coincide():
    for nombre in ("editado_n_sin_refirmar", "editado_un_recuento_con_suma_igual", "editado_id_del_servicio",
                   "editado_k_minimo", "prev_cambiado_sin_refirmar", "firma_alterada"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"]) == (A.NO_COINCIDE, "firma"), nombre
    # y un número, el que sea, de un campo cualquiera: ningún campo queda sin firmar
    b = json.loads(CADENA[1])
    for ruta in (("n",), ("seq",), ("k_minimo",), ("servicio", "id"), ("ventana", "hasta"), ("salida", "probabilidad", 0),
                 ("columnas", "edad_años", "bins", 2), ("columnas", "sexo", "nuevas"), ("reinicio",)):
        m = copy.deepcopy(b)
        destino = m
        for p in ruta[:-1]:
            destino = destino[p]
        antes = destino[ruta[-1]]
        destino[ruta[-1]] = "sin_previo" if ruta == ("reinicio",) else (antes + "x" if isinstance(antes, str) else antes + 1)
        if ruta == ("ventana", "hasta"):
            destino[ruta[-1]] = "2026-11-02T23:59:59Z"
        r = _comprobar([CADENA[0], _linea(m), CADENA[2]])
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "firma", 1), (ruta, r)


def test_sabotaje_borrar_un_bloque_del_medio_da_cadena_rota_y_dice_cual():
    r = _caso("borrado_el_del_medio")
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", 1)
    # cada una de las dos sacadas posibles, sobre una cadena de 3
    assert _comprobar([CADENA[1], CADENA[2]])["estado"] == A.VALIDA  # sacar la CABEZA no es borrar del medio (ver truncar)
    assert _comprobar([CADENA[0], CADENA[2]])["motivo"] == "cadena"


def test_sabotaje_reordenar_da_cadena_rota():
    r = _caso("reordenado_1_y_2")
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", 1)
    r = _comprobar(list(reversed(CADENA)))
    assert (r["estado"], r["motivo"]) == (A.NO_COINCIDE, "cadena")


def test_sabotaje_truncar_la_cola_NO_se_detecta_y_la_salida_lo_dice():
    """Lo que el formato no puede ver: el contrato (§4) lo escribe y el verificador lo repite en pantalla, en es y en en."""
    r = _caso("TRUNCADA_LA_COLA")
    assert r["estado"] == A.VALIDA and r["bloques"] == 2 and r["ultimo_seq"] == 1
    # lo que SÍ da: el último bloque visto, para cotejarlo con el que anuncia el servicio
    assert r["ultimo_hash"] == T.hash_de_bloque(json.loads(CADENA[1]))
    for idioma, frase in (("es", "Que no falten los últimos bloques"), ("en", "That no trailing blocks are missing")):
        pantalla = "\n".join(T.describir(r, idioma))
        assert frase in pantalla and f"seq {r['ultimo_seq']}" in pantalla
        assert pantalla.index(frase) > pantalla.index("NO")  # dentro de «lo que esto NO demuestra / does NOT show»
    # la cabeza, igual de invisible, salvo que quien importa por primera vez exija el génesis
    assert _caso("TRUNCADA_LA_CABEZA")["empieza_en"] == 1
    assert _caso("TRUNCADA_LA_CABEZA_esperando_genesis")["motivo"] == "cadena"
    assert "génesis" in "\n".join(T.describir(_caso("TRUNCADA_LA_CABEZA"), "es"))


def test_sabotaje_otra_clave_da_cadena_rota_o_ancla_y_sin_ancla_solo_se_ve_en_huellas():
    r = _caso("otra_clave_en_el_ultimo_bloque")
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", 2)
    assert _caso("clave_cambiada_sin_refirmar")["motivo"] == "firma"
    assert _caso("huella_que_no_es_la_de_su_clave")["motivo"] == "clave"
    assert _caso("otra_cadena_entera_con_ancla")["motivo"] == "ancla"
    sin_ancla = _caso("otra_cadena_entera_SIN_ancla")
    assert sin_ancla["estado"] == A.VALIDA and sin_ancla["huellas"] == [CLAVE_B["huella"]]
    assert "Sin huella anclada" in "\n".join(T.describir(sin_ancla, "es"))
    assert "No anchored fingerprint" in "\n".join(T.describir(sin_ancla, "en"))
    assert _comprobar(CADENA, CLAVE_A["huella"])["anclada"] is True


def test_sabotaje_prev_cambiado_y_refirmado_lo_ve_la_cadena_no_la_firma():
    for nombre in ("prev_cambiado_y_refirmado", "seq_saltado_y_refirmado"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", 1), nombre


def test_sabotaje_sin_separacion_de_dominio_una_firma_sin_el_prefijo_no_vale():
    """Un bloque firmado SOLO sobre el JCS (sin `matrixai.traza.firma/v1\\n`) no es una traza válida: la firma de una
    traza no vale como firma de recibo ni al revés (145-D8)."""
    clave = A.cargar_clave_privada(bytes.fromhex(CLAVE_A["semilla_hex"]))
    b = json.loads(CADENA[0])
    sin = {k: v for k, v in b.items() if k != "firma"}
    sin_dominio = {**sin, "firma": {"alg": "ed25519", "valor": A.firmar(clave, jcs_bytes(sin))}}
    r = _comprobar([_linea(sin_dominio)])
    assert (r["estado"], r["motivo"]) == (A.NO_COINCIDE, "firma")
    con_dominio = {**sin, "firma": {"alg": "ed25519", "valor": A.firmar(clave, T.PREFIJO_FIRMA + jcs_bytes(sin))}}
    assert _comprobar([_linea(con_dominio)])["estado"] == A.VALIDA
    # y al revés: la firma de un bloque no vale con el prefijo de OTRO dominio (el del recibo del 145)
    otro = {**sin, "firma": {"alg": "ed25519", "valor": A.firmar(clave, b"matrixai.recibo.firma/v1\n" + jcs_bytes(sin))}}
    assert _comprobar([_linea(otro)])["motivo"] == "firma"


# ───────────────────────────────────────────────────────────────── lo que dice cada estado

def test_tres_estados_y_el_vocabulario_de_145_D11_nunca_verificado():
    estados = {CASOS[n]["esperado"]["estado"] for n in CASOS}
    assert estados == set(A.ESTADOS)
    prohibidas = re.compile(r"verificad|verified|aut[eé]ntic|authentic|certificad|certified|garantiz|guarantee", re.I)
    textos = []
    for nombre in CASOS:
        r = _caso(nombre)
        for idioma in ("es", "en"):
            textos.extend(T.describir(r, idioma))
        if r["detalle"]:
            textos.extend(r["detalle"].values())
    textos.extend(v for d in T.MOTIVOS.values() for v in d.values())
    textos.extend(f for lista in T.LO_QUE_NO_DEMUESTRA.values() for f in lista)
    assert textos and not [t for t in textos if prohibidas.search(t)]
    assert T.describir(_caso("cadena_valida"), "es")[0] == "Traza: firma válida"
    assert T.describir(_caso("borrado_el_del_medio"), "es")[0] == "Traza: la firma no coincide"
    assert T.describir(_caso("borrado_el_del_medio"), "en")[0] == "Trace: the signature does not match"
    assert T.describir(_caso("fichero_vacio"), "en")[0] == "Trace: could not be checked"


def test_el_detalle_de_cada_fallo_esta_en_los_dos_idiomas():
    for nombre in CASOS:
        r = _caso(nombre)
        if r["estado"] != A.VALIDA:
            assert set(r["detalle"]) == {"es", "en"} and all(r["detalle"].values()), nombre


def test_las_firmas_se_miran_todas_antes_que_la_cadena_y_las_sumas_despues_de_las_firmas():
    """Si hay una firma mala en el bloque 2 y la cadena rota en el 1, sale `firma`; y un bloque editado dice
    «la firma no coincide», no «formato» (aunque sus sumas ya no cuadren)."""
    editado = json.loads(CADENA[2]); editado["n"] += 1
    r = _comprobar([CADENA[0], CADENA[2], _linea(editado)])  # el bloque 1 rompe la cadena y el 2 está editado
    assert r["motivo"] == "firma" and r["bloque"] == 2
    r = _comprobar([CADENA[0], _linea(editado), CADENA[2]])
    assert (r["estado"], r["motivo"]) == (A.NO_COINCIDE, "firma")


def test_la_hora_no_se_comprueba_pero_la_ventana_no_puede_estar_al_reves():
    """DS11: la hora la pone el reloj del servicio. Una ventana en 1999 o en 2099 es válida; al revés, no."""
    clave = A.cargar_clave_privada(bytes.fromhex(CLAVE_A["semilla_hex"]))
    b = json.loads(CADENA[0])
    for desde, hasta in (("1999-01-01T00:00:00Z", "1999-01-01T01:00:00Z"), ("2099-12-31T22:00:00Z", "2099-12-31T23:00:00Z")):
        m = {**b, "ventana": {"desde": desde, "hasta": hasta}}
        assert _comprobar([_linea(T.firmar_bloque(clave, m))])["estado"] == A.VALIDA
    assert _caso("ventana_al_reves")["estado"] == A.NO_SE_PUDO_COMPROBAR


def test_k_minimo_dentro_del_bloque_y_la_regla_de_DS2():
    assert _caso("completo_con_n_menor_que_k")["motivo"] == "formato"
    assert _caso("incompleto_con_n_mayor_o_igual_que_k")["motivo"] == "formato"
    assert _caso("k_minimo_cero")["estado"] == A.NO_SE_PUDO_COMPROBAR
    assert _caso("sin_k_minimo")["estado"] == A.NO_SE_PUDO_COMPROBAR
    assert _caso("k_minimo_propio_de_cada_bloque")["estado"] == A.VALIDA
    assert T.K_MINIMO_POR_DEFECTO == 20 and all(json.loads(l)["k_minimo"] == 20 for l in CADENA)


def test_regresion_el_valor_predicho_va_en_salida_valor_y_lo_demas_a_null():
    r = _caso("regresion_valida")
    assert r["estado"] == A.VALIDA and r["bloques"] == 2
    b = json.loads(CASOS["regresion_valida"]["lineas"][0])
    assert b["salida"]["probabilidad"] is None and b["salida"]["etiquetas"] is None
    assert len(b["salida"]["valor"]["bins"]) == 10 and sum(b["salida"]["valor"]["bins"]) == b["n"]
    assert "__proto__" in b["columnas"]
    for nombre in ("regresion_con_probabilidad", "regresion_valor_no_suma_n", "regresion_fuera_mayor_que_casillas"):
        assert _caso(nombre)["estado"] == A.NO_SE_PUDO_COMPROBAR, nombre
    assert _caso("regresion_editado_un_recuento_del_valor")["motivo"] == "firma"


def test_incompleto_no_lleva_ni_un_cero_inventado():
    """Ausente no es cero (criterio 7): un bloque incompleto tiene salida y columnas a null, no `{}` ni ceros."""
    r = _caso("incompleto_con_columnas")
    assert r["estado"] == A.NO_SE_PUDO_COMPROBAR
    b = json.loads(CASOS["incompleto_solo_n"]["lineas"][2])
    assert b["incompleto"] is True and b["salida"] is None and b["columnas"] is None


def test_ningun_bloque_de_los_vectores_lleva_un_valor_de_entrada_ni_mas_campos_que_el_formato():
    """Criterio 1 en miniatura, sobre lo que produce el generador: solo enteros, y los textos que hay son los
    nombres de columna, las etiquetas y el id del servicio."""
    def hojas(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from hojas(v)
        elif isinstance(x, list):
            for v in x:
                yield from hojas(v)
        else:
            yield x
    for c in VECTORES["cadena_valida"]["bloques"]:
        b = json.loads(c["linea"])
        assert not [h for h in hojas(b) if isinstance(h, float)]
        assert set(b) == {"formato", "seq", "prev", "servicio", "ventana", "n", "k_minimo", "incompleto", "reinicio",
                          "salida", "columnas", "firma"}


# ───────────────────────────────────────────────────────────────── lectura de ficheros y casos raros

def test_lineas_de_texto_parte_solo_por_salto_de_linea():
    assert T.lineas_de_texto("a\nb\n") == ["a", "b"] and T.lineas_de_texto("a\nb") == ["a", "b"]
    assert T.lineas_de_texto("a\n\nb\n") == ["a", "", "b"]
    assert T.lineas_de_texto("a b\n") == ["a b"]  # un separador Unicode no es una línea
    assert T.lineas_de_texto("") == []


def test_comprobar_fichero_con_bytes_con_utf8_roto_y_con_el_fichero_vacio():
    ok = ("\n".join(CADENA) + "\n").encode("utf-8")
    assert T.comprobar_fichero(ok)["estado"] == A.VALIDA
    assert T.comprobar_fichero(ok.decode("utf-8"))["estado"] == A.VALIDA
    roto = T.comprobar_fichero(b"\xff\xfe{}")
    assert (roto["estado"], roto["motivo"]) == (A.NO_SE_PUDO_COMPROBAR, "formato")
    assert T.comprobar_fichero(b"")["estado"] == A.NO_SE_PUDO_COMPROBAR


def test_una_huella_anclada_que_no_es_una_huella_es_un_error_de_quien_llama_no_un_veredicto():
    for mala in ("zz", "A" * 64, "a" * 63, 7):
        with pytest.raises(ValueError):
            _comprobar(CADENA, mala)


def test_leer_bloques_es_el_lector_estricto():
    assert [b["seq"] for b in T.leer_bloques(CADENA)] == [0, 1, 2]
    with pytest.raises(T.TrazaIlegible) as e:
        T.leer_bloques([CADENA[0], '{"a":1,"a":2}'])
    assert e.value.bloque == 1 and e.value.motivo == "formato"


def test_un_numero_enorme_en_un_json_no_hace_lanzar_al_comprobador():
    r = _comprobar([CADENA[0], '{"n":' + "9" * 5000 + "}"])
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 1)


# ───────────────────────────────────────────────────────────────── sin cryptography

_BLOQUEO = textwrap.dedent('''
    import importlib.abc, sys
    class _SinCrypto(importlib.abc.MetaPathFinder):
        def find_spec(self, nombre, ruta=None, destino=None):
            if nombre == "cryptography" or nombre.startswith("cryptography."):
                raise ImportError("cryptography bloqueada por la prueba")
    sys.meta_path.insert(0, _SinCrypto())
''')


def test_sin_cryptography_importa_y_dice_no_se_pudo_comprobar_con_su_motivo(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(CADENA) + "\n", encoding="utf-8")
    codigo = _BLOQUEO + textwrap.dedent(f'''
        import json, sys
        from matrixai.pipelines import traza as T
        r = T.comprobar_fichero(open({str(f)!r}, "rb").read())
        print(json.dumps({{"estado": r["estado"], "motivo": r["motivo"]}}))
        r2 = T.comprobar_traza(["no es json"])
        print(r2["motivo"])
        assert "cryptography" not in sys.modules
    ''')
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=RAIZ)
    assert r.returncode == 0, r.stderr[-600:]
    primera, segunda = r.stdout.strip().splitlines()
    assert json.loads(primera) == {"estado": "no_se_pudo_comprobar", "motivo": "falta_cryptography"}
    assert segunda == "formato"  # lo que no necesita la firma se sigue diciendo
    cli = _BLOQUEO + "import sys\nfrom matrixai.cli import main\nsys.argv = ['matrixai'] + sys.argv[1:]\nsys.exit(main())\n"
    r = subprocess.run([sys.executable, "-c", cli, "trace", "verify", str(f), "--locale", "es"], capture_output=True, text=True, cwd=RAIZ)
    assert r.returncode == 3 and "no se pudo comprobar" in r.stdout and "cryptography" in r.stdout


# ───────────────────────────────────────────────────────────────── la CLI

def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "matrixai", *argv], capture_output=True, text=True, cwd=RAIZ)


def _fichero(tmp_path, nombre, lineas):
    f = tmp_path / nombre
    f.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
    return str(f)


def test_cli_trace_verify_valida_sale_con_0_y_dice_lo_que_no_demuestra_en_es_y_en(tmp_path):
    f = _fichero(tmp_path, "ok.jsonl", CADENA)
    es = _cli("trace", "verify", f, "--locale", "es")
    assert es.returncode == 0 and "firma válida" in es.stdout and "Que no falten los últimos bloques" in es.stdout
    en = _cli("trace", "verify", f, "--locale", "en")
    assert en.returncode == 0 and "valid signature" in en.stdout and "That no trailing blocks are missing" in en.stdout
    assert not re.search(r"verificad|verified", es.stdout + en.stdout, re.I)
    js = json.loads(_cli("trace", "verify", f, "--json").stdout)
    assert js["estado"] == "valida" and js["bloques"] == 3 and js["huellas"] == [CLAVE_A["huella"]]


def test_cli_trace_verify_el_codigo_de_salida_distingue_los_tres_estados(tmp_path):
    ok = _cli("trace", "verify", _fichero(tmp_path, "a.jsonl", CADENA))
    roto = _cli("trace", "verify", _fichero(tmp_path, "b.jsonl", [CADENA[0], CADENA[2]]), "--locale", "es")
    basura = _cli("trace", "verify", _fichero(tmp_path, "c.jsonl", [CADENA[0], "esto no es json"]), "--locale", "es")
    assert (ok.returncode, roto.returncode, basura.returncode) == (0, 2, 3)
    assert "la firma no coincide" in roto.stdout and "en la línea 2" in roto.stdout
    assert "no se pudo comprobar" in basura.stdout
    editado = json.loads(CADENA[1]); editado["n"] += 1
    assert _cli("trace", "verify", _fichero(tmp_path, "d.jsonl", [CADENA[0], _linea(editado), CADENA[2]])).returncode == 2


def test_cli_trace_verify_huella_y_desde_genesis(tmp_path):
    f = _fichero(tmp_path, "ok.jsonl", CADENA)
    assert _cli("trace", "verify", f, "--huella", CLAVE_A["huella"]).returncode == 0
    assert _cli("trace", "verify", f, "--huella", CLAVE_A["huella"].upper()).returncode == 0  # se normaliza
    otra = _cli("trace", "verify", f, "--huella", CLAVE_B["huella"], "--locale", "es")
    assert otra.returncode == 2 and "huella" in otra.stdout
    cabeza = _fichero(tmp_path, "cabeza.jsonl", CADENA[1:])
    assert _cli("trace", "verify", cabeza).returncode == 0
    assert _cli("trace", "verify", cabeza, "--desde-genesis").returncode == 2
    mala = _cli("trace", "verify", f, "--huella", "no-es-una-huella")
    assert mala.returncode == 1 and "huella" in mala.stderr and mala.stdout == ""


def test_cli_trace_verify_un_fichero_que_no_existe_sale_con_1(tmp_path):
    r = _cli("trace", "verify", str(tmp_path / "no-existe.jsonl"))
    assert r.returncode == 1 and "no se pudo leer" in r.stderr


def test_cli_trace_verify_un_fichero_con_los_bytes_rotos_no_lanza(tmp_path):
    f = tmp_path / "raro.jsonl"
    f.write_bytes(b"\xff\xfe\x00{")
    r = _cli("trace", "verify", str(f), "--locale", "es")
    assert r.returncode == 3 and "Traceback" not in r.stderr


# ───────────────────────────────────────────────────────────────── paridad con el comprobador de JavaScript

def _node(entrada):
    if _NODE is None:
        if os.environ.get("MATRIXAI_SIN_NODE") == "1":
            pytest.skip("MATRIXAI_SIN_NODE=1: sin node no se comprueba la paridad con JavaScript")
        raise AssertionError("falta node: la paridad Python<->JS no se puede comprobar (MATRIXAI_SIN_NODE=1 para saltarla a propósito)")
    r = subprocess.run([_NODE, str(FIXTURES / "paridad_stdin.mjs")], input=json.dumps(entrada, ensure_ascii=False), capture_output=True,
                       text=True, encoding="utf-8", cwd=RAIZ)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout)


def test_paridad_js_el_corredor_de_vectores_de_node_da_cero_fallos():
    if _NODE is None:
        _node({})  # levanta o salta según MATRIXAI_SIN_NODE
    r = subprocess.run([_NODE, str(FIXTURES / "comprobar_traza.mjs")], capture_output=True, text=True, cwd=RAIZ)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]
    assert f"{len(CASOS)} casos, 0 fallos" in r.stdout


def test_paridad_js_mismo_estado_motivo_y_bloque_en_cada_vector():
    casos = [{"lineas": c["lineas"], "ancla": c.get("ancla"), "esperar_genesis": c.get("esperar_genesis", False)} for c in VECTORES["casos"]]
    js = _node({"modo": "comprobar", "casos": casos})["resultados"]
    for c, j in zip(VECTORES["casos"], js):
        p = _comprobar(c["lineas"], c.get("ancla"), c.get("esperar_genesis", False))
        assert (p["estado"], p["motivo"], p["bloque"]) == (j["estado"], j["motivo"], j["bloque"]), c["nombre"]


def _valor_jcs_al_azar(rng, profundidad=0):
    """Estructuras con lo difícil del JCS: controles, comillas, barras, separadores Unicode, astrales, claves en orden UTF-16."""
    pool = ["a", "ñ", "ó", "Z", "\u0000", "\u0001", "\u001f", "\u007f", " ", " ", '"', "\\", "/", "\b", "\f", "\n", "\r", "\t",
            "ｚ", "퟿", "", "￿", "😀", "𝄞", "\U0010ffff", " ", "__proto__"]
    texto = lambda: "".join(rng.choice(pool) for _ in range(rng.randint(0, 5)))
    eleccion = rng.randint(0, 6 if profundidad < 3 else 3)
    if eleccion == 0:
        return rng.choice([0, 1, 7, 52, 2 ** 31, 2 ** 53 - 1])
    if eleccion == 1:
        return texto()
    if eleccion == 2:
        return rng.choice([True, False, None])
    if eleccion == 3:
        return rng.randint(0, 99)
    if eleccion in (4, 5):
        return {texto(): _valor_jcs_al_azar(rng, profundidad + 1) for _ in range(rng.randint(0, 5))}
    return [_valor_jcs_al_azar(rng, profundidad + 1) for _ in range(rng.randint(0, 4))]


def test_paridad_js_el_JCS_de_JavaScript_es_el_de_jcs_bytes_en_valores_al_azar():
    rng = random.Random(153)
    valores = [_valor_jcs_al_azar(rng) for _ in range(400)]
    js = _node({"modo": "jcs", "valores": valores})["hex"]
    esperados = [jcs_bytes(v).hex() for v in valores]
    assert js == esperados


def _mutaciones(rng, base_lineas, cuantas):
    """Líneas estropeadas de muchas maneras: a nivel de TEXTO (un carácter cambiado, borrado o duplicado) y a nivel de
    ESTRUCTURA (un campo cambiado por un valor de otro tipo), esta última a veces REFIRMADA con la clave buena para que el
    fallo llegue a la coherencia o a la cadena y no se quede en «la firma no coincide»."""
    clave = bytes.fromhex(CLAVE_A["semilla_hex"])
    pool_texto = list('0123456789."{}[],:-eE \t\\u"ñ\r') + ["null", "true", "NaN", "\\ud800", "9007199254740993", "-0", "1.0"]
    valores = [0, 1, 19, 20, 21, 2 ** 53 - 1, 2 ** 53, -1, "", "x", "0" * 64, "A" * 64, None, True, False, [], {}, [1] * 10, [0] * 9, 52.5]
    salida = []
    for _ in range(cuantas):
        lineas = list(base_lineas)
        i = rng.randrange(len(lineas))
        if rng.random() < 0.5:
            t = lineas[i]
            j = rng.randrange(len(t))
            op = rng.choice(["cambiar", "borrar", "duplicar", "insertar"])
            if op == "cambiar":
                t = t[:j] + rng.choice(pool_texto) + t[j + 1:]
            elif op == "borrar":
                t = t[:j] + t[j + 1:]
            elif op == "duplicar":
                t = t[:j] + t[j] + t[j:]
            else:
                t = t[:j] + rng.choice(pool_texto) + t[j:]
            lineas[i] = t
        else:
            b = json.loads(lineas[i])
            destino, ruta = b, []
            while isinstance(destino, (dict, list)) and rng.random() < 0.8:
                clave_ = rng.choice(list(destino)) if isinstance(destino, dict) else rng.randrange(len(destino)) if destino else None
                if clave_ is None:
                    break
                ruta.append(clave_)
                if not isinstance(destino[clave_], (dict, list)):
                    break
                destino = destino[clave_]
            if ruta:
                padre = b
                for p in ruta[:-1]:
                    padre = padre[p]
                if rng.random() < 0.15 and isinstance(padre, dict):
                    del padre[ruta[-1]]
                else:
                    padre[ruta[-1]] = rng.choice(valores)
            if rng.random() < 0.6 and isinstance(b, dict):
                try:
                    b = T.firmar_bloque(clave, b)
                except Exception:  # un bloque tan roto que ni se puede canonicalizar: queda sin refirmar
                    pass
            lineas[i] = json.dumps(b, ensure_ascii=rng.random() < 0.5)
        if rng.random() < 0.1 and len(lineas) > 1:
            del lineas[rng.randrange(len(lineas))]
        if rng.random() < 0.05:
            lineas.reverse()
        salida.append({"lineas": lineas, "ancla": rng.choice([None, None, CLAVE_A["huella"], CLAVE_B["huella"]]),
                       "esperar_genesis": rng.random() < 0.2})
    return salida


def test_paridad_js_600_trazas_estropeadas_al_azar_dan_el_mismo_veredicto_en_python_y_en_javascript():
    """El banco que más discrepancias caza: lo que ni los 65 vectores previeron. Semilla fija, así que es repetible."""
    rng = random.Random(1530)
    casos = (_mutaciones(rng, CADENA, 250) + _mutaciones(rng, CASOS["regresion_valida"]["lineas"], 150)
             + _mutaciones(rng, CASOS["reinicio_sin_previo_con_otra_clave"]["lineas"], 200))
    js = _node({"modo": "comprobar", "casos": casos})["resultados"]
    distintas, estados = [], {}
    for c, j in zip(casos, js):
        p = _comprobar(c["lineas"], c["ancla"], c["esperar_genesis"])
        estados[(p["estado"], p["motivo"])] = estados.get((p["estado"], p["motivo"]), 0) + 1
        if (p["estado"], p["motivo"], p["bloque"]) != (j["estado"], j["motivo"], j["bloque"]):
            distintas.append((c["lineas"], (p["estado"], p["motivo"], p["bloque"]), (j["estado"], j["motivo"], j["bloque"])))
    assert not distintas, distintas[:3]
    # que el banco tenga dientes: salen de todo, no solo de «la firma no coincide»
    assert {("valida", None), ("no_coincide", "firma"), ("no_coincide", "cadena"), ("no_se_pudo_comprobar", "formato")} <= set(estados), estados
    assert estados.get(("no_se_pudo_comprobar", "formato"), 0) > 20 and estados.get(("no_coincide", "cadena"), 0) > 5, estados
