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


def _expandir(x):
    """Una línea de los vectores: texto, o `{"base", "rellenar_con_espacios_hasta"}` (los casos de los topes; mismo criterio en el JS)."""
    if isinstance(x, str):
        return x
    return x["base"] + " " * (x["rellenar_con_espacios_hasta"] - len(x["base"].encode("utf-8")))


def _comprobar_caso(c):
    """Un caso de los vectores, por la entrada que trae: los BYTES del fichero (`fichero_b64`, `fichero_de_espacios`) o las líneas."""
    if "fichero_b64" in c:
        return T.comprobar_fichero(base64.b64decode(c["fichero_b64"]), c.get("ancla"), c.get("esperar_genesis", False))
    if "fichero_de_espacios" in c:
        return T.comprobar_fichero(b" " * c["fichero_de_espacios"], c.get("ancla"), c.get("esperar_genesis", False))
    if "fichero_de_lineas" in c:
        datos = ("\n".join(_expandir(x) for x in c["fichero_de_lineas"]) + ("\n" if c["salto_final"] else "")).encode("utf-8")
        return T.comprobar_fichero(datos, c.get("ancla"), c.get("esperar_genesis", False))
    return _comprobar([_expandir(x) for x in c["lineas"]], c.get("ancla"), c.get("esperar_genesis", False))


def _caso(nombre):
    return _comprobar_caso(CASOS[nombre])


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
    assert len(CASOS) >= 110 and len(CASOS) == len(VECTORES["casos"])


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
            destino[ruta[-1]] = "2026-11-02T23:00:00Z"  # otra hora en punto: `hasta` va al final de una hora
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


def test_cli_trace_verify_un_fichero_que_no_existe_sale_con_1_y_lo_dice_en_su_idioma(tmp_path):
    """rep-c1 (menor 4): el error de stderr respeta `--locale` (antes salía en castellano bajo una CLI en inglés)."""
    en = _cli("trace", "verify", str(tmp_path / "no-existe.jsonl"))
    assert en.returncode == 1 and "could not read" in en.stderr and "no se pudo" not in en.stderr
    es = _cli("trace", "verify", str(tmp_path / "no-existe.jsonl"), "--locale", "es")
    assert es.returncode == 1 and "no se pudo leer" in es.stderr
    mala = _cli("trace", "verify", _fichero(tmp_path, "ok.jsonl", CADENA), "--huella", "zz")
    assert mala.returncode == 1 and "fingerprint" in mala.stderr and "huella es" not in mala.stderr


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
    """Todos los vectores (los 65 de C1 y los de la reparación), por la misma entrada en los dos: líneas o BYTES del fichero.
    Y las `cadenas` de un resultado válido, iguales."""
    js = _node({"modo": "comprobar", "casos": VECTORES["casos"]})["resultados"]
    assert len(js) == len(VECTORES["casos"])
    for c, j in zip(VECTORES["casos"], js):
        p = _comprobar_caso(c)
        assert (p["estado"], p["motivo"], p["bloque"]) == (j["estado"], j["motivo"], j["bloque"]), c["nombre"]
        assert p.get("cadenas") == j["cadenas"], c["nombre"]


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


# ───────────────────────────────────────────────────────────────── reparación de la auditoría de C1 (rep-c1)
# B1 (reinicios), I1 (claves de orden pequeño), I2 (topes de tamaño) y los menores. Cada prueba mata un sabotaje con su nombre.

_REP_B1 = [n for n in CASOS if n.startswith("B1_")]
_REP_I1 = [n for n in CASOS if n.startswith("I1_")]


def test_B1_un_bloque_repetido_en_el_fichero_es_cadena_rota_aunque_vaya_tras_un_reinicio():
    """Arreglo (a): el hash de cada bloque, una sola vez por fichero. Antes, duplicar una cadena de reinicio daba «firma válida»."""
    for nombre in ("B1_cadena_de_reinicio_duplicada", "B1_bloque_de_reinicio_duplicado_al_final", "B1_solo_una_cadena_de_reinicio_dos_veces",
                   "B1_bloque_duplicado_con_otro_formato", "B1_bloque_de_una_hora_duplicado"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", CASOS[nombre]["esperado"]["bloque"]), nombre
        assert "repetido" in r["detalle"]["es"] and "repeated" in r["detalle"]["en"], nombre


def test_B1_las_cadenas_de_una_misma_clave_van_en_el_orden_de_sus_ventanas_firmadas():
    """Arreglo (b): una cadena nueva de una clave no puede empezar antes de la HORA en que acaba la anterior de ESA clave (remate 153:
    `hasta` es el final de una hora, así que en la misma hora vale). No es la hora (DS11): es el orden de lo que la propia clave firmó.
    Igual (desde == hasta) vale; claves distintas no se ordenan."""
    for nombre in ("B1_cadenas_intercambiadas", "B1_cadenas_intercambiadas_con_ancla", "B1_reinicio_que_se_solapa_con_la_anterior"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", CASOS[nombre]["esperado"]["bloque"]), nombre
        assert "antes de la hora en que acaba" in r["detalle"]["es"] and "before the hour in which" in r["detalle"]["en"], nombre
    for nombre in ("B1_tres_cadenas_legitimas", "B1_reinicio_que_empieza_justo_cuando_acaba_la_anterior",
                   "B1_cadena_de_otra_clave_antes_y_fuera_de_hora_NO_se_detecta_sin_ancla"):
        assert _caso(nombre)["estado"] == A.VALIDA, nombre


def test_B1_las_sondas_del_auditor_borrar_reordenar_y_duplicar_cadenas():
    """Las ocho manipulaciones de `p1_cadenas.py` del auditor, reconstruidas con los vectores: lo detectable sale `cadena`, y lo que
    no se puede detectar (borrar ANTES de un reinicio) sigue «válida» y está dicho en el §4."""
    a0, a1, a2 = CASOS["B1_tres_cadenas_legitimas"]["lineas"][:3], CASOS["B1_tres_cadenas_legitimas"]["lineas"][3:5], CASOS["B1_tres_cadenas_legitimas"]["lineas"][5:]
    h = CLAVE_A["huella"]
    esperado = {
        "legitimo": (a0 + a1 + a2, A.VALIDA),
        "borrar_cola_antes_del_reinicio": (a0[:2] + a1 + a2, A.VALIDA),  # no detectable: §4
        "borrar_una_cadena_entera": (a0 + a2, A.VALIDA),                 # no detectable: §4
        "reordenar_cadenas": (a0 + a2 + a1, A.NO_COINCIDE),
        "duplicar_la_ultima": (a0 + a1 + a2 + a2, A.NO_COINCIDE),
        "duplicar_una_entera": (a0 + a1 + a1 + a2, A.NO_COINCIDE),
        "reinicio_al_principio": (a2 + a0 + a1, A.NO_COINCIDE),
        "solo_una_dos_veces": (a1 + a1, A.NO_COINCIDE),
    }
    for nombre, (lineas, estado) in esperado.items():
        for ancla in (None, h):
            r = _comprobar(lineas, ancla)
            assert r["estado"] == estado, (nombre, ancla, r)
            if estado == A.NO_COINCIDE:
                assert r["motivo"] == "cadena", (nombre, r)


def test_B1_lo_que_no_demuestra_y_describir_dicen_que_en_un_reinicio_pueden_faltar_los_bloques_de_antes():
    """Arreglo (c): el §4 (es/en) dice que en cada reinicio pueden faltar los bloques anteriores y que borrar o reordenar solo se
    detecta DENTRO de una cadena; `describir` lista las cadenas (menor 2) y lo repite donde hay reinicios."""
    assert T.AVISO_REINICIO["es"] in T.LO_QUE_NO_DEMUESTRA["es"] and T.AVISO_REINICIO["en"] in T.LO_QUE_NO_DEMUESTRA["en"]
    assert "DENTRO" in T.AVISO_REINICIO["es"] and "INSIDE" in T.AVISO_REINICIO["en"]
    assert "reinicio" in T.AVISO_REINICIO["es"] and "restart" in T.AVISO_REINICIO["en"]
    r = _caso("B1_tres_cadenas_legitimas")
    es, en = "\n".join(T.describir(r, "es")), "\n".join(T.describir(r, "en"))
    assert "3 cadenas (reinicio sin_previo en las líneas 4, 6)" in es and "3 chains (sin_previo restart at lines 4, 6)" in en
    assert "- líneas 1–3: seq 0–2" in es and "- líneas 4–5: seq 0–1" in es and "- lines 6–6: seq 0–0" in en
    assert "En cada reinicio pueden faltar los bloques de antes" in es and "At each restart the earlier blocks may be missing" in en
    sin = "\n".join(T.describir(_caso("cadena_valida"), "es"))
    assert "cadenas (" not in sin and "En cada reinicio" not in sin  # sin reinicios no se habla de cadenas
    for texto in T.LO_QUE_NO_DEMUESTRA["es"] + T.LO_QUE_NO_DEMUESTRA["en"]:
        assert not re.search(r"verificad|verified|garantiz|guarantee", texto, re.I), texto


def test_B1_las_cadenas_del_resultado_valido_dicen_inicio_fin_seq_y_huella():
    r = _caso("B1_cadena_de_otra_clave_antes_y_fuera_de_hora_NO_se_detecta_sin_ancla")
    assert r["cadenas"] == CASOS["B1_cadena_de_otra_clave_antes_y_fuera_de_hora_NO_se_detecta_sin_ancla"]["esperado"]["cadenas"]
    assert _caso("cadena_valida")["cadenas"] == [{"inicio": 0, "fin": 2, "seq_inicial": 0, "seq_final": 2, "huella": CLAVE_A["huella"]}]


def test_menor_7_el_primer_bloque_incoherente_es_cadena_y_el_de_otro_dia_es_valido():
    """Decisión de rep-c1: sin `reinicio`, seq 0 lleva prev de ceros y ningún otro lo lleva. Un fichero que EMPIEZA con seq 0 y un
    prev real, o con seq > 0 y prev de ceros, es `cadena` en su línea 1; el fichero de otro día (seq 2 con el prev del 1) sigue valiendo."""
    for nombre in ("primer_bloque_seq_0_con_prev_que_no_es_de_ceros", "primer_bloque_seq_2_con_prev_de_ceros"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", 0), nombre
        assert "incoherente" in r["detalle"]["es"], nombre
    assert _caso("primer_bloque_seq_2_con_prev_real")["estado"] == A.VALIDA


# ───── I1

def _edwards():
    """Una suma de Edwards de juguete (lenta, solo para las pruebas) para comprobar la lista de puntos de orden pequeño."""
    p, d = T._P, T._D

    def recuperar(y, signo):
        u, v = (y * y - 1) % p, (d * y * y + 1) % p
        x2 = u * pow(v, -1, p) % p
        x = pow(x2, (p + 3) // 8, p)
        if (x * x - x2) % p:
            x = x * pow(2, (p - 1) // 4, p) % p
        if (x * x - x2) % p:
            return None
        return (x if x % 2 == signo else (-x) % p, y)

    def sumar(P1, P2):
        (x1, y1), (x2, y2) = P1, P2
        t = d * x1 * x2 * y1 * y2
        return ((x1 * y2 + y1 * x2) * pow(1 + t, -1, p) % p, (y1 * y2 + x1 * x2) * pow(1 - t, -1, p) % p)

    return recuperar, sumar


def test_I1_la_lista_de_puntos_de_orden_pequeno_son_los_8_y_solo_esos():
    """Las 5 coordenadas y de `_Y_DE_ORDEN_PEQUENO` dan, con los dos signos, exactamente los 8 puntos P con 8·P = identidad."""
    recuperar, sumar = _edwards()
    puntos = set()
    for y in T._Y_DE_ORDEN_PEQUENO:
        for signo in (0, 1):
            P = recuperar(y, signo)
            assert P is not None
            Q = (0, 1)
            for _ in range(8):
                Q = sumar(Q, P)
            assert Q == (0, 1), hex(y)
            puntos.add(P)
    assert len(puntos) == 8
    pub_a = bytes.fromhex(CLAVE_A["publica_hex"])
    assert T._clave_publica_utilizable(pub_a) and T._clave_publica_utilizable(bytes.fromhex(CLAVE_B["publica_hex"]))


def test_I1_una_clave_de_orden_pequeno_o_no_canonica_es_clave_antes_de_verificar(monkeypatch):
    """Con la clave identidad y R = identidad, S = 0, OpenSSL da por buena cualquier firma (medido por la auditoría: 10 «válidas»).
    Ahora es `clave`, ANTES de llamar a la biblioteca: aunque esta aceptara todo, sigue siendo `clave`."""
    for nombre in _REP_I1:
        if nombre.startswith("I1_clave_"):
            assert (_caso(nombre)["estado"], _caso(nombre)["motivo"]) == (A.NO_COINCIDE, "clave"), nombre
    assert len([n for n in _REP_I1 if n.startswith("I1_clave_")]) >= 13
    monkeypatch.setattr(A, "comprobar", lambda *a, **k: True)  # una biblioteca que no rechazara nada
    for nombre in _REP_I1:
        if nombre.startswith("I1_clave_"):
            assert _caso(nombre)["motivo"] == "clave", nombre
    r = _caso("I1_clave_identidad")
    assert "orden pequeño" in r["detalle"]["es"] and "small order" in r["detalle"]["en"]


def test_I1_R_de_orden_pequeno_y_S_mayor_o_igual_que_L_no_son_firma_aunque_la_biblioteca_los_aceptara(monkeypatch):
    """S >= L (S + L sobre una firma buena) y R = identidad son `firma`; se miran en el comprobador, no solo en la biblioteca."""
    for nombre in ("I1_firma_con_S_mas_L", "I1_firma_R_identidad_con_clave_buena"):
        assert (_caso(nombre)["motivo"], _caso(nombre)["bloque"]) == ("firma", 1), nombre
    monkeypatch.setattr(A, "comprobar", lambda *a, **k: True)
    for nombre in ("I1_firma_con_S_mas_L", "I1_firma_R_identidad_con_clave_buena"):
        assert (_caso(nombre)["motivo"], _caso(nombre)["bloque"]) == ("firma", 1), nombre
    assert _caso("cadena_valida")["estado"] == A.VALIDA  # el parche no estropea lo bueno


def test_I1_en_javascript_tambien_con_una_plataforma_que_acepta_cualquier_firma():
    casos = [CASOS[n] for n in _REP_I1]
    for js, c in zip(_node({"modo": "comprobar", "casos": casos, "acepta_todo": True})["resultados"], casos):
        assert (js["estado"], js["motivo"]) == (c["esperado"]["estado"], c["esperado"]["motivo"]), c["nombre"]


# ───── I2

def test_I2_los_topes_son_los_mismos_en_python_y_en_javascript_y_cubren_un_ano_de_bloques():
    js = _node({"modo": "topes"})
    assert (js["TOPE_LINEA"], js["TOPE_FICHERO"]) == (T.TOPE_LINEA, T.TOPE_FICHERO) == (1 << 20, 64 << 20)
    # un año de bloques horarios con 42 columnas pesa 41,5 MB (medido en rep-c1, `anio.py`): cabe con un 60 % de holgura
    assert T.TOPE_FICHERO > 1.5 * 41_537_810


def test_I2_los_topes_se_miran_antes_de_leer_el_json(monkeypatch):
    """Una línea de más de 1 MiB, o un fichero de más de 64 MiB, es `formato` SIN llegar a `json.loads` (que con 100 MB usaba
    ×11 la memoria) y sin decodificar los bytes."""
    llamadas = []
    monkeypatch.setattr(T.json, "loads", lambda *a, **k: llamadas.append(1) or (_ for _ in ()).throw(AssertionError("se leyó el JSON")))
    r = _comprobar([" " * (T.TOPE_LINEA + 1)])
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 0) and "1 MiB" in r["detalle"]["es"]
    r = T.comprobar_fichero(b"[" * (T.TOPE_FICHERO + 1))
    assert (r["motivo"], r["bloque"]) == ("formato", None) and "64 MiB" in r["detalle"]["es"] and "64 MiB" in r["detalle"]["en"]
    r = _comprobar(["ñ" * (T.TOPE_LINEA // 2 + 1)])  # 524.289 caracteres, 1 MiB + 2 bytes: se cuentan BYTES, no caracteres
    assert (r["motivo"], r["bloque"]) == ("formato", 0)
    assert llamadas == []
    for nombre in ("I2_linea_de_1_MiB_mas_1_byte", "I2_linea_de_basura_de_20_MiB", "I2_fichero_de_64_MiB_mas_1_byte", "I2_fichero_en_bytes_de_64_MiB_mas_1"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", CASOS[nombre]["esperado"]["bloque"]), nombre


def test_I2_justo_en_los_topes_cabe():
    assert _caso("I2_linea_de_1_MiB_justo")["estado"] == A.VALIDA
    r = _caso("I2_fichero_de_64_MiB_justo")
    assert (r["estado"], r["bloques"]) == (A.VALIDA, 64)


def test_I2_la_cli_no_lee_mas_alla_del_tope(tmp_path):
    f = tmp_path / "grande.jsonl"
    with open(f, "wb") as fh:
        fh.truncate(T.TOPE_FICHERO + 10)  # disperso: no gasta disco
    r = _cli("trace", "verify", str(f), "--locale", "es")
    assert r.returncode == 3 and "64 MiB" in r.stdout and "Traceback" not in r.stderr


def test_I2_el_lector_de_javascript_no_construye_cadenas_caracter_a_caracter():
    """En JS, `s += c` con una cadena de millones de caracteres crea millones de cadenas; el lector usa `slice`."""
    fuente = (FIXTURES / "comprobar_traza.mjs").read_text(encoding="utf-8")
    lector = fuente[fuente.index("function leerJSON"):fuente.index("// ───────────────────────── JCS")]
    assert "s += c;" not in lector and "texto.slice(ini, i)" in lector


# ───── menores

def test_menor_1_por_bytes_el_BOM_y_el_UTF8_invalido_son_formato_igual_en_los_dos():
    for nombre in ("bytes_con_BOM", "bytes_utf8_invalido", "bytes_utf8_sobrelargo", "bytes_sustituto_codificado"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", CASOS[nombre]["esperado"]["bloque"]), nombre
    assert _caso("bytes_cadena_valida")["estado"] == A.VALIDA
    fuente = (FIXTURES / "comprobar_traza.mjs").read_text(encoding="utf-8")
    assert "new TextDecoder('utf-8', { fatal: true, ignoreBOM: true })" in fuente


def test_menor_4_la_cli_con_json_lleva_lo_que_no_demuestra_y_el_aviso_de_la_cola(tmp_path):
    f = _fichero(tmp_path, "ok.jsonl", CADENA)
    en = json.loads(_cli("trace", "verify", f, "--json").stdout)
    assert en["lo_que_no_demuestra"] == T.LO_QUE_NO_DEMUESTRA["en"] and en["aviso_cola"] == T.AVISO_COLA["en"]
    es = json.loads(_cli("trace", "verify", f, "--json", "--locale", "es").stdout)
    assert es["lo_que_no_demuestra"] == T.LO_QUE_NO_DEMUESTRA["es"] and es["aviso_cola"] == T.AVISO_COLA["es"]
    assert es["estado"] == "valida" and es["cadenas"][0]["seq_final"] == 2


def test_menor_3_el_consejo_de_cotejar_la_cola_dice_con_que():
    """El consejo «compara el último bloque» se queda porque C3 se obliga (anotado en el contrato) a anunciar `ultimo_seq` y
    `ultimo_hash` en `/health`: el texto dice qué comparar y dónde."""
    assert "/health" in T.AVISO_COLA["es"] and "seq y hash" in T.AVISO_COLA["es"]
    assert "/health" in T.AVISO_COLA["en"] and "seq and hash" in T.AVISO_COLA["en"]


def test_menor_5_cli_reference_no_promete_de_mas():
    """CLI_REFERENCE (es/en) ya no dice que un bloque borrado o reordenado sale siempre «no coincide»: dice DENTRO de una cadena,
    y que en un reinicio pueden faltar bloques; y su ejemplo es la salida de hoy."""
    for idioma, dentro, reinicio in (("es", "dentro de una cadena", "reinicio"), ("en", "inside a chain", "restart")):
        texto = (RAIZ / "docs" / idioma / "api" / "CLI_REFERENCE.md").read_text(encoding="utf-8")
        seccion = texto[texto.index("### matrixai trace verify"):]
        seccion = seccion[:seccion.index("\n---")]
        plano = " ".join(seccion.replace("**", "").split())
        assert dentro in plano and reinicio in plano, idioma
        assert "64 MiB" in seccion and "1 MiB" in seccion, idioma
        for frase in T.LO_QUE_NO_DEMUESTRA[idioma]:
            assert frase in seccion, (idioma, frase[:40])


# ───── remate de C1 (re-auditoría: I2 y menores 1, 2 y el cosmético)

@pytest.mark.parametrize("patron", [b"ab\n", b"\n", b"abcdefg\n"])
def test_remate_I2_un_fichero_de_lineas_diminutas_no_crea_la_lista_de_lineas(patron):
    """Re-auditoría, I2: 64 MiB de líneas diminutas gastaban 535–1.284 MB en Python porque se partía el fichero entero (`split`)
    antes de mirar los topes: 22 millones de objetos `bytes`. Ahora los topes saltan de ventana en ventana y cada línea se
    decodifica al leerla. Con 8 MiB (2,8 millones de líneas), el pico de lo asignado por Python tiene que quedar POR DEBAJO del
    tamaño del fichero (medido: ~2,5 MB, los trozos de 1 MiB de la comprobación de UTF-8; con `split`, 68–122 MB). Mismo veredicto."""
    import tracemalloc
    datos = patron * ((8 << 20) // len(patron))
    tracemalloc.start()
    try:
        r = T.comprobar_fichero(datos)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 0)
    assert pico < 6 << 20, f"pico de {pico / 2**20:.1f} MiB con un fichero de 8 MiB: ¿se vuelve a crear la lista de líneas?"


def test_remate_I2_un_fichero_valido_guarda_poco_por_bloque():
    """Lo peor tras leer por líneas es un fichero de bloques VÁLIDOS lo más pequeños posible (incompletos, 672 bytes): de cada uno se
    guarda solo lo que la cadena necesita. Medido con 64 MiB de ellos (100.029 bloques, `~/.agentes/153/remate-c1/`): 222 → 141 MB
    en Python. Aquí, el pico por bloque con 600 bloques, con holgura sobre lo medido (~1,3 KB, con los trozos de UTF-8 dentro)."""
    import tracemalloc
    from matrixai.pipelines.firma_asimetrica import cargar_clave_privada, huella_de_clave_publica
    k = cargar_clave_privada(hashlib.sha256(b"remate-c1 PRUEBA pico por bloque").digest())
    pub = k.public_key().public_bytes_raw().hex()
    prev, lineas = T.GENESIS, []
    for seq in range(600):
        b = {"formato": T.FORMATO, "seq": seq, "prev": prev,
             "servicio": {"id": "s", "huella_clave": huella_de_clave_publica(k), "clave_publica": pub, "huella_modelo": "0" * 64},
             "ventana": {"desde": "2026-11-03T00:00:00Z", "hasta": "2026-11-03T01:00:00Z"}, "n": 1, "k_minimo": 20,
             "incompleto": True, "reinicio": None, "salida": None, "columnas": None}
        b = T.firmar_bloque(k, b)
        lineas.append(_linea(b)); prev = T.hash_de_bloque(b)
    datos = ("\n".join(lineas) + "\n").encode("utf-8")
    tracemalloc.start()
    try:
        r = T.comprobar_fichero(datos)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (r["estado"], r["bloques"]) == (A.VALIDA, 600)
    assert pico < (2 << 20) + 600 * 2048, f"{pico / 600:.0f} bytes por bloque"


def test_remate_I2_los_topes_por_bytes_saltando_ventanas_dan_lo_mismo_que_mirar_cada_linea(monkeypatch):
    """`_fuera_de_tope_en_bytes` (64 pasos para 64 MiB) contra la definición (cada línea de `split`, quitando la última vacía, más
    su salto), con topes diminutos para que los bordes salgan a menudo: 4.000 contenidos al azar."""
    monkeypatch.setattr(T, "TOPE_LINEA", 5)
    monkeypatch.setattr(T, "TOPE_FICHERO", 24)
    rng = random.Random(15301)
    for _ in range(4000):
        datos = bytes(rng.choice(b"\nab") if rng.random() < 0.5 else rng.choice(b"\n" + b"x" * 6) for _ in range(rng.randrange(0, 30)))
        partes = datos.split(b"\n")
        if partes and partes[-1] == b"":
            partes.pop()
        assert T._fuera_de_tope_en_bytes(datos) == T._fuera_de_tope(len(p) for p in partes), datos


def test_remate_I2_paridad_js_los_topes_por_bytes_en_ficheros_al_azar():
    """Los mismos topes por bytes en el JS (`fueraDeTopeEnBytes`, `lastIndexOf`): ficheros con líneas de 1 MiB ± 2 al azar
    entre líneas cortas, mismo estado, motivo y bloque que Python."""
    rng = random.Random(15302)
    casos = []
    for _ in range(24):
        lineas = []
        for _ in range(rng.randrange(1, 6)):
            if rng.random() < 0.5:
                lineas.append(rng.choice(["", "x", "[", CADENA[0]]))
            else:
                lineas.append({"base": rng.choice(["[", CADENA[0]]), "rellenar_con_espacios_hasta": T.TOPE_LINEA + rng.randrange(-2, 3)})
        casos.append({"fichero_de_lineas": lineas, "salto_final": rng.random() < 0.5})
    js = _node({"modo": "comprobar", "casos": casos})["resultados"]
    vistos = set()
    for c, j in zip(casos, js):
        r = _comprobar_caso(c)
        assert (j["estado"], j["motivo"], j["bloque"]) == (r["estado"], r["motivo"], r["bloque"]), c["fichero_de_lineas"]
        vistos.add((r["motivo"], r["bloque"] is None))
    assert ("formato", False) in vistos  # alguno cayó en el tope de línea


def test_remate_I2_javascript_no_parte_el_fichero_ni_lo_decodifica_entero():
    """Re-auditoría, I2 en Node: 64 MiB de líneas diminutas, 502–1.013 MB, por `TextDecoder.decode` del fichero entero y
    `split('\\n')`. Un espía: ningún `split('\\n')` de más de 1 MiB, ningún `decode` de más de 1 MiB; y el veredicto, el de siempre."""
    if _NODE is None:
        pytest.skip("sin node") if os.environ.get("MATRIXAI_SIN_NODE") == "1" else pytest.fail("falta node")
    guion = textwrap.dedent(f"""
        const partir = String.prototype.split;
        String.prototype.split = function (sep, ...r) {{
          if (sep === '\\n' && this.length > (1 << 20)) throw new Error('split de ' + this.length);
          return partir.call(this, sep, ...r);
        }};
        const decodificar = TextDecoder.prototype.decode;
        TextDecoder.prototype.decode = function (x, ...r) {{
          if (x && x.byteLength > (1 << 20)) throw new Error('decode de ' + x.byteLength);
          return decodificar.call(this, x, ...r);
        }};
        const {{ comprobarFichero }} = await import({json.dumps((FIXTURES / "comprobar_traza.mjs").as_uri())});
        const salida = [];
        for (const p of ['ab\\n', '\\n', 'abcdefg\\n']) {{
          const r = await comprobarFichero(new Uint8Array(Buffer.alloc(8 << 20, p)));
          salida.push([r.estado, r.motivo, r.bloque]);
        }}
        const valida = new TextEncoder().encode({json.dumps(chr(10).join(CADENA) + chr(10))});
        const r = await comprobarFichero(valida);
        salida.push([r.estado, r.bloques]);
        console.log(JSON.stringify(salida));
    """)
    p = subprocess.run([_NODE, "--input-type=module", "-e", guion], capture_output=True, text=True, cwd=RAIZ)
    assert p.returncode == 0, p.stderr[-800:]
    assert json.loads(p.stdout) == [["no_se_pudo_comprobar", "formato", 0]] * 3 + [["valida", 3]]


def test_remate_menor_1_S_igual_a_L_exacto_no_es_firma_aunque_la_biblioteca_la_aceptara(monkeypatch):
    """Re-auditoría, menor 1: S = L es el borde de S < L (RFC 8032 §5.1.7). Sin este vector, cambiar `<` por `<=` dejaba la
    suite verde. Con una biblioteca que aceptara todo, sigue siendo `firma`, en Python y en JS."""
    c = CASOS["I1_firma_con_S_igual_a_L"]
    S = int.from_bytes(base64.b64decode(json.loads(c["lineas"][1])["firma"]["valor"])[32:], "little")
    assert S == 2 ** 252 + 27742317777372353535851937790883648493  # el vector es lo que dice
    assert (_caso(c["nombre"])["motivo"], _caso(c["nombre"])["bloque"]) == ("firma", 1)
    monkeypatch.setattr(A, "comprobar", lambda *a, **k: True)
    assert (_caso(c["nombre"])["motivo"], _caso(c["nombre"])["bloque"]) == ("firma", 1)
    js = _node({"modo": "comprobar", "casos": [c], "acepta_todo": True})["resultados"][0]
    assert (js["estado"], js["motivo"], js["bloque"]) == (A.NO_COINCIDE, "firma", 1)


def test_remate_menor_2_desde_genesis_exige_que_la_primera_linea_no_sea_un_reinicio(tmp_path):
    """Re-auditoría, menor 2: borrada la cadena del génesis entera, el fichero empieza en un reinicio `sin_previo` (seq 0, prev de
    ceros) y `--desde-genesis --huella` decía «válida». Ahora el génesis es seq 0, prev de ceros y SIN reinicio."""
    assert _caso("B1_borrada_la_cadena_del_genesis_NO_se_detecta_sin_esperar_genesis")["estado"] == A.VALIDA
    r = _caso("B1_borrada_la_cadena_del_genesis_esperando_genesis")
    assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", 0)
    assert "reinicio" in r["detalle"]["es"] and "restart" in r["detalle"]["en"]
    assert _caso("cadena_valida_esperando_genesis")["estado"] == A.VALIDA  # el génesis de verdad sigue pasando
    tres = CASOS["B1_tres_cadenas_legitimas"]["lineas"]
    sin_genesis = _fichero(tmp_path, "sin_genesis.jsonl", tres[3:])
    huella = json.loads(tres[0])["servicio"]["huella_clave"]
    assert _cli("trace", "verify", sin_genesis, "--huella", huella).returncode == 0
    assert _cli("trace", "verify", sin_genesis, "--huella", huella, "--desde-genesis").returncode == 2
    assert _cli("trace", "verify", _fichero(tmp_path, "tres.jsonl", tres), "--huella", huella, "--desde-genesis").returncode == 0
    for idioma, frase in (("es", "antes del primer reinicio"), ("en", "before the first restart")):
        assert frase in (RAIZ / "docs" / idioma / "api" / "CLI_REFERENCE.md").read_text(encoding="utf-8")


def test_remate_cosmetico_describir_concuerda_en_singular():
    """«1 bloques», «1 cadenas» y «en las líneas 1» con un solo bloque, una sola cadena o un solo reinicio."""
    un = T.comprobar_traza(CADENA[:1])
    es, en = T.describir(un, "es"), T.describir(un, "en")
    assert "1 bloque, del seq 0 al 0" in es and "1 block, from seq 0 to 0" in en
    reinicio = T.comprobar_traza(CASOS["B1_tres_cadenas_legitimas"]["lineas"][5:])  # la cadena A2: un bloque, un reinicio
    es, en = "\n".join(T.describir(reinicio, "es")), "\n".join(T.describir(reinicio, "en"))
    assert "1 bloque en 1 cadena\n" in es and "1 block in 1 chain\n" in en
    assert "1 cadena (reinicio sin_previo en la línea 1):" in es and "1 chain (sin_previo restart at line 1):" in en
    assert not re.search(r"\b1 (bloques|cadenas|blocks|chains)\b", es + en)
    tres = "\n".join(T.describir(_caso("B1_tres_cadenas_legitimas"), "es"))
    assert "6 bloques en 3 cadenas" in tres and "en las líneas 4, 6" in tres


def test_c2_menor_las_huellas_salen_sin_repetir_y_en_orden_de_aparicion_en_python_y_en_js():
    """153-C2, menor de la 3.ª auditoría de C1: `huellas` se arma con un dict/Set (era un `not in` sobre una lista, cuadrático: 77 s
    de 173 con 64 MiB de bloques con una clave cada uno). Pero el ORDEN de `huellas` es parte de la salida: la primera aparición de
    cada clave, sin repetir. Se comprueba contra una cuenta independiente (leyendo la huella de cada línea) en TODOS los vectores
    válidos, y el JS da la misma lista."""
    validos = [c for c in VECTORES["casos"] if _comprobar_caso(c)["estado"] == A.VALIDA]
    assert len(validos) >= 10
    js = _node({"modo": "comprobar", "casos": validos})["resultados"]
    varias_claves = 0
    for c, j in zip(validos, js):
        r = _comprobar_caso(c)
        if any(k in c for k in ("fichero_b64", "fichero_de_espacios", "fichero_de_lineas")):
            continue  # entran por los bytes del fichero, no por una lista de líneas
        lineas = [_expandir(x) for x in c["lineas"]]
        esperado = []
        for linea in lineas:
            h = json.loads(linea)["servicio"]["huella_clave"]
            if h not in esperado:
                esperado.append(h)
        assert r["huellas"] == esperado == j["huellas"], c["nombre"]
        varias_claves += len(esperado) > 1
    assert varias_claves >= 1, "ningún vector válido mezcla claves: la prueba no mide el orden"


def test_c2_menor_el_orden_es_el_de_la_primera_aparicion_no_el_alfabetico_ni_con_repetidos():
    """El orden de `huellas` tiene que distinguirse de `sorted` (con las claves de los vectores el orden de aparición coincide con el
    alfabético y un `sorted` pasaba): se construye una traza con la clave de huella MAYOR primero, luego la menor, y la mayor otra vez
    (tres cadenas, `sin_previo`). Salen las dos huellas, una vez cada una, la mayor delante; igual en JS."""
    mayor, menor = sorted((CLAVE_A, CLAVE_B), key=lambda k: k["huella"], reverse=True)
    base = json.loads(CADENA[0])

    def bloque(clave, desde, primero):
        b = {k: v for k, v in base.items() if k != "firma"}
        b["servicio"] = {**base["servicio"], "huella_clave": clave["huella"], "clave_publica": clave["publica_hex"]}
        b.update(seq=0, prev=T.GENESIS, ventana={"desde": desde, "hasta": desde[:11] + f"{int(desde[11:13]) + 1:02d}:00:00Z"}, n=1, incompleto=True, salida=None, columnas=None,
                 reinicio=None if primero else "sin_previo")
        return _linea(_firmado_con(clave, b))

    lineas = [bloque(mayor, "2026-11-02T10:00:00Z", True), bloque(menor, "2026-11-02T11:00:00Z", False),
              bloque(mayor, "2026-11-02T12:00:00Z", False)]
    r = _comprobar(lineas)
    assert r["estado"] == A.VALIDA, r.get("detalle")
    assert r["huellas"] == [mayor["huella"], menor["huella"]] != sorted(r["huellas"])
    assert _node({"modo": "comprobar", "casos": [{"lineas": lineas}]})["resultados"][0]["huellas"] == r["huellas"]


# ───────────────────────────────────────────────────────────────── 153-C4: `ventana.desde` es el inicio de una HORA UTC

_DESDE_CON_HORA_EXACTA = ("ventana_desde_con_segundos", "ventana_desde_con_minutos", "ventana_desde_a_mitad_de_hora_sin_segundos")


def test_c4_desde_que_no_es_el_inicio_de_una_hora_utc_no_es_v1_firmado_o_no():
    """Decisión de Roberto (07-11, opción A): `ventana.desde` es el inicio de la hora UTC de la primera petición contada, nunca su segundo
    exacto (la hora de la petición de UNA persona). Un bloque con minutos o segundos, aunque la clave buena lo firme, es `formato`."""
    for nombre in _DESDE_CON_HORA_EXACTA:
        r = _comprobar_caso(CASOS[nombre])
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 1), (nombre, r)
        assert "inicio de una hora UTC" in r["detalle"]["es"] and "start of a UTC hour" in r["detalle"]["en"], r


def test_c4_todos_los_desde_de_los_vectores_validos_estan_en_punto():
    """El generador no produce ni un `desde` con minutos o segundos en los bloques que dan «válida»: lo que el escritor escribe y lo que el
    comprobador acepta son la misma regla."""
    visto = 0
    for c in VECTORES["casos"]:
        if c["esperado"].get("estado") != A.VALIDA:
            continue
        for linea in c["lineas"]:
            if isinstance(linea, str) and linea.startswith("{"):
                desde = json.loads(linea)["ventana"]["desde"]
                assert desde.endswith(":00:00Z"), (c["nombre"], desde)
                visto += 1
    assert visto > 20


def test_c4_paridad_js_los_desde_con_hora_exacta_dan_formato_tambien_en_javascript():
    js = _node({"modo": "comprobar", "casos": [CASOS[n] for n in _DESDE_CON_HORA_EXACTA]})["resultados"]
    assert len(js) == len(_DESDE_CON_HORA_EXACTA)
    for nombre, r in zip(_DESDE_CON_HORA_EXACTA, js):
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 1), (nombre, r)


# ───────────────────────────────────────── remate de la auditoría de C4–C7: `hasta` al final de su hora (Roberto, 07-11, A) y M4

_HASTA_QUE_NO_VALE = ("ventana_hasta_con_minutos", "ventana_hasta_con_segundos", "ventana_vacia_desde_igual_a_hasta")
_FECHAS_QUE_NO_SON = ("ventana_30_de_febrero", "ventana_hora_24", "ventana_29_de_febrero_de_2100_no_es_bisiesto")
_MIXTAS = ("M4_clave_de_orden_mixto_2_con_h_multiplo_de_8", "M4_clave_de_orden_mixto_4_con_h_multiplo_de_8",
           "M4_clave_de_orden_mixto_8_con_h_multiplo_de_8", "M4_clave_de_orden_mixto_8_con_h_no_multiplo_de_8_en_el_segundo")


def test_remate_hasta_que_no_es_el_final_de_una_hora_utc_no_es_v1_y_una_ventana_nunca_esta_vacia():
    """Decisión de Roberto (07-11, A): `hasta` es el FINAL de la hora UTC de la última petición contada (la hora en punto siguiente). Con
    minutos o segundos, aunque la clave buena lo firme, es `formato`; y como `desde` es el inicio de la hora de la primera, `desde == hasta`
    no lo escribe nadie: también `formato`. En Python y en JS, el mismo bloque."""
    js = _node({"modo": "comprobar", "casos": [CASOS[n] for n in _HASTA_QUE_NO_VALE]})["resultados"]
    for nombre, r_js in zip(_HASTA_QUE_NO_VALE, js):
        r = _comprobar_caso(CASOS[nombre])
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 1) == (r_js["estado"], r_js["motivo"], r_js["bloque"]), (nombre, r, r_js)
    assert "final de una hora UTC" in _caso("ventana_hasta_con_minutos")["detalle"]["es"]
    assert "end of a UTC hour" in _caso("ventana_hasta_con_segundos")["detalle"]["en"]
    assert "ventana vacía" in _caso("ventana_vacia_desde_igual_a_hasta")["detalle"]["es"]


def test_remate_todos_los_hasta_de_los_vectores_validos_estan_en_punto_y_por_delante_de_desde():
    visto = 0
    for c in VECTORES["casos"]:
        if c["esperado"].get("estado") != A.VALIDA:
            continue
        for linea in c["lineas"]:
            if isinstance(linea, str) and linea.startswith("{"):
                v = json.loads(linea)["ventana"]
                assert v["hasta"].endswith(":00:00Z") and T.segundos_utc(v["hasta"]) - T.segundos_utc(v["desde"]) >= 3600, (c["nombre"], v)
                visto += 1
    assert visto > 20


def test_remate_una_cadena_nueva_puede_empezar_en_la_hora_en_que_acaba_la_anterior_y_no_antes():
    """Con horas enteras, el servicio que se cae a las 12:20 (última línea a medias) y vuelve a las 12:25 abre una cadena con `desde` 12:00
    cuando la anterior acaba a las 13:00: con la regla de antes (`desde` >= `hasta`) eso era «la firma no coincide» en una traza HONRADA.
    La regla es `desde` >= `hasta` anterior − 1 h; también cruzando la medianoche (la cuenta es de fechas, no de texto)."""
    for nombre in ("B1_reinicio_en_la_misma_hora_en_que_acaba_la_anterior", "B1_reinicio_en_la_misma_hora_pasada_la_medianoche"):
        assert _caso(nombre)["estado"] == A.VALIDA, (nombre, _caso(nombre))
    for nombre in ("B1_reinicio_una_hora_antes_pasada_la_medianoche", "B1_reinicio_que_se_solapa_con_la_anterior"):
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "cadena", CASOS[nombre]["esperado"]["bloque"]), (nombre, r)
        assert "antes de la hora en que acaba" in r["detalle"]["es"], r
    nombres = ("B1_reinicio_en_la_misma_hora_en_que_acaba_la_anterior", "B1_reinicio_en_la_misma_hora_pasada_la_medianoche",
               "B1_reinicio_una_hora_antes_pasada_la_medianoche", "B1_reinicio_que_se_solapa_con_la_anterior")
    js = _node({"modo": "comprobar", "casos": [CASOS[n] for n in nombres]})["resultados"]
    assert [(r["estado"], r["motivo"]) for r in js] == [(A.VALIDA, None), (A.VALIDA, None), (A.NO_COINCIDE, "cadena"), (A.NO_COINCIDE, "cadena")]


def test_remate_las_fechas_de_la_ventana_son_del_calendario_y_su_cuenta_es_la_misma_en_python_y_en_js():
    """La regla entre cadenas resta una hora: necesita fechas reales. `segundos_utc` (Python) y `segundosUtc` (JS) son la misma aritmética
    propia; aquí, contra `calendar.timegm` en 3.000 fechas al azar de 0000 a 9999 y en los bordes, y `None` en las que no existen."""
    import calendar
    rng = random.Random(15308)
    fechas = ["0000-01-01T00:00:00Z", "0000-02-29T23:59:59Z", "1969-12-31T23:59:59Z", "1970-01-01T00:00:00Z", "2000-02-29T12:00:00Z",
              "2028-02-29T23:00:00Z", "9999-12-31T23:59:59Z"]
    for _ in range(3000):
        a, m = rng.randrange(0, 10000), rng.randrange(1, 13)
        d = rng.randrange(1, calendar.monthrange(max(a, 1) if a else 2000, m)[1] + 1)   # el año 0000 es bisiesto, como el 2000
        fechas.append(f"{a:04d}-{m:02d}-{d:02d}T{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}Z")
    malas = ["2026-02-29T00:00:00Z", "2100-02-29T00:00:00Z", "2026-13-01T00:00:00Z", "2026-00-10T00:00:00Z", "2026-04-31T00:00:00Z",
             "2026-01-00T00:00:00Z", "2026-11-02T24:00:00Z", "2026-11-02T23:60:00Z", "2026-11-02T23:00:60Z", "2026-02-30T10:00:00Z"]
    py = [T.segundos_utc(f) for f in fechas + malas]
    for f, s in zip(fechas, py):
        a = int(f[:4])
        esperado = calendar.timegm((a if a else 2000, *map(int, (f[5:7], f[8:10], f[11:13], f[14:16], f[17:19])), 0, 0, 0))
        if not a:   # timegm no admite el año 0: se compara con el 2000 (mismo ciclo de 400 años), restando 2000 años
            esperado -= 146097 * 5 * 86400
        assert s == esperado, (f, s, esperado)
    assert py[len(fechas):] == [None] * len(malas)
    assert _node({"modo": "fechas", "fechas": fechas + malas})["segundos"] == py
    for nombre in _FECHAS_QUE_NO_SON:
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_SE_PUDO_COMPROBAR, "formato", 1), (nombre, r)
        assert "fecha del calendario" in r["detalle"]["es"], r
    assert _caso("ventana_29_de_febrero_bisiesto")["estado"] == A.VALIDA
    assert _caso("ventana_de_varias_horas")["estado"] == A.VALIDA


def test_M4_una_clave_de_orden_mixto_es_clave_antes_de_verificar_aunque_openssl_la_aceptara():
    """M4 de la auditoría de C4–C7 (deuda del C6): con A' = [a]B + T (T de orden 2, 4 u 8), OpenSSL (Python, Node), WebCrypto de cada
    navegador y @noble decían cosas distintas de la misma traza. Ahora `[L]A ≠ identidad` es `clave` en el primer bloque, ANTES de llamar a
    la biblioteca. El diente: en los vectores con h ≡ 0 (mod 8) la biblioteca de aquí (OpenSSL) da la firma por buena."""
    for nombre in _MIXTAS:
        r = _caso(nombre)
        assert (r["estado"], r["motivo"], r["bloque"]) == (A.NO_COINCIDE, "clave", 0), (nombre, r)
        assert "orden mixto" in r["detalle"]["es"] and "mixed order" in r["detalle"]["en"], r
        if "h_multiplo" in nombre:
            b = json.loads(CASOS[nombre]["lineas"][0])
            assert A.comprobar(bytes.fromhex(b["servicio"]["clave_publica"]), T.mensaje_firmado(b), b["firma"]["valor"]) is True, nombre
    for acepta_todo in (False, True):
        js = _node({"modo": "comprobar", "casos": [CASOS[n] for n in _MIXTAS], "acepta_todo": acepta_todo})["resultados"]
        assert [(r["estado"], r["motivo"], r["bloque"]) for r in js] == [(A.NO_COINCIDE, "clave", 0)] * len(_MIXTAS), (acepta_todo, js)


def test_M4_las_claves_de_verdad_siguen_valiendo():
    """El otro lado: la comprobación de orden no rechaza una clave Ed25519 normal (las de `cryptography`, como la del escritor)."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    for _ in range(64):
        crudo = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        assert T._clave_publica_utilizable(crudo), crudo.hex()
    for c in (CLAVE_A, CLAVE_B):
        assert T._clave_publica_utilizable(bytes.fromhex(c["publica_hex"]))
