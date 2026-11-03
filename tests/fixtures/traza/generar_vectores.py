#!/usr/bin/env python3
"""153-C1 · Genera `vectores_traza_v1.json`: cadenas válidas de `matrixai.traza/v1` y casos que deben fallar.

Es el fichero que comparten el núcleo (Python), `comprobar_traza.mjs` (JS) y, en C6, `/verificar`. Para
regenerarlo:  python3 tests/fixtures/traza/generar_vectores.py   (escribe el JSON junto a este guion).

CLAVES DE PRUEBA: las dos semillas de abajo son CONSTANTES PÚBLICAS escritas a propósito, y salen de una frase
que lo dice. No protegen nada, no son de ninguna instalación y NO deben usarse nunca para firmar nada real. Este
guion no lee ~/.config ni ninguna clave de nadie.

Ed25519 es determinista (RFC 8032): con la misma semilla y los mismos bytes sale SIEMPRE la misma firma, así que
el JSON es reproducible bit a bit (se comprueba al final de `main`, y una prueba lo repite).

Los veredictos esperados están ESCRITOS A MANO aquí: no salen del comprobador (sería circular). El prefijo de
firma y el hash del bloque se declaran AQUÍ otra vez, a propósito: si el comprobador del núcleo cambiara el suyo,
los vectores ya generados dejarían de cuadrar y se vería.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys

sys.dont_write_bytecode = True
# el núcleo de ESTE árbol (no el que el .pth de la máquina resuelva): el repo está tres niveles por encima
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from matrixai.continual.drift import recuentos_en_casillas  # noqa: E402
from matrixai.pipelines.canonical import jcs_bytes, jcs_canonical  # noqa: E402
from matrixai.pipelines.firma_asimetrica import (  # noqa: E402
    cargar_clave_privada, firmar, huella_de_clave_publica)

PREFIJO_FIRMA = b"matrixai.traza.firma/v1\n"
GENESIS = "0" * 64
K = 20

SEMILLA_A = hashlib.sha256(b"matrixai.traza/v1 vectores 153-C1: CLAVE DE PRUEBA A, no usar").hexdigest()
SEMILLA_B = hashlib.sha256(b"matrixai.traza/v1 vectores 153-C1: CLAVE DE PRUEBA B, no usar").hexdigest()

HUELLA_MODELO = hashlib.sha256(b"checkpoint de prueba").hexdigest()
SERVICIO_ID = "servicio-ñandú-de-prueba"  # no ASCII a propósito: UTF-8 del canonicalizador

# columnas vigiladas con la referencia de ENTRENAMIENTO (mín, máx). Nombres no ASCII y con orden UTF-16 distinto del
# orden por puntos de código ("😀_cara" va ANTES que "ｚ_ancho" en UTF-16 y DESPUÉS por puntos de código)
COLUMNAS = {
    "edad_años": ("numerica", 18.0, 90.0),
    "Z_presión": ("numerica", 0.0, 1.0),
    "a": ("numerica", -5.0, 5.0),
    "ｚ_ancho": ("numerica", 0.0, 10.0),
    "😀_cara": ("numerica", -1.0, 1.0),
    "sexo": ("categorica", ["F", "M"]),
}
COLUMNAS_REGRESION = {"__proto__": ("numerica", 0.0, 1.0), "sexo": ("categorica", ["F", "M"])}
OBJETIVO = (0.0, 100.0)  # referencia del objetivo en regresión


def clave(semilla_hex):
    k = cargar_clave_privada(bytes.fromhex(semilla_hex))
    return k, k.public_key().public_bytes_raw().hex(), huella_de_clave_publica(k)


def hash_de_bloque(b):
    return hashlib.sha256(jcs_bytes(b)).hexdigest()


def mensaje_firmado(b):
    return PREFIJO_FIRMA + jcs_bytes({k: v for k, v in b.items() if k != "firma"})


def firmado(k, b):
    b = {x: v for x, v in b.items() if x != "firma"}
    b["firma"] = {"alg": "ed25519", "valor": firmar(k, mensaje_firmado(b))}
    return b


def linea(b):
    return jcs_canonical(b)


def ventana(rng, n, dist, definiciones, regresion):
    """n peticiones sintéticas -> SOLO los recuentos (los valores no salen de aquí)."""
    preds, cols = [], {nombre: [] for nombre in definiciones}
    for _ in range(n):
        preds.append(min(max(rng.betavariate(2 + dist, 3), 0.0), 1.0) if not regresion else rng.gauss(50 + 20 * dist, 25))
        for nombre, d in definiciones.items():
            if d[0] == "numerica":
                lo, hi = d[1], d[2]
                cols[nombre].append(None if rng.random() < 0.05 else lo + (hi - lo) * min(max(rng.gauss(0.5 + 0.2 * dist, 0.25), -0.1), 1.1))
            else:
                cols[nombre].append(None if rng.random() < 0.03 else rng.choice(["F", "M", "M", "F", "X?"]))
    if regresion:
        bins, _, fuera = recuentos_en_casillas(preds, OBJETIVO[0], OBJETIVO[1])
        salida = {"probabilidad": None, "etiquetas": None, "valor": {"bins": bins, "fuera": fuera}}
    else:
        prob_bins, etiquetas = [0] * 10, {"no": 0, "sí": 0}
        for p in preds:
            prob_bins[min(int(p * 10), 9)] += 1
            etiquetas["sí" if p >= 0.5 else "no"] += 1
        salida = {"probabilidad": prob_bins, "etiquetas": etiquetas, "valor": None}
    columnas = {}
    for nombre, d in definiciones.items():
        if d[0] == "numerica":
            bins, faltantes, fuera = recuentos_en_casillas(cols[nombre], d[1], d[2])
            columnas[nombre] = {"tipo": "numerica", "bins": bins, "faltantes": faltantes, "fuera": fuera}
        else:
            conocidas = set(d[1])
            columnas[nombre] = {"tipo": "categorica",
                                "nuevas": sum(1 for v in cols[nombre] if v is not None and v not in conocidas),
                                "faltantes": sum(1 for v in cols[nombre] if v is None)}
    return salida, columnas


def bloque(k, pub_hex, huella, seq, prev, desde, hasta, rng, n=40, dist=0.0, incompleto=False, reinicio=None,
           k_minimo=K, regresion=False, servicio_id=SERVICIO_ID):
    definiciones = COLUMNAS_REGRESION if regresion else COLUMNAS
    salida, columnas = (None, None) if incompleto else ventana(rng, n, dist, definiciones, regresion)
    b = {
        "formato": "matrixai.traza/v1", "seq": seq, "prev": prev,
        "servicio": {"id": servicio_id, "huella_clave": huella, "clave_publica": pub_hex, "huella_modelo": HUELLA_MODELO},
        "ventana": {"desde": desde, "hasta": hasta}, "n": n, "k_minimo": k_minimo, "incompleto": incompleto,
        "reinicio": reinicio, "salida": salida, "columnas": columnas,
    }
    return firmado(k, b)


def construir():
    kA, pubA, hA = clave(SEMILLA_A)
    kB, pubB, hB = clave(SEMILLA_B)
    rng = random.Random(153)
    D = "2026-11-02T"
    b0 = bloque(kA, pubA, hA, 0, GENESIS, D + "10:00:00Z", D + "11:00:00Z", rng, 40, 0.0)
    b1 = bloque(kA, pubA, hA, 1, hash_de_bloque(b0), D + "11:00:00Z", D + "12:00:00Z", rng, 52, 0.0)
    b2 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", rng, 47, 1.5)
    cadena = [b0, b1, b2]
    L = [linea(b) for b in cadena]
    casos = []

    def caso(nombre, que, lineas, esperado, **opc):
        c = {"nombre": nombre, "que_prueba": que, "lineas": lineas, "esperado": esperado}
        c.update(opc)
        casos.append(c)

    def en_b1(viejo, nuevo):
        f = L[1].replace(viejo, nuevo, 1)
        assert f != L[1], (viejo, nuevo)
        return [L[0], f, L[2]]

    caso("cadena_valida", "Los 3 bloques intactos", L,
         {"estado": "valida", "motivo": None, "bloque": None, "bloques": 3, "empieza_en": 0, "ultimo_seq": 2, "huellas": [hA], "reinicios": []})
    caso("cadena_valida_con_ancla", "Misma cadena con la huella anclada correcta", L, {"estado": "valida", "bloques": 3}, ancla=hA)
    caso("cadena_valida_esperando_genesis", "Empieza en el génesis, como se exige", L, {"estado": "valida", "bloques": 3}, esperar_genesis=True)

    # --- que DEBEN fallar: editar ---
    m = copy.deepcopy(b1); m["n"] += 1
    caso("editado_n_sin_refirmar", "Se cambia n en el bloque 1 y se deja la firma vieja", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(b1); m["columnas"]["edad_años"]["bins"][3] += 1; m["columnas"]["edad_años"]["bins"][4] -= 1
    caso("editado_un_recuento_con_suma_igual", "Mueve un recuento de casilla sin cambiar ninguna suma", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(b0); m["servicio"]["id"] = "otro-servicio"
    caso("editado_id_del_servicio", "Se cambia el id del servicio en el génesis", [linea(m), L[1], L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 0})
    m = copy.deepcopy(b1); m["k_minimo"] = 1
    caso("editado_k_minimo", "Se rebaja k_minimo para que un bloque pequeño parezca legítimo, sin refirmar", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    # --- borrar, reordenar ---
    caso("borrado_el_del_medio", "Falta el bloque 1: los otros dos son válidos por separado", [L[0], L[2]],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 1})
    caso("reordenado_1_y_2", "Los bloques 1 y 2 intercambiados", [L[0], L[2], L[1]],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 1})
    # --- otra clave ---
    b2B = bloque(kB, pubB, hB, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(1), 47, 0.0)
    caso("otra_clave_en_el_ultimo_bloque", "El bloque 2 lo firma OTRA clave (con prev y seq correctos)", [L[0], L[1], linea(b2B)],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 2})
    m = copy.deepcopy(b2); m["servicio"]["clave_publica"], m["servicio"]["huella_clave"] = pubB, hB
    caso("clave_cambiada_sin_refirmar", "Se pone la clave B en el bloque 2 pero la firma es la de A", [L[0], L[1], linea(m)],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 2})
    m = copy.deepcopy(b2); m["servicio"]["huella_clave"] = hB
    caso("huella_que_no_es_la_de_su_clave", "La huella no corresponde a la clave pública del bloque", [L[0], L[1], linea(firmado(kA, m))],
         {"estado": "no_coincide", "motivo": "clave", "bloque": 2})
    rng2 = random.Random(7)
    x0 = bloque(kB, pubB, hB, 0, GENESIS, D + "10:00:00Z", D + "11:00:00Z", rng2, 40, 0.0)
    x1 = bloque(kB, pubB, hB, 1, hash_de_bloque(x0), D + "11:00:00Z", D + "12:00:00Z", rng2, 52, 0.0)
    x2 = bloque(kB, pubB, hB, 2, hash_de_bloque(x1), D + "12:00:00Z", D + "13:00:00Z", rng2, 47, 0.0)
    cB = [linea(x0), linea(x1), linea(x2)]
    caso("otra_cadena_entera_con_ancla", "Fichero sustituido por otra cadena válida de otra clave; la huella anclada es la de A", cB,
         {"estado": "no_coincide", "motivo": "ancla", "bloque": 0}, ancla=hA)
    caso("otra_cadena_entera_SIN_ancla", "LO MISMO sin ancla: NO se detecta (solo se ve en `huellas`). Por eso el Studio ancla la huella.", cB,
         {"estado": "valida", "motivo": None, "bloques": 3, "huellas": [hB]})
    # --- prev, seq, firma ---
    m = copy.deepcopy(b1); m["prev"] = GENESIS
    caso("prev_cambiado_sin_refirmar", "Se pone prev = ceros en el bloque 1 sin refirmar", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(b1); m["prev"] = hashlib.sha256(b"otro").hexdigest(); m = firmado(kA, m)
    caso("prev_cambiado_y_refirmado", "prev cambiado y refirmado con la clave buena: firma válida, cadena rota", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 1})
    m = copy.deepcopy(b1); m["seq"] = 5; m = firmado(kA, m)
    caso("seq_saltado_y_refirmado", "seq 5 tras 0, refirmado: la cadena no es contigua", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 1})
    m = copy.deepcopy(b1); m["firma"]["valor"] = "AAAA" + m["firma"]["valor"][4:]
    caso("firma_alterada", "Se estropean los 4 primeros caracteres de la firma", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    v = b1["firma"]["valor"]
    assert v.endswith("==") and len(v) == 88
    alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    i = alfabeto.index(v[85])
    alt = alfabeto[i ^ 1]  # mismos 2 bits altos del último sexteto, distintos bits bajos (que no se usan)
    m = copy.deepcopy(b1); m["firma"]["valor"] = v[:85] + alt + v[86:]
    assert base64_igual(v, m["firma"]["valor"])
    caso("firma_con_base64_no_canonico", "Los mismos 64 bytes de firma escritos con otro texto base64: no coincide (una sola forma por firma)",
         [L[0], linea(m), L[2]], {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(b1); m["firma"]["valor"] = ""
    caso("firma_vacia", "El valor de la firma es la cadena vacía", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    pub_malo = "ff" * 32
    m = copy.deepcopy(b1)
    m["servicio"]["clave_publica"], m["servicio"]["huella_clave"] = pub_malo, hashlib.sha256(bytes.fromhex(pub_malo)).hexdigest()
    caso("clave_publica_que_no_es_un_punto", "32 bytes que no son una clave Ed25519 (con su huella bien calculada): ninguna firma cuadra",
         [linea(m)], {"estado": "no_coincide", "motivo": "firma", "bloque": 0})
    # --- LO QUE NO SE DETECTA ---
    caso("TRUNCADA_LA_COLA", "Se borra el bloque 2: NO se detecta (el fichero parece completo). Solo se ve ultimo_seq=1", L[:2],
         {"estado": "valida", "motivo": None, "bloques": 2, "empieza_en": 0, "ultimo_seq": 1, "huellas": [hA]})
    caso("TRUNCADA_LA_CABEZA", "Se borra el bloque 0: NO se detecta, se informa empieza_en=1", L[1:],
         {"estado": "valida", "motivo": None, "bloques": 2, "empieza_en": 1, "ultimo_seq": 2})
    caso("TRUNCADA_LA_CABEZA_esperando_genesis", "Lo mismo, pero quien importa por primera vez exige el génesis: SÍ se ve", L[1:],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 0}, esperar_genesis=True)
    caso("fichero_de_dia_siguiente_empieza_en_2", "Rotación diaria legítima: empieza en seq 2 con prev fuera; válido, no es manipulación", L[2:],
         {"estado": "valida", "empieza_en": 2, "bloques": 1})
    # --- reinicio ---
    r0 = bloque(kB, pubB, hB, 0, GENESIS, D + "14:00:00Z", D + "15:00:00Z", random.Random(3), 44, 0.0, reinicio="sin_previo")
    r1 = bloque(kB, pubB, hB, 1, hash_de_bloque(r0), D + "15:00:00Z", D + "16:00:00Z", random.Random(4), 31, 0.0)
    caso("reinicio_sin_previo_con_otra_clave", "Cadena nueva (carpeta borrada, clave nueva) dentro del mismo fichero: hueco explicado",
         L + [linea(r0), linea(r1)], {"estado": "valida", "bloques": 5, "huellas": [hA, hB], "reinicios": [3]})
    caso("reinicio_con_ancla_de_la_primera", "El mismo fichero con la huella A anclada: la cadena B se ve", L + [linea(r0), linea(r1)],
         {"estado": "no_coincide", "motivo": "ancla", "bloque": 3}, ancla=hA)
    m = copy.deepcopy(b2); m["reinicio"] = "sin_previo"
    caso("reinicio_falso", "reinicio sin_previo con seq 2 y prev de un bloque real", [L[0], L[1], linea(firmado(kA, m))],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 2})
    m = copy.deepcopy(b1); m["seq"], m["prev"] = 0, GENESIS
    caso("genesis_en_medio_sin_reinicio", "seq 0 y prev de ceros en medio, sin marcar reinicio", [L[0], linea(firmado(kA, m))],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 1})
    # --- incompleto y k ---
    i2 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "12:07:00Z", random.Random(5), 11, 0.0, incompleto=True)
    caso("incompleto_solo_n", "Bloque de apagado con n < k: salida y columnas null", [L[0], L[1], linea(i2)],
         {"estado": "valida", "bloques": 3, "ultimo_seq": 2})
    m = copy.deepcopy(i2); m["columnas"] = {}
    caso("incompleto_con_columnas", "incompleto:true pero con columnas (debe ir a null)", [L[0], L[1], linea(firmado(kA, m))],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    pequeno = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "12:07:00Z", random.Random(5), 11, 0.0)
    caso("completo_con_n_menor_que_k", "Un bloque con histogramas y n=11 < k_minimo=20: DS2 no lo deja cerrar (la fila de una persona)",
         [L[0], L[1], linea(pequeno)], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    grande_incompleto = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "12:07:00Z", random.Random(5), 25, 0.0, incompleto=True)
    caso("incompleto_con_n_mayor_o_igual_que_k", "incompleto:true con n=25 >= k_minimo=20: se habría cerrado entero",
         [L[0], L[1], linea(grande_incompleto)], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    k0 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(5), 47, 0.0, k_minimo=0)
    caso("k_minimo_cero", "k_minimo = 0: sin umbral no hay protección", [L[0], L[1], linea(k0)],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    k5 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "12:30:00Z", random.Random(5), 11, 0.0, k_minimo=5)
    caso("k_minimo_propio_de_cada_bloque", "El umbral va grabado en cada bloque: n=11 con k_minimo=5 es un bloque completo válido",
         [L[0], L[1], linea(k5)], {"estado": "valida", "bloques": 3, "ultimo_seq": 2})
    m = copy.deepcopy(b1); del m["k_minimo"]
    caso("sin_k_minimo", "Un bloque con la forma de antes de DS9 (sin k_minimo), firmado", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    # --- regresión ---
    rr = random.Random(11)
    g0 = bloque(kA, pubA, hA, 0, GENESIS, D + "10:00:00Z", D + "11:00:00Z", rr, 60, 0.0, regresion=True)
    g1 = bloque(kA, pubA, hA, 1, hash_de_bloque(g0), D + "11:00:00Z", D + "12:00:00Z", rr, 55, 1.0, regresion=True)
    G = [linea(g0), linea(g1)]
    caso("regresion_valida", "Regresión: salida.valor lleva el histograma del valor predicho; probabilidad y etiquetas son null (columna `__proto__` a propósito)",
         G, {"estado": "valida", "bloques": 2, "ultimo_seq": 1, "huellas": [hA]})
    m = copy.deepcopy(g1); m["salida"]["probabilidad"] = [0] * 9 + [55]
    caso("regresion_con_probabilidad", "salida.valor y salida.probabilidad a la vez: son null en regresión", [G[0], linea(firmado(kA, m))],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(g1); m["salida"]["valor"]["bins"][0] += 1
    caso("regresion_valor_no_suma_n", "salida.valor.bins suma n+1, firmado", [G[0], linea(firmado(kA, m))],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(g1); m["salida"]["valor"]["bins"][2] += 1; m["salida"]["valor"]["bins"][3] -= 1
    caso("regresion_editado_un_recuento_del_valor", "Se mueve un recuento del histograma del valor predicho, sin refirmar", [G[0], linea(m)],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(g1); m["salida"]["valor"]["fuera"] = 500
    caso("regresion_fuera_mayor_que_casillas", "salida.valor.fuera > suma de casillas, firmado", [G[0], linea(firmado(kA, m))],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    # --- no se pudo comprobar ---
    m = copy.deepcopy(b1); m["formato"] = "matrixai.traza/v2"
    caso("version_desconocida", "formato v2: este verificador no sabe leerlo", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "version", "bloque": 1})
    caso("linea_que_no_es_json", "Una línea de basura", [L[0], "esto no es json {", L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("linea_en_blanco_en_medio", "Una línea vacía entre dos bloques no es un bloque", [L[0], "", L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("bloque_que_no_es_un_objeto", "Una línea que es una lista JSON", [L[0], "[1,2]", L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("anidacion_excesiva", "100 niveles de corchetes", [L[0], "[" * 100 + "]" * 100, L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("fichero_vacio", "Sin bloques", [], {"estado": "no_se_pudo_comprobar", "motivo": "formato"})
    m = copy.deepcopy(b1); m["columnas"]["edad_años"]["faltantes"] += 1
    caso("suma_incoherente_firmada", "Firmado con la clave buena pero bins+faltantes != n", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(b1); m["extra"] = 1
    caso("campo_de_mas", "Un campo que v1 no define, firmado", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(b1); m["ventana"]["desde"], m["ventana"]["hasta"] = m["ventana"]["hasta"], m["ventana"]["desde"]
    caso("ventana_al_reves", "ventana.desde > ventana.hasta, firmado", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(b1); m["servicio"]["huella_clave"] = hA.upper()
    caso("huella_en_mayusculas", "La huella en hex MAYÚSCULA: v1 la quiere en minúscula", [L[0], linea(m), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(b1); m["reinicio"] = "otro"
    caso("reinicio_con_valor_desconocido", "reinicio distinto de null y sin_previo, firmado", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    m = copy.deepcopy(b1); m["servicio"]["id"] = "x" * 129
    caso("id_de_129_caracteres", "servicio.id con 129 caracteres, firmado", [L[0], linea(firmado(kA, m)), L[2]],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    astrales = bloque(kA, pubA, hA, 0, GENESIS, D + "10:00:00Z", D + "11:00:00Z", random.Random(2), 40, 0.0, servicio_id="😀" * 128)
    caso("id_de_128_caracteres_astrales", "servicio.id con 128 puntos de código (256 unidades UTF-16): vale, se cuentan caracteres y no unidades",
         [linea(astrales)], {"estado": "valida", "bloques": 1})
    caso("numero_con_punto", "n escrito como 52.0: v1 solo admite enteros (paridad Python/JS)", en_b1('"n":52', '"n":52.0'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("numero_con_exponente", "n escrito como 5.2e1", en_b1('"n":52', '"n":5.2e1'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("numero_menos_cero", "n escrito como -0", en_b1('"n":52', '"n":-0'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("numero_negativo", "n = -1", en_b1('"n":52', '"n":-1'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("entero_mayor_que_2_elevado_a_53", "n = 2^53+1: JavaScript lo redondearía; v1 lo rechaza en los dos", en_b1('"n":52', '"n":9007199254740993'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("constante_NaN", "NaN como número", en_b1('"n":52', '"n":NaN'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("clave_duplicada", "Clave n repetida en el JSON (paridad: JS se queda con la última)", en_b1('{"columnas"', '{"n":52,"columnas"'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("surrogado_suelto", "Un \\ud800 suelto en el id: no hay UTF-8 que firmar", en_b1('"id":"servicio-ñandú-de-prueba"', '"id":"servicio-\\ud800"'),
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    # --- la misma cadena escrita de otras maneras: igual de válida (la firma es sobre el canónico) ---
    caso("misma_cadena_con_espacios", "La cadena válida reescrita con espacios tras `:` y `,`: igual de válida",
         [json.dumps(json.loads(x), ensure_ascii=False, separators=(", ", ": ")) for x in L],
         {"estado": "valida", "bloques": 3, "ultimo_seq": 2})
    caso("misma_cadena_con_escapes_unicode", "La cadena válida con todo lo no ASCII escapado (\\uXXXX): igual de válida",
         [json.dumps(json.loads(x), ensure_ascii=True) for x in L], {"estado": "valida", "bloques": 3, "ultimo_seq": 2})
    caso("misma_cadena_con_crlf", "Líneas acabadas en \\r (fichero de Windows): el espacio en blanco al final es JSON válido",
         [x + "\r" for x in L], {"estado": "valida", "bloques": 3, "ultimo_seq": 2})

    todas = cadena + [g0, g1]
    return {
        "formato_de_vectores": "matrixai.traza.vectores/v1",
        "AVISO": "CLAVES DE PRUEBA. Las semillas de abajo son públicas y fijas; no protegen nada; no usarlas para firmar nada real.",
        "prefijo_de_firma": PREFIJO_FIRMA.decode("utf-8"),
        "canonicalizador": "matrixai.pipelines.canonical.jcs_bytes (JCS, RFC 8785)",
        "claves_de_prueba": {
            "A": {"semilla_hex": SEMILLA_A, "publica_hex": pubA, "huella": hA},
            "B": {"semilla_hex": SEMILLA_B, "publica_hex": pubB, "huella": hB},
        },
        "cadena_valida": {"bloques": [
            {"seq": b["seq"], "linea": linea(b), "canonico_sin_firma": jcs_canonical({k: v for k, v in b.items() if k != "firma"}),
             "sha256_del_mensaje_firmado": hashlib.sha256(mensaje_firmado(b)).hexdigest(), "hash_del_bloque": hash_de_bloque(b)}
            for b in todas]},
        "casos": casos,
    }


def base64_igual(a, b):
    import base64
    return base64.b64decode(a) == base64.b64decode(b) and a != b


def texto_de_los_vectores():
    return json.dumps(construir(), indent=1, ensure_ascii=False) + "\n"


def main():
    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectores_traza_v1.json")
    texto = texto_de_los_vectores()
    assert texto == texto_de_los_vectores(), "los vectores no son reproducibles"
    with open(salida, "w", encoding="utf-8") as fh:
        fh.write(texto)
    print(f"escrito {salida}: {len(json.loads(texto)['casos'])} casos, {len(texto)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
