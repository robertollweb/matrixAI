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
    # (rep-c1, I1) 0xff…ff tiene y = 2^255-1 >= p: NO es canónica, y desde la reparación se rechaza como `clave` antes de verificar
    caso("clave_publica_que_no_es_un_punto", "32 bytes 0xff (y >= p, no canónica; huella bien calculada): la clave no vale, antes de verificar",
         [linea(m)], {"estado": "no_coincide", "motivo": "clave", "bloque": 0})
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
    i2 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(5), 11, 0.0, incompleto=True)
    caso("incompleto_solo_n", "Bloque de apagado con n < k: salida y columnas null", [L[0], L[1], linea(i2)],
         {"estado": "valida", "bloques": 3, "ultimo_seq": 2})
    m = copy.deepcopy(i2); m["columnas"] = {}
    caso("incompleto_con_columnas", "incompleto:true pero con columnas (debe ir a null)", [L[0], L[1], linea(firmado(kA, m))],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    pequeno = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(5), 11, 0.0)
    caso("completo_con_n_menor_que_k", "Un bloque con histogramas y n=11 < k_minimo=20: DS2 no lo deja cerrar (la fila de una persona)",
         [L[0], L[1], linea(pequeno)], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    grande_incompleto = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(5), 25, 0.0, incompleto=True)
    caso("incompleto_con_n_mayor_o_igual_que_k", "incompleto:true con n=25 >= k_minimo=20: se habría cerrado entero",
         [L[0], L[1], linea(grande_incompleto)], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    k0 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(5), 47, 0.0, k_minimo=0)
    caso("k_minimo_cero", "k_minimo = 0: sin umbral no hay protección", [L[0], L[1], linea(k0)],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2})
    k5 = bloque(kA, pubA, hA, 2, hash_de_bloque(b1), D + "12:00:00Z", D + "13:00:00Z", random.Random(5), 11, 0.0, k_minimo=5)
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
    # 153-C4 (Roberto, 07-11, opción A): `desde` es el INICIO DE LA HORA UTC; el segundo exacto de la primera petición no es v1.
    for nombre, desde in (("ventana_desde_con_segundos", "2026-11-02T11:00:07Z"), ("ventana_desde_con_minutos", "2026-11-02T11:30:00Z"),
                          ("ventana_desde_a_mitad_de_hora_sin_segundos", "2026-11-02T11:59:59Z")):
        m = copy.deepcopy(b1); m["ventana"]["desde"] = desde
        caso(nombre, f"ventana.desde = {desde}, firmado: no es el inicio de una hora UTC", [L[0], linea(firmado(kA, m)), L[2]],
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

    casos_de_la_reparacion(caso, kA, pubA, hA, kB, pubB, hB, L, b0, b1, b2)
    casos_del_remate_153(caso, kA, pubA, hA, L, b1)

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


# ─────────────── reparación de la auditoría de C1 (rep-c1): B1 reinicios, I1 claves, I2 topes, menores ───────────────
P = 2 ** 255 - 19
Y8 = 0x7A03AC9277FDC74EC6CC392CFA53202A0F67100D760B3CBA4FD84D3D706A17C7
TOPE_LINEA = 1 << 20     # declarados AQUÍ otra vez, a propósito (como el prefijo): si el comprobador cambia los suyos, se ve
TOPE_FICHERO = 64 << 20


def cadena_de(k, pub, h, n, hora, rng, reinicio="sin_previo", seq0=0, prev0=GENESIS):
    """n bloques horarios desde las `hora`:00 del 2026-11-02 (pasan de día si hace falta); el primero con `reinicio`."""
    import datetime as dt
    t0 = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc) + dt.timedelta(hours=hora)
    hh = lambda x: (t0 + dt.timedelta(hours=x)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out, prev = [], prev0
    for j in range(n):
        b = bloque(k, pub, h, seq0 + j, prev, hh(j), hh(j + 1), rng, 30 + j, 0.0, reinicio=reinicio if j == 0 else None)
        out.append(b)
        prev = hash_de_bloque(b)
    return out


def enc(y, signo=0):
    return (y | (signo << 255)).to_bytes(32, "little")


def con_clave_y_firma(b, crudo, firma64):
    """El bloque con OTRA clave pública (y su huella bien calculada) y una firma de 64 bytes dada, sin firmar de verdad."""
    import base64
    m = copy.deepcopy(b)
    m["servicio"]["clave_publica"], m["servicio"]["huella_clave"] = crudo.hex(), hashlib.sha256(crudo).hexdigest()
    m["firma"]["valor"] = base64.b64encode(firma64).decode("ascii")
    return m


def casos_de_la_reparacion(caso, kA, pubA, hA, kB, pubB, hB, L, b0, b1, b2):
    import base64
    rng = random.Random(1531)
    # Tres cadenas de la clave A en un fichero, como el de un servicio que se reinició dos veces sin poder leer su última línea:
    # A0 (génesis, 10-13 h), A1 (reinicio, 14-16 h), A2 (reinicio, 18 h).
    A0 = cadena_de(kA, pubA, hA, 3, 10, rng, reinicio=None)
    A1 = cadena_de(kA, pubA, hA, 2, 14, rng)
    A2 = cadena_de(kA, pubA, hA, 1, 18, rng)
    LA0, LA1, LA2 = [linea(b) for b in A0], [linea(b) for b in A1], [linea(b) for b in A2]

    def cad(inicio, fin, si, sf, h=hA):
        return {"inicio": inicio, "fin": fin, "seq_inicial": si, "seq_final": sf, "huella": h}

    tres = [cad(0, 2, 0, 2), cad(3, 4, 0, 1), cad(5, 5, 0, 0)]
    caso("B1_tres_cadenas_legitimas", "Génesis y dos reinicios sin_previo de la MISMA clave, en orden: válida, y dice las tres cadenas",
         LA0 + LA1 + LA2, {"estado": "valida", "bloques": 6, "reinicios": [3, 5], "cadenas": tres, "huellas": [hA]})
    caso("B1_tres_cadenas_legitimas_con_ancla", "Lo mismo con la huella anclada", LA0 + LA1 + LA2,
         {"estado": "valida", "bloques": 6, "reinicios": [3, 5], "cadenas": tres}, ancla=hA)
    # lo que NO se detecta (y la salida lo dice): borrar ANTES de un reinicio
    caso("B1_borrada_la_cola_antes_de_un_reinicio_NO_se_detecta",
         "Se borra el último bloque de A0, justo antes del reinicio: el reinicio dice «no hay anterior» y nada lo enlaza. NO se detecta; el §4 lo dice",
         LA0[:2] + LA1 + LA2, {"estado": "valida", "bloques": 5, "reinicios": [2, 4]}, ancla=hA)
    caso("B1_borrada_una_cadena_entera_NO_se_detecta", "Se borra la cadena A1 entera: NO se detecta (el §4 lo dice)",
         LA0 + LA2, {"estado": "valida", "bloques": 4, "reinicios": [3]}, ancla=hA)
    # remate-c1 (menor 2): borrada la cadena del génesis, el fichero empieza en un reinicio (seq 0 y prev de ceros también)
    caso("B1_borrada_la_cadena_del_genesis_NO_se_detecta_sin_esperar_genesis",
         "Se borra A0 entera y el fichero empieza en el reinicio de A1: sin exigir el génesis NO se detecta (el §4 lo dice)",
         LA1 + LA2, {"estado": "valida", "bloques": 3, "reinicios": [0, 2], "cadenas": [cad(0, 1, 0, 1), cad(2, 2, 0, 0)]}, ancla=hA)
    caso("B1_borrada_la_cadena_del_genesis_esperando_genesis",
         "Lo mismo exigiendo el génesis: un reinicio sin_previo no es el génesis aunque tenga seq 0 y prev de ceros: SÍ se ve",
         LA1 + LA2, {"estado": "no_coincide", "motivo": "cadena", "bloque": 0}, ancla=hA, esperar_genesis=True)
    # lo que SÍ (B1, arreglo b: la ventana firmada de cadenas de la misma clave, en orden; la hora NO se comprueba)
    caso("B1_cadenas_intercambiadas", "A2 antes que A1: la cadena A1 empieza (14 h) antes de que acabe A2 (19 h), misma clave",
         LA0 + LA2 + LA1, {"estado": "no_coincide", "motivo": "cadena", "bloque": 4})
    caso("B1_cadenas_intercambiadas_con_ancla", "Lo mismo con la huella anclada: la cadena se mira antes que el ancla",
         LA0 + LA2 + LA1, {"estado": "no_coincide", "motivo": "cadena", "bloque": 4}, ancla=hA)
    caso("B1_una_cadena_de_reinicio_puesta_al_principio", "A1 antes que el génesis: el génesis aparece tras una cadena, sin reinicio",
         LA1 + LA0 + LA2, {"estado": "no_coincide", "motivo": "cadena", "bloque": 2})
    # (arreglo a: un bloque repetido en el fichero es `cadena`)
    caso("B1_cadena_de_reinicio_duplicada", "A1 dos veces seguidas: su primer bloque se repite",
         LA0 + LA1 + LA1 + LA2, {"estado": "no_coincide", "motivo": "cadena", "bloque": 5})
    caso("B1_bloque_de_reinicio_duplicado_al_final", "La cadena de un solo bloque A2, dos veces al final",
         LA0 + LA1 + LA2 + LA2, {"estado": "no_coincide", "motivo": "cadena", "bloque": 6}, ancla=hA)
    caso("B1_solo_una_cadena_de_reinicio_dos_veces", "A1 + A1 y nada más", LA1 + LA1,
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 2})
    caso("B1_bloque_duplicado_dentro_de_la_cadena", "El bloque 1 repetido justo después de sí mismo", [L[0], L[1], L[1], L[2]],
         {"estado": "no_coincide", "motivo": "cadena", "bloque": 2})
    caso("B1_bloque_duplicado_con_otro_formato", "El génesis repetido al final reescrito con espacios: es el MISMO bloque (el hash es del canónico)",
         L + [json.dumps(json.loads(L[0]), separators=(", ", ": "), ensure_ascii=False)], {"estado": "no_coincide", "motivo": "cadena", "bloque": 3})
    # el que SOLO caza el hash repetido: una cadena de reinicio de un bloque (19-20 h, un servicio que se apaga en la misma hora en que
    # arranca) duplicada: el orden de ventanas no la ve (153 remate: la copia empieza a las 19:00, en la hora en que acaba la primera)
    V = bloque(kA, pubA, hA, 0, GENESIS, "2026-11-02T19:00:00Z", "2026-11-02T20:00:00Z", rng, 5, 0.0, incompleto=True, reinicio="sin_previo")
    caso("B1_bloque_de_una_hora_duplicado", "Un reinicio de un solo bloque de una hora, dos veces: solo lo ve el hash repetido",
         LA0 + LA1 + LA2 + [linea(V), linea(V)], {"estado": "no_coincide", "motivo": "cadena", "bloque": 7})
    caso("B1_bloque_de_una_hora_una_vez", "El mismo bloque una sola vez: válido", LA0 + LA1 + LA2 + [linea(V)],
         {"estado": "valida", "bloques": 7, "reinicios": [3, 5, 6]})
    # el límite del orden: una cadena que empieza justo cuando acaba la anterior (desde == hasta) vale
    E = cadena_de(kA, pubA, hA, 1, 13, rng)  # A0 acaba a las 13:00
    caso("B1_reinicio_que_empieza_justo_cuando_acaba_la_anterior", "desde de la cadena nueva == hasta de la anterior: en orden, válida",
         LA0 + [linea(E[0])], {"estado": "valida", "bloques": 4, "reinicios": [3]})
    E = cadena_de(kA, pubA, hA, 1, 11, rng)  # empieza a las 11:00, antes de la hora (12-13) en que acaba A0
    caso("B1_reinicio_que_se_solapa_con_la_anterior", "La cadena nueva de la misma clave empieza antes de la hora en que acaba la anterior (11:00; A0 acaba a las 13:00)",
         LA0 + [linea(E[0])], {"estado": "no_coincide", "motivo": "cadena", "bloque": 3})
    # cadenas de claves DISTINTAS: su orden no se puede exigir (otro servicio, otra hora); sin ancla no se detecta, con ancla sí
    rB = random.Random(1532)
    B0 = cadena_de(kB, pubB, hB, 2, 8, rB)
    LB0 = [linea(b) for b in B0]
    caso("B1_cadena_de_otra_clave_antes_y_fuera_de_hora_NO_se_detecta_sin_ancla",
         "Una cadena de la clave B (8-10 h) después de A1 (14-16 h): claves distintas, el orden no se exige; sin ancla es válida y salen dos huellas",
         LA0 + LA1 + LB0, {"estado": "valida", "huellas": [hA, hB], "reinicios": [3, 5],
                           "cadenas": [cad(0, 2, 0, 2), cad(3, 4, 0, 1), cad(5, 6, 0, 1, hB)]})
    caso("B1_cadena_de_otra_clave_con_ancla", "Lo mismo con la huella A anclada: la cadena B se ve", LA0 + LA1 + LB0,
         {"estado": "no_coincide", "motivo": "ancla", "bloque": 5}, ancla=hA)
    # (menor 7) el primer bloque del fichero, incoherente
    m = copy.deepcopy(b0); m["prev"] = hash_de_bloque(b2)
    caso("primer_bloque_seq_0_con_prev_que_no_es_de_ceros", "seq 0 sin reinicio y con un prev real: el génesis lleva ceros; incoherente",
         [linea(firmado(kA, m)), L[1]], {"estado": "no_coincide", "motivo": "cadena", "bloque": 0})
    m = copy.deepcopy(b2); m["prev"] = GENESIS
    caso("primer_bloque_seq_2_con_prev_de_ceros", "seq 2 sin reinicio y con prev de ceros: solo el génesis los lleva; incoherente",
         [linea(firmado(kA, m))], {"estado": "no_coincide", "motivo": "cadena", "bloque": 0})
    caso("primer_bloque_seq_2_con_prev_real", "El caso legítimo de al lado: un fichero de otro día empieza en seq 2 con el prev del 1",
         [L[2]], {"estado": "valida", "empieza_en": 2})

    # I1: claves públicas que «firman» cualquier cosa, con firmas que lo aprovechan (R = identidad, S = 0, etc.)
    firma_id = enc(1) + b"\x00" * 32
    malas = {
        "identidad": enc(1), "identidad_con_signo": enc(1, 1), "ceros_orden_4": enc(0), "orden_4_con_signo": enc(0, 1),
        "orden_2": enc(P - 1), "orden_2_con_signo": enc(P - 1, 1), "orden_8_a": enc(Y8), "orden_8_a_con_signo": enc(Y8, 1),
        "orden_8_b": enc(P - Y8), "orden_8_b_con_signo": enc(P - Y8, 1), "y_igual_a_p_no_canonica": enc(P),
        "y_p_mas_1_no_canonica_identidad": enc(P + 1), "fuera_de_la_curva_y_2": enc(2),
    }
    for nombre, crudo in malas.items():
        caso(f"I1_clave_{nombre}", f"Clave pública {nombre} con su huella bien calculada y firma R=identidad, S=0: `clave`, antes de verificar",
             [linea(con_clave_y_firma(b0, crudo, firma_id))], {"estado": "no_coincide", "motivo": "clave", "bloque": 0})
    caso("I1_clave_identidad_y_firma_R_de_orden_4", "Clave identidad con R = (y=0), S = 0", [linea(con_clave_y_firma(b0, enc(1), enc(0) + b"\x00" * 32))],
         {"estado": "no_coincide", "motivo": "clave", "bloque": 0})
    sig = base64.b64decode(b1["firma"]["valor"])
    L_ORDEN = 2 ** 252 + 27742317777372353535851937790883648493
    S = int.from_bytes(sig[32:], "little")
    m = copy.deepcopy(b1); m["firma"]["valor"] = base64.b64encode(sig[:32] + (S + L_ORDEN).to_bytes(32, "little")).decode("ascii")
    caso("I1_firma_con_S_mas_L", "La firma buena del bloque 1 con S + L (otra firma «equivalente»): S >= L no es una firma", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(b1); m["firma"]["valor"] = base64.b64encode(sig[:32] + L_ORDEN.to_bytes(32, "little")).decode("ascii")
    caso("I1_firma_con_S_igual_a_L", "La R buena del bloque 1 con S = L exacto (remate-c1, menor 1): el borde, S < L estricto; con "
         "`<=` en vez de `<` una biblioteca que lo aceptara lo daría por firmado", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    m = copy.deepcopy(b1); m["firma"]["valor"] = base64.b64encode(enc(1) + b"\x00" * 32).decode("ascii")
    caso("I1_firma_R_identidad_con_clave_buena", "Clave A buena y firma R = identidad, S = 0", [L[0], linea(m), L[2]],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})

    # I2: topes, ANTES de leer el JSON. Las líneas largas van como {"base", "rellenar_con_espacios_hasta"} (espacio al final: JSON válido)
    caso("I2_linea_de_1_MiB_justo", "El bloque 0 relleno con espacios hasta 1 MiB exacto: cabe y es válido",
         [{"base": L[0], "rellenar_con_espacios_hasta": TOPE_LINEA}, L[1], L[2]], {"estado": "valida", "bloques": 3})
    caso("I2_linea_de_1_MiB_mas_1_byte", "El bloque 1 relleno hasta 1 MiB + 1 byte: `formato` en esa línea, sin leer su JSON",
         [L[0], {"base": L[1], "rellenar_con_espacios_hasta": TOPE_LINEA + 1}, L[2]], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    caso("I2_linea_de_basura_de_20_MiB", "Una línea de 20 MiB (no JSON): `formato` en esa línea; antes, Node se quedaba sin memoria con 100 MB",
         [L[0], {"base": "[", "rellenar_con_espacios_hasta": 20 << 20}], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    rT = random.Random(1533)
    larga = cadena_de(kA, pubA, hA, 64, 0, rT, reinicio=None)
    LL = [linea(b) for b in larga]
    assert len(LL) == 64
    justo = [{"base": x, "rellenar_con_espacios_hasta": TOPE_LINEA - 1} for x in LL]  # 64 × (1 MiB − 1 + el salto) = 64 MiB
    caso("I2_fichero_de_64_MiB_justo", "64 bloques encadenados, cada línea de 1 MiB − 1 byte: con sus saltos, 64 MiB exactos; cabe",
         justo, {"estado": "valida", "bloques": 64, "ultimo_seq": 63})
    caso("I2_fichero_de_64_MiB_mas_1_byte", "Lo mismo con un espacio más en la última línea: 64 MiB + 1; `formato` del fichero, sin línea",
         justo[:-1] + [{"base": LL[-1], "rellenar_con_espacios_hasta": TOPE_LINEA}], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": None})
    caso("I2_fichero_en_bytes_de_64_MiB_mas_1", "Los BYTES de un fichero de 64 MiB + 1 (espacios): `formato` antes de decodificar",
         [], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": None}, fichero_de_espacios=TOPE_FICHERO + 1)
    # remate-c1: el comprobador ya no guarda un fallo por bloque, solo el PRIMERO de firma y el primero de coherencia. Una firma
    # rota DESPUÉS de un bloque incoherente (bien firmado) sigue ganando: las firmas van antes que la coherencia.
    incoherente = bloque(kA, pubA, hA, 0, GENESIS, "2026-11-02T10:00:00Z", "2026-11-02T11:00:00Z", random.Random(15304), 5, 0.0)
    rota = json.loads(L[1]); rota["n"] += 1
    caso("orden_coherencia_rota_antes_y_firma_rota_despues", "Bloque 0 bien firmado pero completo con n < k_minimo (coherencia), bloque 1 "
         "editado (firma): gana la firma, aunque venga después", [linea(incoherente), linea(rota)],
         {"estado": "no_coincide", "motivo": "firma", "bloque": 1})
    # remate-c1 (I2): los mismos topes por BYTES, donde el fichero ya no se parte en una lista y se salta de ventana en ventana.
    # `fichero_de_lineas`: las líneas (con relleno) unidas por `\n`, más uno al final si `salto_final`.
    caso("I2_bytes_linea_de_1_MiB_justo", "Por bytes: el bloque 0 de 1 MiB exacto; cabe", [], {"estado": "valida", "bloques": 3},
         fichero_de_lineas=[{"base": L[0], "rellenar_con_espacios_hasta": TOPE_LINEA}, L[1], L[2]], salto_final=True)
    caso("I2_bytes_linea_de_1_MiB_mas_1_byte", "Por bytes: el bloque 1 de 1 MiB + 1: `formato` en la línea 1", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1},
         fichero_de_lineas=[L[0], {"base": L[1], "rellenar_con_espacios_hasta": TOPE_LINEA + 1}, L[2]], salto_final=True)
    caso("I2_bytes_ultima_linea_de_1_MiB_mas_1_sin_salto", "Por bytes: la ÚLTIMA línea, sin `\\n`, de 1 MiB + 1: `formato` en esa línea", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 2},
         fichero_de_lineas=[L[0], L[1], {"base": L[2], "rellenar_con_espacios_hasta": TOPE_LINEA + 1}], salto_final=False)
    caso("I2_bytes_ultima_linea_de_1_MiB_justo_sin_salto", "Por bytes: la última línea, sin `\\n`, de 1 MiB exacto: cabe", [],
         {"estado": "valida", "bloques": 3},
         fichero_de_lineas=[L[0], L[1], {"base": L[2], "rellenar_con_espacios_hasta": TOPE_LINEA}], salto_final=False)
    caso("I2_bytes_linea_larga_tras_3000_cortas", "Por bytes: 3.000 líneas vacías y luego una de 1 MiB + 1: los topes van antes que el JSON, "
         "así que es `formato` en la línea 3000 (el índice se cuenta, no se lee)", [], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 3000},
         fichero_de_lineas=[""] * 3000 + [{"base": "[", "rellenar_con_espacios_hasta": TOPE_LINEA + 1}], salto_final=True)
    caso("I2_bytes_dos_lineas_largas_la_primera_cuenta", "Por bytes: líneas de 1 MiB + 1 en la 1 y en la 3: la primera", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1},
         fichero_de_lineas=["x", {"base": "[", "rellenar_con_espacios_hasta": TOPE_LINEA + 1}, "y",
                            {"base": "[", "rellenar_con_espacios_hasta": TOPE_LINEA + 1}], salto_final=True)
    caso("I2_bytes_64_MiB_justo", "Por bytes: los 64 bloques de 1 MiB − 1 con su salto: 64 MiB exactos; cabe", [],
         {"estado": "valida", "bloques": 64, "ultimo_seq": 63}, fichero_de_lineas=justo, salto_final=True)
    caso("I2_bytes_64_MiB_justo_sin_salto_final_cuenta_uno_mas", "Por bytes: 64 MiB exactos SIN `\\n` final (la última línea, 1 MiB): la "
         "última línea cuenta su salto aunque no lo lleve, como en las líneas sueltas: 64 MiB + 1, `formato` del fichero", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": None},
         fichero_de_lineas=justo[:-1] + [{"base": LL[-1], "rellenar_con_espacios_hasta": TOPE_LINEA}], salto_final=False)

    # menor 1: por BYTES, UTF-8 estricto, como en Python (`TextDecoder` con fatal e ignoreBOM en el JS)
    texto = ("\n".join(L) + "\n").encode("utf-8")
    b64 = lambda x: base64.b64encode(x).decode("ascii")
    caso("bytes_cadena_valida", "La cadena válida pasada como BYTES de fichero", [], {"estado": "valida", "bloques": 3}, fichero_b64=b64(texto))
    caso("bytes_con_BOM", "El mismo fichero con BOM UTF-8 delante: la primera línea deja de ser JSON (no se quita en silencio)", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 0}, fichero_b64=b64(b"\xef\xbb\xbf" + texto))
    caso("bytes_utf8_invalido", "Un byte 0xff en medio del id: no es UTF-8", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": None}, fichero_b64=b64(texto.replace("ñandú".encode(), b"\xff", 1)))
    caso("bytes_utf8_sobrelargo", "La «/» escrita en 2 bytes (0xc0 0xaf, UTF-8 sobrelargo): no es UTF-8", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": None}, fichero_b64=b64(texto.replace(b"ed25519", b"ed\xc0\xaf25519", 1)))
    caso("bytes_sustituto_codificado", "Un sustituto UTF-16 codificado en UTF-8 (0xed 0xa0 0x80): no es UTF-8", [],
         {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": None}, fichero_b64=b64(texto.replace("ñ".encode(), b"\xed\xa0\x80", 1)))


# ─────────────── remate de la auditoría de C4–C7 (07-11): `hasta` al final de su hora (Roberto, A) y claves de orden mixto (M4) ───────────────
# Aritmética de Ed25519 propia, SOLO para fabricar las claves de orden mixto (afín: lenta y clara; no se usa para comprobar nada)
_Pd = 2 ** 255 - 19
_Dd = (-121665 * pow(121666, -1, _Pd)) % _Pd
_Ld = 2 ** 252 + 27742317777372353535851937790883648493
_Bd = (15112221349535400772501151409588531511454012693041857206046113283949847762202,
       46316835694926478169428394003475163141307993866256225615783033603165251855960)


def _sumar(p, q):
    (x1, y1), (x2, y2) = p, q
    t = _Dd * x1 * x2 * y1 * y2 % _Pd
    return ((x1 * y2 + x2 * y1) * pow(1 + t, -1, _Pd) % _Pd, (y1 * y2 + x1 * x2) * pow(1 - t, -1, _Pd) % _Pd)


def _por(k, p):
    r = (0, 1)
    while k:
        if k & 1:
            r = _sumar(r, p)
        p = _sumar(p, p)
        k >>= 1
    return r


def _codificar(p):
    return (p[1] | ((p[0] & 1) << 255)).to_bytes(32, "little")


def _orden_pequeno(t):
    k, x = 1, t
    while x != (0, 1):
        x, k = _sumar(x, t), k + 1
    return k


def _torsion(orden):
    """Un punto de orden `orden` (2, 4 u 8): los de orden 8 son las raíces de los de orden 4."""
    i = pow(2, (_Pd - 1) // 4, _Pd)
    if orden == 2:
        return (0, _Pd - 1)
    if orden == 4:
        return (i, 0)
    # (x, y) con 2·(x, y) = (i, 0): y² = x² y y² − x² = ... ; se busca entre los múltiplos [L]·P de puntos fijos
    for semilla in range(1, 200):
        y = int.from_bytes(hashlib.sha256(b"153 remate torsion %d" % semilla).digest(), "little") % _Pd
        x2 = (y * y - 1) * pow(_Dd * y * y + 1, -1, _Pd) % _Pd
        if pow(x2, (_Pd - 1) // 2, _Pd) != 1:
            continue
        x = pow(x2, (_Pd + 3) // 8, _Pd)
        if (x * x - x2) % _Pd:
            x = x * i % _Pd
        t = _por(_Ld, (x, y))
        if _orden_pequeno(t) == 8:
            return t
    raise AssertionError("sin punto de orden 8")


def cadena_de_clave_mixta(orden, h_multiplo_de_8, rng):
    """Dos bloques encadenados firmados con A' = [a]B + T (T de orden `orden`): firmas Ed25519 de verdad para A' (S = r + h·a), con
    h = SHA-512(R ‖ A' ‖ M) mod L ≡ 0 (mod 8) en las dos si `h_multiplo_de_8` (entonces también vale SIN cofactor: OpenSSL dice
    «válida»), o con h ≢ 0 en la segunda (solo vale CON cofactor: @noble dice «válida»)."""
    import base64
    a = rng.randrange(1, _Ld)
    publica = _codificar(_sumar(_por(a, _Bd), _torsion(orden)))
    huella = hashlib.sha256(publica).hexdigest()
    out, prev = [], GENESIS
    for seq, (desde, hasta, h0) in enumerate((("10:00", "11:00", True), ("11:00", "12:00", h_multiplo_de_8))):
        kA, pubA, hA = clave(SEMILLA_A)
        b = bloque(kA, pubA, hA, seq, prev, f"2026-11-02T{desde}:00Z", f"2026-11-02T{hasta}:00Z", rng, 30 + seq, 0.0)
        b["servicio"]["clave_publica"], b["servicio"]["huella_clave"] = publica.hex(), huella
        msg = mensaje_firmado(b)
        while True:
            r = rng.randrange(1, _Ld)
            R = _codificar(_por(r, _Bd))
            h = int.from_bytes(hashlib.sha512(R + publica + msg).digest(), "little") % _Ld
            if (h % 8 == 0) == h0:
                break
        b["firma"] = {"alg": "ed25519", "valor": base64.b64encode(R + ((r + h * a) % _Ld).to_bytes(32, "little")).decode("ascii")}
        out.append(b)
        prev = hash_de_bloque(b)
    if h_multiplo_de_8:   # el diente del vector: sin la comprobación de orden, OpenSSL (`cryptography`) la daría por buena
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        for b in out:
            Ed25519PublicKey.from_public_bytes(publica).verify(base64.b64decode(b["firma"]["valor"]), mensaje_firmado(b))
    return out


def casos_del_remate_153(caso, kA, pubA, hA, L, b1):
    rng = random.Random(15307)
    # `hasta` es el FINAL de la hora UTC de la última petición: con minutos o segundos no es v1, y una ventana nunca está vacía
    for nombre, hasta in (("ventana_hasta_con_minutos", "2026-11-02T12:30:00Z"), ("ventana_hasta_con_segundos", "2026-11-02T12:00:01Z"),
                          ("ventana_vacia_desde_igual_a_hasta", "2026-11-02T11:00:00Z")):
        m = copy.deepcopy(b1); m["ventana"]["hasta"] = hasta
        caso(nombre, f"ventana.hasta = {hasta} (desde 11:00), firmado: no es el final de una hora o la ventana queda vacía",
             [L[0], linea(firmado(kA, m)), L[2]], {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    # fechas que casan con AAAA-MM-DDThh:mm:ssZ y no son del calendario (la aritmética de la regla entre cadenas las necesita reales)
    for nombre, desde, hasta in (("ventana_30_de_febrero", "2026-02-30T10:00:00Z", "2026-02-30T11:00:00Z"),
                                 ("ventana_hora_24", "2026-11-02T23:00:00Z", "2026-11-02T24:00:00Z"),
                                 ("ventana_29_de_febrero_de_2100_no_es_bisiesto", "2100-02-28T23:00:00Z", "2100-02-29T00:00:00Z")):
        m = copy.deepcopy(b1); m["ventana"] = {"desde": desde, "hasta": hasta}
        caso(nombre, f"ventana {desde} – {hasta}, firmado: no es una fecha del calendario", [L[0], linea(firmado(kA, m)), L[2]],
             {"estado": "no_se_pudo_comprobar", "motivo": "formato", "bloque": 1})
    g = bloque(kA, pubA, hA, 0, GENESIS, "2028-02-29T23:00:00Z", "2028-03-01T00:00:00Z", rng, 25, 0.0)
    caso("ventana_29_de_febrero_bisiesto", "Un bloque del 29 de febrero de 2028 (bisiesto) a medianoche: válido", [linea(g)],
         {"estado": "valida", "bloques": 1})
    # DS2: una ventana con menos de k peticiones sigue acumulando, así que un bloque puede durar varias horas
    largo = bloque(kA, pubA, hA, 0, GENESIS, "2026-11-02T10:00:00Z", "2026-11-02T14:00:00Z", rng, 21, 0.0)
    caso("ventana_de_varias_horas", "Un bloque de 10:00 a 14:00 (pocas peticiones por hora: no se cerró hasta llegar a k): válido",
         [linea(largo)], {"estado": "valida", "bloques": 1})
    # la regla entre cadenas con horas enteras: una cadena nueva puede empezar en la MISMA hora en que acaba la anterior, no antes
    A0 = cadena_de(kA, pubA, hA, 3, 10, rng, reinicio=None)              # 10-13 h: su último bloque, 12-13 h
    LA0 = [linea(b) for b in A0]
    misma = cadena_de(kA, pubA, hA, 2, 12, rng)                           # empieza a las 12:00, en la hora en que acaba A0
    caso("B1_reinicio_en_la_misma_hora_en_que_acaba_la_anterior",
         "El servicio se cae a las 12:20 con la última línea a medias y vuelve a las 12:25: la cadena nueva empieza a las 12:00 y la anterior "
         "acaba a las 13:00 (el final de su hora). Es honrado: válida", LA0 + [linea(b) for b in misma],
         {"estado": "valida", "bloques": 5, "reinicios": [3]})
    noche = cadena_de(kA, pubA, hA, 2, 22, rng, reinicio=None)            # 22-24 h: acaba a las 00:00 del día 3
    LN = [linea(b) for b in noche]
    for nombre, hora, esperado in (("B1_reinicio_en_la_misma_hora_pasada_la_medianoche", 23, {"estado": "valida", "bloques": 3, "reinicios": [2]}),
                                   ("B1_reinicio_una_hora_antes_pasada_la_medianoche", 22, {"estado": "no_coincide", "motivo": "cadena", "bloque": 2})):
        nueva = cadena_de(kA, pubA, hA, 1, hora, rng)
        caso(nombre, f"La cadena anterior acaba a las 00:00 del día 3; la nueva empieza a las {hora}:00 del día 2: la cuenta cruza el día",
             LN + [linea(nueva[0])], esperado)
    # M4: claves de orden MIXTO, [L]A' ≠ identidad. `clave` en el primer bloque, antes de verificar: así núcleo, WebCrypto (de cualquier
    # navegador) y @noble dicen lo mismo, sea cual sea su ecuación (con o sin cofactor)
    for orden in (2, 4, 8):
        caso(f"M4_clave_de_orden_mixto_{orden}_con_h_multiplo_de_8", f"A' = [a]B + T, T de orden {orden}, firmas con h ≡ 0 (mod 8): OpenSSL las da "
             "por buenas sin la comprobación de orden. `clave`", [linea(b) for b in cadena_de_clave_mixta(orden, True, rng)],
             {"estado": "no_coincide", "motivo": "clave", "bloque": 0})
    caso("M4_clave_de_orden_mixto_8_con_h_no_multiplo_de_8_en_el_segundo", "A' = [a]B + T, T de orden 8; la segunda firma con h ≢ 0 (mod 8): solo la "
         "acepta una comprobación CON cofactor (@noble). `clave`", [linea(b) for b in cadena_de_clave_mixta(8, False, rng)],
         {"estado": "no_coincide", "motivo": "clave", "bloque": 0})


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
