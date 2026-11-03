// 153-C1 · Comprobador de `matrixai.traza/v1` en JavaScript, SIN dependencias, y el corredor de los vectores.
//
// Es la contraparte de `matrixai/pipelines/traza.py` (núcleo): debe dar el MISMO estado, motivo y bloque en cada
// caso de `vectores_traza_v1.json`. C6 lo porta a `comprobar.ts` / `verificar.html` (WebCrypto en vez de `node:crypto`):
// por eso la plataforma (SHA-256 y Ed25519) se INYECTA y el resto es JS puro (TextEncoder, sin Buffer).
//
//   node comprobar_traza.mjs [vectores.json]        -> corre los vectores; sale con 1 si algún caso no da lo esperado
//
// Si este fichero y `traza.py` discrepan, manda `traza.py` y se corrige este.
// Tres trampas que tiene portar el formato (cada una tiene su vector):
//   - JSON.parse no distingue `52` de `52.0`, ni ve las claves duplicadas, ni un `-0`: por eso hay un lector propio.
//   - JCS ordena las claves por unidades UTF-16 (`<` de JS lo hace) y escribe las cadenas como JSON.stringify.
//   - un objeto normal con la clave `__proto__` se rompe: aquí los objetos leídos no tienen prototipo.

import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import nodeCrypto from 'node:crypto';

export const FORMATO = 'matrixai.traza/v1';
export const PREFIJO_FIRMA = 'matrixai.traza.firma/v1\n';
export const GENESIS = '0'.repeat(64);
export const MAX_ENTERO = Number.MAX_SAFE_INTEGER; // 2^53 - 1
// Los MISMOS topes que `traza.py` (I2), comprobados ANTES de leer el JSON: 1 MiB por línea (bytes UTF-8, sin su `\n`) y 64 MiB el
// fichero (con líneas sueltas, la suma de bytes + 1 de cada una). Sin ellos, una línea de 100 MB dejaba a Node sin memoria.
export const TOPE_LINEA = 1 << 20;
export const TOPE_FICHERO = 64 << 20;
const PROFUNDIDAD_MAXIMA = 64;
const MAX_CIFRAS = 4300; // el límite de int() de Python: un entero más largo es «formato» en los dos
const CAMPOS = ['formato', 'seq', 'prev', 'servicio', 'ventana', 'n', 'k_minimo', 'incompleto', 'reinicio', 'salida', 'columnas', 'firma'];
const HEX64 = /^[0-9a-f]{64}$/;
const FECHA = /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/;

export const VALIDA = 'valida';
export const NO_COINCIDE = 'no_coincide';
export const NO_SE_PUDO_COMPROBAR = 'no_se_pudo_comprobar';

class Formato extends Error {
  constructor(motivo, es, en) { super(es); this.motivo = motivo; this.detalle = { es, en }; }
}
const mal = (es, en, motivo = 'formato') => new Formato(motivo, es, en);
const t = (es, en) => ({ es, en });
const plana = (s) => JSON.parse(JSON.stringify(s)); // una copia que no comparte memoria con la línea leída

// ───────────────────────── lector JSON estricto ─────────────────────────

function leerJSON(texto) {
  let i = 0;
  const ws = () => {
    for (;;) {
      const u = texto.charCodeAt(i);
      if (u === 0x20 || u === 0x09 || u === 0x0a || u === 0x0d) i++; else return;
    }
  };
  const fallo = (m) => mal(`no es JSON: ${m}`, `not JSON: ${m}`);

  const SIMPLE = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
  function cadena() {
    i++; // "
    // Los tramos sin escapes se copian con `slice` (no carácter a carácter con `s += c`, que con una cadena de
    // millones de caracteres crea millones de cadenas intermedias).
    const trozos = [];
    let ini = i;
    for (;;) {
      if (i >= texto.length) throw fallo('cadena sin cerrar');
      const u = texto.charCodeAt(i);
      if (u === 0x22) { trozos.push(texto.slice(ini, i)); i++; break; }
      if (u < 0x20) throw fallo('carácter de control sin escapar');
      if (u === 0x5c) {
        trozos.push(texto.slice(ini, i));
        const e = texto[i + 1];
        if (e !== undefined && Object.prototype.hasOwnProperty.call(SIMPLE, e)) { trozos.push(SIMPLE[e]); i += 2; ini = i; continue; }
        if (e === 'u') {
          const h = texto.slice(i + 2, i + 6);
          if (!/^[0-9a-fA-F]{4}$/.test(h)) throw fallo('escape \\u incorrecto');
          trozos.push(String.fromCharCode(parseInt(h, 16))); i += 6; ini = i; continue;
        }
        throw fallo('escape desconocido');
      }
      i++;
    }
    const s = trozos.join('');
    for (let k = 0; k < s.length; k++) { // un sustituto UTF-16 suelto no tiene UTF-8 que firmar
      const u = s.charCodeAt(k);
      if (u >= 0xd800 && u <= 0xdbff && s.charCodeAt(k + 1) >= 0xdc00 && s.charCodeAt(k + 1) <= 0xdfff) k++;
      else if (u >= 0xd800 && u <= 0xdfff) throw mal('una cadena lleva un sustituto UTF-16 suelto', 'a string carries a lone UTF-16 surrogate');
    }
    return s;
  }

  function numero() {
    const m = /^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/.exec(texto.slice(i, i + MAX_CIFRAS + 40));
    if (!m) throw fallo('valor inesperado');
    const tok = m[0];
    i += tok.length;
    if (m[2] !== undefined || m[3] !== undefined) throw mal(`número no entero: ${tok}`, `not an integer: ${tok}`);
    if (tok.startsWith('-')) throw mal(`entero negativo: ${tok}`, `negative integer: ${tok}`);
    if (tok.length > MAX_CIFRAS) throw fallo('entero demasiado largo');
    const n = Number(tok);
    return Number.isSafeInteger(n) ? n : BigInt(tok); // un BigInt nunca pasa de `entero()`
  }

  function valor(d) {
    if (d > PROFUNDIDAD_MAXIMA) throw mal('anidación excesiva', 'nesting too deep');
    ws();
    const c = texto[i];
    if (c === '{') {
      i++;
      const o = Object.create(null);
      ws();
      if (texto[i] === '}') { i++; return o; }
      for (;;) {
        ws();
        if (texto[i] !== '"') throw fallo('se esperaba una clave');
        const k = cadena();
        if (k in o) throw mal(`clave duplicada: ${JSON.stringify(k)}`, `duplicate key: ${JSON.stringify(k)}`);
        ws();
        if (texto[i] !== ':') throw fallo('se esperaba ":"');
        i++;
        o[k] = valor(d + 1);
        ws();
        if (texto[i] === ',') { i++; continue; }
        if (texto[i] === '}') { i++; return o; }
        throw fallo('se esperaba "," o "}"');
      }
    }
    if (c === '[') {
      i++;
      const a = [];
      ws();
      if (texto[i] === ']') { i++; return a; }
      for (;;) {
        a.push(valor(d + 1));
        ws();
        if (texto[i] === ',') { i++; continue; }
        if (texto[i] === ']') { i++; return a; }
        throw fallo('se esperaba "," o "]"');
      }
    }
    if (c === '"') return cadena();
    if (texto.startsWith('true', i)) { i += 4; return true; }
    if (texto.startsWith('false', i)) { i += 5; return false; }
    if (texto.startsWith('null', i)) { i += 4; return null; }
    if (texto.startsWith('NaN', i) || texto.startsWith('Infinity', i) || texto.startsWith('-Infinity', i)) {
      throw mal('constante no permitida', 'constant not allowed');
    }
    return numero();
  }

  const v = valor(0);
  ws();
  if (i !== texto.length) throw fallo('texto después del valor');
  return v;
}

// ───────────────────────── JCS (RFC 8785) para el subconjunto de v1 ─────────────────────────
// v1 solo lleva enteros, cadenas, booleanos, null, listas y objetos: JSON.stringify de una cadena ya es JCS
// (escapes cortos, \u00xx en minúscula para el resto de controles, UTF-8 real) y `<` ordena por unidades UTF-16.

export function jcs(x) {
  if (x === null) return 'null';
  if (x === true) return 'true';
  if (x === false) return 'false';
  if (typeof x === 'string') return JSON.stringify(x);
  if (typeof x === 'number') return String(x);
  if (Array.isArray(x)) return '[' + x.map(jcs).join(',') + ']';
  return '{' + Object.keys(x).sort().map((k) => JSON.stringify(k) + ':' + jcs(x[k])).join(',') + '}';
}
const utf8 = (s) => new TextEncoder().encode(s);
export const jcsBytes = (x) => utf8(jcs(x));

// ───────────────────────── forma y coherencia ─────────────────────────

// un entero fuera de rango se lee como BigInt, que JSON.stringify no sabe escribir
const mostrar = (x) => JSON.stringify(x, (k, v) => (typeof v === 'bigint' ? v.toString() : v));
const esObjeto = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const mismasClaves = (o, lista) => { const k = Object.keys(o).sort(); const l = [...lista].sort(); return k.length === l.length && k.every((x, j) => x === l[j]); };

function entero(x, nombre) {
  if (typeof x !== 'number' || !Number.isInteger(x) || x < 0 || x > MAX_ENTERO) {
    throw mal(`${nombre} no es un entero 0..2^53-1`, `${nombre} is not an integer 0..2^53-1`);
  }
  return x;
}
function listaDeEnteros(x, largo, nombre) {
  if (!Array.isArray(x) || x.length !== largo) throw mal(`${nombre} no es una lista de ${largo} enteros`, `${nombre} is not a list of ${largo} integers`);
  let s = 0;
  for (const v of x) s += entero(v, nombre);
  return s;
}
function histograma(h, nombre) {
  if (!esObjeto(h) || !mismasClaves(h, ['bins', 'fuera'])) throw mal(`${nombre}: campos incorrectos`, `${nombre}: wrong fields`);
  return [listaDeEnteros(h.bins, 10, `${nombre}.bins`), entero(h.fuera, `${nombre}.fuera`)];
}

function estructura(b, coherencia = false) {
  if (!esObjeto(b)) throw mal('el bloque no es un objeto', 'the block is not an object');
  if (b.formato !== FORMATO) throw mal(`formato desconocido: ${mostrar(b.formato)}`, `unknown format: ${mostrar(b.formato)}`, 'version');
  if (!mismasClaves(b, CAMPOS)) throw mal('campos distintos de los de v1', 'fields differ from v1');
  entero(b.seq, 'seq');
  const n = entero(b.n, 'n');
  const k = entero(b.k_minimo, 'k_minimo');
  if (k < 1) throw mal('k_minimo debe ser >= 1', 'k_minimo must be >= 1');
  if (typeof b.prev !== 'string' || !HEX64.test(b.prev)) throw mal('prev no es SHA-256 en hex minúscula', 'prev is not a lowercase hex SHA-256');
  const s = b.servicio;
  if (!esObjeto(s) || !mismasClaves(s, ['id', 'huella_clave', 'clave_publica', 'huella_modelo'])) throw mal('servicio: campos incorrectos', 'servicio: wrong fields');
  if (typeof s.id !== 'string' || [...s.id].length < 1 || [...s.id].length > 128) throw mal('servicio.id: de 1 a 128 caracteres', 'servicio.id: 1 to 128 characters');
  for (const campo of ['huella_clave', 'clave_publica', 'huella_modelo']) {
    if (typeof s[campo] !== 'string' || !HEX64.test(s[campo])) throw mal(`servicio.${campo} no es hex de 64`, `servicio.${campo} is not 64 hex characters`);
  }
  const v = b.ventana;
  if (!esObjeto(v) || !mismasClaves(v, ['desde', 'hasta'])) throw mal('ventana: campos incorrectos', 'ventana: wrong fields');
  for (const campo of ['desde', 'hasta']) {
    if (typeof v[campo] !== 'string' || !FECHA.test(v[campo])) throw mal(`ventana.${campo} no es AAAA-MM-DDThh:mm:ssZ`, `ventana.${campo} is not YYYY-MM-DDThh:mm:ssZ`);
  }
  if (v.desde > v.hasta) throw mal('ventana.desde > ventana.hasta', 'ventana.desde > ventana.hasta');
  if (typeof b.incompleto !== 'boolean') throw mal('incompleto no es booleano', 'incompleto is not a boolean');
  if (b.reinicio !== null && b.reinicio !== 'sin_previo') throw mal('reinicio: solo null o "sin_previo"', 'reinicio: only null or "sin_previo"');
  const f = b.firma;
  if (!esObjeto(f) || !mismasClaves(f, ['alg', 'valor']) || f.alg !== 'ed25519' || typeof f.valor !== 'string') throw mal('firma: se espera {alg:"ed25519", valor:<base64>}', 'firma: expected {alg:"ed25519", valor:<base64>}');
  if (b.incompleto) {
    if (b.salida !== null || b.columnas !== null) throw mal('incompleto: salida y columnas deben ser null (solo n)', 'incompleto: salida and columnas must be null (only n)');
    if (coherencia && n >= k) throw mal('incompleto con n >= k_minimo: se habría cerrado entero', 'incompleto with n >= k_minimo: it would have closed in full');
    return;
  }
  if (coherencia && n < k) throw mal('un bloque completo con n < k_minimo (DS2: no se cierra)', 'a complete block with n < k_minimo (DS2: it does not close)');
  const sal = b.salida;
  if (!esObjeto(sal) || !mismasClaves(sal, ['probabilidad', 'etiquetas', 'valor'])) throw mal('salida: campos incorrectos', 'salida: wrong fields');
  if (sal.probabilidad !== null && listaDeEnteros(sal.probabilidad, 10, 'salida.probabilidad') !== n && coherencia) throw mal('salida.probabilidad no suma n', 'salida.probabilidad does not add up to n');
  if (sal.etiquetas !== null) {
    const e = sal.etiquetas;
    if (!esObjeto(e)) throw mal('salida.etiquetas no es un objeto', 'salida.etiquetas is not an object');
    let suma = 0;
    for (const c of Object.values(e)) suma += entero(c, 'etiquetas');
    if (coherencia && suma !== n) throw mal('salida.etiquetas no suma n', 'salida.etiquetas does not add up to n');
  }
  if (sal.valor !== null) {
    const [suma, fuera] = histograma(sal.valor, 'salida.valor');
    if (sal.probabilidad !== null || sal.etiquetas !== null) throw mal('salida.valor (regresión) con probabilidad o etiquetas: deben ser null', 'salida.valor (regression) with probabilidad or etiquetas: they must be null');
    if (coherencia && (suma !== n || fuera > suma)) throw mal('salida.valor: sus casillas no suman n o fuera > casillas', 'salida.valor: its bins do not add up to n or fuera > bins');
  }
  const cols = b.columnas;
  if (!esObjeto(cols)) throw mal('columnas no es un objeto', 'columnas is not an object');
  for (const nombre of Object.keys(cols)) {
    const c = cols[nombre];
    if (!esObjeto(c) || (c.tipo !== 'numerica' && c.tipo !== 'categorica')) throw mal(`columna ${JSON.stringify(nombre)}: tipo`, `column ${JSON.stringify(nombre)}: type`);
    if (c.tipo === 'numerica') {
      if (!mismasClaves(c, ['tipo', 'bins', 'faltantes', 'fuera'])) throw mal(`columna ${JSON.stringify(nombre)}: campos`, `column ${JSON.stringify(nombre)}: fields`);
      const suma = listaDeEnteros(c.bins, 10, 'bins');
      if ((suma + entero(c.faltantes, 'faltantes') !== n || entero(c.fuera, 'fuera') > suma) && coherencia) throw mal(`columna ${JSON.stringify(nombre)}: bins + faltantes != n, o fuera > bins`, `column ${JSON.stringify(nombre)}: bins + faltantes != n, or fuera > bins`);
    } else {
      if (!mismasClaves(c, ['tipo', 'nuevas', 'faltantes'])) throw mal(`columna ${JSON.stringify(nombre)}: campos`, `column ${JSON.stringify(nombre)}: fields`);
      if (entero(c.nuevas, 'nuevas') + entero(c.faltantes, 'faltantes') > n && coherencia) throw mal(`columna ${JSON.stringify(nombre)}: nuevas + faltantes > n`, `column ${JSON.stringify(nombre)}: nuevas + faltantes > n`);
    }
  }
}

// ───────────────────────── bytes y plataforma ─────────────────────────

const aHex = (u8) => [...u8].map((x) => x.toString(16).padStart(2, '0')).join('');
const deHex = (h) => Uint8Array.from(h.match(/../g).map((x) => parseInt(x, 16)));
const ALFABETO = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/** Base64 estándar CANÓNICO de exactamente 64 bytes (88 caracteres, `==`), o null. Una sola forma por firma. */
function firmaDeBase64(texto) {
  if (!/^[A-Za-z0-9+/]{86}==$/.test(texto)) return null;
  const out = new Uint8Array(64);
  let acc = 0, bits = 0, j = 0;
  for (const ch of texto.slice(0, 86)) {
    acc = (acc << 6) | ALFABETO.indexOf(ch); bits += 6;
    if (bits >= 8) { bits -= 8; if (j < 64) out[j++] = (acc >> bits) & 0xff; acc &= (1 << bits) - 1; }
  }
  if (acc !== 0) return null; // los 4 bits que sobran, a cero
  return out;
}

const SPKI_ED25519 = deHex('302a300506032b6570032100' + '00'.repeat(32)).slice(0, 12);

/** La plataforma de Node. En el navegador (C6): `crypto.subtle.digest` e `importKey('raw', …, 'Ed25519')`. */
export const plataformaNode = {
  sha256: (bytes) => nodeCrypto.createHash('sha256').update(bytes).digest('hex'),
  ed25519: (crudo32, mensaje, firma64) => {
    try {
      const clave = nodeCrypto.createPublicKey({ key: Buffer.concat([SPKI_ED25519, crudo32]), format: 'der', type: 'spki' });
      return nodeCrypto.verify(null, mensaje, clave, firma64);
    } catch { return false; } // 32 bytes que no son un punto de la curva: no firmaron nada
  },
};

const concat = (a, b) => { const r = new Uint8Array(a.length + b.length); r.set(a); r.set(b, a.length); return r; };
export const mensajeFirmado = (b) => { const { firma, ...sin } = b; return concat(utf8(PREFIJO_FIRMA), jcsBytes(sin)); };
export const hashDeBloque = async (b, p = plataformaNode) => p.sha256(jcsBytes(b));

const res = (estado, motivo = null, bloque = null, detalle = null, extra = {}) => ({ estado, motivo, bloque, detalle, ...extra });

// ───────────────────────── claves y firmas Ed25519 que no valen (I1), como en `traza.py` ─────────────────────────
// OpenSSL acepta claves de ORDEN PEQUEÑO y codificaciones no canónicas: con la clave identidad y R = identidad, S = 0, cualquier
// mensaje «está firmado». Se decide ANTES de verificar, con aritmética propia (BigInt), para no depender de la biblioteca ni del
// navegador (C6, WebCrypto).
const P = 2n ** 255n - 19n;
const L_ORDEN = 2n ** 252n + 27742317777372353535851937790883648493n;
const potencia = (b, e, m) => { let r = 1n; b %= m; while (e > 0n) { if (e & 1n) r = (r * b) % m; b = (b * b) % m; e >>= 1n; } return r; };
const inverso = (a) => potencia(((a % P) + P) % P, P - 2n, P);
const D = ((-121665n * inverso(121666n)) % P + P) % P;
const Y8 = 0x7a03ac9277fdc74ec6cc392cfa53202a0f67100d760b3cba4fd84d3d706a17c7n;
const Y_DE_ORDEN_PEQUENO = new Set([0n, 1n, P - 1n, Y8, P - Y8].map(String));
const enteroLE = (u8) => { let x = 0n; for (let k = u8.length - 1; k >= 0; k--) x = (x << 8n) | BigInt(u8[k]); return x; };
const yDe = (crudo32) => enteroLE(crudo32) & ((1n << 255n) - 1n);

export function clavePublicaUtilizable(crudo32) {
  const y = yDe(crudo32);
  if (y >= P || Y_DE_ORDEN_PEQUENO.has(String(y))) return false;
  const x2 = ((y * y - 1n) * inverso(D * y * y + 1n)) % P;
  return x2 === 0n || potencia(x2, (P - 1n) / 2n, P) === 1n;
}
export function firmaBienFormada(firma64) {
  const yR = yDe(firma64.slice(0, 32));
  return yR < P && !Y_DE_ORDEN_PEQUENO.has(String(yR)) && enteroLE(firma64.slice(32)) < L_ORDEN;
}

// ───────────────────────── topes de tamaño (I2) ─────────────────────────

/** Bytes UTF-8 de una línea; un sustituto suelto cuenta 3 (como `surrogatepass` en Python). */
function bytesUtf8(s) {
  if (s.length > TOPE_LINEA) return s.length; // las unidades UTF-16 nunca son más que los bytes
  let n = 0;
  for (let k = 0; k < s.length; k++) {
    const u = s.charCodeAt(k);
    if (u < 0x80) n += 1;
    else if (u < 0x800) n += 2;
    else if (u >= 0xd800 && u <= 0xdbff && k + 1 < s.length && s.charCodeAt(k + 1) >= 0xdc00 && s.charCodeAt(k + 1) <= 0xdfff) { n += 4; k++; }
    else n += 3;
  }
  return n;
}
const FICHERO_GRANDE = t(`el fichero pesa más de ${TOPE_FICHERO >> 20} MiB, el tope de una traza (el servicio escribe un fichero por día: compruébalos por separado)`,
  `the file is over ${TOPE_FICHERO >> 20} MiB, the limit for one trace (the service writes one file per day: check them one by one)`);
const lineaGrande = (i) => res(NO_SE_PUDO_COMPROBAR, 'formato', i, t(`la línea pesa más de ${TOPE_LINEA >> 20} MiB, el tope de un bloque`, `the line is over ${TOPE_LINEA >> 20} MiB, the limit for one block`));
function fueraDeTope(lineas) {
  let total = 0;
  for (let i = 0; i < lineas.length; i++) {
    const n = bytesUtf8(lineas[i]);
    if (n > TOPE_LINEA) return lineaGrande(i);
    total += n + 1;
  }
  if (total > TOPE_FICHERO) return res(NO_SE_PUDO_COMPROBAR, 'formato', null, FICHERO_GRANDE);
  return null;
}
/** `fueraDeTope` sobre los BYTES de un fichero sin partirlo (I2 de la re-auditoría: `split` de 64 MiB de líneas diminutas eran
 *  502–1.013 MB): una línea pasa del tope si y solo si hay TOPE_LINEA + 1 bytes seguidos sin `\n`, así que se salta de ventana en
 *  ventana al ÚLTIMO salto de cada una. El índice de la línea son los saltos de antes. Como `_fuera_de_tope_en_bytes` en Python. */
function fueraDeTopeEnBytes(bytes) {
  const n = bytes.length;
  let pos = 0;
  while (n - pos > TOPE_LINEA) {
    const j = bytes.lastIndexOf(0x0a, pos + TOPE_LINEA);
    if (j < pos) { // TOPE_LINEA + 1 bytes sin salto desde el principio de una línea
      let saltos = 0;
      for (let k = 0; k < pos; k++) if (bytes[k] === 0x0a) saltos++;
      return lineaGrande(saltos);
    }
    pos = j + 1;
  }
  const total = n + (n && bytes[n - 1] !== 0x0a ? 1 : 0); // cada línea, más su `\n` (la última puede no llevarlo)
  if (total > TOPE_FICHERO) return res(NO_SE_PUDO_COMPROBAR, 'formato', null, FICHERO_GRANDE);
  return null;
}
/** `(inicio, fin)` de cada línea, sin crear la lista: las mismas que `split('\n')` quitando la última vacía. */
function* tramos(bytes) {
  let i = 0;
  while (i < bytes.length) {
    const j = bytes.indexOf(0x0a, i);
    if (j < 0) { yield [i, bytes.length]; return; }
    yield [i, j];
    i = j + 1;
  }
}

// ───────────────────────── el comprobador ─────────────────────────

export function lineasDeTexto(texto) {
  const partes = texto.split('\n');
  if (partes.length && partes[partes.length - 1] === '') partes.pop();
  return partes;
}

async function falloDeFirma(b, plataforma, utilizables = new Map()) {
  const s = b.servicio;
  const crudo = deHex(s.clave_publica);
  if (plataforma.sha256(crudo) !== s.huella_clave) return ['clave', t('la huella no es la de la clave pública que lleva el bloque', 'the fingerprint is not the one of the public key the block carries')];
  if (!utilizables.has(s.clave_publica)) utilizables.set(s.clave_publica, clavePublicaUtilizable(crudo)); // una vez por clave
  if (!utilizables.get(s.clave_publica)) {
    return ['clave', t('la clave pública no es una clave Ed25519 utilizable (no canónica, fuera de la curva o de orden pequeño): con ella cualquier contenido parecería firmado',
      'the public key is not a usable Ed25519 key (non-canonical, off the curve or of small order): with it any content would look signed')];
  }
  const firma = firmaDeBase64(b.firma.valor);
  if (firma === null || !firmaBienFormada(firma) || !(await plataforma.ed25519(crudo, mensajeFirmado(b), firma))) {
    return ['firma', t('la firma no coincide con el contenido del bloque', 'the signature does not match the block\'s content')];
  }
  return null;
}

/** Mismo contrato que `comprobar_traza` del núcleo. `lineas`: textos; `opciones`: {huellaAnclada, esperarGenesis}.
 *  Cada bloque se lee una vez y de él se guarda solo lo que la cadena necesita; el orden de los fallos es el de Python. */
export async function comprobarTraza(lineas, { huellaAnclada = null, esperarGenesis = false } = {}, plataforma = plataformaNode) {
  if (huellaAnclada !== null && !HEX64.test(huellaAnclada)) throw new Error('la huella anclada es SHA-256 en hexadecimal minúscula (64 caracteres)');
  return comprobar(lineas, fueraDeTope(lineas), huellaAnclada, esperarGenesis, plataforma);
}

/** El comprobador sobre líneas ya medidas (`fuera`: los topes); `lineas`, cualquier iterable (con un fichero, un generador que
 *  decodifica cada línea al leerla). Como `_comprobar` en Python: del primer fallo de firma y del de coherencia se guarda solo ese. */
async function comprobar(lineas, fuera, huellaAnclada, esperarGenesis, plataforma) {
  if (fuera) return fuera;
  let registros = [];
  const utilizables = new Map();
  let primeraFirma = null, primeraCoherencia = null, vistas = 0, i = -1;
  for (const linea of lineas) {
    i++;
    let b;
    try {
      b = leerJSON(linea);
      estructura(b);
    } catch (e) {
      if (e instanceof Formato) return res(NO_SE_PUDO_COMPROBAR, e.motivo, i, e.detalle);
      throw e;
    }
    vistas++;
    if (primeraFirma) continue;
    const firma = await falloDeFirma(b, plataforma, utilizables);
    if (firma) { primeraFirma = [i, ...firma]; registros = []; continue; }
    if (primeraCoherencia) continue;
    try { estructura(b, true); } catch (e) {
      if (!(e instanceof Formato)) throw e;
      primeraCoherencia = [i, e.motivo, e.detalle]; registros = []; continue;
    }
    // `plana`: el lector JSON saca las cadenas con `slice`, y en V8 un trozo retiene la línea entera: sin la copia, cada registro
    // guardaba su línea (medido: 64 MiB de bloques mínimos, 409 MB; con la copia, 306 MB)
    registros.push({ seq: b.seq, prev: plana(b.prev), hash: await hashDeBloque(b, plataforma), huella: plana(b.servicio.huella_clave),
      reinicio: b.reinicio, desde: plana(b.ventana.desde), hasta: plana(b.ventana.hasta) });
  }
  if (!vistas) return res(NO_SE_PUDO_COMPROBAR, 'formato', null, t('fichero sin bloques', 'file without blocks'));
  if (primeraFirma) return res(NO_COINCIDE, primeraFirma[1], primeraFirma[0], primeraFirma[2]);
  if (primeraCoherencia) return res(NO_SE_PUDO_COMPROBAR, primeraCoherencia[1], primeraCoherencia[0], primeraCoherencia[2]);
  const fallo = falloDeCadena(registros, esperarGenesis);
  if (fallo) return fallo;
  const huellas = [...new Set(registros.map((r) => r.huella))]; // sin repetir, en orden de aparición (un Set, no `includes` sobre un array: era cuadrático)
  if (huellaAnclada !== null) {
    for (let i = 0; i < registros.length; i++) {
      if (registros[i].huella !== huellaAnclada) return res(NO_COINCIDE, 'ancla', i, t('la huella de este bloque no es la anclada', 'this block\'s fingerprint is not the anchored one'));
    }
  }
  const reinicios = [];
  registros.forEach((r, i) => { if (r.reinicio === 'sin_previo') reinicios.push(i); });
  const inicios = [...new Set([0, ...reinicios])].sort((a, b) => a - b);
  const cadenas = inicios.map((a, k) => {
    const z = k + 1 < inicios.length ? inicios[k + 1] - 1 : registros.length - 1;
    return { inicio: a, fin: z, seq_inicial: registros[a].seq, seq_final: registros[z].seq, huella: registros[a].huella };
  });
  const ultimo = registros[registros.length - 1];
  return res(VALIDA, null, null, null, { bloques: registros.length, empieza_en: registros[0].seq, ultimo_seq: ultimo.seq, ultimo_hash: ultimo.hash, huellas, reinicios, cadenas, anclada: huellaAnclada !== null });
}

function falloDeCadena(registros, esperarGenesis) {
  const vistos = new Set();
  const ultimoHasta = new Map(); // huella -> `hasta` del último bloque visto de esa clave
  for (let i = 0; i < registros.length; i++) {
    const b = registros[i];
    if (i === 0 && esperarGenesis && !(b.seq === 0 && b.prev === GENESIS && b.reinicio === null)) {
      return res(NO_COINCIDE, 'cadena', 0, t('se esperaba el bloque génesis (seq 0, prev de ceros, sin reinicio) y el fichero empieza más tarde o en un reinicio sin_previo', 'the genesis block (seq 0, prev of zeros, no restart) was expected and the file starts later or at a sin_previo restart'));
    }
    if (vistos.has(b.hash)) return res(NO_COINCIDE, 'cadena', i, t('este bloque ya estaba antes en el fichero (repetido)', 'this block was already earlier in the file (repeated)'));
    vistos.add(b.hash);
    if (b.reinicio === 'sin_previo') {
      if (!(b.seq === 0 && b.prev === GENESIS)) return res(NO_COINCIDE, 'cadena', i, t('un reinicio sin_previo debe tener seq 0 y prev de ceros', 'a sin_previo restart must have seq 0 and a prev of zeros'));
      if (ultimoHasta.has(b.huella) && b.desde < ultimoHasta.get(b.huella)) {
        const h = ultimoHasta.get(b.huella);
        return res(NO_COINCIDE, 'cadena', i, t(`esta cadena nueva empieza (${b.desde}) antes de que acabe la anterior de la misma clave (${h}): cadenas cambiadas de orden o repetidas`,
          `this new chain starts (${b.desde}) before the previous one of the same key ends (${h}): chains out of order or repeated`));
      }
    } else if ((b.seq === 0) !== (b.prev === GENESIS)) {
      return res(NO_COINCIDE, 'cadena', i, t('bloque incoherente: el génesis (seq 0) lleva prev de ceros y ningún otro lo lleva', 'incoherent block: the genesis block (seq 0) has a prev of zeros and no other block has one'));
    } else if (i > 0) {
      const a = registros[i - 1];
      if (b.seq !== a.seq + 1) return res(NO_COINCIDE, 'cadena', i, t(`seq ${b.seq} no sigue a ${a.seq}`, `seq ${b.seq} does not follow ${a.seq}`));
      if (b.prev !== a.hash) return res(NO_COINCIDE, 'cadena', i, t('prev no es el SHA-256 del bloque anterior', 'prev is not the SHA-256 of the previous block'));
      if (b.huella !== a.huella) return res(NO_COINCIDE, 'cadena', i, t('la clave cambia dentro de la cadena sin un reinicio sin_previo', 'the key changes inside the chain without a sin_previo restart'));
    }
    ultimoHasta.set(b.huella, b.hasta);
  }
  return null;
}

/** Como `comprobar_fichero` del núcleo, sobre los BYTES del fichero: el tope antes de decodificar y UTF-8 estricto (un BOM no
 *  se quita: la primera línea deja de ser JSON y es `formato`, como en Python). */
export async function comprobarFichero(bytes, opciones = {}, plataforma = plataformaNode) {
  if (opciones.huellaAnclada != null && !HEX64.test(opciones.huellaAnclada)) throw new Error('la huella anclada es SHA-256 en hexadecimal minúscula (64 caracteres)');
  if (bytes.length > TOPE_FICHERO) return res(NO_SE_PUDO_COMPROBAR, 'formato', null, FICHERO_GRANDE);
  try { // UTF-8 estricto a trozos de 1 MiB, sin crear el texto entero (como `_es_utf8`)
    const dec = new TextDecoder('utf-8', { fatal: true });
    for (let k = 0; k < bytes.length; k += 1 << 20) dec.decode(bytes.subarray(k, k + (1 << 20)), { stream: true });
    dec.decode();
  } catch {
    return res(NO_SE_PUDO_COMPROBAR, 'formato', null, t('el fichero no es UTF-8', 'the file is not UTF-8'));
  }
  const lector = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }); // un BOM se queda: la línea deja de ser JSON
  function* lineas() { for (const [a, z] of tramos(bytes)) yield lector.decode(bytes.subarray(a, z)); }
  return comprobar(lineas(), fueraDeTopeEnBytes(bytes), opciones.huellaAnclada ?? null, opciones.esperarGenesis ?? false, plataforma);
}

// ───────────────────────── los casos de los vectores que no caben escritos ─────────────────────────
// Una línea puede venir como {"base": <línea>, "rellenar_con_espacios_hasta": N}: la línea más espacios hasta N bytes UTF-8
// (JSON válido: el espacio al final lo es). Y un caso puede traer el FICHERO en vez de las líneas: {"fichero_b64": …},
// {"fichero_de_espacios": N} o {"fichero_de_lineas": [<líneas como arriba>], "salto_final": bool} (unidas por `\n`). Así caben en el JSON los casos de los topes sin que pese megas. Igual en las pruebas de Python.
export function expandirLinea(x) {
  if (typeof x === 'string') return x;
  return x.base + ' '.repeat(x.rellenar_con_espacios_hasta - bytesUtf8(x.base));
}
export function bytesDelCaso(caso) {
  if (caso.fichero_b64 !== undefined) return Uint8Array.from(atob(caso.fichero_b64), (c) => c.charCodeAt(0));
  if (caso.fichero_de_espacios !== undefined) return new Uint8Array(caso.fichero_de_espacios).fill(0x20);
  if (caso.fichero_de_lineas !== undefined) return new TextEncoder().encode(caso.fichero_de_lineas.map(expandirLinea).join('\n') + (caso.salto_final ? '\n' : ''));
  return null;
}
export async function comprobarCaso(caso, plataforma = plataformaNode) {
  const opciones = { huellaAnclada: caso.ancla ?? null, esperarGenesis: caso.esperar_genesis ?? false };
  const bytes = bytesDelCaso(caso);
  if (bytes !== null) return comprobarFichero(bytes, opciones, plataforma);
  return comprobarTraza(caso.lineas.map(expandirLinea), opciones, plataforma);
}

// ───────────────────────── corredor de vectores ─────────────────────────

async function main(argv) {
  const ruta = argv[2] ?? fileURLToPath(new URL('./vectores_traza_v1.json', import.meta.url));
  const v = JSON.parse(readFileSync(ruta, 'utf8'));
  const fallos = [];
  for (const [nombre, k] of Object.entries(v.claves_de_prueba)) {
    if (plataformaNode.sha256(deHex(k.publica_hex)) !== k.huella) fallos.push(`clave ${nombre}: la pública no da la huella declarada`);
  }
  if (v.prefijo_de_firma !== PREFIJO_FIRMA) fallos.push('prefijo de firma distinto');
  for (const c of v.cadena_valida.bloques) {
    const b = leerJSON(c.linea);
    const { firma, ...sin } = b;
    if (jcs(sin) !== c.canonico_sin_firma) fallos.push(`canónico del bloque ${b.seq} distinto`);
    if (plataformaNode.sha256(mensajeFirmado(b)) !== c.sha256_del_mensaje_firmado) fallos.push(`mensaje firmado del bloque ${b.seq} distinto`);
    if (await hashDeBloque(b) !== c.hash_del_bloque) fallos.push(`hash del bloque ${b.seq} distinto`);
  }
  for (const caso of v.casos) {
    const r = await comprobarCaso(caso);
    const malos = Object.entries(caso.esperado).filter(([k, x]) => JSON.stringify(r[k] ?? null) !== JSON.stringify(x));
    console.log(`${malos.length ? 'MAL' : 'ok '} ${caso.nombre.padEnd(46)} ${r.estado.padEnd(22)} ${String(r.motivo).padEnd(8)} bloque=${r.bloque}`);
    if (malos.length) fallos.push(`${caso.nombre}: ${JSON.stringify(malos.map(([k, x]) => [k, x, r[k]]))}`);
  }
  console.log(`\n${v.casos.length} casos, ${fallos.length} fallos`);
  for (const f of fallos) console.log('FALLO:', f);
  return fallos.length ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv).then((c) => process.exit(c));
}
