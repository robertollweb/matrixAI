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

// ───────────────────────── lector JSON estricto ─────────────────────────

function leerJSON(texto) {
  let i = 0;
  const WS = ' \t\n\r';
  const ws = () => { while (i < texto.length && WS.includes(texto[i])) i++; };
  const fallo = (m) => mal(`no es JSON: ${m}`, `not JSON: ${m}`);

  function cadena() {
    i++; // "
    let s = '';
    for (;;) {
      if (i >= texto.length) throw fallo('cadena sin cerrar');
      const c = texto[i];
      if (c === '"') { i++; break; }
      if (c.charCodeAt(0) < 0x20) throw fallo('carácter de control sin escapar');
      if (c === '\\') {
        const e = texto[i + 1];
        const simple = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
        if (e in simple) { s += simple[e]; i += 2; continue; }
        if (e === 'u') {
          const h = texto.slice(i + 2, i + 6);
          if (!/^[0-9a-fA-F]{4}$/.test(h)) throw fallo('escape \\u incorrecto');
          s += String.fromCharCode(parseInt(h, 16)); i += 6; continue;
        }
        throw fallo('escape desconocido');
      }
      s += c; i++;
    }
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

// ───────────────────────── el comprobador ─────────────────────────

export function lineasDeTexto(texto) {
  const partes = texto.split('\n');
  if (partes.length && partes[partes.length - 1] === '') partes.pop();
  return partes;
}

/** Mismo contrato que `comprobar_traza` del núcleo. `lineas`: textos; `opciones`: {huellaAnclada, esperarGenesis}. */
export async function comprobarTraza(lineas, { huellaAnclada = null, esperarGenesis = false } = {}, plataforma = plataformaNode) {
  if (huellaAnclada !== null && !HEX64.test(huellaAnclada)) throw new Error('la huella anclada es SHA-256 en hexadecimal minúscula (64 caracteres)');
  const bloques = [];
  for (let i = 0; i < lineas.length; i++) {
    try {
      const b = leerJSON(lineas[i]);
      estructura(b);
      bloques.push(b);
    } catch (e) {
      if (e instanceof Formato) return res(NO_SE_PUDO_COMPROBAR, e.motivo, i, e.detalle);
      throw e;
    }
  }
  if (!bloques.length) return res(NO_SE_PUDO_COMPROBAR, 'formato', null, t('fichero sin bloques', 'file without blocks'));
  for (let i = 0; i < bloques.length; i++) {
    const b = bloques[i];
    const s = b.servicio;
    const crudo = deHex(s.clave_publica);
    if (plataforma.sha256(crudo) !== s.huella_clave) {
      return res(NO_COINCIDE, 'clave', i, t('la huella no es la de la clave pública que lleva el bloque', 'the fingerprint is not the one of the public key the block carries'));
    }
    const firma = firmaDeBase64(b.firma.valor);
    if (firma === null || !(await plataforma.ed25519(crudo, mensajeFirmado(b), firma))) {
      return res(NO_COINCIDE, 'firma', i, t('la firma no coincide con el contenido del bloque', 'the signature does not match the block\'s content'));
    }
  }
  for (let i = 0; i < bloques.length; i++) {
    try { estructura(bloques[i], true); } catch (e) {
      if (e instanceof Formato) return res(NO_SE_PUDO_COMPROBAR, e.motivo, i, e.detalle);
      throw e;
    }
  }
  const reinicios = [];
  for (let i = 0; i < bloques.length; i++) {
    const b = bloques[i];
    if (i === 0 && esperarGenesis && !(b.seq === 0 && b.prev === GENESIS)) {
      return res(NO_COINCIDE, 'cadena', 0, t('se esperaba el bloque génesis (seq 0, prev de ceros) y el fichero empieza más tarde', 'the genesis block (seq 0, prev of zeros) was expected and the file starts later'));
    }
    if (b.reinicio === 'sin_previo') {
      if (!(b.seq === 0 && b.prev === GENESIS)) return res(NO_COINCIDE, 'cadena', i, t('un reinicio sin_previo debe tener seq 0 y prev de ceros', 'a sin_previo restart must have seq 0 and a prev of zeros'));
      reinicios.push(i);
    } else if (i > 0) {
      const a = bloques[i - 1];
      if (b.seq !== a.seq + 1) return res(NO_COINCIDE, 'cadena', i, t(`seq ${b.seq} no sigue a ${a.seq}`, `seq ${b.seq} does not follow ${a.seq}`));
      if (b.prev !== await hashDeBloque(a, plataforma)) return res(NO_COINCIDE, 'cadena', i, t('prev no es el SHA-256 del bloque anterior', 'prev is not the SHA-256 of the previous block'));
      if (b.servicio.huella_clave !== a.servicio.huella_clave) return res(NO_COINCIDE, 'cadena', i, t('la clave cambia dentro de la cadena sin un reinicio sin_previo', 'the key changes inside the chain without a sin_previo restart'));
    }
  }
  const huellas = [];
  for (const b of bloques) if (!huellas.includes(b.servicio.huella_clave)) huellas.push(b.servicio.huella_clave);
  if (huellaAnclada !== null) {
    for (let i = 0; i < bloques.length; i++) {
      if (bloques[i].servicio.huella_clave !== huellaAnclada) return res(NO_COINCIDE, 'ancla', i, t('la huella de este bloque no es la anclada', 'this block\'s fingerprint is not the anchored one'));
    }
  }
  const ultimo = bloques[bloques.length - 1];
  return res(VALIDA, null, null, null, { bloques: bloques.length, empieza_en: bloques[0].seq, ultimo_seq: ultimo.seq, ultimo_hash: await hashDeBloque(ultimo, plataforma), huellas, reinicios, anclada: huellaAnclada !== null });
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
    const r = await comprobarTraza(caso.lineas, { huellaAnclada: caso.ancla ?? null, esperarGenesis: caso.esperar_genesis ?? false });
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
