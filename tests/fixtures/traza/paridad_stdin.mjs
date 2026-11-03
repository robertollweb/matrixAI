// 153-C1 · Ayudante de las pruebas de paridad Python<->JS. Lee un JSON de la entrada estándar y escribe otro.
//   {"modo":"jcs","valores":[...]}                       -> {"hex":[<JCS en hex de cada valor>]}
//   {"modo":"comprobar","casos":[{lineas|fichero_b64|fichero_de_espacios,ancla,esperar_genesis}...]}
//                                                       -> {"resultados":[{estado,motivo,bloque,cadenas}...]}
//   (con "acepta_todo": true, el Ed25519 de la plataforma acepta cualquier firma)
//   {"modo":"topes"}                                     -> {"TOPE_LINEA","TOPE_FICHERO"}
import { readFileSync } from 'node:fs';
import { comprobarCaso, jcsBytes, plataformaNode, TOPE_FICHERO, TOPE_LINEA } from './comprobar_traza.mjs';

const entrada = JSON.parse(readFileSync(0, 'utf8'));
if (entrada.modo === 'jcs') {
  const hex = entrada.valores.map((v) => Buffer.from(jcsBytes(v)).toString('hex'));
  process.stdout.write(JSON.stringify({ hex }));
} else if (entrada.modo === 'comprobar') {
  // `acepta_todo`: una plataforma cuyo Ed25519 da por buena CUALQUIER firma (la de una biblioteca que no rechazara claves de
  // orden pequeño ni S >= L): lo que siga rechazándose lo rechaza el comprobador, no la biblioteca.
  const plataforma = entrada.acepta_todo ? { ...plataformaNode, ed25519: () => true } : plataformaNode;
  const resultados = [];
  for (const c of entrada.casos) {
    const r = await comprobarCaso(c, plataforma);
    resultados.push({ estado: r.estado, motivo: r.motivo, bloque: r.bloque, cadenas: r.cadenas ?? null });
  }
  process.stdout.write(JSON.stringify({ resultados }));
} else if (entrada.modo === 'topes') {
  process.stdout.write(JSON.stringify({ TOPE_LINEA, TOPE_FICHERO }));
} else {
  throw new Error('modo desconocido');
}
