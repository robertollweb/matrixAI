// 153-C1 · Ayudante de las pruebas de paridad Python<->JS. Lee un JSON de la entrada estándar y escribe otro.
//   {"modo":"jcs","valores":[...]}                       -> {"hex":[<JCS en hex de cada valor>]}
//   {"modo":"comprobar","casos":[{lineas,ancla,esperar_genesis}...]} -> {"resultados":[{estado,motivo,bloque}...]}
import { readFileSync } from 'node:fs';
import { comprobarTraza, jcsBytes } from './comprobar_traza.mjs';

const entrada = JSON.parse(readFileSync(0, 'utf8'));
if (entrada.modo === 'jcs') {
  const hex = entrada.valores.map((v) => Buffer.from(jcsBytes(v)).toString('hex'));
  process.stdout.write(JSON.stringify({ hex }));
} else if (entrada.modo === 'comprobar') {
  const resultados = [];
  for (const c of entrada.casos) {
    const r = await comprobarTraza(c.lineas, { huellaAnclada: c.ancla ?? null, esperarGenesis: c.esperar_genesis ?? false });
    resultados.push({ estado: r.estado, motivo: r.motivo, bloque: r.bloque });
  }
  process.stdout.write(JSON.stringify({ resultados }));
} else {
  throw new Error('modo desconocido');
}
