// 145-C1 · paridad Python <-> WebCrypto (la sonda_2 del inventario de la Ola 2, hecha prueba).
// Entrada por stdin (JSON):
//   {"modo": "verificar", "raw": b64, "pem": "-----BEGIN PUBLIC KEY-----...", "casos": [{"nombre", "msg": b64, "sig": b64}]}
//   {"modo": "firmar", "msg": b64}
// Salida por stdout (JSON). Solo `crypto.subtle`: lo mismo que tendrá la página de verificación.
import { webcrypto as wc, createPublicKey } from 'node:crypto'

const entrada = JSON.parse(await new Promise((ok) => {
  let t = ''
  process.stdin.on('data', (d) => { t += d })
  process.stdin.on('end', () => ok(t))
}))
const b = (s) => Uint8Array.from(Buffer.from(s, 'base64'))

if (entrada.modo === 'verificar') {
  const kRaw = await wc.subtle.importKey('raw', b(entrada.raw), { name: 'Ed25519' }, false, ['verify'])
  const der = createPublicKey(entrada.pem).export({ type: 'spki', format: 'der' })
  const kSpki = await wc.subtle.importKey('spki', der, { name: 'Ed25519' }, false, ['verify'])
  const huella = Buffer.from(await wc.subtle.digest('SHA-256', b(entrada.raw))).toString('hex')
  const resultados = {}
  for (const c of entrada.casos) {
    resultados[c.nombre] = {
      raw: await wc.subtle.verify('Ed25519', kRaw, b(c.sig), b(c.msg)),
      spki: await wc.subtle.verify('Ed25519', kSpki, b(c.sig), b(c.msg)),
    }
  }
  console.log(JSON.stringify({ huella, resultados }))
} else if (entrada.modo === 'firmar') {
  const par = await wc.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify'])
  const sig = Buffer.from(await wc.subtle.sign('Ed25519', par.privateKey, b(entrada.msg))).toString('base64')
  const raw = Buffer.from(await wc.subtle.exportKey('raw', par.publicKey)).toString('base64')
  const spki = Buffer.from(await wc.subtle.exportKey('spki', par.publicKey)).toString('base64')
  console.log(JSON.stringify({ raw, spki, sig }))
} else {
  throw new Error('modo desconocido: ' + entrada.modo)
}
