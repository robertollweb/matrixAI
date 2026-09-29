# Jev (TypeSafe) por OpenRouter — el formato exacto, con fuentes

Condición (4) de 107-C3.0: el mismo respondedor de preguntas atómicas de la
condición (3), pero con **Jev** (`typesafe/jev-1.13-20260917`, TypeSafe) en
vez del modelo local. Este fichero documenta el formato de la petición, con
las URL de origen. **Nada de esto se ha probado contra el servicio real**
(no hay clave en esta máquina de trabajo, y el encargo lo prohíbe): es
documentación leída, no medida.

## Fuentes consultadas

1. `docs.typesafe.ai` (documentación oficial de TypeSafe), vía su índice
   `https://docs.typesafe.ai/llms.txt` (ya presente en el repo como
   `documentacion/112_anexos/fuentes/typesafe_llms.txt`) y las páginas:
   - `https://docs.typesafe.ai/introduction/quickstart.md`
   - `https://docs.typesafe.ai/primitives.md` y `.../primitives/noul.md`
   - `https://docs.typesafe.ai/sdk/python/api/types/questions.md` y
     `.../responses.md`
   - `https://docs.typesafe.ai/sdk/python/api/constants.md`
2. `openrouter.ai` (documentación de OpenRouter):
   - `https://openrouter.ai/docs/guides/community/jev` (guía de Jev en
     OpenRouter)
   - `https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request`
     (referencia de la API, esquema completo)
   - `https://openrouter.ai/docs/guides/community/typesafe-sdk`
   - `https://openrouter.ai/docs/faq` y `https://openrouter.ai/docs/api-reference/limits`
     (límites de tasa)
3. `documentacion/112_anexos/fuentes/priorbench_jev.md` (ya en el repo, no
   descargado en esta sesión): informe independiente de PriorBench, 5.721
   llamadas reales medidas contra `typesafe/jev-1.13-20260917` por
   OpenRouter, con su propia doble verificación (`analysis/recount.py`) —
   citado abajo como fuente de los números de coste/latencia/concurrencia
   que ni TypeSafe ni OpenRouter publican como límite oficial.

## El endpoint: una discrepancia entre fuentes, sin resolver

Tres lecturas de páginas distintas dieron TRES formas distintas de construir
la URL:

1. La guía de Jev en OpenRouter, y el resumen de búsqueda que la cita, dicen
   en prosa: **`POST https://openrouter.ai/api/alpha/decisions`**. Es
   también la que cita, literal, el anexo local
   `priorbench_jev.md` (línea 9: *"Access: OpenRouter `/api/alpha/decisions`"*),
   que fue la pista de partida del encargo.
2. Una lectura de la página de referencia OpenAPI
   (`.../alphadecisions/submit-a-decisions-...`) devolvió la concatenación
   **`https://openrouter.ai/api/v1/api/alpha/decisions`** (servidor por
   omisión `.../api/v1` + la ruta del propio endpoint `/api/alpha/decisions`
   pegadas sin más) — parece un artefacto de cómo el extractor combinó el
   "server" genérico de la especificación OpenAPI con la ruta absoluta
   propia de este endpoint alfa (que normalmente la anularía), no una
   tercera ruta real.
3. La guía del SDK de TypeSafe (`.../guides/community/typesafe-sdk`)
   describe la composición como base `https://openrouter.ai/api` + sufijo
   **`/v1/systemone`** — el MISMO sufijo que usa la API nativa de TypeSafe
   (`https://api.typesafe.ai/v1/systemone`, ver `quickstart.md`), sugiriendo
   que el SDK, cuando se apunta a OpenRouter, reexpone la forma nativa en
   vez de la ruta `/api/alpha/decisions` documentada para la llamada HTTP
   cruda.

**No se pudo resolver esta discrepancia sin hacer una petición real**, que
este corte prohíbe expresamente. Decisión tomada aquí: `respondedor_jev.py`
usa **`https://openrouter.ai/api/alpha/decisions`** como valor por omisión
(la lectura mejor sostenida — tres fuentes independientes de prosa, incluida
la que ya citaba la casa) y lo deja **configurable** (`base_url` /
`--base-url`), precisamente para que la primera llamada real, cuando haya
clave, lo confirme o lo corrija sin tocar código.

## Formato de la petición (Decisions API, tipo `noul`)

**Método y cabeceras:**
```
POST <base_url>
Authorization: Bearer <OPENROUTER_API_KEY>
Content-Type: application/json
```

**Cuerpo** — tres campos: `model`, `state` (el texto a evaluar) y
`questions` (mapa id → definición). Para una pregunta atómica sí/no, el tipo
es `"noul"` (booleano probabilístico; TypeSafe: *"a Noul question asks the
model to evaluate a yes/no question and return the probability that the
answer is yes"*), con un campo `instructions` (la pregunta) y un `criteria`
opcional (`{"true": "...", "false": "..."}`) que aquí NO se usa — se
reutiliza la pregunta LITERAL ya sellada de la condición (3)
(`medir_c30_respondedor.PLANTILLA_PREGUNTA`), sin inventar una descripción
nueva:

```json
{
  "model": "typesafe/jev-1.13-20260917",
  "state": "el texto de la fila, truncado como el embedding de C1",
  "questions": {
    "mobiliario": {
      "type": "noul",
      "instructions": "¿El texto trata de «mobiliario» o de algo equivalente? Responde solo sí o no."
    },
    "urgente": {
      "type": "noul",
      "instructions": "¿El texto trata de «urgente» o de algo equivalente? Responde solo sí o no."
    }
  }
}
```

**Varias preguntas por petición — SÍ, y es el diseño**: "Send every question
that uses the same state in one request... adding questions barely changes
the response time and costs only the tokens for the extra questions"
(`primitives.md`). PriorBench lo mide: 800 preguntas en una sola llamada,
985 ms, $0.000747 — frente a ~430 ms FIJOS por llamada si se encadenan una a
una. Por eso `respondedor_jev.responder_terminos` manda **las 5 preguntas de
una fila en UNA sola petición** (no 5 peticiones), a diferencia del
respondedor local de (3), que sí necesita una generación por pregunta.

**Respuesta** (200 OK):
```json
{
  "model": "typesafe/jev-1.13-20260917",
  "answers": {
    "mobiliario": {"type": "noul", "noul": 0.96},
    "urgente": {"type": "noul", "noul": 0.12}
  },
  "usage": {"input_tokens": 476, "output_tokens": 39, "cost": 0.000019992}
}
```
`answers.<id>.noul`: "probability of a yes answer... 0 to 1, where 0 means
no and 1 means yes" (`primitives/noul.md`). Sin campo de texto libre: no
hace falta parsear "sí"/"no" como en (3) — se umbraliza en 0.5 (el propio
punto medio que define la documentación), y cualquier respuesta que no
tenga la forma esperada (falta la clave, `type` distinto de `"noul"`, valor
no numérico o fuera de `[0, 1]`) se cuenta como **faltante**, igual que (3).

## Errores y reintentos

Los códigos documentados en la referencia de la API:

| código | motivo |
|---|---|
| 400 | petición mal formada |
| 401 | clave ausente o inválida |
| 402 | crédito insuficiente |
| 403 | sin permiso |
| 429 | límite de tasa superado |
| 500/502/503/524/529 | fallo del proveedor o de la pasarela |

`respondedor_jev.py` reintenta con espera exponencial **solo** 429 y 5xx
(`ErrorJevReintentable`); el resto (401, 400, 402, 403...) no tiene sentido
reintentarlo y se propaga en el primer intento (`ErrorJevFatal`).

## Límites de tasa: sin techo oficial publicado, un punto de operación medido

Ni la FAQ de OpenRouter (`docs/faq`) ni la referencia de límites
(`docs/api-reference/limits`) dan un número de peticiones/minuto para
modelos de pago como Jev — solo protección genérica anti-DDoS y límites
*diarios* para modelos GRATIS (no aplica aquí). PriorBench sí lo MIDE (no es
un límite impuesto, es lo que observaron):

| concurrentes | p50 | caudal | nota |
|---|---|---|---|
| 1 | 521 ms | 1,91/s | |
| 4 | 537 ms | 6,88/s | |
| **8** | **576 ms** | **10,41/s** | recomendación de PriorBench: "eight concurrent requests is the operating point" |
| 16 | 821 ms | 11,19/s | +7% de caudal por +43% de latencia: no compensa |

`--estimar` usa **10,41 peticiones/segundo** (8 concurrentes, la fila de la
tabla de arriba) como supuesto **declarado**, no como límite real.

## Coste por token

`https://openrouter.ai/docs/guides/community/jev`: *"Output tokens are
free; billing is per input token"*, contexto de 32.000 tokens. El precio
verificado por PriorBench con su propio recuento independiente
(`analysis/recount.py`, sin importar el código de análisis principal):
**$0,042000 por millón de tokens de ENTRADA**; salida, $0. Ambos números en
`respondedor_jev.PRECIO_USD_POR_MILLON_TOKENS_ENTRADA` /
`_SALIDA`.

## Supuestos de tokens para `--estimar` (SIN medir — declarado)

Jev no publica su propio tokenizador (ni en la documentación de TypeSafe ni
en la de OpenRouter), así que `--estimar` no puede tokenizar de verdad el
texto como hace (3) con el tokenizador real de C1. Se usan dos supuestos,
citados en el propio JSON de salida (`estimacion_c30_jev.json`, campo
`supuestos`), NO medidos:

- **`~1 token / 3,6 caracteres`** (`SUPUESTO_TOKENS_POR_CARACTER = 0.28`):
  heurística común para texto latino con acentuación (más denso que el
  inglés puro); sin tokenizador propio de Jev que lo confirme.
- **`~70 tokens fijos por pregunta`** (envoltorio JSON + `instructions` +
  claves): derivado del único ejemplo con recuento real de la referencia de
  la API (476 tokens de entrada para un `state` corto de ~22 tokens más TRES
  preguntas heterogéneas -- una `noul` con `criteria` breve, una `choice`
  con 3 opciones y descripciones, una `score` con 3 niveles -- que promedian
  ~150 tokens/pregunta; una `noul` sin `criteria` como la que usa este corte
  es más barata que ese promedio, así que 70 es una cifra generosa a
  propósito, para no subestimar el coste).

Si algún día hay clave y se mide de verdad, estos dos supuestos se
sustituyen por la tasa observada (mismo patrón que
`medir_c30_respondedor._tasa_s_por_respuesta`, que prefiere la medida real
sobre la constante del registro).

## Lo que NO se pudo determinar

- **El endpoint exacto** (ver arriba): tres lecturas, tres formas de
  construir la URL. Configurable a propósito.
- **Un límite de peticiones/minuto oficial** para Jev/modelos de pago en
  OpenRouter: no publicado: se usa el punto de operación medido por
  PriorBench como supuesto, declarado como tal.
- **El tokenizador real de Jev**: no documentado públicamente; los tokens de
  `--estimar` son una aproximación por caracteres, declarada como supuesto.
- **Si `criteria` (true/false) mejora la precisión lo bastante como para
  justificar sus tokens extra** para ESTA tarea concreta: no se ha probado
  (no hay clave); se empieza sin `criteria`, reusando la pregunta literal ya
  sellada de (3), y queda para cuando se mida de verdad.

## Resuelto al medir (28 y 29-09-2026)

Lo de arriba se escribió sin clave y sin ninguna llamada real. Esto es lo que dijeron las llamadas:

- **El endpoint**: `https://openrouter.ai/api/alpha/decisions` (la lectura 1). Fijado el 28-09 con UNA
  llamada de prueba con un texto inventado (sin datos de ninguna tarea): responde con el modelo
  `typesafe/jev-1.13-20260917`, `noul` 0,98 a «¿trata de tiempo?», 298 tokens de entrada, 0,0000125 $.
  El envoltorio fijo pesa ~260 tokens por petición, así que el supuesto de 70 por pregunta se queda
  CORTO para una petición con una sola pregunta; con cinco por petición, como aquí, el reparto es otro.
- **El ritmo, medido desde este servidor y de UNA en UNA** (el medidor no hace peticiones concurrentes;
  la estimación de 51 min suponía las 8 de PriorBench): tarea A, 1.543 peticiones en 476 s (~0,31 s por
  petición); tarea C, 30.462 peticiones en CUATRO pasadas y 15.208 s de pared (~0,5 s por fila; entre 1,7 y
  2,3 filas por segundo según la hora). La C se reanudó desde la caché tres veces, así que los segundos de
  su resultado (1.122) son los de la ÚLTIMA pasada: las cuatro, con su desenlace, en `pasadas_c30_jev.json`
  (`registrar_pasadas_jev.py`, de las líneas de la cola), y es lo que usan los datos públicos.
- **El coste**: el medidor NO guarda el `usage.cost` de cada respuesta; se acota con el uso de la cuenta,
  que solo usa esta clave: **0,596 $ en total** (A, C y unas pocas llamadas de prueba de ~0,00001 $), frente
  a los 0,54 $ estimados; A costó menos de 0,045 $, así que C, unos 0,55 $. *(Mejora pendiente: sumar
  `usage.cost` por petición en el resultado.)*
- **Las respuestas de C** (152.310, la caché de la medida: lo que se pagó) van comprimidas en
  `respuestas_c30_jev_C.json.gz`; las de A, sin comprimir, en `respuestas_c30_jev_A.json`.
- **Un error que no estaba en la lista de arriba**: `HTTP 402 «Insufficient credits. This account never
  purchased credits»`, a mitad de C (29-09 07:35), aunque se había cargado saldo: no llegó a la cuenta de
  esta clave (a las 08:15 su total comprado era solo la recarga posterior, de 5 $). El
  adaptador lo trató bien —fatal, sin reintento, la caché guardada— y se siguió desde ella al arreglarlo.
  Y la clave puede llevar un **límite propio** aparte del saldo de la cuenta (esta, 2 $), que se consulta
  en `GET /api/v1/key` (`limit`, `limit_remaining`) sin enseñar la clave.
- **Un corte de red** (`TimeoutError`) paró la primera pasada de C: no era un `HTTPException` ni un
  `OSError` reintentable en la versión de `48b010e`. Desde `b739ba5` se reintenta como los 5xx.
- **Los resultados**, en el contrato 107 (bajo el registro sellado de la (4)), en la comparativa 112
  (§3.8) y, para la web, en `benchmarks/datos_publicos/texto_107_publico.json` (lo compone
  `generar_texto_107.py` desde los registros, sin cifras a mano).
