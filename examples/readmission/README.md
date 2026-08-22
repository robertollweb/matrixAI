# Riesgo de reingreso a 30 días — caso de referencia (contrato 81-C6)

> **DEMOSTRACIÓN. Datos sintéticos. NO validado clínicamente y NO apto
> para decisiones asistenciales: es un sistema de apoyo, y no sustituye al
> profesional.**
>
> Esta frase encabeza también la salida del programa y viaja dentro de
> cada recibo. No está aquí por trámite: un caso de reingreso hospitalario
> que no la lleve **donde se lee el resultado** se lee como una
> herramienta clínica.

## Cómo correrlo

```bash
python3 examples/readmission/run_case.py
```

Sin red, sin claves y sin dependencias externas. El dataset se genera al
vuelo con una semilla.

## Cómo reconstruirlo desde cero

El clasificador entrenado viaja en `registry/` (56 KB), así que lo de
arriba funciona recién clonado. Para rehacerlo:

```bash
# 1 · el dataset, determinista
python3 -c "from pathlib import Path; from matrixai.reference.readmission_datos \
  import generar_dataset; Path('examples/readmission/data/train.csv') \
  .write_text(generar_dataset(400, 42))"

# 2 · entrenar
python3 -m matrixai train examples/readmission/readmission.mxai \
  --training examples/readmission/readmission.mxtrain \
  --output examples/readmission/run

# 3 · evaluar (el registry EXIGE el informe: sin él no se puede publicar)
python3 -m matrixai evaluate examples/readmission/readmission.mxai \
  --training examples/readmission/readmission.mxtrain \
  --params examples/readmission/run/parameter_set.json \
  --data examples/readmission/data/train.csv \
  --output examples/readmission/run/evaluation_report.json

# 4 · publicar en el registry del ejemplo
python3 -c "import shutil; from pathlib import Path; \
  from matrixai.registry.model_registry import ModelRegistry; \
  run=Path('examples/readmission/run'); \
  shutil.copy2('examples/readmission/readmission.mxai', run/'readmission.mxai'); \
  print(ModelRegistry('examples/readmission/registry') \
  .push_run_dir(run, 'readmission_head', 'v1').entry_hash)"
```

*Esto estaba sin escribir y lo encontró la 3ª pasada de auditoría
conduciendo el producto: sin `registry/`, el programa decía «entrena el
caso primero (ver el README del ejemplo)» y el README **no decía cómo**.
Un mensaje que manda a un sitio donde no está la respuesta es la misma
avería que «no hay sandbox» sin decir qué instalar.*

## Qué es

Un pipeline multimodal —variables tabulares + nota clínica libre— que
estima riesgo de reingreso a 30 días, ejecutado por el motor del contrato
81, decidido por su lenguaje de políticas y respaldado por un recibo.

```
tabulares → preproceso + encoder ┐
                                 ├→ fusión → clasificador → política → salida
nota      → texto + encoder ─────┘
```

## Los datos

**Sintéticos, y por eso.** El §16.2 admite datos sintéticos o
desidentificados; se eligen los primeros porque un dataset desidentificado
sigue siendo de personas y esto viaja en un repositorio público.

La etiqueta sale de una **regla publicada**, no de un oráculo opaco:

```
riesgo = 0.35·edad + 0.30·ingresos_previos + 0.20·estancia
       + 0.10·(disnea|edema|oxigeno en la nota) + 0.05·soledad
etiqueta = reingreso si riesgo + ruido(±0.05) > 0.5
```

Con la misma semilla, el CSV sale **byte a byte igual**.

## Las políticas, declaradas

| Situación | Qué hace | Por qué |
|---|---|---|
| Falta una variable **obligatoria** | **se abstiene** | rellenarla con la media convierte «no lo sabemos» en «es normal» |
| Falta una **opcional** | predice, y lo declara **dentro del vector** | un cero solo se leería como una medida; al lado va la bandera de ausencia |
| Categoría desconocida | **se abstiene** | asignarla a la más parecida predice sobre un paciente que no es éste |
| Nota más larga que 512 B | trunca **y lo dice** | el final de una nota es donde suele estar lo agudo |
| Nota vacía | señal **ausente**, declarada | un vector de ceros se lee como «sin hallazgos» |
| Texto que no es UTF-8 | **rechaza** | reemplazar bytes cambia el texto sobre el que se decide |
| Dato posterior a la decisión | **se abstiene** | fuga temporal: el modelo parecería mejor de lo que es |
| Dimensiones que no encajan | **rechaza** | ajustarlas produce un vector que no describe a nadie |
| Confianza < 0,60 | **se abstiene** | una clase con confianza de moneda al aire se lee igual que una segura |

La línea entre **rechazar** y **abstenerse** es deliberada: se rechaza lo
que está mal formado (no hay nada que decidir) y se abstiene ante lo que
está bien formado y no basta (sí hay algo que decir, y es «no me
pronuncio», con la regla al lado).

## Limitaciones conocidas — **léelas antes que las métricas**

**1. Se calla justamente donde importa.** Medido sobre su propio dataset
(400 casos, umbral 0,60):

| | n | se abstiene | cubre |
|---|---:|---:|---:|
| `no_reingreso` | 276 | 16,3 % | 83,7 % |
| **`reingreso`** | **124** | **89,5 %** | **10,5 %** |
| global | 400 | 39,0 % | 61,0 % |

Cuando responde acierta el **94,7 %** — y ese número se consigue
**callándose los casos difíciles**. El sistema casi no se pronuncia sobre
los pacientes que sí reingresan, que es para lo que existiría. La tasa
global del 39 % suena prudente y esconde esto entero.

**2. La calibración solo vale dentro de su distribución.** ECE = 0,077
sobre los 400 casos. No dice nada de un caso que el modelo no ha visto:
ahí puede estar igual de seguro y equivocado. **La calibración no es una
defensa contra lo que queda fuera de distribución**, y el umbral de
abstención tampoco.

**3. El vocabulario de la nota es cerrado** (11 palabras). Una nota real
usaría miles, y las que no están simplemente no se ven.

**4. `alta_voluntaria` no interviene en la regla de la etiqueta**: es
ruido a propósito, para que se vea que el modelo puede agarrarse a algo
que no explica nada.

**5. No hay validación externa, ni temporal, ni por centro.** Nada de lo
que hay aquí sostiene una afirmación clínica.

## Métricas

| | |
|---|---|
| Exactitud (evaluación completa) | 0,8575 |
| Macro-F1 | 0,8421 |
| Mejor pérdida de validación | 0,2888 |
| ECE (dentro de distribución) | 0,0770 |
| Latencia por caso | mediana ~0,5 ms |
| Memoria pico | ~24 MiB |

## Privacidad

Perfil `local-only`, sin red. En la traza y en el recibo van
identificadores, versiones, huellas, tiempos, estados y resultados de
políticas. **Nunca** la nota clínica ni ningún identificador de paciente:
del texto solo sale su huella, y el nodo de texto **no deja salir el
texto**.

Un aviso que conviene no maquillar: una huella **no anonimiza** un campo
de pocos valores posibles —probar los cuatro servicios revierte el
hash—, así que el servicio se declara como categoría y no se vende como
anonimizado.
