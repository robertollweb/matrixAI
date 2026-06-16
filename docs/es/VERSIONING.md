# Política de versiones y compatibilidad

MatrixAI usa [Versionado Semántico 2.0.0](https://semver.org/lang/es/): `MAYOR.MENOR.PARCHE`.

> **English:** [docs/en/VERSIONING.md](../en/VERSIONING.md)

---

## Qué significa la v1.0

La v1.0.0 es la primera release estable pública. Marca el punto en que el lenguaje,
el runtime, la CLI, la API HTTP y el registry de modelos se consideran listos para
uso en producción en entornos críticos.

A partir de la v1.0.0 este proyecto sigue una política de compatibilidad clara,
descrita a continuación.

---

## Qué está estable en v1.0

Las siguientes interfaces son estables y están cubiertas por la garantía de compatibilidad:

| Interfaz | Alcance |
|----------|---------|
| **Lenguaje `.mxai`** | Todos los tipos de nodo, expresiones y construcciones de tipos documentados en la especificación del lenguaje |
| **Formato `.mxtrain`** | Todos los campos de la spec de entrenamiento, optimizadores y funciones de pérdida |
| **Formato `.mxact`** | Todos los campos del contrato de acción y su semántica de ejecución |
| **Formato `.mxcontinual`** | Todos los campos de la política de aprendizaje continuo y sus disparadores |
| **Comandos y flags de la CLI** | Todos los comandos de `matrixai --help` y `CLI_REFERENCE.md` |
| **API HTTP** | Endpoints de `REST_API.md`; estarán disponibles bajo `/api/v1/` una vez versionados en este ciclo de release — se mantendrán alias compatibles hacia atrás |
| **Formato del registry de modelos** | Estructura de entradas, esquema de firma, campo `matrixai_version` |
| **Nombre del paquete PyPI** | `matrixai-core`; punto de entrada `matrixai` |
| **Tags de imagen Docker** | `ghcr.io/robertollweb/matrixai:MAYOR.MENOR.PARCHE` y `:latest` |

---

## Qué no está estable

Lo siguiente puede cambiar sin necesidad de un cambio de versión mayor:

- **Estructura interna de módulos Python** — los imports desde `matrixai.compiler.*`,
  `matrixai.agents.*`, `matrixai.ir.*` son internos. Usa la CLI o la API HTTP.
- **Bridge LLM** — nombres de variables de entorno y comportamiento pueden cambiar
  mientras esta función madura.
- **Formato de salida del playground** — sujeto a mejoras.

---

## Reglas de cambio de versión

| Tipo de cambio | Incremento | Ejemplos |
|----------------|------------|---------|
| Cambio incompatible en lenguaje, CLI, API HTTP o registry | **Mayor** (2.0.0) | Eliminar un tipo de nodo, renombrar un flag obligatorio de la CLI, cambiar la forma de la respuesta HTTP |
| Nueva funcionalidad compatible con versiones anteriores | **Menor** (1.1.0) | Nuevo tipo de nodo, nuevo comando CLI, nuevo endpoint HTTP |
| Corrección de error, sin cambio de API | **Parche** (1.0.1) | Corregir convergencia de entrenamiento, corregir mensaje de error |

Un cambio incompatible es aquel que obliga al usuario a modificar sus ficheros
`.mxai`, `.mxtrain`, `.mxact` o `.mxcontinual`, sus scripts de CLI, o su integración
con la API HTTP.

---

## Compatibilidad de modelos

Cada entrada en el registry almacena la `matrixai_version` que la creó.
Esta es la versión de producto de MatrixAI, separada de la versión de esquema
del índice guardada en `registry.json`.

- Los modelos entrenados con **v1.0.x** se cargan y ejecutan en cualquier **v1.0.y**.
- Los modelos entrenados con **v1.x.y** se cargan en **v1.x.z** (z ≥ y) y en **v1.w.0** (w > x).
- Un **cambio de versión mayor** puede requerir un paso de migración. Las guías de
  migración se publican en `CHANGELOG.md` cuando esto ocurre.
- `matrixai registry verify` valida el hash de la entrada, los checksums de artefactos
  y la firma. Emite un aviso cuando la versión mayor de la entrada difiere de la
  versión en ejecución.

---

## Política de deprecación

1. Una función se marca como deprecada en `CHANGELOG.md` y emite un aviso en tiempo de ejecución cuando se usa.
2. Una función deprecada no se elimina antes de la siguiente versión menor.
3. Las funciones nunca se eliminan en una versión de parche.
4. Cada función pasa por un único ciclo de deprecación: deprecada en v1.x,
   eliminada como pronto en v1.(x+1).

---

## Ciclo de soporte

| Serie | Estado |
|-------|--------|
| 1.0.x | Activa — correcciones de errores y parches de seguridad |
| < 1.0 | Fin de vida — sin correcciones |

Este proyecto está mantenido por un equipo pequeño. Los tiempos de respuesta están
en `SECURITY.md`.
