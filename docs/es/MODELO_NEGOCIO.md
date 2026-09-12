# Modelo de negocio de MatrixAI

MatrixAI son **dos cuerpos de software, con dos licencias y un solo autor**.

| | Licencia | Precio |
|---|---|---|
| **Núcleo y CLI de MatrixAI** (`matrixai-core`) | GNU AGPL v3 (`AGPL-3.0-only`) | gratis |
| **MatrixAI Studio** (este producto) | propietaria — ver `LICENSE` | de pago |

## El núcleo es software libre, y lo sigue siendo

El lenguaje, el runtime, la CLI, el entrenamiento, el registry, el servidor
HTTP, las herramientas de despliegue y la documentación se publican bajo la GNU
Affero General Public License versión 3. El registro de verificación de
licencia está en [VERIFICACION_LICENCIA.md](VERIFICACION_LICENCIA.md).

Esa licencia permite a cualquiera usar, estudiar, modificar y redistribuir el
núcleo bajo sus términos — incluido **construirse su propio estudio encima**,
sin pedir permiso y sin pagar nada. Es una alternativa real a comprar este
producto, y está escrita aquí a propósito.

## MatrixAI Studio es un producto comercial

MatrixAI Studio —la interfaz, el backend del Studio, la web y la API de
licencias— **no** es software libre. Lo cubren los términos propietarios de
`LICENSE`, que viaja en la raíz de este paquete junto a `TERCEROS.md`.

**El Studio existe hoy, y es lo que tienes delante.** Hasta el 2026-09-12 este
documento lo describía como algo previsto *después* del roadmap v1.0, «solo si
la adopción real demuestra demanda», y que debería hablar con el núcleo «por
HTTP en lugar de importar módulos Python directamente». El producto ha
adelantado a las dos frases y esta sección las sustituye: el Studio está
construido, se vende, y su backend importa los módulos Python del núcleo
directamente.

Ese import directo no choca con la AGPL. La AGPL es copyleft fuerte: lo que
enlaza con código AGPL y se distribuye tiene que distribuirse bajo AGPL — *por
cualquiera que no sea el titular del copyright*. El núcleo y el Studio tienen
el mismo y único titular, que puede publicar su propio trabajo como software
libre para todo el mundo **y** usarlo bajo otros términos en su propio
producto. Ningún tercero tiene derechos sobre el núcleo que esto pudiera
infringir.

## Qué recibes al comprar el Studio

- **El Studio**, bajo la licencia propietaria de `LICENSE`.
- **El núcleo, bajo AGPL-3.0**, con su texto de licencia completo y con todos
  los derechos que esa licencia te concede sobre él: usarlo, estudiarlo,
  modificarlo y redistribuirlo. La licencia del Studio no restringe eso ni
  pretende hacerlo.
- **Bibliotecas de terceros**, cada una con sus propios términos.

`TERCEROS.md` las enumera las tres, junto a `LICENSE` en la raíz del paquete.
Lo que se vende aquí es este Studio, no el derecho a usar el núcleo: eso no
necesita comprarlo nadie.

## Donativos

Se podrán aceptar donativos voluntarios mediante GitHub Sponsors o una
plataforma equivalente para ayudar a cubrir costes de servidor y tiempo de
desarrollo del núcleo libre. Los donativos no son necesarios para usarlo.

## Marca

MatrixAI, matrixaistudio.org y la identidad pública del proyecto pertenecen al
autor del proyecto. Los proveedores externos solo se mencionan cuando es
técnicamente necesario para describir compatibilidad configurable.

## Límites de este documento

Esto es un resumen de la posición del autor sobre las licencias, no la licencia
en sí. Los textos que mandan son `LICENSE` para el Studio y el texto de la AGPL
v3 que viaja con el núcleo; donde este resumen y esos textos difieran, **mandan
los textos**.

No es asesoramiento legal, y no responde a qué significa la AGPL *dentro de tu
organización*. Si puedes incorporar, extender o redistribuir algo que lleva un
núcleo AGPL dentro es una pregunta para tus propios asesores — mejor hacerla
antes de desplegar que después.
