# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde
"""101-C3 — UNA PASADA DECLARA SU PROPIO ALCANCE, y el alcance CUADRA.

HALLAZGO GRAVE de la auditoría interna del 2026-09-12:

    «El protocolo registrado pide 40 datasets y 7 motores; la pasada mide 12 y
    4. El script declara el subconjunto de datasets, pero en ninguna parte
    declara el recorte de motores — y los tres que faltan (`sklearn.hgb`,
    `xgboost`, `catboost`) son los rivales DIRECTOS de un GBM.»

Re-medido el 2026-09-13, punto por punto, antes de reparar nada: el protocolo
registrado (`101-C1.v1`, digest `493a6f1d…` — el de ENTONCES; ese mismo día,
más tarde, se re-firmó a `eb54f421…` por la cardinalidad falsa del catálogo y
la sobre-reserva de CPU, sin tocar la regla de cierre) trae **40** datasets y
**7** motores; `pasada_exploratoria_101_c3.py` corre **12** y **4**; y de los diez
aciertos de lightgbm, **ocho** tienen distancia 0,0000 al mejor, que es como
se escribe «es el mejor de los tres que compitieron». El enunciado era exacto.

EL DEFECTO NO ERA EL RECORTE. Recortar para una pasada exploratoria es
legítimo y el subconjunto está bien elegido (verificado aquí abajo: son
EXACTAMENTE los 12 del protocolo que cumplen el criterio). El defecto era que
el ARTEFACTO no llevaba escrito su propio alcance: el JSON de resultado no
tenía un solo campo que dijera contra cuántos motores se midió, así que
«lightgbm 10/12 = 0,833 CUMPLE» se podía leer entero sin enterarse de que tres
de los siete motores —los tres que más aprietan a un GBM— no corrieron.

QUÉ PONE ESTE FICHERO EN ROJO, que es lo que decide si tiene dientes:

  * tocar la lista `DATASETS` o `motores_de_la_pasada()` del script sin
    regenerar el JSON — el alcance declarado dejaría de casar con el medido;
  * quitar un motor del protocolo registrado, o añadirlo, sin re-medir;
  * editar a mano cualquiera de las listas de alcance del JSON;
  * un motor que falle ENTERO y no deje un solo registro: aparecería en
    `declarados_por_la_pasada` y no en `observados_en_los_resultados`;
  * borrar el bloque de alcance, o separarlo del veredicto en otro fichero.

NO comprueba «que el campo exista». Eso lo pasa un campo puesto a mano con
cualquier contenido, y es justo el defecto que se está cerrando: la
comprobación es que la lista DECLARADA case con las CLAVES REALES de los 720
registros y con lo que el script corre HOY.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.fase0.pasada_exploratoria_101_c3 import (
    CRITERIO_DEL_SUBCONJUNTO, DATASETS, _componer_y_guardar,
    nombres_de_los_motores_de_la_pasada)
from benchmarks.fase0.protocolo import (
    EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR, ProtocoloExploratorio, aplicar_regla_de_cierre)
from matrixai.estudio.validacion import digest_canonico

_FASE0 = Path(__file__).resolve().parents[1] / "benchmarks" / "fase0"

#: La evidencia CON alcance. Sin `skipTest` si falta: el fichero está en el
#: árbol, y un salto silencioso ante un fichero ausente es exactamente el banco
#: de pruebas sin dientes que este hallazgo denuncia — se quedaría verde sin
#: haber comprobado nada.
#: LA EVIDENCIA QUE ESTE FICHERO AUDITA — la conforme, desde el 2026-09-14.
#:
#: **El rojo deliberado queda cerrado.** Desde el 13-09,
#: `test_los_motores_declarados_son_los_que_el_SCRIPT_corre_hoy` estaba en rojo
#: a propósito: la evidencia declaraba 4 motores y el script corría 7. Era
#: verdad, y taparlo habría sido apagar el guardia que avisa de que la
#: evidencia se ha quedado atrás. Se cerraba re-midiendo, y se ha re-medido.
#:
#: `..._conforme_20260914.json`: **1.260 intentos, CERO fallos**, 12 datasets,
#: los 7 motores, con el presupuesto del protocolo (120 s cubo pequeño, 300 s
#: mediano, en vez de los 120 s fijos) y UNA configuración. 218,5 minutos, sin
#: reusar un solo intento del caché.
#:
#: **Y el veredicto no se movió**: de las 84 celdas comunes con la pasada del
#: 13-09, **83 salen idénticas al bit** y la que se mueve lo hace 0,0029
#: puntos (`Internet-Advertisements` × densa, el motor que ya se sabía que se
#: mueve con la semilla). Motor a motor, exactamente los mismos: catboost
#: 12/12 CUMPLE, lightgbm 9/12, sklearn.hgb 9/12, xgboost 8/12, sklearn.lineal
#: 7/12, densa 4/12.
#:
#: Que el presupuesto correcto no cambie nada **es un resultado, no un
#: trámite**: dice que la pasada anterior no estaba limitada por reloj, y por
#: tanto que el 9/12 de lightgbm no es un artefacto del presupuesto.
_JSON = "pasada_exploratoria_101_c3_conforme_20260914.json"


class ElArtefactoDeclaraSuAlcanceTest(unittest.TestCase):
    """El bloque existe, está completo, y viaja con el veredicto."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads((_FASE0 / _JSON).read_text(encoding="utf-8"))
        cls.resultados = cls.payload["resultados"]
        cls.bloque = cls.payload["alcance_y_veredicto"]
        cls.alcance = cls.bloque["alcance"]
        cls.protocolo = ProtocoloExploratorio.cargar(
            str(_FASE0 / "protocolo_exploratorio.json"))

    def test_el_veredicto_y_el_alcance_estan_en_el_MISMO_objeto(self):
        """Lo no negociable de la reparación. Si mañana alguien saca el
        alcance a un fichero aparte «para que el resultado quede limpio», los
        dos se separan y el número vuelve a viajar solo — que es como llegó
        hasta aquí."""
        self.assertIn("alcance_y_veredicto", self.payload)
        self.assertIn("alcance", self.bloque)
        self.assertIn("por_motor", self.bloque)
        for motor, veredicto in self.bloque["por_motor"].items():
            self.assertIn("cumple_la_regla", veredicto, motor)
            self.assertIn("como_hay_que_leer_este_numero", veredicto, motor)

    def test_el_veredicto_no_se_puede_leer_sin_el_recorte_DELANTE(self):
        """La frase que acompaña a cada número nombra los motores que faltan.
        No es decoración: es lo único que ve quien lee el JSON por encima."""
        faltan = self.alcance["motores"]["que_faltan"]
        if not faltan:
            # Desde el 2026-09-13 la pasada puede correr los SIETE motores del
            # protocolo, y entonces no hay recorte que anunciar. No se salta la
            # comprobación en silencio: se dice por qué no aplica, porque un
            # `skip` mudo es indistinguible de una prueba que dejó de mirar.
            self.skipTest("esta pasada corre los siete motores: no hay recorte de "
                          "motores que anunciar en la lectura del veredicto")
        for motor, veredicto in self.bloque["por_motor"].items():
            lectura = veredicto["como_hay_que_leer_este_numero"]
            for ausente in faltan:
                self.assertIn(ausente, lectura,
                              f"la lectura del veredicto de {motor} no nombra a "
                              f"{ausente}, que NO corrió")

    def test_el_fichero_cuadra_con_su_propio_digest_CON_el_alcance_dentro(self):
        """El alcance va DENTRO del sello. Fuera de él se podría reescribir en
        silencio, y un alcance editable sin dejar rastro no declara nada."""
        payload = dict(self.payload)
        guardado = payload.pop("digest_resultados_crudos")
        self.assertEqual(digest_canonico(payload), guardado,
                         f"{_JSON} ha cambiado desde que se escribió: su propio "
                         "digest ya no cuadra")


class ElSELLO_CUBRE_AL_ALCANCE_EnElCODIGO_NoSoloEnElFicheroTest(unittest.TestCase):
    """Que el JSON que ya existe cuadre con su digest NO dice nada del código.

    SABOTAJE VERDE del 2026-09-13, el décimo de la sesión. El script lleva
    escrito, justo encima de las dos líneas, por qué el alcance va dentro del
    sello: «fuera del sello se podría editar sin que el fichero dejara de
    cuadrar consigo mismo, y un alcance que se puede reescribir en silencio no
    declara nada». Intercambié esas dos líneas —el digest se calcula ANTES de
    meter el alcance— y las 19 pruebas de este fichero siguieron en VERDE.

    Por qué escapaba: el único test que miraba el sello comprobaba el JSON YA
    ESCRITO, que se escribió bien y no cambia porque el script cambie. Probar
    el artefacto no es probar el código que lo produce.

    Es el mismo patrón que ya mordió dos veces hoy: **una línea que explica
    por qué NO hace lo obvio necesita una prueba con su nombre**, o el
    siguiente que pase la «simplifica» y nadie se entera.
    """

    def _payload_recien_compuesto(self):
        """Compone uno DE VERDAD, con el mismo código que corre la pasada."""
        reales = json.loads((_FASE0 / _JSON).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            return _componer_y_guardar(
                reales["resultados"], reales["procedencia"], {},
                Path(tmp) / "salida.json",
                total_wall_s=1.0, reusados=0, parcial=False)

    def test_quitar_el_alcance_del_payload_CAMBIA_su_digest(self):
        """La comprobación directa: si el alcance estuviera fuera del sello,
        quitarlo no movería el digest ni un bit."""
        payload = self._payload_recien_compuesto()
        guardado = dict(payload)
        digest = guardado.pop("digest_resultados_crudos")

        sin_alcance = dict(guardado)
        sin_alcance.pop("alcance_y_veredicto")

        self.assertEqual(digest_canonico(guardado), digest,
                         "el payload recién compuesto no cuadra con su propio digest")
        self.assertNotEqual(
            digest_canonico(sin_alcance), digest,
            "QUITAR el alcance no cambia el digest: está FUERA del sello, así que "
            "se puede reescribir sin que el fichero deje de cuadrar consigo mismo")

    def test_el_alcance_del_payload_recien_compuesto_NO_esta_vacio(self):
        """Un aserto negativo lo pasa un payload en blanco: el de arriba
        seguiría cumpliendo si `alcance_y_veredicto` fuera `{}`, porque quitar
        una clave vacía también mueve el digest. Esta es la otra mitad."""
        bloque = self._payload_recien_compuesto()["alcance_y_veredicto"]
        motores = bloque.get("alcance", {}).get("motores", {})
        # **Esto exigía que FALTARA algún motor**, y el 2026-09-13 dejó de ser
        # verdad: se construyeron los tres que faltaban y la pasada corre los
        # siete del protocolo. Un aserto que da por supuesto el recorte
        # convierte el recorte en contrato, y quien lo cerrara vería la suite
        # en rojo creyendo que se equivoca él.
        #
        # Lo que hay que exigir no es que falte alguien: es que el alcance
        # DIGA algo comprobable. Reescrito conservando su intención.
        self.assertTrue(motores.get("del_protocolo"),
                        "el alcance recién compuesto no dice qué motores pide el protocolo")
        self.assertTrue(motores.get("declarados_por_la_pasada"),
                        "el alcance recién compuesto no dice qué motores corre la pasada")
        # CON el mapa de equivalencias, no a pelo. El protocolo llama `dummy`
        # al `baseline` y `sklearn.logreg` al `sklearn.lineal`; restadas a
        # secas, las listas dicen que faltan CINCO motores en vez de los que
        # de verdad faltan — un número falso en la dirección alarmista, que
        # miente igual que el tranquilizador. Está documentado desde el 13-09
        # y aun así caí en ello al escribir este aserto.
        corren = {EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR.get(m, m)
                  for m in motores["declarados_por_la_pasada"]}
        self.assertEqual(
            set(motores["del_protocolo"]) - corren,
            set(motores.get("que_faltan") or []),
            "«los que faltan» no es la resta real entre lo que el protocolo pide y "
            "lo que la pasada corre: es un texto, no un hecho derivado")
        self.assertTrue(bloque.get("por_motor"),
                        "el alcance recién compuesto no trae veredicto por motor")


class ElAlcanceDeclaradoCUADRAConLoMedidoTest(unittest.TestCase):
    """El corazón del asunto: declarado vs REAL, en las dos direcciones.

    Cada pareja caza un fallo distinto:
      * declarado vs protocolo → el recorte, que es lo que no se declaraba;
      * declarado vs observado → la lista que se tocó sin regenerar el JSON, o
        el motor que se cayó entero y no dejó un registro.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads((_FASE0 / _JSON).read_text(encoding="utf-8"))
        cls.resultados = cls.payload["resultados"]
        cls.alcance = cls.payload["alcance_y_veredicto"]["alcance"]
        cls.protocolo = ProtocoloExploratorio.cargar(
            str(_FASE0 / "protocolo_exploratorio.json"))

    # -- motores ---------------------------------------------------------

    def test_los_motores_DECLARADOS_son_los_que_aparecen_en_los_720(self):
        """Las claves reales del resultado, no una lista escrita a mano."""
        observados = sorted({r["motor"] for r in self.resultados})
        self.assertEqual(sorted(self.alcance["motores"]["declarados_por_la_pasada"]),
                         observados,
                         "el JSON declara unos motores y sus registros traen otros")
        self.assertEqual(sorted(self.alcance["motores"]["observados_en_los_resultados"]),
                         observados)

    def test_los_motores_declarados_son_los_que_el_SCRIPT_corre_hoy(self):
        """La comprobación que caza «cambiaron la lista y no regeneraron el
        JSON». Se le preguntan al script, que a su vez se los pregunta a los
        objetos motor: si uno cambia de `nombre`, esto cambia con él."""
        self.assertEqual(self.alcance["motores"]["declarados_por_la_pasada"],
                         nombres_de_los_motores_de_la_pasada(),
                         "la lista de motores del script ya no es la del JSON: "
                         "hay que volver a generar la evidencia")

    #: El digest del protocolo CONTRA EL QUE SE MIDIERON los 720 intentos.
    #:
    #: Cambió el 2026-09-13 y **eso es correcto, no una regresión**: la
    #: evidencia vigente ya no es la del 12-09 sino una pasada nueva, lanzada
    #: DESPUÉS de re-firmar el protocolo y con todas las reparaciones del día
    #: dentro. Un artefacto cita el protocolo contra el que corrió **él**, no
    #: el que hubiera entonces ni el de hoy.
    #:
    #: Antes de aquí valía `493a6f1d…`, el del primer registro (2026-09-06
    #: 16:32, commit `a3d551a`), que es el que citaba el fichero del 12-09.
    #: Queda escrito para que el cambio se pueda seguir: si mañana alguien ve
    #: este número moverse sin que se haya vuelto a medir, eso SÍ es un fallo.
    DIGEST_CONTRA_EL_QUE_SE_MIDIO = (
        "eb54f42166835ad158b4d879ebfbe085ad8c8eaedf4895cfcb5d220a032756dc")

    def test_los_motores_del_protocolo_son_los_del_protocolo_REGISTRADO(self):
        """**Esto comparaba el digest del artefacto con el del protocolo de
        HOY, y el 2026-09-13 el protocolo se RE-FIRMÓ** (cardinalidad falsa
        del catálogo + reserva de 24 hilos sobre 8 CPUs). Con la comparación
        vieja, una corrección legítima del catálogo invalidaba una evidencia
        que no tenía nada malo — y, peor, la única forma de volver a verde
        habría sido editar el JSON de los 720 intentos, que es evidencia.

        Lo que el artefacto graba es un HECHO HISTÓRICO: contra qué protocolo
        se midió. Eso no cambia nunca, y por eso ahora se compara con el
        literal de entonces. Lo que sí hay que seguir exigiendo —y va debajo,
        separado— es que lo que la pasada USÓ no se haya movido: los motores,
        y la regla con la que se lee el resultado."""
        self.assertEqual(self.alcance["motores"]["del_protocolo"],
                         [m.id for m in self.protocolo.motores])
        self.assertEqual(self.alcance["protocolo"]["digest_sha256"],
                         self.DIGEST_CONTRA_EL_QUE_SE_MIDIO)

    def test_la_RE_FIRMA_no_toco_nada_de_lo_que_esta_pasada_uso(self):
        """La otra mitad, y la que impide que «es un hecho histórico» se
        convierta en una excusa para cualquier cambio. El digest del artefacto
        puede diferir del vigente SOLO si lo que la pasada usó sigue siendo lo
        mismo: los siete motores, los 40 datasets, la partición y —sobre todo—
        la regla de cierre con la que se lee «10/12 CUMPLE».

        Si alguien re-firma tocando alguna de esas cosas, esta prueba se pone
        roja y el veredicto guardado deja de poder leerse con el protocolo de
        hoy, que es exactamente lo que tiene que pasar."""
        self.assertEqual(self.alcance["protocolo"]["version"],
                         self.protocolo.version_protocolo)
        self.assertEqual(self.alcance["protocolo"]["n_motores"], len(self.protocolo.motores))
        self.assertEqual(self.alcance["protocolo"]["n_datasets"], len(self.protocolo.datasets))
        self.assertEqual(self.protocolo.regla_de_cierre.puntos, 2.0)
        self.assertEqual(self.protocolo.regla_de_cierre.fraccion_minima, 0.80)
        # CONTRA EL MOTOR QUE CUMPLE, no contra uno escrito a mano. Aquí ponía
        # `motor="lightgbm"` y `(10, 12)`, que era el veredicto de la evidencia
        # de CUATRO motores. Con los siete, el que cumple es otro — y una
        # prueba que nombra a mano al ganador de ayer mide el ayer, no la
        # propiedad: lo que esto defiende es que **el veredicto guardado se
        # siga obteniendo con el protocolo vigente**, sea de quien sea.
        for motor, guardado in self.payload["alcance_y_veredicto"]["por_motor"].items():
            veredicto = aplicar_regla_de_cierre(
                self.resultados, self.protocolo.regla_de_cierre, motor=motor)
            self.assertEqual((veredicto["cumplidos"], veredicto["datasets"]),
                             (guardado["cumplidos"], guardado["datasets"]),
                             f"el veredicto guardado de {motor} ya no sale con el protocolo "
                             "vigente: la re-firma tocó algo que la pasada usaba")
            self.assertEqual(veredicto["cumple_la_regla"], guardado["cumple_la_regla"], motor)

    def test_los_que_FALTAN_salen_de_restar_las_dos_listas_con_su_mapeo(self):
        """Y se restan traduciendo los nombres. Comparadas a pelo, las dos
        listas dirían que faltan CINCO —`baseline` es el `dummy` del protocolo
        y `sklearn.lineal` su `sklearn.logreg`— y ese número sería falso en la
        dirección alarmista, que miente igual que la tranquilizadora."""
        declarados = self.alcance["motores"]["declarados_por_la_pasada"]
        traducidos = [EQUIVALENCIAS_DE_NOMBRE_DE_MOTOR.get(m, m) for m in declarados]
        esperado = [m.id for m in self.protocolo.motores if m.id not in traducidos]
        self.assertEqual(self.alcance["motores"]["que_faltan"], esperado)

    def test_el_que_NO_compitio_se_nombra_y_hoy_no_falta_NINGUNO(self):
        """El corazón del hallazgo, conservado al cambiar la evidencia.

        Decía «los tres rivales directos de un GBM están nombrados» y exigía
        `{sklearn.hgb, xgboost, catboost}`, porque la evidencia de entonces
        corría 4 motores de 7. **Hoy corren los siete**, así que esa lista
        tiene que estar VACÍA — y forzar la prueba a seguir esperando tres
        ausentes habría convertido una limitación en un contrato.

        Lo que la prueba defiende es lo mismo que defendía: **quien falte se
        nombra**, nunca un recuento. Por eso se miden las dos mitades: que la
        lista cuadre con la diferencia real entre protocolo y pasada, y que
        hoy esa diferencia sea cero con los siete corriendo."""
        faltan = self.alcance["motores"]["que_faltan"]
        m = self.alcance["motores"]
        # La mitad general: lo que falta son NOMBRES y cuadran con las listas.
        esperados = set(m["del_protocolo"]) - set(
            self.alcance["motores"]["equivalencias_de_nombre"].get(x, x)
            for x in m["declarados_por_la_pasada"])
        self.assertEqual(set(faltan), esperados,
                         "la lista de ausentes no cuadra con las dos listas de motores")
        # Y la mitad de HOY, que es la que caduca ruidosa si alguien recorta.
        self.assertEqual(faltan, [],
                         "esta evidencia corre los SIETE motores del protocolo: si vuelve "
                         "a faltar alguno, la promoción de un motor se estrecha otra vez")
        self.assertEqual(m["n_que_corrieron"], m["n_del_protocolo"])

    def test_las_cuentas_de_motores_cuadran_con_sus_listas(self):
        """Un contador escrito a mano es el primero que se queda atrás."""
        m = self.alcance["motores"]
        self.assertEqual(m["n_del_protocolo"], len(m["del_protocolo"]))
        self.assertEqual(m["n_que_corrieron"], len(m["declarados_por_la_pasada"]))
        self.assertEqual(len(m["que_faltan"]),
                         m["n_del_protocolo"] - m["n_que_corrieron"])

    # -- datasets --------------------------------------------------------

    def test_los_datasets_DECLARADOS_son_los_que_aparecen_en_los_720(self):
        observados = sorted({r["dataset"] for r in self.resultados})
        self.assertEqual(sorted(self.alcance["datasets"]["declarados_por_la_pasada"]),
                         observados,
                         "el JSON declara unos datasets y sus registros traen otros")
        self.assertEqual(sorted(self.alcance["datasets"]["observados_en_los_resultados"]),
                         observados)

    def test_los_datasets_declarados_son_los_que_el_SCRIPT_corre_hoy(self):
        self.assertEqual(self.alcance["datasets"]["declarados_por_la_pasada"],
                         [nombre for _id, nombre, _c, _p, _n in DATASETS])

    def test_los_que_faltan_mas_los_que_corrieron_son_los_40_del_protocolo(self):
        d = self.alcance["datasets"]
        self.assertEqual(len(d["del_protocolo"]), 40)
        self.assertEqual(sorted(d["que_faltan"] + d["declarados_por_la_pasada"]),
                         sorted(d["del_protocolo"]))

    def test_el_CRITERIO_del_subconjunto_describe_el_subconjunto_de_verdad(self):
        """Medido, no creído: los 12 que corren son EXACTAMENTE los 12 del
        protocolo que son binarios, de cubo pequeño o mediano y no sellados.
        Si alguien añadiera un dataset que no cumple el criterio, o dejara
        fuera uno que sí, el criterio escrito pasaría a ser una media verdad
        tranquilizadora — y las tranquilizadoras son las peores."""
        elegibles = {d.nombre for d in self.protocolo.datasets
                     if d.tarea == "binary_classification"
                     and d.cubo_de_tamano in ("pequeno", "mediano")
                     and not d.sellado}
        self.assertEqual(set(self.alcance["datasets"]["declarados_por_la_pasada"]),
                         elegibles)
        self.assertEqual(self.alcance["datasets"]["criterio_del_subconjunto"],
                         CRITERIO_DEL_SUBCONJUNTO)

    def test_declara_las_tareas_y_los_cubos_que_se_quedan_SIN_MEDIR(self):
        """Un motor de menos se ve contando. Una TAREA entera sin un solo
        dataset no se ve en ningún recuento, y es más grave: de esta pasada no
        se sigue NADA sobre regresión ni multiclase, por alto que sea el
        0,833. Se deriva del protocolo, no se escribe a mano."""
        corridos = set(self.alcance["datasets"]["declarados_por_la_pasada"])
        cubiertas = {d.tarea for d in self.protocolo.datasets if d.nombre in corridos}
        cubos = {d.cubo_de_tamano for d in self.protocolo.datasets if d.nombre in corridos}
        self.assertEqual(set(self.alcance["sin_medir"]["tareas"]),
                         {d.tarea for d in self.protocolo.datasets} - cubiertas)
        self.assertEqual(set(self.alcance["sin_medir"]["cubos_de_tamano"]),
                         {d.cubo_de_tamano for d in self.protocolo.datasets} - cubos)
        # Y las dos que de verdad faltan, por nombre: una lista vacía pasaría
        # las igualdades de arriba si el protocolo cambiara, y un aserto
        # negativo lo pasa un render en blanco.
        self.assertEqual(set(self.alcance["sin_medir"]["tareas"]),
                         {"multiclass_classification", "regression"})
        self.assertEqual(self.alcance["sin_medir"]["cubos_de_tamano"], ["grande"])


class GanarYGanarPorNadaNoSeLeenIgualTest(unittest.TestCase):
    """«8 de los 10 aciertos de lightgbm son distancia 0,0000.»

    Re-derivado aquí, no copiado: `distancia_en_puntos` vale 0,0000 SIEMPRE que
    el motor sea el mejor, así que sin una segunda columna «ganó de calle» y
    «ganó por cinco centésimas» se escriben idéntico.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads((_FASE0 / _JSON).read_text(encoding="utf-8"))
        cls.protocolo = ProtocoloExploratorio.cargar(
            str(_FASE0 / "protocolo_exploratorio.json"))
        # EL MOTOR QUE CUMPLE, quienquiera que sea — no `["lightgbm"]`.
        #
        # Esta clase examinaba a lightgbm por su nombre porque era el que
        # cumplía con la evidencia de CUATRO motores. Con los siete cumple
        # otro (catboost), y dejar el nombre escrito habría hecho que estas
        # pruebas midieran el veredicto de ayer. Lo que defienden no es quién
        # gana: es que **un veredicto que CUMPLE viaje con lo que le costó**
        # —cuántos ganó por nada, contra quién, y a cuánto está del borde—,
        # y eso vale para cualquier motor.
        por_motor = cls.payload["alcance_y_veredicto"]["por_motor"]
        cumplen = [m for m, v in por_motor.items() if v["cumple_la_regla"]]
        assert len(cumplen) == 1, (
            f"esta clase lee EL veredicto que cumple y hay {len(cumplen)}: {cumplen}. "
            "Con ninguno no hay nada que auditar; con varios hay que decir cuál se audita.")
        cls.motor_que_cumple = cumplen[0]
        cls.veredicto = por_motor[cls.motor_que_cumple]

    def test_el_numero_del_JSON_sale_de_RE_APLICAR_la_regla_registrada(self):
        """El número no se cree, se re-deriva — sobre los registros del propio
        fichero y con la regla que 101-C1 selló antes de medir."""
        re_derivado = aplicar_regla_de_cierre(
            self.payload["resultados"], self.protocolo.regla_de_cierre,
            motor=self.motor_que_cumple)
        self.assertEqual((re_derivado["cumplidos"], re_derivado["datasets"]),
                         (self.veredicto["cumplidos"], self.veredicto["datasets"]))
        self.assertEqual(re_derivado["cumple_la_regla"], self.veredicto["cumple_la_regla"])

    def test_cuantos_aciertos_son_por_SER_el_mejor_se_cuentan_a_mano(self):
        """La re-derivación independiente del «8 de 10»: contados aquí desde el
        detalle, sin usar el contador que el propio bloque trae."""
        a_mano = sum(1 for d in self.veredicto["detalle"]
                     if d["cumple"] and d["distancia_en_puntos"] is not None
                     and abs(d["distancia_en_puntos"]) < 1e-12)
        self.assertEqual(a_mano, self.veredicto["aciertos_por_ser_el_mejor"])
        # Y los números de HOY, escritos para que caduquen ruidosos: catboost
        # cumple los 12 y **5 de esos 12 los gana por ser el mejor**, no por
        # entrar en el margen. Antes aquí ponía 8 de 10, que era lightgbm con
        # cuatro motores compitiendo; con siete, las dos cifras cambian.
        self.assertEqual((self.motor_que_cumple, a_mano, self.veredicto["cumplidos"]),
                         ("catboost", 5, 12))

    def test_cada_dataset_GANADO_dice_por_CUANTO_y_a_quien(self):
        """Sin esto, los ocho empates a cero se leen como ocho dominios."""
        ganados = [d for d in self.veredicto["detalle"]
                   if d["distancia_en_puntos"] is not None
                   and abs(d["distancia_en_puntos"]) < 1e-12]
        self.assertEqual(len(ganados), self.veredicto["aciertos_por_ser_el_mejor"])
        self.assertEqual(len(ganados), 5, "hoy son 5; eran 8 con cuatro motores")
        for d in ganados:
            self.assertIsNotNone(d["ventaja_sobre_el_segundo_en_puntos"],
                                 f"{d['dataset']}: ganó y no dice por cuánto")
            self.assertIsNotNone(d["segundo"], f"{d['dataset']}: ganó y no dice a quién")
            self.assertGreaterEqual(d["ventaja_sobre_el_segundo_en_puntos"], 0.0)

    def test_la_ventaja_sobre_el_segundo_se_RE_CALCULA_desde_los_crudos(self):
        """El número que distingue «ganó» de «ganó por nada», re-derivado
        desde los 720 intentos sin pasar por el bloque que lo declara."""
        medias: dict[str, dict[str, list[float]]] = {}
        for r in self.payload["resultados"]:
            if r.get("estado") != "completed" or r.get("auroc") is None:
                continue
            if r["motor"] == "baseline":   # la regla lo excluye de «el mejor»
                continue
            medias.setdefault(r["dataset"], {}).setdefault(r["motor"], []).append(
                float(r["auroc"]))
        for d in self.veredicto["detalle"]:
            if d["ventaja_sobre_el_segundo_en_puntos"] is None:
                continue
            por_motor = {m: sum(v) / len(v) for m, v in medias[d["dataset"]].items()}
            orden = sorted(por_motor, key=lambda m: por_motor[m], reverse=True)
            self.assertEqual(orden[0], self.motor_que_cumple, d["dataset"])
            self.assertEqual(d["segundo"], orden[1], d["dataset"])
            self.assertAlmostEqual(
                d["ventaja_sobre_el_segundo_en_puntos"],
                (por_motor[orden[0]] - por_motor[orden[1]]) * 100.0, places=9,
                msg=d["dataset"])

    def test_CUANTO_MARGEN_tiene_el_que_cumple_viaja_con_el_veredicto(self):
        """Que un veredicto que cumple esté en el borde —o no— es parte de
        cómo hay que leerlo, y por eso viaja con él.

        Se llamaba «el margen es de UN solo dataset» y exigía CERO: era
        lightgbm con cuatro motores, 10/12, que no podía perder ninguno más.
        Hoy cumple catboost con 12/12 y le sobran **dos**. Fijar el cero
        habría convertido «estaba en el borde» en un requisito.

        Lo que se comprueba es la propiedad: el margen declarado es el que
        sale de la regla, y perder uno más de la cuenta baja del listón."""
        self.assertTrue(self.veredicto["cumple_la_regla"])
        margen = self.veredicto["datasets_que_puede_perder_sin_incumplir"]
        self.assertIsNotNone(margen, "un veredicto que cumple tiene que decir su margen")
        minimo = self.protocolo.regla_de_cierre.fraccion_minima
        n = self.veredicto["datasets"]
        # Con el margen declarado sigue cumpliendo...
        self.assertGreaterEqual((self.veredicto["cumplidos"] - margen) / n, minimo)
        # ...y con uno más, no. Las dos mitades: sin la segunda, un margen
        # inflado pasaría igual.
        self.assertLess((self.veredicto["cumplidos"] - margen - 1) / n, minimo)
        self.assertEqual(margen, 2, "hoy sobran dos; con cuatro motores el margen era cero")


if __name__ == "__main__":
    unittest.main()
