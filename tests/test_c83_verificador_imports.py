"""EL VERIFICADOR Y LAS PIEZAS IMPORTADAS — contrato 83.

Medido el 2026-08-20: `matrixai validate` y `matrixai lint` RECHAZABAN
todo programa compuesto, incluidos **los ejemplos que viajan en este
mismo repositorio**:

    $ matrixai validate examples/text-routing/text_routing_pipeline.mxai
    Error: GRAPH references undeclared node 'RouteClassifier'
    Error: GRAPH references undeclared node 'FeatureExtractor'

`VerifierAgent._declared_nodes` reunía vectors, sequences, functions,
distributions, actions y networks — y no `program.imports`, así que el
alias de una pieza importada nunca estaba declarado y el `GRAPH` que la
usa se declaraba inválido.

Es el MISMO fallo que esa función ya documenta para `SEQUENCE`, y por el
mismo motivo vivió tanto: ningún test pasaba el verificador sobre un
programa con `IMPORT`. Este fichero es ese test.
"""

from pathlib import Path

from matrixai.agents.verifier import VerifierAgent
from matrixai.parser.parser import parse_file, parse_text

EJEMPLOS = Path(__file__).resolve().parent.parent / "examples"


def _errores(program):
    return VerifierAgent().verify(program).errors


class TestElAliasImportadoEsUnNodoDeclarado:
    def test_el_ejemplo_del_repositorio_se_verifica(self):
        """El cierre real: lo que el producto trae de fábrica pasa."""
        program = parse_file(str(EJEMPLOS / "text-routing" / "text_routing_pipeline.mxai"))
        assert _errores(program) == []

    def test_el_segundo_ejemplo_ya_no_señala_sus_piezas_importadas(self):
        """El aserto es el de los ALIAS, no el del fichero entero.

        Medido: `examples/text-routing-pipeline.mxai` tiene además dos
        defectos PROPIOS que este arreglo no toca y que no debe tapar —
        un nodo `Output` que nadie declara, y un `NETWORK Router` cuyo
        `INPUT` es un alias importado (ver la nota de abajo). Pedirle al
        fichero que salga limpio sería exigirle a este arreglo algo que
        no es suyo.
        """
        program = parse_file(str(EJEMPLOS / "text-routing-pipeline.mxai"))
        errores = " ".join(_errores(program))
        for imp in program.imports:
            assert f"undeclared node '{imp.alias}'" not in errores
        # Y el defecto propio del ejemplo sigue señalado: si esto dejara
        # de verse, el arreglo habría apagado un aviso verdadero.
        assert "undeclared node 'Output'" in errores

    def test_el_alias_importado_cuenta_como_declarado(self):
        program = parse_file(str(EJEMPLOS / "text-routing" / "text_routing_pipeline.mxai"))
        alias = {imp.alias for imp in program.imports}
        assert alias, "el ejemplo debe traer imports; si no, esta prueba no mide nada"
        # Lo que se arregla: el alias entra en el conjunto de nodos
        # declarados, igual que un VECTOR o un NETWORK.
        assert alias <= VerifierAgent()._declared_nodes(program)


class TestSigueCazandoLoQueDebeCazar:
    """Un arreglo que hiciera pasar TODO no sería un arreglo.

    Sin este bloque, `_declared_nodes` podría devolver cualquier cosa y
    las tres pruebas de arriba seguirían verdes: un aserto que solo pide
    «no hay errores» lo aprueba un verificador que no verifica.
    """

    FUENTE = "\n".join([
        "PROJECT Suelto",
        "",
        "IMPORT Enc FROM registry signal_encoder@v1 FROZEN",
        "",
        "VECTOR Features[2]",
        "  a: Score",
        "  b: Score",
        "END",
        "",
        "GRAPH",
        "  Features -> Enc -> NoExiste",
        "END",
        "",
    ])

    def test_un_nodo_que_nadie_declara_sigue_siendo_un_error(self):
        errores = " ".join(_errores(parse_text(self.FUENTE)))
        assert "NoExiste" in errores
        # Y el alias importado NO se señala: es el caso que se arregla.
        assert "'Enc'" not in errores


class TestLoQueEsteARREGLONoCierra:
    """Un hueco HERMANO, medido al escribir esto y dejado escrito.

    `examples/text-routing-pipeline.mxai` declara
    `NETWORK Router` con `INPUT TextEncoder`, donde `TextEncoder` es un
    alias importado, y el verificador responde:

        NETWORK Router: INPUT 'TextEncoder' is not a declared VECTOR

    O sea: la comprobación de las entradas de un NETWORK tiene el mismo
    hueco que tenía `_declared_nodes` — no conoce los imports. NO se
    arregla aquí porque no está medido si alimentar un NETWORK con la
    salida de una pieza importada debe ser legal (es una decisión de
    diseño del lenguaje, no un descuido evidente), y arreglar de más
    también rompe.

    Esta prueba FIJA el comportamiento de hoy para que el día que se
    decida, se decida a la vista y no por accidente.
    """

    def test_un_network_alimentado_por_un_import_sigue_rechazandose(self):
        program = parse_file(str(EJEMPLOS / "text-routing-pipeline.mxai"))
        errores = " ".join(_errores(program))
        assert "INPUT 'TextEncoder' is not a declared VECTOR" in errores
