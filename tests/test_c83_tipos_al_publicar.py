"""LOS TIPOS DE INTERFAZ, AL PUBLICAR EN EL REGISTRY — contrato 83.

Medido el 2026-08-19: `matrixai registry push` no escribía `input_type`
ni `output_type`, así que los manifiestos salían con `{}` —incluidos los
dos que ya viajan en `examples/text-routing/registry/`— y
`check_composite_program_types` hace `if not dst_in: continue`. Una
conexión imposible daba `Typecheck OK`, `rc=0`: **la comprobación de
tipos de un compuesto no podía fallar nunca**.

`push_run_dir` YA aceptaba los dos parámetros; quien no los pasaba era el
llamante. Se arregla en `push_run_dir` y no en el CLI a propósito: una
protección que vive en un llamante no protege a los demás — es la misma
lección que dejó el `matrixai_registry/` que apareció en el repo.

Los tipos NO entran en `entry_hash` (`compute_entry_hash` los excluye),
así que anotarlos no rompe la cadena de integridad de P21.
"""

import json
from pathlib import Path

from matrixai.registry.model_registry import ModelRegistry

EJEMPLOS = Path(__file__).resolve().parent.parent / "examples"


def _run_dir(tmp_path, mxai: Path | None, nombre: str = "run"):
    d = tmp_path / nombre
    d.mkdir()
    (d / "evaluation_report.json").write_text(json.dumps({"metrics": {}}))
    if mxai is not None:
        (d / mxai.name).write_text(mxai.read_text())
    return d


class TestSeDeducenDelModeloQueSePublica:
    def test_el_vector_de_entrada_y_la_salida_de_la_red(self, tmp_path):
        origen = EJEMPLOS / "text-routing" / "feature_extractor.mxai"
        reg = ModelRegistry(tmp_path / "reg")
        entry = reg.push_run_dir(_run_dir(tmp_path, origen), "text_encoder", "v1")

        # La ENTRADA es el VECTOR por el que entra el GRAPH, con su talla:
        # sin `size` no se puede comprobar que una conexión encaje.
        assert entry.input_type["name"] == "TicketBOW"
        assert entry.input_type["kind"] == "VECTOR"
        assert entry.input_type["size"] == 30

        # La SALIDA es la que declara la red, DICHA EN EL MISMO
        # VOCABULARIO que la entrada (2026-08-23). El ejemplo declara
        # `OUTPUT routing_signal: Vector[1]` porque es lo que un
        # `Dense(1)` emite y es lo que el siguiente modelo consume; antes
        # decía `Score` y su propio `typecheck` rechazaba un pipeline que
        # acierta 9 de 9.
        assert entry.output_type["name"] == "routing_signal"
        assert entry.output_type["kind"] == "VECTOR"
        assert entry.output_type["size"] == 1

    def test_viajan_en_el_manifiesto_que_se_escribe(self, tmp_path):
        """En el objeto no basta: lo que lee el compositor es el fichero."""
        origen = EJEMPLOS / "text-routing" / "feature_extractor.mxai"
        reg = ModelRegistry(tmp_path / "reg")
        reg.push_run_dir(_run_dir(tmp_path, origen), "text_encoder", "v1")

        manifest = json.loads(
            (tmp_path / "reg" / "entries" / "text_encoder" / "v1" / "manifest.json").read_text())
        assert manifest["input_type"]["size"] == 30
        assert manifest["output_type"]["kind"] == "VECTOR"


class TestNoSeInventaNada:
    """Un valor ausente no es un cero, y un tipo inventado es peor que
    ninguno: el compositor daría por buena una conexión que no lo es."""

    def test_sin_modelo_no_hay_tipos(self, tmp_path):
        reg = ModelRegistry(tmp_path / "reg")
        entry = reg.push_run_dir(_run_dir(tmp_path, None), "sin_modelo", "v1")
        assert not entry.input_type
        assert not entry.output_type

    def test_lo_que_declara_quien_publica_manda(self, tmp_path):
        """Deducir es la ayuda, no la ley: un tipo explícito no se pisa."""
        origen = EJEMPLOS / "text-routing" / "feature_extractor.mxai"
        reg = ModelRegistry(tmp_path / "reg")
        suyo = {"name": "Otro", "kind": "Tensor", "shape": [7]}
        entry = reg.push_run_dir(
            _run_dir(tmp_path, origen), "text_encoder", "v1", input_type=suyo)
        assert entry.input_type == suyo
        # Y el que NO declaró sí se deduce: media ayuda es peor que ninguna.
        assert entry.output_type["name"] == "routing_signal"


class TestLaIntegridadDeP21NoSeToca:
    def test_los_tipos_no_cambian_el_entry_hash(self, tmp_path):
        origen = EJEMPLOS / "text-routing" / "feature_extractor.mxai"
        con = ModelRegistry(tmp_path / "a").push_run_dir(
            _run_dir(tmp_path, origen, "run_a"), "text_encoder", "v1")
        sin = ModelRegistry(tmp_path / "b").push_run_dir(
            _run_dir(tmp_path, origen, "run_b"), "text_encoder", "v1",
            input_type={}, output_type={})
        assert con.entry_hash == sin.entry_hash


class TestLoQueElModeloNoDICENoSeINVENTA:
    """`route_classifier.mxai` declara `OUTPUT probs: Tensor` **sin forma**,
    y su última capa es `Dense units=3`. Se podría inferir `shape: [3]` —y
    se decide NO hacerlo: el core no lo ha dicho, y una forma equivocada
    rompe conexiones válidas, que es peor que no comprobarlas.

    Consecuencia medida y aceptada: un `Tensor` sin forma casa con
    CUALQUIER `Tensor` (`_composite_types_compatible` solo compara la
    forma cuando los dos la traen). O sea, se comprueba el tipo pero no la
    talla — que es exactamente lo que ese modelo declara de sí mismo.
    """

    def test_la_salida_sin_forma_se_publica_sin_forma(self, tmp_path):
        origen = EJEMPLOS / "text-routing" / "route_classifier.mxai"
        reg = ModelRegistry(tmp_path / "reg")
        entry = reg.push_run_dir(_run_dir(tmp_path, origen), "route_classifier", "v1")
        assert entry.output_type == {"name": "probs", "kind": "Tensor"}
        assert "shape" not in entry.output_type

    def test_y_esa_salida_casa_con_cualquier_tensor(self):
        from matrixai.types import _composite_types_compatible
        sin_forma = {"name": "probs", "kind": "Tensor"}
        assert _composite_types_compatible(sin_forma, {"kind": "Tensor", "shape": [3]})
        # Pero el TIPO sí se comprueba: no es una comprobación vacía.
        assert not _composite_types_compatible(sin_forma, {"kind": "VECTOR", "size": 3})


class TestUnVectorQueNoCasaSeDETECTA:
    """Publicar la talla no sirve de nada si nadie la mira.

    `_composite_types_compatible` comparaba `kind` y, solo para `Tensor`,
    la forma: un `VECTOR[2]` casaba con un `VECTOR[30]` (medido
    2026-08-20). Con los manifiestos vacíos de antes daba igual —no había
    talla que comparar—, pero en cuanto `push_run_dir` la publica, ignorarla
    convierte el `Typecheck OK` en una media verdad tranquilizadora.

    Simétrico a lo de `Tensor`, y con la misma prudencia: si a UNO de los
    dos lados no le consta la talla, no se inventa un desencaje.
    """

    def test_tallas_distintas_no_encajan(self):
        from matrixai.types import _composite_types_compatible as compatible
        assert not compatible({"kind": "VECTOR", "size": 2},
                              {"kind": "VECTOR", "size": 30})

    def test_la_misma_talla_encaja(self):
        from matrixai.types import _composite_types_compatible as compatible
        assert compatible({"kind": "VECTOR", "size": 30},
                          {"kind": "VECTOR", "size": 30})

    def test_sin_talla_declarada_no_se_inventa_un_desencaje(self):
        from matrixai.types import _composite_types_compatible as compatible
        # Es el caso de todo lo publicado ANTES de este corte: sin talla,
        # se comprueba el tipo y nada más. Declarar un fallo aquí tiraría
        # compuestos que hoy funcionan.
        assert compatible({"kind": "VECTOR"}, {"kind": "VECTOR", "size": 30})
        assert compatible({"kind": "VECTOR", "size": 2}, {"kind": "VECTOR"})

    def test_talla_cero_es_un_dato_no_un_hueco(self):
        from matrixai.types import _composite_types_compatible as compatible
        # `0` es falsy en Python: comprobarlo con `if src_size` lo trataría
        # como «no consta» y dejaría pasar un vector vacío contra uno de 30.
        assert not compatible({"kind": "VECTOR", "size": 0},
                              {"kind": "VECTOR", "size": 30})


class TestPorElPRODUCTONoSoloPorLaFUNCION:
    """Probar la función no es probar el producto.

    Medido: con `_composite_types_compatible` ya comparando la talla, el
    CLI seguía diciendo `Typecheck OK` — porque `_node_output_type`
    describía un vector local como `{"kind": "VECTOR", "name": ...}`, **sin
    talla**, y no había nada que comparar. Dos arreglos, y solo este
    bloque los mide juntos: publicar la talla y usarla.
    """

    def _registry_con_encoder(self, tmp_path):
        origen = EJEMPLOS / "text-routing" / "feature_extractor.mxai"
        reg = ModelRegistry(tmp_path / "reg")
        reg.push_run_dir(_run_dir(tmp_path, origen), "enc", "v1")
        return tmp_path / "reg"

    def _typecheck(self, fuente, registry):
        """Por donde pasa el producto: parsear y comprobar el compuesto."""
        from matrixai.parser.parser import parse_text
        from matrixai.types import check_composite_program_types
        from matrixai.registry.model_registry import ModelRegistry as R
        programa = parse_text(fuente)
        return check_composite_program_types(programa, R(registry))

    def _compuesto(self, talla):
        campos = "\n".join(f"  c{i}: Score" for i in range(talla))
        return (f"PROJECT P\n\nIMPORT Enc FROM registry enc@v1 FROZEN\n\n"
                f"VECTOR TicketBOW[{talla}]\n{campos}\nEND\n\n"
                f"GRAPH\n  TicketBOW -> Enc\nEND\n")

    def test_una_talla_que_no_encaja_se_rechaza(self, tmp_path):
        import pytest
        from matrixai.types import TypeCompatibilityError
        registry = self._registry_con_encoder(tmp_path)
        with pytest.raises(TypeCompatibilityError) as caja:
            self._typecheck(self._compuesto(2), registry)
        # El mensaje enseña LAS DOS tallas: «no encaja» a secas obliga a
        # abrir los dos modelos para saber cuál cambiar.
        errores = " ".join(getattr(caja.value, "errors", []) or [str(caja.value)])
        assert "'size': 2" in errores
        assert "'size': 30" in errores

    def test_la_talla_correcta_pasa(self, tmp_path):
        registry = self._registry_con_encoder(tmp_path)
        # Sin excepción: si este caso también fallara, el arreglo no
        # comprobaría nada, solo rompería.
        self._typecheck(self._compuesto(30), registry)


class TestLosDosExtremosHablanElMismoIdioma:
    """La entrada salía normalizada (`{kind: VECTOR, size: n}`) y la
    salida en CRUDO, la cadena tal cual la escribió el `.mxai`. Medido el
    2026-08-23: `"Vector[1]"` no casaba ni con un `VECTOR`, así que
    **ninguna pareja publicada por `registry push` podía encajar jamás** —
    el ejemplo `text-routing`, que acierta 9 de 9, era rechazado por su
    propio verificador.

    Esto NO afloja nada: un `Score` sigue sin entrar en un `VECTOR[1]`, y
    las tallas se siguen comparando. Solo hace comparables las dos
    descripciones."""

    def test_un_vector_se_dice_como_se_dice_en_la_entrada(self):
        from matrixai.registry.interface_types import _describir_salida
        assert _describir_salida("s", "Vector[1]") == {"name": "s", "kind": "VECTOR", "size": 1}

    def test_un_tensor_lleva_su_forma(self):
        from matrixai.registry.interface_types import _describir_salida
        assert _describir_salida("p", "Tensor[3]") == {"name": "p", "kind": "Tensor", "shape": [3]}

    def test_un_escalar_se_queda_como_estaba(self):
        """Lo que no es contenedor no cambia: `Score` sigue siendo `Score`,
        y por eso sigue SIN encajar en un `VECTOR[1]`."""
        from matrixai.registry.interface_types import _describir_salida
        from matrixai.types import _composite_types_compatible
        assert _describir_salida("s", "Score") == {"name": "s", "kind": "Score"}
        assert not _composite_types_compatible({"kind": "Score"}, {"kind": "VECTOR", "size": 1})

    def test_las_etiquetas_no_son_el_tipo(self):
        """`ProbabilityMap[a,b]` y `ProbabilityMap[c,d]` son el mismo tipo
        con distintas etiquetas: en crudo se leían como tipos distintos."""
        from matrixai.registry.interface_types import _describir_salida
        uno = _describir_salida("p", "ProbabilityMap[billing,tech]")
        otro = _describir_salida("p", "ProbabilityMap[a,b]")
        assert uno == otro == {"name": "p", "kind": "ProbabilityMap"}

    def test_lo_que_no_se_sabe_leer_no_se_inventa(self):
        from matrixai.registry.interface_types import _describir_salida
        assert _describir_salida("x", "NoSoyUnTipo") is None


class TestElEjemploPasaSuPropioVerificador:
    """La regresión exacta: `examples/text-routing` viaja publicado en el
    repo y su `typecheck` fallaba. Se comprueba contra el registry
    COMMITEADO, que es lo que se descarga cualquiera."""

    def test_el_pipeline_del_ejemplo_typechequea(self):
        from matrixai.parser import parse_file
        from matrixai.registry import ModelRegistry
        from matrixai.types import check_composite_program_types

        raiz = EJEMPLOS / "text-routing"
        programa = parse_file(raiz / "text_routing_pipeline.mxai")
        registro = ModelRegistry(raiz / "registry")
        assert check_composite_program_types(programa, registro).errors == []
