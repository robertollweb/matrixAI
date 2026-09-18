# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Roberto Llamosas Conde

"""Edge bundle: packages .mxai + params + model.onnx + manifests into a deployable directory."""
from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matrixai.ir import MatrixAIProgram
from matrixai.export.onnx_exporter import validate_export_parameter_set
from matrixai.parameters.store import ParameterSet, program_hash
from matrixai.export.onnx_exporter import OnnxExporter, OnnxExportResult, OnnxExportError
from matrixai.export.inference_spec import (
    build_inference_spec,
    build_example_input,
    InferenceSpecError,
    _matrixai_version,
)

# The standalone predict.py shipped inside every usable bundle (copied verbatim).
_PREDICT_TEMPLATE = str(Path(__file__).resolve().parent / "predict_template.py")
_REQUIREMENTS = "numpy>=1.24\nonnxruntime>=1.16\n"
from matrixai.export.space import SPACE_DIR, space_index_html, space_readme_md
# `predict.js` del Space: EL MISMO que genera el bundle WASM (contrato
# WASM_EXPORT), no una segunda implementación — ver el comentario junto a
# donde se escribe, más abajo.
from matrixai.export.wasm_exporter import _build_predict_js
from matrixai.export.reproduce import (
    TRAINING_ARTIFACT_NAME,
    write_reproduce_manifest,
)
from matrixai.export.equivalence import (
    OnnxEquivalenceResult,
    OnnxEquivalenceValidator,
    write_export_manifest,
    ort_available,
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    _DEFAULT_N_SAMPLES,
)


class EdgeBundleError(ValueError):
    pass


@dataclass(frozen=True)
class EdgeBundleResult:
    bundle_dir: str
    files: list[str]
    model_hash: str
    parameter_set_id: str
    export_result: OnnxExportResult
    equivalence_result: OnnxEquivalenceResult | None = None
    # Why the bundle has no inference_spec.json (SEQUENCE/multi-input, unlabelled
    # classification, ...). None means the spec was produced. The bundle stays valid
    # either way; this makes the omission observable instead of silent.
    inference_spec_skipped_reason: str | None = None
    # PESOS_GRANDES C7b: por qué la equivalencia ONNX==referencia NO se
    # verificó (modelo grande vía `state_dict` — verificarla exigiría
    # materializar tensores a listas Python, el `.tolist()` que este export
    # evita). `None` significa que SÍ se verificó (comportamiento de siempre).
    equivalence_skipped_reason: str | None = None
    # PESOS_GRANDES C7 auditoría: en el modo STREAMING (mxw_path), el bundle NO
    # escribe `model.onnx.data` — el caller (Studio) lo streamea desde el `.mxw`
    # directo al zip final. Aquí van los tensores del `.mxw` en el orden EXACTO
    # en que deben concatenarse en el `.data` (offsets del grafo). `None` en el
    # modo normal/state_dict (el `.onnx.data` ya está en `bundle_dir`).
    external_data_layout: list[dict] | None = None
    # Contrato 82-C1: el `reproduce.json` que se escribió, tal cual. Se
    # devuelve entero (y no solo un booleano) para que el llamante pueda
    # enseñar el motivo de un `reproducible: false` sin volver a leer el
    # fichero — y para que no haya un segundo sitio decidiendo qué falta.
    reproduce: dict[str, Any] | None = None

    @property
    def equivalence_passed(self) -> bool:
        return self.equivalence_result is not None and self.equivalence_result.passed

    @property
    def has_inference_spec(self) -> bool:
        return self.inference_spec_skipped_reason is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "bundle_dir": self.bundle_dir,
            "files": self.files,
            "model_hash": self.model_hash,
            "parameter_set_id": self.parameter_set_id,
            "equivalence_passed": self.equivalence_passed,
            "export": self.export_result.to_dict(),
            "inference_spec_skipped_reason": self.inference_spec_skipped_reason,
            "equivalence_skipped_reason": self.equivalence_skipped_reason,
            "reproduce": self.reproduce,
        }
        if self.equivalence_result is not None:
            d["equivalence_check"] = self.equivalence_result.to_dict()
        return d


class EdgeBundler:
    """Creates a self-contained edge bundle directory from a .mxai + ParameterSet."""

    def bundle(
        self,
        program: MatrixAIProgram,
        parameter_set: ParameterSet | None,
        mxai_path: str | Path,
        params_path: str | Path | None,
        outdir: str | Path,
        *,
        state_dict: dict[str, Any] | None = None,
        mxw_path: str | Path | None = None,
        mxw_header: dict[str, Any] | None = None,
        materialize_external_data: bool = False,
        model_hash: str | None = None,
        parameter_schema_hash: str | None = None,
        validate: bool = True,
        atol: float = _DEFAULT_ATOL,
        rtol: float = _DEFAULT_RTOL,
        n_samples: int = _DEFAULT_N_SAMPLES,
        force: bool = False,
        field_ranges: dict[str, tuple[float, float]] | None = None,
        field_categories: dict[str, list[str]] | None = None,
        field_types: dict[str, str] | None = None,
        field_seq: dict[str, dict[str, Any]] | None = None,
        labels: list[str] | None = None,
        example_input: dict[str, Any] | None = None,
        target_range: tuple[float, float] | None = None,
        data_recipe: str | None = None,
        mxtrain_path: str | Path | None = None,
        dataset_sha256: str | None = None,
        dataset_rows: int | None = None,
        generation: dict[str, Any] | None = None,
        metrics: list[dict[str, Any]] | None = None,
        run_provenance: dict[str, Any] | None = None,
        weights_source: str | None = None,
        # CONTRATO 103 C1 — el problema confirmado (un `ProblemSpec` del 104-C0
        # o su `a_json()`). Viaja al manifiesto para que el orden de las clases
        # y la clase positiva salgan DENTRO del paquete: un vector de
        # probabilidades sin el orden de sus clases no se puede leer.
        problem: Any | None = None,
        # CONTRATO 109-C3 — el expediente clínico. `expediente_clinico.py` ya
        # sabe LEER `clinical_profile.json` (lo MIDIÓ el core, un
        # `PerfilClinico` de 109-C2 o su `a_json()`) y `team_declaration.json`
        # (lo DECLARÓ una persona: población, uso previsto, criterios de
        # inclusión...). Los dos son opcionales y los dos se omiten del
        # paquete cuando no hay nada que escribir — el núcleo no fabrica una
        # declaración que nadie hizo.
        clinical_profile: Any | None = None,
        team_declaration: dict[str, Any] | None = None,
    ) -> EdgeBundleResult:
        """PESOS_GRANDES C7b: `state_dict` (tensores torch crudos de un modelo
        grande guardado en `.mxw`) es la alternativa a un `parameter_set` con
        valores. Cuando se da, `parameter_set` sigue siendo obligatorio para
        `_build_model_manifest`/`build_inference_spec` (metadata: hash/shape/
        schema, nunca `.values`) — puede ser una PLANTILLA sin valores
        (`values=None`, el mismo objeto que ya construye
        `build_parameter_template_for_state`). `params_path=None` omite
        `params.best.json` del zip (escribirlo exigiría el JSON completo, el
        mismo `.tolist()` que este camino evita); la validación de
        equivalencia (que sí necesita una REFERENCIA con valores reales) se
        salta con motivo registrado.

        Con ``mxw_path``, ``materialize_external_data=True`` escribe el
        sidecar en el staging del bundle (CLI); ``False`` devuelve su layout
        para que un caller como Studio lo streamee directamente a un ZIP.
        """
        using_state_dict = state_dict is not None
        # PESOS_GRANDES C7 auditoría: modo STREAMING — los pesos NUNCA se traen
        # a RAM; el grafo ONNX se construye desde la cabecera del `.mxw` y el
        # `model.onnx.data` lo streamea el caller (Studio) directo al zip.
        using_mxw_streaming = mxw_path is not None
        external_data_layout: list[dict] | None = None
        outdir = Path(outdir)
        if outdir.exists() and not force:
            raise EdgeBundleError(
                f"Bundle directory {outdir} already exists. "
                "Pass force=True to overwrite."
            )

        # Validate ParameterSet shapes/schema before touching disk. Se salta
        # con state_dict/streaming: el validador genérico (`validate_parameter_
        # set`) exige VALORES reales (no solo shapes) incluso para una
        # plantilla — `OnnxExporter.export`/`export_dense_onnx_graph_external`
        # ya validan `model_hash` y la presencia de cada tensor esperado por su
        # cuenta, así que esto no es un hueco, es la MISMA validación en el
        # sitio correcto.
        if not using_state_dict and not using_mxw_streaming:
            val = validate_export_parameter_set(program, parameter_set)
            if not val.ok:
                raise EdgeBundleError(
                    f"ParameterSet validation failed: {'; '.join(val.errors)}"
                )

        # Build in a temp dir; rename to outdir only when everything succeeds.
        tmp_parent = outdir.parent
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent, prefix=".bundle_tmp_") as _tmp:
            work = Path(_tmp)

            # 1. Copy .mxai and params (params.best.json se omite para un
            # grande vía state_dict — ver docstring).
            shutil.copy2(str(mxai_path), str(work / "model.mxai"))
            if params_path is not None:
                shutil.copy2(str(params_path), str(work / "params.best.json"))

            # 1b. EL CONTRATO DE ENTRENAMIENTO (contrato 82-C1).
            #
            # El `.mxai` dice qué es el modelo; el `.mxtrain` dice cómo se
            # entrenó —dataset, split, pérdida, optimizador, épocas, backend—.
            # Sin él el paquete se puede USAR y no se puede REHACER: quien lo
            # recibe puede predecir, pero no reentrenar ni comprobar que las
            # métricas publicadas son ciertas.
            #
            # Entra con nombre FIJO (`model.mxtrain`), como el `.mxai`: el
            # nombre del fichero de origen es del proyecto de quien exporta y
            # no tiene por qué viajar dentro del paquete.
            training_filename: str | None = None
            if mxtrain_path is not None:
                shutil.copy2(str(mxtrain_path), str(work / TRAINING_ARTIFACT_NAME))
                training_filename = TRAINING_ARTIFACT_NAME

            # 2. Export ONNX
            onnx_dest = work / "model.onnx"
            try:
                if using_mxw_streaming:
                    # PESOS_GRANDES C7 auditoría: solo el GRAFO se escribe aquí
                    # (`model.onnx`, initializers EXTERNAL); el `model.onnx.data`
                    # lo streamea el caller desde el `.mxw`. `ordered_metas` da el
                    # orden de concatenación que casa con los offsets del grafo.
                    from matrixai.export.onnx_exporter import export_onnx_graph_external
                    export_result, external_data_layout = export_onnx_graph_external(
                        program, mxw_header, onnx_dest,
                        model_hash=model_hash, parameter_schema_hash=parameter_schema_hash,
                    )
                else:
                    export_result = OnnxExporter().export(
                        program, parameter_set, onnx_dest,
                        state_dict=state_dict, model_hash=model_hash,
                        parameter_schema_hash=parameter_schema_hash,
                    )
            except OnnxExportError as exc:
                raise EdgeBundleError(f"ONNX export failed: {exc}") from exc

            # 3. Equivalence validation
            eq_result: OnnxEquivalenceResult | None = None
            eq_skipped_reason: str | None = None
            if validate and (using_state_dict or using_mxw_streaming):
                # PESOS_GRANDES C7b: verificar equivalencia exige correr un
                # forward de REFERENCIA con valores reales (`parameter_set`
                # con `.values`) — para un state_dict grande eso es
                # exactamente el `.tolist()` que este camino evita. Se salta
                # con el motivo registrado (mismo patrón que
                # `inference_spec_skipped_reason`), nunca en silencio.
                eq_skipped_reason = (
                    "Equivalencia no verificada: el modelo es grande (pesos en "
                    ".mxw binario) y verificarla exigiría materializar los "
                    "tensores a listas Python — el mismo .tolist() que este "
                    "export evita. El ONNX se generó directamente desde los "
                    "mismos tensores entrenados; validado por separado en el "
                    "cierre duro del contrato."
                )
            elif validate:
                if not ort_available():
                    raise EdgeBundleError(
                        "Equivalence validation requires 'onnxruntime'. "
                        "Install with: pip install matrixai-core[export]"
                    )
                eq_result = OnnxEquivalenceValidator().validate(
                    program, parameter_set, onnx_dest,
                    atol=atol, rtol=rtol, n_samples=n_samples,
                )
                if not eq_result.passed:
                    raise EdgeBundleError(
                        f"Equivalence check FAILED: max_abs_diff={eq_result.max_abs_diff:.2e} "
                        f"exceeds tolerance atol={atol:.0e} + rtol={rtol:.0e}. "
                        "Bundle not created."
                    )

            # 4. model_manifest.json
            (work / "model_manifest.json").write_text(
                json.dumps(_build_model_manifest(program, parameter_set), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

            # 4b. inference_spec.json — the "tokenizer": how a raw record becomes the
            # normalized float32 vector the ONNX graph expects. Without it the bundle
            # is not self-usable (see EXPORT_MODELO_DESCARGABLE_CONTRACT C1).
            # Best-effort: models that cannot produce a usable spec (SEQUENCE,
            # multi-input, unlabelled classification) keep producing the rest of the
            # bundle as before; they just don't get an inference_spec (nor, later,
            # predict.py). The reason is surfaced on the result and as a warning so the
            # omission is observable instead of silent (the Studio turns it into a
            # user-facing message in C4).
            spec_skipped_reason: str | None = None
            try:
                inference_spec = build_inference_spec(
                    program, parameter_set, export_result,
                    field_ranges=field_ranges,
                    field_categories=field_categories,
                    field_types=field_types,
                    field_seq=field_seq,
                    labels=labels,
                    example_input=example_input,
                    target_range=target_range,
                )
                (work / "inference_spec.json").write_text(
                    json.dumps(inference_spec, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except InferenceSpecError as exc:
                inference_spec = None
                spec_skipped_reason = str(exc)
                warnings.warn(
                    f"inference_spec.json omitted from bundle: {spec_skipped_reason} "
                    "The bundle is still valid but is not self-usable for prediction.",
                    stacklevel=2,
                )

            # 4c. Self-usable prediction artifacts (EXPORT C2): the standalone
            # predict.py wrapper, its requirements, a safe raw example and the
            # expected output of running it. Only when a usable spec exists.
            example_record: dict[str, Any] | None = None
            smoke_test_skipped = False
            if inference_spec is not None:
                shutil.copy2(_PREDICT_TEMPLATE, str(work / "predict.py"))
                (work / "requirements.txt").write_text(_REQUIREMENTS, encoding="utf-8")
                example_record = example_input or build_example_input(inference_spec)
                (work / "example_input.json").write_text(
                    json.dumps(example_record, indent=2, ensure_ascii=False), encoding="utf-8")
                if using_mxw_streaming:
                    # PESOS_GRANDES C7 auditoría: NO se ejecuta el smoke-test
                    # (`_run_prediction`) para un modelo grande — el
                    # `model.onnx.data` ni siquiera está presente aquí (lo
                    # streamea el caller directo al zip), y cargar un ONNX de
                    # GiBs en onnxruntime + un forward CPU dentro del POST de
                    # export es exactamente el coste que este corte evita. El
                    # bundle es self-usable igual (predict.py + inference_spec +
                    # example_input); solo omite el `expected_output.json` de
                    # referencia (que el usuario puede regenerar con `python
                    # predict.py example_input.json`).
                    smoke_test_skipped = True
                else:
                    # A usable bundle is only shipped fully formed: predict.py is smoke-tested
                    # at packaging time and expected_output.json is its baseline. If we cannot
                    # run it (no onnxruntime), we refuse to stamp the bundle as usable rather
                    # than emit it half-built with a README that references a missing file.
                    if not ort_available():
                        raise EdgeBundleError(
                            "Model yields a usable inference_spec, but onnxruntime is not "
                            "installed, so predict.py cannot be smoke-tested and "
                            "expected_output.json cannot be generated. Install it "
                            "(pip install 'matrixai-core[export]') to produce a self-usable bundle."
                        )
                    expected = _run_prediction(work, example_record)
                    (work / "expected_output.json").write_text(
                        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")

            # 5. export_manifest.json
            if eq_result is not None:
                write_export_manifest(export_result, eq_result, work / "export_manifest.json")
            else:
                _write_export_manifest_no_eq(export_result, work / "export_manifest.json")

            # 5b. LA RECETA DE LOS DATOS (contrato 80-C3).
            #
            # Un paquete dice hoy con qué modelo se predice, pero no CON QUÉ
            # DATOS se entrenó. Si esos datos se generaron con una receta,
            # va aquí en texto llano: quien recibe el paquete puede volver a
            # generar el mismo dataset —misma receta, misma semilla, mismas
            # filas— y comprobar por su cuenta lo que decimos.
            #
            # Es lo que separa «créenos» de «compruébalo», y es la mitad que
            # hacía falta para publicar un caso reproducible.
            recipe_filename: str | None = None
            if data_recipe and data_recipe.strip():
                (work / "data_recipe.txt").write_text(
                    data_recipe.strip() + "\n", encoding="utf-8")
                recipe_filename = "data_recipe.txt"

            # 5b2. EL EXPEDIENTE CLÍNICO (contrato 109-C3).
            #
            # `matrixai.export.expediente_clinico` ya sabe LEER estos dos
            # ficheros y de ellos componer la ficha TRIPOD+AI y los huecos
            # PROBAST+AI — pero hasta este corte nadie los ESCRIBÍA: el
            # lector estaba probado sobre un fichero que ningún paquete real
            # llegaba a traer nunca. `clinical_profile.json` es lo que MIDIÓ
            # el core (109-C2, `PerfilClinico.a_json()`, tal cual, sin
            # tocarlo — dos sitios recomponiendo el mismo sobre acaban
            # divergiendo). `team_declaration.json` es DISTINTO: lo declara
            # una persona, y el núcleo no puede inventar población, uso
            # previsto, criterios de inclusión, ni el resto — solo lo acepta
            # si se lo dan (109-C3, invariante 5).
            #
            # LOS DOS SE OMITEN EN SILENCIO cuando no hay nada que escribir:
            # un fichero vacío o inventado sería peor que su ausencia, porque
            # `ExpedienteClinico.desde_paquete()` lo daría por bueno y diría
            # «declarado» de algo que nadie declaró.
            perfil_bloque = build_clinical_profile_block(clinical_profile)
            if perfil_bloque is not None:
                (work / "clinical_profile.json").write_text(
                    json.dumps(perfil_bloque, indent=2, ensure_ascii=False),
                    encoding="utf-8")
            declaracion_bloque = build_team_declaration_block(team_declaration)
            if declaracion_bloque is not None:
                (work / "team_declaration.json").write_text(
                    json.dumps(declaracion_bloque, indent=2, ensure_ascii=False),
                    encoding="utf-8")

            # 5c. EL MANIFIESTO REPRODUCIBLE (contrato 82-C1).
            #
            # Se escribe SIEMPRE, tenga receta o no: un paquete que calla no
            # dice «esto no se puede reproducir», dice nada. Sin receta sale
            # con `reproducible: false` y su motivo (§6.2) — y no se le
            # fabrica ninguna, porque un modelo entrenado con datos reales no
            # tiene receta que compartir y fingir que sí es lo único peor que
            # no poder reproducirlo.
            #
            # `run_provenance` es LA CAPTURA que el core guardó en el run al
            # entrenar, y es lo único desde lo que el manifiesto AFIRMA (82-C2).
            # Lo demás —`dataset_sha256`, `dataset_rows`, `generation`— es lo
            # que manda quien exporta: sirve para detectar que la pantalla dice
            # una cosa y el run dijo otra, nunca para rellenar. Sin captura el
            # paquete sale `reproducible: false` diciendo exactamente eso: no
            # puede demostrar su relación con los pesos que lleva.
            reproduce_manifest = write_reproduce_manifest(
                work,
                training_filename=training_filename,
                recipe_filename=recipe_filename,
                dataset_sha256=dataset_sha256,
                dataset_rows=dataset_rows,
                generation=generation,
                metrics=metrics,
                run_provenance=run_provenance,
                # EL ESTADO DE LOS PESOS QUE ACABAMOS DE EMPAQUETAR. Lo declara
                # quien empaqueta porque es el único que los ha visto: la
                # captura se compone al empezar el run y no sabe qué bytes
                # eligió alguien después. Sin esto el manifiesto no puede decir
                # `reproducible: true` — «no consta» no es «entrenado»— y el
                # aviso se quedaba fuera del ZIP, solo en la respuesta HTTP.
                weights_source=weights_source,
                problem=problem,
            )

            # 6. README.md — refleja los ficheros REALES del bundle (BAJA C7
            # auditoría): con external-data lista `model.onnx.data`; sin
            # `params.best.json` cuando se omitió (grande); sin
            # `expected_output.json` si el smoke-test se saltó.
            # 82-C4 · LA PLANTILLA DEL SPACE, dentro del paquete.
            #
            # Es TEXTO y está ahí para quien la quiera: no crea nada en la
            # cuenta de nadie. Publicar el Space es una casilla al
            # publicar, nunca un efecto de exportar — crear repositorios
            # porque sí es lo que este proyecto evita en todo lo demás.
            #
            # REVISIÓN 2026-09-16: Space ESTÁTICO (HTML + ONNX Runtime Web
            # en el navegador de quien visita), no Gradio — decisión de
            # Roberto, por lo medido el 2026-08-21: un Space de Gradio
            # devuelve 402 a cualquier cuenta sin HF PRO, así que ningún
            # usuario gratuito llegaba nunca a ver la demo; uno estático se
            # crea igual con una cuenta gratuita. El historial de por qué
            # la plantilla anterior llevaba `gradio` y su propio
            # `requirements.txt` queda en el historial de git de
            # `matrixai/export/space.py` — ya no describe lo que se genera.
            _space = work / SPACE_DIR
            _space.mkdir(parents=True, exist_ok=True)
            _space_name = getattr(program, "name", None) or "matrixai-model"
            (_space / "index.html").write_text(
                space_index_html(_space_name), encoding="utf-8")
            (_space / "README.md").write_text(
                space_readme_md(_space_name), encoding="utf-8")
            # `predict.js` — EL MISMO generador que usa el bundle WASM
            # (`wasm_exporter._build_predict_js`), no una segunda
            # implementación: dos generadores de la parte que habla con
            # ONNX Runtime acabarían divergiendo, y la que correría en el
            # navegador de alguien no sería la que se probó aquí.
            # `export_result` es el mismo `OnnxExportResult` que ya
            # describe el `model.onnx` de este mismo paquete.
            #
            # Un Space estático no tiene Python, así que ya no lleva
            # `requirements.txt` propio (era `gradio` + `matrixai-core`
            # para el botón «Is this package intact?», que ejecutaba
            # `matrixai verify` dentro del Space — imposible sin proceso).
            # La página lo dice y da el comando para correrlo en local.
            (_space / "predict.js").write_text(
                _build_predict_js(program, export_result), encoding="utf-8")

            (work / "README.md").write_text(
                _build_readme(program, export_result, eq_result,
                              inference_spec=inference_spec,
                              example_input=example_record,
                              has_params_json=params_path is not None,
                              external_data=export_result.external_data,
                              smoke_test_skipped=smoke_test_skipped,
                              has_mxtrain=training_filename is not None,
                              has_recipe=recipe_filename is not None,
                              reproduce=reproduce_manifest),
                encoding="utf-8",
            )

            # CLI C6 materializa el sidecar por chunks DENTRO del staging para
            # que la promoción del bundle siga siendo atómica. Studio mantiene
            # `materialize_external_data=False`: lo escribe directo al ZIP y
            # evita una copia intermedia multi-GiB.
            if using_mxw_streaming and materialize_external_data:
                from matrixai.parameters.binary_store import stream_mxw_tensors_to_file
                stream_mxw_tensors_to_file(
                    mxw_path,
                    external_data_layout or [],
                    work / "model.onnx.data",
                )

            # EL BYTECODE DE LA PRUEBA DE HUMO NO SE ENTREGA.
            #
            # `_run_prediction` importa `predict.py` con importlib para probarlo
            # aquí mismo, y ese import deja un `__pycache__/` dentro del staging
            # que luego viajaba entero al bundle. Medido el 2026-08-13 sobre un
            # export real del Studio: el zip traía
            # `__pycache__/predict.cpython-312.pyc`, que además NO figura en la
            # lista `files` de más abajo (solo mira ficheros sueltos) — el
            # paquete llevaba algo que su propio manifiesto no declaraba, y con
            # la versión del intérprete de la máquina que lo empaquetó dentro.
            #
            # Se limpia AQUÍ, antes de la promoción, y no en quien empaqueta:
            # las dos rutas de export del Studio (zip por `make_archive` y zip
            # streaming de pesos grandes) recortaban distinto, así que el mismo
            # modelo salía con ficheros distintos según su tamaño.
            for pycache in work.rglob("__pycache__"):
                shutil.rmtree(str(pycache), ignore_errors=True)

            # Atomic promotion: remove stale outdir then rename temp into place
            if outdir.exists():
                shutil.rmtree(str(outdir))
            # EL INVENTARIO, AL FINAL: cuando ya está todo dentro. El
            # `space/` y el `README.md` se escriben después del
            # manifiesto, así que hacerlo antes dejaría fuera justo los
            # ficheros que la página del Space carga (`predict.js`, y
            # `model.onnx`/`inference_spec.json` de la raíz del paquete —
            # revisión 2026-09-16: ya no es `predict.py`, la página no
            # ejecuta Python). (Refutación 2026-08-20.)
            from matrixai.export.reproduce import añadir_inventario_de_ficheros

            # Y EL RESULTADO SE QUEDA CON EL MANIFIESTO DE VERDAD, el que
            # va dentro del paquete. Devolver el de antes del inventario
            # dejaba al llamante con una copia que ya no coincide con el
            # fichero —otro `manifest_sha256`, sin `files`—, o sea dos
            # manifiestos distintos para el mismo paquete. Lo cazó
            # `test_the_full_package_is_declared_reproducible`, que
            # compara lo devuelto contra el disco, y tenía razón.
            reproduce_manifest = añadir_inventario_de_ficheros(work)
            shutil.copytree(str(work), str(outdir))

        # RECURSIVO, no solo el primer nivel. Con `iterdir()` los ficheros
        # de `space/` viajaban en el ZIP y la lista NO los declaraba: un
        # paquete que lleva dentro cosas que su propio manifiesto no
        # nombra es justo la omisión que este contrato combate. Lo cazó
        # `test_bundle_no_entrega_bytecode`, que compara disco contra
        # declaración — y tenía razón.
        #
        # Las rutas van con `/` siempre: en Windows saldrían `space\\app.py`
        # y no casarían con lo que declara el manifiesto ni con lo que
        # busca quien abra el paquete.
        files = sorted(
            p.relative_to(outdir).as_posix()
            for p in outdir.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
        if using_mxw_streaming:
            files = sorted(set(files) | {"model.onnx.data"})

        return EdgeBundleResult(
            bundle_dir=str(outdir),
            files=files,
            # `export_result` (no `parameter_set`) es la fuente de verdad: para
            # el camino state_dict, `parameter_set_id` es "torch_state" ahí,
            # mientras que la plantilla pasada como `parameter_set` podría
            # llevar otro id de plantilla — evita la discrepancia.
            model_hash=export_result.model_hash,
            parameter_set_id=export_result.parameter_set_id,
            export_result=export_result,
            equivalence_result=eq_result,
            inference_spec_skipped_reason=spec_skipped_reason,
            equivalence_skipped_reason=eq_skipped_reason,
            external_data_layout=external_data_layout,
            reproduce=reproduce_manifest,
        )


def create_edge_bundle(
    program: MatrixAIProgram,
    parameter_set: ParameterSet | None,
    mxai_path: str | Path,
    params_path: str | Path | None,
    outdir: str | Path,
    *,
    state_dict: dict[str, Any] | None = None,
    mxw_path: str | Path | None = None,
    mxw_header: dict[str, Any] | None = None,
    materialize_external_data: bool = False,
    model_hash: str | None = None,
    parameter_schema_hash: str | None = None,
    validate: bool = True,
    force: bool = False,
    field_ranges: dict[str, tuple[float, float]] | None = None,
    field_categories: dict[str, list[str]] | None = None,
    field_types: dict[str, str] | None = None,
    field_seq: dict[str, dict[str, Any]] | None = None,
    labels: list[str] | None = None,
    example_input: dict[str, Any] | None = None,
    target_range: tuple[float, float] | None = None,
    data_recipe: str | None = None,
    mxtrain_path: str | Path | None = None,
    dataset_sha256: str | None = None,
    dataset_rows: int | None = None,
    generation: dict[str, Any] | None = None,
    metrics: list[dict[str, Any]] | None = None,
    run_provenance: dict[str, Any] | None = None,
    weights_source: str | None = None,
    problem: Any | None = None,
    clinical_profile: Any | None = None,
    team_declaration: dict[str, Any] | None = None,
) -> EdgeBundleResult:
    return EdgeBundler().bundle(
        program, parameter_set, mxai_path, params_path, outdir,
        state_dict=state_dict, mxw_path=mxw_path, mxw_header=mxw_header,
        materialize_external_data=materialize_external_data,
        model_hash=model_hash, parameter_schema_hash=parameter_schema_hash,
        validate=validate, force=force,
        field_ranges=field_ranges,
        field_categories=field_categories,
        field_types=field_types,
        field_seq=field_seq,
        labels=labels,
        example_input=example_input,
        target_range=target_range,
        data_recipe=data_recipe,
        mxtrain_path=mxtrain_path,
        dataset_sha256=dataset_sha256,
        dataset_rows=dataset_rows,
        generation=generation,
        metrics=metrics,
        run_provenance=run_provenance,
        weights_source=weights_source,
        problem=problem,
        clinical_profile=clinical_profile,
        team_declaration=team_declaration,
    )


# ---------------------------------------------------------------------------
# Prediction artifacts (C2)
# ---------------------------------------------------------------------------

def _run_prediction(bundle_work: Path, record: dict[str, Any]) -> Any | None:
    """Run the bundled predict.py on the example to produce expected_output.json.

    Best-effort smoke test executed at packaging time: it exercises the very code
    the consumer will run. Returns None if onnxruntime is unavailable.
    """
    if not ort_available():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_matrixai_bundled_predict", str(bundle_work / "predict.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MatrixAIModel(str(bundle_work / "inference_spec.json"))
    return model.predict(record)


# ---------------------------------------------------------------------------
# El expediente clínico (contrato 109-C3): qué se escribe, y qué no se fabrica
# ---------------------------------------------------------------------------

def build_clinical_profile_block(clinical_profile: Any) -> dict[str, Any] | None:
    """El perfil clínico de 109-C2 que viaja en el paquete, o `None` si no hay.

    Acepta un `PerfilClinico` (o cualquier objeto con `.a_json()`) o el mapa
    que ya produce esa llamada — el mismo doble camino que `build_problem_
    block` para el problema confirmado. Ausente es `None` y se ve: un
    paquete sin perfil clínico no es un paquete con un perfil vacío, que es
    justo lo que `ExpedienteClinico` necesita para no dar un hueco por
    relleno (`hay_perfil` mira si el fichero está, no si está vacío).

    Fail-closed, como el resto de este fichero: NO reimplementa las
    comprobaciones de `PerfilClinico.__post_init__` —eso dejaría dos sitios
    validando lo mismo, la fuente de fallos que más se repite en este
    repositorio—, solo confirma que lo que llega es de verdad el SOBRE de un
    `PerfilClinico` (su `schema`). Un objeto de otra forma revienta aquí, no
    se empaqueta a medias ni con el esquema equivocado.
    """
    if clinical_profile is None:
        return None
    cuerpo = clinical_profile.a_json() if hasattr(clinical_profile, "a_json") \
        else clinical_profile
    if not isinstance(cuerpo, dict) or not cuerpo:
        raise EdgeBundleError(
            "clinical_profile must be a PerfilClinico (matrixai.estudio."
            f"perfil_clinico) or its a_json() mapping; got {cuerpo!r}")
    from matrixai.estudio.perfil_clinico import PerfilClinico  # noqa: PLC0415
    esquema = cuerpo.get("schema")
    if esquema != PerfilClinico.ESQUEMA:
        raise EdgeBundleError(
            f"clinical_profile carries schema {esquema!r}, expected "
            f"{PerfilClinico.ESQUEMA!r}: packaging it under the wrong "
            "schema is exactly what schema_version exists to prevent")
    return cuerpo


def build_team_declaration_block(team_declaration: Any) -> dict[str, Any] | None:
    """Lo que el equipo declaró sobre el uso clínico, o `None` si nadie lo hizo.

    Población, uso previsto, criterios de inclusión, definición y momento
    del desenlace, proceso actual: el núcleo NO PUEDE inventarlos (109-C3,
    invariante 5). Esta función no juzga su contenido campo a campo —quién
    decide qué cuenta como declarado y qué falta es `ExpedienteClinico`, y
    duplicar ese criterio aquí sería el mismo hueco de «dos sitios
    declarando lo mismo»—; si el equipo lo dio, se empaqueta tal cual.

    Un mapa vacío (`{}`, `None`) es «nadie declaró nada» y no se escribe:
    un `team_declaration.json` vacío es peor que su ausencia, porque
    `ExpedienteClinico.tiene()` lo daría por presente.
    """
    if not team_declaration:
        return None
    if not isinstance(team_declaration, dict):
        raise EdgeBundleError(
            "team_declaration must be a mapping, got "
            f"{type(team_declaration).__name__}")
    return dict(team_declaration)


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def _build_model_manifest(program: MatrixAIProgram, parameter_set: ParameterSet) -> dict:
    from matrixai.compiler import BackendContractAnalyzer
    from matrixai.parameters.network_params import transformer_block_export_metadata
    report = BackendContractAnalyzer().analyze(program)
    inputs: list[dict] = []
    for v in program.vectors:
        inputs.append({"kind": "vector", "name": v.name, "size": v.size,
                       "dtype": "float32", "fields": list(v.fields)})
    for s in program.sequences:
        inputs.append({"kind": "sequence", "name": s.name, "length": s.length,
                       "vocab_size": s.vocab_size, "dtype": "int64"})
    manifest: dict[str, Any] = {
        "project": program.project,
        "matrixai_version": _matrixai_version(),
        "model_hash": parameter_set.model_hash,
        "parameter_schema_hash": parameter_set.parameter_schema_hash,
        "parameter_set_id": parameter_set.parameter_set_id,
        "inputs": inputs,
        "vectors": [
            {"name": v.name, "size": v.size, "fields": list(v.fields)}
            for v in program.vectors
        ],
        "sequences": [
            {"name": s.name, "length": s.length, "vocab_size": s.vocab_size}
            for s in program.sequences
        ],
        "functions": [
            {"name": f.name, "kind": f.semantic.kind}
            for f in program.functions
        ],
        "backend_contract": {
            "target": report.target,
            "ok": report.ok,
            # BackendNodeReport is a dataclass-like domain object, not JSON
            # serializable.  Transformer programs deliberately keep one
            # unsupported program-level forward node even though export is
            # supported, so C5 is the first bundle path that reliably reaches
            # this non-empty branch.
            "unsupported_nodes": [node.to_dict() for node in report.unsupported_nodes],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    transformer_entry = next(
        (e for e in report.layer_manifest if e.get("layer_type") == "TransformerBlock"),
        None,
    )
    if transformer_entry is not None and "layers" in transformer_entry:
        manifest["transformer_block"] = transformer_block_export_metadata(transformer_entry)
    return manifest


def _write_export_manifest_no_eq(export_result: OnnxExportResult, path: Path) -> None:
    data = {
        "model_hash": export_result.model_hash,
        "parameter_schema_hash": export_result.parameter_schema_hash,
        "parameter_set_id": export_result.parameter_set_id,
        "format": "onnx",
        "format_version": export_result.opset_version,
        "input_name": export_result.input_name,
        "input_shape": export_result.input_shape,
        "output_name": export_result.output_name,
        "output_shape": export_result.output_shape,
        "exported_function": export_result.exported_functions[0] if export_result.exported_functions else None,
        "tolerance": None,
        "equivalence_check": None,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def readme_reproducing_section(reproduce: dict[str, Any] | None) -> str:
    """La sección «Reproducing this» del README (contrato 82-C3).

    Criterio de cierre del contrato: **quien llega al repositorio sabe en
    un minuto qué ejecutar y qué debería salir**. Por eso lleva las dos
    cosas —el comando Y el resultado esperado—: un comando sin resultado
    esperado deja a quien lo corre sin saber si lo que ve está bien.

    Devuelve cadena VACÍA si el paquete no trae manifiesto: un ONNX suelto
    no tiene nada que reproducir, y escribir una sección hueca parecería
    un dato que falta en vez de algo que no aplica.

    Y **no promete lo que el paquete no puede cumplir**: si el manifiesto
    dice que no es reproducible, la sección lo dice PRIMERO, con el motivo
    que redactó el core, en vez de invitar a ejecutar algo que no va a
    salir. Mandar a alguien a perder una tarde es peor que no decir nada.
    """
    if not isinstance(reproduce, dict) or not reproduce:
        return ""

    artefactos = reproduce.get("artifacts") or {}
    generacion = reproduce.get("generation") or {}
    entorno = reproduce.get("environment") or {}

    def _fila(etiqueta: str, clave: str) -> str | None:
        art = artefactos.get(clave)
        if not isinstance(art, dict) or not art.get("path"):
            return None
        huella = str(art.get("sha256") or "")
        return f"| {etiqueta} | `{art['path']}` | `{huella[:16]}…` |"

    lineas = ["## Reproducing this", ""]

    if not reproduce.get("reproducible"):
        motivo = reproduce.get("reproducible_reason") or "This package is not reproducible."
        lineas += [
            f"**This package is not reproducible.** {motivo}",
            "",
            "You can still download it, inspect it and run it — what you "
            "cannot do is rebuild its data and check that the same numbers "
            "come out.",
            "",
        ]
    else:
        lineas += [
            "Everything needed to rebuild this model's data and check it "
            "travels inside the package:",
            "",
        ]

    filas = [f for f in (_fila("model", "model"),
                         _fila("training contract", "training"),
                         _fila("data recipe", "recipe")) if f]
    if filas:
        lineas += ["| What | File | sha256 |", "|---|---|---|", *filas, ""]

    dataset = artefactos.get("dataset")
    if isinstance(dataset, dict) and dataset.get("sha256"):
        filas_ds = f"- **dataset**: {dataset.get('rows')} rows, sha256 `{str(dataset['sha256'])[:16]}…`"
        lineas.append(filas_ds)
    semillas = generacion.get("seeds") or {}
    if semillas.get("dataset") is not None:
        lineas.append(f"- **generation seed**: `{semillas['dataset']}`")
    if generacion.get("mode"):
        lineas.append(f"- **generation mode**: `{generacion['mode']}`")
    if entorno:
        # El entorno, porque sin él una diferencia de versión parece una
        # manipulación en vez de lo que es.
        piezas = ", ".join(f"{k} {v}" for k, v in sorted(entorno.items()) if v)
        if piezas:
            lineas.append(f"- **trained with**: {piezas}")
    lineas.append("")

    if reproduce.get("reproducible"):
        lineas += [
            "```bash",
            "matrixai verify .              # integrity + rebuild the dataset",
            "matrixai verify . --retrain    # …and train again (slow)",
            "matrixai verify . --json       # same report, machine readable",
            "```",
            "",
            "**What you should see:** `manifest PASS` and `R1 PASS`. With "
            "`--retrain`, also `training PASS`. `R3` reports `INCOMPARABLE` "
            "unless the package publishes metrics with their tolerance — that "
            "is a missing datum, not a failure.",
            "",
            "Exit codes: `0` nothing failed · `2` something does not match · "
            "`3` it could not be checked.",
            "",
        ]

    return "\n".join(lineas)


def _readme_quickstart(inference_spec: dict[str, Any], example_input: dict[str, Any] | None,
                       smoke_test_skipped: bool = False) -> str:
    out = inference_spec.get("output", {})
    kind = out.get("kind", "")
    labels = out.get("labels") or []
    if kind == "classification" or kind == "binary_classification":
        out_desc = f"a probability per class ({', '.join(labels)})"
    elif kind == "regression":
        out_desc = "a single numeric value"
    else:
        out_desc = "a raw output vector"
    example_json = json.dumps(example_input or {}, ensure_ascii=False)
    # PESOS_GRANDES C7 auditoría: para un modelo grande no se generó
    # `expected_output.json` (el smoke-test se saltó) — el README no promete
    # reproducirlo; en su lugar explica cómo generarlo.
    reproduce = (
        "That produces the prediction; there is no bundled `expected_output.json` for "
        "this (large) model — running the command above once writes your own baseline."
        if smoke_test_skipped else
        "That should reproduce `expected_output.json`."
    )
    return f"""
## Quick start

This model is self-usable: feed **raw, human-readable values** and get back {out_desc}.
Normalization and category encoding are handled for you by `predict.py`.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python predict.py --input example_input.json
```

{reproduce} From your own code:

```python
from predict import MatrixAIModel

model = MatrixAIModel()                 # loads inference_spec.json next to predict.py
print(model.predict({example_json}))
```
"""


def _build_readme(
    program: MatrixAIProgram,
    export_result: OnnxExportResult,
    eq_result: OnnxEquivalenceResult | None,
    *,
    inference_spec: dict[str, Any] | None = None,
    example_input: dict[str, Any] | None = None,
    has_params_json: bool = True,
    external_data: bool = False,
    smoke_test_skipped: bool = False,
    has_mxtrain: bool = False,
    has_recipe: bool = False,
    reproduce: dict[str, Any] | None = None,
) -> str:
    project = program.project
    out_name = export_result.output_name
    out_shape = export_result.output_shape

    is_sequence = bool(program.sequences)
    if is_sequence:
        seq = program.sequences[0]
        input_name = seq.name
        mask_name = f"{seq.name}_mask"
        input_desc = (
            f"`{seq.name}` shape `{export_result.input_shape}` (int64 token IDs) + "
            f"`{mask_name}` with the same shape (float32; 1 real / 0 padding)"
        )
        inference_snippet = (
            f"sess = ort.InferenceSession(\"model.onnx\")\n"
            f"ids = np.array([[...]], dtype=np.int64)  # shape [batch, {seq.length}]\n"
            f"mask = np.ones_like(ids, dtype=np.float32)\n"
            f"result = sess.run(None, {{\"{seq.name}\": ids, \"{mask_name}\": mask}})[0]  # {out_shape}"
        )
    else:
        vec = program.vectors[0] if program.vectors else None
        input_name = vec.name if vec else "input"
        vec_size = vec.size if vec else "?"
        input_desc = f"`{input_name}` shape `{export_result.input_shape}` (float32)"
        inference_snippet = (
            f"sess = ort.InferenceSession(\"model.onnx\")\n"
            f"x = np.array([[...]], dtype=np.float32)  # shape [batch, {vec_size}]\n"
            f"result = sess.run(None, {{\"{input_name}\": x}})[0]  # {out_shape}"
        )

    eq_line = ""
    if eq_result is not None:
        status = "PASS" if eq_result.passed else "FAIL"
        eq_line = (
            f"\nEquivalence check: {status} "
            f"(max_abs_diff={eq_result.max_abs_diff:.2e}, "
            f"atol={eq_result.atol:.0e}, n={eq_result.n_samples})\n"
        )

    skipped = ""
    if export_result.skipped_functions:
        skipped = (
            f"\nNote: the following functions were not exported (unsupported kind): "
            f"{', '.join(export_result.skipped_functions)}\n"
        )

    quickstart = ""
    usable_files = ""
    # 82-C3 · «Reproducing this» se compone AQUÍ, FUERA del condicional, y
    # se inyecta en el README. Dos motivos:
    #  1. Escribir la función y no llamarla es el hueco de cableado que
    #     este proyecto lleva repitiendo dieciséis veces.
    #  2. Dentro del `if` quedaba SIN DEFINIR en el otro camino y el README
    #     reventaba con `UnboundLocalError` — lo cazó la prueba del
    #     paquete sin `inference_spec`, que es justo el caso que un ONNX
    #     suelto produce.
    _seccion = readme_reproducing_section(reproduce)
    _reproducing = f"{_seccion}\n" if _seccion else ""
    if inference_spec is not None:
        quickstart = _readme_quickstart(inference_spec, example_input, smoke_test_skipped)
        usable_files = (
            "| `inference_spec.json` | How a raw record maps to the model input (the \"tokenizer\") |\n"
            "| `predict.py` | Standalone wrapper: raw values in, labelled prediction out |\n"
            "| `requirements.txt` | Minimal deps to run predict.py (numpy + onnxruntime) |\n"
            "| `example_input.json` | A ready-to-run raw example |\n"
        )
        # PESOS_GRANDES C7 auditoría (BAJA): el README lista los ficheros
        # REALES — `expected_output.json` solo si el smoke-test corrió (para un
        # grande se salta, ver docstring del bundle).
        if not smoke_test_skipped:
            usable_files += (
                "| `expected_output.json` | The output predict.py should produce for that example |\n"
            )

    # BAJA C7: `params.best.json` solo se lista si de verdad va en el zip
    # (se omite para un modelo grande — el ONNX lleva los pesos); con
    # external-data, además hay un `model.onnx.data` que hay que listar.
    # DE DÓNDE SALEN LOS PESOS, en la tabla y arriba del todo.
    #
    # La fila decía «Trained parameter weights» pasara lo que pasara, y un
    # paquete exportado antes de terminar de entrenar —o después de borrar los
    # pesos— la llevaba igual: un dibujo afirma por omisión, y una tabla
    # también. Lo que se escribe aquí sale del manifiesto (`weights.source`),
    # que es quien lo sabe; no se vuelve a deducir, porque dos sitios
    # declarando lo mismo acaban divergiendo.
    pesos = ((reproduce or {}).get("weights") or {}).get("source")
    if not has_params_json:
        params_row = ""
    elif reproduce is None or pesos == "trained":
        params_row = "| `params.best.json` | Trained parameter weights |\n"
    elif pesos == "untrained":
        params_row = ("| `params.best.json` | Parameter weights — **random "
                      "initialisation, NOT trained** |\n")
    else:
        params_row = ("| `params.best.json` | Parameter weights — this package does "
                      "not state whether they were trained |\n")

    # Y el aviso va ANTES del «quick start», que empieza diciendo «This model is
    # self-usable»: quien descarga esto lee el README antes que el JSON, y
    # enterarse después de haber ejecutado la predicción no sirve de nada.
    aviso_pesos = ""
    if pesos == "untrained":
        aviso_pesos = (
            "\n> **WARNING — the weights in this package are random initialisation, "
            "not the result of a training run.** Anything it predicts comes from an "
            "untrained model: nothing here was learned. See `reproduce.json`.\n"
        )
    elif reproduce is not None and pesos is None:
        aviso_pesos = (
            "\n> **This package does not state whether its weights come from a "
            "training run**, and \"not stated\" is not \"trained\". See "
            "`reproduce.json`.\n"
        )
    onnx_data_row = (
        "| `model.onnx.data` | ONNX external weights (loaded automatically next to model.onnx) |\n"
        if external_data else ""
    )

    # Contrato 82-C1: la tabla sigue listando los ficheros REALES (misma
    # regla que `params.best.json`/`model.onnx.data` arriba). `reproduce.json`
    # va siempre; el `.mxtrain` y la receta, solo si de verdad viajan.
    mxtrain_row = (
        "| `model.mxtrain` | Training contract: dataset, split, loss, optimizer, epochs |\n"
        if has_mxtrain else ""
    )
    recipe_row = (
        "| `data_recipe.txt` | The recipe the training data was generated from |\n"
        if has_recipe else ""
    )
    reproduce_row = (
        # La fila describe el fichero, no lo que el paquete consigue: en un
        # paquete sin receta `reproduce.json` NO lleva «lo que hace falta para
        # rehacer el modelo», lleva lo que falta. Un dibujo afirma por
        # omisión, y una tabla también.
        "| `reproduce.json` | Whether this model can be rebuilt, and the digest of each "
        "artifact that travels |\n"
    )
    # Y el estado se DECLARA, no se deduce del README: un paquete sin receta
    # dice que no se puede reproducir y por qué, en vez de callar (§6.2).
    reproduce_note = ""
    if reproduce is not None:
        if reproduce.get("reproducible"):
            # Y con el mismo criterio, al revés: «reproducible» NO quiere decir
            # que todo se pueda comprobar. Si falta lo que solo hace falta para
            # R3 —la semilla de inicialización, el motor, o una métrica con su
            # tolerancia— se dice aquí. Un aviso a medias que tranquiliza es
            # peor que callar, y quien descarga esto lo lee antes que el JSON.
            r3 = (reproduce.get("verifiable") or {}).get("r3") or {}
            aviso_r3 = ""
            if r3 and not r3.get("possible", True):
                aviso_r3 = f"\n> {r3.get('reason', '')} See `reproduce.json`.\n"
            # Y se dice DE DÓNDE sale lo que afirma. «Reproducible» sin eso
            # se leía como «alguien escribió estos valores en la pantalla de
            # exportar y salieron bien»: lo que lo convierte en una prueba es
            # que cada artefacto casa con la captura que el core guardó
            # mientras entrenaba, no que los campos estén rellenos.
            reproduce_note = (
                "\nThis package is **reproducible**: `reproduce.json` carries the recipe, "
                "the training contract, the expected dataset sha256 and the exact "
                "environment, all of it taken from the capture the core recorded "
                "while training this model — and every artifact here matches it. "
                "It proves internal consistency, not authorship.\n"
                f"{aviso_r3}"
            )
        else:
            # El motivo se imprime TAL CUAL lo escribió `reproduce.json` (ya es
            # una frase entera): componer aquí una segunda versión sería el
            # segundo sitio redactando lo mismo, y se leía repetido
            # («not reproducible. Not reproducible: …»).
            reproduce_note = (
                f"\n> {reproduce.get('reproducible_reason', '')} "
                f"See `reproduce.json`.\n"
            )

    return f"""# {project} Edge Bundle

MatrixAI model exported for edge/production inference.
Actions remain `simulate_only`. This bundle only provides predictions.
{aviso_pesos}{quickstart}
## Files

| File | Description |
|------|-------------|
| `model.mxai` | MatrixAI model definition (source of truth) |
{mxtrain_row}{params_row}| `model.onnx` | ONNX model, opset {export_result.opset_version} |
{onnx_data_row}| `model_manifest.json` | Model metadata, hashes and backend contract |
| `export_manifest.json` | Export metadata, tolerance and equivalence check |
{recipe_row}{reproduce_row}{usable_files}| `space/` | A static Hugging Face Space template (runs the model in the browser with ONNX Runtime Web) — yours to publish, or to ignore |
| `README.md` | This file |
{reproduce_note}
## Model info

- Project: `{project}`
- Model hash: `{export_result.model_hash}`
- Parameter set: `{export_result.parameter_set_id}`
- Input: {input_desc}
- Output: `{out_name}` shape `{out_shape}`
{eq_line}{skipped}
## Advanced: raw onnxruntime access

For most uses prefer `predict.py` above (it handles normalization and labels).
The raw ONNX graph expects an already-normalized float32 vector:

```python
import onnxruntime as ort
import numpy as np

{inference_snippet}
```

{_reproducing}## Verifying integrity

```python
import json
with open("model_manifest.json") as f:
    manifest = json.load(f)
assert manifest["model_hash"] == "{export_result.model_hash}"
assert manifest["parameter_schema_hash"] == "{export_result.parameter_schema_hash}"
```
"""
