# MatrixAI — Test Index

3618 tests across 140 files, organised by theme.

```bash
python -m pytest tests/                  # run all
python -m pytest tests/test_<file>.py    # run one section
```

## Contents

1. [Mathematical Core (.mx language)](#mathematical-core-mx-language) — 100 tests
2. [Language, Parser and IR (.mxai)](#language-parser-and-ir-mxai) — 122 tests
3. [Agents and Pipeline](#agents-and-pipeline) — 183 tests
4. [Training and Parameters](#training-and-parameters) — 59 tests
5. [Synthetic Data (P12)](#synthetic-data-p12) — 54 tests
6. [Serving and HTTP Server](#serving-and-http-server) — 12 tests
7. [PyTorch Backend](#pytorch-backend) — 11 tests
9. [Language Expressivity and Typed Layers (P10)](#language-expressivity-and-typed-layers-p10) — 178 tests
10. [Transformer Architecture (P11)](#transformer-architecture-p11) — 292 tests
11. [Regression (P17)](#regression-p17) — 158 tests
12. [Dense Neural Networks (P18)](#dense-neural-networks-p18) — 326 tests
13. [Composite Networks and Embeddings (P19)](#composite-networks-and-embeddings-p19) — 313 tests
14. [GPU Acceleration (P14)](#gpu-acceleration-p14) — 118 tests
15. [ONNX, WASM and Edge Export (P15)](#onnx-wasm-and-edge-export-p15) — 315 tests
16. [Real Actions and Action Contracts (P20)](#real-actions-and-action-contracts-p20) — 281 tests
17. [Model Registry (P21)](#model-registry-p21) — 227 tests
18. [Continual Learning (P22)](#continual-learning-p22) — 419 tests
19. [Deployment Suite (PR4)](#deployment-suite-pr4) — 149 tests

---

## Mathematical Core (.mx language) (100 tests)

| Test | Description |
|------|-------------|
| **[test_core.py](../tests/test_core.py)** | Lexer, parser, AST nodes, evaluator, trace, computation graph, math functions library (100 tests) |
| &nbsp;&nbsp;**TestLexer** | *TestLexer* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_number_integer` | number integer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_number_float` | number float |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_number_scientific` | number scientific |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ident` | ident |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dotted_ident` | dotted ident |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_operators` | operators |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parens_comma_equals` | parens comma equals |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_comment_ignored` | comment ignored |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eof_present` | eof present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unexpected_char_raises` | unexpected char raises |
| &nbsp;&nbsp;**TestParser** | *TestParser* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_literal` | literal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_variable` | variable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assignment_no_params` | assignment no params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assignment_with_params` | assignment with params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assignment_two_params` | assignment two params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binop_add` | binop add |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_precedence_mul_before_add` | precedence mul before add |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parentheses_override_precedence` | parentheses override precedence |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unary_minus` | unary minus |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_function_call_no_args` | function call no args |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_function_call_one_arg` | function call one arg |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_function_call_nested` | function call nested |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_statements` | multiple statements |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bad_syntax_raises` | bad syntax raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict` | to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_str` | str |
| &nbsp;&nbsp;**TestAstNodes** | *TestAstNodes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_number_node_eval` | number node eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_var_node_eval` | var node eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_var_node_dotted` | var node dotted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_var_node_missing_raises` | var node missing raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binop_add` | binop add |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binop_mul` | binop mul |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binop_div_by_zero` | binop div by zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_number_to_dict` | number to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_var_to_dict` | var to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binop_to_dict` | binop to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assign_to_dict` | assign to dict |
| &nbsp;&nbsp;**TestEvaluator** | *TestEvaluator* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_literal` | literal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_variable_from_env` | variable from env |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binop` | binop |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_precedence_in_eval` | precedence in eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_function` | registry function |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_user_defined_function_call` | user defined function call |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_score_example` | score(x) = 0.6 * relevance(x) + 0.4 * coherence(x) |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_utility_example` | utility(x) = score(x) - 0.2 * cost(x)  → depends on score |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_call_method` | call method |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_undefined_raises` | undefined raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_registry_fn_raises` | missing registry fn raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_nested_calls` | nested calls |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unary_minus_eval` | unary minus eval |
| &nbsp;&nbsp;**TestTrace** | *TestTrace* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_recorded` | output recorded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_steps_nonempty` | steps nonempty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_contains_function_calls` | trace contains function calls |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_to_dict` | trace to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_node_name` | trace node name |
| &nbsp;&nbsp;**TestGraph** | *TestGraph* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_nodes_and_edges` | has nodes and edges |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_input_node` | has input node |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_output_node` | has output node |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_call_nodes` | has call nodes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_edges_reference_valid_nodes` | edges reference valid nodes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_params_graph` | no params graph |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_to_text` | graph to text |
| &nbsp;&nbsp;**TestMathOps** | *TestMathOps* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_add` | add |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sub` | sub |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mul` | mul |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_div` | div |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_div_by_zero` | div by zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pow` | pow |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sqrt` | sqrt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_abs_negative` | abs negative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_min_varargs` | min varargs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_varargs` | max varargs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mean` | mean |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sum` | sum |
| &nbsp;&nbsp;**TestTransforms** | *TestTransforms* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_normalize_clips_high` | normalize clips high |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_normalize_clips_low` | normalize clips low |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_normalize_passthrough` | normalize passthrough |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_clip` | clip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_scale` | scale |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_scale_zero_span` | scale zero span |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_sums_to_one` | softmax sums to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_monotone` | softmax monotone |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_at_zero` | sigmoid at zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_large_positive` | sigmoid large positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_large_negative` | sigmoid large negative |
| &nbsp;&nbsp;**TestScoring** | *TestScoring* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relevance_from_dict` | relevance from dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherence_from_dict` | coherence from dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_confidence_missing_returns_zero` | confidence missing returns zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cost_from_dict` | cost from dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relevance_from_scalar` | relevance from scalar |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_argmax` | argmax |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_topk` | topk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rank` | rank |
| &nbsp;&nbsp;**TestEndToEnd** | *TestEndToEnd* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_score_and_utility` | Full pipeline: parse p1_demo.mx → eval → expected values. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_score` | trace score |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_score` | graph score |
| &nbsp;&nbsp;**TestCliEval** | *TestCliEval* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_basic` | eval basic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_json` | eval json |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_trace` | eval trace |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_specific_call` | eval specific call |

## Language, Parser and IR (.mxai) (122 tests)

| Test | Description |
|------|-------------|
| **[test_mvp.py](../tests/test_mvp.py)** | End-to-end MVP pipeline: prompt → .semantic → .mxai → IR → runtime → auditor (80 tests) |
| &nbsp;&nbsp;**MatrixAIMVPTest** | *MatrixAIMVPTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_planner_verifier_accepts_valid_plan` | planner verifier accepts valid plan |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_planner_verifier_rejects_unsafe_action` | planner verifier rejects unsafe action |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_architect_generates_valid_email_agent` | architect generates valid email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_agent_generates_supervised_email_agent` | prompt agent generates supervised email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_agent_extracts_fields_rules_and_lineage` | prompt agent extracts fields rules and lineage |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_rules_are_executable_action_evidence` | prompt rules are executable action evidence |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_supervisor_accepts_prompt_pipeline` | prompt supervisor accepts prompt pipeline |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_supervisor_rejects_unsafe_semantic_proposal` | prompt supervisor rejects unsafe semantic proposal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_proposal_agent_accepts_deterministic_provider` | llm proposal agent accepts deterministic provider |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_proposal_agent_tries_next_candidate_after_rejection` | llm proposal agent tries next candidate after rejection |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_proposal_agent_rejects_when_no_candidate_passes` | llm proposal agent rejects when no candidate passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_generates_supervised_candidate` | chat completions provider generates supervised candidate |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_from_env_requires_api_key` | chat completions provider from env requires api key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_rejects_dotenv_placeholder` | chat completions provider rejects dotenv placeholder |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_loads_dotenv_file` | chat completions provider loads dotenv file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_architect_generates_valid_risk_agent` | architect generates valid risk agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_parses_and_validates` | email agent parses and validates |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_runs_simulated_action` | email agent runs simulated action |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_goal_translator_derives_rules_from_goals` | goal translator derives rules from goals |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_planner_verifier_fails_goal_rule_violation` | planner verifier fails goal rule violation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_optimizer_suggests_merge_for_email_agent` | optimizer suggests merge for email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_optimizer_no_suggestions_on_minimal_graph` | optimizer no suggestions on minimal graph |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_python_compiler_generated_email_matches_runtime` | python compiler generated email matches runtime |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_python_compiler_generated_pharmacy_matches_runtime` | python compiler generated pharmacy matches runtime |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_simple_rule` | mathematical agent translates simple rule |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_and_rule` | mathematical agent translates and rule |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_or_rule` | mathematical agent translates or rule |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_classify_rule` | mathematical agent translates classify rule |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_marks_unresolved` | mathematical agent marks unresolved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_negative_threshold_simple` | mathematical agent translates negative threshold simple |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_negative_threshold_and` | mathematical agent translates negative threshold and |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translates_negative_threshold_or` | mathematical agent translates negative threshold or |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_positive_threshold_still_works` | mathematical agent positive threshold still works |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_mixed_batch` | mathematical agent mixed batch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_call_trace_recorded_on_success` | llm call trace recorded on success |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_call_trace_prompt_hash_is_stable` | llm call trace prompt hash is stable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_call_trace_token_usage_captured` | llm call trace token usage captured |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_call_trace_recorded_on_http_error` | llm call trace recorded on http error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_retries_on_429` | chat completions provider retries on 429 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_retries_on_5xx` | chat completions provider retries on 5xx |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_no_retry_on_400` | chat completions provider no retry on 400 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_retries_exhausted` | chat completions provider retries exhausted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_no_choices_raises` | chat completions provider no choices raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_candidate_without_project_skipped` | chat completions provider candidate without project skipped |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chat_completions_provider_budget_exceeded` | chat completions provider budget exceeded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_propose_and_supervise_exhaust_all_evaluates_all_candidates` | propose and supervise exhaust all evaluates all candidates |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_proposal_batch_call_traces_in_json` | llm proposal batch call traces in json |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_literal_eval` | expr literal eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_var_eval` | expr var eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_var_dotted_eval` | expr var dotted eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_binop_add_mul` | expr binop add mul |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_call_normalize` | expr call normalize |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_call_sigmoid` | expr call sigmoid |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_call_clip_and_scale` | expr call clip and scale |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_expr_weighted_sum_structure` | parse expr weighted sum structure |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_extract_weighted_sum_two_terms` | extract weighted sum two terms |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_extract_weighted_sum_eval` | extract weighted sum eval |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_collect_vars` | collect vars |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_to_dict_round_trip` | expr to dict round trip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expr_division_by_zero_raises` | expr division by zero raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_expr_unknown_function_resolves_from_env` | parse expr unknown function resolves from env |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_build_symbolic_weighted_sum` | mathematical agent build symbolic weighted sum |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_build_symbolic_expr` | mathematical agent build symbolic expr |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_build_symbolic_invalid` | mathematical agent build symbolic invalid |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_assign` | mathematical agent translate assign |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_aggregate_max` | mathematical agent translate aggregate max |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_aggregate_mean` | mathematical agent translate aggregate mean |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_aggregate_softmax` | mathematical agent translate aggregate softmax |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_aggregate_vote` | mathematical agent translate aggregate vote |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_normalize` | mathematical agent translate normalize |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_normalize_with_range` | mathematical agent translate normalize with range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mathematical_agent_translate_select` | mathematical agent translate select |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_evaluates_aggregate_max` | runtime evaluates aggregate max |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_evaluates_aggregate_mean` | runtime evaluates aggregate mean |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_evaluates_normalize` | runtime evaluates normalize |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_evaluates_symbolic_weighted_sum` | runtime evaluates symbolic weighted sum |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_vector_fields_exposed_in_state` | runtime vector fields exposed in state |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p1_mxai_symbolic_normalize_aggregate_runtime_matches_compiled` | p1 mxai symbolic normalize aggregate runtime matches compiled |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p1_mxai_select_argmax_runtime_matches_compiled` | p1 mxai select argmax runtime matches compiled |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unknown_function_semantics_fail_explicitly` | unknown function semantics fail explicitly |
| **[test_backend_contract.py](../tests/test_backend_contract.py)** | BackendContractAnalyzer: portability checks, parameter manifest, autodiff plan (21 tests) |
| &nbsp;&nbsp;**MatrixAIBackendContractTest** | *MatrixAIBackendContractTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_typed_email_example_fits_differentiable_backend_contract` | typed email example fits differentiable backend contract |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_linear_manifest_uses_scalar_output_shape` | sigmoid linear manifest uses scalar output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_report_accepts_email_classifier` | torch backend report accepts email classifier |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_report_accepts_fall_risk_binary_classifier` | torch backend report accepts fall risk binary classifier |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_report_blocks_deferred_symbolic_function` | torch backend report blocks deferred symbolic function |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_discrete_argmax_blocks_differentiable_backend_contract` | discrete argmax blocks differentiable backend contract |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_report_blocks_discrete_argmax` | torch backend report blocks discrete argmax |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_report_json_returns_trainable_manifest` | cli backend report json returns trainable manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_report_torch_json_returns_optional_backend_metadata` | cli backend report torch json returns optional backend metadata |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_parameters_torch_target_outputs_manifest` | cli backend parameters torch target outputs manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_compiler_runs_typed_email_boundaries` | differentiable python compiler runs typed email boundaries |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_runs_continuous_p1_typed_example` | differentiable python runs continuous p1 typed example |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_run_accepts_initial_parameters` | differentiable python run accepts initial parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_rejects_invalid_parameter_shape` | differentiable python rejects invalid parameter shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_reports_ragged_parameter_errors` | differentiable python reports ragged parameter errors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_compiler_rejects_discrete_argmax` | differentiable python compiler rejects discrete argmax |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_compile_differentiable_python_outputs_module` | cli compile differentiable python outputs module |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_run_supports_initial_parameters` | cli backend run supports initial parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_run_rejects_invalid_parameter_shape` | cli backend run rejects invalid parameter shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_parameters_outputs_initial_parameters` | cli backend parameters outputs initial parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_parameters_validates_parameter_file` | cli backend parameters validates parameter file |
| **[test_types.py](../tests/test_types.py)** | Type system: base types, AI-native types, structured types, range validation (11 tests) |
| &nbsp;&nbsp;**MatrixAITypeSystemTest** | *MatrixAITypeSystemTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_base_ai_and_structured_types` | parse base ai and structured types |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mx_parser_accepts_typed_params_and_return` | mx parser accepts typed params and return |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mx_typecheck_rejects_incompatible_return` | mx typecheck rejects incompatible return |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mx_typecheck_accepts_score_pipeline` | mx typecheck accepts score pipeline |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_accepts_typed_fields_and_outputs` | mxai parser accepts typed fields and outputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_program_typecheck_rejects_function_output_mismatch` | program typecheck rejects function output mismatch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_rejects_typed_vector_field_out_of_range` | runtime rejects typed vector field out of range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_compiled_python_rejects_typed_vector_field_out_of_range` | compiled python rejects typed vector field out of range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_typed_domain_examples_typecheck` | typed domain examples typecheck |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_typed_domain_examples_runtime_matches_compiled` | typed domain examples runtime matches compiled |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_typed_domain_example_rejects_out_of_range_field` | typed domain example rejects out of range field |
| **[test_snapshots.py](../tests/test_snapshots.py)** | Snapshot regression: .mxai IR JSON, compiled Python output, trace JSON (3 tests) |
| &nbsp;&nbsp;**MatrixAISnapshotTest** | *MatrixAISnapshotTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p1_mx_trace_json_snapshot` | p1 mx trace json snapshot |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_ir_json_snapshot` | mxai ir json snapshot |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_compiled_python_output_snapshot` | compiled python output snapshot |
| **[test_supervision_snapshots.py](../tests/test_supervision_snapshots.py)** | Supervision snapshot regression: accepted and rejected proposals (3 tests) |
| &nbsp;&nbsp;**MatrixAISupervisionSnapshotTest** | *MatrixAISupervisionSnapshotTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_claim_risk_accepted_snapshot` | prompt claim risk accepted snapshot |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposal_email_accepted_snapshot` | proposal email accepted snapshot |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposal_email_rejected_snapshot` | proposal email rejected snapshot |
| **[test_sandbox.py](../tests/test_sandbox.py)** | Sandbox policy: capability classification, simulate_only enforcement (4 tests) |
| &nbsp;&nbsp;**MatrixAISandboxTest** | *MatrixAISandboxTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capabilities_for_simulated_domain_calls` | capabilities for simulated domain calls |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sandbox_allows_mvp_simulated_action` | sandbox allows mvp simulated action |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sandbox_blocks_external_action` | sandbox blocks external action |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_permissions_json_reports_capabilities` | cli permissions json reports capabilities |

## Agents and Pipeline (183 tests)

| Test | Description |
|------|-------------|
| **[test_tooling.py](../tests/test_tooling.py)** | Language tooling: lint, format, graph, diagnose runtime-vs-compiler (11 tests) |
| &nbsp;&nbsp;**MatrixAILanguageToolingTest** | *MatrixAILanguageToolingTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_format_semantic_canonicalizes_blocks` | format semantic canonicalizes blocks |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_lint_semantic_reports_planner_errors` | lint semantic reports planner errors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_format_mxai_preserves_p2_type_annotations` | format mxai preserves p2 type annotations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_format_mxai_preserves_explicit_params` | format mxai preserves explicit params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_lint_mxai_reports_verifier_errors` | lint mxai reports verifier errors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_lint_json_reports_ok_for_example` | cli lint json reports ok for example |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_format_check_detects_unformatted_file` | cli format check detects unformatted file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_mermaid_renders_nodes_and_edges` | graph mermaid renders nodes and edges |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_json_supports_semantic_source` | graph json supports semantic source |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_compiler_diagnostic_matches_example` | runtime compiler diagnostic matches example |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_diagnose_json_reports_match` | cli diagnose json reports match |
| **[test_p7_llm_bridge.py](../tests/test_p7_llm_bridge.py)** | LLM bridge: chat-completions-compatible provider, supervision source, deterministic fallback (5 tests) |
| &nbsp;&nbsp;**TestP7LLMBridge** | *TestP7LLMBridge* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fake_transport_llm_branch_accepted` | transport fake: supervise_prompt uses the LLM branch when from_env |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_deterministic_branch_when_no_api_key` | rama determinista: supervise_prompt falls back to the deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_safety_agent_blocks_unsafe_llm_proposal` | SafetyAgent bloqueando LLM: a fake transport that returns a semantic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_playground_exposes_llm_mode_field` | playground llm_mode: analyze_playground_request (mode=prompt) returns |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_playground_supervision_source_field` | playground supervision_source: the playground result carries a |
| **[test_p13_refinement_loop.py](../tests/test_p13_refinement_loop.py)** | Refinement agent: audit-driven and metric-driven prompt refinement, iteration limit (82 tests) |
| &nbsp;&nbsp;**TestP13RefinementAgentAuditDriven** | *TestP13RefinementAgentAuditDriven* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refine_returns_proposal` | refine returns proposal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposal_has_refinement_id` | proposal has refinement id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refinement_id_is_deterministic` | refinement id is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refinement_id_differs_for_different_prompts` | refinement id differs for different prompts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposed_prompt_differs_from_original` | proposed prompt differs from original |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposed_prompt_includes_action_name_from_audit` | proposed prompt includes action name from audit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hints_applied_not_empty` | hints applied not empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explanation_references_audit` | explanation references audit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervision_report_is_dict_with_accepted_key` | supervision report is dict with accepted key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervision_accepted_matches_report` | supervision accepted matches report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_is_serializable` | to dict is serializable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_activated_action_detected_in_audit` | activated action detected in audit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_actions_audit_adds_generic_hint` | no actions audit adds generic hint |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_context_is_injected` | mxai context is injected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_count_adds_warning_if_high` | iteration count adds warning if high |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_xml_tags_in_prompt` | xml tags in prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_user_hints_are_appended` | user hints are appended |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_default_mode_is_audit_driven` | default mode is audit driven |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_prompt_raises` | empty prompt raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_whitespace_prompt_raises` | whitespace prompt raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_invalid_mode_raises_value_error` | invalid mode raises value error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_driven_without_audit_raises` | audit driven without audit raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_without_evaluation_raises` | metric driven without evaluation raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_importable_from_agents_package` | importable from agents package |
| &nbsp;&nbsp;**TestP13IterationLimit** | *TestP13IterationLimit* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_default_max_iterations_is_3` | default max iterations is 3 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_at_boundary_does_not_raise` | iteration_count == max_iterations debe ser aceptado. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_exceeds_default_limit_raises` | iteration exceeds default limit raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_error_message_includes_current_and_max` | error message includes current and max |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_custom_limit_blocks_at_threshold` | custom limit blocks at threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_custom_limit_allows_below_threshold` | custom limit allows below threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_limit_applies_to_metric_driven_too` | limit applies to metric driven too |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exception_is_runtime_error_subclass` | exception is runtime error subclass |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_importable_from_agents_package` | importable from agents package |
| &nbsp;&nbsp;**TestP13RefinementAgentMetricDriven** | *TestP13RefinementAgentMetricDriven* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_returns_proposal` | metric driven returns proposal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_refinement_id_format` | metric driven refinement id format |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_id_is_deterministic` | metric driven id is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_proposed_prompt_contains_original` | metric driven proposed prompt contains original |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_low_accuracy_hint_in_hints` | metric driven low accuracy hint in hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_high_loss_hint_in_hints` | metric driven high loss hint in hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_low_f1_label_hint_in_hints` | metric driven low f1 label hint in hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_all_ok_still_returns_proposal` | metric driven all ok still returns proposal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_explanation_references_accuracy` | metric driven explanation references accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_has_system_feedback_tags` | metric driven has system feedback tags |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_supervision_report_present` | metric driven supervision report present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_supervision_accepted_matches_report` | metric driven supervision accepted matches report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_user_hints_appended` | metric driven user hints appended |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_iteration_count_stored` | metric driven iteration count stored |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_high_iteration_adds_warning` | metric driven high iteration adds warning |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_to_dict_serializable` | metric driven to dict serializable |
| &nbsp;&nbsp;**TestP13Traceability** | *TestP13Traceability* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_single_iteration_chain_has_one_element` | single iteration chain has one element |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_single_iteration_parent_hash_is_sha256_of_prompt` | single iteration parent hash is sha256 of prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parent_hash_is_64_hex_chars` | parent hash is 64 hex chars |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_grows_across_iterations` | chain grows across iterations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parent_hash_stable_across_iterations` | parent hash stable across iterations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explicit_parent_hash_is_preserved` | explicit parent hash is preserved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_three_iteration_chain_length` | three iteration chain length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervision_report_has_refinement_chain` | supervision report has refinement chain |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervision_report_has_parent_prompt_hash` | supervision report has parent prompt hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_chain_present` | metric driven chain present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_parent_hash_present` | metric driven parent hash present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_survives_to_dict` | chain survives to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_prompt_helper` | hash prompt helper |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_prompt_deterministic` | hash prompt deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_prompt_differs_for_different_prompts` | hash prompt differs for different prompts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_list_refinement_chain_equals_none` | refinement_chain=[] debe comportarse igual que refinement_chain=None. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_string_parent_hash_computes_from_prompt` | parent_prompt_hash='' debe calcular el hash del prompt, igual que None. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_has_refinement_chain_comment` | mxai has refinement chain comment |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_has_parent_prompt_hash_comment` | mxai has parent prompt hash comment |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_metadata_roundtrip` | mxai metadata roundtrip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_metadata_chain_grows_across_iterations` | mxai metadata chain grows across iterations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_refinement_metadata_no_metadata` | parse refinement metadata no metadata |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embed_refinement_metadata_empty_mxai` | embed refinement metadata empty mxai |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embed_refinement_metadata_roundtrip` | embed refinement metadata roundtrip |
| &nbsp;&nbsp;**TestP13MetricHelpers** | *TestP13MetricHelpers* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_metric_explanation_accuracy` | build metric explanation accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_metric_explanation_loss_only` | build metric explanation loss only |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_metric_explanation_empty_eval` | build metric explanation empty eval |
| &nbsp;&nbsp;**TestP13RefinementHelpers** | *TestP13RefinementHelpers* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_make_refinement_id_format` | make refinement id format |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_make_refinement_id_deterministic` | make refinement id deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_refined_prompt_no_hints` | build refined prompt no hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_refined_prompt_with_hints` | build refined prompt with hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_explanation_contains_audit_preview` | build explanation contains audit preview |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_explanation_no_actions` | build explanation no actions |
| **[test_p13_cli_refine.py](../tests/test_p13_cli_refine.py)** | CLI `matrixai refine`: flags, --accept guard, --chain, --max-iterations exit code 2 (29 tests) |
| &nbsp;&nbsp;**TestP13CliRefineArgParsing** | *Fast argparse-level tests — no PromptSupervisor calls.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_help_exits_zero` | help exits zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_audit_and_evaluation_returns_2` | missing audit and evaluation returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_both_audit_and_evaluation_returns_2` | both audit and evaluation returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_prompt_returns_2` | empty prompt returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_audit_file_returns_2` | missing audit file returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_evaluation_file_returns_2` | missing evaluation file returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_malformed_audit_json_returns_2` | malformed audit json returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_mxai_file_returns_2` | missing mxai file returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_malformed_chain_file_returns_2` | malformed chain file returns 2 |
| &nbsp;&nbsp;**TestP13CliRefineAuditDriven** | *Integration tests with PromptSupervisor — audit_driven mode.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_json_output_has_refinement_id` | json output has refinement id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_json_output_has_chain_and_parent_hash` | json output has chain and parent hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_json_output_mode_is_audit_driven` | json output mode is audit driven |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_human_readable_output_contains_refinement_id` | human readable output contains refinement id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_accept_no_output_files_written` | no accept no output files written |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accept_without_outputs_emits_warning` | accept without outputs emits warning |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accept_writes_proposed_prompt` | accept writes proposed prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accept_writes_chain_output` | accept writes chain output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accept_writes_mxai_with_metadata` | accept writes mxai with metadata |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hint_flag_appears_in_json_hints_applied` | hint flag appears in json hints applied |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_flag_stored_in_proposal` | iteration flag stored in proposal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_stdin_prompt_with_dash` | stdin prompt with dash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_file_propagated_to_second_iteration` | chain file propagated to second iteration |
| &nbsp;&nbsp;**TestP13CliRefineIterationLimit** | *Corte 5 — hard iteration limit via CLI.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_exceeds_default_limit_returns_2` | iteration exceeds default limit returns 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_iterations_flag_blocks_when_exceeded` | max iterations flag blocks when exceeded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_iterations_flag_allows_at_boundary` | max iterations flag allows at boundary |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_error_message_includes_counts` | error message includes counts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_iterations_appears_in_help` | max iterations appears in help |
| &nbsp;&nbsp;**TestP13CliRefineMetricDriven** | *Integration tests — metric_driven mode.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_json_mode` | metric driven json mode |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_driven_chain_in_proposal` | metric driven chain in proposal |
| **[test_p13_playground_refine.py](../tests/test_p13_playground_refine.py)** | Playground refinement UI: refine button, delta view, apply, iteration counter (20 tests) |
| &nbsp;&nbsp;**TestP13PlaygroundRefineValidation** | *TestP13PlaygroundRefineValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_prompt_returns_error` | missing prompt returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_prompt_returns_error` | empty prompt returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_run_result_returns_error` | missing run result returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_run_result_returns_error` | empty run result returns error |
| &nbsp;&nbsp;**TestP13PlaygroundRefineSuccess** | *TestP13PlaygroundRefineSuccess* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ok_response_has_required_keys` | ok response has required keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mode_is_audit_driven` | mode is audit driven |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_starts_with_one_element` | chain starts with one element |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parent_hash_is_64_hex_chars` | parent hash is 64 hex chars |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_defaults_to_1` | iteration defaults to 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explicit_iteration_reflected` | explicit iteration reflected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_field_stripped_from_run_result` | audit field stripped from run result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hints_accepted` | hints accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_text_accepted` | mxai text accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_grows_across_iterations` | chain grows across iterations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parent_hash_stable_across_iterations` | parent hash stable across iterations |
| &nbsp;&nbsp;**TestP13PlaygroundRefineIterationLimit** | *TestP13PlaygroundRefineIterationLimit* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_limit_reached_returns_error` | iteration limit reached returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_limit_error_message_has_counts` | iteration limit error message has counts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_default_max_iterations_is_3` | default max iterations is 3 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_at_max_iterations_boundary_succeeds` | at max iterations boundary succeeds |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_custom_max_iterations_respected` | custom max iterations respected |
| **[test_p13_cut7_regression.py](../tests/test_p13_cut7_regression.py)** | Refinement regression: P1–P12 imports and full pipeline integrity (36 tests) |
| &nbsp;&nbsp;**TestP13RegressionPromptSupervisionReport** | *P13 added refinement_chain and parent_prompt_hash with defaults; old callers unaffected.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_new_fields_default_to_empty` | new fields default to empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_is_still_frozen` | report is still frozen |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervise_prompt_returns_empty_chain_without_p13_context` | supervise prompt returns empty chain without p13 context |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervise_prompt_returns_accepted_or_rejected` | supervise prompt returns accepted or rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervise_prompt_report_to_dict_has_no_spurious_keys` | supervise prompt report to dict has no spurious keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervise_semantic_still_works` | supervise semantic still works |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervise_semantic_new_fields_still_empty_without_p13` | supervise semantic new fields still empty without p13 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_summary_method_still_returns_string` | summary method still returns string |
| &nbsp;&nbsp;**TestP13RegressionAgentsExports** | *P13 adds new exports; existing ones must still be importable and functional.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_auditor_agent_importable` | auditor agent importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_supervisor_importable` | prompt supervisor importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_safety_agent_importable` | safety agent importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_agent_importable` | verifier agent importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_llm_proposal_provider_importable` | llm proposal provider importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p13_new_exports_coexist_with_old` | p13 new exports coexist with old |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_list_unchanged_for_old_symbols` | all list unchanged for old symbols |
| &nbsp;&nbsp;**TestP13RegressionCLICommands** | *New refine subcommand must not shadow or break existing CLI subcommands.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refine_help_exits_0` | refine help exits 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_help_unaffected` | train help unaffected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_help_unaffected` | evaluate help unaffected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_supervised_help_unaffected` | train supervised help unaffected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_training_help_unaffected` | validate training help unaffected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_training_help_unaffected` | generate training help unaffected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_dataset_help_unaffected` | generate dataset help unaffected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_top_level_help_lists_refine` | matrixai top level help lists refine |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_top_level_help_still_lists_train` | matrixai top level help still lists train |
| &nbsp;&nbsp;**TestP13RegressionPlaygroundAnalyze** | *P13 added _refine_prompt and modified renderRunView; analyze pipeline must be unchanged.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_analyze_without_input_still_returns_ok_structure` | analyze without input still returns ok structure |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_analyze_with_input_returns_run_result` | analyze with input returns run result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_result_still_has_audit_text` | run result still has audit text |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refine_prompt_helper_does_not_strip_audit_from_analyze_result` | analyze_playground_request() and _refine_prompt() are independent; |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_analyze_workflow_steps_still_present` | analyze workflow steps still present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_analyze_checks_still_present` | analyze checks still present |
| &nbsp;&nbsp;**TestP13RegressionRefinePromptIsolation** | *_refine_prompt strips 'audit' key from run_result before passing to RefinementAgent;* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_original_run_result_dict_not_mutated` | original run result dict not mutated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refine_prompt_ok_response_does_not_include_audit` | refine prompt ok response does not include audit |
| &nbsp;&nbsp;**TestP13RegressionAuditorAgent** | *P13 imports AuditorAgent in playground.py; the agent itself must be unchanged.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explain_returns_non_empty_string` | explain returns non empty string |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_result_still_has_actions_key` | run result still has actions key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_result_still_has_trace_key` | run result still has trace key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_auditor_explain_does_not_mutate_run_result` | auditor explain does not mutate run result |

## Training and Parameters (59 tests)

| Test | Description |
|------|-------------|
| **[test_training_contract.py](../tests/test_training_contract.py)** | .mxtrain spec: parser, fields, TrainingVerifier, DifferentiabilityVerifier (15 tests) |
| &nbsp;&nbsp;**MatrixAITrainingContractTest** | *MatrixAITrainingContractTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxtrain_parser_accepts_email_training_spec` | mxtrain parser accepts email training spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxtrain_parser_accepts_fall_risk_binary_training_spec` | mxtrain parser accepts fall risk binary training spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxtrain_parser_accepts_fall_risk_probability_target_spec` | mxtrain parser accepts fall risk probability target spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_training_accepts_email_training_spec` | validate training accepts email training spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_training_accepts_fall_risk_binary_training_spec` | validate training accepts fall risk binary training spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_training_accepts_fall_risk_probability_target_spec` | validate training accepts fall risk probability target spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_cross_entropy_requires_two_label_values` | binary cross entropy requires two label values |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_cross_entropy_rejects_probability_target_out_of_range` | binary cross entropy rejects probability target out of range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiability_verifier_reports_parameter_paths` | differentiability verifier reports parameter paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_target_missing_fails` | loss target missing fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_update_unknown_parameter_fails` | update unknown parameter fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_non_differentiable_training_path_fails` | non differentiable training path fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probability_range_validation` | probability range validation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_label_domain_validation` | label domain validation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_validate_training_json` | cli validate training json |
| **[test_training_pipeline.py](../tests/test_training_pipeline.py)** | Supervised training: softmax/sigmoid trainers, SGD, ParameterSet, EvaluationResult (17 tests) |
| &nbsp;&nbsp;**MatrixAITrainingPipelineTest** | *MatrixAITrainingPipelineTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_data_adapter_emits_matrixai_batches_with_metadata` | csv data adapter emits matrixai batches with metadata |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_in_memory_data_adapter_emits_reproducible_batches` | in memory data adapter emits reproducible batches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_softmax_linear_reduces_loss_and_writes_artifacts` | train softmax linear reduces loss and writes artifacts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_sigmoid_linear_binary_cross_entropy_reduces_loss` | train sigmoid linear binary cross entropy reduces loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_sigmoid_linear_binary_cross_entropy_probability_target` | train sigmoid linear binary cross entropy probability target |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_fall_risk_binary_classifier_reports_metrics` | evaluate fall risk binary classifier reports metrics |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_trace_includes_split_fingerprint` | training trace includes split fingerprint |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trained_params_validate_and_run_with_runtime` | trained params validate and run with runtime |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_trained_params_reports_metrics` | evaluate trained params reports metrics |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_can_use_separate_test_dataset` | evaluate can use separate test dataset |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trained_parameter_set_matches_differentiable_compiler_runtime` | trained parameter set matches differentiable compiler runtime |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_trained_params_runtime_matches_compiled` | fall risk trained params runtime matches compiled |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_train_writes_parameter_artifacts` | cli train writes parameter artifacts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_evaluate_uses_parameter_set_and_training_spec` | cli evaluate uses parameter set and training spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_run_accepts_trained_parameter_set` | cli backend run accepts trained parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_train_rejects_mismatched_model_argument` | cli train rejects mismatched model argument |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_trace_snapshot` | training trace snapshot |
| **[test_training_generation.py](../tests/test_training_generation.py)** | Training generation: generate-training, generate-supervised, train-supervised CLI (14 tests) |
| &nbsp;&nbsp;**MatrixAITrainingGenerationTest** | *MatrixAITrainingGenerationTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_fall_risk_binary_training_contract_from_prompt` | generate fall risk binary training contract from prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_email_softmax_training_contract_from_prompt` | generate email softmax training contract from prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_training_rejects_wrong_softmax_label_count` | generate training rejects wrong softmax label count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_generate_training_writes_mxtrain_and_dataset_template` | cli generate training writes mxtrain and dataset template |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_supervised_prompt_package_writes_valid_relative_artifacts` | generate supervised prompt package writes valid relative artifacts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_generate_supervised_writes_valid_classification_package` | cli generate supervised writes valid classification package |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_supervised_prompt_with_real_fall_risk_datasets_trains_and_evaluates` | run supervised prompt with real fall risk datasets trains and evaluates |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_supervised_prompt_accepts_dataset_manifest` | run supervised prompt accepts dataset manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_manifest_verifies_versioned_split` | dataset manifest verifies versioned split |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_supervised_prompt_accepts_dataset_manifest_split` | run supervised prompt accepts dataset manifest split |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_manifest_rejects_split_fingerprint_mismatch` | dataset manifest rejects split fingerprint mismatch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_manifest_rejects_hash_mismatch` | dataset manifest rejects hash mismatch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_train_supervised_generates_trains_and_evaluates_email_package` | cli train supervised generates trains and evaluates email package |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_train_supervised_accepts_dataset_manifest` | cli train supervised accepts dataset manifest |
| **[test_parameters.py](../tests/test_parameters.py)** | ParameterSet: versioning, schema hash, init-parameters, validate-parameters (13 tests) |
| &nbsp;&nbsp;**MatrixAIParameterStoreTest** | *MatrixAIParameterStoreTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_initial_parameter_set_from_backend_manifest` | build initial parameter set from backend manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_store_roundtrip` | parameter store roundtrip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_parameter_schema_fails` | wrong parameter schema fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_parameter_value_shape_fails` | wrong parameter value shape fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_init_parameters_writes_versioned_parameter_set` | cli init parameters writes versioned parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_validate_parameters_accepts_parameter_set` | cli validate parameters accepts parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_parameters_torch_outputs_parameter_set_without_torch_dependency` | cli backend parameters torch outputs parameter set without torch dependency |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_parameters_torch_validates_parameter_set_without_torch_dependency` | cli backend parameters torch validates parameter set without torch dependency |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bridge_validates_parameter_set_before_torch_import` | tensor bridge validates parameter set before torch import |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bridge_requires_optional_torch_dependency_when_absent` | tensor bridge requires optional torch dependency when absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bridge_materializes_torch_tensors` | tensor bridge materializes torch tensors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bridge_roundtrip_preserves_hashes` | tensor bridge roundtrip preserves hashes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bridge_rejects_torch_tensor_shape_mismatch` | tensor bridge rejects torch tensor shape mismatch |

## Synthetic Data (P12) (54 tests)

| Test | Description |
|------|-------------|
| **[test_p12_synthetic_data.py](../tests/test_p12_synthetic_data.py)** | SyntheticDataGenerator: random/coherent modes, DatasetManifest, playground integration (39 tests) |
| &nbsp;&nbsp;**TestP12SyntheticDataGenerator** | *TestP12SyntheticDataGenerator* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_mode_generation` | random mode generation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reproducibility` | reproducibility |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_mode_has_zero_fallback_count` | random mode has zero fallback count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_mode_fallback_counted_when_runtime_cannot_resolve` | coherent mode fallback counted when runtime cannot resolve |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_mode_no_warning_when_no_fallback` | coherent mode no warning when no fallback |
| &nbsp;&nbsp;**TestP12SyntheticManifest** | *TestP12SyntheticManifest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_spec_round_trip` | generator spec round trip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_synthetic_manifest` | build synthetic manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_to_dict_includes_origin_and_generator` | manifest to dict includes origin and generator |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_from_dict_round_trip` | manifest from dict round trip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_load_manifest_from_json_file` | load manifest from json file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_manifest_is_synthetic_flag` | verify manifest is synthetic flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_manifest_non_synthetic_is_not_flagged` | verify manifest non synthetic is not flagged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_includes_is_synthetic_when_true` | to dict includes is synthetic when true |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_without_origin_preserves_backwards_compat` | manifest without origin preserves backwards compat |
| &nbsp;&nbsp;**TestP12GenerateDatasetCLI** | *TestP12GenerateDatasetCLI* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_csv_and_manifest` | generates csv and manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_has_correct_schema` | manifest has correct schema |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_headers_and_rows` | csv headers and rows |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reproducibility_same_seed` | reproducibility same seed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_seeds_produce_different_data` | different seeds produce different data |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_manifest_passes_on_generated_files` | verify manifest passes on generated files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rows_too_small_returns_error` | rows too small returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rows_too_large_returns_error` | rows too large returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_human_readable_output` | human readable output |
| &nbsp;&nbsp;**TestP12SyntheticOriginPropagation** | *Corte 5: synthetic_origin propagation via train-supervised with synthetic manifest.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_supervised_with_synthetic_manifest_propagates_synthetic_origin` | train supervised with synthetic manifest propagates synthetic origin |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_supervised_with_real_manifest_has_no_synthetic_origin` | train supervised with real manifest has no synthetic origin |
| &nbsp;&nbsp;**TestP12PlaygroundGenerateSyntheticDataset** | *Unit tests for the _generate_synthetic_dataset playground backend function.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ok_returns_csv_text` | ok returns csv text |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_row_count_matches_request` | row count matches request |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_present_and_deterministic` | fingerprint present and deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_seeds_give_different_fingerprints` | different seeds give different fingerprints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_origin_is_synthetic` | origin is synthetic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_columns_in_response` | columns in response |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_row_clamp_respects_p9_max` | row clamp respects p9 max |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_row_clamp_min_2` | row clamp min 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_mxai_returns_error` | empty mxai returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_training_returns_error` | empty training returns error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_mode_returns_ok` | coherent mode returns ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_mode_fallback_exposed_in_response` | coherent mode fallback exposed in response |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_mode_has_no_fallback_key` | random mode has no fallback key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_invalid_mode_falls_back_to_random` | invalid mode falls back to random |
| **[test_p12_cut7_regression.py](../tests/test_p12_cut7_regression.py)** | Synthetic data regression: P1–P11 imports and end-to-end pipeline (15 tests) |
| &nbsp;&nbsp;**TestP12RegressionManifestCompat** | *P12 added origin/generator to DatasetManifest; real manifests must be unchanged.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_without_origin_round_trips_cleanly` | manifest without origin round trips cleanly |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_real_manifest_is_not_synthetic` | verify real manifest is not synthetic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_real_manifest_to_dict_no_is_synthetic_key` | verify real manifest to dict no is synthetic key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_load_real_manifest_preserves_all_existing_fields` | load real manifest preserves all existing fields |
| &nbsp;&nbsp;**TestP12RegressionSupervisedPrompt** | *P12 added synthetic_origin to SupervisedPromptRunResult; non-synthetic runs unaffected.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_result_has_synthetic_origin_false_with_direct_csvs` | run result has synthetic origin false with direct csvs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_trace_has_no_synthetic_origin_with_direct_csvs` | training trace has no synthetic origin with direct csvs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_has_no_synthetic_origin_with_direct_csvs` | evaluation report has no synthetic origin with direct csvs |
| &nbsp;&nbsp;**TestP12RegressionSyntheticLabelLookup** | *Fix: generator now checks parameters['args'] before parameters['labels'].* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_labels_key_still_works` | labels key still works |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_args_key_works_as_real_parser_produces` | args key works as real parser produces |
| &nbsp;&nbsp;**TestP12RegressionCLICommands** | *New generate-dataset subcommand must not shadow or break existing commands.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_training_help` | validate training help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_training_help` | generate training help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_supervised_help` | train supervised help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_help` | train help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_help` | evaluate help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generate_dataset_help` | generate dataset help |

## Serving and HTTP Server (12 tests)

| Test | Description |
|------|-------------|
| **[test_server.py](../tests/test_server.py)** | HTTP prediction server: /predict, /health, batch input, Bearer auth, error codes (9 tests) |
| &nbsp;&nbsp;**TestServerHandler** | *TestServerHandler* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_health_check` | health check |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_openapi` | openapi |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_docs` | docs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_invalid_auth` | predict invalid auth |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_missing_auth` | predict missing auth |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_success` | predict success |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_batch` | predict batch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_invalid_json_counts_failed_request` | predict invalid json counts failed request |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_empty_payload_counts_failed_request` | predict empty payload counts failed request |
| **[test_pack.py](../tests/test_pack.py)** | matrixai pack --docker: Dockerfile, docker-compose.yml, .env.example with UTF-8 encoding (3 tests) |
| &nbsp;&nbsp;**TestPack** | *TestPack* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pack_model_without_docker` | pack model without docker |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pack_model_with_docker_and_params` | pack model with docker and params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pack_model_rejects_invalid_params` | pack model rejects invalid params |

## PyTorch Backend (11 tests)

| Test | Description |
|------|-------------|
| **[test_torch_forward.py](../tests/test_torch_forward.py)** | Torch forward pass: softmax_linear, sigmoid_linear, TensorParameterBridge (5 tests) |
| &nbsp;&nbsp;**MatrixAITorchForwardTest** | *MatrixAITorchForwardTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_blocks_deferred_continuous_program_before_torch_import` | torch forward blocks deferred continuous program before torch import |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_run_torch_requires_optional_dependency_when_absent` | cli backend run torch requires optional dependency when absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_email_matches_differentiable_python_initial_parameters` | torch forward email matches differentiable python initial parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_fall_risk_matches_differentiable_python_initial_parameters` | torch forward fall risk matches differentiable python initial parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_backend_run_torch_outputs_forward_report` | cli backend run torch outputs forward report |
| **[test_torch_training.py](../tests/test_torch_training.py)** | Torch training: SGD with torch, loss, evaluate, backend=torch in CLI (6 tests) |
| &nbsp;&nbsp;**MatrixAITorchTrainingTest** | *MatrixAITorchTrainingTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_train_torch_requires_optional_dependency_when_absent` | cli train torch requires optional dependency when absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_evaluate_torch_requires_optional_dependency_when_absent` | cli evaluate torch requires optional dependency when absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_train_softmax_linear_writes_p4_artifacts` | torch train softmax linear writes p4 artifacts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_train_sigmoid_linear_binary_cross_entropy_writes_valid_params` | torch train sigmoid linear binary cross entropy writes valid params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_train_sigmoid_linear_probability_target_writes_valid_params` | torch train sigmoid linear probability target writes valid params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_torch_backend_matches_stdlib` | evaluate torch backend matches stdlib |

## Language Expressivity and Typed Layers (P10) (178 tests)

| Test | Description |
|------|-------------|
| **[test_p10_layers.py](../tests/test_p10_layers.py)** | LAYER blocks: declaration, inline PARAM, call_layer IR, hierarchical serialisation (22 tests) |
| &nbsp;&nbsp;**LayerParseTest** | *LayerParseTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_parsed` | attention layer parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_has_three_params` | attention layer has three params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_param_shapes` | attention layer param shapes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_input_type` | attention layer input type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_output_type` | attention layer output type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_trainable_by_default` | layer params trainable by default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_simple_layer_parsed` | simple layer parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bare_layer_no_params` | bare layer no params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_layers_by_default` | no layers by default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_inline_param_trainable_false` | layer inline param trainable false |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_inline_param_init` | layer inline param init |
| &nbsp;&nbsp;**CallLayerExpressionTest** | *CallLayerExpressionTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_call_layer_kind` | call layer kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_call_layer_layer_name` | call layer layer name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_call_layer_input` | call layer input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_call_layer_simple_linear` | call layer simple linear |
| &nbsp;&nbsp;**LayerToDictTest** | *LayerToDictTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layers_appear_in_to_dict` | layers appear in to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_dict_has_name` | layer dict has name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_dict_has_parameters` | layer dict has parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_dict_has_types` | layer dict has types |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_layers_key_when_empty` | no layers key when empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_layers_key_when_truly_empty` | no layers key when truly empty |
| &nbsp;&nbsp;**BackwardCompatLayerTest** | *Programs without LAYER blocks must parse identically to before.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_existing_program_unchanged` | existing program unchanged |
| **[test_p10_types.py](../tests/test_p10_types.py)** | Structured types: Tensor, Sequence, Embedding, type annotations inside LAYER (35 tests) |
| &nbsp;&nbsp;**TensorTypeTest** | *TensorTypeTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_1d` | tensor 1d |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_2d` | tensor 2d |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_3d` | tensor 3d |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_shape_helper` | tensor shape helper |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_shape_1d_helper` | tensor shape 1d helper |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bare_has_no_shape` | tensor bare has no shape |
| &nbsp;&nbsp;**EmbeddingTypeTest** | *EmbeddingTypeTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_single_dim` | embedding single dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_vocab_dim` | embedding vocab dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_dims_helper` | embedding dims helper |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_dims_single_returns_none` | embedding dims single returns none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_shape_for_embedding_vocab_dim` | tensor shape for embedding vocab dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_shape_for_embedding_single_dim` | tensor shape for embedding single dim |
| &nbsp;&nbsp;**SequenceTypeTest** | *SequenceTypeTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_tensor_1d` | sequence tensor 1d |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_tensor_2d` | sequence tensor 2d |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_scalar` | sequence scalar |
| &nbsp;&nbsp;**ListTypeTest** | *ListTypeTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_probability` | list probability |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_tensor` | list tensor |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_bare_still_works` | list bare still works |
| &nbsp;&nbsp;**MapTypeTest** | *MapTypeTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_map_string_score` | map string score |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_map_bare_still_works` | map bare still works |
| &nbsp;&nbsp;**ValidateStructuredTypesTest** | *ValidateStructuredTypesTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_validates_elements` | list validates elements |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_rejects_non_list` | list rejects non list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_max_length` | sequence max length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_within_length_ok` | sequence within length ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_map_validates_dict` | map validates dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_map_rejects_non_dict` | map rejects non dict |
| &nbsp;&nbsp;**BackwardCompatTest** | *Existing P1-P5 type annotations must still parse correctly.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probability` | probability |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_score_with_range` | score with range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_single_dim_unchanged` | embedding single dim unchanged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_2d_unchanged` | tensor 2d unchanged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_vector` | vector |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_label_args` | label args |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_bare` | record bare |
| &nbsp;&nbsp;**TypeCheckerParamShapeTest** | *check_program_types warns on Tensor PARAM without declared shape.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_with_shape_no_warning` | tensor with shape no warning |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bare_emits_warning` | tensor bare emits warning |
| **[test_p10_params.py](../tests/test_p10_params.py)** | Hierarchical ParameterSet: layer.param paths, manifest shapes, round-trip (19 tests) |
| &nbsp;&nbsp;**HierarchicalManifestTest** | *HierarchicalManifestTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_appear_in_manifest` | layer params appear in manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_have_hierarchical_path` | layer params have hierarchical path |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_flat_params_path_equals_name` | flat params path equals name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_have_function_field` | layer params have function field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_shapes` | layer params shapes |
| &nbsp;&nbsp;**HierarchicalParameterSetTest** | *HierarchicalParameterSetTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_stored_with_path_key` | layer params stored with path key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_flat_params_stored_with_name_key` | flat params stored with name key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_param_entry_has_shape` | layer param entry has shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_param_entry_has_is_layer_flag` | layer param entry has is layer flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_flat_param_no_is_layer_flag` | flat param no is layer flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_params_have_initial_values` | layer params have initial values |
| &nbsp;&nbsp;**HierarchicalRuntimeParametersTest** | *HierarchicalRuntimeParametersTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_path_key_in_runtime_params` | layer path key in runtime params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_flat_params_still_expose_function_dot_name` | flat params still expose function dot name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_path_not_doubled` | Attention.Wq must not appear as Attention.Attention.Wq. |
| &nbsp;&nbsp;**ValidateHierarchicalParameterSetTest** | *ValidateHierarchicalParameterSetTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_valid_layer_parameter_set` | valid layer parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_valid_flat_parameter_set_unchanged` | valid flat parameter set unchanged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_layer_param_detected` | missing layer param detected |
| &nbsp;&nbsp;**BackwardCompatFlatParamsTest** | *Existing flat ParameterSet programs must continue to work identically.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_flat_ps_roundtrip` | flat ps roundtrip |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_flat_param_keys_unchanged` | flat param keys unchanged |
| **[test_p10_primitives.py](../tests/test_p10_primitives.py)** | Tensor primitives: dot, matmul, relu, gelu, layer_norm, residual — IR and eval (21 tests) |
| &nbsp;&nbsp;**TensorPrimitiveParseTest** | *TensorPrimitiveParseTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_kind` | dot kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_inputs` | dot inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matmul_kind` | matmul kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_kind` | relu kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_inputs` | relu inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gelu_kind` | gelu kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_norm_kind` | layer norm kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_norm_inputs` | layer norm inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_kind` | residual kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_inputs` | residual inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_primitive_has_ast_in_parameters` | primitive has ast in parameters |
| &nbsp;&nbsp;**TensorPrimitiveBackendContractTest** | *TensorPrimitiveBackendContractTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_is_supported` | dot is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matmul_is_supported` | matmul is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_is_supported` | relu is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gelu_is_supported` | gelu is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_norm_is_supported` | layer norm is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_is_supported` | residual is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_kind_in_node_report` | dot kind in node report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_kind_in_node_report` | relu kind in node report |
| &nbsp;&nbsp;**TensorPrimitiveBackwardCompatTest** | *Existing programs must not be affected by adding tensor primitives.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_linear_unchanged` | softmax linear unchanged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_symbolic_expr_unchanged` | symbolic expr unchanged |
| **[test_p10_lowering.py](../tests/test_p10_lowering.py)** | Lowering to differentiable_python: code generation per primitive kind (14 tests) |
| &nbsp;&nbsp;**TensorPrimitiveLoweringTest** | *TensorPrimitiveLoweringTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_compiles` | dot compiles |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_runs` | dot runs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_compiles` | relu compiles |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_runs_positive` | relu runs positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_runs_negative` | relu runs negative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gelu_compiles` | gelu compiles |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gelu_runs` | gelu runs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_compiles` | residual compiles |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_runs` | residual runs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_compiles` | layer call compiles |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_runs` | layer call runs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_produces_list` | layer call produces list |
| &nbsp;&nbsp;**LoweringBackwardCompatTest** | *Existing programs must still compile and run correctly.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_linear_unchanged` | softmax linear unchanged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_symbolic_expr_unchanged` | symbolic expr unchanged |
| **[test_p10_backend_extended.py](../tests/test_p10_backend_extended.py)** | BackendContractAnalyzer extended: layer_manifest, param count per layer (14 tests) |
| &nbsp;&nbsp;**LayerManifestTest** | *LayerManifestTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_present` | layer manifest present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_count` | layer manifest count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_in_manifest` | attention layer in manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_layer_in_manifest` | linear layer in manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_layer_param_count` | attention layer param count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_layer_param_count` | linear layer param count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_param_paths` | layer param paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_param_shapes` | layer param shapes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_has_input_type` | layer has input type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_has_output_type` | layer has output type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_layer_manifest_for_flat_program` | empty layer manifest for flat program |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_dtype` | layer manifest dtype |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_trainable_flag` | layer manifest trainable flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_non_trainable_param_excluded` | non trainable param excluded |
| **[test_p10_attention_primitives.py](../tests/test_p10_attention_primitives.py)** | Attention primitives: embedding_lookup, positional_encoding, attention, mean/cls pooling (15 tests) |
| &nbsp;&nbsp;**AttentionPrimitiveParseTest** | *AttentionPrimitiveParseTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_kind` | embedding lookup kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_inputs` | embedding lookup inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_positional_encoding_kind` | positional encoding kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_kind` | attention kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_inputs` | attention inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mean_pooling_kind` | mean pooling kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cls_pooling_kind` | cls pooling kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_primitives_have_ast` | all primitives have ast |
| &nbsp;&nbsp;**AttentionPrimitiveBackendContractTest** | *AttentionPrimitiveBackendContractTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_supported_not_differentiable` | embedding lookup supported not differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_supported_not_differentiable` | attention supported not differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mean_pooling_supported_differentiable` | mean pooling supported differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cls_pooling_supported_differentiable` | cls pooling supported differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_positional_encoding_supported_differentiable` | positional encoding supported differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_kind_in_node` | embedding lookup kind in node |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_kind_in_node` | attention kind in node |
| **[test_p10_regression.py](../tests/test_p10_regression.py)** | P10 regression: P1–P9 import integrity and pipeline (13 tests) |
| &nbsp;&nbsp;**P1LanguageRegressionTest** | *P1LanguageRegressionTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_program_parses` | email program parses |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_program_parses` | fall risk program parses |
| &nbsp;&nbsp;**P2TypeSystemRegressionTest** | *P2TypeSystemRegressionTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_p2_types_parse` | all p2 types parse |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probability_range` | probability range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_program_types_no_errors` | check program types no errors |
| &nbsp;&nbsp;**P3BackendContractRegressionTest** | *P3BackendContractRegressionTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_contract_ok_for_softmax_linear` | backend contract ok for softmax linear |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_manifest_has_path` | parameter manifest has path |
| &nbsp;&nbsp;**P4TrainingRegressionTest** | *P4TrainingRegressionTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_initial_parameter_set` | build initial parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_parameter_set_ok` | validate parameter set ok |
| &nbsp;&nbsp;**P1P5DifferentiablePythonRegressionTest** | *P1P5DifferentiablePythonRegressionTest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_compiles` | differentiable python compiles |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_linear_compiles` | sigmoid linear compiles |
| &nbsp;&nbsp;**P10NewFeaturesCoexistTest** | *P10 features (layers, structured types) coexist with P1-P9 features.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_program_with_layers_and_flat_params` | program with layers and flat params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_structured_types_in_vector_fields` | structured types in vector fields |
| **[test_p10_audit_fixes.py](../tests/test_p10_audit_fixes.py)** | P10 audit fixes: edge cases in primitives, lowering, type constraints (25 tests) |
| &nbsp;&nbsp;**TestTensorShapeValidation** | *TestTensorShapeValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_correct_shape_passes` | correct shape passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_shape_reports_error` | wrong shape reports error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_rank_reports_error` | wrong rank reports error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_without_shape_accepts_any_list` | tensor without shape accepts any list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_non_list_rejected` | non list rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_3d_tensor_shape` | 3d tensor shape |
| &nbsp;&nbsp;**TestFunctionOutputShapeNewPrimitives** | *TestFunctionOutputShapeNewPrimitives* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_reported_as_supported` | dot reported as supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_relu_reported_as_supported` | relu reported as supported |
| &nbsp;&nbsp;**TestCallLayerValidatesExistence** | *TestCallLayerValidatesExistence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_layer_not_supported` | missing layer not supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_present_layer_supported` | present layer supported |
| &nbsp;&nbsp;**TestLayerManifestInToDict** | *TestLayerManifestInToDict* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_present_in_to_dict` | layer manifest present in to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_has_correct_layer` | layer manifest has correct layer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_layers_no_manifest_key` | no layers no manifest key |
| &nbsp;&nbsp;**TestRecordTypeSyntax** | *TestRecordTypeSyntax* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_record` | empty record |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_single_scalar_field` | single scalar field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_fields` | multiple fields |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_field` | tensor field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_with_tensor_comma_inside_brackets` | record with tensor comma inside brackets |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_in_param_parse` | record in param parse |
| &nbsp;&nbsp;**TestOperandShapeCompatibility** | *TestOperandShapeCompatibility* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_shape_mismatch_blocked` | residual shape mismatch blocked |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_same_shape_ok` | residual same shape ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dot_shape_mismatch_blocked` | dot shape mismatch blocked |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_with_mask_supported` | attention with mask supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_without_mask_supported` | attention without mask supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_q_k_dim_mismatch_blocked` | attention q k dim mismatch blocked |

## Transformer Architecture (P11) (292 tests)

| Test | Description |
|------|-------------|
| **[test_p11_cut1_layer_body.py](../tests/test_p11_cut1_layer_body.py)** | Layer body parsing: multi-statement LAYER, PARAM types, call_layer in IR (19 tests) |
| &nbsp;&nbsp;**TestLayerBodyOp** | *TestLayerBodyOp* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_single_body_op_parsed` | single body op parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_body_ops_parsed` | chain body ops parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_body_ops_when_only_params` | no body ops when only params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_body_ops_in_ir_to_dict` | body ops in ir to dict |
| &nbsp;&nbsp;**TestLayerExecutorGenerated** | *TestLayerExecutorGenerated* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_function_in_compiled_source` | executor function in compiled source |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_passthrough_when_no_body_ops` | passthrough when no body ops |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_executor_contains_relu` | chain executor contains relu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_body_binds_layer_param` | body binds layer param |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_param_binding_uses_is_not_none_not_or` | param binding uses is not none not or |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_param_binding_falsy_qualified_key_not_replaced_by_fallback` | param binding falsy qualified key not replaced by fallback |
| &nbsp;&nbsp;**TestLayerBodyExecution** | *TestLayerBodyExecution* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matmul_body_returns_transformed_input` | matmul body returns transformed input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_chain_body_applies_relu_after_matmul` | chain body applies relu after matmul |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matmul_with_non_identity_w` | matmul with non identity w |
| &nbsp;&nbsp;**TestMatchUpdatePatterns** | *TestMatchUpdatePatterns* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wildcard_star_matches_all` | wildcard star matches all |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exact_path_matches_one` | exact path matches one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prefix_wildcard_matches_subtree` | prefix wildcard matches subtree |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_patterns_union` | multiple patterns union |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_match_returns_empty` | no match returns empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exact_encoder_prefix_matches` | exact encoder prefix matches |
| **[test_p11_cut1_trainer_integration.py](../tests/test_p11_cut1_trainer_integration.py)** | Transformer trainer integration: forward+backward, ParameterSet round-trip (21 tests) |
| &nbsp;&nbsp;**TestVerifierUpdatePatterns** | *TestVerifierUpdatePatterns* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wildcard_star_accepted_for_all_params` | wildcard star accepted for all params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exact_hierarchical_path_accepted` | exact hierarchical path accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exact_flat_name_accepted` | exact flat name accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prefix_wildcard_accepted_when_params_match` | prefix wildcard accepted when params match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prefix_wildcard_matches_all_layers_of_prefix` | prefix wildcard matches all layers of prefix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unknown_prefix_rejected` | unknown prefix rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unknown_exact_rejected` | unknown exact rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mix_valid_and_invalid_reports_only_invalid` | mix valid and invalid reports only invalid |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_original_error_message_preserved` | original error message preserved |
| &nbsp;&nbsp;**TestGenericSupervisedTrainer** | *TestGenericSupervisedTrainer* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_keys_with_wildcard_star` | trainable keys with wildcard star |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_keys_with_layer_prefix` | trainable keys with layer prefix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_keys_with_exact_path` | trainable keys with exact path |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_matching_pattern_raises` | no matching pattern raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_trace_has_correct_length` | epoch trace has correct length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_trace_entry_keys` | epoch trace entry keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_numbers_are_sequential` | epoch numbers are sequential |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_best_and_final_params` | returns best and final params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_best_params_is_matrix` | best params is matrix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_is_non_negative` | loss is non negative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_decreases_on_deterministic_task` | loss decreases on deterministic task |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_callback_called_each_epoch` | epoch callback called each epoch |
| **[test_p11_cut2_template.py](../tests/test_p11_cut2_template.py)** | Transformer template: embed → attn → ffn → classifier with layer_norm (47 tests) |
| &nbsp;&nbsp;**TestTemplateParses** | *TestTemplateParses* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_template_file_exists` | template file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_project_name` | project name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_three_layers_declared` | three layers declared |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_one_vector_declared` | one vector declared |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_three_functions_declared` | three functions declared |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_topology` | graph topology |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_edges` | graph edges |
| &nbsp;&nbsp;**TestLayerStructure** | *TestLayerStructure* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_param_count` | encoder attn param count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_param_names` | encoder attn param names |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_body_op_count` | encoder attn body op count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_body_ops_include_attention` | encoder attn body ops include attention |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_param_count` | encoder ffn param count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_param_names` | encoder ffn param names |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_body_op_count` | encoder ffn body op count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_body_ops_include_gelu` | encoder ffn body ops include gelu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_param_count` | classifier param count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_param_names` | classifier param names |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_body_op_count` | classifier body op count |
| &nbsp;&nbsp;**TestBackendContract** | *TestBackendContract* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_ok` | report ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_unsupported_nodes` | no unsupported nodes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_parameter_errors` | no parameter errors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_parameter_count` | trainable parameter count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_expected_paths_present` | all expected paths present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrix_shapes` | matrix shapes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_vector_shapes` | vector shapes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_field_in_manifest_dict` | trainable field in manifest dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bias_params_have_bias_role` | bias params have bias role |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gain_params_have_gain_role` | gain params have gain role |
| &nbsp;&nbsp;**TestCompilation** | *TestCompilation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_source_non_empty` | source non empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_exec_encoder_attn_generated` | layer exec encoder attn generated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_exec_encoder_ffn_generated` | layer exec encoder ffn generated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_exec_classifier_generated` | layer exec classifier generated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_param_keys_in_source` | all param keys in source |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_is_not_none_sentinel_used` | is not none sentinel used |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_source_executes_without_error` | source executes without error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_forward_pass_returns_logits_length_2_and_finite` | forward pass returns logits length 2 and finite |
| &nbsp;&nbsp;**TestParameterSet** | *TestParameterSet* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_id` | parameter set id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_expected_keys_present` | all expected keys present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_total_key_count` | total key count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrix_values_have_correct_shape` | matrix values have correct shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_w1_has_shape_8x32` | w1 has shape 8x32 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_w_has_shape_8x2` | classifier w has shape 8x2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bias_vectors_are_lists` | bias vectors are lists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bias_values_are_zeros` | bias values are zeros |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gain_values_are_ones` | gain values are ones |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_non_empty` | model hash non empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash_non_empty` | parameter schema hash non empty |
| **[test_p11_cut3_forward_pass.py](../tests/test_p11_cut3_forward_pass.py)** | Transformer forward: embedding lookup, scaled dot-product attention, FFN, residuals (33 tests) |
| &nbsp;&nbsp;**TestForwardPassShapes** | *TestForwardPassShapes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_vector_shape` | input vector shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attn_block_output_shape` | attn block output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ffn_block_output_shape` | ffn block output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_output_shape` | logits output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attn_block_alias_matches` | attn block alias matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ffn_block_alias_matches` | ffn block alias matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_alias_matches` | logits alias matches |
| &nbsp;&nbsp;**TestForwardPassState** | *TestForwardPassState* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_graph_nodes_in_state` | all graph nodes in state |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_output_refs_in_state` | all output refs in state |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_scalar_fields_in_state` | scalar fields in state |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_extra_junk_keys` | no extra junk keys |
| &nbsp;&nbsp;**TestForwardPassTrace** | *TestForwardPassTrace* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_has_four_entries` | trace has four entries |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_node_order` | trace node order |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_node_types` | trace node types |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_output_refs` | trace output refs |
| &nbsp;&nbsp;**TestForwardPassValues** | *TestForwardPassValues* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_are_finite` | logits are finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_intermediate_outputs_are_finite` | intermediate outputs are finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_forward_pass_is_deterministic` | forward pass is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_input_produces_different_logits` | different input produces different logits |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_params_produce_different_logits` | different params produce different logits |
| &nbsp;&nbsp;**TestForwardPassMetadata** | *TestForwardPassMetadata* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_autodiff_plan_ready` | autodiff plan ready |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_parameters_count` | trainable parameters count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_manifest_count` | parameter manifest count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_manifest_has_trainable_field` | parameter manifest has trainable field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_shapes_includes_input` | tensor shapes includes input |
| &nbsp;&nbsp;**TestTorchBackendContract** | *Verify the torch backend contract accepts the transformer template — no PyTorch required.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_contract_ok` | torch contract ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_contract_layer_call_nodes_supported` | torch contract layer call nodes supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_contract_still_ok` | differentiable python contract still ok |
| &nbsp;&nbsp;**TestForwardPassTorchEquivalence** | *Verify differentiable_python and torch backends produce equivalent logits.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_match_within_tolerance` | logits match within tolerance |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_state_has_expected_shapes` | torch forward state has expected shapes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unknown_body_op_raises_torch_forward_error` | unknown body op raises torch forward error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_deferred_body_op_raises_torch_forward_error` | deferred body op raises torch forward error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_with_mask_applies_mask_to_score` | attention with mask applies mask to score |
| **[test_p11_cut4_training_spec.py](../tests/test_p11_cut4_training_spec.py)** | Transformer .mxtrain: SEQUENCE input, INT64 vocab, epochs and loss (44 tests) |
| &nbsp;&nbsp;**TestTrainingSpecParses** | *TestTrainingSpecParses* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_file_exists` | file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parses_without_exception` | parses without exception |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_field` | model field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_name` | dataset name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_source_kind` | dataset source kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_input_vector` | dataset input vector |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_input_columns` | dataset input columns |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_target_name` | dataset target name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_target_labels` | dataset target labels |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_type` | loss type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_prediction` | loss prediction |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_target` | loss target |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_optimizer_type` | optimizer type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_optimizer_learning_rate` | optimizer learning rate |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_optimizer_update_patterns` | optimizer update patterns |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_epochs` | run epochs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metric_type` | metric type |
| &nbsp;&nbsp;**TestTrainingVerifier** | *TestTrainingVerifier* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_ok` | verifier ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_errors` | no errors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_path_resolved` | model path resolved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_path_resolved` | dataset path resolved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_count` | trainable params count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_params_present` | encoder attn params present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_params_present` | encoder ffn params present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_params_present` | classifier params present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiability_ok` | differentiability ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiability_prediction_node` | differentiability prediction node |
| &nbsp;&nbsp;**TestUpdatePatterns** | *Verify UPDATE glob patterns match the correct trainable parameters.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_glob_matches_six_params` | encoder attn glob matches six params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_glob_matches_six_params` | encoder ffn glob matches six params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_glob_matches_two_params` | classifier glob matches two params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_patterns_together_match_fourteen_keys` | all patterns together match fourteen keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wildcard_matches_all_fourteen` | wildcard matches all fourteen |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_glob_includes_bias_and_gain` | encoder attn glob includes bias and gain |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_glob_includes_b1_and_b2` | encoder ffn glob includes b1 and b2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_glob_includes_b` | classifier glob includes b |
| &nbsp;&nbsp;**TestDatasetFile** | *Verify the training CSV is structurally correct.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_row_count` | row count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_all_input_columns` | has all input columns |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_label_column` | has label column |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_input_values_are_numeric` | all input values are numeric |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_labels_are_valid` | all labels are valid |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_both_classes_represented` | both classes represented |
| &nbsp;&nbsp;**TestLayerCallAcceptedForCrossEntropy** | *Regression: verifier must not reject layer_call prediction for cross_entropy.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_does_not_error_on_layer_call_loss` | verifier does not error on layer call loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_accepts_layer_call_prediction_kind` | verifier accepts layer call prediction kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_old_softmax_linear_models_still_accepted` | old softmax linear models still accepted |
| **[test_p11_cut5_training_run.py](../tests/test_p11_cut5_training_run.py)** | Transformer training run: end-to-end with real CSV, loss convergence check (25 tests) |
| &nbsp;&nbsp;**TestDifferentiabilityVerifierGlob** | *TestDifferentiabilityVerifierGlob* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ok_no_errors` | ok no errors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_paths_warning_absent` | no paths warning absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fourteen_paths_verified` | fourteen paths verified |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_attn_paths_go_through_attn_block` | encoder attn paths go through attn block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_ffn_paths_go_through_ffn_block` | encoder ffn paths go through ffn block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_paths_reach_logits_directly` | classifier paths reach logits directly |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_to_node_map_correct` | layer to node map correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_paths_only_contain_graph_nodes` | all paths only contain graph nodes |
| &nbsp;&nbsp;**TestCrossEntropyOverLogits** | *TestCrossEntropyOverLogits* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_is_finite_for_list_prediction` | loss is finite for list prediction |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_not_max_sentinel_when_labels_provided` | loss not max sentinel when labels provided |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_correct_class_has_lower_loss` | correct class has lower loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_applied_probabilities_sum_to_one` | softmax applied probabilities sum to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backward_compat_int_string_target_no_labels` | backward compat int string target no labels |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backward_compat_dict_prediction` | backward compat dict prediction |
| &nbsp;&nbsp;**TestGenericTrainerOnTransformer** | *Train only classifier.* (18 scalars) for speed — O(2*18*4) = 144 forward passes.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runs_without_exception` | runs without exception |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_keys_are_classifier_params` | trainable keys are classifier params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_trace_has_three_entries` | epoch trace has three entries |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_loss_decreases_over_epochs` | train loss decreases over epochs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_final_loss_is_finite` | final loss is finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_params_changed_after_training` | params changed after training |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_best_params_present` | best params present |
| &nbsp;&nbsp;**TestBackwardCompatAfterFixes** | *TestBackwardCompatAfterFixes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cut4_verifier_still_ok` | cut4 verifier still ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cut4_diff_now_has_14_paths` | cut4 diff now has 14 paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cut4_no_paths_warning_gone` | cut4 no paths warning gone |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_p4_model_unaffected` | email agent p4 model unaffected |
| **[test_p11_cut6_evaluation_report.py](../tests/test_p11_cut6_evaluation_report.py)** | Transformer evaluation: accuracy, confusion matrix, evaluation_report.json (8 tests) |
| &nbsp;&nbsp;**TestP11Cut6EvaluationReport** | *TestP11Cut6EvaluationReport* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainer_returns_valid_parameter_sets` | trainer returns valid parameter sets |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_result_shape_matches_p4_contract` | evaluation result shape matches p4 contract |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_metadata_identifies_generic_evaluator` | backend metadata identifies generic evaluator |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_fingerprint_and_schema_are_preserved` | dataset fingerprint and schema are preserved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metrics_are_finite_and_bounded` | metrics are finite and bounded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_confusion_matrix_covers_all_rows` | confusion matrix covers all rows |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_is_json_serializable` | report is json serializable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_evaluate_writes_generic_evaluation_report` | cli evaluate writes generic evaluation report |
| &nbsp;&nbsp;**TestP11Cut7PlaygroundIntegration** | *TestP11Cut7PlaygroundIntegration* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_classifier_in_examples_dict` | transformer classifier in examples dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_classifier_in_project_example_index` | transformer classifier in project example index |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_get_prediction_kind_layer_call_for_transformer` | get prediction kind layer call for transformer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_get_prediction_kind_softmax_linear_for_fall_risk` | get prediction kind softmax linear for fall risk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_get_prediction_kind_returns_empty_on_invalid_input` | get prediction kind returns empty on invalid input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_returns_ok` | run playground training generic returns ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_has_two_epochs` | run playground training generic has two epochs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_epoch_has_losses` | run playground training generic epoch has losses |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_best_epoch_positive` | run playground training generic best epoch positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_params_best_has_id` | run playground training generic params best has id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_params_best_has_transformer_keys` | run playground training generic params best has transformer keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_has_evaluation_report` | run playground training generic has evaluation report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_backend_stdlib` | run playground training generic backend stdlib |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_playground_training_generic_has_metrics_field` | run playground training generic has metrics field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_submit_training_job_layer_call_returns_job_id` | submit training job layer call returns job id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_submit_training_job_layer_call_completes` | submit training job layer call completes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_training_unaffected` | fall risk training unaffected |
| **[test_p11_cut8_regression.py](../tests/test_p11_cut8_regression.py)** | Transformer regression: P1–P10 imports, template compatibility (19 tests) |
| &nbsp;&nbsp;**TestP11RegressionVerifier** | *P11 extended _cross_entropy_kinds; P4 softmax_linear must still pass.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_training_verifier_still_ok` | email agent training verifier still ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_training_verifier_still_ok` | fall risk training verifier still ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cross_entropy_still_accepts_softmax_linear` | cross entropy still accepts softmax linear |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_cross_entropy_still_accepts_sigmoid_linear` | binary cross entropy still accepts sigmoid linear |
| &nbsp;&nbsp;**TestP11RegressionDifferentiability** | *P11 added glob expansion and _layer_to_node_map; P4 exact-path params must still verify.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_differentiability_ok` | email agent differentiability ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_differentiability_ok` | fall risk differentiability ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_parameter_paths_still_verified` | email agent parameter paths still verified |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_spurious_warning_for_p4_exact_paths` | no spurious warning for p4 exact paths |
| &nbsp;&nbsp;**TestP11RegressionSupervisedTrainer** | *P11 added GenericSupervisedTrainer; SupervisedTrainer must be unchanged.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervised_trainer_runs_without_error` | supervised trainer runs without error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervised_trainer_has_best_epoch` | supervised trainer has best epoch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervised_evaluator_produces_accuracy` | supervised evaluator produces accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generic_cross_entropy_loss_without_labels_unchanged` | _generic_cross_entropy_loss(labels=None) must behave as before P11 for dict predictions. |
| &nbsp;&nbsp;**TestP11RegressionTorchBackend** | *P11 added layer_call support to torch; P4/P5 softmax_linear forward must be unchanged.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_runner_ok_for_email_agent` | torch forward runner ok for email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_contract_ok_for_email_agent` | torch backend contract ok for email agent |
| &nbsp;&nbsp;**TestP11RegressionPlaygroundRouting** | *P11 added layer_call routing; P4/P5 models must still take the SupervisedTrainer path.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_get_prediction_kind_email_agent_not_layer_call` | get prediction kind email agent not layer call |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_get_prediction_kind_fall_risk_not_layer_call` | get prediction kind fall risk not layer call |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_playground_training_uses_supervised_path` | Fall-risk goes through SupervisedTrainer (not GenericSupervisedTrainer). |
| &nbsp;&nbsp;**TestP11RegressionP10LayerPassthrough** | *P11 added _layer_exec_<Name>; P10 LAYERs without body ops must still use passthrough.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p10_layer_without_body_ops_compiles_with_passthrough` | p10 layer without body ops compiles with passthrough |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p10_layer_with_body_ops_still_generates_executor` | p10 layer with body ops still generates executor |
| **[test_p11_5_cut1_liveness.py](../tests/test_p11_5_cut1_liveness.py)** | Liveness analysis: dead parameter detection in hierarchical LAYER graphs (3 tests) |
| &nbsp;&nbsp;**TestLayerLivenessAnalysis** | *TestLayerLivenessAnalysis* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_params_used` | all params used |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_completely_dead_param` | completely dead param |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_used_in_disconnected_branch` | used in disconnected branch |
| **[test_p11_5_cut2_sequence.py](../tests/test_p11_5_cut2_sequence.py)** | Sequence hardening: SEQUENCE block, embedding_lookup + mean_pooling rewrite (41 tests) |
| &nbsp;&nbsp;**TestSequenceSpec** | *TestSequenceSpec* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_template_parses` | template parses |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequences_list_has_one_entry` | sequences list has one entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_name` | sequence name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_length` | sequence length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_vocab_size` | sequence vocab size |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_vectors_list_is_empty` | vectors list is empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_node_type_sequence` | graph node type sequence |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_nodes_order` | graph nodes order |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_embed_layer_exists` | encoder embed layer exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_embed_has_embed_table_param` | encoder embed has embed table param |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_embed_has_embedding_lookup_op` | encoder embed has embedding lookup op |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_encoder_embed_has_mean_pooling_op` | encoder embed has mean pooling op |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_spec_to_dict` | sequence spec to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_spec_not_in_vectors_in_dict` | sequence spec not in vectors in dict |
| &nbsp;&nbsp;**TestBackendContractWithSequence** | *TestBackendContractWithSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_contract_ok` | contract ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_node_is_supported` | sequence node is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_param_count_includes_embed_table` | param count includes embed table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_contract_ok` | torch contract ok |
| &nbsp;&nbsp;**TestDifferentiablePythonForwardWithSequence** | *TestDifferentiablePythonForwardWithSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_state_has_logits` | state has logits |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_have_two_elements` | logits have two elements |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_are_finite` | logits are finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_state_has_embedded` | state has embedded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedded_has_size_8` | embedded has size 8 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_node_in_state` | sequence node in state |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_input_stored_as_integers` | sequence input stored as integers |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_tokens_produce_different_logits` | different tokens produce different logits |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_forward_is_deterministic` | forward is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_has_sequence_node` | trace has sequence node |
| &nbsp;&nbsp;**TestTorchForwardWithSequence** | *TestTorchForwardWithSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_are_finite` | logits are finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_logits_have_two_elements` | logits have two elements |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trace_has_sequence_node_type` | trace has sequence node type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_both_backends_agree_on_logit_shape` | both backends agree on logit shape |
| &nbsp;&nbsp;**TestEmbeddingLookupAndMeanPooling** | *TestEmbeddingLookupAndMeanPooling* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_2d_table` | embedding lookup 2d table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mean_pooling_2d_averages_rows` | mean pooling 2d averages rows |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_parser_with_inline_definition` | sequence parser with inline definition |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_parser_requires_length` | sequence parser requires length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sequence_parser_requires_vocab_size` | sequence parser requires vocab size |
| &nbsp;&nbsp;**TestTrainingCsvFormat** | *TestTrainingCsvFormat* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_has_token_columns` | csv has token columns |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_has_label_column` | csv has label column |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_values_are_integers` | csv values are integers |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_spec_references_sequence_columns` | train spec references sequence columns |
| **[test_p11_5_cut3_attention.py](../tests/test_p11_5_cut3_attention.py)** | Attention hardening: explicit dot/scale/softmax decomposition, gradient flow (11 tests) |
| &nbsp;&nbsp;**TestAttentionUnrolling** | *TestAttentionUnrolling* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_attention_op_is_removed` | attention op is removed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unrolled_ops_are_present` | unrolled ops are present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_contract_accepts_unrolled_attention` | backend contract accepts unrolled attention |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_contract_accepts_unrolled_attention` | torch backend contract accepts unrolled attention |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_differentiable_python_forward_is_finite` | differentiable python forward is finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_is_finite` | torch forward is finite |
| &nbsp;&nbsp;**TestScaleSoftmaxInContract** | *Verify that scale and softmax are accepted as top-level FUNCTION kinds* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_scale_accepted_by_dp_backend_contract` | scale accepted by dp backend contract |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_accepted_by_dp_backend_contract` | softmax accepted by dp backend contract |
| &nbsp;&nbsp;**TestAttentionDegeneracy** | *Documents the known architectural limitation of encoder_attn with pooled 1D input.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_of_scalar_is_one_differentiable_python` | softmax of scalar is one differentiable python |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_of_scalar_is_one_torch` | softmax of scalar is one torch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wq_wk_have_no_effect_on_forward_output` | Changing Wq and Wk by 100× must not alter the model output. |
| **[test_nightly_transformer.py](../tests/test_nightly_transformer.py)** | Nightly dense gradient: all weights produce non-zero gradients after one step (4 tests) |
| &nbsp;&nbsp;**TestNightlyDenseGradient** | *TestNightlyDenseGradient* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_completed` | training completed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_parameters_were_updated` | all parameters were updated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_is_finite` | loss is finite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensors_moved` | tensors moved |

## Regression (P17) (158 tests)

| Test | Description |
|------|-------------|
| **[test_p17_cut1_ir.py](../tests/test_p17_cut1_ir.py)** | Regression IR: linear_regression kind, Scalar output type, .mxtrain LOSS mse (22 tests) |
| &nbsp;&nbsp;**TestParserLinearRegression** | *TestParserLinearRegression* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_linear_produces_linear_regression_kind` | parse linear produces linear regression kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_linear_inputs_correct` | parse linear inputs correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_linear_parameters_correct` | parse linear parameters correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_linear_output_name` | parse linear output name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_linear_multi_feature` | parse linear multi feature |
| &nbsp;&nbsp;**TestSemanticKindOutputType** | *TestSemanticKindOutputType* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_regression_returns_scalar` | linear regression returns scalar |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_linear_still_probability` | sigmoid linear still probability |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_linear_still_probability_map` | softmax linear still probability map |
| &nbsp;&nbsp;**TestBackendContractLinearRegression** | *TestBackendContractLinearRegression* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_contract_ok` | backend contract ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_parameters_found` | trainable parameters found |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_weights_shape_matches_input_dim` | weights shape matches input dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bias_shape_is_scalar` | bias shape is scalar |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_weights_role` | weights role |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bias_role` | bias role |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_function_output_shape_scalar` | function output shape scalar |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multi_feature_weights_shape` | multi feature weights shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_function_node_differentiable` | function node differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_backend_gates_regression_for_p17_1` | torch backend gates regression for p17 1 |
| &nbsp;&nbsp;**TestDifferentiablePythonCompiler** | *TestDifferentiablePythonCompiler* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_compile_succeeds` | compile succeeds |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_with_known_weights_computes_kelvin` | run with known weights computes kelvin |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_100_celsius` | run 100 celsius |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_negative_celsius` | run negative celsius |
| **[test_p17_cut2_verifier.py](../tests/test_p17_cut2_verifier.py)** | Regression verifier: Scalar target, mse loss compatibility, verifier checks (17 tests) |
| &nbsp;&nbsp;**TestVerifierMSEAccepts** | *TestVerifierMSEAccepts* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_kelvin_mxtrain_verifies_ok` | kelvin mxtrain verifies ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_kelvin_trainable_parameters_found` | kelvin trainable parameters found |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_kelvin_differentiability_ok` | kelvin differentiability ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mae_metric_accepted_for_mse` | mae metric accepted for mse |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_errors_in_dataset_validation` | no errors in dataset validation |
| &nbsp;&nbsp;**TestVerifierMSERejects** | *TestVerifierMSERejects* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_with_label_target_rejected` | mse with label target rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_with_accuracy_metric_rejected` | mse with accuracy metric rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_with_rmse_metric_accepted` | mse with rmse metric accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_with_r2_metric_accepted` | mse with r2 metric accepted |
| &nbsp;&nbsp;**TestVerifierClassificationRejectsScalarTarget** | *TestVerifierClassificationRejectsScalarTarget* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cross_entropy_with_scalar_target_rejected` | cross entropy with scalar target rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_cross_entropy_with_scalar_target_rejected` | binary cross entropy with scalar target rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_with_regression_metric_rejected` | classification with regression metric rejected |
| &nbsp;&nbsp;**TestVerifierDatasetContinuousTarget** | *TestVerifierDatasetContinuousTarget* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_numeric_target_accepted` | dataset numeric target accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_non_numeric_target_rejected` | dataset non numeric target rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_out_of_range_target_rejected` | dataset out of range target rejected |
| &nbsp;&nbsp;**TestEvaluationResultRegression** | *TestEvaluationResultRegression* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_is_regression_when_no_labels` | is regression when no labels |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_is_not_regression_when_labels` | is not regression when labels |
| **[test_p17_cut3_trainer.py](../tests/test_p17_cut3_trainer.py)** | Regression trainer: MSE loss, mae/rmse/r2 metrics, SGD convergence (26 tests) |
| &nbsp;&nbsp;**TestExpectedSemanticKind** | *TestExpectedSemanticKind* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_returns_linear_regression` | mse returns linear regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cross_entropy_still_softmax` | cross entropy still softmax |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_cross_entropy_still_sigmoid` | binary cross entropy still sigmoid |
| &nbsp;&nbsp;**TestLabelsForMSE** | *TestLabelsForMSE* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_labels_empty_for_mse` | labels empty for mse |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_labels_non_empty_for_classification` | labels non empty for classification |
| &nbsp;&nbsp;**TestMSERegressionGradients** | *TestMSERegressionGradients* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_direction_weight` | gradient direction weight |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_direction_bias` | gradient direction bias |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_values_exact` | gradient values exact |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_batch_averaged` | gradient batch averaged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_zero_error` | gradient zero error |
| &nbsp;&nbsp;**TestMSERegressionMetrics** | *TestMSERegressionMetrics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_perfect_prediction` | mse perfect prediction |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_known_error` | mse known error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_r2_below_one_for_imperfect` | r2 below one for imperfect |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_confusion_matrix` | no confusion matrix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accuracy_zero` | accuracy zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mae_is_mean_abs_error` | mae is mean abs error |
| &nbsp;&nbsp;**TestSupervisedTrainerRegressionIntegration** | *TestSupervisedTrainerRegressionIntegration* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_runs_without_error` | train runs without error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_task_kind_regression_in_trace` | task kind regression in trace |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mae_in_metrics_json` | mae in metrics json |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_canonical_celsius_example_converges` | canonical celsius example converges |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_regression_is_explicitly_gated` | torch regression is explicitly gated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_decreases_with_training` | loss decreases with training |
| &nbsp;&nbsp;**TestSupervisedEvaluatorRegression** | *TestSupervisedEvaluatorRegression* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluator_returns_mae_rmse_r2` | evaluator returns mae rmse r2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluator_perfect_weights_gives_low_mae` | evaluator perfect weights gives low mae |
| &nbsp;&nbsp;**TestSupervisedExampleTargetValue** | *TestSupervisedExampleTargetValue* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_adapter_sets_target_value_for_regression` | csv adapter sets target value for regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_csv_adapter_no_target_value_for_classification` | csv adapter no target value for classification |
| **[test_p17_cut4_generator.py](../tests/test_p17_cut4_generator.py)** | Regression training generator: generate-training with MODE regression (16 tests) |
| &nbsp;&nbsp;**TestGeneratorRegressionBasic** | *TestGeneratorRegressionBasic* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_mse_loss` | generates mse loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_labels_empty_for_regression` | labels empty for regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_target_defaults_to_function_output` | target defaults to function output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_target_name_override` | target name override |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_scalar_target_no_range` | generates scalar target no range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_scalar_target_with_range` | generates scalar target with range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_mae_metric_not_accuracy` | generates mae metric not accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_label_bracket_in_target` | no label bracket in target |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assumptions_mention_linear_regression` | assumptions mention linear regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epochs_default_fifty` | epochs default fifty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_learning_rate_default` | learning rate default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_template_numeric_target` | dataset template numeric target |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_template_uses_scalar_range` | dataset template uses scalar range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prediction_matches_function_output` | prediction matches function output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generated_mxtrain_is_parseable_and_verifiable` | generated mxtrain is parseable and verifiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_still_works` | classification still works |
| **[test_p17_cut5_synthetic.py](../tests/test_p17_cut5_synthetic.py)** | Regression synthetic data: coherent mode generating continuous targets (16 tests) |
| &nbsp;&nbsp;**TestSyntheticRegressionRandom** | *TestSyntheticRegressionRandom* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_returns_adapter` | random returns adapter |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_labels_empty` | random labels empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_correct_row_count` | random correct row count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_target_is_numeric` | random target is numeric |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_target_within_scalar_range` | random target within scalar range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_target_value_field_set` | random target value field set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_input_vector_present` | random input vector present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_reproducible_with_same_seed` | random reproducible with same seed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_random_different_with_different_seed` | random different with different seed |
| &nbsp;&nbsp;**TestSyntheticRegressionCoherent** | *TestSyntheticRegressionCoherent* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_returns_adapter` | coherent returns adapter |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_labels_empty` | coherent labels empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_target_is_numeric` | coherent target is numeric |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_coherent_target_value_is_float` | coherent target value is float |
| &nbsp;&nbsp;**TestSyntheticRegressionRangeHelper** | *TestSyntheticRegressionRangeHelper* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_range_uses_scalar_bounds` | regression range uses scalar bounds |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_range_defaults_when_no_range` | regression range defaults when no range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_spec_still_raises_without_labels` | classification spec still raises without labels |
| **[test_p17_cut6_prompt_agent.py](../tests/test_p17_cut6_prompt_agent.py)** | Regression prompt agent: MODE regression, OUTPUT, LOSS, METRIC in .semantic (27 tests) |
| &nbsp;&nbsp;**TestPromptAgentRegressionTemplateSelection** | *TestPromptAgentRegressionTemplateSelection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_precio_selects_regression` | precio selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_price_selects_regression` | price selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predecir_selects_regression` | predecir selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_celsius_selects_regression` | celsius selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_kelvin_selects_regression` | kelvin selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_estim_selects_regression` | estim selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regres_selects_regression` | regres selects regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classify_still_selects_classification` | classify still selects classification |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_still_selects_fall_risk` | fall risk still selects fall risk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_not_regression` | email not regression |
| &nbsp;&nbsp;**TestPromptAgentRegressionOutput** | *TestPromptAgentRegressionOutput* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mode_is_regression` | mode is regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_semantic_text_has_mode_regression` | semantic text has mode regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_semantic_text_has_mse_loss` | semantic text has mse loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_semantic_text_has_mae_metric` | semantic text has mae metric |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_semantic_text_has_output_scalar` | semantic text has output scalar |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_semantic_text_no_action_simulate_only` | semantic text no action simulate only |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_extracted_rules_empty_for_regression` | extracted rules empty for regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assumptions_mention_trainable_package` | assumptions mention trainable package |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_assumptions_mention_mse` | assumptions mention mse |
| &nbsp;&nbsp;**TestPromptAgentExactFormulaDistinction** | *TestPromptAgentExactFormulaDistinction* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_celsius_kelvin_notes_deterministic_formula` | celsius kelvin notes deterministic formula |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_price_prediction_no_formula_note` | price prediction no formula note |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_convertir_notes_formula` | convertir notes formula |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_centigrados_kelvin_uses_auditable_domain_names` | centigrados kelvin uses auditable domain names |
| &nbsp;&nbsp;**TestPromptRegressionEndToEnd** | *TestPromptRegressionEndToEnd* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_supervisor_accepts_deterministic_regression_semantic` | prompt supervisor accepts deterministic regression semantic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_regression_generates_linear_mxai` | prompt regression generates linear mxai |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_regression_mxai_is_trainable_shape` | prompt regression mxai is trainable shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_supervise_prompt_falls_back_when_llm_classifies_continuous_prompt` | supervise prompt falls back when llm classifies continuous prompt |
| &nbsp;&nbsp;**TestPlaygroundRegressionPromptPackage** | *TestPlaygroundRegressionPromptPackage* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_generates_celsius_project` | prompt generates celsius project |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_generates_mxtrain` | prompt generates mxtrain |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_generates_dataset_template` | prompt generates dataset template |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_step_no_longer_pending` | training step no longer pending |
| &nbsp;&nbsp;**TestPlaygroundRegressionTrainingShape** | *TestPlaygroundRegressionTrainingShape* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_ok` | training ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_task_kind_is_regression` | task kind is regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_has_mae_key` | result has mae key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_has_rmse_key` | result has rmse key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_has_r2_key` | result has r2 key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mae_is_float_or_none` | mae is float or none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rmse_is_float_or_none` | rmse is float or none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_r2_is_float_or_none` | r2 is float or none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_best_epoch_positive` | best epoch positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epochs_list_populated` | epochs list populated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_params_best_populated` | params best populated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_is_stdlib` | backend is stdlib |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_has_task_kind_key` | result has task kind key |
| &nbsp;&nbsp;**TestPlaygroundRegressionEvaluationReport** | *TestPlaygroundRegressionEvaluationReport* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_present` | evaluation report present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_labels_empty` | evaluation report labels empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_has_mae` | evaluation report has mae |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_has_rmse` | evaluation report has rmse |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_has_r2` | evaluation report has r2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_mae_is_float` | evaluation report mae is float |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_report_rows_positive` | evaluation report rows positive |
| &nbsp;&nbsp;**TestPlaygroundEvaluationArtifactsRegression** | *TestPlaygroundEvaluationArtifactsRegression* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_artifacts_available_for_regression_report` | evaluation artifacts available for regression report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_artifacts_ok_for_regression_report` | evaluation artifacts ok for regression report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_artifacts_no_missing_fields_regression` | evaluation artifacts no missing fields regression |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_artifacts_report_has_mae` | evaluation artifacts report has mae |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_artifacts_report_has_rmse` | evaluation artifacts report has rmse |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluation_artifacts_report_has_r2` | evaluation artifacts report has r2 |
| &nbsp;&nbsp;**TestPlaygroundClassificationUnaffected** | *Ensure classification training still works and has correct task_kind.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_training_ok` | classification training ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_task_kind` | classification task kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_mae_is_none` | classification mae is none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prompt_generated_regression_uses_unavailable_confidence_and_scalar_category` | prompt generated regression uses unavailable confidence and scalar category |

## Dense Neural Networks (P18) (326 tests)

| Test | Description |
|------|-------------|
| **[test_p18_cut1_network_parser.py](../tests/test_p18_cut1_network_parser.py)** | NETWORK block parser: LAYER Dense declarations, activations, OUTPUT type, IR (19 tests) |
| &nbsp;&nbsp;`test_network_parser_accepts_dense_regression_network` | network parser accepts dense regression network |
| &nbsp;&nbsp;`test_network_parser_accepts_dense_classification_network` | network parser accepts dense classification network |
| &nbsp;&nbsp;`test_network_parser_accepts_dense_binary_network` | network parser accepts dense binary network |
| &nbsp;&nbsp;`test_network_parser_rejects_unknown_layer_type` | network parser rejects unknown layer type |
| &nbsp;&nbsp;`test_network_parser_rejects_unknown_activation` | network parser rejects unknown activation |
| &nbsp;&nbsp;`test_network_parser_rejects_missing_input` | network parser rejects missing input |
| &nbsp;&nbsp;`test_network_parser_rejects_missing_output` | network parser rejects missing output |
| &nbsp;&nbsp;`test_network_parser_rejects_missing_layers` | network parser rejects missing layers |
| &nbsp;&nbsp;`test_network_parser_rejects_zero_units` | network parser rejects zero units |
| &nbsp;&nbsp;`test_network_parser_rejects_multiple_inputs` | network parser rejects multiple inputs |
| &nbsp;&nbsp;`test_network_ir_has_correct_layer_count` | network ir has correct layer count |
| &nbsp;&nbsp;`test_network_ir_layer_indices_are_sequential` | network ir layer indices are sequential |
| &nbsp;&nbsp;`test_network_ir_layer_units_and_activations` | network ir layer units and activations |
| &nbsp;&nbsp;`test_network_ir_output_name_and_type` | network ir output name and type |
| &nbsp;&nbsp;`test_network_program_contains_networks_list` | network program contains networks list |
| &nbsp;&nbsp;`test_network_graph_node_type_is_dense_network` | network graph node type is dense network |
| &nbsp;&nbsp;`test_network_to_dict_serializes_layers` | network to dict serializes layers |
| &nbsp;&nbsp;`test_network_parser_accepts_tanh_activation` | network parser accepts tanh activation |
| &nbsp;&nbsp;`test_network_dense_layer_spec_type` | network dense layer spec type |
| **[test_p18_cut2_network_types.py](../tests/test_p18_cut2_network_types.py)** | Dense network types: shape inference per layer, type constraints (20 tests) |
| &nbsp;&nbsp;`test_dense_network_shape_inference_regression` | dense network shape inference regression |
| &nbsp;&nbsp;`test_dense_network_shape_inference_multiclass` | dense network shape inference multiclass |
| &nbsp;&nbsp;`test_dense_network_shape_inference_single_hidden_layer` | dense network shape inference single hidden layer |
| &nbsp;&nbsp;`test_dense_network_output_type_scalar_from_linear` | dense network output type scalar from linear |
| &nbsp;&nbsp;`test_dense_network_output_type_probability_from_sigmoid` | dense network output type probability from sigmoid |
| &nbsp;&nbsp;`test_dense_network_output_type_probability_map_from_softmax` | dense network output type probability map from softmax |
| &nbsp;&nbsp;`test_dense_network_softmax_requires_probability_map` | dense network softmax requires probability map |
| &nbsp;&nbsp;`test_dense_network_sigmoid_requires_probability_output` | dense network sigmoid requires probability output |
| &nbsp;&nbsp;`test_dense_network_linear_scalar_output_requires_one_unit` | dense network linear scalar output requires one unit |
| &nbsp;&nbsp;`test_dense_network_softmax_requires_min_2_units` | dense network softmax requires min 2 units |
| &nbsp;&nbsp;`test_dense_network_relu_final_rejected_for_probability` | dense network relu final rejected for probability |
| &nbsp;&nbsp;`test_dense_network_relu_final_rejected_for_probability_map` | dense network relu final rejected for probability map |
| &nbsp;&nbsp;`test_dense_network_rejects_non_numeric_vector_fields` | dense network rejects non numeric vector fields |
| &nbsp;&nbsp;`test_dense_network_rejects_unknown_input_vector` | dense network rejects unknown input vector |
| &nbsp;&nbsp;`test_dense_network_relu_hidden_layer_accepted` | dense network relu hidden layer accepted |
| &nbsp;&nbsp;`test_dense_network_valid_regression_produces_interpretability_warning` | dense network valid regression produces interpretability warning |
| &nbsp;&nbsp;`test_dense_network_check_integrated_in_program_typecheck` | dense network check integrated in program typecheck |
| &nbsp;&nbsp;`test_dense_network_valid_regression_has_no_errors` | dense network valid regression has no errors |
| &nbsp;&nbsp;`test_dense_network_valid_multiclass_has_no_errors` | dense network valid multiclass has no errors |
| &nbsp;&nbsp;`test_dense_network_valid_binary_has_no_errors` | dense network valid binary has no errors |
| **[test_p18_cut3_parameter_manifest.py](../tests/test_p18_cut3_parameter_manifest.py)** | Dense parameter manifest: He/Xavier initialisation, W/b naming per layer (21 tests) |
| &nbsp;&nbsp;`test_network_parameter_manifest_has_correct_entry_count` | network parameter manifest has correct entry count |
| &nbsp;&nbsp;`test_network_parameter_manifest_contains_all_layers` | network parameter manifest contains all layers |
| &nbsp;&nbsp;`test_network_parameter_manifest_weights_shape` | network parameter manifest weights shape |
| &nbsp;&nbsp;`test_network_parameter_manifest_bias_shape` | network parameter manifest bias shape |
| &nbsp;&nbsp;`test_network_parameter_manifest_qualified_names` | network parameter manifest qualified names |
| &nbsp;&nbsp;`test_network_parameter_manifest_relu_uses_he_initializer` | network parameter manifest relu uses he initializer |
| &nbsp;&nbsp;`test_network_parameter_manifest_sigmoid_uses_xavier_initializer` | network parameter manifest sigmoid uses xavier initializer |
| &nbsp;&nbsp;`test_network_parameter_manifest_bias_always_zeros` | network parameter manifest bias always zeros |
| &nbsp;&nbsp;`test_network_parameter_schema_hash_changes_when_units_change` | network parameter schema hash changes when units change |
| &nbsp;&nbsp;`test_network_parameter_schema_hash_changes_when_activation_changes` | network parameter schema hash changes when activation changes |
| &nbsp;&nbsp;`test_network_parameter_schema_hash_is_stable` | network parameter schema hash is stable |
| &nbsp;&nbsp;`test_build_network_parameter_set_structure` | build network parameter set structure |
| &nbsp;&nbsp;`test_build_network_parameter_set_weights_not_zero_for_he` | build network parameter set weights not zero for he |
| &nbsp;&nbsp;`test_build_network_parameter_set_bias_is_zeros` | build network parameter set bias is zeros |
| &nbsp;&nbsp;`test_build_network_parameter_set_weight_shape_correct` | build network parameter set weight shape correct |
| &nbsp;&nbsp;`test_build_network_parameter_set_is_deterministic` | build network parameter set is deterministic |
| &nbsp;&nbsp;`test_build_network_parameter_set_roundtrip` | build network parameter set roundtrip |
| &nbsp;&nbsp;`test_validate_network_parameter_set_accepts_valid` | validate network parameter set accepts valid |
| &nbsp;&nbsp;`test_validate_network_parameter_set_rejects_missing_layer_weight` | validate network parameter set rejects missing layer weight |
| &nbsp;&nbsp;`test_validate_network_parameter_set_rejects_wrong_shape` | validate network parameter set rejects wrong shape |
| &nbsp;&nbsp;`test_validate_network_parameter_set_rejects_wrong_model_hash` | validate network parameter set rejects wrong model hash |
| **[test_p18_cut4_backend_contract.py](../tests/test_p18_cut4_backend_contract.py)** | Dense backend contract: BackendContractAnalyzer for NETWORK nodes (25 tests) |
| &nbsp;&nbsp;`test_dense_network_node_is_supported` | dense network node is supported |
| &nbsp;&nbsp;`test_dense_network_node_is_differentiable` | dense network node is differentiable |
| &nbsp;&nbsp;`test_dense_network_node_type` | dense network node type |
| &nbsp;&nbsp;`test_dense_network_kind` | dense network kind |
| &nbsp;&nbsp;`test_dense_network_output_shape_scalar` | dense network output shape scalar |
| &nbsp;&nbsp;`test_dense_network_output_shape_multiclass` | dense network output shape multiclass |
| &nbsp;&nbsp;`test_dense_network_appears_in_differentiable_nodes` | dense network appears in differentiable nodes |
| &nbsp;&nbsp;`test_dense_network_generates_w_and_b_per_layer` | dense network generates w and b per layer |
| &nbsp;&nbsp;`test_dense_network_param_count` | dense network param count |
| &nbsp;&nbsp;`test_dense_network_weight_roles` | dense network weight roles |
| &nbsp;&nbsp;`test_dense_network_weight_shapes` | dense network weight shapes |
| &nbsp;&nbsp;`test_dense_network_bias_shapes` | dense network bias shapes |
| &nbsp;&nbsp;`test_dense_network_param_paths` | dense network param paths |
| &nbsp;&nbsp;`test_dense_network_param_function_field` | dense network param function field |
| &nbsp;&nbsp;`test_dense_network_emits_interpretability_warning` | dense network emits interpretability warning |
| &nbsp;&nbsp;`test_dense_network_interpretability_warning_mentions_network` | dense network interpretability warning mentions network |
| &nbsp;&nbsp;`test_dense_network_in_layer_manifest` | dense network in layer manifest |
| &nbsp;&nbsp;`test_dense_network_layer_manifest_entry_count` | dense network layer manifest entry count |
| &nbsp;&nbsp;`test_dense_network_layer_manifest_param_shapes` | dense network layer manifest param shapes |
| &nbsp;&nbsp;`test_dense_network_layer_manifest_relu_initializer` | dense network layer manifest relu initializer |
| &nbsp;&nbsp;`test_dense_network_layer_manifest_bias_zeros` | dense network layer manifest bias zeros |
| &nbsp;&nbsp;`test_report_ok_with_dense_network_only` | report ok with dense network only |
| &nbsp;&nbsp;`test_two_dense_networks_both_supported` | two dense networks both supported |
| &nbsp;&nbsp;`test_two_dense_networks_param_count` | two dense networks param count |
| &nbsp;&nbsp;`test_dense_network_to_dict_structure` | dense network to dict structure |
| **[test_p18_cut5_dense_forward.py](../tests/test_p18_cut5_dense_forward.py)** | Dense forward pass: stdlib multi-layer forward with activations (22 tests) |
| &nbsp;&nbsp;`test_dense_forward_output_is_list` | dense forward output is list |
| &nbsp;&nbsp;`test_dense_forward_output_shape_matches_last_layer` | dense forward output shape matches last layer |
| &nbsp;&nbsp;`test_dense_forward_scalar_output_length_one` | dense forward scalar output length one |
| &nbsp;&nbsp;`test_dense_forward_relu_non_negative` | dense forward relu non negative |
| &nbsp;&nbsp;`test_dense_forward_sigmoid_in_range` | dense forward sigmoid in range |
| &nbsp;&nbsp;`test_dense_forward_softmax_sums_to_one` | dense forward softmax sums to one |
| &nbsp;&nbsp;`test_dense_forward_softmax_all_positive` | dense forward softmax all positive |
| &nbsp;&nbsp;`test_dense_forward_tanh_in_range` | dense forward tanh in range |
| &nbsp;&nbsp;`test_dense_forward_linear_is_identity_for_zero_weights` | dense forward linear is identity for zero weights |
| &nbsp;&nbsp;`test_dense_forward_known_values_single_layer` | dense forward known values single layer |
| &nbsp;&nbsp;`test_dense_forward_relu_clips_negative` | dense forward relu clips negative |
| &nbsp;&nbsp;`test_dense_forward_trace_returns_trace_object` | dense forward trace returns trace object |
| &nbsp;&nbsp;`test_dense_forward_trace_activations_length` | dense forward trace activations length |
| &nbsp;&nbsp;`test_dense_forward_trace_pre_activations_length` | dense forward trace pre activations length |
| &nbsp;&nbsp;`test_dense_forward_trace_input_preserved` | dense forward trace input preserved |
| &nbsp;&nbsp;`test_dense_forward_trace_output_matches_dense_forward` | dense forward trace output matches dense forward |
| &nbsp;&nbsp;`test_dense_forward_is_deterministic` | dense forward is deterministic |
| &nbsp;&nbsp;`test_dense_forward_wrong_input_dim_raises` | dense forward wrong input dim raises |
| &nbsp;&nbsp;`test_dense_forward_missing_param_raises` | dense forward missing param raises |
| &nbsp;&nbsp;`test_dense_forward_sigmoid_large_positive` | dense forward sigmoid large positive |
| &nbsp;&nbsp;`test_dense_forward_sigmoid_large_negative` | dense forward sigmoid large negative |
| &nbsp;&nbsp;`test_dense_forward_softmax_uniform_for_equal_inputs` | dense forward softmax uniform for equal inputs |
| **[test_p18_cut6_backprop.py](../tests/test_p18_cut6_backprop.py)** | Dense backprop: SGD gradient computation for W/b across layers (26 tests) |
| &nbsp;&nbsp;`test_mse_loss_zero_for_perfect_prediction` | mse loss zero for perfect prediction |
| &nbsp;&nbsp;`test_mse_loss_correct_value` | mse loss correct value |
| &nbsp;&nbsp;`test_binary_cross_entropy_loss_perfect_prediction` | binary cross entropy loss perfect prediction |
| &nbsp;&nbsp;`test_binary_cross_entropy_loss_bad_prediction` | binary cross entropy loss bad prediction |
| &nbsp;&nbsp;`test_cross_entropy_loss_perfect_onehot` | cross entropy loss perfect onehot |
| &nbsp;&nbsp;`test_compute_loss_dispatches_mse` | compute loss dispatches mse |
| &nbsp;&nbsp;`test_compute_loss_unknown_raises` | compute loss unknown raises |
| &nbsp;&nbsp;`test_gradients_have_all_param_keys` | gradients have all param keys |
| &nbsp;&nbsp;`test_gradient_dW1_shape` | gradient dW1 shape |
| &nbsp;&nbsp;`test_gradient_db1_shape` | gradient db1 shape |
| &nbsp;&nbsp;`test_train_step_returns_tuple` | train step returns tuple |
| &nbsp;&nbsp;`test_train_step_loss_is_float` | train step loss is float |
| &nbsp;&nbsp;`test_train_step_returns_parameter_set` | train step returns parameter set |
| &nbsp;&nbsp;`test_train_step_preserves_schema_hash` | train step preserves schema hash |
| &nbsp;&nbsp;`test_train_step_parameters_change` | train step parameters change |
| &nbsp;&nbsp;`test_train_step_source_is_trained` | train step source is trained |
| &nbsp;&nbsp;`test_loss_decreases_after_one_mse_step` | loss decreases after one mse step |
| &nbsp;&nbsp;`test_loss_decreases_after_one_cross_entropy_step` | loss decreases after one cross entropy step |
| &nbsp;&nbsp;`test_loss_decreases_after_one_bce_step` | loss decreases after one bce step |
| &nbsp;&nbsp;`test_mse_gradient_known_values_linear` | mse gradient known values linear |
| &nbsp;&nbsp;`test_sgd_moves_weight_toward_target` | sgd moves weight toward target |
| &nbsp;&nbsp;`test_softmax_cross_entropy_fused_gradient` | softmax cross entropy fused gradient |
| &nbsp;&nbsp;`test_sigmoid_bce_fused_gradient` | sigmoid bce fused gradient |
| &nbsp;&nbsp;`test_training_converges_mse_regression` | training converges mse regression |
| &nbsp;&nbsp;`test_training_converges_binary_classification` | training converges binary classification |
| &nbsp;&nbsp;`test_training_converges_multiclass` | training converges multiclass |
| **[test_p18_cut7_training_verifier.py](../tests/test_p18_cut7_training_verifier.py)** | Dense training verifier: TrainingVerifier accepts NETWORK, UPDATE rules (19 tests) |
| &nbsp;&nbsp;`test_symbol_type_finds_scalar_for_network_output` | symbol type finds scalar for network output |
| &nbsp;&nbsp;`test_symbol_type_finds_probabilitymap_for_softmax_network` | symbol type finds probabilitymap for softmax network |
| &nbsp;&nbsp;`test_symbol_type_finds_probability_for_sigmoid_network` | symbol type finds probability for sigmoid network |
| &nbsp;&nbsp;`test_symbol_type_returns_none_for_unknown` | symbol type returns none for unknown |
| &nbsp;&nbsp;`test_symbol_type_also_matches_by_network_name` | symbol type also matches by network name |
| &nbsp;&nbsp;`test_prediction_node_finds_regression_network` | prediction node finds regression network |
| &nbsp;&nbsp;`test_prediction_node_finds_multiclass_network` | prediction node finds multiclass network |
| &nbsp;&nbsp;`test_prediction_node_also_matches_network_name` | prediction node also matches network name |
| &nbsp;&nbsp;`test_prediction_node_returns_empty_for_unknown` | prediction node returns empty for unknown |
| &nbsp;&nbsp;`test_differentiability_verifier_no_errors_for_regression_network` | differentiability verifier no errors for regression network |
| &nbsp;&nbsp;`test_differentiability_verifier_prediction_node_is_network` | differentiability verifier prediction node is network |
| &nbsp;&nbsp;`test_differentiability_verifier_parameter_paths_populated` | differentiability verifier parameter paths populated |
| &nbsp;&nbsp;`test_verifier_accepts_mse_dense_network` | verifier accepts mse dense network |
| &nbsp;&nbsp;`test_verifier_accepts_cross_entropy_dense_network` | verifier accepts cross entropy dense network |
| &nbsp;&nbsp;`test_verifier_accepts_binary_cross_entropy_dense_network` | verifier accepts binary cross entropy dense network |
| &nbsp;&nbsp;`test_verifier_reports_trainable_params_for_network` | verifier reports trainable params for network |
| &nbsp;&nbsp;`test_verifier_rejects_wrong_loss_for_scalar_network` | verifier rejects wrong loss for scalar network |
| &nbsp;&nbsp;`test_verifier_accepts_update_wildcard_for_network` | verifier accepts update wildcard for network |
| &nbsp;&nbsp;`test_verifier_rejects_invalid_update_pattern` | verifier rejects invalid update pattern |
| **[test_p18_cut8_evaluation.py](../tests/test_p18_cut8_evaluation.py)** | Dense evaluation: precision/recall/f1/macro_f1, mae/rmse/r2 metrics (22 tests) |
| &nbsp;&nbsp;`test_mae_perfect_prediction` | mae perfect prediction |
| &nbsp;&nbsp;`test_mae_correct_value` | mae correct value |
| &nbsp;&nbsp;`test_rmse_perfect_prediction` | rmse perfect prediction |
| &nbsp;&nbsp;`test_rmse_correct_value` | rmse correct value |
| &nbsp;&nbsp;`test_r2_perfect_prediction` | r2 perfect prediction |
| &nbsp;&nbsp;`test_r2_constant_baseline` | r2 constant baseline |
| &nbsp;&nbsp;`test_r2_worse_than_baseline_is_negative` | r2 worse than baseline is negative |
| &nbsp;&nbsp;`test_accuracy_binary_all_correct` | accuracy binary all correct |
| &nbsp;&nbsp;`test_accuracy_multiclass_all_correct` | accuracy multiclass all correct |
| &nbsp;&nbsp;`test_accuracy_all_wrong` | accuracy all wrong |
| &nbsp;&nbsp;`test_evaluate_regression_returns_result` | evaluate regression returns result |
| &nbsp;&nbsp;`test_evaluate_regression_row_count` | evaluate regression row count |
| &nbsp;&nbsp;`test_evaluate_regression_loss_nonneg` | evaluate regression loss nonneg |
| &nbsp;&nbsp;`test_evaluate_regression_mae_nonneg` | evaluate regression mae nonneg |
| &nbsp;&nbsp;`test_evaluate_regression_is_regression_flag` | evaluate regression is regression flag |
| &nbsp;&nbsp;`test_evaluate_regression_to_dict_has_mae` | evaluate regression to dict has mae |
| &nbsp;&nbsp;`test_evaluate_crossentropy_returns_accuracy` | evaluate crossentropy returns accuracy |
| &nbsp;&nbsp;`test_evaluate_crossentropy_not_regression` | evaluate crossentropy not regression |
| &nbsp;&nbsp;`test_evaluate_bce_returns_accuracy` | evaluate bce returns accuracy |
| &nbsp;&nbsp;`test_evaluate_crossentropy_perfect_accuracy_after_training` | evaluate crossentropy perfect accuracy after training |
| &nbsp;&nbsp;`test_evaluate_regression_mae_zero_after_convergence` | evaluate regression mae zero after convergence |
| &nbsp;&nbsp;`test_evaluate_empty_examples_raises` | evaluate empty examples raises |
| **[test_p18_cut9_torch.py](../tests/test_p18_cut9_torch.py)** | Dense Torch: nn.Linear per layer, parameter bridge, equivalence with stdlib (19 tests) |
| &nbsp;&nbsp;`test_module_created_without_error` | module created without error |
| &nbsp;&nbsp;`test_module_has_correct_linear_layer_count` | module has correct linear layer count |
| &nbsp;&nbsp;`test_module_linear_shapes` | module linear shapes |
| &nbsp;&nbsp;`test_module_weights_match_parameter_set` | module weights match parameter set |
| &nbsp;&nbsp;`test_module_bias_match_parameter_set` | module bias match parameter set |
| &nbsp;&nbsp;`test_module_missing_param_raises` | module missing param raises |
| &nbsp;&nbsp;`test_torch_forward_output_is_list` | torch forward output is list |
| &nbsp;&nbsp;`test_torch_forward_output_length` | torch forward output length |
| &nbsp;&nbsp;`test_torch_forward_matches_stdlib_forward` | torch forward matches stdlib forward |
| &nbsp;&nbsp;`test_torch_forward_softmax_sums_to_one` | torch forward softmax sums to one |
| &nbsp;&nbsp;`test_torch_forward_relu_nonneg` | torch forward relu nonneg |
| &nbsp;&nbsp;`test_torch_forward_sigmoid_in_range` | torch forward sigmoid in range |
| &nbsp;&nbsp;`test_torch_forward_tanh_in_range` | torch forward tanh in range |
| &nbsp;&nbsp;`test_torch_forward_deterministic` | torch forward deterministic |
| &nbsp;&nbsp;`test_torch_module_to_parameter_set_roundtrip` | torch module to parameter set roundtrip |
| &nbsp;&nbsp;`test_torch_module_to_parameter_set_source_is_torch` | torch module to parameter set source is torch |
| &nbsp;&nbsp;`test_torch_module_to_parameter_set_preserves_schema_hash` | torch module to parameter set preserves schema hash |
| &nbsp;&nbsp;`test_torch_module_parameters_have_grad` | torch module parameters have grad |
| &nbsp;&nbsp;`test_torch_backward_runs` | torch backward runs |
| **[test_p18_cut10_dense_generator.py](../tests/test_p18_cut10_dense_generator.py)** | DenseNetworkGenerator: prompt-driven dense architecture generation (29 tests) |
| &nbsp;&nbsp;`test_regression_intent_loss` | regression intent loss |
| &nbsp;&nbsp;`test_regression_intent_activation` | regression intent activation |
| &nbsp;&nbsp;`test_binary_intent_loss` | binary intent loss |
| &nbsp;&nbsp;`test_binary_intent_activation` | binary intent activation |
| &nbsp;&nbsp;`test_multiclass_intent_loss` | multiclass intent loss |
| &nbsp;&nbsp;`test_multiclass_intent_activation` | multiclass intent activation |
| &nbsp;&nbsp;`test_output_type_regression` | output type regression |
| &nbsp;&nbsp;`test_output_type_binary` | output type binary |
| &nbsp;&nbsp;`test_output_type_multiclass` | output type multiclass |
| &nbsp;&nbsp;`test_output_units_regression` | output units regression |
| &nbsp;&nbsp;`test_output_units_binary` | output units binary |
| &nbsp;&nbsp;`test_output_units_multiclass` | output units multiclass |
| &nbsp;&nbsp;`test_mxai_text_contains_network_block` | mxai text contains network block |
| &nbsp;&nbsp;`test_mxai_text_contains_layer_dense` | mxai text contains layer dense |
| &nbsp;&nbsp;`test_mxai_text_is_parseable` | mxai text is parseable |
| &nbsp;&nbsp;`test_mxai_text_network_has_correct_activation` | mxai text network has correct activation |
| &nbsp;&nbsp;`test_mxai_multiclass_parseable` | mxai multiclass parseable |
| &nbsp;&nbsp;`test_training_text_contains_loss_function` | training text contains loss function |
| &nbsp;&nbsp;`test_training_text_contains_update_wildcard` | training text contains update wildcard |
| &nbsp;&nbsp;`test_training_text_binary_contains_labels` | training text binary contains labels |
| &nbsp;&nbsp;`test_labels_extracted_from_prompt` | labels extracted from prompt |
| &nbsp;&nbsp;`test_input_dim_from_explicit_fields` | input dim from explicit fields |
| &nbsp;&nbsp;`test_hidden_layers_scale_with_input_dim` | hidden layers scale with input dim |
| &nbsp;&nbsp;`test_network_spec_returns_networkspec` | network spec returns networkspec |
| &nbsp;&nbsp;`test_network_spec_layer_count` | network spec layer count |
| &nbsp;&nbsp;`test_to_dict_contains_keys` | to dict contains keys |
| &nbsp;&nbsp;`test_assumptions_not_empty` | assumptions not empty |
| &nbsp;&nbsp;`test_empty_prompt_raises` | empty prompt raises |
| &nbsp;&nbsp;`test_whitespace_only_prompt_raises` | whitespace only prompt raises |
| &nbsp;&nbsp;`test_architecture_text_contains_input` | architecture text contains input |
| &nbsp;&nbsp;`test_architecture_text_contains_dense_layers` | architecture text contains dense layers |
| &nbsp;&nbsp;`test_architecture_text_contains_output_type` | architecture text contains output type |
| &nbsp;&nbsp;`test_architecture_text_arrow_separated` | architecture text arrow separated |
| &nbsp;&nbsp;`test_view_layers_count` | view layers count |
| &nbsp;&nbsp;`test_view_last_layer_is_output` | view last layer is output |
| &nbsp;&nbsp;`test_view_layer_units_correct` | view layer units correct |
| &nbsp;&nbsp;`test_view_layer_param_count` | view layer param count |
| &nbsp;&nbsp;`test_view_total_params` | view total params |
| &nbsp;&nbsp;`test_view_interpretability_level_reduced` | view interpretability level reduced |
| &nbsp;&nbsp;`test_view_interpretability_warning_not_empty` | view interpretability warning not empty |
| &nbsp;&nbsp;`test_view_loss_type_regression` | view loss type regression |
| &nbsp;&nbsp;`test_view_loss_type_binary` | view loss type binary |
| &nbsp;&nbsp;`test_view_loss_type_multiclass` | view loss type multiclass |
| &nbsp;&nbsp;`test_view_has_trained_weights_false_without_ps` | view has trained weights false without ps |
| &nbsp;&nbsp;`test_view_has_trained_weights_true_with_ps` | view has trained weights true with ps |
| &nbsp;&nbsp;`test_to_dict_has_required_keys` | to dict has required keys |
| &nbsp;&nbsp;`test_explanation_points_not_empty` | explanation points not empty |
| &nbsp;&nbsp;`test_explanation_points_contain_interpretability_warning` | explanation points contain interpretability warning |
| &nbsp;&nbsp;`test_executive_result_model_origin` | executive result model origin |
| &nbsp;&nbsp;`test_executive_result_decision_contains_network_name` | executive result decision contains network name |
| &nbsp;&nbsp;`test_executive_result_without_weights_not_production_ready` | executive result without weights not production ready |
| &nbsp;&nbsp;`test_executive_result_with_weights_ready_for_inspection` | executive result with weights ready for inspection |
| &nbsp;&nbsp;`test_executive_result_no_score_without_evaluation` | executive result no score without evaluation |
| &nbsp;&nbsp;`test_executive_result_technical_reference_is_network_name` | executive result technical reference is network name |
| &nbsp;&nbsp;`test_e2e_parse_network_spec` | e2e parse network spec |
| &nbsp;&nbsp;`test_e2e_parsed_network_has_correct_shapes` | e2e parsed network has correct shapes |
| &nbsp;&nbsp;`test_e2e_build_parameter_set` | e2e build parameter set |
| &nbsp;&nbsp;`test_e2e_validate_parameter_set` | e2e validate parameter set |
| &nbsp;&nbsp;`test_e2e_forward_stdlib_returns_list` | e2e forward stdlib returns list |
| &nbsp;&nbsp;`test_e2e_forward_trace_has_activations` | e2e forward trace has activations |
| &nbsp;&nbsp;`test_e2e_forward_deterministic` | e2e forward deterministic |
| &nbsp;&nbsp;`test_e2e_train_step_returns_new_ps` | e2e train step returns new ps |
| &nbsp;&nbsp;`test_e2e_training_loop_loss_decreases` | e2e training loop loss decreases |
| &nbsp;&nbsp;`test_e2e_gradients_not_nan` | e2e gradients not nan |
| &nbsp;&nbsp;`test_e2e_evaluate_regression_result` | e2e evaluate regression result |
| &nbsp;&nbsp;`test_e2e_evaluate_has_mae_rmse_r2` | e2e evaluate has mae rmse r2 |
| &nbsp;&nbsp;`test_e2e_backend_contract_dense_network_ok` | e2e backend contract dense network ok |
| &nbsp;&nbsp;`test_e2e_backend_contract_dense_network_trainable_params` | e2e backend contract dense network trainable params |
| &nbsp;&nbsp;`test_e2e_generator_output_parseable` | e2e generator output parseable |
| &nbsp;&nbsp;`test_e2e_generator_network_spec_compatible_with_parameter_builder` | e2e generator network spec compatible with parameter builder |
| &nbsp;&nbsp;`test_e2e_executive_result_trained_network` | e2e executive result trained network |
| &nbsp;&nbsp;`test_e2e_torch_forward_matches_stdlib` | e2e torch forward matches stdlib |
| &nbsp;&nbsp;`test_regression_classic_parse_text` | regression classic parse text |
| &nbsp;&nbsp;`test_regression_backend_contract_softmax_linear` | regression backend contract softmax linear |
| &nbsp;&nbsp;`test_regression_ir_schema_exports` | regression ir schema exports |
| &nbsp;&nbsp;`test_regression_parameters_exports` | regression parameters exports |
| &nbsp;&nbsp;`test_regression_forward_exports` | regression forward exports |
| &nbsp;&nbsp;`test_regression_training_exports` | regression training exports |
| &nbsp;&nbsp;`test_regression_program_to_dict_with_network` | regression program to dict with network |
| &nbsp;&nbsp;`test_regression_program_without_networks_unaffected` | regression program without networks unaffected |
| &nbsp;&nbsp;`test_regression_check_network_types` | regression check network types |
| &nbsp;&nbsp;`test_regression_training_verifier_imports` | regression training verifier imports |
| &nbsp;&nbsp;`test_regression_parameter_set_source_field` | regression parameter set source field |
| **[test_p18_audit_fixes.py](../tests/test_p18_audit_fixes.py)** | P18 audit fixes: initialiser override, parameter manifest edge cases, hash correctness (48 tests) |
| &nbsp;&nbsp;**TestInitializerOverride** | *TestInitializerOverride* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_weight_initializer_is_he_normal_for_relu` | weight initializer is he normal for relu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bias_initializer_is_zeros` | bias initializer is zeros |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_layer_uses_xavier_normal` | linear layer uses xavier normal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_initializer_override_field_accessible` | initializer override field accessible |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_default_weight_without_override_is_deterministic_uniform` | default weight without override is deterministic uniform |
| &nbsp;&nbsp;**TestValidateParameterSetCompatibility** | *TestValidateParameterSetCompatibility* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_parameter_set_compatible_with_dense_network` | validate parameter set compatible with dense network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_parameter_set_schema_hashes_match` | validate parameter set schema hashes match |
| &nbsp;&nbsp;**TestRuntimeNetworkExecution** | *TestRuntimeNetworkExecution* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_executes_network_node` | runtime executes network node |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_network_output_is_list` | runtime network output is list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_network_trace_has_dense_network_step` | runtime network trace has dense network step |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_network_output_bound_to_output_field` | runtime network output bound to output field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_skips_network_gracefully_without_params` | runtime skips network gracefully without params |
| &nbsp;&nbsp;**TestClassificationMetrics** | *TestClassificationMetrics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_evaluation_has_precision` | binary evaluation has precision |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_evaluation_has_recall` | binary evaluation has recall |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_evaluation_has_f1` | binary evaluation has f1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_binary_evaluation_has_macro_f1` | binary evaluation has macro f1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiclass_evaluation_has_per_class_metrics` | multiclass evaluation has per class metrics |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_includes_precision_recall_f1` | to dict includes precision recall f1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_does_not_include_classification_metrics` | regression does not include classification metrics |
| &nbsp;&nbsp;**TestIRParameterRefs** | *TestIRParameterRefs* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_network_to_dict_has_parameters_per_layer` | network to dict has parameters per layer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_refs_use_correct_keys` | parameter refs use correct keys |
| &nbsp;&nbsp;**TestDenseSupervisedTrainer** | *TestDenseSupervisedTrainer* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_supervised_trainer_importable` | dense supervised trainer importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_supervised_evaluator_importable` | dense supervised evaluator importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_trainer_exported_from_training` | dense trainer exported from training |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_trainer_train_method_exists` | dense trainer train method exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_evaluator_evaluate_method_exists` | dense evaluator evaluate method exists |
| &nbsp;&nbsp;**TestInitialValueRespectOverride** | *TestInitialValueRespectOverride* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_initial_value_uses_he_normal_when_override_set` | initial value uses he normal when override set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_initial_value_uses_xavier_normal_when_override_set` | initial value uses xavier normal when override set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_initial_value_deterministic_without_override` | initial value deterministic without override |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_he_vs_xavier_initial_values_differ` | he vs xavier initial values differ |
| &nbsp;&nbsp;**TestSchemaHashOutputName** | *TestSchemaHashOutputName* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_without_output_name_stable` | schema hash without output name stable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_changes_with_output_name` | schema hash changes with output name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_same_output_name_stable` | schema hash same output name stable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_starts_with_params` | schema hash starts with params |
| &nbsp;&nbsp;**TestDenseSupervisedTrainerIntegration** | *TestDenseSupervisedTrainerIntegration* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_trainer_runs_and_returns_result` | dense trainer runs and returns result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_trainer_parameter_set_has_correct_model_hash` | dense trainer parameter set has correct model hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_trainer_parameter_set_passes_validate` | dense trainer parameter set passes validate |
| &nbsp;&nbsp;**TestCLIImports** | *TestCLIImports* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_imports_dense_trainer` | cli imports dense trainer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_imports_dense_evaluator` | cli imports dense evaluator |
| &nbsp;&nbsp;**TestVerifierNetworkDeclaration** | *TestVerifierNetworkDeclaration* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_ok_for_network_model` | verifier ok for network model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_no_undeclared_node_error_for_network` | verifier no undeclared node error for network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifier_declared_nodes_includes_network_names` | verifier declared nodes includes network names |
| &nbsp;&nbsp;**TestBCENumericProbabilityTarget** | *TestBCENumericProbabilityTarget* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bce_numeric_target_0_9_maps_to_0_9` | bce numeric target 0 9 maps to 0 9 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bce_numeric_target_0_0_maps_to_0_0` | bce numeric target 0 0 maps to 0 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bce_label_fallback_when_no_target_value` | bce label fallback when no target value |
| &nbsp;&nbsp;**TestOutputNameWiredInBuildPS** | *TestOutputNameWiredInBuildPS* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_output_name_gives_different_schema_hash` | different output name gives different schema hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_same_output_name_gives_same_schema_hash` | same output name gives same schema hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_network_parameter_set_consistent_with_output_name` | validate network parameter set consistent with output name |

## Composite Networks and Embeddings (P19) (313 tests)

| Test | Description |
|------|-------------|
| **[test_p19_cut1_network_parser.py](../tests/test_p19_cut1_network_parser.py)** | Composite parser: EMBEDDING, CONCAT, BLOCK, RESIDUAL, POOL, Dropout, LayerNorm (31 tests) |
| &nbsp;&nbsp;**TestNetworkP19Basics** | *TestNetworkP19Basics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_embedding_declaration` | parser accepts embedding declaration |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_concat_with_multiple_sources` | parser accepts concat with multiple sources |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_block_with_residual_from_previous` | parser accepts block with residual from previous |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_block_with_residual_from_named` | parser accepts block with residual from named |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_pool_mean_at_top_level` | parser accepts pool mean at top level |
| &nbsp;&nbsp;**TestNetworkP19Layers** | *TestNetworkP19Layers* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_dropout_with_valid_rate` | parser accepts dropout with valid rate |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_layernorm` | parser accepts layernorm |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_standalone_activation_relu` | parser accepts standalone activation relu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_activation_gelu` | parser accepts activation gelu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parser_accepts_reshape_with_valid_target` | parser accepts reshape with valid target |
| &nbsp;&nbsp;**TestNetworkP19KindDetection** | *TestNetworkP19KindDetection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p18_dense_only_is_kind_dense_network` | p18 dense only is kind dense network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p18_layers_populated_in_dense_network` | p18 layers populated in dense network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_promotes_to_composite_network` | embedding promotes to composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_promotes_to_composite_network` | block promotes to composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pool_promotes_to_composite_network` | pool promotes to composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_layers_field_is_empty` | composite network layers field is empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_graph_node_type_is_composite_network` | graph node type is composite network |
| &nbsp;&nbsp;**TestNetworkP19Errors** | *TestNetworkP19Errors* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_embedding_with_zero_vocab` | rejects embedding with zero vocab |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_embedding_with_zero_dim` | rejects embedding with zero dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_dropout_rate_ge_1` | rejects dropout rate ge 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_dropout_rate_le_0` | rejects dropout rate le 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_empty_block` | rejects empty block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_nested_blocks` | rejects nested blocks |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_unknown_activation_for_standalone_activation_layer` | rejects unknown activation for standalone activation layer |
| &nbsp;&nbsp;**TestNetworkP19IR** | *TestNetworkP19IR* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_spec_has_correct_fields` | embedding spec has correct fields |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_spec_has_layers_and_residual_from` | block spec has layers and residual from |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_to_dict_includes_embeddings` | composite network to dict includes embeddings |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_to_dict_includes_blocks` | composite network to dict includes blocks |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_layer_to_dict_dense_has_hierarchical_param_paths` | composite layer to dict dense has hierarchical param paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_dense_layer_has_hierarchical_param_path` | block dense layer has hierarchical param path |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p18_dense_network_to_dict_unchanged` | p18 dense network to dict unchanged |
| **[test_p19_cut2_shape_inference.py](../tests/test_p19_cut2_shape_inference.py)** | Composite shape inference: hierarchical propagation through blocks (24 tests) |
| &nbsp;&nbsp;**TestCompositeNetworkShapeEmbeddingConcat** | *TestCompositeNetworkShapeEmbeddingConcat* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_check_ok_for_valid_model` | composite network check ok for valid model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_named_shape_is_dim` | embedding named shape is dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_scalar_field_named_shape_is_one` | scalar field named shape is one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_concat_output_shape_is_sum_of_source_dims` | concat output shape is sum of source dims |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_first_top_layer_input_shape_equals_concat_output` | first top layer input shape equals concat output |
| &nbsp;&nbsp;**TestCompositeNetworkLayerShapes** | *TestCompositeNetworkLayerShapes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_preserves_input_shape` | layernorm preserves input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_preserves_input_shape` | dropout preserves input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_activation_preserves_input_shape` | activation preserves input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reshape_output_equals_target_shape` | reshape output equals target shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_after_reshape_uses_reshaped_shape` | dense after reshape uses reshaped shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reshape_rejects_wrong_product_of_dims` | reshape rejects wrong product of dims |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_layers_check_returns_ok` | all layers check returns ok |
| &nbsp;&nbsp;**TestCompositeNetworkBlockShapes** | *TestCompositeNetworkBlockShapes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_layers_have_correct_shapes` | block layers have correct shapes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_output_shape_equals_last_block_layer_output` | block output shape equals last block layer output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_from_previous_ok_when_shapes_match` | residual from previous ok when shapes match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_from_named_ok_when_shapes_match` | residual from named ok when shapes match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_top_layer_after_block_uses_block_output_shape` | top layer after block uses block output shape |
| &nbsp;&nbsp;**TestCompositeNetworkShapeErrors** | *TestCompositeNetworkShapeErrors* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_from_previous_errors_when_block_changes_shape` | residual from previous errors when block changes shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_from_named_errors_when_shape_mismatches` | residual from named errors when shape mismatches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_concat_errors_when_undeclared_source` | concat errors when undeclared source |
| &nbsp;&nbsp;**TestCompositeNetworkDispatch** | *TestCompositeNetworkDispatch* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_network_types_dispatches_to_composite_for_composite_kind` | check network types dispatches to composite for composite kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_network_types_dispatches_to_dense_for_dense_kind` | check network types dispatches to dense for dense kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_interpretability_warning_present_in_composite_result` | interpretability warning present in composite result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_output_type_inferred_correctly` | composite network output type inferred correctly |
| **[test_p19_cut3_parameter_manifest.py](../tests/test_p19_cut3_parameter_manifest.py)** | Composite parameter manifest: hierarchical paths, embedding table vocab×dim (23 tests) |
| &nbsp;&nbsp;**TestCompositeManifestEmbedding** | *TestCompositeManifestEmbedding* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_includes_embedding_table` | manifest includes embedding table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_embedding_table_shape_is_vocab_by_dim` | manifest embedding table shape is vocab by dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_embedding_table_initializer_is_xavier_normal` | manifest embedding table initializer is xavier normal |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_no_entries_for_concat_dropout_pool` | manifest no entries for concat dropout pool |
| &nbsp;&nbsp;**TestCompositeManifestLayerNorm** | *TestCompositeManifestLayerNorm* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_includes_layernorm_gamma_and_beta` | manifest includes layernorm gamma and beta |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_layernorm_gamma_initializer_is_ones` | manifest layernorm gamma initializer is ones |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_layernorm_beta_initializer_is_zeros` | manifest layernorm beta initializer is zeros |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_layernorm_gamma_shape_equals_features` | manifest layernorm gamma shape equals features |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_top_level_layernorm_has_correct_path` | manifest top level layernorm has correct path |
| &nbsp;&nbsp;**TestCompositeManifestHierarchicalPaths** | *TestCompositeManifestHierarchicalPaths* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_dense_path_includes_block_name` | block dense path includes block name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_layernorm_path_includes_block_name` | block layernorm path includes block name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_top_level_dense_path_does_not_include_block_name` | top level dense path does not include block name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_full_path_structure_for_block_layers` | full path structure for block layers |
| &nbsp;&nbsp;**TestCompositeSchemaHash** | *TestCompositeSchemaHash* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_stable_for_same_architecture` | schema hash stable for same architecture |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_changes_when_embedding_dim_changes` | schema hash changes when embedding dim changes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_changes_when_dense_units_change` | schema hash changes when dense units change |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_schema_hash_changes_when_residual_source_changes` | schema hash changes when residual source changes |
| &nbsp;&nbsp;**TestCompositeParameterSet** | *TestCompositeParameterSet* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_composite_parameter_set_contains_all_params` | build composite parameter set contains all params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_composite_parameter_set_embedding_table_shape` | build composite parameter set embedding table shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_composite_parameter_set_gamma_values_are_ones` | build composite parameter set gamma values are ones |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_composite_parameter_set_beta_values_are_zeros` | build composite parameter set beta values are zeros |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_composite_parameter_set_ok_for_matching_ps` | validate composite parameter set ok for matching ps |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_detects_missing_parameter` | validate detects missing parameter |
| **[test_p19_cut4_backend_contract.py](../tests/test_p19_cut4_backend_contract.py)** | Composite backend contract: layer_manifest for composite_network kind (25 tests) |
| &nbsp;&nbsp;**TestCompositeNetworkNodeReport** | *TestCompositeNetworkNodeReport* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_node_type_is_composite_network` | node type is composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_is_supported` | composite network is supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_is_differentiable` | composite network is differentiable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_kind_is_composite_network` | composite network kind is composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_network_output_shape_from_final_dense` | composite network output shape from final dense |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_ok_for_valid_composite_model` | report ok for valid composite model |
| &nbsp;&nbsp;**TestCompositeTrainableParameters** | *TestCompositeTrainableParameters* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_include_embedding_table` | trainable params include embedding table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_include_dense_weights` | trainable params include dense weights |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_include_layernorm_gamma_beta` | trainable params include layernorm gamma beta |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_have_hierarchical_block_paths` | trainable params have hierarchical block paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_table_shape_in_parameter_manifest` | embedding table shape in parameter manifest |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_table_initializer_in_parameter_manifest` | embedding table initializer in parameter manifest |
| &nbsp;&nbsp;**TestCompositeLayerManifest** | *TestCompositeLayerManifest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_includes_embedding_entry` | layer manifest includes embedding entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_manifest_includes_block_entry` | layer manifest includes block entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_entry_has_sub_layers` | block entry has sub layers |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_entry_reports_residual_from` | block entry reports residual from |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_layer_has_dropout_active_flag` | dropout layer has dropout active flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_entry_has_correct_layer_type` | dense entry has correct layer type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_includes_layer_manifest` | to dict includes layer manifest |
| &nbsp;&nbsp;**TestCompositeInterpretabilityWarnings** | *TestCompositeInterpretabilityWarnings* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_warns_interpretability_reduced_for_composite` | warns interpretability reduced for composite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_warns_very_reduced_for_two_residual_blocks` | warns very reduced for two residual blocks |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_warns_very_reduced_for_large_embedding_dim` | warns very reduced for large embedding dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_not_very_reduced_for_single_small_residual_block` | not very reduced for single small residual block |
| &nbsp;&nbsp;**TestDenseNetworkUnchangedByC4** | *TestDenseNetworkUnchangedByC4* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_network_still_reports_dense_network_kind` | dense network still reports dense network kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dense_network_p18_warning_unchanged` | dense network p18 warning unchanged |
| **[test_p19_cut5_forward_stdlib.py](../tests/test_p19_cut5_forward_stdlib.py)** | Composite forward: composite_forward with embedding lookup, residual, dropout (25 tests) |
| &nbsp;&nbsp;**TestEmbeddingForward** | *TestEmbeddingForward* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_produces_correct_dim` | embedding lookup produces correct dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_picks_correct_row` | embedding lookup picks correct row |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_indices_produce_different_embeddings` | different indices produce different embeddings |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_concat_shape_in_named_tensors` | embedding concat shape in named tensors |
| &nbsp;&nbsp;**TestLayerNormForward** | *TestLayerNormForward* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_output_has_correct_length` | layernorm output has correct length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_with_identity_gamma_beta_is_normalized` | layernorm with identity gamma beta is normalized |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_specific_values` | layernorm specific values |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_scaled_gamma` | layernorm scaled gamma |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_applied_in_block` | layernorm applied in block |
| &nbsp;&nbsp;**TestDropoutForward** | *TestDropoutForward* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_in_eval_mode_is_identity` | dropout in eval mode is identity |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_output_has_correct_shape` | dropout output has correct shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_seeded_is_deterministic` | dropout seeded is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_trace_stores_mask` | dropout trace stores mask |
| &nbsp;&nbsp;**TestResidualForward** | *TestResidualForward* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_from_previous_adds_block_input` | residual from previous adds block input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_from_named_adds_named_tensor` | residual from named adds named tensor |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_output_shape_matches_residual_input` | residual output shape matches residual input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_result_equals_block_input_plus_block_output` | residual result equals block input plus block output |
| &nbsp;&nbsp;**TestFullCompositeForward** | *TestFullCompositeForward* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_concat_dense_output_shape` | embedding concat dense output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_output_sums_to_one` | softmax output sums to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_all_outputs_positive` | softmax all outputs positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_network_output_shape` | residual network output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_layers_model_output_shape_and_sum` | all layers model output shape and sum |
| &nbsp;&nbsp;**TestActivationPoolReshapeForward** | *TestActivationPoolReshapeForward* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_standalone_activation_preserves_shape` | standalone activation preserves shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pool_pass_through_for_flat_input` | pool pass through for flat input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_forward_trace_returns_named_tensors` | composite forward trace returns named tensors |
| **[test_p19_cut6_backprop.py](../tests/test_p19_cut6_backprop.py)** | Composite backprop: gradients for Embedding, LayerNorm, Dropout, Residual, Concat (22 tests) |
| &nbsp;&nbsp;**TestGradientShapes** | *TestGradientShapes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradients_include_all_dense_keys` | gradients include all dense keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradients_include_embedding_table` | gradients include embedding table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_table_gradient_shape` | embedding table gradient shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_table_gradient_nonzero_at_used_index` | embedding table gradient nonzero at used index |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradients_include_block_layernorm_params` | gradients include block layernorm params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_gamma_gradient_shape` | layernorm gamma gradient shape |
| &nbsp;&nbsp;**TestGradientCheck** | *TestGradientCheck* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_dense_weight` | gradient check dense weight |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_dense_bias` | gradient check dense bias |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_layernorm_gamma` | gradient check layernorm gamma |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_layernorm_beta` | gradient check layernorm beta |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_embedding_table_used_row` | gradient check embedding table used row |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_embedding_table_second_dim` | gradient check embedding table second dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_block_dense_weight` | gradient check block dense weight |
| &nbsp;&nbsp;**TestTrainingStep** | *TestTrainingStep* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_step_returns_tuple` | train step returns tuple |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_step_returns_float_loss` | train step returns float loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_step_updates_parameters` | train step updates parameters |
| &nbsp;&nbsp;**TestLossReduction** | *TestLossReduction* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_steps_reduce_mse_loss` | multiple steps reduce mse loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_steps_reduce_cross_entropy` | multiple steps reduce cross entropy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_table_updates_on_train` | embedding table updates on train |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layernorm_gamma_updates_on_train` | layernorm gamma updates on train |
| &nbsp;&nbsp;**TestResidualGradient** | *TestResidualGradient* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_block_gradient_flows_to_both_paths` | residual block gradient flows to both paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gradient_check_passes_for_residual_network` | gradient check passes for residual network |
| **[test_p19_cut7_training_verifier.py](../tests/test_p19_cut7_training_verifier.py)** | Composite training verifier: UPDATE Net.*, differentiability with embedding table (23 tests) |
| &nbsp;&nbsp;**TestSymbolType** | *TestSymbolType* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probabilitymap_for_softmax_composite_network` | probabilitymap for softmax composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_scalar_for_regression_composite_network` | scalar for regression composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_symbol_type_also_matches_composite_network_name` | symbol type also matches composite network name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prediction_function_is_none_for_composite_network` | prediction function is none for composite network |
| &nbsp;&nbsp;**TestPredictionNode** | *TestPredictionNode* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_finds_composite_network_by_output_variable` | finds composite network by output variable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_finds_residual_composite_network_by_output` | finds residual composite network by output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_finds_composite_network_by_network_name` | finds composite network by network name |
| &nbsp;&nbsp;**TestDifferentiabilityVerifier** | *TestDifferentiabilityVerifier* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_errors_for_cross_entropy_embedding_network` | no errors for cross entropy embedding network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_prediction_node_is_composite_network` | prediction node is composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_paths_include_embedding_table` | parameter paths include embedding table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_paths_include_dense_weights` | parameter paths include dense weights |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_errors_for_mse_residual_network` | no errors for mse residual network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_paths_include_layernorm_params` | parameter paths include layernorm params |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dropout_does_not_block_differentiability` | dropout does not block differentiability |
| &nbsp;&nbsp;**TestTrainingVerifierFull** | *TestTrainingVerifierFull* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_cross_entropy_composite_with_embedding` | accepts cross entropy composite with embedding |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_mse_composite_with_residual` | accepts mse composite with residual |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_dropout_composite_network` | accepts dropout composite network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_include_embedding_table` | trainable params include embedding table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_trainable_params_include_layernorm_gamma_beta` | trainable params include layernorm gamma beta |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_update_wildcard_covers_embedding_table` | update wildcard covers embedding table |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_update_wildcard_covers_block_layernorm` | update wildcard covers block layernorm |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_cross_entropy_for_scalar_output_composite` | rejects cross entropy for scalar output composite |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_params_use_hierarchical_names` | block params use hierarchical names |
| **[test_p19_cut8_evaluation.py](../tests/test_p19_cut8_evaluation.py)** | Composite evaluation: Dropout disabled in eval mode, P18 metrics reused (22 tests) |
| &nbsp;&nbsp;**TestCompositeExamplesFromCsv** | *TestCompositeExamplesFromCsv* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_produces_list_of_tuples` | produces list of tuples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_is_dict_with_column_keys` | input is dict with column keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_categorical_column_is_numeric` | categorical column is numeric |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classification_target_is_onehot` | classification target is onehot |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_target_is_scalar_list` | regression target is scalar list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loads_all_rows` | loads all rows |
| &nbsp;&nbsp;**TestDropoutDisabledOnEval** | *TestDropoutDisabledOnEval* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_is_deterministic_with_dropout_network` | eval is deterministic with dropout network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_output_differs_from_training_with_dropout` | eval output differs from training with dropout |
| &nbsp;&nbsp;**TestEvaluateCrossEntropy** | *TestEvaluateCrossEntropy* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_dense_evaluation_result` | returns dense evaluation result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rows_count_matches_input` | rows count matches input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cross_entropy_loss_is_positive` | cross entropy loss is positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accuracy_is_between_zero_and_one` | accuracy is between zero and one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_confusion_matrix_present` | confusion matrix present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_labels_stored_in_result` | labels stored in result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_macro_f1_between_zero_and_one` | macro f1 between zero and one |
| &nbsp;&nbsp;**TestEvaluateMse** | *TestEvaluateMse* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_mae_rmse_r2` | returns mae rmse r2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_loss_is_nonnegative` | mse loss is nonnegative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mse_rows_count` | mse rows count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_reduces_eval_loss` | training reduces eval loss |
| &nbsp;&nbsp;**TestCategoricalColumnHandling** | *TestCategoricalColumnHandling* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_lookup_uses_csv_integer` | embedding lookup uses csv integer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_category_ids_give_different_predictions` | different category ids give different predictions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_examples_raises` | empty examples raises |
| **[test_p19_cut9_torch.py](../tests/test_p19_cut9_torch.py)** | Composite Torch: nn.Embedding, nn.LayerNorm, nn.Dropout, atol<1e-4 equivalence (22 tests) |
| &nbsp;&nbsp;**TestModuleCreation** | *TestModuleCreation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_module_created_for_embedding_network` | module created for embedding network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_module_created_for_residual_network` | module created for residual network |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_module_has_embedding_layer` | module has embedding layer |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_layer_correct_shape` | embedding layer correct shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_weights_match_parameter_set` | embedding weights match parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_module_has_trainable_parameters` | module has trainable parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_module_parameter_count_matches_manifest` | module parameter count matches manifest |
| &nbsp;&nbsp;**TestForwardMatchesStdlib** | *TestForwardMatchesStdlib* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_forward_matches_stdlib` | embedding forward matches stdlib |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_residual_forward_matches_stdlib` | residual forward matches stdlib |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_forward_output_is_list` | forward output is list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_forward_different_ids_give_different_outputs` | embedding forward different ids give different outputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_output_sums_to_one` | softmax output sums to one |
| &nbsp;&nbsp;**TestDropoutEvalMode** | *TestDropoutEvalMode* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_mode_is_deterministic` | eval mode is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_eval_mode_matches_stdlib_eval` | eval mode matches stdlib eval |
| &nbsp;&nbsp;**TestBackwardMatchesStdlib** | *TestBackwardMatchesStdlib* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_one_step_loss_matches_stdlib` | one step loss matches stdlib |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_one_step_weights_match_stdlib` | one step weights match stdlib |
| &nbsp;&nbsp;**TestParameterRoundtrip** | *TestParameterRoundtrip* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_roundtrip_preserves_parameter_set_structure` | roundtrip preserves parameter set structure |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_roundtrip_preserves_initial_weights` | roundtrip preserves initial weights |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_roundtrip_source_is_torch` | roundtrip source is torch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_roundtrip_preserves_embedding_weights` | roundtrip preserves embedding weights |
| &nbsp;&nbsp;**TestErrorHandling** | *TestErrorHandling* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_composite_torch_error_is_value_error` | composite torch error is value error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_missing_parameter_raises_composite_torch_error` | missing parameter raises composite torch error |
| **[test_p19_cut10_composite_generator.py](../tests/test_p19_cut10_composite_generator.py)** | CompositeNetworkGenerator: 6 heuristic rules, categorical fields, force_residual (28 tests) |
| &nbsp;&nbsp;**TestCategoricalDetection** | *TestCategoricalDetection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_creates_embedding_for_categorical_field` | generator creates embedding for categorical field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_embedding_vocab_and_dim` | generator embedding vocab and dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_multiple_embeddings` | generator multiple embeddings |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_small_vocab_no_embedding` | generator small vocab no embedding |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_categorical_keyword_in_field_name_auto_detected` | generator categorical keyword in field name auto detected |
| &nbsp;&nbsp;**TestResidualDetection** | *TestResidualDetection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_creates_residual_block_for_complex_prompt` | generator creates residual block for complex prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_block_has_layernorm` | generator block has layernorm |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_block_has_dropout` | generator block has dropout |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_no_residual_without_complexity_keyword` | generator no residual without complexity keyword |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_no_residual_with_few_features` | generator no residual with few features |
| &nbsp;&nbsp;**TestSequenceInput** | *TestSequenceInput* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_creates_pool_mean_for_sequence_input` | generator creates pool mean for sequence input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_no_pool_for_non_sequence` | generator no pool for non sequence |
| &nbsp;&nbsp;**TestFallback** | *TestFallback* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_falls_back_to_p18_dense_for_simple_prompt` | generator falls back to p18 dense for simple prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_fallback_result_structure` | generator fallback result structure |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_fallback_emits_dense_network` | generator fallback emits dense network |
| &nbsp;&nbsp;**TestMxaiText** | *TestMxaiText* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_emits_valid_mxai_text` | generator emits valid mxai text |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_mxai_contains_embedding` | generator mxai contains embedding |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_mxai_contains_block` | generator mxai contains block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_emits_valid_mxtrain_text` | generator emits valid mxtrain text |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_mxai_pool_in_text` | generator mxai pool in text |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_dense_fallback_text_parseable` | generator dense fallback text parseable |
| &nbsp;&nbsp;**TestExplicitHints** | *TestExplicitHints* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_explicit_vocab_respected` | generator explicit vocab respected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_explicit_force_residual` | generator explicit force residual |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generator_respects_explicit_architecture_hint` | generator respects explicit architecture hint |
| &nbsp;&nbsp;**TestResultStructure** | *TestResultStructure* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_to_dict_contains_keys` | result to dict contains keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_assumptions_not_empty` | result assumptions not empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_error_on_empty_prompt` | error on empty prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_error_on_whitespace_prompt` | error on whitespace prompt |
| &nbsp;&nbsp;**TestArchitectureText** | *TestArchitectureText* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_contains_input` | arch text contains input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_contains_embedding` | arch text contains embedding |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_contains_concat` | arch text contains concat |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_contains_block` | arch text contains block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_contains_arrow_separator` | arch text contains arrow separator |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_ends_with_output_type_prefix` | arch text ends with output type prefix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_arch_text_block_has_residual_marker` | arch text block has residual marker |
| &nbsp;&nbsp;**TestEmbeddingInfo** | *TestEmbeddingInfo* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embeddings_list_length` | embeddings list length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_vocab_and_dim` | embedding vocab and dim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_params_count` | embedding params count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_embeddings_for_residual_network` | no embeddings for residual network |
| &nbsp;&nbsp;**TestBlocksInfo** | *TestBlocksInfo* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_blocks_list_not_empty_for_residual` | blocks list not empty for residual |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_name_correct` | block name correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_residual_from_set` | block residual from set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_block_layer_types_listed` | block layer types listed |
| &nbsp;&nbsp;**TestParamCounting** | *TestParamCounting* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_total_params_positive_with_ps` | total params positive with ps |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_embedding_params_in_total` | embedding params in total |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_total_params_zero_without_ps_for_dense_layers` | total params zero without ps for dense layers |
| &nbsp;&nbsp;**TestInterpretability** | *TestInterpretability* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_interpretability_reduced_single_block` | interpretability reduced single block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_interpretability_very_reduced_two_blocks` | interpretability very reduced two blocks |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_interpretability_warning_not_empty` | interpretability warning not empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_interpretability_warning_mentions_embedding` | interpretability warning mentions embedding |
| &nbsp;&nbsp;**TestFlags** | *TestFlags* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_trained_weights_false_without_ps` | has trained weights false without ps |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_trained_weights_true_with_ps` | has trained weights true with ps |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_has_composite_keys` | to dict has composite keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_type_cross_entropy_for_probabilitymap` | loss type cross entropy for probabilitymap |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_type_mse_for_scalar` | loss type mse for scalar |
| &nbsp;&nbsp;**TestExecutiveResult** | *TestExecutiveResult* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executive_result_model_origin` | executive result model origin |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executive_result_decision_contains_network_name` | executive result decision contains network name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executive_result_without_weights_not_production_ready` | executive result without weights not production ready |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executive_result_with_weights_ready_for_inspection` | executive result with weights ready for inspection |
| &nbsp;&nbsp;`test_e2e_composite_parse_network_spec` | e2e composite parse network spec |
| &nbsp;&nbsp;`test_e2e_composite_type_check_ok` | e2e composite type check ok |
| &nbsp;&nbsp;`test_e2e_residual_type_check_ok` | e2e residual type check ok |
| &nbsp;&nbsp;`test_e2e_composite_build_parameter_set` | e2e composite build parameter set |
| &nbsp;&nbsp;`test_e2e_composite_parameter_set_embedding_shape` | e2e composite parameter set embedding shape |
| &nbsp;&nbsp;`test_e2e_composite_validate_parameter_set` | e2e composite validate parameter set |
| &nbsp;&nbsp;`test_e2e_composite_forward_returns_list` | e2e composite forward returns list |
| &nbsp;&nbsp;`test_e2e_composite_forward_deterministic` | e2e composite forward deterministic |
| &nbsp;&nbsp;`test_e2e_composite_forward_softmax_sums_to_one` | e2e composite forward softmax sums to one |
| &nbsp;&nbsp;`test_e2e_residual_forward_returns_scalar` | e2e residual forward returns scalar |
| &nbsp;&nbsp;`test_e2e_composite_train_step_returns_new_ps` | e2e composite train step returns new ps |
| &nbsp;&nbsp;`test_e2e_residual_training_loop_loss_decreases` | e2e residual training loop loss decreases |
| &nbsp;&nbsp;`test_e2e_composite_gradients_not_nan` | e2e composite gradients not nan |
| &nbsp;&nbsp;`test_e2e_gradient_check_passes_for_composite` | e2e gradient check passes for composite |
| &nbsp;&nbsp;`test_e2e_composite_evaluate_has_accuracy` | e2e composite evaluate has accuracy |
| &nbsp;&nbsp;`test_e2e_composite_evaluate_rows_count` | e2e composite evaluate rows count |
| &nbsp;&nbsp;`test_e2e_residual_evaluate_has_mae_rmse_r2` | e2e residual evaluate has mae rmse r2 |
| &nbsp;&nbsp;`test_e2e_backend_contract_composite_ok` | e2e backend contract composite ok |
| &nbsp;&nbsp;`test_e2e_backend_contract_composite_has_embedding_param` | e2e backend contract composite has embedding param |
| &nbsp;&nbsp;`test_e2e_backend_contract_composite_interpretability_warning` | e2e backend contract composite interpretability warning |
| &nbsp;&nbsp;`test_e2e_composite_generator_output_parseable` | e2e composite generator output parseable |
| &nbsp;&nbsp;`test_e2e_composite_generator_fallback_parseable` | e2e composite generator fallback parseable |
| &nbsp;&nbsp;`test_e2e_composite_executive_result_trained` | e2e composite executive result trained |
| &nbsp;&nbsp;`test_regression_composite_ir_exports` | regression composite ir exports |
| &nbsp;&nbsp;`test_regression_composite_forward_exports` | regression composite forward exports |
| &nbsp;&nbsp;`test_regression_composite_training_exports` | regression composite training exports |
| &nbsp;&nbsp;`test_regression_p18_dense_forward_unaffected` | regression p18 dense forward unaffected |
| &nbsp;&nbsp;`test_regression_p18_dense_generator_unaffected` | regression p18 dense generator unaffected |
| &nbsp;&nbsp;`test_regression_p18_backend_contract_unaffected` | regression p18 backend contract unaffected |
| &nbsp;&nbsp;`test_regression_classic_parse_unaffected` | regression classic parse unaffected |
| &nbsp;&nbsp;`test_regression_check_composite_network_types_exported` | regression check composite network types exported |
| &nbsp;&nbsp;`test_regression_training_verifier_composite_aware` | regression training verifier composite aware |
| &nbsp;&nbsp;`test_regression_composite_program_to_dict` | regression composite program to dict |

## GPU Acceleration (P14) (118 tests)

| Test | Description |
|------|-------------|
| **[test_p14_backend_spec.py](../tests/test_p14_backend_spec.py)** | BackendSpec: TARGET/DEVICE in .mxtrain, invalid combinations rejected (29 tests) |
| &nbsp;&nbsp;**TestBackendSpecContract** | *TestBackendSpecContract* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_importable_from_training` | importable from training |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_importable_from_spec` | importable from spec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_defaults_are_stdlib_cpu` | defaults are stdlib cpu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_has_target_and_device` | to dict has target and device |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_is_frozen` | is frozen |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_invalid_target_raises` | invalid target raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_invalid_device_raises` | invalid device raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_valid_combinations` | valid combinations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_stdlib_non_cpu_raises` | stdlib non cpu raises |
| &nbsp;&nbsp;**TestTrainingSpecBackendField** | *TestTrainingSpecBackendField* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_none_when_block_absent` | backend none when block absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_has_no_backend_key_when_none` | to dict has no backend key when none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_real_mxtrain_files_have_no_backend` | real mxtrain files have no backend |
| &nbsp;&nbsp;**TestBackendBlockParser** | *TestBackendBlockParser* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_target_torch_device_cpu` | backend target torch device cpu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_target_torch_device_cuda` | backend target torch device cuda |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_target_torch_device_mps` | backend target torch device mps |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_target_stdlib` | backend target stdlib |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_default_device_cpu_when_only_target` | backend default device cpu when only target |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_default_target_stdlib_when_only_device` | backend default target stdlib when only device |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_in_to_dict` | backend in to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_invalid_target_raises_parse_error` | invalid target raises parse error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_invalid_device_raises_parse_error` | invalid device raises parse error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unknown_backend_line_raises_parse_error` | unknown backend line raises parse error |
| &nbsp;&nbsp;**TestCLIDeviceValidation** | *TestCLIDeviceValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_train_help_shows_device` | train help shows device |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_evaluate_help_shows_device` | evaluate help shows device |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_cuda_without_torch_backend_exits_1` | device cuda without torch backend exits 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_mps_without_torch_backend_exits_1` | device mps without torch backend exits 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_cpu_with_stdlib_backend_is_default_and_valid` | device cpu with stdlib backend is default and valid |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_cuda_unavailable_exits_1` | When cuda hardware is absent, --device cuda must fail with exit 1. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_mps_unavailable_exits_1` | device mps unavailable exits 1 |
| **[test_p14_device_detection.py](../tests/test_p14_device_detection.py)** | Device detection: torch_device_info, cpu/cuda/mps availability (29 tests) |
| &nbsp;&nbsp;**TestTorchDeviceInfoContract** | *Core contract: torch_device_info() always returns a well-formed dict.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_dict` | returns dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_torch_available_key` | has torch available key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_torch_version_key` | has torch version key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_available_devices_list` | has available devices list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cpu_always_in_available_devices` | cpu always in available devices |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_cuda_available_bool` | has cuda available bool |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_mps_available_bool` | has mps available bool |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_cuda_version_key` | has cuda version key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_device_name_key` | has device name key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_importable_from_parameters` | importable from parameters |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_importable_from_tensor_bridge` | importable from tensor bridge |
| &nbsp;&nbsp;**TestTorchDeviceInfoWithTorchInstalled** | *Tests for the current environment where torch IS installed.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_available_is_true` | torch available is true |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_version_is_string` | torch version is string |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_available_true_in_info` | torch available true in info |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cpu_in_devices` | cpu in devices |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_absent_when_no_hardware` | cuda absent when no hardware |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mps_absent_when_no_hardware` | mps absent when no hardware |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_available_devices_has_no_duplicates` | available devices has no duplicates |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_available_devices_only_known_values` | available devices only known values |
| &nbsp;&nbsp;**TestTorchDeviceInfoCudaPresent** | *Tests that run only when CUDA is available.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_in_devices` | cuda in devices |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_version_is_string` | cuda version is string |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_name_is_string` | device name is string |
| &nbsp;&nbsp;**TestTorchDeviceInfoMpsPresent** | *Tests that run only when MPS is available.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mps_in_devices` | mps in devices |
| &nbsp;&nbsp;**TestTorchDeviceInfoWithoutTorch** | *Simulate environment where torch is not installed.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_dict_when_torch_absent` | returns dict when torch absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_available_false_when_absent` | torch available false when absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cpu_still_in_devices_when_torch_absent` | cpu still in devices when torch absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_version_none_when_absent` | torch version none when absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_available_false_when_torch_absent` | cuda available false when torch absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mps_available_false_when_torch_absent` | mps available false when torch absent |
| **[test_p14_device_propagation.py](../tests/test_p14_device_propagation.py)** | Device propagation: CLI --device overrides .mxtrain DEVICE, _resolve_backend_spec (15 tests) |
| &nbsp;&nbsp;**TestTorchForwardRunnerDeviceValidation** | *TestTorchForwardRunnerDeviceValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cpu_device_always_valid` | cpu device always valid |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_float64_dtype_raises` | float64 dtype raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_unavailable_raises_forward_error` | cuda unavailable raises forward error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mps_unavailable_raises_forward_error` | mps unavailable raises forward error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_error_message_contains_available_devices` | error message contains available devices |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_old_p5_error_message_no_longer_present` | P5 hard-coded error is gone; only dtype and availability checks remain. |
| &nbsp;&nbsp;**TestTorchForwardRunnerCpuRun** | *Smoke test: forward run on cpu still works after Corte 3 changes.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_cpu_produces_result` | run cpu produces result |
| &nbsp;&nbsp;**TestBatchTensorsDeviceParameter** | *TestBatchTensorsDeviceParameter* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_tensors_cpu_default` | batch tensors cpu default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_tensors_cpu_explicit` | batch tensors cpu explicit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_tensors_binary_objective_cpu` | batch tensors binary objective cpu |
| &nbsp;&nbsp;**TestTorchSupervisedTrainerDeviceReading** | *TestTorchSupervisedTrainerDeviceReading* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_read_from_backend_cpu` | Training with explicit BackendSpec cpu completes. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_defaults_to_cpu_when_no_backend` | If training.backend is None, device defaults to cpu — no AttributeError. |
| &nbsp;&nbsp;**TestCLIDeviceInjection** | *TestCLIDeviceInjection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cmd_train_injects_backend_spec_into_training` | After _cmd_train runs _validate_device and parses training, it injects BackendSpec. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cmd_train_cuda_unavailable_exits_before_training` | cmd train cuda unavailable exits before training |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_metadata_reflects_device` | torch_backend_metadata(device=X) returns dict with device=X. |
| **[test_p14_backend_runtime.py](../tests/test_p14_backend_runtime.py)** | Backend runtime: train/evaluate/backend-run routing with --device flag (19 tests) |
| &nbsp;&nbsp;**TestEvaluationResultBackendRuntimeField** | *TestEvaluationResultBackendRuntimeField* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_field_exists_with_empty_default` | field exists with empty default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_omits_backend_runtime_when_empty` | to dict omits backend runtime when empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_includes_backend_runtime_when_set` | to dict includes backend runtime when set |
| &nbsp;&nbsp;**TestBuildBackendRuntime** | *TestBuildBackendRuntime* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_dict_with_required_keys` | returns dict with required keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_target_is_torch` | target is torch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_device_reflects_argument` | device reflects argument |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_seed_key_present` | seed key present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_version_absent_on_cpu_only_machine` | cuda version absent on cpu only machine |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_version_present_when_cuda_available` | cuda version present when cuda available |
| &nbsp;&nbsp;**TestTrainingTraceBackendRuntime** | *TestTrainingTraceBackendRuntime* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_trace_has_backend_runtime_key` | training trace has backend runtime key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_runtime_target_is_torch` | backend runtime target is torch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_runtime_device_is_cpu` | backend runtime device is cpu |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_runtime_has_torch_version` | backend runtime has torch version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_runtime_has_device_name_key` | backend runtime has device name key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_runtime_has_seed_key` | backend runtime has seed key |
| &nbsp;&nbsp;**TestStdlibTrainingTraceNoBackendRuntime** | *TestStdlibTrainingTraceNoBackendRuntime* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_stdlib_trace_has_no_backend_runtime` | stdlib trace has no backend runtime |
| &nbsp;&nbsp;**TestTorchEvaluatorBackendRuntime** | *TestTorchEvaluatorBackendRuntime* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_evaluator_sets_backend_runtime` | torch evaluator sets backend runtime |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_evaluator_backend_runtime_in_to_dict` | torch evaluator backend runtime in to dict |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_stdlib_evaluator_no_backend_runtime` | stdlib evaluator no backend runtime |
| **[test_p14_numerical_equivalence.py](../tests/test_p14_numerical_equivalence.py)** | Numerical equivalence: differentiable_python vs torch CPU atol=1e-5 (17 tests) |
| &nbsp;&nbsp;**TestEmailAgentEquivalence** | *TestEmailAgentEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_classifier_output_matches` | classifier output matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reply_activation_matches` | reply activation matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_value_matches` | action value matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_activated_agrees` | action activated agrees |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_float_state_keys_match` | all float state keys match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_inputs_agree` | multiple inputs agree |
| &nbsp;&nbsp;**TestFallRiskEquivalence** | *TestFallRiskEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_risk_model_output_matches` | risk model output matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_alert_activation_matches` | alert activation matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_value_matches` | action value matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_float_state_keys_match` | all float state keys match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_inputs_agree` | multiple inputs agree |
| &nbsp;&nbsp;**TestToleranceContract** | *TestToleranceContract* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_documented_atol_value` | documented atol value |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_documented_rtol_value` | documented rtol value |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_observed_diff_within_atol` | Spot-check: worst observed difference stays well below atol. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_observed_diff_fall_risk_within_atol` | max observed diff fall risk within atol |
| &nbsp;&nbsp;**TestCudaEquivalence** | *TestCudaEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cuda_matches_cpu_within_tolerance` | cuda matches cpu within tolerance |
| &nbsp;&nbsp;**TestMpsEquivalence** | *TestMpsEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mps_matches_cpu_within_tolerance` | mps matches cpu within tolerance |
| **[test_p14_torch_skip_activation.py](../tests/test_p14_torch_skip_activation.py)** | Torch graceful skip: tests skipped cleanly when PyTorch is not installed (9 tests) |
| &nbsp;&nbsp;**TestTorchAvailableInEnvironment** | *TestTorchAvailableInEnvironment* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_available_returns_true` | torch available returns true |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_importable` | torch importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_device_info_torch_available_true` | torch device info torch available true |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cpu_always_in_available_devices` | cpu always in available devices |
| &nbsp;&nbsp;**TestSkipUnlessGuardedTestsAreActive** | *Run the three target test files in isolation and confirm no skip for torch tests.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_forward_tests_active` | test_torch_forward.py: 3 skipUnless tests must not be skipped. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_training_tests_active` | test_torch_training.py: 4 skipUnless tests must not be skipped. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameters_torch_tests_active` | test_parameters.py: 3 skipUnless tests must not be skipped. |
| &nbsp;&nbsp;**TestAbsencePathsStillIntact** | *The 'when torch absent' code paths (lazy import, error handling) compile and are reachable.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tensor_bridge_import_is_lazy` | torch is NOT imported at module load — only inside functions. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_torch_available_function_exists` | torch available function exists |

## ONNX, WASM and Edge Export (P15) (315 tests)

| Test | Description |
|------|-------------|
| **[test_p15_cut1_onnx_export.py](../tests/test_p15_cut1_onnx_export.py)** | ONNX export: OnnxExporter, Opset 17, softmax_linear, sigmoid_linear, layer_call (54 tests) |
| &nbsp;&nbsp;**TestOnnxAvailable** | *TestOnnxAvailable* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_true_when_installed` | returns true when installed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_false_when_onnx_missing` | returns false when onnx missing |
| &nbsp;&nbsp;**TestOnnxImportError** | *TestOnnxImportError* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_raises_when_onnx_unavailable` | export raises when onnx unavailable |
| &nbsp;&nbsp;**TestUnsupportedKind** | *TestUnsupportedKind* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unsupported_kind_raises` | unsupported kind raises |
| &nbsp;&nbsp;**TestHashMismatchValidation** | *TestHashMismatchValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_parameter_set_raises` | ParameterSet trained on fall-risk must not export for email-agent. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_correct_parameter_set_succeeds` | Matching ParameterSet must export without error. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash_in_onnx_metadata` | Both model_hash and parameter_schema_hash must be embedded in ONNX metadata. |
| &nbsp;&nbsp;**TestSoftmaxLinearOnnxValidity** | *TestSoftmaxLinearOnnxValidity* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_file_exists` | output file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_checker_passes` | onnx checker passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_opset_version` | opset version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_input_shape` | result input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_output_shape` | result output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_input_name` | result input name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_output_name` | result output name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exported_function_name` | exported function name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_skipped_functions` | skipped functions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_set` | model hash set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_id_set` | parameter set id set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash_set` | parameter schema hash set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metadata_embedded` | metadata embedded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_keys` | to dict keys |
| &nbsp;&nbsp;**TestSigmoidLinearOnnxValidity** | *TestSigmoidLinearOnnxValidity* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_file_exists` | output file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_checker_passes` | onnx checker passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_input_shape` | result input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_output_shape` | result output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_input_name` | result input name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_output_name` | result output name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exported_function_name` | exported function name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_skipped_functions` | skipped functions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metadata_kind` | metadata kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_labels_empty_for_sigmoid` | labels empty for sigmoid |
| &nbsp;&nbsp;**TestSoftmaxInference** | *TestSoftmaxInference* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape_batch_1` | output shape batch 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape_batch_4` | output shape batch 4 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probabilities_sum_to_one` | probabilities sum to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probabilities_non_negative` | probabilities non negative |
| &nbsp;&nbsp;**TestSigmoidInference** | *TestSigmoidInference* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape_batch_1` | output shape batch 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape_batch_4` | output shape batch 4 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_probabilities_in_range` | probabilities in range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_specific_input` | specific input |
| &nbsp;&nbsp;**TestSoftmaxOnnxVsDiffPy** | *TestSoftmaxOnnxVsDiffPy* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_typical_input` | equivalence typical input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_uniform_input` | equivalence uniform input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_zero_input` | equivalence zero input |
| &nbsp;&nbsp;**TestSigmoidOnnxVsDiffPy** | *TestSigmoidOnnxVsDiffPy* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_typical_input` | equivalence typical input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_low_risk_input` | equivalence low risk input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_uniform_input` | equivalence uniform input |
| &nbsp;&nbsp;**TestOnnxDeterminism** | *TestOnnxDeterminism* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_deterministic` | softmax deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_deterministic` | sigmoid deterministic |
| &nbsp;&nbsp;**TestCliExportOnnx** | *TestCliExportOnnx* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_help_shows_export_onnx` | cli help shows export onnx |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_export_produces_file` | cli export produces file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_export_json_flag` | cli export json flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_fall_risk_export` | cli fall risk export |
| &nbsp;&nbsp;**TestExportOnnxWrapper** | *TestExportOnnxWrapper* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrapper_returns_result` | wrapper returns result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrapper_accepts_path_object` | wrapper accepts path object |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrapper_creates_parent_directory` | wrapper creates parent directory |
| **[test_p15_cut2_equivalence.py](../tests/test_p15_cut2_equivalence.py)** | ONNX equivalence: OnnxEquivalenceValidator, numpy.allclose vs onnxruntime (39 tests) |
| &nbsp;&nbsp;**TestOrtAvailable** | *TestOrtAvailable* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_true_when_installed` | returns true when installed |
| &nbsp;&nbsp;**TestEquivalenceResultStructure** | *TestEquivalenceResultStructure* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_is_dataclass` | is dataclass |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_keys` | to dict keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_atol_default` | atol default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rtol_default` | rtol default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_n_samples_default` | n samples default |
| &nbsp;&nbsp;**TestSoftmaxEquivalence** | *TestSoftmaxEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_passes` | passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_abs_diff_within_atol` | max abs diff within atol |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_n_outputs_per_sample` | n outputs per sample |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_abs_diff_non_negative` | max abs diff non negative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_rel_diff_non_negative` | max rel diff non negative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_custom_n_samples` | custom n samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_deterministic_with_seed` | deterministic with seed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_seeds_may_differ` | different seeds may differ |
| &nbsp;&nbsp;**TestSigmoidEquivalence** | *TestSigmoidEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_passes` | passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_abs_diff_within_atol` | max abs diff within atol |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_n_outputs_per_sample` | n outputs per sample |
| &nbsp;&nbsp;**TestExportManifest** | *TestExportManifest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_file_exists` | manifest file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_required_keys_present` | required keys present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_matches` | model hash matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash_matches` | parameter schema hash matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_id_matches` | parameter set id matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_format_is_onnx` | format is onnx |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_format_version_is_opset` | format version is opset |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tolerance_keys` | tolerance keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_check_passed` | equivalence check passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_check_max_abs_diff` | equivalence check max abs diff |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_check_n_samples` | equivalence check n samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exported_at_is_iso_string` | exported at is iso string |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_shape_correct` | input shape correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape_correct` | output shape correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_creates_parent_directories` | manifest creates parent directories |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_manifest` | fall risk manifest |
| &nbsp;&nbsp;**TestCliValidateFlag** | *TestCliValidateFlag* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_exits_zero` | validate exits zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_json_flag_includes_equivalence_check` | validate json flag includes equivalence check |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_manifest_flag_writes_file` | validate manifest flag writes file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_without_validate_warns` | manifest without validate warns |
| &nbsp;&nbsp;**TestValidateConvenienceWrapper** | *TestValidateConvenienceWrapper* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_onnx_equivalence_wrapper` | validate onnx equivalence wrapper |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validator_class_directly` | validator class directly |
| **[test_p15_cut3_edge_bundle.py](../tests/test_p15_cut3_edge_bundle.py)** | Edge bundle: EdgeBundler, model.onnx + model_manifest + export_manifest + README (35 tests) |
| &nbsp;&nbsp;**TestEmailAgentBundle** | *TestEmailAgentBundle* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_dir_exists` | bundle dir exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expected_files` | expected files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_mxai_content_matches` | model mxai content matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_params_best_json_is_valid_json` | params best json is valid json |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_onnx_passes_checker` | model onnx passes checker |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_keys` | model manifest keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_hashes` | model manifest hashes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_vectors` | model manifest vectors |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_functions` | model manifest functions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_backend_contract_ok` | model manifest backend contract ok |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_manifest_keys` | export manifest keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_manifest_hashes` | export manifest hashes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_manifest_format` | export manifest format |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_manifest_equivalence_passed` | export manifest equivalence passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_manifest_exported_at` | export manifest exported at |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_readme_exists_and_nonempty` | readme exists and nonempty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_readme_contains_hash` | readme contains hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_model_hash` | result model hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_parameter_set_id` | result parameter set id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_equivalence_passed` | result equivalence passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_to_dict_keys` | result to dict keys |
| &nbsp;&nbsp;**TestFallRiskBundle** | *TestFallRiskBundle* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_expected_files` | expected files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_project` | model manifest project |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_manifest_vectors_size_5` | model manifest vectors size 5 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_manifest_input_shape` | export manifest input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_passed` | equivalence passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_readme_contains_fall_risk` | readme contains fall risk |
| &nbsp;&nbsp;**TestBundleForceFlag** | *TestBundleForceFlag* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_existing_dir_raises_without_force` | existing dir raises without force |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_force_true_overwrites` | force true overwrites |
| &nbsp;&nbsp;**TestBundleNoValidate** | *TestBundleNoValidate* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_validate_skips_equivalence` | no validate skips equivalence |
| &nbsp;&nbsp;**TestCliBundleCommand** | *TestCliBundleCommand* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_bundle_creates_files` | cli bundle creates files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_bundle_json_output` | cli bundle json output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_bundle_no_validate` | cli bundle no validate |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_help_shows_export_bundle` | cli help shows export bundle |
| &nbsp;&nbsp;**TestCreateEdgeBundleWrapper** | *TestCreateEdgeBundleWrapper* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrapper_returns_result` | wrapper returns result |
| **[test_p15_cut4_p10_primitives.py](../tests/test_p15_cut4_p10_primitives.py)** | P10 primitives in ONNX: matmul, residual, gelu, layer_norm, attention ops (38 tests) |
| &nbsp;&nbsp;**TestTransformerOnnxExport** | *TestTransformerOnnxExport* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_file_exists` | onnx file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_checker_passes` | onnx checker passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_name_and_shape` | input name and shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_name_and_shape` | output name and shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_opset_version` | opset version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exported_functions` | exported functions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_skipped_functions_empty` | skipped functions empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_in_result` | model hash in result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash_in_result` | parameter schema hash in result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_metadata_model_hash` | onnx metadata model hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_metadata_kind` | onnx metadata kind |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_to_dict_keys` | to dict keys |
| &nbsp;&nbsp;**TestTransformerOnnxInference** | *TestTransformerOnnxInference* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_size_1` | batch size 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_size_4` | batch size 4 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_is_float32` | output is float32 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_deterministic_output` | deterministic output |
| &nbsp;&nbsp;**TestTransformerEquivalence** | *TestTransformerEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_validator_passes` | equivalence validator passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_abs_diff_within_atol` | max abs diff within atol |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_n_outputs_per_sample` | n outputs per sample |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manual_sample_equivalence` | manual sample equivalence |
| &nbsp;&nbsp;**TestEquivalenceValidatorLayerCall** | *TestEquivalenceValidatorLayerCall* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_returns_result` | validate returns result |
| &nbsp;&nbsp;**TestPrimitiveCoverage** | *TestPrimitiveCoverage* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matmul_present` | matmul present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_add_present` | add present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_norm_present` | layer norm present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tanh_present` | tanh present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_present` | sigmoid present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reduce_sum_present` | reduce sum present |
| &nbsp;&nbsp;**TestSupportedKinds** | *TestSupportedKinds* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_in_supported_kinds` | layer call in supported kinds |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_linear_still_supported` | softmax linear still supported |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_linear_still_supported` | sigmoid linear still supported |
| &nbsp;&nbsp;**TestUnsupportedPrimitive** | *TestUnsupportedPrimitive* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unsupported_op_kind_raises_with_name` | A layer with an unrecognized primitive should raise OnnxExportError naming it. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_mismatch_raises` | hash mismatch raises |
| &nbsp;&nbsp;**TestTransformerEdgeBundle** | *TestTransformerEdgeBundle* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_created_with_6_files` | bundle created with 6 files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_equivalence_passed` | bundle equivalence passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_model_manifest_project` | bundle model manifest project |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_export_manifest_format` | bundle export manifest format |
| &nbsp;&nbsp;**TestExistingModelsUnaffected** | *TestExistingModelsUnaffected* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_still_exports` | email agent still exports |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_still_exports` | fall risk still exports |
| **[test_p15_cut5_transformer_p11.py](../tests/test_p15_cut5_transformer_p11.py)** | Transformer ONNX: P11 encoder exported and validated against onnxruntime (30 tests) |
| &nbsp;&nbsp;**TestSequenceTransformerExport** | *TestSequenceTransformerExport* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_file_exists` | onnx file exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_checker_passes` | onnx checker passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_is_int64` | input is int64 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_name_and_shape` | input name and shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_name_and_shape` | output name and shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exported_functions` | exported functions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_skipped_functions_empty` | skipped functions empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_opset_version` | opset version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_in_result` | model hash in result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_metadata_kind` | onnx metadata kind |
| &nbsp;&nbsp;**TestSequencePrimitiveCoverage** | *TestSequencePrimitiveCoverage* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_gather_present` | gather present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reduce_mean_present` | reduce mean present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reduce_sum_present` | reduce sum present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_present` | softmax present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_normalization_present` | layer normalization present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matmul_present` | matmul present |
| &nbsp;&nbsp;**TestSequenceTransformerInference** | *TestSequenceTransformerInference* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_size_1` | batch size 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_batch_size_3` | batch size 3 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_is_float32` | output is float32 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ids_in_range_respected` | ids in range respected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_deterministic_output` | deterministic output |
| &nbsp;&nbsp;**TestSequenceTransformerEquivalence** | *TestSequenceTransformerEquivalence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_validator_passes` | equivalence validator passes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_max_abs_diff_within_atol` | max abs diff within atol |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_n_outputs_per_sample` | n outputs per sample |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manual_sample_equivalence` | manual sample equivalence |
| &nbsp;&nbsp;**TestEquivalenceValidatorSequence** | *TestEquivalenceValidatorSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_returns_passing_result` | validate returns passing result |
| &nbsp;&nbsp;**TestSequenceTransformerEdgeBundle** | *TestSequenceTransformerEdgeBundle* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_created` | bundle created |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_input_shape_in_manifest` | bundle input shape in manifest |
| &nbsp;&nbsp;**TestVectorModelsUnaffected** | *TestVectorModelsUnaffected* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_vector_still_works` | transformer vector still works |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_still_works` | email agent still works |
| **[test_p15_cut6_wasm.py](../tests/test_p15_cut6_wasm.py)** | WASM bundle: WasmExporter, predict.js, wasm_manifest.json, ORT Web compatibility (68 tests) |
| &nbsp;&nbsp;**TestWasmBundleSequence** | *TestWasmBundleSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_dir_created` | bundle dir created |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_three_files_present` | three files present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_onnx_exists` | model onnx exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wasm_manifest_exists` | wasm manifest exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_js_exists` | predict js exists |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wasm_runtime_field` | wasm runtime field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash` | model hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_id` | parameter set id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash` | parameter schema hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_shape` | input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape` | output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_opset_version` | opset version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_passed` | equivalence passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_result_not_none` | equivalence result not none |
| &nbsp;&nbsp;**TestWasmManifestSequence** | *TestWasmManifestSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_format_field` | format field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wasm_runtime_field` | wasm runtime field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ort_web_min_version` | ort web min version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash` | model hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash` | parameter schema hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_id` | parameter set id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_name` | input name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_shape` | input shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_name` | output name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_shape` | output shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_opset` | onnx opset |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_check_passed` | equivalence check passed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tolerance_present` | tolerance present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exported_at_present` | exported at present |
| &nbsp;&nbsp;**TestPredictJsSequence** | *TestPredictJsSequence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ort_cdn_reference` | ort cdn reference |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_predict_function_present` | predict function present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_name_present` | input name present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_output_name_present` | output name present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_present` | model hash present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_int64_dtype` | int64 dtype |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bigint64array` | bigint64array |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_float32array` | no float32array |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_onnx_reference` | model onnx reference |
| &nbsp;&nbsp;**TestPredictJsVector** | *TestPredictJsVector* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_float32_dtype` | float32 dtype |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_float32array` | float32array |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_bigint64array` | no bigint64array |
| &nbsp;&nbsp;**TestWasmModelOnnxValid** | *TestWasmModelOnnxValid* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_checker_passes_sequence` | onnx checker passes sequence |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_onnx_checker_passes_vector` | onnx checker passes vector |
| &nbsp;&nbsp;**TestWasmExportResultToDict** | *TestWasmExportResultToDict* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bundle_dir_key` | bundle dir key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_files_key` | files key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_key` | model hash key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wasm_runtime_key` | wasm runtime key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_passed_key` | equivalence passed key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_equivalence_check_key` | equivalence check key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_input_output_shapes` | input output shapes |
| &nbsp;&nbsp;**TestWasmNoValidate** | *TestWasmNoValidate* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_without_validate` | export without validate |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_manifest_no_tolerance_when_no_validate` | manifest no tolerance when no validate |
| &nbsp;&nbsp;**TestWasmForceFlag** | *TestWasmForceFlag* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_force_false_raises_on_existing_dir` | force false raises on existing dir |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_force_true_overwrites` | force true overwrites |
| &nbsp;&nbsp;**TestExportWasmFunction** | *TestExportWasmFunction* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_wasm_returns_result` | export wasm returns result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_wasm_vector_model` | export wasm vector model |
| &nbsp;&nbsp;**TestCliExportWasm** | *TestCliExportWasm* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_export_wasm_exits_zero` | cli export wasm exits zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_export_wasm_json_flag` | cli export wasm json flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_force_flag` | cli force flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_no_force_existing_dir_fails` | cli no force existing dir fails |
| &nbsp;&nbsp;**TestWasmHashMismatch** | *TestWasmHashMismatch* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_mismatch_raises` | hash mismatch raises |
| &nbsp;&nbsp;**TestWasmNodeJsValidation** | *Validate WASM bundle with onnxruntime-node (same runtime as onnxruntime-web).* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_node_validates_vector_model` | node validates vector model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_node_validates_sequence_model` | node validates sequence model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_node_output_shape_matches_manifest` | node output shape matches manifest |
| &nbsp;&nbsp;**TestWasmOrtWebValidation** | *Validate WASM bundle with onnxruntime-web WASM execution provider.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ortWeb_wasm_backend_vector_model` | ortWeb wasm backend vector model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ortWeb_wasm_backend_sequence_model` | ortWeb wasm backend sequence model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ortWeb_wasm_output_shape_correct` | Output shape from onnxruntime-web WASM matches wasm_manifest.json. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ortWeb_wasm_deterministic` | Two runs with same input produce identical output (WASM is deterministic). |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ortWeb_wasm_matches_python_ort` | onnxruntime-web WASM output matches onnxruntime Python/CPU within atol=1e-4. |
| **[test_p15_cut7_regression.py](../tests/test_p15_cut7_regression.py)** | ONNX/WASM regression: P1–P14 import integrity, export pipeline (51 tests) |
| &nbsp;&nbsp;**TestCoreParseValidate** | *P1-P3 regression: parse and validate work for all archetypes.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_email_agent` | parse email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_fall_risk` | parse fall risk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_transformer_vector` | parse transformer vector |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parse_transformer_sequence` | parse transformer sequence |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_email_agent` | validate email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_fall_risk` | validate fall risk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_validate_transformer_vector` | validate transformer vector |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_backend_contract_email_agent` | backend contract email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_compile_dp_email_agent` | compile dp email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_compile_dp_transformer_vector` | compile dp transformer vector |
| &nbsp;&nbsp;**TestParameterSetCoreFlows** | *P4 regression: ParameterSet build, validate, and hashing work.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_parameter_set_email_agent` | build parameter set email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_has_hashes` | parameter set has hashes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_hash_stable` | parameter set hash stable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_write_and_load_parameter_set` | write and load parameter set |
| &nbsp;&nbsp;**TestRuntimeInferenceUnaffected** | *P5/P10/P11 regression: differentiable_python runtime produces correct output.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dp_email_agent_produces_output` | dp email agent produces output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dp_fall_risk_produces_output` | dp fall risk produces output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dp_transformer_vector_produces_output` | dp transformer vector produces output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dp_transformer_sequence_produces_output` | dp transformer sequence produces output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dp_output_deterministic_email_agent` | dp output deterministic email agent |
| &nbsp;&nbsp;**TestExportNoMutation** | *Export operations must not alter program IR or ParameterSet state.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_does_not_mutate_program_email_agent` | export does not mutate program email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_does_not_mutate_parameter_set_email_agent` | export does not mutate parameter set email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_runtime_output_unchanged_after_export` | DP runtime produces identical output before and after exporting to ONNX. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_transformer_vec_no_mutation` | export transformer vec no mutation |
| &nbsp;&nbsp;**TestAllArchetypesOnnxExport** | *Regression: all four archetypes produce valid ONNX files.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_linear_email_agent` | softmax linear email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_linear_fall_risk` | sigmoid linear fall risk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_vector` | layer call vector |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_sequence` | layer call sequence |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_embedded_email_agent` | hash embedded email agent |
| &nbsp;&nbsp;**TestAllArchetypesEquivalence** | *Regression: OnnxEquivalenceValidator passes for all archetypes.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_softmax_linear_email_agent` | softmax linear email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sigmoid_linear_fall_risk` | sigmoid linear fall risk |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_vector` | layer call vector |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_layer_call_sequence` | layer call sequence |
| &nbsp;&nbsp;**TestAllArchetypesEdgeBundle** | *Regression: EdgeBundler produces correct 6-file bundles for all archetypes.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_bundle_files` | email agent bundle files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_bundle_files` | fall risk bundle files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_vector_bundle_files` | transformer vector bundle files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_sequence_bundle_files` | transformer sequence bundle files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_bundles_equivalence_passed` | all bundles equivalence passed |
| &nbsp;&nbsp;**TestAllArchetypesWasmBundle** | *Regression: WasmExporter produces correct 3-file bundles for all archetypes.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_agent_wasm_files` | email agent wasm files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fall_risk_wasm_files` | fall risk wasm files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_vector_wasm_files` | transformer vector wasm files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_transformer_sequence_wasm_files` | transformer sequence wasm files |
| &nbsp;&nbsp;**TestHashStability** | *model_hash and parameter_schema_hash are deterministic across exports.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_stable_email_agent` | model hash stable email agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_stable_transformer_vec` | model hash stable transformer vec |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_differs_between_models` | model hash differs between models |
| &nbsp;&nbsp;**TestImportIsolation** | *Core matrixai modules must not import onnx or onnxruntime at load time.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_core_importable` | matrixai core importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_runtime_importable` | matrixai runtime importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_compiler_importable` | matrixai compiler importable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_export_module_symbols_accessible` | export module symbols accessible |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ort_available_returns_bool` | ort available returns bool |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ort_available_true_when_installed` | ort available true when installed |
| &nbsp;&nbsp;**TestUnsupportedModelExplicitError** | *P15 contract: unsupported models raise OnnxExportError with clear message.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_mismatch_raises_onnx_export_error` | hash mismatch raises onnx export error |

## Real Actions and Action Contracts (P20) (281 tests)

| Test | Description |
|------|-------------|
| **[test_p20_cut1_mxact_parser.py](../tests/test_p20_cut1_mxact_parser.py)** | .mxact parser: ACTION_CONTRACT, SCOPE, SANDBOX_LIMITS, ROLLBACK blocks (25 tests) |
| &nbsp;&nbsp;**TestMxactParserEmailSend** | *TestMxactParserEmailSend* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_accepts_email_send_contract` | mxact parser accepts email send contract |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_contract_scope_parsed` | email contract scope parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_contract_rollback_resolved` | email contract rollback resolved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_contract_rate_limit` | email contract rate limit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_contract_flags` | email contract flags |
| &nbsp;&nbsp;**TestMxactParserHttpPost** | *TestMxactParserHttpPost* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_accepts_http_post_with_rollback` | mxact parser accepts http post with rollback |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_post_scope_url_list` | http post scope url list |
| &nbsp;&nbsp;**TestMxactParserFilesystemWrite** | *TestMxactParserFilesystemWrite* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_accepts_filesystem_write_with_sandbox` | mxact parser accepts filesystem write with sandbox |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_filesystem_sandbox_limits_parsed` | filesystem sandbox limits parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_filesystem_rollback_scope` | filesystem rollback scope |
| &nbsp;&nbsp;**TestMxactParserRejections** | *TestMxactParserRejections* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_rejects_unknown_capability` | mxact parser rejects unknown capability |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_rejects_high_risk_without_sandbox` | mxact parser rejects high risk without sandbox |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_rejects_human_approval_without_channel` | mxact parser rejects human approval without channel |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_rejects_wildcard_only_scope` | mxact parser rejects wildcard only scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_rejects_wildcard_path_scope` | mxact parser rejects wildcard path scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxact_parser_rejects_missing_rollback_for_mutating_action` | mxact parser rejects missing rollback for mutating action |
| &nbsp;&nbsp;**TestMxaiParserP20Extensions** | *TestMxaiParserP20Extensions* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_accepts_policy_real_with_audit` | mxai parser accepts policy real with audit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_action_has_target` | mxai parser action has target |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_action_condition_from_condition_keyword` | mxai parser action condition from condition keyword |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_input_params_parsed` | mxai parser input params parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_keeps_simulate_only_as_default` | mxai parser keeps simulate only as default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mxai_parser_simulate_only_call_preserved` | mxai parser simulate only call preserved |
| &nbsp;&nbsp;**TestCapabilityRegistry** | *TestCapabilityRegistry* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_nine_capabilities_present` | all nine capabilities present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_high_risk_capabilities_classified` | high risk capabilities classified |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mutating_capabilities_require_rollback` | mutating capabilities require rollback |
| **[test_p20_cut2_action_contract.py](../tests/test_p20_cut2_action_contract.py)** | Action contract: canonical hash SHA256, validate_action_contract, require_signing_key (20 tests) |
| &nbsp;&nbsp;**TestActionContractHash** | *TestActionContractHash* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_hash_is_deterministic` | action contract hash is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_hash_has_sha256_prefix` | action contract hash has sha256 prefix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_hash_is_hex_string` | action contract hash is hex string |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_hash_changes_when_scope_changes` | action contract hash changes when scope changes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_hash_changes_when_rollback_changes` | action contract hash changes when rollback changes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_hash_changes_when_capability_changes` | action contract hash changes when capability changes |
| &nbsp;&nbsp;**TestCanonicalDict** | *TestCanonicalDict* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_canonical_dict_includes_risk_level` | canonical dict includes risk level |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_canonical_dict_scope_keys_sorted` | canonical dict scope keys sorted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_canonical_dict_includes_rollback` | canonical dict includes rollback |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_canonical_dict_rollback_scope_sorted` | canonical dict rollback scope sorted |
| &nbsp;&nbsp;**TestValidateActionContract** | *TestValidateActionContract* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_compatibility_with_mxai_action` | action contract compatibility with mxai action |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_validate_result_ok_when_compatible` | action contract validate result ok when compatible |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_rejects_action_not_found_in_program` | action contract rejects action not found in program |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_rejects_capability_mismatch` | action contract rejects capability mismatch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_rejects_simulate_only_policy` | action contract rejects simulate only policy |
| &nbsp;&nbsp;**TestSigningKey** | *TestSigningKey* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_signing_key_available_returns_true_when_set` | check signing key available returns true when set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_signing_key_available_returns_false_when_missing` | check signing key available returns false when missing |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_rejects_signature_without_key` | action contract rejects signature without key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_accepts_signature_with_key` | action contract accepts signature with key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_contract_validate_accepts_no_signing_key_when_not_required` | action contract validate accepts no signing key when not required |
| **[test_p20_cut3_capability_registry.py](../tests/test_p20_cut3_capability_registry.py)** | Capability registry: 9 capabilities, validate_scope, resolve_scope, REQUIRED_SCOPE_FIELDS (29 tests) |
| &nbsp;&nbsp;**TestCapabilityRegistryListing** | *TestCapabilityRegistryListing* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capability_registry_lists_all_supported_capabilities` | capability registry lists all supported capabilities |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capability_registry_returns_sorted_list` | capability registry returns sorted list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capability_registry_rejects_unknown_capability` | capability registry rejects unknown capability |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_rejects_unknown_in_required_scope_fields` | registry rejects unknown in required scope fields |
| &nbsp;&nbsp;**TestCapabilityRegistryRiskLevels** | *TestCapabilityRegistryRiskLevels* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capability_registry_classifies_risk_levels` | capability registry classifies risk levels |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_low_for_http_get` | registry risk level low for http get |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_low_for_notification` | registry risk level low for notification |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_medium_for_email_send` | registry risk level medium for email send |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_medium_for_http_post` | registry risk level medium for http post |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_high_for_database_write` | registry risk level high for database write |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_high_for_filesystem_write` | registry risk level high for filesystem write |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_risk_level_high_for_subprocess_spawn` | registry risk level high for subprocess spawn |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capability_high_risk_forces_sandbox_required` | capability high risk forces sandbox required |
| &nbsp;&nbsp;**TestRequiredScopeFields** | *TestRequiredScopeFields* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_capability_registry_validates_required_scope_fields` | capability registry validates required scope fields |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_required_scope_email_send_includes_recipients_and_domains` | registry required scope email send includes recipients and domains |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_required_scope_http_get_includes_allowed_urls` | registry required scope http get includes allowed urls |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_required_scope_filesystem_write_includes_allowed_paths` | registry required scope filesystem write includes allowed paths |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_required_scope_database_write_includes_tables_and_operations` | registry required scope database write includes tables and operations |
| &nbsp;&nbsp;**TestScopeValidation** | *TestScopeValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_validate_scope_accepts_complete_scope` | registry validate scope accepts complete scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_validate_scope_rejects_missing_required_field` | registry validate scope rejects missing required field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_validate_scope_rejects_unknown_capability` | registry validate scope rejects unknown capability |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_validate_scope_effective_scope_populated_on_success` | registry validate scope effective scope populated on success |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_resolve_scope_returns_effective_scope` | registry resolve scope returns effective scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_resolve_scope_raises_on_invalid_scope` | registry resolve scope raises on invalid scope |
| &nbsp;&nbsp;**TestRollbackRequirement** | *TestRollbackRequirement* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_requires_rollback_for_email_send` | registry requires rollback for email send |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_requires_rollback_for_http_post` | registry requires rollback for http post |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_requires_rollback_for_database_write` | registry requires rollback for database write |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_does_not_require_rollback_for_http_get` | registry does not require rollback for http get |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_does_not_require_rollback_for_database_read` | registry does not require rollback for database read |
| **[test_p20_cut4_dryrun.py](../tests/test_p20_cut4_dryrun.py)** | Dry-run simulator: DryRunReport, RateTracker sliding window, input type validation (20 tests) |
| &nbsp;&nbsp;**TestDryRunReportStructure** | *TestDryRunReportStructure* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_has_report_id` | dry run report has report id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_has_valid_until_field` | dry run report has valid until field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_has_input_hash` | dry run report has input hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_has_action_contract_hash` | dry run report has action contract hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_ok_when_all_valid` | dry run report ok when all valid |
| &nbsp;&nbsp;**TestDryRunExpiry** | *TestDryRunExpiry* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_expires_after_five_minutes` | dry run report expires after five minutes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_valid_until_is_five_minutes_after_executed_at` | dry run valid until is five minutes after executed at |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_validity_minutes_configurable` | dry run validity minutes configurable |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_validity_capped_at_one_hour` | dry run validity capped at one hour |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_is_not_expired_before_valid_until` | dry run report is not expired before valid until |
| &nbsp;&nbsp;**TestDryRunScopeValidation** | *TestDryRunScopeValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_validates_scope_against_model_output` | dry run validates scope against model output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_blocks_execution_with_out_of_scope_recipient` | dry run blocks execution with out of scope recipient |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_report_not_ok_when_scope_violated` | dry run report not ok when scope violated |
| &nbsp;&nbsp;**TestDryRunInputTypes** | *TestDryRunInputTypes* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_validates_input_types` | dry run validates input types |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_blocks_missing_required_input_field` | dry run blocks missing required input field |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_blocks_wrong_input_type` | dry run blocks wrong input type |
| &nbsp;&nbsp;**TestDryRunRateLimit** | *TestDryRunRateLimit* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_validates_rate_limit` | dry run validates rate limit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_blocks_execution_when_rate_limit_exceeded` | dry run blocks execution when rate limit exceeded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_rate_tracker_records_call` | dry run rate tracker records call |
| &nbsp;&nbsp;**TestDryRunRollback** | *TestDryRunRollback* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_validates_rollback_invocability` | dry run validates rollback invocability |
| **[test_p20_cut5_executor.py](../tests/test_p20_cut5_executor.py)** | Action executor: in-process handlers, pre-flight checks, HIGH_RISK capability rejection (23 tests) |
| &nbsp;&nbsp;**TestExecutorPreFlight** | *TestExecutorPreFlight* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_without_allow_real_actions_flag` | executor blocks without allow real actions flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_without_signing_key` | executor blocks without signing key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_without_valid_dry_run` | executor blocks without valid dry run |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_expired_dry_run` | executor blocks expired dry run |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_dry_run_contract_hash_mismatch` | executor blocks dry run contract hash mismatch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_dry_run_input_hash_mismatch` | executor blocks dry run input hash mismatch |
| &nbsp;&nbsp;**TestHttpGetExecutor** | *TestHttpGetExecutor* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_get_executor_respects_scope_urls` | http get executor respects scope urls |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_get_executor_rejects_url_not_in_scope` | http get executor rejects url not in scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_get_executor_respects_timeout` | http get executor respects timeout |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_result_ok_on_successful_http_get` | action result ok on successful http get |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_result_has_latency_ms` | action result has latency ms |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_result_has_executor_kind_in_process` | action result has executor kind in process |
| &nbsp;&nbsp;**TestHttpPostExecutor** | *TestHttpPostExecutor* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_post_executor_rejects_url_not_in_scope` | http post executor rejects url not in scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_post_executor_validates_required_headers` | http post executor validates required headers |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_http_post_executor_rejects_missing_required_header` | http post executor rejects missing required header |
| &nbsp;&nbsp;**TestEmailExecutor** | *TestEmailExecutor* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_executor_respects_allowed_recipients` | email executor respects allowed recipients |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_executor_rejects_recipient_not_in_list` | email executor rejects recipient not in list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_email_executor_rejects_domain_not_in_allowed_list` | email executor rejects domain not in allowed list |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_returns_action_result` | executor returns action result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_action_result_has_error_field_on_failure` | action result has error field on failure |
| &nbsp;&nbsp;**TestNotificationExecutor** | *TestNotificationExecutor* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_executor_sends_to_webhook` | notification executor sends to webhook |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_executor_rejects_recipient_not_in_scope` | notification executor rejects recipient not in scope |
| &nbsp;&nbsp;**TestExecutorBlocksHighRisk** | *TestExecutorBlocksHighRisk* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_executor_blocks_high_risk_capability` | executor blocks high risk capability |
| **[test_p20_cut6_sandbox.py](../tests/test_p20_cut6_sandbox.py)** | Sandboxed executor: POSIX resource limits, SIGKILL on timeout, scope enforcement (23 tests) |
| &nbsp;&nbsp;`test_sandboxed_executor_importable` | sandboxed executor importable |
| &nbsp;&nbsp;`test_sandbox_params_dataclass` | sandbox params dataclass |
| &nbsp;&nbsp;`test_sandbox_result_dataclass` | sandbox result dataclass |
| &nbsp;&nbsp;`test_sandboxed_executor_accepts_injectable_runner` | sandboxed executor accepts injectable runner |
| &nbsp;&nbsp;`test_sandbox_params_built_from_contract` | sandbox params built from contract |
| &nbsp;&nbsp;`test_sandbox_params_include_scope_and_input` | sandbox params include scope and input |
| &nbsp;&nbsp;`test_scope_check_filesystem_blocks_disallowed_path` | scope check filesystem blocks disallowed path |
| &nbsp;&nbsp;`test_scope_check_filesystem_allows_valid_path` | scope check filesystem allows valid path |
| &nbsp;&nbsp;`test_scope_check_database_blocks_disallowed_table` | scope check database blocks disallowed table |
| &nbsp;&nbsp;`test_scope_check_database_allows_valid_table` | scope check database allows valid table |
| &nbsp;&nbsp;`test_scope_check_unknown_capability_returns_none` | scope check unknown capability returns none |
| &nbsp;&nbsp;`test_executor_returns_error_on_scope_violation` | executor returns error on scope violation |
| &nbsp;&nbsp;`test_preflight_rejects_disabled_real_actions` | preflight rejects disabled real actions |
| &nbsp;&nbsp;`test_preflight_rejects_expired_dry_run` | preflight rejects expired dry run |
| &nbsp;&nbsp;`test_preflight_rejects_contract_hash_mismatch` | preflight rejects contract hash mismatch |
| &nbsp;&nbsp;`test_preflight_rejects_input_hash_mismatch` | preflight rejects input hash mismatch |
| &nbsp;&nbsp;`test_preflight_rejects_non_high_risk_capability` | SandboxedActionExecutor must only accept HIGH_RISK capabilities. |
| &nbsp;&nbsp;`test_result_ok_on_exit_code_zero` | result ok on exit code zero |
| &nbsp;&nbsp;`test_result_error_on_nonzero_exit` | result error on nonzero exit |
| &nbsp;&nbsp;`test_result_error_on_timeout` | result error on timeout |
| &nbsp;&nbsp;`test_result_has_positive_latency_ms` | result has positive latency ms |
| &nbsp;&nbsp;`test_default_runner_executes_simple_script` | default runner executes simple script |
| &nbsp;&nbsp;`test_default_runner_sets_network_env_when_no_network` | default runner sets network env when no network |
| **[test_p20_cut7_trace.py](../tests/test_p20_cut7_trace.py)** | Action trace: ActionTrace, HMAC-SHA256 canonical signing, verify_action_trace (21 tests) |
| &nbsp;&nbsp;`test_action_trace_importable` | action trace importable |
| &nbsp;&nbsp;`test_action_trace_dataclass_fields` | action trace dataclass fields |
| &nbsp;&nbsp;`test_action_trace_executed_at_is_iso8601` | action trace executed at is iso8601 |
| &nbsp;&nbsp;`test_action_trace_without_signing_key_has_none_signature` | action trace without signing key has none signature |
| &nbsp;&nbsp;`test_sign_action_trace_returns_hmac_prefix` | sign action trace returns hmac prefix |
| &nbsp;&nbsp;`test_sign_action_trace_hex_is_64_chars` | sign action trace hex is 64 chars |
| &nbsp;&nbsp;`test_sign_action_trace_is_deterministic` | sign action trace is deterministic |
| &nbsp;&nbsp;`test_build_action_trace_with_signing_key_sets_signature` | build action trace with signing key sets signature |
| &nbsp;&nbsp;`test_canonical_message_contains_all_fields` | canonical message contains all fields |
| &nbsp;&nbsp;`test_canonical_message_ok_field_is_tamper_detectable` | canonical message ok field is tamper detectable |
| &nbsp;&nbsp;`test_canonical_message_response_summary_is_tamper_detectable` | canonical message response summary is tamper detectable |
| &nbsp;&nbsp;`test_canonical_message_error_field_is_tamper_detectable` | canonical message error field is tamper detectable |
| &nbsp;&nbsp;`test_canonical_message_latency_ms_is_tamper_detectable` | canonical message latency ms is tamper detectable |
| &nbsp;&nbsp;`test_verify_action_trace_returns_true_for_valid_signature` | verify action trace returns true for valid signature |
| &nbsp;&nbsp;`test_verify_action_trace_returns_false_for_wrong_key` | verify action trace returns false for wrong key |
| &nbsp;&nbsp;`test_verify_action_trace_returns_false_when_signature_none` | verify action trace returns false when signature none |
| &nbsp;&nbsp;`test_verify_action_trace_returns_false_for_tampered_model_hash` | verify action trace returns false for tampered model hash |
| &nbsp;&nbsp;`test_verify_action_trace_returns_false_for_tampered_executed_at` | verify action trace returns false for tampered executed at |
| &nbsp;&nbsp;`test_build_action_trace_from_sandbox_result` | build action trace from sandbox result |
| &nbsp;&nbsp;`test_build_action_trace_from_failed_result` | build action trace from failed result |
| &nbsp;&nbsp;`test_sign_matches_manual_hmac` | sign matches manual hmac |
| **[test_p20_cut8_rollback.py](../tests/test_p20_cut8_rollback.py)** | Rollback manager: execute_rollback, RollbackResult, RollbackSpec materialisation (16 tests) |
| &nbsp;&nbsp;`test_rollback_manager_importable` | rollback manager importable |
| &nbsp;&nbsp;`test_rollback_result_importable` | rollback result importable |
| &nbsp;&nbsp;`test_rollback_error_is_action_executor_error` | rollback error is action executor error |
| &nbsp;&nbsp;`test_rollback_not_attempted_when_no_rollback_declared` | rollback not attempted when no rollback declared |
| &nbsp;&nbsp;`test_rollback_executed_with_injected_handler` | rollback executed with injected handler |
| &nbsp;&nbsp;`test_rollback_result_names_rollback_contract` | rollback result names rollback contract |
| &nbsp;&nbsp;`test_rollback_result_not_ok_when_handler_fails` | rollback result not ok when handler fails |
| &nbsp;&nbsp;`test_rollback_spec_to_contract_inherits_capability` | rollback spec to contract inherits capability |
| &nbsp;&nbsp;`test_rollback_spec_to_contract_dry_run_not_required` | rollback spec to contract dry run not required |
| &nbsp;&nbsp;`test_rollback_spec_to_contract_no_nested_rollback` | rollback spec to contract no nested rollback |
| &nbsp;&nbsp;`test_rollback_spec_to_contract_no_sandbox_required` | rollback spec to contract no sandbox required |
| &nbsp;&nbsp;`test_rollback_spec_to_contract_not_signature_required` | rollback spec to contract not signature required |
| &nbsp;&nbsp;`test_rollback_spec_to_contract_inherits_scope` | rollback spec to contract inherits scope |
| &nbsp;&nbsp;`test_rollback_works_for_http_post_capability` | rollback works for http post capability |
| &nbsp;&nbsp;`test_rollback_dry_run_does_not_expire_immediately_with_default_now` | Regression: _build_rollback_dry_run used to set valid_until == now, |
| &nbsp;&nbsp;`test_rollback_dry_run_has_validity_window` | The synthesized rollback DryRunReport should have valid_until strictly |
| **[test_p20_cut9_approval.py](../tests/test_p20_cut9_approval.py)** | Human-in-the-loop: PendingExecution, ApprovalStore, HumanApprovalGate, expiry (27 tests) |
| &nbsp;&nbsp;`test_pending_execution_importable` | pending execution importable |
| &nbsp;&nbsp;`test_approval_store_importable` | approval store importable |
| &nbsp;&nbsp;`test_human_approval_gate_importable` | human approval gate importable |
| &nbsp;&nbsp;`test_submit_creates_pending_execution` | submit creates pending execution |
| &nbsp;&nbsp;`test_submit_sets_expiry_from_ttl` | submit sets expiry from ttl |
| &nbsp;&nbsp;`test_submit_sets_channel` | submit sets channel |
| &nbsp;&nbsp;`test_submit_stores_action_contract_hash` | submit stores action contract hash |
| &nbsp;&nbsp;`test_get_returns_submitted_pending` | get returns submitted pending |
| &nbsp;&nbsp;`test_get_returns_none_for_unknown_id` | get returns none for unknown id |
| &nbsp;&nbsp;`test_approve_changes_status_to_approved` | approve changes status to approved |
| &nbsp;&nbsp;`test_reject_changes_status_to_rejected` | reject changes status to rejected |
| &nbsp;&nbsp;`test_approve_raises_on_already_decided` | approve raises on already decided |
| &nbsp;&nbsp;`test_reject_raises_on_already_decided` | reject raises on already decided |
| &nbsp;&nbsp;`test_approve_raises_on_expired` | approve raises on expired |
| &nbsp;&nbsp;`test_approve_raises_on_unknown_id` | approve raises on unknown id |
| &nbsp;&nbsp;`test_pending_not_expired_before_ttl` | pending not expired before ttl |
| &nbsp;&nbsp;`test_pending_expired_after_ttl` | pending expired after ttl |
| &nbsp;&nbsp;`test_is_approved_false_when_expired` | is approved false when expired |
| &nbsp;&nbsp;`test_list_pending_returns_only_pending` | list pending returns only pending |
| &nbsp;&nbsp;`test_list_pending_excludes_expired` | list pending excludes expired |
| &nbsp;&nbsp;`test_gate_requires_approval_when_human_approval_true` | gate requires approval when human approval true |
| &nbsp;&nbsp;`test_gate_does_not_require_approval_when_false` | gate does not require approval when false |
| &nbsp;&nbsp;`test_gate_check_true_when_no_approval_required` | gate check true when no approval required |
| &nbsp;&nbsp;`test_gate_check_false_when_approval_required_but_none_submitted` | gate check false when approval required but none submitted |
| &nbsp;&nbsp;`test_gate_check_true_after_approval` | gate check true after approval |
| &nbsp;&nbsp;`test_gate_check_false_after_rejection` | gate check false after rejection |
| &nbsp;&nbsp;`test_gate_check_false_when_approval_expired` | gate check false when approval expired |
| **[test_p20_cut10_cli.py](../tests/test_p20_cut10_cli.py)** | Action CLI: validate-actions, dry-run-action, execute-action, audit-action commands (15 tests) |
| &nbsp;&nbsp;`test_validate_actions_ok` | validate actions ok |
| &nbsp;&nbsp;`test_validate_actions_json_output` | validate actions json output |
| &nbsp;&nbsp;`test_validate_actions_fail_on_wrong_target` | validate actions fail on wrong target |
| &nbsp;&nbsp;`test_validate_actions_fail_on_bad_contract_file` | validate actions fail on bad contract file |
| &nbsp;&nbsp;`test_dry_run_action_ok` | dry run action ok |
| &nbsp;&nbsp;`test_dry_run_action_json_output` | dry run action json output |
| &nbsp;&nbsp;`test_dry_run_action_missing_contract_name` | dry run action missing contract name |
| &nbsp;&nbsp;`test_dry_run_action_scope_violation` | dry run action scope violation |
| &nbsp;&nbsp;`test_execute_action_requires_allow_flag` | execute action requires allow flag |
| &nbsp;&nbsp;`test_execute_action_json_output_with_flag` | execute action json output with flag |
| &nbsp;&nbsp;`test_execute_action_missing_contract` | execute action missing contract |
| &nbsp;&nbsp;`test_audit_action_no_key_reports_not_verified` | audit action no key reports not verified |
| &nbsp;&nbsp;`test_audit_action_valid_signature` | audit action valid signature |
| &nbsp;&nbsp;`test_audit_action_invalid_signature` | audit action invalid signature |
| &nbsp;&nbsp;`test_audit_action_json_output` | audit action json output |
| &nbsp;&nbsp;`test_action_contract_view_importable` | action contract view importable |
| &nbsp;&nbsp;`test_build_action_contract_view_fields` | build action contract view fields |
| &nbsp;&nbsp;`test_build_action_contract_view_scope_summary` | build action contract view scope summary |
| &nbsp;&nbsp;`test_build_action_contract_view_rate_limit_summary` | build action contract view rate limit summary |
| &nbsp;&nbsp;`test_build_action_contract_view_sandbox_summary` | build action contract view sandbox summary |
| &nbsp;&nbsp;`test_build_action_contract_view_high_risk` | build action contract view high risk |
| &nbsp;&nbsp;`test_server_execute_action_no_contracts_returns_404` | server execute action no contracts returns 404 |
| &nbsp;&nbsp;`test_server_execute_action_requires_allow_real_actions` | server execute action requires allow real actions |
| &nbsp;&nbsp;`test_server_execute_action_unknown_contract_returns_404` | server execute action unknown contract returns 404 |
| &nbsp;&nbsp;`test_server_execute_action_scope_violation_fails_dry_run` | server execute action scope violation fails dry run |
| &nbsp;&nbsp;`test_server_execute_action_returns_trace` | server execute action returns trace |
| &nbsp;&nbsp;`test_server_execute_action_with_signing_key_includes_hmac` | server execute action with signing key includes hmac |
| **[test_p20_cut12_regression_e2e.py](../tests/test_p20_cut12_regression_e2e.py)** | Action regression + canonical E2E: P1–P19 imports, email-notifier example (21 tests) |
| &nbsp;&nbsp;`test_regression_matrixai_ir_imports` | regression matrixai ir imports |
| &nbsp;&nbsp;`test_regression_matrixai_parser_imports` | regression matrixai parser imports |
| &nbsp;&nbsp;`test_regression_matrixai_runtime_imports` | regression matrixai runtime imports |
| &nbsp;&nbsp;`test_regression_matrixai_types_imports` | regression matrixai types imports |
| &nbsp;&nbsp;`test_regression_matrixai_training_imports` | regression matrixai training imports |
| &nbsp;&nbsp;`test_regression_matrixai_actions_all_exports` | regression matrixai actions all exports |
| &nbsp;&nbsp;`test_regression_p19_composite_network_still_works` | regression p19 composite network still works |
| &nbsp;&nbsp;`test_regression_existing_cli_commands_importable` | regression existing cli commands importable |
| &nbsp;&nbsp;`test_regression_server_module_importable` | regression server module importable |
| &nbsp;&nbsp;`test_e2e_parse_and_validate_contract` | e2e parse and validate contract |
| &nbsp;&nbsp;`test_e2e_compute_contract_hash` | e2e compute contract hash |
| &nbsp;&nbsp;`test_e2e_hash_is_deterministic` | e2e hash is deterministic |
| &nbsp;&nbsp;`test_e2e_dry_run_passes` | e2e dry run passes |
| &nbsp;&nbsp;`test_e2e_dry_run_fails_on_scope_violation` | e2e dry run fails on scope violation |
| &nbsp;&nbsp;`test_e2e_execute_with_injectable_handler` | e2e execute with injectable handler |
| &nbsp;&nbsp;`test_e2e_build_signed_trace` | e2e build signed trace |
| &nbsp;&nbsp;`test_e2e_verify_trace_signature` | e2e verify trace signature |
| &nbsp;&nbsp;`test_e2e_rollback_on_failed_execution` | e2e rollback on failed execution |
| **[test_p20_audit_fixes.py](../tests/test_p20_audit_fixes.py)** | P20 audit fixes: HMAC canonical message covers all fields, signing-key via context (26 tests) |
| &nbsp;&nbsp;`test_canonical_message_includes_report_id` | canonical message includes report id |
| &nbsp;&nbsp;`test_canonical_message_includes_executor_kind` | canonical message includes executor kind |
| &nbsp;&nbsp;`test_canonical_message_includes_ok` | canonical message includes ok |
| &nbsp;&nbsp;`test_canonical_message_includes_response_summary` | canonical message includes response summary |
| &nbsp;&nbsp;`test_canonical_message_includes_error` | canonical message includes error |
| &nbsp;&nbsp;`test_canonical_message_includes_latency_ms` | canonical message includes latency ms |
| &nbsp;&nbsp;`test_tamper_ok_invalidates_signature` | tamper ok invalidates signature |
| &nbsp;&nbsp;`test_tamper_response_summary_invalidates_signature` | tamper response summary invalidates signature |
| &nbsp;&nbsp;`test_tamper_error_invalidates_signature` | tamper error invalidates signature |
| &nbsp;&nbsp;`test_tamper_latency_ms_invalidates_signature` | tamper latency ms invalidates signature |
| &nbsp;&nbsp;`test_executor_preflight_accepts_signing_key_from_context` | When contract.signature_required, a key in context is enough (no env var). |
| &nbsp;&nbsp;`test_executor_preflight_raises_when_signature_required_and_no_key` | Without key in context or env var, signature_required must raise. |
| &nbsp;&nbsp;`test_executor_preflight_blocks_human_approval_without_store` | human_approval=True and no approval_store → raises. |
| &nbsp;&nbsp;`test_executor_preflight_blocks_human_approval_without_approval` | human_approval=True, store provided but no approved record → raises. |
| &nbsp;&nbsp;`test_executor_preflight_passes_human_approval_when_approved` | human_approval=True and valid approval → executes without raising. |
| &nbsp;&nbsp;`test_executor_preflight_skips_approval_gate_when_not_required` | human_approval=False → no gate check even without store. |
| &nbsp;&nbsp;`test_validate_action_contract_fails_on_missing_scope_field` | validate_action_contract must reject a contract with invalid scope. |
| &nbsp;&nbsp;`test_validate_action_contract_passes_valid_scope` | validate_action_contract must accept a contract with valid email_send scope. |
| &nbsp;&nbsp;`test_rate_tracker_blocks_when_limit_exceeded` | DryRunSimulator with a saturated RateTracker must fail rate limit check. |
| &nbsp;&nbsp;`test_find_approved_rejects_different_model_hash` | Approval for model_v1 must not satisfy a context with model_v2. |
| &nbsp;&nbsp;`test_find_approved_rejects_different_parameter_set_id` | Approval for ps_1 must not satisfy a context with ps_2. |
| &nbsp;&nbsp;`test_find_approved_accepts_exact_match` | Approval must be found when all four binding fields match. |
| &nbsp;&nbsp;`test_dry_run_action_rejects_contract_with_wrong_target` | dry-run-action must fail if validate_action_contract rejects the contract. |
| &nbsp;&nbsp;`test_find_approved_binds_all_four_fields` | Verify find_approved requires action_contract_hash + input_hash + model_hash + ps_id. |
| &nbsp;&nbsp;`test_executor_uses_context_model_hash_in_trace` | ActionTrace.model_hash must come from ExecutionContext, not the ParameterSet directly. |
| &nbsp;&nbsp;`test_execute_action_param_set_file_without_model_hash_uses_artifact_identity` | A certified ParameterSet file must not require an explicit --model-hash flag. |

## Model Registry (P21) (227 tests)

| Test | Description |
|------|-------------|
| **[test_p21_cut1_registry_entry.py](../tests/test_p21_cut1_registry_entry.py)** | Registry entry: RegistryEntry, manifest fields, parameter hash, signature record (33 tests) |
| &nbsp;&nbsp;`test_entry_hash_is_deterministic` | entry hash is deterministic |
| &nbsp;&nbsp;`test_entry_hash_is_sha256_prefixed` | entry hash is sha256 prefixed |
| &nbsp;&nbsp;`test_entry_hash_changes_when_model_hash_changes` | entry hash changes when model hash changes |
| &nbsp;&nbsp;`test_entry_hash_changes_when_evaluation_report_changes` | entry hash changes when evaluation report changes |
| &nbsp;&nbsp;`test_entry_hash_changes_when_training_trace_changes` | entry hash changes when training trace changes |
| &nbsp;&nbsp;`test_entry_hash_changes_when_version_changes` | entry hash changes when version changes |
| &nbsp;&nbsp;`test_entry_hash_changes_when_name_changes` | entry hash changes when name changes |
| &nbsp;&nbsp;`test_entry_hash_stable_regardless_of_kwargs_order` | Canonical JSON sorts keys — kwarg order must not affect the result. |
| &nbsp;&nbsp;`test_sha256_str_and_sha256_bytes_agree` | sha256 str and sha256 bytes agree |
| &nbsp;&nbsp;`test_registry_entry_is_frozen` | registry entry is frozen |
| &nbsp;&nbsp;`test_registry_entry_error_is_exception` | registry entry error is exception |
| &nbsp;&nbsp;`test_registry_entry_to_manifest_roundtrip` | registry entry to manifest roundtrip |
| &nbsp;&nbsp;`test_registry_entry_manifest_contains_required_keys` | registry entry manifest contains required keys |
| &nbsp;&nbsp;`test_registry_entry_from_manifest_defaults_optional_fields` | registry entry from manifest defaults optional fields |
| &nbsp;&nbsp;`test_sign_returns_hmac_sha256_prefix` | sign returns hmac sha256 prefix |
| &nbsp;&nbsp;`test_verify_passes_for_valid_signature` | verify passes for valid signature |
| &nbsp;&nbsp;`test_verify_fails_for_wrong_key` | verify fails for wrong key |
| &nbsp;&nbsp;`test_verify_rejects_tampered_entry_hash` | verify rejects tampered entry hash |
| &nbsp;&nbsp;`test_verify_rejects_empty_signature` | verify rejects empty signature |
| &nbsp;&nbsp;`test_build_signature_record_shape` | build signature record shape |
| &nbsp;&nbsp;`test_sign_verify_roundtrip_is_consistent` | sign verify roundtrip is consistent |
| &nbsp;&nbsp;`test_env_key_takes_precedence_over_local` | env key takes precedence over local |
| &nbsp;&nbsp;`test_local_key_file_creates_with_restricted_permissions` | local key file creates with restricted permissions |
| &nbsp;&nbsp;`test_local_key_file_reuses_existing` | local key file reuses existing |
| &nbsp;&nbsp;`test_get_signing_key_returns_empty_when_no_path_and_no_env` | get signing key returns empty when no path and no env |
| &nbsp;&nbsp;`test_registry_layout_index_path` | registry layout index path |
| &nbsp;&nbsp;`test_registry_layout_entries_dir` | registry layout entries dir |
| &nbsp;&nbsp;`test_registry_layout_tags_dir` | registry layout tags dir |
| &nbsp;&nbsp;`test_registry_layout_entry_dir` | registry layout entry dir |
| &nbsp;&nbsp;`test_registry_layout_entry_file_manifest` | registry layout entry file manifest |
| &nbsp;&nbsp;`test_registry_layout_entry_file_model` | registry layout entry file model |
| &nbsp;&nbsp;`test_registry_layout_entry_file_unknown_raises` | registry layout entry file unknown raises |
| &nbsp;&nbsp;`test_registry_layout_tag_path` | registry layout tag path |
| **[test_p21_cut2_import_parser.py](../tests/test_p21_cut2_import_parser.py)** | IMPORT parser: IMPORT name FROM registry name@version FROZEN|TRAINABLE (22 tests) |
| &nbsp;&nbsp;`test_import_spec_is_frozen` | import spec is frozen |
| &nbsp;&nbsp;`test_import_spec_resolved_fields_default_empty` | import spec resolved fields default empty |
| &nbsp;&nbsp;`test_import_spec_accepts_explicit_resolved_fields` | import spec accepts explicit resolved fields |
| &nbsp;&nbsp;`test_program_imports_field_empty_by_default` | program imports field empty by default |
| &nbsp;&nbsp;`test_parser_accepts_import_frozen` | parser accepts import frozen |
| &nbsp;&nbsp;`test_parser_accepts_import_trainable` | parser accepts import trainable |
| &nbsp;&nbsp;`test_parser_accepts_multiple_imports` | parser accepts multiple imports |
| &nbsp;&nbsp;`test_parser_accepts_tag_version` | parser accepts tag version |
| &nbsp;&nbsp;`test_parser_accepts_dotted_version` | parser accepts dotted version |
| &nbsp;&nbsp;`test_parser_accepts_prod_tag` | parser accepts prod tag |
| &nbsp;&nbsp;`test_parsed_import_resolved_fields_are_empty` | parsed import resolved fields are empty |
| &nbsp;&nbsp;`test_parser_rejects_import_without_version` | parser rejects import without version |
| &nbsp;&nbsp;`test_parser_rejects_import_invalid_mode` | parser rejects import invalid mode |
| &nbsp;&nbsp;`test_parser_rejects_import_missing_from` | parser rejects import missing from |
| &nbsp;&nbsp;`test_parser_rejects_import_missing_registry_keyword` | parser rejects import missing registry keyword |
| &nbsp;&nbsp;`test_parser_rejects_import_uppercase_registry_name` | Registry names must start lowercase per contract naming convention. |
| &nbsp;&nbsp;`test_parser_rejects_duplicate_alias` | parser rejects duplicate alias |
| &nbsp;&nbsp;`test_parser_rejects_alias_colliding_with_vector` | parser rejects alias colliding with vector |
| &nbsp;&nbsp;`test_parser_rejects_alias_colliding_with_network` | parser rejects alias colliding with network |
| &nbsp;&nbsp;`test_program_to_dict_includes_imports` | program to dict includes imports |
| &nbsp;&nbsp;`test_program_to_dict_omits_imports_when_empty` | program to dict omits imports when empty |
| &nbsp;&nbsp;`test_existing_program_without_imports_unchanged` | Parsing a P1-P20 program must produce imports=[] — no regressions. |
| **[test_p21_cut3_model_registry.py](../tests/test_p21_cut3_model_registry.py)** | Model registry: push, pull, list, show, tag, verify — local append-only store (27 tests) |
| &nbsp;&nbsp;`test_registry_initializes_directory_structure` | registry initializes directory structure |
| &nbsp;&nbsp;`test_registry_index_file_has_correct_format` | registry index file has correct format |
| &nbsp;&nbsp;`test_registry_push_writes_manifest` | registry push writes manifest |
| &nbsp;&nbsp;`test_registry_push_requires_evaluation_report` | registry push requires evaluation report |
| &nbsp;&nbsp;`test_registry_push_fails_on_duplicate_version` | registry push fails on duplicate version |
| &nbsp;&nbsp;`test_registry_is_append_only` | registry is append only |
| &nbsp;&nbsp;`test_registry_push_updates_index` | registry push updates index |
| &nbsp;&nbsp;`test_registry_push_signs_with_env_key` | registry push signs with env key |
| &nbsp;&nbsp;`test_registry_get_returns_full_entry` | registry get returns full entry |
| &nbsp;&nbsp;`test_registry_get_raises_for_missing_entry` | registry get raises for missing entry |
| &nbsp;&nbsp;`test_registry_list_empty_when_no_entries` | registry list empty when no entries |
| &nbsp;&nbsp;`test_registry_list_returns_all_entries` | registry list returns all entries |
| &nbsp;&nbsp;`test_registry_list_filters_by_name` | registry list filters by name |
| &nbsp;&nbsp;`test_registry_tag_creates_alias` | registry tag creates alias |
| &nbsp;&nbsp;`test_registry_tag_moves_existing_alias` | registry tag moves existing alias |
| &nbsp;&nbsp;`test_registry_get_resolves_tag_to_version` | registry get resolves tag to version |
| &nbsp;&nbsp;`test_registry_tag_points_to_correct_version` | registry tag points to correct version |
| &nbsp;&nbsp;`test_registry_verify_passes_for_valid_entry` | registry verify passes for valid entry |
| &nbsp;&nbsp;`test_registry_verify_rejects_tampered_manifest` | registry verify rejects tampered manifest |
| &nbsp;&nbsp;`test_registry_verify_rejects_wrong_signature` | registry verify rejects wrong signature |
| &nbsp;&nbsp;`test_registry_pull_copies_entry_between_registries` | registry pull copies entry between registries |
| &nbsp;&nbsp;`test_registry_pull_copies_signature_when_present` | registry pull copies signature when present |
| &nbsp;&nbsp;`test_verify_detects_tampered_params_file` | verify detects tampered params file |
| &nbsp;&nbsp;`test_verify_detects_tampered_model_file` | verify detects tampered model file |
| &nbsp;&nbsp;`test_verify_detects_tampered_evaluation_report` | verify detects tampered evaluation report |
| &nbsp;&nbsp;`test_verify_detects_tampered_training_trace` | verify detects tampered training trace |
| &nbsp;&nbsp;`test_verify_passes_for_untampered_push_run_dir_entry` | verify passes for untampered push run dir entry |
| **[test_p21_cut4_resolver.py](../tests/test_p21_cut4_resolver.py)** | Registry resolver: IMPORT resolution, frozen execution, composite model hash (12 tests) |
| &nbsp;&nbsp;`test_resolver_resolves_existing_import` | resolver resolves existing import |
| &nbsp;&nbsp;`test_resolver_records_resolved_entry_hash` | resolver records resolved entry hash |
| &nbsp;&nbsp;`test_resolver_sets_resolved_at_timestamp` | resolver sets resolved at timestamp |
| &nbsp;&nbsp;`test_resolver_preserves_mode` | resolver preserves mode |
| &nbsp;&nbsp;`test_resolver_resolves_multiple_imports` | resolver resolves multiple imports |
| &nbsp;&nbsp;`test_resolver_fails_on_missing_entry` | resolver fails on missing entry |
| &nbsp;&nbsp;`test_resolver_fails_on_missing_version` | resolver fails on missing version |
| &nbsp;&nbsp;`test_resolver_warns_on_mutable_tag_without_flag` | resolver warns on mutable tag without flag |
| &nbsp;&nbsp;`test_resolver_resolves_tag_alias` | resolver resolves tag alias |
| &nbsp;&nbsp;`test_resolver_allows_mutable_tag_with_flag` | resolver allows mutable tag with flag |
| &nbsp;&nbsp;`test_resolver_caches_resolved_components` | resolver caches resolved components |
| &nbsp;&nbsp;`test_resolver_empty_program_returns_empty` | resolver empty program returns empty |
| **[test_p21_cut5_type_system.py](../tests/test_p21_cut5_type_system.py)** | Registry type system: matrixai_version in manifest entries, compatibility checks (10 tests) |
| &nbsp;&nbsp;`test_type_check_accepts_compatible_components` | type check accepts compatible components |
| &nbsp;&nbsp;`test_type_check_empty_imports_ok` | type check empty imports ok |
| &nbsp;&nbsp;`test_type_check_no_edges_between_imports_ok` | type check no edges between imports ok |
| &nbsp;&nbsp;`test_type_check_rejects_shape_mismatch` | type check rejects shape mismatch |
| &nbsp;&nbsp;`test_type_check_handles_tensor_dim_mismatch` | type check handles tensor dim mismatch |
| &nbsp;&nbsp;`test_type_check_rejects_kind_mismatch` | type check rejects kind mismatch |
| &nbsp;&nbsp;`test_type_check_propagates_through_chain_of_components` | type check propagates through chain of components |
| &nbsp;&nbsp;`test_type_check_detects_error_in_chain` | type check detects error in chain |
| &nbsp;&nbsp;`test_type_check_rejects_unknown_output_type` | type check rejects unknown output type |
| &nbsp;&nbsp;`test_type_check_skips_edge_when_dst_has_no_input_type` | type check skips edge when dst has no input type |
| **[test_p21_cut6_backend_analyzer.py](../tests/test_p21_cut6_backend_analyzer.py)** | Registry backend analyzer: BackendContractAnalyzer with imported frozen nodes (12 tests) |
| &nbsp;&nbsp;`test_analyzer_detects_composite_model_nodes` | analyzer detects composite model nodes |
| &nbsp;&nbsp;`test_analyzer_composite_node_is_supported` | analyzer composite node is supported |
| &nbsp;&nbsp;`test_analyzer_composite_node_kind_reflects_mode` | analyzer composite node kind reflects mode |
| &nbsp;&nbsp;`test_analyzer_builds_component_manifest` | analyzer builds component manifest |
| &nbsp;&nbsp;`test_analyzer_component_manifest_includes_entry_hash` | analyzer component manifest includes entry hash |
| &nbsp;&nbsp;`test_analyzer_component_manifest_empty_without_registry` | analyzer component manifest empty without registry |
| &nbsp;&nbsp;`test_analyzer_warns_about_interpretability_reduced` | analyzer warns about interpretability reduced |
| &nbsp;&nbsp;`test_analyzer_no_interp_warning_for_full` | analyzer no interp warning for full |
| &nbsp;&nbsp;`test_analyzer_propagates_blockers_from_imported_component` | analyzer propagates blockers from imported component |
| &nbsp;&nbsp;`test_analyzer_rejects_action_in_intermediate_component` | analyzer rejects action in intermediate component |
| &nbsp;&nbsp;`test_analyzer_reports_matrixai_version_mismatch` | analyzer reports matrixai version mismatch |
| &nbsp;&nbsp;`test_analyzer_no_version_warning_for_current_version` | analyzer no version warning for current version |
| **[test_p21_cut7_parameter_set.py](../tests/test_p21_cut7_parameter_set.py)** | Registry ParameterSet: versioned parameters, push/pull round-trip integrity (13 tests) |
| &nbsp;&nbsp;`test_parameter_set_supports_namespaced_paths` | parameter set supports namespaced paths |
| &nbsp;&nbsp;`test_parameter_set_deep_namespaced_paths` | parameter set deep namespaced paths |
| &nbsp;&nbsp;`test_parameter_set_mixed_namespaced_and_flat` | parameter set mixed namespaced and flat |
| &nbsp;&nbsp;`test_parameter_schema_hash_excludes_frozen_components` | parameter schema hash excludes frozen components |
| &nbsp;&nbsp;`test_composite_schema_hash_with_no_frozen_equals_full` | composite schema hash with no frozen equals full |
| &nbsp;&nbsp;`test_composite_schema_hash_same_for_same_trainable_params` | composite schema hash same for same trainable params |
| &nbsp;&nbsp;`test_parameter_set_loads_frozen_from_registry` | parameter set loads frozen from registry |
| &nbsp;&nbsp;`test_load_frozen_skips_trainable_imports` | load frozen skips trainable imports |
| &nbsp;&nbsp;`test_load_frozen_returns_empty_for_no_imports` | load frozen returns empty for no imports |
| &nbsp;&nbsp;`test_load_frozen_handles_multiple_frozen` | load frozen handles multiple frozen |
| &nbsp;&nbsp;`test_parameter_set_separates_trainable_and_frozen` | parameter set separates trainable and frozen |
| &nbsp;&nbsp;`test_separate_parameters_all_trainable` | separate parameters all trainable |
| &nbsp;&nbsp;`test_separate_parameters_no_imports` | separate parameters no imports |
| **[test_p21_cut8_composite_forward.py](../tests/test_p21_cut8_composite_forward.py)** | Composite forward with registry: _frozen_execute runs real registered models (16 tests) |
| &nbsp;&nbsp;`test_composite_forward_loads_frozen_params_from_registry` | composite forward loads frozen params from registry |
| &nbsp;&nbsp;`test_composite_forward_combines_frozen_and_trainable` | composite forward combines frozen and trainable |
| &nbsp;&nbsp;`test_composite_forward_returns_result_type` | composite forward returns result type |
| &nbsp;&nbsp;`test_composite_forward_includes_composite_model_hash` | composite forward includes composite model hash |
| &nbsp;&nbsp;`test_training_updates_only_trainable_components` | training updates only trainable components |
| &nbsp;&nbsp;`test_training_does_not_modify_registry_entries` | training does not modify registry entries |
| &nbsp;&nbsp;`test_training_gradient_flows_only_through_trainable` | training gradient flows only through trainable |
| &nbsp;&nbsp;`test_training_with_all_frozen_components_fails_validation` | training with all frozen components fails validation |
| &nbsp;&nbsp;`test_composite_evaluation_uses_composite_model_hash` | composite evaluation uses composite model hash |
| &nbsp;&nbsp;`test_validate_composite_trainability_true_when_trainable_params` | validate composite trainability true when trainable params |
| &nbsp;&nbsp;`test_validate_composite_trainability_false_when_all_frozen` | validate composite trainability false when all frozen |
| &nbsp;&nbsp;`test_frozen_execute_produces_non_identity_output` | Frozen component must execute real model, not return input unchanged. |
| &nbsp;&nbsp;`test_frozen_execute_output_varies_with_weight` | Different weights → different outputs from frozen component. |
| &nbsp;&nbsp;`test_frozen_execute_falls_back_to_identity_when_no_artifacts` | Mock registry entry (no files) must gracefully fall back to passthrough. |
| &nbsp;&nbsp;`test_frozen_execute_scalar_input_mapped_correctly` | Scalar input_val is wrapped into the frozen model's vector field. |
| &nbsp;&nbsp;`test_frozen_execute_dict_input_passed_directly` | Dict input_val is forwarded as-is to MatrixAIRuntime. |
| **[test_p21_cut9_hash_chain.py](../tests/test_p21_cut9_hash_chain.py)** | Hash chain: compute_composite_model_hash, tamper detection, signature binding (16 tests) |
| &nbsp;&nbsp;`test_composite_model_hash_includes_import_entry_hashes` | composite model hash includes import entry hashes |
| &nbsp;&nbsp;`test_composite_model_hash_stable_with_same_imports` | composite model hash stable with same imports |
| &nbsp;&nbsp;`test_composite_model_hash_changes_when_import_version_changes` | composite model hash changes when import version changes |
| &nbsp;&nbsp;`test_composite_model_hash_stable_regardless_of_import_declaration_order` | composite model hash stable regardless of import declaration order |
| &nbsp;&nbsp;`test_composite_model_hash_changes_when_new_import_added` | composite model hash changes when new import added |
| &nbsp;&nbsp;`test_composite_model_hash_empty_imports_is_deterministic` | composite model hash empty imports is deterministic |
| &nbsp;&nbsp;`test_composite_model_hash_raises_for_missing_import` | composite model hash raises for missing import |
| &nbsp;&nbsp;`test_composite_model_hash_changes_when_local_graph_changes` | Changing the local GRAPH topology changes the composite hash. |
| &nbsp;&nbsp;`test_composite_model_hash_changes_when_import_mode_changes` | Switching a component from FROZEN to TRAINABLE changes the composite hash. |
| &nbsp;&nbsp;`test_composite_model_hash_changes_when_local_network_changes` | Changing the local NETWORK definition changes the composite hash. |
| &nbsp;&nbsp;`test_composite_model_hash_same_imports_different_modes_both_stable` | Two programs with different modes are each individually stable. |
| &nbsp;&nbsp;`test_verify_composite_model_passes_for_valid_entries` | verify composite model passes for valid entries |
| &nbsp;&nbsp;`test_action_trace_verification_detects_imported_component_change` | If a component manifest is tampered, verify_composite_model detects it. |
| &nbsp;&nbsp;`test_verify_composite_model_fails_for_missing_entry` | verify composite model fails for missing entry |
| &nbsp;&nbsp;`test_action_trace_signature_includes_composite_chain` | action trace signature includes composite chain |
| &nbsp;&nbsp;`test_action_trace_signature_bound_to_composite_hash` | If the composite hash changes (different component version), old sig is invalid. |
| **[test_p21_cut10_p20_integration.py](../tests/test_p21_cut10_p20_integration.py)** | Registry + actions: registered model used inside action contract pipeline (10 tests) |
| &nbsp;&nbsp;`test_action_contract_attaches_to_composite_terminal` | action contract attaches to composite terminal |
| &nbsp;&nbsp;`test_intermediate_component_cannot_have_real_action` | intermediate component cannot have real action |
| &nbsp;&nbsp;`test_terminal_with_real_action_is_allowed` | terminal with real action is allowed |
| &nbsp;&nbsp;`test_validate_composite_returns_composite_hash` | validate composite returns composite hash |
| &nbsp;&nbsp;`test_action_trace_firms_composite_model_hash` | action trace firms composite model hash |
| &nbsp;&nbsp;`test_audit_action_shows_full_component_chain` | audit action shows full component chain |
| &nbsp;&nbsp;`test_component_chain_includes_mode_info` | component chain includes mode info |
| &nbsp;&nbsp;`test_composite_dry_run_resolves_all_imports` | composite dry run resolves all imports |
| &nbsp;&nbsp;`test_composite_dry_run_fails_for_missing_import` | composite dry run fails for missing import |
| &nbsp;&nbsp;`test_composite_dry_run_includes_composite_hash` | composite dry run includes composite hash |
| &nbsp;&nbsp;`test_registry_list_cli` | registry list cli |
| &nbsp;&nbsp;`test_registry_list_cli_json` | registry list cli json |
| &nbsp;&nbsp;`test_registry_list_cli_filters_by_name` | registry list cli filters by name |
| &nbsp;&nbsp;`test_registry_show_cli_outputs_manifest` | registry show cli outputs manifest |
| &nbsp;&nbsp;`test_registry_verify_cli_returns_exit_code` | registry verify cli returns exit code |
| &nbsp;&nbsp;`test_registry_verify_cli_fails_for_missing` | registry verify cli fails for missing |
| &nbsp;&nbsp;`test_registry_diff_cli_shows_differences` | registry diff cli shows differences |
| &nbsp;&nbsp;`test_workbench_links_to_imported_component_evaluation` | workbench links to imported component evaluation |
| **[test_p21_cut12_regression.py](../tests/test_p21_cut12_regression.py)** | Registry regression: P1–P20 import integrity, full pipeline (21 tests) |
| &nbsp;&nbsp;`test_existing_models_without_imports_unchanged` | Programs without IMPORT produce identical structure as before P21. |
| &nbsp;&nbsp;`test_program_without_imports_has_empty_imports_list` | program without imports has empty imports list |
| &nbsp;&nbsp;`test_to_dict_excludes_imports_when_empty` | to dict excludes imports when empty |
| &nbsp;&nbsp;`test_p18_dense_networks_can_be_registered` | A P18-style entry (no blockers, evaluation_report_hash set) pushes successfully. |
| &nbsp;&nbsp;`test_p18_registry_entry_survives_round_trip` | p18 registry entry survives round trip |
| &nbsp;&nbsp;`test_p17_regression_models_can_be_registered` | p17 regression models can be registered |
| &nbsp;&nbsp;`test_p17_multiple_versions_coexist` | p17 multiple versions coexist |
| &nbsp;&nbsp;`test_suite_passes_without_signing_key` | Registry push/get works with no signing key in env. |
| &nbsp;&nbsp;`test_verify_without_signing_key_passes` | verify without signing key passes |
| &nbsp;&nbsp;`test_torch_remains_optional` | Core registry operations must not import torch. |
| &nbsp;&nbsp;`test_ir_schema_does_not_require_torch` | IR schema (including ImportSpec) must be importable without torch. |
| &nbsp;&nbsp;`test_p20_action_traces_remain_valid_for_non_composite_models` | ActionTrace created with a plain model_hash (non-composite) still signs/verifies. |
| &nbsp;&nbsp;`test_p20_trace_model_hash_field_unchanged` | Existing ActionTrace.model_hash field accepts any sha256 value. |
| &nbsp;&nbsp;`test_canonical_example_file_exists` | canonical example file exists |
| &nbsp;&nbsp;`test_canonical_example_parses_successfully` | canonical example parses successfully |
| &nbsp;&nbsp;`test_canonical_example_has_two_imports` | canonical example has two imports |
| &nbsp;&nbsp;`test_canonical_example_import_modes` | canonical example import modes |
| &nbsp;&nbsp;`test_canonical_example_import_aliases` | canonical example import aliases |
| &nbsp;&nbsp;`test_canonical_example_has_router_network` | canonical example has router network |
| &nbsp;&nbsp;`test_canonical_example_graph_has_edges` | canonical example graph has edges |
| &nbsp;&nbsp;`test_canonical_example_composite_hash` | composite_model_hash can be computed once the imports are in the registry. |
| **[test_p21_post_audit.py](../tests/test_p21_post_audit.py)** | P21 post-audit: tamper detection fix, composite_forward with root output nodes (24 tests) |
| &nbsp;&nbsp;`test_composite_forward_hash_matches_canonical` | composite_forward.composite_model_hash must equal compute_composite_model_hash. |
| &nbsp;&nbsp;`test_composite_forward_hash_changes_with_local_graph` | composite_forward reflects local GRAPH changes in its composite_model_hash. |
| &nbsp;&nbsp;`test_parser_marks_import_alias_as_composite_model` | parser marks import alias as composite model |
| &nbsp;&nbsp;`test_parser_marks_multiple_import_aliases` | parser marks multiple import aliases |
| &nbsp;&nbsp;`test_parser_non_import_nodes_unaffected` | parser non import nodes unaffected |
| &nbsp;&nbsp;`test_push_run_dir_succeeds_with_all_artifacts` | push run dir succeeds with all artifacts |
| &nbsp;&nbsp;`test_push_run_dir_copies_artifacts_to_registry` | push run dir copies artifacts to registry |
| &nbsp;&nbsp;`test_push_run_dir_fails_without_evaluation_report` | push run dir fails without evaluation report |
| &nbsp;&nbsp;`test_push_run_dir_reads_parameter_set_id_from_params` | push run dir reads parameter set id from params |
| &nbsp;&nbsp;`test_push_run_dir_entry_is_retrievable` | push run dir entry is retrievable |
| &nbsp;&nbsp;`test_push_run_dir_works_without_optional_artifacts` | push run dir works without optional artifacts |
| &nbsp;&nbsp;`test_push_run_dir_is_append_only` | push run dir is append only |
| &nbsp;&nbsp;`test_push_run_dir_prefers_params_best_json` | push run dir prefers params best json |
| &nbsp;&nbsp;`test_cli_registry_push_registers_entry` | cli registry push registers entry |
| &nbsp;&nbsp;`test_cli_registry_push_fails_without_evaluation_report` | cli registry push fails without evaluation report |
| &nbsp;&nbsp;`test_cli_typecheck_accepts_registry_path_flag` | --registry-path flag on typecheck must be accepted without error. |
| &nbsp;&nbsp;`test_composite_typecheck_rejects_mutable_tag_by_default` | @latest import rejected without allow_mutable_tags=True. |
| &nbsp;&nbsp;`test_composite_typecheck_accepts_mutable_tag_with_flag` | @latest import accepted when allow_mutable_tags=True (registry lookup may fail but not due to policy). |
| &nbsp;&nbsp;`test_composite_typecheck_pinned_version_not_affected` | Pinned version (v1) passes through without policy rejection. |
| &nbsp;&nbsp;`test_cli_typecheck_mutable_import_rejected_without_flag` | CLI typecheck with @latest import fails unless --allow-mutable-imports passed. |
| &nbsp;&nbsp;`test_cli_typecheck_mutable_import_accepted_with_flag` | CLI typecheck with @latest import succeeds when --allow-mutable-imports is passed. |
| &nbsp;&nbsp;`test_pull_copies_all_artifacts` | pull() transfers model.mxai, params.json, training_trace.json, evaluation_report.json. |
| &nbsp;&nbsp;`test_pull_preserves_artifact_content` | Pulled artifacts have the same content as the originals. |
| &nbsp;&nbsp;`test_pull_entry_is_verifiable_in_target` | Pulled entry verifies correctly in the target registry. |

## Continual Learning (P22) (419 tests)

| Test | Description |
|------|-------------|
| **[test_p22_cut1_continual_parser.py](../tests/test_p22_cut1_continual_parser.py)** | .mxcontinual parser: CONTINUAL_POLICY, GROUND_TRUTH, DRIFT_DETECTION, TRAINING blocks (42 tests) |
| &nbsp;&nbsp;**TestContinualParserAccepts** | *TestContinualParserAccepts* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_full_policy_parses_name` | full policy parses name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_full_policy_target_model` | full policy target model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_full_policy_base_parameter_set` | full policy base parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_full_policy_registry_and_version` | full policy registry and version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ground_truth_parsed` | ground truth parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_detection_features` | drift detection features |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_detection_methods` | drift detection methods |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_detection_min_samples` | drift detection min samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_concept_drift_parsed` | concept drift parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_update_trigger_parsed` | update trigger parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_training_parsed` | training parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dataset_mix_parsed` | dataset mix parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_approval_gate_parsed` | approval gate parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_guard_parsed` | regression guard parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_parsed` | rollback parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_notify_scope` | rollback notify scope |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_parsed` | audit parsed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_minimal_policy_no_concept_drift` | minimal policy no concept drift |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_minimal_policy_no_registry` | minimal policy no registry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_recency_decay` | linear recency decay |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_comments_ignored` | comments ignored |
| &nbsp;&nbsp;**TestPolicyHash** | *TestPolicyHash* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_is_deterministic` | hash is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_has_sha256_prefix` | hash has sha256 prefix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_changes_when_threshold_changes` | hash changes when threshold changes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hash_changes_when_lr_factor_changes` | hash changes when lr factor changes |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_canonical_dict_has_required_keys` | canonical dict has required keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_compute_policy_hash_matches_spec` | compute policy hash matches spec |
| &nbsp;&nbsp;**TestContinualParserRejects** | *TestContinualParserRejects* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_missing_target_model` | rejects missing target model |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_missing_base_parameter_set` | rejects missing base parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_registry_without_version` | rejects registry without version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_invalid_drift_method` | rejects invalid drift method |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_mix_weights_not_summing_to_one` | rejects mix weights not summing to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_invalid_learning_rate_factor_too_low` | rejects invalid learning rate factor too low |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_invalid_learning_rate_factor_too_high` | rejects invalid learning rate factor too high |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_holdout_fraction_out_of_range_low` | rejects holdout fraction out of range low |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_holdout_fraction_out_of_range_high` | rejects holdout fraction out of range high |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_degradation_threshold_out_of_range` | rejects degradation threshold out of range |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_method_for_undeclared_feature` | rejects method for undeclared feature |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_missing_ground_truth_block` | rejects missing ground truth block |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_invalid_training_method` | rejects invalid training method |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_empty_source` | rejects empty source |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_missing_continual_policy_header` | rejects missing continual policy header |
| **[test_p22_cut2_collector.py](../tests/test_p22_cut2_collector.py)** | Production data collector: ActionTrace ingestion, HMAC validation, label time window (30 tests) |
| &nbsp;&nbsp;**TestParseValidLabels** | *TestParseValidLabels* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parses_label_type` | parses label type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_none_for_no_label_type` | returns none for no label type |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_none_for_unrecognized_format` | returns none for unrecognized format |
| &nbsp;&nbsp;**TestCollectorIngest** | *TestCollectorIngest* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_valid_trace_with_ground_truth` | accepts valid trace with ground truth |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_stores_sample_with_timestamp` | stores sample with timestamp |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_sample_has_unique_id` | sample has unique id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ingest_records_source` | ingest records source |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_samples_returns_accumulated` | all samples returns accumulated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ingest_from_dict_reconstructs_trace` | ingest from dict reconstructs trace |
| &nbsp;&nbsp;**TestIngestById** | *TestIngestById* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ingests_registered_trace` | ingests registered trace |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_unknown_trace_id` | rejects unknown trace id |
| &nbsp;&nbsp;**TestGroundTruthWindow** | *TestGroundTruthWindow* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_trace_within_window` | accepts trace within window |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_trace_outside_window` | rejects trace outside window |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_get_samples_in_window_filters_by_executed_at` | get samples in window filters by executed at |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_respects_ground_truth_window_boundary` | respects ground truth window boundary |
| &nbsp;&nbsp;**TestGroundTruthType** | *TestGroundTruthType* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_valid_label` | accepts valid label |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_invalid_label` | rejects invalid label |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_any_label_when_type_not_declared` | accepts any label when type not declared |
| &nbsp;&nbsp;**TestSignatureValidation** | *TestSignatureValidation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_unsigned_trace_when_not_required` | accepts unsigned trace when not required |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_unsigned_when_signature_required` | rejects unsigned when signature required |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_correctly_signed_trace` | accepts correctly signed trace |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_tampered_signature` | rejects tampered signature |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_unsigned_when_signing_key_provided` | rejects unsigned when signing key provided |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_signed_trace_when_required_but_no_key_to_verify` | rejects signed trace when required but no key to verify |
| &nbsp;&nbsp;**TestContinualIngestCLI** | *TestContinualIngestCLI* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_ingest_accepts_signed_trace_file` | cli ingest accepts signed trace file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cli_ingest_rejects_required_signature_without_key` | cli ingest rejects required signature without key |
| &nbsp;&nbsp;**TestFileWatch** | *TestFileWatch* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_picks_up_new_json_files` | picks up new json files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ignores_non_json_files` | ignores non json files |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_skips_files_with_unknown_trace_id` | skips files with unknown trace id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_handles_missing_directory_gracefully` | handles missing directory gracefully |
| **[test_p22_cut3_drift_detector.py](../tests/test_p22_cut3_drift_detector.py)** | Drift detector: PSI, KS, chi-square, JS, Wasserstein, ConceptDriftDetector, cardinality guard (55 tests) |
| &nbsp;&nbsp;**TestComputePSI** | *TestComputePSI* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_drift_when_distributions_match` | no drift when distributions match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_detects_drift_when_distribution_shifts` | detects drift when distribution shifts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_symmetric_is_not_guaranteed` | symmetric is not guaranteed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_identical_small_sample` | identical small sample |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_observed_returns_zero` | empty observed returns zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_constant_reference_returns_zero` | constant reference returns zero |
| &nbsp;&nbsp;**TestComputeKS** | *TestComputeKS* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_identical_distributions_zero` | identical distributions zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_detects_continuous_distribution_change` | detects continuous distribution change |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bounded_zero_to_one` | bounded zero to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_returns_zero` | empty returns zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_small_sample_handled` | small sample handled |
| &nbsp;&nbsp;**TestComputeChiSquare** | *TestComputeChiSquare* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_identical_counts_zero` | identical counts zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_detects_categorical_distribution_change` | detects categorical distribution change |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_returns_zero` | empty returns zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_new_category_in_observed` | new category in observed |
| &nbsp;&nbsp;**TestComputeJS** | *TestComputeJS* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_identical_distributions_near_zero` | identical distributions near zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_js_divergence_bounded_zero_to_one` | js divergence bounded zero to one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_detects_distribution_shift` | detects distribution shift |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_returns_zero` | empty returns zero |
| &nbsp;&nbsp;**TestComputeWasserstein** | *TestComputeWasserstein* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_identical_distributions_zero` | identical distributions zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_detects_shift` | detects shift |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_handles_small_samples` | handles small samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_returns_zero` | empty returns zero |
| &nbsp;&nbsp;**TestDriftDetector** | *TestDriftDetector* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_produces_drift_report` | produces drift report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_drift_when_distributions_match` | no drift when distributions match |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_detected_when_distribution_shifts` | drift detected when distribution shifts |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_threshold_exceeded_sets_feature_flag` | drift threshold exceeded sets feature flag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_contains_all_features` | report contains all features |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_respects_min_samples` | respects min samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_enough_samples_flag_set` | enough samples flag set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_handles_unknown_feature_gracefully` | handles unknown feature gracefully |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_feature_result_has_observed_value` | feature result has observed value |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_feature_without_declared_method_is_skipped` | feature without declared method is skipped |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_ks_feature_drift_detected_above_threshold` | ks feature drift detected above threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_per_feature_min_samples_not_global_max` | A feature with insufficient samples must NOT mark drift even if |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_psi_handles_observed_values_below_reference_range` | PSI must not produce negative bin indices when observed values |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_psi_handles_observed_values_above_reference_range` | PSI must not overflow when observed values exceed the reference max. |
| &nbsp;&nbsp;**TestChiSquareCardinality** | *TestChiSquareCardinality* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_low_cardinality_reference_succeeds` | low cardinality reference succeeds |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_high_cardinality_raises_value_error` | high cardinality raises value error |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_error_message_suggests_alternative` | error message suggests alternative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exactly_at_limit_succeeds` | exactly at limit succeeds |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_one_over_limit_raises` | one over limit raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_detector_raises_on_continuous_chi_square` | DriftDetector.run_check propagates ValueError when chi_square is |
| &nbsp;&nbsp;**TestConceptDriftDetector** | *TestConceptDriftDetector* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_drift_above_alert_threshold` | no drift above alert threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_detected_below_alert_threshold` | drift detected below alert threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_insufficient_labeled_samples_suppresses_detection` | insufficient labeled samples suppresses detection |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_fields_populated_correctly` | report fields populated correctly |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exact_boundary_not_flagged` | exact boundary not flagged |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_policy_without_concept_drift_raises` | policy without concept drift raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_checked_at_is_iso8601` | checked at is iso8601 |
| &nbsp;&nbsp;**TestConceptDriftParserValidation** | *Parser must reject CONCEPT_DRIFT configs that make the detector always-silent.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_threshold_equal_to_reference_raises` | threshold equal to reference raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_threshold_greater_than_reference_raises` | threshold greater than reference raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_threshold_zero_raises` | threshold zero raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_threshold_negative_raises` | threshold negative raises |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_valid_config_parses_ok` | valid config parses ok |
| **[test_p22_cut4_continual_dataset.py](../tests/test_p22_cut4_continual_dataset.py)** | Continual dataset: base + production mix with linear/exponential recency decay (26 tests) |
| &nbsp;&nbsp;**TestContinualDatasetConstruction** | *TestContinualDatasetConstruction* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_constructs_with_valid_inputs` | constructs with valid inputs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_on_timestamp_count_mismatch` | raises on timestamp count mismatch |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_base_count_and_production_count` | base count and production count |
| &nbsp;&nbsp;**TestExamples** | *TestExamples* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_base_examples_always_included` | base examples always included |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_examples_are_supervised_examples` | all examples are supervised examples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_production_examples_with_nonzero_weight_included` | production examples with nonzero weight included |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_decay_excludes_examples_beyond_window` | linear decay excludes examples beyond window |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_production_weight_zero_excludes_production` | production weight zero excludes production |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_production_returns_base_only` | empty production returns base only |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_total_count_with_recent_production` | total count with recent production |
| &nbsp;&nbsp;**TestRecencyDecay** | *TestRecencyDecay* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exponential_decay_formula` | w = exp(-ln(2) * age_days / half_life_days) |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exponential_zero_age_weight_is_one` | exponential zero age weight is one |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exponential_weight_decreases_with_age` | exponential weight decreases with age |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_decay_formula` | w = max(0, 1 - age_days / window_days) |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_decay_zero_at_boundary` | linear decay zero at boundary |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_linear_decay_beyond_window_clamped_to_zero` | linear decay beyond window clamped to zero |
| &nbsp;&nbsp;**TestWeights** | *TestWeights* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_weights_parallel_to_examples` | weights parallel to examples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_weights_positive` | all weights positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_recent_production_has_higher_weight_than_old` | recent production has higher weight than old |
| &nbsp;&nbsp;**TestFingerprint** | *TestFingerprint* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_starts_with_continual` | fingerprint starts with continual |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_is_deterministic` | fingerprint is deterministic |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_changes_with_different_base` | fingerprint changes with different base |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_changes_when_production_added` | fingerprint changes when production added |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_changes_with_different_window_days` | Different window_days changes the effective decay → different fingerprint. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_changes_with_different_reference_date` | Different reference date changes effective weights → different fingerprint. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fingerprint_changes_with_different_production_timestamps` | Same examples but different timestamps → different decay → different fingerprint. |
| **[test_p22_cut5_incremental_trainer.py](../tests/test_p22_cut5_incremental_trainer.py)** | Incremental trainer: fine-tune from ParameterSet base, auto-detect P4 mode, replay_buffer (33 tests) |
| &nbsp;&nbsp;**TestIncrementalTrainingResultShape** | *TestIncrementalTrainingResultShape* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_incremental_training_result` | returns incremental training result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_has_candidate_parameter_set` | result has candidate parameter set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_result_has_parent_parameter_set_id` | result has parent parameter set id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_metrics_contain_parent_id` | candidate metrics contain parent id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_trace_not_empty` | epoch trace not empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_trace_contains_expected_keys` | epoch trace contains expected keys |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epochs_run_matches_trace_length` | epochs run matches trace length |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_best_epoch_within_range` | best epoch within range |
| &nbsp;&nbsp;**TestParameterUpdates** | *TestParameterUpdates* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameters_change_after_training` | parameters change after training |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_has_same_parameter_schema` | candidate has same parameter schema |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_preserves_model_hash` | candidate preserves model hash |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_source_is_incremental_finetune` | candidate source is incremental finetune |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_id_derived_from_parent` | candidate id derived from parent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_metrics_contain_validation_loss` | candidate metrics contain validation loss |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_metrics_contain_dataset_fingerprint` | candidate metrics contain dataset fingerprint |
| &nbsp;&nbsp;**TestEpochsAndLearningRate** | *TestEpochsAndLearningRate* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_respects_max_epochs` | respects max epochs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_learning_rate_factor_applied` | Smaller LEARNING_RATE_FACTOR should produce less total parameter change. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_epoch_callback_is_called` | epoch callback is called |
| &nbsp;&nbsp;**TestEarlyStopping** | *TestEarlyStopping* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_early_stop_limits_epochs_below_max` | early stop limits epochs below max |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_stopped_early_flag_set_when_triggered` | stopped early flag set when triggered |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_early_stop_without_spec` | Without EARLY_STOP, the trainer must run for exactly MAX_EPOCHS. |
| &nbsp;&nbsp;**TestTrainingReducesLoss** | *TestTrainingReducesLoss* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_decreases_after_training` | Final validation_loss should be below initial loss on a learnable dataset. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accuracy_is_non_negative` | accuracy is non negative |
| &nbsp;&nbsp;**TestMiscellaneous** | *TestMiscellaneous* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_on_empty_dataset` | raises on empty dataset |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_without_program_and_non_p4_params` | ParameterSet without W1/b1 keys and no program should raise. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_p4_mode_detected_when_w1_b1_present` | p4 mode detected when w1 b1 present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_metrics_epochs_run_matches_result` | candidate metrics epochs run matches result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_stopped_early_flag_matches_result` | candidate stopped early flag matches result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_deterministic_with_same_seed` | deterministic with same seed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_seeds_may_produce_different_results` | different seeds may produce different results |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_on_unsupported_training_method` | TRAINING.METHOD values outside the MVP set must be rejected at init. |
| &nbsp;&nbsp;**TestReplayBufferAlias** | *TestReplayBufferAlias* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_replay_buffer_accepted_as_method` | replay buffer accepted as method |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_replay_buffer_produces_incremental_finetune_source` | replay_buffer is an alias: candidate source is always 'incremental_finetune'. |
| **[test_p22_cut6_approval_gate.py](../tests/test_p22_cut6_approval_gate.py)** | Approval gate: holdout evaluation, REGRESSION_GUARD states, PendingApproval signed token (34 tests) |
| &nbsp;&nbsp;**TestApprovalGateReportShape** | *TestApprovalGateReportShape* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_approval_gate_report` | returns approval gate report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_has_correct_candidate_id` | report has correct candidate id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_has_correct_baseline_id` | report has correct baseline id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_has_holdout_samples_count` | report has holdout samples count |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_report_has_evaluated_at` | report has evaluated at |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_baseline_and_candidate_metrics_are_holdout_metrics` | baseline and candidate metrics are holdout metrics |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_guard_result_in_report` | regression guard result in report |
| &nbsp;&nbsp;**TestAutomaticPass** | *TestAutomaticPass* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_automatic_pass_when_candidate_better_and_no_human_required` | automatic pass when candidate better and no human required |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_passed_property_true_for_automatic_pass` | passed property true for automatic pass |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_pending_approval_for_automatic_pass` | no pending approval for automatic pass |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_accuracy_exceeds_baseline` | candidate accuracy exceeds baseline |
| &nbsp;&nbsp;**TestRejection** | *TestRejection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejects_when_candidate_worse_and_strict_guard` | rejects when candidate worse and strict guard |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_passed_false_when_rejected` | passed false when rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejection_reasons_not_empty` | rejection reasons not empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_regression_guard_passed_false_when_rejected` | regression guard passed false when rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_pending_approval_when_rejected` | no pending approval when rejected |
| &nbsp;&nbsp;**TestHumanApproval** | *TestHumanApproval* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_human_when_guard_passes_and_human_required` | pending human when guard passes and human required |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_passed_true_for_pending_human` | passed true for pending human |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_approval_not_none_when_human_required` | pending approval not none when human required |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_approval_has_candidate_id` | pending approval has candidate id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_approval_has_approval_token` | pending approval has approval token |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_approval_hmac_token_when_signing_key_provided` | pending approval hmac token when signing key provided |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_approval_status_is_pending_by_default` | pending approval status is pending by default |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_approve_pending_approval_sets_decision_fields` | approve pending approval sets decision fields |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pending_approval_expires_at_set_when_timeout_declared` | pending approval expires at set when timeout declared |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rejected_even_with_human_approval_when_guard_fails` | rejected even with human approval when guard fails |
| &nbsp;&nbsp;**TestHoldoutMetrics** | *TestHoldoutMetrics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_perfect_ps_achieves_high_accuracy` | perfect ps achieves high accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metrics_loss_is_non_negative` | metrics loss is non negative |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_per_label_metrics_present_for_classification` | per label metrics present for classification |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_holdout_metrics_samples_count_correct` | holdout metrics samples count correct |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_loss_metric_guard` | With METRIC=loss, a lower candidate loss should pass the gate. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_holdout_does_not_raise` | empty holdout does not raise |
| &nbsp;&nbsp;**TestNonP4Rejection** | *ApprovalGate must reject ParameterSets without W1/b1 with a clear error.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_for_non_p4_candidate` | A candidate without W1/b1 must raise ValueError, not return dummy metrics. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_for_non_p4_baseline` | A baseline without W1/b1 must also raise ValueError. |
| **[test_p22_cut7_versioning.py](../tests/test_p22_cut7_versioning.py)** | Continual versioner: minor bump in registry, SIGNATURE_REQUIRED enforcement, token rejection (36 tests) |
| &nbsp;&nbsp;**TestPromoteBasics** | *TestPromoteBasics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_versioning_result` | returns versioning result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_new_version_is_v1_1` | new version is v1 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_previous_version_is_base_version` | previous version is base version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_registry_name_in_result` | registry name in result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_entry_hash_is_non_empty` | entry hash is non empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_pushed_at_matches_now` | pushed at matches now |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parent_parameter_set_id_in_result` | parent parameter set id in result |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_candidate_parameter_set_id_in_result` | candidate parameter set id in result |
| &nbsp;&nbsp;**TestRegistryState** | *TestRegistryState* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_new_version_retrievable_from_registry` | new version retrievable from registry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_current_tag_updated_to_new_version` | current tag updated to new version |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_hash_preserved_in_entry` | model hash preserved in entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_schema_hash_preserved` | parameter schema hash preserved |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_entry_metrics_include_parent_ps_id` | entry metrics include parent ps id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_entry_metrics_include_continual_update_id` | entry metrics include continual update id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_params_json_stored_in_entry_dir` | params json stored in entry dir |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_approval_gate_report_stored_in_entry_dir` | approval gate report stored in entry dir |
| &nbsp;&nbsp;**TestSequentialPromotions** | *TestSequentialPromotions* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_second_promotion_gives_v1_2` | second promotion gives v1 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_current_tag_follows_latest_promotion` | current tag follows latest promotion |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_base_version_untouched_after_promotion` | base version untouched after promotion |
| &nbsp;&nbsp;**TestErrors** | *TestErrors* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_on_failed_approval_report` | raises on failed approval report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_when_no_registry_name_in_policy` | raises when no registry name in policy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_continual_update_id_auto_generated` | continual update id auto generated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_custom_update_id_preserved` | custom update id preserved |
| &nbsp;&nbsp;**TestSecurityBinding** | *Verify cryptographic binding between report, candidate, and policy.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_when_report_candidate_mismatch` | Report for a different candidate must not allow promotion. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_when_report_policy_hash_mismatch` | Report generated under a different policy must not allow promotion. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_on_pending_human_without_explicit_acknowledgement` | pending_human status must not promote without human_approved=True. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_when_acknowledged_but_pending_not_approved` | human_approved=True is not enough without an approved PendingApproval state. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promotes_with_pending_human_when_acknowledged` | human_approved=True must allow promotion of pending_human reports. |
| &nbsp;&nbsp;**TestPendingApprovalExpiry** | *Verify that expired PendingApproval tokens are rejected.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_when_token_expired` | Token expired before NOW must raise ContinualVersioningError. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_allows_when_token_not_yet_expired` | Token with future expires_at must allow promotion. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_allows_when_expires_at_is_none` | No expiry date means token never expires — promotion must succeed. |
| &nbsp;&nbsp;**TestPendingApprovalTokenIntegrity** | *Verify that expires_at is covered by the HMAC token and malformed values fail closed.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_malformed_expires_at_fails_closed` | Unparseable expires_at must raise ContinualVersioningError, not allow through. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_tampered_expires_at_invalidates_token` | Changing expires_at after token generation must cause CLI HMAC mismatch. |
| &nbsp;&nbsp;**TestSignatureRequired** | *Verify that SIGNATURE_REQUIRED true forces HMAC-signed tokens in promote().* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_raises_on_unsigned_token_when_signature_required` | SIGNATURE_REQUIRED true + unsigned sha256: token must raise, not promote. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_allows_hmac_signed_token_when_signature_required` | SIGNATURE_REQUIRED true + HMAC-signed token + matching key must succeed. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unsigned_token_still_works_without_signature_required` | SIGNATURE_REQUIRED false (default) must still allow sha256: tokens. |
| **[test_p22_cut8_monitor.py](../tests/test_p22_cut8_monitor.py)** | Production monitor: sliding window accuracy, per-label metrics, degradation_detected flag (34 tests) |
| &nbsp;&nbsp;**TestOnlineObservation** | *TestOnlineObservation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_returns_observation` | record returns observation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_correct_flag_true_when_prediction_matches` | correct flag true when prediction matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_correct_flag_false_when_prediction_wrong` | correct flag false when prediction wrong |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_observation_stores_trace_id` | observation stores trace id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_observation_stores_parameter_set_id` | observation stores parameter set id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_all_observations_accumulates` | all observations accumulates |
| &nbsp;&nbsp;**TestWindowMetricsShape** | *TestWindowMetricsShape* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_window_metrics_returns_window_metrics` | window metrics returns window metrics |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_window_hours_matches_policy` | window hours matches policy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_window_end_matches_now` | window end matches now |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_window_start_is_window_hours_before_now` | window start is window hours before now |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_window_gives_zero_accuracy` | empty window gives zero accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_samples_counts_in_window_only` | samples counts in window only |
| &nbsp;&nbsp;**TestWindowFiltering** | *TestWindowFiltering* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_excludes_observations_outside_window` | excludes observations outside window |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_includes_observations_inside_window` | includes observations inside window |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_boundary_observation_included` | boundary observation included |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_mixed_in_and_out_of_window` | mixed in and out of window |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_observations_in_window_helper_matches` | observations in window helper matches |
| &nbsp;&nbsp;**TestAccuracyComputation** | *TestAccuracyComputation* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_perfect_accuracy` | perfect accuracy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_zero_accuracy_all_wrong` | zero accuracy all wrong |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_partial_accuracy` | partial accuracy |
| &nbsp;&nbsp;**TestDegradationDetection** | *TestDegradationDetection* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_degradation_detected_when_accuracy_drops` | degradation detected when accuracy drops |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_degradation_when_accuracy_above_threshold` | no degradation when accuracy above threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_degradation_without_enough_samples` | no degradation without enough samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_enough_samples_true_at_min_threshold` | enough samples true at min threshold |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_degradation_when_reference_not_set` | no degradation when reference not set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_actual_degradation_computed_correctly` | actual degradation computed correctly |
| &nbsp;&nbsp;**TestPerLabelMetrics** | *TestPerLabelMetrics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_per_label_metrics_present_for_known_labels` | per label metrics present for known labels |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_per_label_precision_recall_f1` | per label precision recall f1 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_per_label_empty_without_labels` | per label empty without labels |
| &nbsp;&nbsp;**TestParameterSetTracking** | *TestParameterSetTracking* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_ids_collected` | parameter set ids collected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_parameter_set_ids_deduplicated` | parameter set ids deduplicated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_empty_parameter_set_id_excluded` | empty parameter set id excluded |
| &nbsp;&nbsp;**TestUtility** | *TestUtility* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_clear_removes_all_observations` | clear removes all observations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_set_reference_accuracy_updates_detection` | set reference accuracy updates detection |
| **[test_p22_cut9_rollback.py](../tests/test_p22_cut9_rollback.py)** | Rollback manager: check/execute/run, RollbackEvent signed with HMAC or SHA256 (32 tests) |
| &nbsp;&nbsp;**TestRollbackCheckResultShape** | *TestRollbackCheckResultShape* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_returns_result_instance` | check returns result instance |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_fields_populated_when_triggered` | check fields populated when triggered |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_reason_is_online_degradation` | check reason is online degradation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_check_window_accuracy_is_present` | check window accuracy is present |
| &nbsp;&nbsp;**TestCheckNoRollback** | *TestCheckNoRollback* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_rollback_when_auto_trigger_false` | no rollback when auto trigger false |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_rollback_when_not_enough_samples` | no rollback when not enough samples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_rollback_when_no_degradation` | no rollback when no degradation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_rollback_when_registry_get_fails` | no rollback when registry get fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_rollback_when_no_parent_ps_id` | no rollback when no parent ps id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_rollback_when_parent_version_not_found` | no rollback when parent version not found |
| &nbsp;&nbsp;**TestExecute** | *TestExecute* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_execute_returns_rollback_event` | execute returns rollback event |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_execute_updates_current_tag` | execute updates current tag |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_execute_event_has_correct_versions` | execute event has correct versions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_execute_event_has_correct_trigger_reason` | execute event has correct trigger reason |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_execute_event_notification_sent_false` | execute event notification sent false |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_execute_event_rollback_id_format` | execute event rollback id format |
| &nbsp;&nbsp;**TestSignature** | *TestSignature* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_signature_is_sha256_without_key` | signature is sha256 without key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_signature_is_hmac_sha256_with_key` | signature is hmac sha256 with key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_signature_is_deterministic_for_same_inputs` | signature is deterministic for same inputs |
| &nbsp;&nbsp;**TestRun** | *TestRun* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_returns_event_when_degraded` | run returns event when degraded |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_returns_none_when_no_degradation` | run returns none when no degradation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_tags_current_when_rollback_executed` | run tags current when rollback executed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_does_not_tag_when_no_rollback` | run does not tag when no rollback |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_event_window_accuracy_matches_monitor` | run event window accuracy matches monitor |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_samples_in_window_in_event` | run samples in window in event |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_run_uses_default_now_when_not_passed` | run uses default now when not passed |
| &nbsp;&nbsp;**TestRollbackNotification** | *RollbackManager calls notification_fn when NOTIFY_CAPABILITY is declared.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_fn_called_when_capability_declared` | notification_fn receives the RollbackEvent when notify_capability is set. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_sent_true_when_fn_succeeds` | notification_sent=True when notification_fn returns True. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_sent_false_when_no_capability` | notification_sent=False when notify_capability is not declared. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_sent_false_when_fn_raises` | Exceptions in notification_fn are swallowed; notification_sent=False. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_fn_not_called_when_no_rollback` | notification_fn is not called when rollback does not trigger. |
| &nbsp;&nbsp;**TestSignatureCoverage** | *Verify that notification_sent is included in the RollbackEvent signature payload.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_notification_sent_covered_by_signature` | Two events identical except notification_sent must produce different signatures. |
| **[test_p22_cut10_refinement_bridge.py](../tests/test_p22_cut10_refinement_bridge.py)** | Drift refinement bridge: maybe_refine, EMIT_REFINEMENT_HINT_ON_SUSTAINED_DRIFT, persistence days (29 tests) |
| &nbsp;&nbsp;**TestRefinementAgentDriftDriven** | *TestRefinementAgentDriftDriven* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_driven_mode_accepted` | drift driven mode accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_driven_mode_field_set` | drift driven mode field set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_driven_requires_drift_report` | drift driven requires drift report |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unknown_mode_rejected` | unknown mode rejected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hints_applied_contains_drift_hints` | hints applied contains drift hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hints_applied_references_feature_name` | hints applied references feature name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_hints_applied_references_method` | hints applied references method |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multiple_drifted_features_generate_multiple_hints` | multiple drifted features generate multiple hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_skipped_features_do_not_generate_hints` | skipped features do not generate hints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_extra_hints_appended_after_derived` | extra hints appended after derived |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposed_prompt_contains_feedback_section` | proposed prompt contains feedback section |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explanation_mentions_drift` | explanation mentions drift |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refinement_id_contains_drift_driven` | refinement id contains drift driven |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_refinement_chain_populated` | refinement chain populated |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_iteration_count_stored` | iteration count stored |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_original_prompt_preserved` | original prompt preserved |
| &nbsp;&nbsp;**TestDriftRefinementBridge** | *TestDriftRefinementBridge* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_proposal_when_emit_on_and_drift_detected` | returns proposal when emit on and drift detected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_none_when_emit_off` | returns none when emit off |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_none_when_no_drift_detected` | returns none when no drift detected |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_none_when_emit_off_even_with_drift` | returns none when emit off even with drift |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposal_mode_is_drift_driven` | proposal mode is drift driven |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_proposal_original_prompt_matches` | proposal original prompt matches |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_extra_hints_forwarded_to_agent` | extra hints forwarded to agent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_multi_feature_drift_report_handled` | multi feature drift report handled |
| &nbsp;&nbsp;**TestDriftPersistence** | *Verify that REFINEMENT_DRIFT_PERSISTENCE_DAYS gates the bridge correctly.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_persistence_check_blocks_when_too_few_days` | When drift has not persisted long enough, maybe_refine returns None. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_persistence_check_allows_when_enough_days` | When drift has persisted >= required days, a proposal is returned. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_persistence_check_allows_when_more_than_required` | When drift has persisted well beyond the threshold, proposal is returned. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_persistence_omitted_treated_as_zero` | When drift_persistence_days is omitted and required > 0, None is treated as 0 → blocks. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_persistence_blocks_when_one_day_short` | Boundary: 13 days when 14 required must still block. |
| &nbsp;&nbsp;**TestCliHelp** | *TestCliHelp* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_continual_help_exits_zero` | continual help exits zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_continual_help_lists_all_subcommands` | continual help lists all subcommands |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_help` | init help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_status_help` | status help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_help` | promote help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_help` | rollback help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_help` | audit help |
| &nbsp;&nbsp;**TestCliInit** | *TestCliInit* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_exits_zero` | init exits zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_shows_policy_name` | init shows policy name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_shows_registry_name` | init shows registry name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_json_output` | init json output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_bad_policy_exits_2` | init bad policy exits 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_init_shows_rollback_config` | init shows rollback config |
| &nbsp;&nbsp;**TestCliStatus** | *TestCliStatus* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_status_exits_1_when_registry_missing` | status exits 1 when registry missing |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_status_error_mentions_registry` | status error mentions registry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_status_bad_policy_exits_2` | status bad policy exits 2 |
| &nbsp;&nbsp;**TestCliRollbackDryRun** | *TestCliRollbackDryRun* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_dry_run_exits_1_no_registry` | rollback dry run exits 1 no registry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_bad_policy_exits_2` | rollback bad policy exits 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_dry_run_json_flag_accepted` | rollback dry run json flag accepted |
| &nbsp;&nbsp;**TestCliPromote** | *TestCliPromote* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_missing_approval_report_exits_2` | promote missing approval report exits 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_bad_policy_exits_2` | promote bad policy exits 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_missing_candidate_params_exits_2` | promote missing candidate params exits 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_happy_path_automatic_pass` | promote happy path automatic pass |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_happy_path_human_approved_hmac` | promote happy path human approved hmac |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_promote_exits_1_on_hmac_tampered_report` | promote exits 1 on hmac tampered report |
| &nbsp;&nbsp;**TestCliAudit** | *TestCliAudit* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_exits_zero` | audit exits zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_shows_emit_hint` | audit shows emit hint |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_json_output` | audit json output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_bad_policy_exits_2` | audit bad policy exits 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_with_drift_report_no_prompt` | audit with drift report no prompt |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_with_drift_report_and_prompt_json` | audit with drift report and prompt json |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_proposal_none_when_persistence_not_provided` | Without --drift-persistence-days, bridge conservatively returns None. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_audit_no_proposal_when_drift_not_detected` | audit no proposal when drift not detected |
| &nbsp;&nbsp;**TestContinualPolicyView** | *TestContinualPolicyView* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_build_returns_view_instance` | build returns view instance |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_name_matches_policy` | view name matches policy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_registry_name` | view registry name |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_rollback_fields` | view rollback fields |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_audit_fields` | view audit fields |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_policy_hash_present` | view policy hash present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_current_version_none_without_entry` | view current version none without entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_current_version_from_entry` | view current version from entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_is_dataclass` | view is dataclass |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_view_asdict_serializable` | view asdict serializable |
| **[test_p22_cut12_regression_e2e.py](../tests/test_p22_cut12_regression_e2e.py)** | Continual regression + canonical E2E: parse → drift → dataset → train → approve → version → rollback (25 tests) |
| &nbsp;&nbsp;**TestP22Regression** | *Verify that all P22 components are importable and minimally functional.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c1_parse_mxcontinual` | c1 parse mxcontinual |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c1_parse_error_on_bad_input` | c1 parse error on bad input |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c2_collector_imports` | c2 collector imports |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c3_drift_detector_imports` | c3 drift detector imports |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c3_drift_report_structure` | c3 drift report structure |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c4_continual_dataset_examples` | c4 continual dataset examples |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c4_continual_dataset_fingerprint` | c4 continual dataset fingerprint |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c5_incremental_trainer_runs` | c5 incremental trainer runs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c5_candidate_has_parent_ps_id` | c5 candidate has parent ps id |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c6_approval_gate_runs` | c6 approval gate runs |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c7_versioner_imports` | c7 versioner imports |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c8_monitor_records_observations` | c8 monitor records observations |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c8_window_metrics_shape` | c8 window metrics shape |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c9_rollback_manager_check` | c9 rollback manager check |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c9_rollback_event_structure` | c9 rollback event structure |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c10_drift_bridge_returns_proposal_on_drift` | c10 drift bridge returns proposal on drift |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c11_policy_view_built` | c11 policy view built |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c11_policy_view_with_entry` | c11 policy view with entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c3_concept_drift_detector_smoke` | c3 concept drift detector smoke |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c7_signature_required_enforced_smoke` | c7 signature required enforced smoke |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_c7_signature_required_allows_hmac_tokens_smoke` | SIGNATURE_REQUIRED true + well-formed HMAC tokens → promote succeeds. |
| &nbsp;&nbsp;**TestContinualE2E** | *Canonical end-to-end demonstration of the P22 continual learning cycle.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_full_cycle_parse_to_rollback` | full cycle parse to rollback |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_e2e_run_rollback_convenience` | run() = check() + execute() combined. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_e2e_no_rollback_on_healthy_monitor` | No rollback when production accuracy is healthy. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_e2e_policy_hash_stable` | Policy hash is deterministic across parses. |

## Deployment Suite (PR4) (149 tests)

| Test | Description |
|------|-------------|
| **[test_pr4_c1_deployment.py](../tests/test_pr4_c1_deployment.py)** | Deployment: matrixai pack --docker, Dockerfile correctness, docker-compose.yml, .env.example (36 tests) |
| &nbsp;&nbsp;**TestPackBasic** | *TestPackBasic* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_copies_model_file` | copies model file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_copies_framework_directory` | copies framework directory |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_docker_files_when_flag_absent` | no docker files when flag absent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_1_for_missing_model` | returns 1 for missing model |
| &nbsp;&nbsp;**TestDockerArtifacts** | *TestDockerArtifacts* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_dockerfile` | generates dockerfile |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_compose` | generates compose |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_generates_env_example` | generates env example |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_0_on_success` | returns 0 on success |
| &nbsp;&nbsp;**TestDockerfileContent** | *TestDockerfileContent* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_uses_python311_slim` | uses python311 slim |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_healthcheck` | has healthcheck |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exposes_port_8000` | exposes port 8000 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cmd_includes_model_filename` | cmd includes model filename |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cmd_includes_host_0000` | cmd includes host 0000 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_hardcoded_api_key` | no hardcoded api key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_copies_matrixai_framework` | copies matrixai framework |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_copies_model_file` | copies model file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_params_not_copied_when_absent` | params not copied when absent |
| &nbsp;&nbsp;**TestDockerfileWithParams** | *Tests that --params wires correctly into the generated Dockerfile.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_copies_params_file` | copies params file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cmd_includes_params_flag` | cmd includes params flag |
| &nbsp;&nbsp;**TestComposeContent** | *TestComposeContent* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_service_definition` | has service definition |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_uses_env_file` | uses env file |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_exposes_port_8000` | exposes port 8000 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_healthcheck` | has healthcheck |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_restart_policy` | has restart policy |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_registry_volume` | has registry volume |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_has_build_directive` | has build directive |
| &nbsp;&nbsp;**TestEnvExample** | *TestEnvExample* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_documents_api_key` | documents api key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_documents_signing_key` | documents signing key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_documents_allow_real_actions` | documents allow real actions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_documents_registry_signing_key` | documents registry signing key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_allow_real_actions_defaults_false` | allow real actions defaults false |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_real_secret_values` | no real secret values |
| &nbsp;&nbsp;**TestAllowRealActionsEnvVar** | *serve_model must honour MATRIXAI_ALLOW_REAL_ACTIONS from the environment.* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_env_var_true_enables_real_actions` | env var true enables real actions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_env_var_false_leaves_flag_false` | env var false leaves flag false |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_env_var_accepts_1_as_true` | env var accepts 1 as true |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explicit_flag_true_overrides_env_false` | explicit flag true overrides env false |
| **[test_pr4_c2_observability.py](../tests/test_pr4_c2_observability.py)** | Observability: GET /metrics Prometheus format, 7 core metrics, drift metrics (nil-safe) (22 tests) |
| &nbsp;&nbsp;**TestMetricsEndpoint** | *TestMetricsEndpoint* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_200` | returns 200 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_content_type_prometheus` | content type prometheus |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_auth_required` | no auth required |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_contains_help_lines` | contains help lines |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_contains_type_lines` | contains type lines |
| &nbsp;&nbsp;**TestMetricNames** | *TestMetricNames* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_requests_total_present` | requests total present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_requests_successful_present` | requests successful present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_requests_failed_present` | requests failed present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_requests_rate_limited_present` | requests rate limited present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_items_processed_present` | items processed present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_last_request_duration_present` | last request duration present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_uptime_seconds_present` | uptime seconds present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_project_label_present` | project label present |
| &nbsp;&nbsp;**TestMetricValues** | *TestMetricValues* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_uptime_positive` | uptime positive |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_requests_total_increments` | requests total increments |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_failed_increments_on_wrong_auth` | failed increments on wrong auth |
| &nbsp;&nbsp;**TestDriftMetrics** | *TestDriftMetrics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_metrics_absent_without_monitor` | drift metrics absent without monitor |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_metrics_present_with_monitor` | drift metrics present with monitor |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_drift_accuracy_value` | drift accuracy value |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_degradation_when_accuracy_perfect` | no degradation when accuracy perfect |
| &nbsp;&nbsp;**TestOpenAPIMetrics** | *TestOpenAPIMetrics* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metrics_in_openapi` | metrics in openapi |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metrics_path_has_get` | metrics path has get |
| **[test_pr4_c3_fixes.py](../tests/test_pr4_c3_fixes.py)** | PR4 fixes: reference_accuracy from params metrics, rollback persist, metrics nil-safe on startup (12 tests) |
| &nbsp;&nbsp;**TestF1ReferenceAccuracy** | *TestF1ReferenceAccuracy* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reference_accuracy_read_from_params` | ProductionMonitor._reference is set from params.json metrics.accuracy. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_reference_accuracy_none_when_no_params` | Without --params, reference_accuracy is None and degradation stays False. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_degradation_detected_with_reference` | With reference_accuracy set, degradation_detected fires when accuracy drops. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_explicit_reference_accuracy_overrides_params` | --reference-accuracy explicit value takes priority over params file. |
| &nbsp;&nbsp;**TestF2RollbackPersist** | *TestF2RollbackPersist* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rollback_event_persisted` | continual rollback writes .{name}_last_rollback.json with rolled_back_at. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_status_reads_rolled_back_at_not_executed_at` | continual status reads 'rolled_back_at' (real field), not 'executed_at'. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_auto_rollback_in_runbook_en` | RUNBOOK.md EN must NOT contain 'fires automatically' claim. |
| &nbsp;&nbsp;**TestF3ModelInfoGauge** | *TestF3ModelInfoGauge* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_info_present_with_params` | matrixai_model_info appears with populated labels when params loaded. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_info_line_value_is_1` | The matrixai_model_info gauge value must be 1. |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_model_info_emitted_without_params` | matrixai_model_info is emitted even without params (empty label values). |
| &nbsp;&nbsp;**TestF4DryRunCounter** | *TestF4DryRunCounter* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_counter_starts_at_zero` | dry run counter starts at zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_dry_run_counter_no_double_count` | Dry-run counter is exactly the number of execute-action calls, not doubled. |
| **[test_pr4_c4_key_rotation.py](../tests/test_pr4_c4_key_rotation.py)** | Key rotation: KeyStore, keys rotate/list CLI, fingerprint, purpose collision, history file (42 tests) |
| &nbsp;&nbsp;**TestKeyFingerprint** | *TestKeyFingerprint* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_returns_sha256_prefix` | returns sha256 prefix |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_length_is_7_plus_16` | length is 7 plus 16 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_keys_different_fingerprints` | different keys different fingerprints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_same_key_same_fingerprint` | same key same fingerprint |
| &nbsp;&nbsp;**TestKeyStoreRecord** | *TestKeyStoreRecord* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_adds_entry` | record adds entry |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_returns_fingerprint` | record returns fingerprint |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_is_idempotent` | record is idempotent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_entry_is_active` | record entry is active |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_record_sets_purpose` | record sets purpose |
| &nbsp;&nbsp;**TestKeyStoreRetire** | *TestKeyStoreRetire* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_retire_records_key` | retire records key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_retire_sets_rotated_at` | retire sets rotated at |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_retire_already_recorded_key` | retire already recorded key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_retire_returns_fingerprint` | retire returns fingerprint |
| &nbsp;&nbsp;**TestKeyStoreLookup** | *TestKeyStoreLookup* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_find_by_fingerprint` | find by fingerprint |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_find_by_fingerprint_missing` | find by fingerprint missing |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_keys_for_purpose_action` | keys for purpose action |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_keys_for_purpose_registry` | keys for purpose registry |
| &nbsp;&nbsp;**TestKeyStorePersistence** | *TestKeyStorePersistence* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_save_and_reload` | save and reload |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_history_file_permissions` | history file permissions |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_load_nonexistent_returns_empty` | load nonexistent returns empty |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_json_is_valid` | json is valid |
| &nbsp;&nbsp;**TestVerifyActionTraceWithKeystore** | *TestVerifyActionTraceWithKeystore* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifies_with_current_key` | verifies with current key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fails_with_wrong_key` | fails with wrong key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifies_with_historical_key_after_rotation` | verifies with historical key after rotation |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fails_when_key_not_in_history` | fails when key not in history |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_action_trace_with_keystore_helper` | verify action trace with keystore helper |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_unsigned_trace_returns_false` | unsigned trace returns false |
| &nbsp;&nbsp;**TestVerifyRegistryEntryWithKeystore** | *TestVerifyRegistryEntryWithKeystore* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifies_with_current_key` | verifies with current key |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verifies_via_fingerprint_lookup` | verifies via fingerprint lookup |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_fails_with_wrong_key_and_empty_store` | fails with wrong key and empty store |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_verify_entry_signature_with_keystore_helper` | verify entry signature with keystore helper |
| &nbsp;&nbsp;**TestCLIKeysRotate** | *TestCLIKeysRotate* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rotate_records_key_in_history` | rotate records key in history |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rotate_marks_key_as_retired` | rotate marks key as retired |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rotate_prints_next_steps` | rotate prints next steps |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rotate_without_key_fails` | rotate without key fails |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rotate_explicit_key_flag` | rotate explicit key flag |
| &nbsp;&nbsp;**TestCLIKeysList** | *TestCLIKeysList* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_shows_fingerprints` | list shows fingerprints |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_shows_retired_status` | list shows retired status |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_json_output` | list json output |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_list_empty_store` | list empty store |
| &nbsp;&nbsp;**TestKeyStoreDefaultPath** | *TestKeyStoreDefaultPath* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_uses_env_var_when_set` | uses env var when set |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_uses_registry_path_when_given` | uses registry path when given |
| **[test_pr4_c5_server_hardening.py](../tests/test_pr4_c5_server_hardening.py)** | Server hardening: RateLimiter sliding window, X-API-Key, CORS, do_OPTIONS, 429 Retry-After (37 tests) |
| &nbsp;&nbsp;**TestRateLimiter** | *TestRateLimiter* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_allows_up_to_limit` | allows up to limit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_blocks_at_limit` | blocks at limit |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_different_ips_independent` | different ips independent |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_disabled_at_zero` | disabled at zero |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_negative_also_disabled` | negative also disabled |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_thread_safe` | thread safe |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_window_allows_after_expiry` | window allows after expiry |
| &nbsp;&nbsp;**TestAuth** | *TestAuth* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_bearer_token_accepted` | bearer token accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_x_api_key_accepted` | x api key accepted |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_no_auth_returns_401` | no auth returns 401 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_bearer_returns_401` | wrong bearer returns 401 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_wrong_x_api_key_returns_401` | wrong x api key returns 401 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_health_no_auth_returns_200` | health no auth returns 200 |
| &nbsp;&nbsp;**TestRateLimitingIntegration** | *TestRateLimitingIntegration* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_first_two_requests_allowed` | first two requests allowed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_third_request_returns_429` | third request returns 429 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_disabled_rate_limit_never_blocks` | disabled rate limit never blocks |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_health_not_rate_limited` | health not rate limited |
| &nbsp;&nbsp;**TestCORSWildcard** | *TestCORSWildcard* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cors_wildcard_on_health` | cors wildcard on health |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cors_wildcard_on_predict` | cors wildcard on predict |
| &nbsp;&nbsp;**TestCORSSpecificOrigin** | *TestCORSSpecificOrigin* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matching_origin_echoed` | matching origin echoed |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_vary_header_present` | vary header present |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_non_matching_origin_omits_acao_header` | non matching origin omits acao header |
| &nbsp;&nbsp;**TestCORSPreflight** | *TestCORSPreflight* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_options_returns_204` | options returns 204 |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_options_allows_methods` | options allows methods |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_options_allows_auth_headers` | options allows auth headers |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_options_max_age_set` | options max age set |
| &nbsp;&nbsp;**TestServeModelSignature** | *TestServeModelSignature* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_rate_limit_param` | accepts rate limit param |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_accepts_cors_origins_param` | accepts cors origins param |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rate_limit_default_is_none` | rate limit default is none |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cors_origins_default_is_none` | cors origins default is none |
| &nbsp;&nbsp;**TestCLIHardeningArgs** | *TestCLIHardeningArgs* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rate_limit_in_help` | rate limit in help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cors_origin_in_help` | cors origin in help |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_rate_limit_env_mentioned` | matrixai rate limit env mentioned |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_matrixai_cors_origins_env_mentioned` | matrixai cors origins env mentioned |
| &nbsp;&nbsp;**TestEnvVarConfig** | *TestEnvVarConfig* |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_rate_limit_env_var` | rate limit env var |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_cors_origins_env_var` | cors origins env var |
| &nbsp;&nbsp;&nbsp;&nbsp;`test_metrics_tracks_rate_limited` | metrics tracks rate limited |

