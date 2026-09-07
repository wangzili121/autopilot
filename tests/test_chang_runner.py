from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_autopilot.calibration.models import (
    CalibrationSpec,
    ConfigurationSpec,
    EnvironmentContract,
    MetricConstraint,
    ObjectiveSpec,
    ProtocolSpec,
    SemanticContract,
    WorkloadContract,
)
from inference_autopilot.calibration.planning import build_plan, build_run_manifest
from inference_autopilot.runners.chang import (
    _REQUIRED_SOURCE_FILES,
    _load_toml,
    _validate_native_binding,
    _validate_prefix_sharded_runtime,
    _validated_wavefront_release_counts,
    arrival_trace_sha256,
    build_chang_run_bundle,
    observation_from_chang_result,
    source_snapshot_sha256,
)
from inference_autopilot.runners.chang_pressure import (
    _CallTraceBackend,
    _apply_model_runner_environment,
    _build_stage_wavefront_plan,
    _validate_and_apply,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_source(root: Path) -> tuple[Path, Path]:
    for relative in _REQUIRED_SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "# source fixture\n"
        if relative.endswith("gsm8k_async_benchmark.py"):
            content += "# conditional_is_small_proposal proposal_backend rollout_backend\n"
        if relative.endswith(("loader.py", "vllm_backend.py")):
            content += "# score_priority\n"
        if relative.endswith("vllm_backend.py"):
            content += "# runtime_metrics\n"
        path.write_text(content, encoding="utf-8")
    for model in ("base", "proposal"):
        model_path = root / model
        model_path.mkdir()
        (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    config = root / "configs" / "gsm8k.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """
[run]
subset_seed = 7
seed = 8

[models]
base = "base"
proposal = "proposal"
base_weight_sha256 = "6a09e667bb67ae853c6ef372a54ff53a510e527fade682d1b3d01b54a32f0e11"
proposal_weight_sha256 = "bb67ae8584caa73b3c6ef372fe94f82ba54ff53a5f1d36f1c510e5279b05688c"

[runtime]
backend = "vllm"
max_batch_size = 64
max_batch_tokens = 8192

[vllm]
enable_prefix_caching = true

[vllm.base]
gpu_memory_utilization = 0.54
max_num_seqs = 256
max_num_batched_tokens = 65536

[vllm.proposal]
gpu_memory_utilization = 0.36
max_num_seqs = 896
max_num_batched_tokens = 147456

[vllm.engine_kwargs]
enable_chunked_prefill = true

[generation]
max_new_tokens = 512

[conditional_is]
candidate_count = 15
rollout_count = 3
block_size = 128
apply_importance_correction = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    data = root / "data" / "gsm8k.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"question":"1+1","answer":"2"}\n', encoding="utf-8")
    return config, data


def _plan(
    data: Path,
    source_repo: Path,
    graph_settings: dict[str, object] | None = None,
):
    parameters = {
        "dataset": "GSM8K",
        "requests": 96,
        "workers": 96,
        "arrival_qps": 0.0,
        "prefix_mode": "unique",
        "prompt_length_regime": "short",
    }
    settings = {
        "base_max_num_seqs": 256,
        "proposal_max_num_seqs": 896,
        "base_max_num_batched_tokens": 65536,
        "proposal_max_num_batched_tokens": 147456,
        "base_memory_fraction": 0.54,
        "proposal_memory_fraction": 0.36,
        "base_batch_wait_seconds": 0.01,
        "proposal_batch_wait_seconds": 0.01,
        "base_score_priority": 1,
    }
    settings.update(graph_settings or {})
    spec = CalibrationSpec(
        campaign_id="runner-test",
        strong_baseline=True,
        protocol=ProtocolSpec("ABBA", 1, (101, 202)),
        semantic_contract=SemanticContract(
            _ALGORITHM,
            "exact",
            hashlib.sha256(b"conditional-is-graph").hexdigest(),
            {
                "candidate_count": 15,
                "rollout_count": 3,
                "block_size": 128,
                "total_length": 512,
                "apply_importance_correction": True,
                "automatic_prefix_caching": True,
                "chunked_prefill": True,
            },
        ),
        workload_contract=WorkloadContract(
            "short-p96",
            _sha256(data),
            arrival_trace_sha256(parameters),
            parameters,
        ),
        environment_contract=EnvironmentContract(
            "test-environment",
            {"device": "Ascend", "count": 1},
            {
                "inference_scaling_source_sha256": source_snapshot_sha256(
                    source_repo
                ),
                "vllm": "0.18",
            },
            {
                "base": {
                    "model_id": "qwen-1.5b",
                    "weights_sha256": (
                        "6a09e667bb67ae853c6ef372a54ff53a510e527fade682d1b3d01b54a32f0e11"
                    ),
                },
                "proposal": {
                    "model_id": "qwen-0.5b",
                    "weights_sha256": (
                        "bb67ae8584caa73b3c6ef372fe94f82ba54ff53a5f1d36f1c510e5279b05688c"
                    ),
                },
            },
        ),
        objective=ObjectiveSpec(
            "completed_qps",
            "maximize",
            (
                MetricConstraint("latency_seconds.p95", "<=", 120.0),
                MetricConstraint("accuracy", ">=", 0.4),
            ),
        ),
        required_metrics=(
            "completed_qps",
            "latency_seconds.p50",
            "latency_seconds.p95",
            "latency_seconds.p99",
            "accuracy",
            "preemptions",
            "proposal_kv_peak_fraction",
        ),
        baseline=ConfigurationSpec("manual", settings),
        candidates=(
            ConfigurationSpec(
                "candidate", {**settings, "proposal_max_num_seqs": 768}
            ),
        ),
    )
    return build_plan(spec)


_ALGORITHM = "conditional_is_small_proposal"


class ChangRunnerTest(unittest.TestCase):
    def test_call_trace_records_actual_sequence_and_token_shapes(self) -> None:
        class Request:
            def __init__(self, prefix, max_new_tokens=0, continuations=()):
                self.prefix = prefix
                self.max_new_tokens = max_new_tokens
                self.continuations = continuations

        class Backend:
            def sample_batch(self, requests):
                return list(requests)

            def score_batch(self, requests):
                return [tuple(0.0 for _ in request.continuations) for request in requests]

        traced = _CallTraceBackend(Backend(), "base", "engine_batch_wall_service_time")
        traced.sample_batch(
            [Request((1, 2), 4), Request((1, 2, 3), 2)]
        )
        traced.score_batch(
            [Request((1, 2), continuations=((3,), (4, 5, 6)))]
        )

        sample, score = traced.events
        self.assertEqual(sample["sequence_count"], 2)
        self.assertEqual(sample["shape_tokens"]["maximum"], 6)
        self.assertEqual(sample["token_extent"]["total"], 6)
        self.assertEqual(score["request_groups"], 1)
        self.assertEqual(score["sequence_count"], 2)
        self.assertEqual(score["shape_tokens"]["maximum"], 5)
        self.assertEqual(score["measurement_semantics"], "engine_batch_wall_service_time")

    def test_call_trace_preserves_native_completion_callbacks(self) -> None:
        class Backend:
            def sample_batch_with_callback(self, requests, on_complete):
                outputs = [f"sample-{index}" for index, _request in enumerate(requests)]
                for index, output in enumerate(outputs):
                    on_complete(index, output)
                return outputs

        requests = [
            type("Request", (), {"prefix": (1,), "max_new_tokens": 2})(),
            type("Request", (), {"prefix": (1, 2), "max_new_tokens": 3})(),
        ]
        completed = []
        traced = _CallTraceBackend(
            Backend(), "proposal", "engine_batch_wall_service_time"
        )

        outputs = traced.sample_batch_with_callback(
            requests, lambda index, output: completed.append((index, output))
        )

        self.assertEqual(outputs, ["sample-0", "sample-1"])
        self.assertEqual(completed, [(0, "sample-0"), (1, "sample-1")])
        self.assertEqual(len(traced.events), 1)
        self.assertEqual(traced.events[0]["sequence_count"], 2)

    @staticmethod
    def _prefix_runtime() -> dict[str, object]:
        waves = [
            {
                "shards": 2,
                "sequences": 6,
                "prefill_tokens": 12288,
                "parent_groups": 2,
                "maximum_shards_for_one_parent": 1,
                "maximum_shard_sequences": 3,
                "maximum_shard_prefill_tokens": 6144,
                "fairness_violation": False,
                "release_reason": "prefill_token_capacity",
                "inflight_waves_after_admission": 1,
                "inflight_sequences_after_admission": 6,
            },
            {
                "shards": 2,
                "sequences": 6,
                "prefill_tokens": 12288,
                "parent_groups": 2,
                "maximum_shards_for_one_parent": 1,
                "maximum_shard_sequences": 3,
                "maximum_shard_prefill_tokens": 6144,
                "fairness_violation": False,
                "release_reason": "prefill_token_capacity",
                "inflight_waves_after_admission": 1,
                "inflight_sequences_after_admission": 6,
            },
        ]
        return {
            "mode": "prefix_sharded",
            "max_wave_sequences": 48,
            "target_wave_sequences": 48,
            "max_wait_seconds": 0.5,
            "max_wave_prefill_tokens": 12288,
            "max_shard_sequences": 3,
            "max_shard_prefill_tokens": 8192,
            "max_shards_per_parent_per_wave": 1,
            "max_inflight_waves": 16,
            "max_inflight_sequences": 48,
            "admitted_groups": 2,
            "admitted_parent_groups": 2,
            "admitted_shards": 4,
            "admitted_sequences": 12,
            "admitted_prefill_tokens": 24576,
            "completed_parent_groups": 2,
            "repeated_prefix_run_count": 4,
            "preserved_prefix_run_count": 4,
            "forced_split_run_count": 0,
            "oversized_atomic_run_count": 0,
            "cancelled_shard_count": 0,
            "wave_count": 2,
            "partial_wave_count": 2,
            "oversized_wave_count": 0,
            "fairness_limited_wave_count": 1,
            "fairness_violation_count": 0,
            "cross_parent_wave_count": 2,
            "parent_groups_across_waves": 4,
            "maximum_shards_for_one_parent": 1,
            "inflight_waves": 0,
            "inflight_sequences": 0,
            "maximum_inflight_waves": 1,
            "maximum_inflight_sequences": 6,
            "sequence_credit_stall_count": 0,
            "sequence_credit_stall_seconds": 0.0,
            "streaming_sequence_credit_release_count": 12,
            "batch_sequence_credit_release_count": 0,
            "dispatch_failure_count": 0,
            "mean_admission_wait_seconds": 0.1,
            "maximum_admission_wait_seconds": 0.2,
            "mean_wave_sequence_utilization": 0.125,
            "mean_wave_prefill_utilization": 1.0,
            "mean_parent_groups_per_wave": 2.0,
            "mean_parent_completion_span_seconds": 0.3,
            "maximum_parent_completion_span_seconds": 0.4,
            "release_reason_counts": {"prefill_token_capacity": 2},
            "waves": waves,
        }

    def test_wavefront_release_reasons_are_fail_closed(self) -> None:
        runtime = {
            "release_reason_counts": {
                "target_sequences": 1,
                "prefill_token_capacity": 1,
            },
            "waves": [
                {"release_reason": "target_sequences"},
                {"release_reason": "prefill_token_capacity"},
            ],
        }
        self.assertEqual(
            _validated_wavefront_release_counts(runtime, 2, required=True),
            runtime["release_reason_counts"],
        )
        with self.assertRaisesRegex(ValueError, "does not match wave count"):
            _validated_wavefront_release_counts(runtime, 3, required=True)
        with self.assertRaisesRegex(ValueError, "lacks required"):
            _validated_wavefront_release_counts({}, 1, required=True)

    def test_model_runner_setting_maps_to_vllm_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _apply_model_runner_environment({"model_runner": "MRV1"})
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "0")

            _apply_model_runner_environment({"model_runner": "MRV2"})
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "1")

            with self.assertRaisesRegex(ValueError, "MRV1 or MRV2"):
                _apply_model_runner_environment({"model_runner": "unknown"})

    def test_prepare_bundle_passes_static_formal_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config, data = _write_fake_source(root)
            plan = _plan(data, root)
            output = Path(directory) / "bundle"
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config,
                data=data,
                output_dir=output,
            )
            bundle.write(output)

            self.assertTrue(bundle.formal_eligible)
            self.assertTrue((output / "run-manifest.json").is_file())
            self.assertTrue((output / "launch.json").is_file())
            self.assertEqual(
                bundle.launch["autopilot"]["snapshot_sha256"],
                bundle.effective_config["autopilot_snapshot_sha256"],
            )
            self.assertEqual(
                bundle.effective_config["deployment_settings"][
                    "proposal_max_num_seqs"
                ],
                896,
            )

    def test_prepare_bundle_reports_missing_score_priority_capability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config, data = _write_fake_source(root)
            for name in ("loader.py", "vllm_backend.py"):
                path = root / f"src/inference_scaling/arllm/backends/{name}"
                path.write_text("# no priority support\n", encoding="utf-8")
            plan = _plan(data, root)
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config,
                data=data,
                output_dir=Path(directory) / "bundle",
            )

        self.assertFalse(bundle.formal_eligible)
        self.assertIn(
            "base_score_priority_capability",
            bundle.launch["compatibility"]["failed_checks"],
        )

    def test_complete_graph_policy_is_manifest_bound_and_applied_per_role(self) -> None:
        graph_settings = {
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 2, 8, 16],
            "proposal_graph_mode": "NONE",
            "proposal_graph_capture_sizes": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config_path, data = _write_fake_source(root)
            plan = _plan(data, root, graph_settings)
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config_path,
                data=data,
                output_dir=Path(directory) / "bundle",
            )
            config, settings, _workload = _validate_and_apply(
                bundle.manifest,
                _load_toml(config_path),
                root,
            )

        self.assertTrue(bundle.formal_eligible)
        self.assertEqual(settings, bundle.manifest["configuration"]["settings"])
        self.assertEqual(
            config["vllm"]["base"]["engine_kwargs"]["compilation_config"],
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 8, 16],
            },
        )
        self.assertEqual(
            config["vllm"]["proposal"]["engine_kwargs"]["compilation_config"],
            {"cudagraph_mode": "NONE"},
        )

    def test_partial_graph_policy_fails_formal_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config, data = _write_fake_source(root)
            plan = _plan(data, root, {"base_graph_mode": "FULL_DECODE_ONLY"})
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config,
                data=data,
                output_dir=Path(directory) / "bundle",
            )

        self.assertFalse(bundle.formal_eligible)
        self.assertIn(
            "graph_policy_settings",
            bundle.launch["compatibility"]["failed_checks"],
        )

    def test_stage_wavefront_is_manifest_bound_and_planned(self) -> None:
        wavefront_settings = {
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 64, 128],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 128, 256, 384, 512],
            "proposal_stage_wavefront_mode": "auto",
            "proposal_graph_capture_ceiling": 512,
            "proposal_stage_wavefront_max_wait_seconds": 0.5,
            "proposal_stage_wavefront_min_utilization": 0.65,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config_path, data = _write_fake_source(root)
            plan = _plan(data, root, wavefront_settings)
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config_path,
                data=data,
                output_dir=Path(directory) / "bundle",
            )
            manifest = bundle.manifest
            wavefront = _build_stage_wavefront_plan(
                manifest["configuration"]["settings"],
                manifest["workload_contract"]["parameters"],
                manifest["semantic_contract"]["invariants"],
            )

        self.assertTrue(bundle.formal_eligible)
        self.assertIsNotNone(wavefront)
        assert wavefront is not None
        self.assertEqual(wavefront.selected.sequences_per_wave, 360)
        self.assertEqual(wavefront.selected.groups_per_wave, 8)

    def test_auto_wavefront_rejects_unattested_graph_ceiling(self) -> None:
        wavefront_settings = {
            "proposal_stage_wavefront_mode": "auto",
            "proposal_graph_capture_ceiling": 512,
            "proposal_stage_wavefront_max_wait_seconds": 0.5,
            "proposal_stage_wavefront_min_utilization": 0.65,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config_path, data = _write_fake_source(root)
            plan = _plan(data, root, wavefront_settings)
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config_path,
                data=data,
                output_dir=Path(directory) / "bundle",
            )

        self.assertFalse(bundle.formal_eligible)
        self.assertIn(
            "stage_wavefront_graph_ceiling",
            bundle.launch["compatibility"]["failed_checks"],
        )

    def test_auto_wavefront_rejects_an_oversized_context_lower_bound(self) -> None:
        wavefront_settings = {
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 64, 128],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 24, 48],
            "proposal_stage_wavefront_mode": "auto",
            "proposal_graph_capture_ceiling": 48,
            "proposal_stage_wavefront_max_wait_seconds": 0.5,
            "proposal_stage_wavefront_min_utilization": 0.65,
            "proposal_max_num_batched_tokens": 12288,
            "proposal_max_num_seqs": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config_path, data = _write_fake_source(root)
            short_plan = _plan(data, root, wavefront_settings)
            medium_parameters = {
                **short_plan.spec.workload_contract.parameters,
                "context_tokens": 2048,
                "prompt_length_regime": "medium2k",
            }
            medium_spec = replace(
                short_plan.spec,
                workload_contract=WorkloadContract(
                    "medium2k-p96",
                    _sha256(data),
                    arrival_trace_sha256(medium_parameters),
                    medium_parameters,
                ),
            )
            medium_plan = build_plan(medium_spec)
            bundle = build_chang_run_bundle(
                medium_plan,
                medium_plan.runs[0].run_id,
                source_repo=root,
                source_config=config_path,
                data=data,
                output_dir=Path(directory) / "bundle",
            )
            manifest = bundle.manifest

        self.assertFalse(bundle.formal_eligible)
        self.assertIn(
            "stage_wavefront_group_prefill_lower_bound",
            bundle.launch["compatibility"]["failed_checks"],
        )
        with self.assertRaisesRegex(ValueError, "validated group sharding"):
            _build_stage_wavefront_plan(
                manifest["configuration"]["settings"],
                manifest["workload_contract"]["parameters"],
                manifest["semantic_contract"]["invariants"],
            )

    def test_prefix_sharded_wavefront_accepts_a_medium_context_contract(self) -> None:
        wavefront_settings = {
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 64, 128],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 24, 48],
            "proposal_stage_wavefront_mode": "prefix_sharded",
            "proposal_graph_capture_ceiling": 48,
            "proposal_stage_wavefront_max_wait_seconds": 0.5,
            "proposal_stage_wavefront_min_utilization": 0.65,
            "proposal_stage_wavefront_max_shards_per_parent": 1,
            "proposal_stage_wavefront_max_shard_sequences": 3,
            "proposal_stage_wavefront_max_shard_prefill_tokens": 8192,
            "proposal_stage_wavefront_max_inflight_waves": 16,
            "proposal_max_num_batched_tokens": 12288,
            "proposal_max_num_seqs": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config_path, data = _write_fake_source(root)
            short_plan = _plan(data, root, wavefront_settings)
            medium_parameters = {
                **short_plan.spec.workload_contract.parameters,
                "context_tokens": 2048,
                "prompt_length_regime": "medium2k",
            }
            medium_spec = replace(
                short_plan.spec,
                workload_contract=WorkloadContract(
                    "medium2k-p96",
                    _sha256(data),
                    arrival_trace_sha256(medium_parameters),
                    medium_parameters,
                ),
            )
            medium_plan = build_plan(medium_spec)
            bundle = build_chang_run_bundle(
                medium_plan,
                medium_plan.runs[0].run_id,
                source_repo=root,
                source_config=config_path,
                data=data,
                output_dir=Path(directory) / "bundle",
            )
            manifest = bundle.manifest
            wavefront = _build_stage_wavefront_plan(
                manifest["configuration"]["settings"],
                manifest["workload_contract"]["parameters"],
                manifest["semantic_contract"]["invariants"],
            )

        self.assertTrue(bundle.formal_eligible)
        self.assertIsNotNone(wavefront)
        assert wavefront is not None
        self.assertEqual(wavefront.selected.sequences_per_wave, 48)
        self.assertNotIn(
            "stage_wavefront_group_prefill_lower_bound",
            bundle.launch["compatibility"]["failed_checks"],
        )
        self.assertNotIn(
            "stage_wavefront_prefix_run_prefill_lower_bound",
            bundle.launch["compatibility"]["failed_checks"],
        )

    def test_prefix_sharded_wavefront_requires_a_parent_fairness_cap(self) -> None:
        wavefront_settings = {
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 64, 128],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 24, 48],
            "proposal_stage_wavefront_mode": "prefix_sharded",
            "proposal_graph_capture_ceiling": 48,
            "proposal_stage_wavefront_max_wait_seconds": 0.5,
            "proposal_stage_wavefront_min_utilization": 0.65,
            "proposal_stage_wavefront_max_shard_sequences": 3,
            "proposal_stage_wavefront_max_shard_prefill_tokens": 8192,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            config_path, data = _write_fake_source(root)
            plan = _plan(data, root, wavefront_settings)
            bundle = build_chang_run_bundle(
                plan,
                plan.runs[0].run_id,
                source_repo=root,
                source_config=config_path,
                data=data,
                output_dir=Path(directory) / "bundle",
            )

        self.assertFalse(bundle.formal_eligible)
        self.assertIn(
            "stage_wavefront_values",
            bundle.launch["compatibility"]["failed_checks"],
        )

    def test_prefix_sharded_runtime_evidence_is_fail_closed(self) -> None:
        runtime = self._prefix_runtime()
        _validate_prefix_sharded_runtime(
            runtime,
            max_wave_sequences=48,
            max_wave_prefill_tokens=12288,
            max_shards_per_parent_per_wave=1,
            max_shard_sequences=3,
            max_shard_prefill_tokens=8192,
            max_inflight_waves=16,
        )

        mutations = {
            "missing forced split count": lambda value: value.pop(
                "forced_split_run_count"
            ),
            "incomplete barrier": lambda value: value.update(
                {"completed_parent_groups": 1}
            ),
            "fairness violation": lambda value: value.update(
                {"fairness_violation_count": 1}
            ),
            "oversized atomic run": lambda value: value.update(
                {"oversized_atomic_run_count": 1}
            ),
            "unfinished pipeline": lambda value: value.update(
                {"inflight_waves": 1, "inflight_sequences": 6}
            ),
            "dispatch failure": lambda value: value.update(
                {"dispatch_failure_count": 1}
            ),
            "sequence credit overflow": lambda value: value.update(
                {"maximum_inflight_sequences": 49}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                tampered = json.loads(json.dumps(runtime))
                mutate(tampered)
                with self.assertRaisesRegex(ValueError, "prefix-sharded"):
                    _validate_prefix_sharded_runtime(
                        tampered,
                        max_wave_sequences=48,
                        max_wave_prefill_tokens=12288,
                        max_shards_per_parent_per_wave=1,
                        max_shard_sequences=3,
                        max_shard_prefill_tokens=8192,
                        max_inflight_waves=16,
                    )

    def test_prefix_sharded_native_binding_includes_runtime_attestation(self) -> None:
        wavefront_settings = {
            "base_graph_mode": "FULL_DECODE_ONLY",
            "base_graph_capture_sizes": [1, 64, 128],
            "proposal_graph_mode": "FULL_DECODE_ONLY",
            "proposal_graph_capture_sizes": [1, 24, 48],
            "proposal_stage_wavefront_mode": "prefix_sharded",
            "proposal_graph_capture_ceiling": 48,
            "proposal_stage_wavefront_max_wait_seconds": 0.5,
            "proposal_stage_wavefront_min_utilization": 0.65,
            "proposal_stage_wavefront_max_shards_per_parent": 1,
            "proposal_stage_wavefront_max_shard_sequences": 3,
            "proposal_stage_wavefront_max_shard_prefill_tokens": 8192,
            "proposal_stage_wavefront_max_inflight_waves": 16,
            "proposal_max_num_batched_tokens": 12288,
            "proposal_max_num_seqs": 48,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            _config_path, data = _write_fake_source(root)
            plan = _plan(data, root, wavefront_settings)
            manifest = build_run_manifest(plan, plan.runs[0].run_id)
            wavefront = _build_stage_wavefront_plan(
                manifest["configuration"]["settings"],
                manifest["workload_contract"]["parameters"],
                manifest["semantic_contract"]["invariants"],
            )
            assert wavefront is not None
            runtime = self._prefix_runtime()
            native = {
                "run_manifest_sha256": manifest["run_manifest_sha256"],
                "launch_manifest_sha256": "c" * 64,
                "formal_preflight": True,
                "source_snapshot_sha256": source_snapshot_sha256(root),
                "method": _ALGORITHM,
                "runtime": manifest["configuration"]["settings"],
                "workload_parameters": manifest["workload_contract"][
                    "parameters"
                ],
                "semantic_invariants": manifest["semantic_contract"][
                    "invariants"
                ],
                "proposal_stage_wavefront": {
                    "enabled": True,
                    "mode": "prefix_sharded",
                    "plan": wavefront.to_dict(),
                    "runtime": runtime,
                },
            }
            _validate_native_binding(manifest, native)
            runtime.pop("cancelled_shard_count")
            with self.assertRaisesRegex(ValueError, "cancelled_shard_count"):
                _validate_native_binding(manifest, native)

    def test_native_result_is_strictly_bound_and_converted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            _config, data = _write_fake_source(root)
            plan = _plan(data, root)
            manifest = build_run_manifest(plan, plan.runs[0].run_id)
            native = {
                "schema_version": "1.0",
                "run_manifest_sha256": manifest["run_manifest_sha256"],
                "launch_manifest_sha256": "c" * 64,
                "formal_preflight": True,
                "source_snapshot_sha256": source_snapshot_sha256(root),
                "method": _ALGORITHM,
                "started_at_unix": 10.0,
                "finished_at_unix": 20.0,
                "elapsed_seconds": 10.0,
                "completed_qps": 9.6,
                "latency_seconds": {
                    "mean": 8.0,
                    "p50": 8.0,
                    "p95": 9.0,
                    "p99": 9.5,
                },
                "queue_wait_seconds": {"mean": 0.0, "p95": 0.0},
                "service_seconds": {"mean": 8.0, "p95": 9.0},
                "accuracy": 0.5,
                "runtime": manifest["configuration"]["settings"],
                "workload_parameters": manifest["workload_contract"]["parameters"],
                "semantic_invariants": manifest["semantic_contract"]["invariants"],
                "workload": {"prompt_tokens": {"mean": 100.0}},
                "continuous_batching": {"base": {}, "proposal": {}},
                "vllm_runtime_metrics": {
                    "base": {
                        "vllm:num_preemptions": {"maximum": 0.0}
                    },
                    "proposal": {
                        "vllm:num_preemptions": {"maximum": 0.0},
                        "vllm:kv_cache_usage_perc": {"maximum": 0.16},
                    },
                    "sample_count": 10,
                },
                "compute": {"total_forward_token_slots": 1000},
                "algorithm": {"rollout_count": 3},
            }
            path = Path(directory) / "native.json"
            path.write_text(json.dumps(native), encoding="utf-8")
            observation = observation_from_chang_result(manifest, path)

            self.assertEqual(observation.status, "success")
            self.assertEqual(observation.metrics["preemptions"], 0.0)
            self.assertEqual(observation.metrics["proposal_kv_peak_fraction"], 0.16)
            self.assertEqual(observation.artifact.sha256, _sha256(path))

            del native["vllm_runtime_metrics"]["base"]["vllm:num_preemptions"]
            path.write_text(json.dumps(native), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preemptions"):
                observation_from_chang_result(manifest, path)

            optional_spec = replace(
                plan.spec,
                required_metrics=tuple(
                    metric
                    for metric in plan.spec.required_metrics
                    if metric != "preemptions"
                ),
            )
            optional_plan = build_plan(optional_spec)
            optional_manifest = build_run_manifest(
                optional_plan, optional_plan.runs[0].run_id
            )
            native["run_manifest_sha256"] = optional_manifest[
                "run_manifest_sha256"
            ]
            path.write_text(json.dumps(native), encoding="utf-8")
            optional_observation = observation_from_chang_result(
                optional_manifest, path
            )
            self.assertNotIn("preemptions", optional_observation.metrics)

            extended_spec = replace(
                optional_spec,
                required_metrics=(
                    *optional_spec.required_metrics,
                    "compute.total_forward_token_slots",
                    "proposal_stage_wavefront_oversized_wave_count",
                ),
            )
            extended_plan = build_plan(extended_spec)
            extended_manifest = build_run_manifest(
                extended_plan, extended_plan.runs[0].run_id
            )
            native["run_manifest_sha256"] = extended_manifest[
                "run_manifest_sha256"
            ]
            path.write_text(json.dumps(native), encoding="utf-8")
            extended_observation = observation_from_chang_result(
                extended_manifest, path
            )
            self.assertEqual(
                extended_observation.metrics[
                    "proposal_stage_wavefront_oversized_wave_count"
                ],
                0.0,
            )

            native["runtime"] = {**native["runtime"], "base_max_num_seqs": 1}
            path.write_text(json.dumps(native), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "runtime settings"):
                observation_from_chang_result(extended_manifest, path)


if __name__ == "__main__":
    unittest.main()
