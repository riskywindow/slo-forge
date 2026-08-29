#!/usr/bin/env python3
"""Sole paid-resource coordinator for the one Experiment 004 kill arm.

The mature integrated coordinator retains reservation, provider-freshness,
bounded-process, immutable-download, settlement, and cleanup-input ownership.
This entry point installs only the exact kill config, 660-second envelope,
content-addressed seal, Modal surface, and result schema before delegating.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "artifacts/branchfabric/gpu-validation/experiment-004"
LEDGER_REFERENCE = "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
ATTEMPT_ID = "exp004-v11-kill-recompute-s41-g"
ATTEMPT_A_ID = "exp004-v11-kill-recompute-s41-a"
ATTEMPT_B_ID = "exp004-v11-kill-recompute-s41-b"
ATTEMPT_C_ID = "exp004-v11-kill-recompute-s41-c"
ATTEMPT_D_ID = "exp004-v11-kill-recompute-s41-d"
ATTEMPT_E_ID = "exp004-v11-kill-recompute-s41-e"
ATTEMPT_F_ID = "exp004-v11-kill-recompute-s41-f"
APP_NAME = "sloforge-branchfabric-gpu-reclamation-004-v11-kill-recompute"
REMOTE_PREFIX = "experiment-004/v11/kill-recompute/modal"
REMOTE_INVENTORY_ROOT = "experiment-004/v11"
FUNCTION_WALL_SECONDS = 660.0
FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS = 840.0
FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS = 720.0
COORDINATOR_PROCESS_GRACE_SECONDS = 60.0
KILL_MODAL_RETRIEVAL_TIMEOUT_SECONDS = (
    FIRST_FUNCTION_CALL_POLL_TIMEOUT_SECONDS
    + FUNCTION_CALL_REPOLL_TIMEOUT_SECONDS
    + COORDINATOR_PROCESS_GRACE_SECONDS
)
GPU_COUNT = 2
RESULTS_VOLUME = "sloforge-branchfabric-results"
SEAL_PATH = (
    EXPERIMENT_ROOT / "v11-final/kill-recompute/authorization/seal-verification-attempt-g.json"
)
MAKE_CHECK_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "authorization/make-check-attempt-g.json"
)
PRIOR_ATTEMPT_STATUS_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "status-attempt-a.json"
)
PRIOR_ATTEMPT_STATUS_SHA256 = "bc4e41571790e45a1984e3e56bd0f496941d1783db2fa1d36ce3f9ef4a92fad6"
PRIOR_ATTEMPT_CAPTURE_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "preflight/attempt-a-remote-prefix-inventory-failure.json"
)
PRIOR_ATTEMPT_CAPTURE_SHA256 = "fd0ad21bfd9df293cdbd9fbbf180a5617cd77786aeff94a00de858c15b6dce1b"
ATTEMPT_A_LEDGER_SHA256 = "e9b3b521da9f0a6eb3ac2e69030b7ce5560e4f5de11504644cf8f707237d8a16"
ATTEMPT_B_STATUS_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "status-attempt-b.json"
)
ATTEMPT_B_STATUS_SHA256 = "9c43d27066325accb5d5ab620df58391dbc63e3f49ea0b79e8db8f699c457147"
ATTEMPT_B_PROVIDER_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-cleanup-attempt-b.json"
)
ATTEMPT_B_PROVIDER_SHA256 = "8fd8d7f3a929a41bf8ebb86f936735b2790fedda024bf65cb5f4d2b3f84d9b34"
ATTEMPT_B_CHARGE_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "failures/exp004-v11-kill-recompute-s41-b-conservative-charge.json"
)
ATTEMPT_B_CHARGE_SHA256 = "523d078decd6c31b3b6107957331084166063083812b7d970ddb7194b19a8a1c"
ATTEMPT_C_STATUS_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "status-attempt-c.json"
)
ATTEMPT_C_STATUS_SHA256 = "9a916eb596edcca599c7461939521a14e6b8b7da74911d2d6091eb7bfe761bfb"
ATTEMPT_C_PROVIDER_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-cleanup-attempt-c.json"
)
ATTEMPT_C_PROVIDER_SHA256 = "736a05801aba8c5e5cef5375574d7ec4c32de6c79645767283f55036dd991818"
ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-inventory-attempt-c.json"
)
ATTEMPT_C_PROVIDER_INVENTORY_SHA256 = (
    "feb2e0d4ea2a73a7ad9c2788b6324ef9c41fe7651d45380626ba1e92acbab423"
)
ATTEMPT_C_MANIFEST_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/raw/"
    "exp004-v11-kill-recompute-s41-c/REMOTE_MANIFEST.json"
)
ATTEMPT_C_MANIFEST_SHA256 = "d4dca87d9f7f5899a463c118def8bfdcd5f5a5acfab6ccf7492afaadda245cc6"
ATTEMPT_C_CHARGE_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "failures/exp004-v11-kill-recompute-s41-c-conservative-charge.json"
)
ATTEMPT_C_CHARGE_SHA256 = "08c9b0c3cd50b58b81b170d8de5ec47141b3c3d3b4c8e1d7315c862ba6e68bb0"
ATTEMPT_C_SETTLED_LEDGER_SHA256 = "0c4c600b64eb58307758ee782d41f122c90b7610976e208b20a4b4b81ac57d9c"
ATTEMPT_C_REMAINING_GPU_SECONDS = 16236.58500987602
ATTEMPT_C_REVIEW_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
    "agent11-attempt-c-kill-history-boundary-review.json"
)
ATTEMPT_C_REVIEW_SHA256 = "3e367aeaa3418531c59f38ff77eaac2b7c6d3c036d9d226171aab15d11a82f20"
ATTEMPT_D_STATUS_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "status-attempt-d.json"
)
ATTEMPT_D_STATUS_SHA256 = "c82e72ea97933ee983e3610f12bdb6af9237f0f01f7d3e09447fc9143488ac63"
ATTEMPT_D_CAPTURE_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "preflight/attempt-d-terminal-prefix-policy-failure.json"
)
ATTEMPT_D_CAPTURE_SHA256 = "a33859db2c4d6ec9153dd0e54516934c24c7c4dee2bb941928aed909ccd3078a"
ATTEMPT_D_LEDGER_SHA256 = "0c4c600b64eb58307758ee782d41f122c90b7610976e208b20a4b4b81ac57d9c"
CONTINUATION_AUTHORIZATION_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "authorization/continuation-authorization-attempt-g.json"
)
CONTINUATION_AUTHORIZATION_SHA256 = (
    "e437bbac9b1626aca9e9296f67fc26ea80dd4479b8368e4f8843bf142ec9e734"
)
ATTEMPT_E_STATUS_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "status-attempt-e.json"
)
ATTEMPT_E_STATUS_SHA256 = "c043685ef74e853e45b24bf8324bb1abf4cde174fb85c1cca9367ef33f0faf5f"
ATTEMPT_E_PROVIDER_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-cleanup-attempt-e.json"
)
ATTEMPT_E_PROVIDER_SHA256 = "da208a07339e98dc631fee6be7be1bfc5fb3cb22e4b38674e5dba93339903a21"
ATTEMPT_E_PROVIDER_INVENTORY_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-inventory-attempt-e.json"
)
ATTEMPT_E_PROVIDER_INVENTORY_SHA256 = (
    "ecb836bab99b7a02b64b8b3e3088a510585c77bff14d5c40e0c74b3a5e3e9c66"
)
ATTEMPT_E_MANIFEST_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/raw/"
    "exp004-v11-kill-recompute-s41-e/REMOTE_MANIFEST.json"
)
ATTEMPT_E_MANIFEST_SHA256 = "935a51578d9a1e4676d3d86b59e65e901efc36f8b28154d55cadeecd1dd2d394"
ATTEMPT_E_CHARGE_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "failures/exp004-v11-kill-recompute-s41-e-conservative-charge.json"
)
ATTEMPT_E_CHARGE_SHA256 = "5bc8f8798e81af9b06634358574147c5c8930b322c284fecba4bef6603602e47"
ATTEMPT_E_RECONCILIATION_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "preflight/attempt-e-function-call-reconciliation.json"
)
ATTEMPT_E_RECONCILIATION_SHA256 = "0afbf3198d6c24c163e8d8d98e98bf4858821f1ee7619dbb3adff9d07e10259b"
ATTEMPT_E_SETTLED_LEDGER_SHA256 = "fec6652110c39d28447991c87b03dbc806b345de323483479dc3af36b0a6f17b"
ATTEMPT_E_REMAINING_GPU_SECONDS = 14916.58500987602
ATTEMPT_F_STATUS_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "status-attempt-f.json"
)
ATTEMPT_F_STATUS_SHA256 = "dbee5570ea716d6fc6ed602792ee2fb5c6182fdbb268b341a9b2c51d51a519be"
ATTEMPT_F_PROVIDER_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-cleanup-attempt-f.json"
)
ATTEMPT_F_PROVIDER_SHA256 = "16922c669509bcc4357b169633e89bcf862b6a0b51723ff0a884fe09c65dbd70"
ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "provider-inventory-attempt-f.json"
)
ATTEMPT_F_PROVIDER_INVENTORY_SHA256 = (
    "eb2f74c350b41b6bb66ce01233fefa58b9a2ef4ee910764fc035928d11491654"
)
ATTEMPT_F_MANIFEST_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/raw/"
    "exp004-v11-kill-recompute-s41-f/REMOTE_MANIFEST.json"
)
ATTEMPT_F_MANIFEST_SHA256 = "23655d7275730175054a63d18fc7d16b34eae093ab5fa2682bcc8b46e124dc83"
ATTEMPT_F_CHARGE_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
    "failures/exp004-v11-kill-recompute-s41-f-conservative-charge.json"
)
ATTEMPT_F_CHARGE_SHA256 = "38905e7cbd92ced0ea523a001954135155c2451140beb861b41a31a417aec174"
ATTEMPT_F_SETTLED_LEDGER_SHA256 = "aa81df8d3a26e370af1e715f86cb13a1b670cdccd4bbf6d4707dae1802a4ca26"
ATTEMPT_F_REMAINING_GPU_SECONDS = 13596.58500987602
CONTROL_GATE_POLICY_REVIEW_REFERENCE = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
    "agent13-attempt-f-kill-control-gate-methodology-review.json"
)
CONTROL_GATE_POLICY_REVIEW_SHA256 = (
    "b6828316d33d0fa2349b37c7d94abe5b22fd9e5d4a983593126940cd69ff6aa8"
)
FUNCTION_CALL_EVIDENCE_PARENT = EXPERIMENT_ROOT / "v11-final/kill-recompute/function-calls"
SOURCE_BINDINGS = {
    "methodology": "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py",
    "controller": "experiments/branchfabric/gpu_reclamation_kill_recompute_controller_v11.py",
    "worker": "experiments/branchfabric/gpu_reclamation_kill_recompute_worker_v11.py",
    "modal": "experiments/branchfabric/modal_gpu_reclamation_kill_recompute_v11.py",
    "coordinator": "tools/branchfabric-experiment-004-v11-kill-recompute.py",
    "tests": "tests/python/test_gpu_reclamation_kill_recompute_v11.py",
    "policy_review": CONTROL_GATE_POLICY_REVIEW_REFERENCE,
    "admission_tests": "tests/python/test_gpu_reclamation_kill_recompute_g_admission.py",
}
_POST_WORKER_CONTROLLER_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "attempt_id",
        "controller_pid",
        "started_ns",
        "ended_ns",
        "operation_deadline_ns",
        "cleanup_deadline_ns",
        "absolute_wall_seconds",
        "execution_mode",
        "terminal_phase",
        "sealed_evidence",
        "inventory_before",
        "inventory_after",
        "stable_physical_gpu_identity",
        "worker_pids",
        "worker_process_groups",
        "worker_session_ids",
        "worker_returncodes",
        "engine_start_evidence",
        "readiness_evidence",
        "readiness_comparability",
        "readiness_deadline_ns",
        "targeted_12rps_reproduction",
        "sanity_guard_pair",
        "worker_results",
        "cleanup_actions",
        "in_function_cleanup",
        "compute_processes_after",
        "cuda_clean_import_audits",
        "controller_error",
        "cleanup_error",
        "no_preservation_output_artifacts",
    }
)
_FORBIDDEN_PRESERVATION_TOKENS = (
    "checkpoint",
    "source-capture-commit",
    "pre-export",
    "post-export",
    "source-release",
    "state-pass",
    "movement",
    "optimized-export",
)
PREIMPORT_BINDINGS = {
    "base_coordinator": (
        "tools/branchfabric-experiment-004-v11-final.py",
        "e4686dd1c5a43fcb620d193806ecf8869a1c2bdde243a09ec3898bc9d0fcd73e",
    ),
    "base_modal": (
        "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py",
        "638555b54abb6804ced704f06fc6016c176c4f5f44abe7e32544e3b6dbc32cde",
    ),
    "methodology": (
        "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py",
        "23b8c3b2b6521c0dea1982c903f1a32c62a3e96d72712b7d35b08c82b20d1368",
    ),
    "kill_controller": (
        "experiments/branchfabric/gpu_reclamation_kill_recompute_controller_v11.py",
        "05d790d5d1ac1e23ba98cccc167a24855cc2b5ec5619fc295d7a3556c9e040c1",
    ),
    "kill_worker": (
        "experiments/branchfabric/gpu_reclamation_kill_recompute_worker_v11.py",
        "6266db3fbd3809ec7ab5561f33b8283df93801264068614fd6ec4bd9c7ee2579",
    ),
    "kill_modal": (
        "experiments/branchfabric/modal_gpu_reclamation_kill_recompute_v11.py",
        "858007ec9ab938e8199cc02fe5f5794ad1260de56d6903f5a6a7c588cc8f30ef",
    ),
    "kill_tests": (
        "tests/python/test_gpu_reclamation_kill_recompute_v11.py",
        "bef33d8d2c1363e6de0544564d93ed7b1d69bb92412d6500f23d37540a771d8f",
    ),
    "admission_tests": (
        "tests/python/test_gpu_reclamation_kill_recompute_g_admission.py",
        "fa7ffa29b70bcd6cc93e1cf2dfb1516cb78a16f5aa37edd98fe8003854fec0cc",
    ),
    "frozen_integrated_controller": (
        "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py",
        "c7c07ea0444874a100aa6d6da273058edd4f8cee03548e2c08c62bca5ecae2fc",
    ),
    "frozen_integrated_worker": (
        "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
        "a9312a017116ffc86ac311b578da67f2afb97198815f30369c8ee3e0159c2002",
    ),
    "frozen_trigger": (
        "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py",
        "9e81ab90ed6d7cb8f7f77e9fd63f2a0b7f3001422c53bcef84eb01344c83d2cc",
    ),
    "frozen_v11_worker": (
        "experiments/branchfabric/gpu_reclamation_worker_v11.py",
        "08449e5af89759847add7433054b2383eb9b1659785fb7e1184f25dc5e02de11",
    ),
    "v11_pipeline": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11.py",
        "697d9d9786fbd1759bc4e4242f5b507a7123d1f6daf1af42c12cffc790dde458",
    ),
    "v11_ownership": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py",
        "0ca553fbe21b431bc315553e792b7b4de523da3c44f84f36e95ff469ba3e6de0",
    ),
    "v11_sync": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11_sync.py",
        "05dc160d881f0b772d77db4f6b96298fe4290a7de81be9abcc876e83cada2820",
    ),
    "capacity_worker": (
        "experiments/branchfabric/gpu_capacity_calibration_worker.py",
        "2c41499a847879b813ce01470a7e32625d772fc07233753404296125c4afc225",
    ),
    "frozen_worker": (
        "experiments/branchfabric/gpu_reclamation_worker.py",
        "e6858e445ddbca912877fc67155dd9422c82d6f6fd4dcbf71871831dee15008a",
    ),
    "frozen_gpu1_serving": (
        "experiments/branchfabric/gpu_reclamation_v10_serving.py",
        "d53b007be4fc72d760ae7a83223773a206e0a0eed52137a834ce5b601bcad3ba",
    ),
    "frozen_sanity": (
        "experiments/branchfabric/gpu_reclamation_controller.py",
        "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9",
    ),
    "allocator_epochs": (
        "python/sloforge/continuum/adapters/vllm_allocator_epochs.py",
        "db543a8c4e7c2c7628e78f7dd0c4b04bcab60a4234bc2829787cca21214e35ce",
    ),
    "canonical_capture_plan": (
        "python/sloforge/continuum/adapters/vllm_reclamation.py",
        "fa8f59b6e6755d0eb5c6c008d26df48b5fcbb4ad85f2af3b84233808c0ba953f",
    ),
    "capacity_model": (
        "python/sloforge/helix/characterization/gpu_capacity_calibration.py",
        "7ffb993004fbc9b595cc3c5fc2e503ffd9dd2370befdeb8714d017badc4e0b3d",
    ),
    "real_runtime_adapter": (
        "python/sloforge/continuum/adapters/real_runtime.py",
        "b7b790e03da5f31c5e68dd1985b8163451d48ea8ee46c805fe20ba9f38aba8f9",
    ),
    "vllm_live_adapter": (
        "python/sloforge/continuum/adapters/vllm_live.py",
        "e22243b768c07d35afb6f98a40e979576a165c804b84944dba53d1aa1fca7ad4",
    ),
    "gpu_reclamation_accounting": (
        "python/sloforge/helix/characterization/gpu_reclamation_accounting.py",
        "0285017813cb07367bb26acb24798bfc2b036ca3955907c82c2c6a1f55ba2a1c",
    ),
    "gpu_reclamation_instrumentation": (
        "python/sloforge/helix/characterization/gpu_reclamation_instrumentation.py",
        "32b3d134496b627354f4813d94f9c827a6ca4d04b1e0092bd20c985419ecaf53",
    ),
    "gpu_reclamation_methodology": (
        "python/sloforge/helix/characterization/gpu_reclamation_methodology.py",
        "3a6217ec2b9c8ed4302a347ce5e36a0e1b1a19e8c1b1d38df6f6ea186b5507c3",
    ),
    "vllm_metadata_0230": (
        "python/sloforge/continuum/adapters/vllm_metadata_0230.py",
        "779c446cbc1ae302ee76bf9d9717f7037aafa09216758b2d0d6e7b1bd06dcf95",
    ),
    "capacity_controller": (
        "experiments/branchfabric/gpu_capacity_calibration_controller.py",
        "31fb70855e49d3f59493592de84120fc82f6a950e5b5edd0e775d5ba47978aa2",
    ),
    "v11_accounting": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11_accounting.py",
        "fe8c913af0414ec402ce230ffb19e8db9474bfe19b8ad539f80f0419c99b23ec",
    ),
    "vllm_live_trace": (
        "python/sloforge/continuum/adapters/vllm_live_trace.py",
        "1e1cbf417e3996d4f22e27e1097be5c0c6282b5d86cd3090a705337511266bb6",
    ),
    "external_adapter": (
        "python/sloforge/continuum/adapters/external.py",
        "94784ebdf6392f824bfde953f184042cb4027a158a2c8e9a34e1b14b4e7759f4",
    ),
    "sdk_adapter": (
        "python/sloforge/continuum/adapters/sdk.py",
        "6a1ca84eebf8865ee0e784961bc6e635789352b3c821fdbcafd5a512906bf361",
    ),
    "continuum_canonical": (
        "python/sloforge/continuum/ir/canonical.py",
        "ba84929cb375e0ac8b391970b5ce863b2680be0e98552b7e049f952bab7046c2",
    ),
    "gpu_reclamation_model": (
        "python/sloforge/helix/characterization/gpu_reclamation.py",
        "27c38db4bb4b7c1b61a29a54ab33f4e4769725a95754d86f4ce54e9a73a175ce",
    ),
    "gpu_reclamation_trace": (
        "python/sloforge/helix/characterization/gpu_reclamation_trace.py",
        "b9e416500c79a6e3620f6962d8fbdd187b8dfe40260758e9b7a3fe2e3659cc73",
    ),
    "trace_init": (
        "python/sloforge/helix/characterization/trace/__init__.py",
        "73cf51ae90fadf84a0fee028818ec91cafc0ba8aaa002c185c701ae533dd2542",
    ),
    "trace_adapters": (
        "python/sloforge/helix/characterization/trace/adapters.py",
        "84a9e8c190b9016843df89c18af7beb071bbde08097315a45104cdeb6dd889f2",
    ),
    "trace_buffer": (
        "python/sloforge/helix/characterization/trace/buffer.py",
        "4f768e042a07dc3acf206fa598589869233ddaa4b0365ed0dfb709cf7230d7f9",
    ),
    "trace_canonical": (
        "python/sloforge/helix/characterization/trace/canonical.py",
        "7a765e489858b834424b810e0a0936c3062c301972eb720357c117de02b534c2",
    ),
    "trace_conversion": (
        "python/sloforge/helix/characterization/trace/conversion.py",
        "f74e631d511bd62d28221c40f7e0ba9f55e83035dd70ddf9d0fa2ac7e2b70798",
    ),
    "trace_io": (
        "python/sloforge/helix/characterization/trace/io.py",
        "fecb5332fb35a614c4e6ef9e48b4991afe66a6203cbc346380cb63a0c6a2eba6",
    ),
    "trace_manifest": (
        "python/sloforge/helix/characterization/trace/manifest.py",
        "1fa378df8019c5767a865f70e5a7d4262bf85ebf55704aa88cac6848c91781a1",
    ),
    "trace_models": (
        "python/sloforge/helix/characterization/trace/models.py",
        "da3a5d4cbbe0c418c3ab0506821781f29c3bf74f26741695196a8dfb5ff3b8bf",
    ),
    "trace_parquet": (
        "python/sloforge/helix/characterization/trace/parquet.py",
        "4d450f39bc4c86783aaac69daaaf1b7dad1ff80175059e68a8ce2e9fec5a0bd0",
    ),
    "trace_perfetto": (
        "python/sloforge/helix/characterization/trace/perfetto.py",
        "4edc9cdfc3b2c00b50d3667cefdf2aa33eb43b2b4a4e6fae769b58b56047f67c",
    ),
    "v10_authorization": (
        "python/sloforge/helix/characterization/gpu_reclamation_v10_authorization.py",
        "3a06207c4865bf6df5fc38027e66d793726b89e0ca35a5472d9fa1f0d44ad840",
    ),
}


def _validate_preimport_closure(
    root: Path = ROOT,
    bindings: Mapping[str, tuple[str, str]] = PREIMPORT_BINDINGS,
) -> dict[str, dict[str, str]]:
    resolved_root = root.resolve(strict=True)
    verified: dict[str, dict[str, str]] = {}
    for label, (reference, expected_sha256) in bindings.items():
        path = root / reference
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"kill/recompute pre-import source is not regular: {label}")
        resolved = path.resolve(strict=True)
        if resolved_root not in resolved.parents or _sha256(path) != expected_sha256:
            raise RuntimeError(f"kill/recompute pre-import source changed: {label}")
        verified[label] = {"artifact": reference, "sha256": expected_sha256}
    return verified


def _load_base() -> Any:
    path = ROOT / "tools/branchfabric-experiment-004-v11-final.py"
    spec = importlib.util.spec_from_file_location("_sloforge_exp004_v11_coordinator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the bounded integrated coordinator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_function_call_evidence_fresh(repository_root: Path) -> None:
    expected_parent = FUNCTION_CALL_EVIDENCE_PARENT
    if repository_root.resolve() != ROOT.resolve():
        expected_parent = (
            repository_root
            / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
            "function-calls"
        )
    if expected_parent.exists() and (expected_parent.is_symlink() or not expected_parent.is_dir()):
        raise FileExistsError("kill/recompute FunctionCall evidence parent is not a safe directory")
    evidence_root = expected_parent / ATTEMPT_ID
    candidates = (
        evidence_root,
        evidence_root / "function-call.json",
        evidence_root / "first-poll-timeout.json",
        evidence_root / "terminal-poll-timeout.json",
    )
    collisions = [path for path in candidates if path.exists() or path.is_symlink()]
    if collisions:
        raise FileExistsError(
            "fresh kill/recompute attempt already has FunctionCall evidence: "
            + ", ".join(str(path) for path in collisions)
        )


def _validate_completed_function_call_evidence(
    *,
    config: Any,
    reservation_id: str,
    reservation_commitment_sha256: str,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    evidence_root = FUNCTION_CALL_EVIDENCE_PARENT / ATTEMPT_ID
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise RuntimeError("kill/recompute FunctionCall evidence root is absent or unsafe")
    allowed_names = {
        "function-call.json",
        "first-poll-timeout.json",
        "terminal-poll-timeout.json",
    }
    observed_children = tuple(evidence_root.iterdir())
    if (
        any(path.is_symlink() or not path.is_file() for path in observed_children)
        or {path.name for path in observed_children} - allowed_names
    ):
        raise RuntimeError("kill/recompute FunctionCall evidence file set changed")
    capture_path = evidence_root / "function-call.json"
    if capture_path.is_symlink() or not capture_path.is_file():
        raise RuntimeError("kill/recompute FunctionCall capture is absent or unsafe")
    capture = json.loads(capture_path.read_text())
    common_fields = {
        "attempt_id",
        "function_call_id",
        "reservation_id",
        "reservation_commitment_sha256",
        "config_sha256",
        "function_timeout_seconds",
        "function_startup_timeout_seconds",
        "first_poll_timeout_seconds",
        "repoll_timeout_seconds",
        "publication_grace_seconds",
    }
    call_id = capture.get("function_call_id") if isinstance(capture, dict) else None
    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        Experiment004V11KillRecomputeConfig,
    )

    validated_config = Experiment004V11KillRecomputeConfig.model_validate(config, strict=True)
    config_sha256 = hashlib.sha256(_BASE.canonical_json_bytes(validated_config)).hexdigest()
    if (
        not isinstance(capture, dict)
        or set(capture) != {"schema_version", "status", "recorded_at_utc", *common_fields}
        or capture.get("schema_version")
        != "sloforge.branchfabric.kill-recompute-function-call-capture/v1"
        or capture.get("status") != "SPAWNED_AND_PERSISTED_BEFORE_FIRST_POLL"
        or capture.get("attempt_id") != ATTEMPT_ID
        or re.fullmatch(r"fc-[0-9A-HJKMNP-TV-Z]{26}", str(call_id)) is None
        or capture.get("reservation_id") != reservation_id
        or capture.get("reservation_commitment_sha256") != reservation_commitment_sha256
        or capture.get("config_sha256") != config_sha256
        or capture.get("function_timeout_seconds") != 660
        or capture.get("function_startup_timeout_seconds") != 180
        or capture.get("first_poll_timeout_seconds") != 840
        or capture.get("repoll_timeout_seconds") != 720
        or capture.get("publication_grace_seconds") != 60
        or not isinstance(capture.get("recorded_at_utc"), str)
        or not capture["recorded_at_utc"]
    ):
        raise RuntimeError("kill/recompute FunctionCall capture differs from its exact contract")

    timeout_rows: dict[str, dict[str, Any]] = {}
    timeout_specs = {
        "first-poll-timeout.json": (
            "sloforge.branchfabric.kill-recompute-function-call-timeout/v1",
            "FIRST_POLL_TIMED_OUT_REPOLLING_SAME_CALL",
        ),
        "terminal-poll-timeout.json": (
            "sloforge.branchfabric.kill-recompute-function-call-terminal-timeout/v1",
            "SECOND_POLL_TIMED_OUT_FAILING_CLOSED",
        ),
    }
    for filename, (schema, status) in timeout_specs.items():
        path = evidence_root / filename
        if path.is_symlink():
            raise RuntimeError("kill/recompute FunctionCall timeout evidence is a symlink")
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "schema_version",
                "status",
                "same_function_call_repolled",
                "error",
                "recorded_at_utc",
                *common_fields,
            }
            or row.get("schema_version") != schema
            or row.get("status") != status
            or any(row.get(field) != capture.get(field) for field in common_fields)
            or row.get("same_function_call_repolled") is not True
            or not isinstance(row.get("error"), dict)
            or set(row["error"]) != {"type", "message"}
            or row["error"].get("type") != "TimeoutError"
            or not isinstance(row["error"].get("message"), str)
            or not isinstance(row.get("recorded_at_utc"), str)
            or not row["recorded_at_utc"]
        ):
            raise RuntimeError("kill/recompute FunctionCall timeout evidence changed")
        timeout_rows[filename] = row
    if "terminal-poll-timeout.json" in timeout_rows and completed.returncode == 0:
        raise RuntimeError("kill/recompute returned a result after a terminal poll timeout")
    if (
        "terminal-poll-timeout.json" in timeout_rows
        and "first-poll-timeout.json" not in timeout_rows
    ):
        raise RuntimeError("kill/recompute terminal timeout lacks its first timeout evidence")
    if "terminal-poll-timeout.json" in timeout_rows and "TimeoutError" not in completed.stderr:
        raise RuntimeError("kill/recompute terminal timeout evidence lacks a typed CLI failure")
    if completed.returncode == 0:
        envelope = _BASE._parse_exact_result(completed.stdout)
        remote = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(remote, dict) or remote.get("function_call_id") != call_id:
            raise RuntimeError("kill/recompute result changed FunctionCall identity")
    return {
        "schema_version": "sloforge.branchfabric.function-call-retrieval-validation/v1",
        "status": "PASS",
        "attempt_id": ATTEMPT_ID,
        "function_call_id": call_id,
        "capture_sha256": _sha256(capture_path),
        "timeout_evidence_sha256": {
            filename: _sha256(evidence_root / filename) for filename in sorted(timeout_rows)
        },
        "completed_returncode": completed.returncode,
    }


def _load_exact_json_binding(root: Path, reference: str, expected_sha256: str) -> dict[str, Any]:
    path = root / reference
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise RuntimeError(f"kill/recompute bound evidence changed: {reference}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"kill/recompute bound evidence is not an object: {reference}")
    return payload


def _validate_control_gate_policy_review(repository_root: Path) -> dict[str, Any]:
    review = _load_exact_json_binding(
        repository_root,
        CONTROL_GATE_POLICY_REVIEW_REFERENCE,
        CONTROL_GATE_POLICY_REVIEW_SHA256,
    )
    expected_top_level = {
        "schema_version",
        "status",
        "reviewer",
        "role",
        "review_mode",
        "attempt_id",
        "authorized_successor_attempt_id",
        "classification",
        "attempt_f_scientific_validity",
        "attempt_f_offline_promotion_permitted",
        "gpu_or_modal_invoked_by_reviewer",
        "network_invoked_by_reviewer",
        "runtime_source_edited_by_reviewer",
        "methodology_decision",
        "attempt_f_evidence",
        "attempt_f_exact_replay",
        "frozen_valid_and_same_allocation_comparators",
        "unchanged_fail_closed_control_gates",
        "required_rejections_and_boundary_behavior",
        "implementation_review",
        "d_independent_runtime_and_test_bindings",
        "frozen_integrated_k_binding",
        "consumer_checkpoint_and_cycle_free_trust",
        "verification",
        "authorization_recommendation",
    }
    expected_methodology = {
        "decision": "PASS_FOR_FRESH_ATTEMPT_G_ONLY",
        "old_predicate": "slope>0 AND final>initial AND second-half-mean>first-half-mean",
        "old_predicate_problem": (
            "A one-request net change in bounded in-service work over five one-second samples "
            "is sufficient to fail the old predicate even when waiting depth, TTFT, completion "
            "fraction, total outstanding, eventual accounting, cadence, and outputs all pass."
        ),
        "corrected_predicate": "slope>0.10 AND last-quarter-median>first-quarter-median+2",
        "corrected_predicate_origin": (
            "The exact material-and-persistent queue-drift predicate already used by "
            "gpu_capacity_calibration.py; this is not a threshold invented from Attempt F."
        ),
        "corrected_predicate_scope": "KILL_RECOMPUTE_ONLY",
        "quarter_count": "max(2,n//4)",
        "both_conditions_required": True,
        "strict_boundaries": True,
        "frozen_integrated_k_changed": False,
        "fresh_runtime_attempt_required": True,
    }
    expected_bindings = {
        "kill_worker": {
            "artifact": SOURCE_BINDINGS["worker"],
            "sha256": "6266db3fbd3809ec7ab5561f33b8283df93801264068614fd6ec4bd9c7ee2579",
        },
        "kill_controller": {
            "artifact": SOURCE_BINDINGS["controller"],
            "sha256": "05d790d5d1ac1e23ba98cccc167a24855cc2b5ec5619fc295d7a3556c9e040c1",
        },
        "kill_modal_surface": {
            "artifact": SOURCE_BINDINGS["modal"],
            "sha256": "858007ec9ab938e8199cc02fe5f5794ad1260de56d6903f5a6a7c588cc8f30ef",
        },
        "kill_tests": {
            "artifact": SOURCE_BINDINGS["tests"],
            "sha256": "ed5b76c2e8074a8cd6a33408532a0a4d60ca9fb6457dd37586b16516f3e50c2b",
        },
        "methodology": {
            "artifact": SOURCE_BINDINGS["methodology"],
            "sha256": "23b8c3b2b6521c0dea1982c903f1a32c62a3e96d72712b7d35b08c82b20d1368",
        },
        "established_capacity_policy": {
            "artifact": "python/sloforge/helix/characterization/gpu_capacity_calibration.py",
            "sha256": "7ffb993004fbc9b595cc3c5fc2e503ffd9dd2370befdeb8714d017badc4e0b3d",
        },
        "frozen_sanity_assessor": {
            "artifact": "experiments/branchfabric/gpu_reclamation_controller.py",
            "sha256": "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9",
        },
        "frozen_integrated_controller": {
            "artifact": PREIMPORT_BINDINGS["frozen_integrated_controller"][0],
            "sha256": PREIMPORT_BINDINGS["frozen_integrated_controller"][1],
        },
        "frozen_integrated_worker": {
            "artifact": PREIMPORT_BINDINGS["frozen_integrated_worker"][0],
            "sha256": PREIMPORT_BINDINGS["frozen_integrated_worker"][1],
        },
        "integrated_controller_tests": {
            "artifact": "tests/python/test_gpu_reclamation_integrated_controller_v11.py",
            "sha256": "00f5f2bee68735b5b84da52c731aa181f3a2e7466fbde67d1450d1167156496f",
        },
    }
    bindings = review.get("d_independent_runtime_and_test_bindings")
    if (
        set(review) != expected_top_level
        or review.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-attempt-f-kill-control-gate-methodology-review/v1"
        or review.get("status") != "PASS"
        or review.get("reviewer") != "agent13"
        or review.get("role") != "independent-scientific-methodology-reviewer"
        or review.get("review_mode") != "CPU_ONLY_READ_ONLY_IMMUTABLE_EVIDENCE_AND_SOURCE_REPLAY"
        or review.get("attempt_id") != ATTEMPT_F_ID
        or review.get("authorized_successor_attempt_id") != ATTEMPT_ID
        or review.get("classification") != "KILL_CONTROL_DRIFT_GATE_METHODOLOGY_BUG"
        or review.get("attempt_f_scientific_validity") is not False
        or review.get("attempt_f_offline_promotion_permitted") is not False
        or review.get("gpu_or_modal_invoked_by_reviewer") is not False
        or review.get("network_invoked_by_reviewer") is not False
        or review.get("runtime_source_edited_by_reviewer") is not False
        or review.get("methodology_decision") != expected_methodology
        or bindings != expected_bindings
    ):
        raise RuntimeError("kill/recompute Agent13 control-gate review semantics changed")
    # The review remains immutable pre-Attempt-G methodology provenance. Its
    # two test bindings record the exact prelaunch fixtures it reviewed; after
    # the measured G settlement those fixtures became terminal-aware and are
    # independently pinned by PREIMPORT_BINDINGS and the merge commit. Runtime
    # implementation bindings remain current-equality requirements here.
    current_runtime_labels = set(expected_bindings) - {
        "kill_tests",
        "integrated_controller_tests",
    }
    for label in current_runtime_labels:
        row = expected_bindings[label]
        path = repository_root / row["artifact"]
        if path.is_symlink() or not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError("kill/recompute Agent13 reviewed source binding changed")

    evidence = review.get("attempt_f_evidence")
    replay = review.get("attempt_f_exact_replay")
    frozen = review.get("frozen_integrated_k_binding")
    cycle = review.get("consumer_checkpoint_and_cycle_free_trust")
    if (
        not isinstance(evidence, dict)
        or evidence.get("status")
        != {
            "artifact": ATTEMPT_F_STATUS_REFERENCE,
            "sha256": ATTEMPT_F_STATUS_SHA256,
            "status": "INVALID",
            "scientifically_valid": False,
        }
        or evidence.get("provider_cleanup", {}).get("sha256") != ATTEMPT_F_PROVIDER_SHA256
        or evidence.get("provider_inventory_capture", {}).get("sha256")
        != ATTEMPT_F_PROVIDER_INVENTORY_SHA256
        or evidence.get("remote_manifest")
        != {
            "artifact": ATTEMPT_F_MANIFEST_REFERENCE,
            "sha256": ATTEMPT_F_MANIFEST_SHA256,
            "declared_artifact_count": 508,
            "verified_artifact_count": 508,
            "hash_size_or_set_mismatches": 0,
            "symlink_count": 0,
        }
        or evidence.get("conservative_charge", {}).get("sha256") != ATTEMPT_F_CHARGE_SHA256
        or evidence.get("settled_ledger")
        != {
            "artifact": "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json",
            "sha256": ATTEMPT_F_SETTLED_LEDGER_SHA256,
            "remaining_gpu_seconds": ATTEMPT_F_REMAINING_GPU_SECONDS,
            "reservations": [],
        }
        or not isinstance(replay, dict)
        or replay.get("evaluation_arrivals") != 36
        or replay.get("evaluation_completions") != 34
        or replay.get("completion_fraction") != 34 / 36
        or replay.get("completed_rate_requests_per_second") != 8.5
        or replay.get("p95_ttft_seconds") != 0.05750730925
        or replay.get("waiting_depth_samples") != [1, 1, 1, 1, 0]
        or replay.get("total_outstanding_samples") != [10, 12, 12, 12, 11]
        or replay.get("slope_requests_per_second") != 0.2
        or replay.get("first_quarter_median_depth") != 11.0
        or replay.get("last_quarter_median_depth") != 11.5
        or replay.get("old_drift_predicate") is not True
        or replay.get("corrected_material_persistent_drift") is not False
        or replay.get("corrected_control_interval_pass") is not True
        or review.get("unchanged_fail_closed_control_gates", {}).get("only_replaced_field")
        != "outstanding_queue_non_positive_trend"
        or review.get("required_rejections_and_boundary_behavior", {}).get(
            "material_persistent_samples_10_12_15_17_19"
        )
        != "FAIL"
        or not isinstance(frozen, dict)
        or frozen.get("integrated_controller_current_hash_matches_freeze") is not True
        or frozen.get("integrated_worker_current_hash_matches_freeze") is not True
        or frozen.get("freeze_record", {}).get("sha256")
        != "76bddc6fcd35c304ced2202e044babdb8bc56d108e3d7d63ac36c448b08ddbf3"
        or not isinstance(cycle, dict)
        or cycle.get("reviewed_pre_review_coordinator", {}).get("sha256")
        != "41455d1611f8bd3a8f039d2e2f85caf75ed137320187ca9d9b118800f3041983"
        or cycle.get("review_alone_authorizes_gpu") is not False
        or review.get("verification", {}).get("focused_kill_suite") != "PASS; 196 passed"
        or review.get("authorization_recommendation")
        != (
            "The kill-scoped control-drift correction is scientifically justified and "
            "fail-closed for a fresh Attempt G after a fresh repository-wide make check and "
            "content-addressed seal. Attempt F remains invalid and cannot be promoted offline. "
            "The final seal and loader must independently bind this review and the post-review "
            "coordinator; this review does not itself launch or authorize paid resources."
        )
    ):
        raise RuntimeError("kill/recompute Agent13 control-gate review evidence changed")
    return review


def _verify_attempt_c_manifest_closure(repository_root: Path) -> None:
    manifest_path = repository_root / ATTEMPT_C_MANIFEST_REFERENCE
    manifest = _load_exact_json_binding(
        repository_root, ATTEMPT_C_MANIFEST_REFERENCE, ATTEMPT_C_MANIFEST_SHA256
    )
    raw_root = manifest_path.parent
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("kill/recompute Attempt-C raw root is not a regular directory")
    descendants = list(raw_root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise RuntimeError("kill/recompute Attempt-C raw closure contains a symlink")
    rows = manifest.get("artifacts")
    if (
        manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or manifest.get("attempt_id") != ATTEMPT_C_ID
        or manifest.get("remote_prefix") != f"{REMOTE_PREFIX}/{ATTEMPT_C_ID}"
        or not isinstance(rows, list)
        or len(rows) != 478
    ):
        raise RuntimeError("kill/recompute Attempt-C manifest header changed")
    declared: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("kill/recompute Attempt-C manifest row changed")
        relative = row.get("relative_path")
        byte_count = row.get("bytes")
        sha256 = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or Path(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or type(byte_count) is not int
            or byte_count < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(sha256)) is None
            or relative in declared
        ):
            raise RuntimeError("kill/recompute Attempt-C manifest path is unsafe or duplicate")
        declared[relative] = (byte_count, str(sha256))
    actual = {
        path.relative_to(raw_root).as_posix(): path
        for path in descendants
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(declared):
        raise RuntimeError("kill/recompute Attempt-C manifest file set changed")
    for relative, path in actual.items():
        expected_bytes, expected_sha256 = declared[relative]
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
            raise RuntimeError("kill/recompute Attempt-C manifest content changed")


def _verify_attempt_e_manifest_closure(repository_root: Path) -> None:
    manifest_path = repository_root / ATTEMPT_E_MANIFEST_REFERENCE
    manifest = _load_exact_json_binding(
        repository_root, ATTEMPT_E_MANIFEST_REFERENCE, ATTEMPT_E_MANIFEST_SHA256
    )
    raw_root = manifest_path.parent
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("kill/recompute Attempt-E raw root is not a regular directory")
    descendants = list(raw_root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise RuntimeError("kill/recompute Attempt-E raw closure contains a symlink")
    rows = manifest.get("artifacts")
    if (
        manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or manifest.get("attempt_id") != ATTEMPT_E_ID
        or manifest.get("remote_prefix") != f"{REMOTE_PREFIX}/{ATTEMPT_E_ID}"
        or not isinstance(rows, list)
        or len(rows) != 16
    ):
        raise RuntimeError("kill/recompute Attempt-E manifest header changed")
    declared: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("kill/recompute Attempt-E manifest row changed")
        relative = row.get("relative_path")
        byte_count = row.get("bytes")
        sha256 = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or Path(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or type(byte_count) is not int
            or byte_count < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(sha256)) is None
            or relative in declared
        ):
            raise RuntimeError("kill/recompute Attempt-E manifest path is unsafe or duplicate")
        declared[relative] = (byte_count, str(sha256))
    actual = {
        path.relative_to(raw_root).as_posix(): path
        for path in descendants
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(declared):
        raise RuntimeError("kill/recompute Attempt-E manifest file set changed")
    for relative, path in actual.items():
        expected_bytes, expected_sha256 = declared[relative]
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
            raise RuntimeError("kill/recompute Attempt-E manifest content changed")


def _verify_attempt_f_manifest_closure(repository_root: Path) -> None:
    manifest_path = repository_root / ATTEMPT_F_MANIFEST_REFERENCE
    manifest = _load_exact_json_binding(
        repository_root, ATTEMPT_F_MANIFEST_REFERENCE, ATTEMPT_F_MANIFEST_SHA256
    )
    raw_root = manifest_path.parent
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("kill/recompute Attempt-F raw root is not a regular directory")
    descendants = list(raw_root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise RuntimeError("kill/recompute Attempt-F raw closure contains a symlink")
    rows = manifest.get("artifacts")
    if (
        manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or manifest.get("attempt_id") != ATTEMPT_F_ID
        or manifest.get("remote_prefix") != f"{REMOTE_PREFIX}/{ATTEMPT_F_ID}"
        or not isinstance(rows, list)
        or len(rows) != 508
    ):
        raise RuntimeError("kill/recompute Attempt-F manifest header changed")
    declared: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("kill/recompute Attempt-F manifest row changed")
        relative = row.get("relative_path")
        byte_count = row.get("bytes")
        sha256 = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or Path(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
            or type(byte_count) is not int
            or byte_count < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(sha256)) is None
            or relative in declared
        ):
            raise RuntimeError("kill/recompute Attempt-F manifest path is unsafe or duplicate")
        declared[relative] = (byte_count, str(sha256))
    actual = {
        path.relative_to(raw_root).as_posix(): path
        for path in descendants
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(declared):
        raise RuntimeError("kill/recompute Attempt-F manifest file set changed")
    for relative, path in actual.items():
        expected_bytes, expected_sha256 = declared[relative]
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
            raise RuntimeError("kill/recompute Attempt-F manifest content changed")


def _kill_attempt_history_is_valid(ledger_payload: Any) -> bool:
    history_fields = ("intervals", "conservative_failure_charges", "reservations")
    if not isinstance(ledger_payload, dict) or not all(
        isinstance(ledger_payload.get(field), list) for field in history_fields
    ):
        return False
    attempt_b_charges = [
        row
        for row in ledger_payload["conservative_failure_charges"]
        if isinstance(row, dict) and row.get("invocation_id") == ATTEMPT_B_ID
    ]
    attempt_c_charges = [
        row
        for row in ledger_payload["conservative_failure_charges"]
        if isinstance(row, dict) and row.get("invocation_id") == ATTEMPT_C_ID
    ]
    attempt_e_charges = [
        row
        for row in ledger_payload["conservative_failure_charges"]
        if isinstance(row, dict) and row.get("invocation_id") == ATTEMPT_E_ID
    ]
    attempt_f_charges = [
        row
        for row in ledger_payload["conservative_failure_charges"]
        if isinstance(row, dict) and row.get("invocation_id") == ATTEMPT_F_ID
    ]
    return bool(
        ledger_payload["reservations"] == []
        and not any(
            isinstance(row, dict)
            and row.get("invocation_id") in {ATTEMPT_A_ID, ATTEMPT_D_ID, ATTEMPT_ID}
            for field in history_fields
            for row in ledger_payload[field]
        )
        and not any(
            isinstance(row, dict)
            and row.get("invocation_id") in {ATTEMPT_B_ID, ATTEMPT_C_ID, ATTEMPT_E_ID, ATTEMPT_F_ID}
            for row in ledger_payload["intervals"]
        )
        and len(attempt_b_charges) == 1
        and attempt_b_charges[0].get("charged_wall_seconds") == 660.0
        and attempt_b_charges[0].get("gpu_count") == 2
        and attempt_b_charges[0].get("requested_gpu") == "A100-80GB"
        and attempt_b_charges[0].get("reservation_id") == "exp004-v11-kill-b-b2ed62946125171fbd29"
        and attempt_b_charges[0].get("config_sha256")
        == "afdd02aa5783e1c1387460cf7c594bdb22a0c6bbbbf8fee792902411e124e467"
        and attempt_b_charges[0].get("failure_stage") == "modal-launch"
        and attempt_b_charges[0].get("failure_evidence", {}).get("artifact_sha256")
        == ATTEMPT_B_CHARGE_SHA256
        and len(attempt_c_charges) == 1
        and attempt_c_charges[0].get("charged_wall_seconds") == 660.0
        and attempt_c_charges[0].get("gpu_count") == 2
        and attempt_c_charges[0].get("requested_gpu") == "A100-80GB"
        and attempt_c_charges[0].get("reservation_id") == "exp004-v11-kill-c-c580c2896a62212fb83f"
        and attempt_c_charges[0].get("config_sha256")
        == "e139a62ae055c5c749193628e9bb881005d44c009744c1a446d67385a2952c0b"
        and attempt_c_charges[0].get("failure_stage") == "result-validation"
        and attempt_c_charges[0].get("failure_evidence", {}).get("artifact_sha256")
        == ATTEMPT_C_CHARGE_SHA256
        and len(attempt_e_charges) == 1
        and attempt_e_charges[0].get("charged_wall_seconds") == 660.0
        and attempt_e_charges[0].get("gpu_count") == 2
        and attempt_e_charges[0].get("requested_gpu") == "A100-80GB"
        and attempt_e_charges[0].get("reservation_id") == "exp004-v11-kill-e-a2132b54b248c90b8385"
        and attempt_e_charges[0].get("config_sha256")
        == "9622b798da11ccf5090680c7d10f26841946d881bc62dfa0583043ba8b6eac78"
        and attempt_e_charges[0].get("failure_stage") == "modal-launch"
        and attempt_e_charges[0].get("failure_evidence", {}).get("artifact_sha256")
        == ATTEMPT_E_CHARGE_SHA256
        and len(attempt_f_charges) == 1
        and attempt_f_charges[0].get("charged_wall_seconds") == 660.0
        and attempt_f_charges[0].get("gpu_count") == 2
        and attempt_f_charges[0].get("requested_gpu") == "A100-80GB"
        and attempt_f_charges[0].get("reservation_id") == "exp004-v11-kill-f-21196ad66e1c8dfa16a0"
        and attempt_f_charges[0].get("config_sha256")
        == "0880d14f78a16108c53f72cd3f0669a9ffd6ccb15d90869568c3f669cff0e865"
        and attempt_f_charges[0].get("failure_stage") == "modal-launch"
        and attempt_f_charges[0].get("failure_evidence", {}).get("artifact_sha256")
        == ATTEMPT_F_CHARGE_SHA256
    )


def _validate_attempt_e_terminal_evidence(
    *,
    repository_root: Path,
    ledger_reference: str,
    ledger_sha256: str,
    remaining_gpu_seconds: float,
) -> None:
    status = _load_exact_json_binding(
        repository_root, ATTEMPT_E_STATUS_REFERENCE, ATTEMPT_E_STATUS_SHA256
    )
    provider = _load_exact_json_binding(
        repository_root, ATTEMPT_E_PROVIDER_REFERENCE, ATTEMPT_E_PROVIDER_SHA256
    )
    inventory = _load_exact_json_binding(
        repository_root,
        ATTEMPT_E_PROVIDER_INVENTORY_REFERENCE,
        ATTEMPT_E_PROVIDER_INVENTORY_SHA256,
    )
    charge = _load_exact_json_binding(
        repository_root, ATTEMPT_E_CHARGE_REFERENCE, ATTEMPT_E_CHARGE_SHA256
    )
    reconciliation = _load_exact_json_binding(
        repository_root,
        ATTEMPT_E_RECONCILIATION_REFERENCE,
        ATTEMPT_E_RECONCILIATION_SHA256,
    )
    _verify_attempt_e_manifest_closure(repository_root)
    raw_root = repository_root / ATTEMPT_E_MANIFEST_REFERENCE
    raw_root = raw_root.parent
    completion = json.loads((raw_root / "function-completion.json").read_text())
    controller = completion.get("controller")
    cleanup = completion.get("in_function_cleanup")
    abort = json.loads((raw_root / "barriers/abort.json").read_text())
    if (
        not isinstance(controller, dict)
        or controller.get("status") != "failed"
        or controller.get("controller_error") != {"type": "KeyboardInterrupt", "message": ""}
        or controller.get("readiness_evidence") != []
        or controller.get("worker_results") != []
        or controller.get("worker_returncodes") != {"serving": -15, "rollout": -15}
        or len(controller.get("engine_start_evidence", ())) != 2
        or controller.get("started_ns") != 24709260670
        or controller.get("ended_ns") != 122119152670
        or completion.get("function_call_id") != "fc-01M152JAGCBMQSBQ1TD55Y9HXV"
        or completion.get("controller_and_analysis_interval_seconds") != 119.882310463
        or completion.get("scientific_status") != "invalid"
        or abort
        != {
            "schema_version": "sloforge.branchfabric.integrated-abort/v1",
            "error": "KeyboardInterrupt",
            "issued_at_monotonic_ns": 120563690477,
        }
        or not isinstance(cleanup, dict)
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or cleanup.get("cleanup_errors") != []
        or cleanup.get("compute_processes_after") != []
    ):
        raise RuntimeError("kill/recompute Attempt-E remote terminal evidence is inconsistent")

    lifecycle = status.get("lifecycle")
    accounting = status.get("accounting")
    remote_bundle = status.get("remote_bundle")
    bindings = status.get("evidence_bindings")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-status/v1"
        or status.get("attempt_id") != ATTEMPT_E_ID
        or status.get("status") != "INVALID"
        or status.get("scientifically_valid") is not False
        or status.get("terminal_classification")
        != "INVALID_PAID_INFRASTRUCTURE_CLIENT_POLL_TIMEOUT_CANCELED_REMOTE_STARTUP"
        or status.get("failure_stage") != "RETAINED_ENGINE_INITIALIZATION_BEFORE_READINESS"
        or not isinstance(lifecycle, dict)
        or any(
            lifecycle.get(field) is not True
            for field in ("function_entry_reached", "controller_started", "both_engines_started")
        )
        or any(
            lifecycle.get(field) is not False
            for field in (
                "retained_engine_readiness_passed",
                "transaction_command_issued",
                "reclaim_trigger_reached",
                "source_state_discarded",
                "recompute_reached",
                "continuation_correctness_reached",
            )
        )
        or status.get("cleanup", {}).get("in_function_status") != "PASS"
        or status.get("cleanup", {}).get("provider_status") != "PASS"
        or status.get("cleanup", {}).get("provider_artifact_sha256") != ATTEMPT_E_PROVIDER_SHA256
        or not isinstance(accounting, dict)
        or accounting.get("charged_gpu_seconds") != 1320.0
        or accounting.get("charged_wall_seconds") != 660.0
        or accounting.get("actual_gpu_seconds") is not None
        or accounting.get("ledger_reservations_empty") is not True
        or accounting.get("ledger_remaining_gpu_seconds") != remaining_gpu_seconds
        or not isinstance(remote_bundle, dict)
        or remote_bundle.get("artifact") != ATTEMPT_E_MANIFEST_REFERENCE
        or remote_bundle.get("sha256") != ATTEMPT_E_MANIFEST_SHA256
        or remote_bundle.get("declared_files") != 16
        or remote_bundle.get("actual_files") != 16
        or any(
            remote_bundle.get(field) != 0
            for field in ("missing_files", "extra_files", "hash_or_size_mismatches", "symlinks")
        )
        or not isinstance(bindings, dict)
        or bindings.get("provider_cleanup")
        != [ATTEMPT_E_PROVIDER_REFERENCE, ATTEMPT_E_PROVIDER_SHA256]
        or bindings.get("provider_inventory")
        != [ATTEMPT_E_PROVIDER_INVENTORY_REFERENCE, ATTEMPT_E_PROVIDER_INVENTORY_SHA256]
        or bindings.get("conservative_charge")
        != [ATTEMPT_E_CHARGE_REFERENCE, ATTEMPT_E_CHARGE_SHA256]
        or bindings.get("settled_ledger") != [ledger_reference, ledger_sha256]
    ):
        raise RuntimeError("kill/recompute Attempt-E status evidence is inconsistent")

    apps = provider.get("apps")
    commands = inventory.get("commands")
    expected_nouns = ("app", "container", "volume", "endpoint", "queue", "dict")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != ATTEMPT_E_ID
        or provider.get("failure_boundary")
        != "CLIENT_FUNCTIONCALL_POLL_TIMEOUT_DURING_RETAINED_ENGINE_INITIALIZATION"
        or provider.get("in_function_cleanup_status") != "PASS"
        or provider.get("provider_cleanup_status") != "PASS"
        or not isinstance(apps, list)
        or [app.get("app_id") for app in apps]
        != ["ap-vp2tIIoVqVm8eDydpH9bOw", "ap-l4XMcxxRhQMFxS625O0utk"]
        or any(app.get("state") != "stopped" or app.get("tasks") != "0" for app in apps)
        or provider.get("active_apps") != []
        or any(
            provider.get(field) != []
            for field in ("containers", "endpoints", "queues", "dicts", "reservations")
        )
        or provider.get("volumes") != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or provider.get("raw_attempt_prefix_present") is not True
        or provider.get("raw_evidence_materialized") is not True
        or provider.get("valid_kill_recompute_measurement_materialized") is not False
        or provider.get("function_call_id") != "fc-01M152JAGCBMQSBQ1TD55Y9HXV"
        or provider.get("reservation_id") != "exp004-v11-kill-e-a2132b54b248c90b8385"
        or inventory.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-provider-inventory-capture/v1"
        or inventory.get("attempt_id") != ATTEMPT_E_ID
        or inventory.get("modal_client_version") != "1.5.3"
        or inventory.get("ledger_reservations") != []
        or not isinstance(commands, list)
        or len(commands) != len(expected_nouns)
    ):
        raise RuntimeError("kill/recompute Attempt-E provider evidence is inconsistent")
    for command, noun in zip(commands, expected_nouns, strict=True):
        if (
            not isinstance(command, dict)
            or command.get("argv") != ["uv", "run", "--locked", "modal", noun, "list", "--json"]
            or command.get("returncode") != 0
            or command.get("stderr") != ""
            or not isinstance(command.get("stdout"), str)
            or not isinstance(command.get("rows"), list)
        ):
            raise RuntimeError("kill/recompute Attempt-E provider command evidence changed")
        if noun in {"container", "endpoint", "queue", "dict"} and (
            command["stdout"] != "[]\n" or command["rows"] != []
        ):
            raise RuntimeError("kill/recompute Attempt-E zero-resource evidence changed")

    if (
        charge.get("schema_version") != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or charge.get("status") != "CONSERVATIVELY_CHARGED"
        or charge.get("attempt_id") != ATTEMPT_E_ID
        or charge.get("reservation_id") != "exp004-v11-kill-e-a2132b54b248c90b8385"
        or charge.get("failure_stage") != "modal-launch"
        or charge.get("charged_wall_seconds") != 660.0
        or charge.get("charged_gpu_seconds") != 1320.0
        or charge.get("actual_gpu_seconds") is not None
        or charge.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
        or charge.get("verified_remote_failure") is not None
        or reconciliation.get("schema_version")
        != "sloforge.branchfabric.kill-recompute-function-call-reconciliation/v1"
        or reconciliation.get("attempt_id") != ATTEMPT_E_ID
        or reconciliation.get("function_call_id") != "fc-01M152JAGCBMQSBQ1TD55Y9HXV"
        or reconciliation.get("result_status") != "REMOTE_ERROR"
        or reconciliation.get("exception")
        != {"type": "modal.exception.RemoteError", "args": [""], "message": ""}
        or reconciliation.get("result_retrieved") is not False
    ):
        raise RuntimeError(
            "kill/recompute Attempt-E charge/reconciliation evidence is inconsistent"
        )


def _validate_attempt_f_terminal_evidence(
    *,
    repository_root: Path,
    ledger_reference: str,
    ledger_sha256: str,
    remaining_gpu_seconds: float,
) -> None:
    status = _load_exact_json_binding(
        repository_root, ATTEMPT_F_STATUS_REFERENCE, ATTEMPT_F_STATUS_SHA256
    )
    provider = _load_exact_json_binding(
        repository_root, ATTEMPT_F_PROVIDER_REFERENCE, ATTEMPT_F_PROVIDER_SHA256
    )
    inventory = _load_exact_json_binding(
        repository_root,
        ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE,
        ATTEMPT_F_PROVIDER_INVENTORY_SHA256,
    )
    charge = _load_exact_json_binding(
        repository_root, ATTEMPT_F_CHARGE_REFERENCE, ATTEMPT_F_CHARGE_SHA256
    )
    _verify_attempt_f_manifest_closure(repository_root)
    raw_root = (repository_root / ATTEMPT_F_MANIFEST_REFERENCE).parent
    completion = json.loads((raw_root / "function-completion.json").read_text())
    controller = json.loads((raw_root / "controller-result.json").read_text())
    cleanup = json.loads((raw_root / "in_function_cleanup.json").read_text())
    control = json.loads((raw_root / "barriers/v11-gpu0-control-interval.json").read_text())
    rollout = json.loads((raw_root / "rollout/result.json").read_text())
    serving_failure = json.loads((raw_root / "serving/failure.json").read_text())
    lifecycle = status.get("lifecycle")
    accounting = status.get("accounting")
    bundle = status.get("remote_bundle")
    bindings = status.get("evidence_bindings")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-status/v1"
        or status.get("attempt_id") != ATTEMPT_F_ID
        or status.get("status") != "INVALID"
        or status.get("scientifically_valid") is not False
        or status.get("valid_kill_recompute_arm") is not False
        or status.get("terminal_classification") != "INVALID_SCIENTIFIC_CONTROL_STABILITY_GATE"
        or status.get("secondary_classification")
        != "LOCAL_FUNCTION_CALL_CONFIG_COMMITMENT_REPRESENTATION_GATE_BUG"
        or status.get("failure_stage") != "STRICT_NINE_RPS_CONTROL_STABILITY_INTERVAL"
        or not isinstance(lifecycle, dict)
        or any(
            lifecycle.get(field) is not True
            for field in (
                "function_entry_reached",
                "controller_started",
                "both_engines_started",
                "retained_engine_readiness_passed",
                "twelve_rps_probe_a_passed",
                "twelve_rps_probe_b_passed",
                "normal_twelve_rps_sanity_passed",
                "normal_fifteen_rps_sanity_passed",
                "strict_nine_rps_control_interval_reached",
                "transaction_command_issued",
                "source_state_discarded",
                "recompute_reached",
                "continuation_correctness_reached",
            )
        )
        or lifecycle.get("strict_nine_rps_control_interval_passed") is not False
        or lifecycle.get("controller_two_worker_success_pair_present") is not False
        or status.get("cleanup", {}).get("in_function_status") != "PASS"
        or status.get("cleanup", {}).get("provider_status") != "PASS"
        or status.get("cleanup", {}).get("provider_artifact_sha256") != ATTEMPT_F_PROVIDER_SHA256
        or not isinstance(accounting, dict)
        or accounting.get("charged_gpu_seconds") != 1320.0
        or accounting.get("charged_wall_seconds") != 660.0
        or accounting.get("actual_gpu_seconds") is not None
        or accounting.get("measured_settlement_performed") is not False
        or accounting.get("ledger_reservations_empty") is not True
        or accounting.get("ledger_remaining_gpu_seconds") != remaining_gpu_seconds
        or not isinstance(bundle, dict)
        or bundle.get("artifact") != ATTEMPT_F_MANIFEST_REFERENCE
        or bundle.get("sha256") != ATTEMPT_F_MANIFEST_SHA256
        or bundle.get("declared_files") != 508
        or bundle.get("actual_files") != 508
        or any(
            bundle.get(field) != 0
            for field in ("missing_files", "extra_files", "hash_or_size_mismatches", "symlinks")
        )
        or not isinstance(bindings, dict)
        or bindings.get("provider_cleanup")
        != [ATTEMPT_F_PROVIDER_REFERENCE, ATTEMPT_F_PROVIDER_SHA256]
        or bindings.get("provider_inventory")
        != [ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE, ATTEMPT_F_PROVIDER_INVENTORY_SHA256]
        or bindings.get("remote_manifest")
        != [ATTEMPT_F_MANIFEST_REFERENCE, ATTEMPT_F_MANIFEST_SHA256]
        or bindings.get("conservative_charge")
        != [ATTEMPT_F_CHARGE_REFERENCE, ATTEMPT_F_CHARGE_SHA256]
        or bindings.get("settled_ledger") != [ledger_reference, ledger_sha256]
    ):
        raise RuntimeError("kill/recompute Attempt-F status evidence is inconsistent")
    if (
        completion.get("status") != "failed"
        or completion.get("scientific_status") != "invalid"
        or completion.get("attempt_id") != ATTEMPT_F_ID
        or completion.get("function_call_id") != "fc-01M159JSQJM88WHXC702WG8GNS"
        or controller.get("status") != "failed"
        or controller.get("attempt_id") != ATTEMPT_F_ID
        or controller.get("worker_returncodes") != {"serving": 1, "rollout": 0}
        or controller.get("worker_results") != []
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or cleanup.get("cleanup_errors") != []
        or cleanup.get("compute_processes_after") != []
        or control.get("schema_version") != "sloforge.branchfabric.v11-control-interval/v2"
        or control.get("arrivals") != 36
        or control.get("completions_in_interval") != 34
        or control.get("completed_rate_per_second") != 8.5
        or control.get("completion_tracks_offer") is not True
        or control.get("p95_ttft_ns") != 57507309.25
        or control.get("waiting_queue_maximum_depth") != 1
        or control.get("total_outstanding_diagnostic", {}).get("samples")
        != [
            {"timestamp_ns": 213429975292, "total_outstanding": 10},
            {"timestamp_ns": 214429975292, "total_outstanding": 12},
            {"timestamp_ns": 215429975292, "total_outstanding": 12},
            {"timestamp_ns": 216429975292, "total_outstanding": 12},
            {"timestamp_ns": 217429975292, "total_outstanding": 11},
        ]
        or control.get("total_outstanding_diagnostic", {}).get("slope_requests_per_second") != 0.2
        or control.get("outstanding_queue_non_positive_trend") is not False
        or control.get("passed") is not False
        or rollout.get("schema_version") != "sloforge.branchfabric.kill-recompute-gpu1-result/v1"
        or rollout.get("status") != "succeeded"
        or rollout.get("attempt_id") != ATTEMPT_F_ID
        or rollout.get("recompute", {}).get("actual_submitted_replay_tokens") != 133128
        or rollout.get("recompute", {}).get("actual_computed_boundary_tokens") != 133120
        or rollout.get("recompute", {}).get("actual_live_uncomputed_tail_tokens") != 8
        or rollout.get("recompute", {}).get("passed") is not True
        or serving_failure
        != {
            "schema_version": "sloforge.branchfabric.kill-recompute-worker-failure/v1",
            "status": "failed",
            "error_type": "RuntimeError",
            "error_message": "integrated v11 9-rps control interval was not stable",
        }
    ):
        raise RuntimeError("kill/recompute Attempt-F remote evidence is inconsistent")
    apps = provider.get("apps")
    commands = inventory.get("commands")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != ATTEMPT_F_ID
        or provider.get("in_function_cleanup_status") != "PASS"
        or provider.get("provider_cleanup_status") != "PASS"
        or not isinstance(apps, list)
        or [app.get("app_id") for app in apps]
        != ["ap-pAywJ5QYg3BmZ0abZxdzhO", "ap-rjdwfRRFAldeVCQJhbpuCb"]
        or any(app.get("state") != "stopped" or app.get("tasks") != "0" for app in apps)
        or provider.get("active_apps") != []
        or any(
            provider.get(field) != []
            for field in ("containers", "endpoints", "queues", "dicts", "reservations")
        )
        or provider.get("volumes") != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or provider.get("raw_attempt_prefix_present") is not True
        or provider.get("valid_kill_recompute_measurement_materialized") is not False
        or provider.get("partial_diagnostic_recompute_evidence_materialized") is not True
        or provider.get("function_call_id") != "fc-01M159JSQJM88WHXC702WG8GNS"
        or inventory.get("attempt_id") != ATTEMPT_F_ID
        or inventory.get("modal_client_version") != "1.5.3"
        or inventory.get("ledger_reservations") != []
        or not isinstance(commands, list)
        or len(commands) != 6
        or any(
            not isinstance(command, dict)
            or command.get("returncode") != 0
            or command.get("stderr") != ""
            or not isinstance(command.get("stdout"), str)
            or not isinstance(command.get("rows"), list)
            for command in commands
        )
        or charge.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or charge.get("status") != "CONSERVATIVELY_CHARGED"
        or charge.get("attempt_id") != ATTEMPT_F_ID
        or charge.get("reservation_id") != "exp004-v11-kill-f-21196ad66e1c8dfa16a0"
        or charge.get("failure_stage") != "modal-launch"
        or charge.get("charged_gpu_seconds") != 1320.0
        or charge.get("actual_gpu_seconds") is not None
        or charge.get("verified_remote_failure") is not None
    ):
        raise RuntimeError("kill/recompute Attempt-F provider/charge evidence is inconsistent")


def _validate_attempt_c_d_e_and_f_continuation(
    *,
    repository_root: Path,
    ledger_reference: str,
    ledger_sha256: str,
    remaining_gpu_seconds: float,
) -> None:
    _validate_control_gate_policy_review(repository_root)
    status = _load_exact_json_binding(
        repository_root, ATTEMPT_C_STATUS_REFERENCE, ATTEMPT_C_STATUS_SHA256
    )
    provider = _load_exact_json_binding(
        repository_root, ATTEMPT_C_PROVIDER_REFERENCE, ATTEMPT_C_PROVIDER_SHA256
    )
    provider_inventory = _load_exact_json_binding(
        repository_root,
        ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE,
        ATTEMPT_C_PROVIDER_INVENTORY_SHA256,
    )
    charge = _load_exact_json_binding(
        repository_root, ATTEMPT_C_CHARGE_REFERENCE, ATTEMPT_C_CHARGE_SHA256
    )
    review = _load_exact_json_binding(
        repository_root, ATTEMPT_C_REVIEW_REFERENCE, ATTEMPT_C_REVIEW_SHA256
    )
    attempt_d_status = _load_exact_json_binding(
        repository_root, ATTEMPT_D_STATUS_REFERENCE, ATTEMPT_D_STATUS_SHA256
    )
    attempt_d_capture = _load_exact_json_binding(
        repository_root, ATTEMPT_D_CAPTURE_REFERENCE, ATTEMPT_D_CAPTURE_SHA256
    )
    continuation = _load_exact_json_binding(
        repository_root,
        CONTINUATION_AUTHORIZATION_REFERENCE,
        CONTINUATION_AUTHORIZATION_SHA256,
    )
    _verify_attempt_c_manifest_closure(repository_root)
    _validate_attempt_e_terminal_evidence(
        repository_root=repository_root,
        ledger_reference=ledger_reference,
        ledger_sha256=ATTEMPT_E_SETTLED_LEDGER_SHA256,
        remaining_gpu_seconds=ATTEMPT_E_REMAINING_GPU_SECONDS,
    )
    _validate_attempt_f_terminal_evidence(
        repository_root=repository_root,
        ledger_reference=ledger_reference,
        ledger_sha256=ledger_sha256,
        remaining_gpu_seconds=remaining_gpu_seconds,
    )

    expected_d_invocation = (
        "/usr/bin/caffeinate -dimsu env SLOFORGE_GPU_BUDGET_USD=80 uv run --locked "
        "python tools/branchfabric-experiment-004-v11-kill-recompute.py --config-path "
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/kill-recompute/"
        "exp004-v11-kill-recompute-s41-d-config.json"
    )
    expected_d_tail = (
        "FileExistsError: fresh kill/recompute attempt remote prefix exists before reservation: "
        f"{REMOTE_PREFIX}/{ATTEMPT_C_ID}"
    )
    if (
        attempt_d_capture.get("schema_version")
        != (
            "sloforge.branchfabric.experiment-004-v11-kill-recompute-"
            "pre-reservation-terminal-capture/v1"
        )
        or attempt_d_capture.get("status") != "FAIL_CLOSED"
        or attempt_d_capture.get("attempt_id") != ATTEMPT_D_ID
        or attempt_d_capture.get("failure_stage") != "REMOTE_PREFIX_FRESHNESS_BEFORE_PREBUILD"
        or attempt_d_capture.get("failure_classification")
        != "INVALID_PRE_RESERVATION_HISTORICAL_C_REMOTE_PREFIX_POLICY_BUG"
        or attempt_d_capture.get("terminal_invocation") != expected_d_invocation
        or attempt_d_capture.get("exit_code") != 1
        or attempt_d_capture.get("observed_at_utc") is not None
        or attempt_d_capture.get("full_pty_combined_output_available") is not False
        or attempt_d_capture.get("pty_combined_output_exact_tail") != expected_d_tail
        or attempt_d_capture.get("original_cli_inventory_rows_available") is not False
        or attempt_d_capture.get("original_cli_inventory_rows") is not None
        or any(
            attempt_d_capture.get(field) is not False
            for field in (
                "prebuild_started",
                "reservation_created",
                "gpu_or_function_invoked",
                "provider_resource_created",
            )
        )
        or attempt_d_capture.get("ledger_sha256_before") != ATTEMPT_D_LEDGER_SHA256
        or attempt_d_capture.get("ledger_sha256_after") != ATTEMPT_D_LEDGER_SHA256
    ):
        raise RuntimeError("kill/recompute Attempt-D terminal evidence is inconsistent")

    attempt_d_bindings = attempt_d_status.get("evidence_bindings")
    if (
        attempt_d_status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-pre-reservation-status/v1"
        or attempt_d_status.get("status") != "INVALID"
        or attempt_d_status.get("scientifically_valid") is not False
        or attempt_d_status.get("attempt_id") != ATTEMPT_D_ID
        or attempt_d_status.get("failure_classification")
        != "INVALID_PRE_RESERVATION_HISTORICAL_C_REMOTE_PREFIX_POLICY_BUG"
        or attempt_d_status.get("failure_stage") != "REMOTE_PREFIX_FRESHNESS_BEFORE_PREBUILD"
        or any(
            attempt_d_status.get(field) is not False
            for field in (
                "scientific_work_started",
                "prebuild_started",
                "reservation_created",
                "function_invoked",
                "gpu_invoked",
                "controller_started",
                "source_state_discarded",
                "recompute_started",
                "original_cli_inventory_rows_available",
            )
        )
        or attempt_d_status.get("cleanup")
        != {
            "in_function": "NOT_APPLICABLE_FUNCTION_NOT_INVOKED",
            "provider": "NOT_APPLICABLE_NO_PROVIDER_RESOURCE_CREATED",
            "ledger_unchanged": True,
        }
        or attempt_d_bindings
        != {
            "config": {
                "artifact": (
                    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                    "kill-recompute/exp004-v11-kill-recompute-s41-d-config.json"
                ),
                "sha256": "d51a8164fff20dd105c5f7f8e60d508629011ba8f9451e620ce1f49157cc98f4",
            },
            "seal": {
                "artifact": (
                    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                    "kill-recompute/authorization/seal-verification-attempt-d.json"
                ),
                "sha256": "fe7ea3909d4e8216e5622669cd3e28724baaf29642a6cdca88bdee01e884daf0",
            },
            "make_check": {
                "artifact": (
                    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
                    "kill-recompute/authorization/make-check-attempt-d.json"
                ),
                "sha256": "31f6c82ca30ccc87aecce73eda319125bfcf12d7b723722210bfdabc15d5e0f1",
            },
            "terminal_capture": {
                "artifact": ATTEMPT_D_CAPTURE_REFERENCE,
                "sha256": ATTEMPT_D_CAPTURE_SHA256,
            },
            "ledger_before_and_after": {
                "artifact": ledger_reference,
                "sha256": ATTEMPT_D_LEDGER_SHA256,
            },
        }
        or attempt_d_status.get("kill_and_recompute_measurement_available") is not False
    ):
        raise RuntimeError("kill/recompute Attempt-D status evidence is inconsistent")
    for label in ("config", "seal", "make_check"):
        binding = attempt_d_bindings[label]
        _load_exact_json_binding(repository_root, binding["artifact"], binding["sha256"])

    lifecycle = status.get("lifecycle")
    measurement = status.get("measurement")
    accounting = status.get("accounting")
    remote_bundle = status.get("remote_bundle")
    bindings = status.get("evidence_bindings")
    provider_bindings = provider.get("evidence_bindings")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-status/v1"
        or status.get("attempt_id") != ATTEMPT_C_ID
        or status.get("status") != "INVALID"
        or status.get("scientifically_valid") is not False
        or status.get("terminal_classification")
        != "INVALID_KILL_BRANCH_HISTORY_LIVE_KV_BOUNDARY_PRE_DISCARD"
        or status.get("failure_stage") != "KILL_TRANSACTION_HISTORY_VALIDATION_BEFORE_STATE_DISCARD"
        or status.get("primary_failure", {}).get("message")
        != "kill/recompute branch history differs from its live KV boundary"
        or not isinstance(lifecycle, dict)
        or any(
            lifecycle.get(field) is not True
            for field in (
                "function_entry_reached",
                "controller_started",
                "both_engines_started",
                "retained_engine_readiness_passed",
                "twelve_rps_probe_a_passed",
                "twelve_rps_probe_b_passed",
                "fifteen_rps_overload_probe_passed",
                "reclaim_trigger_reached",
            )
        )
        or lifecycle.get("history_validation_passed") is not False
        or any(
            lifecycle.get(field) is not False
            for field in (
                "ownership_capture_reached",
                "source_state_discarded",
                "source_release_reached",
                "hbm_reclaim_reached",
                "temporary_gpu1_serving_reached",
                "recompute_reached",
                "continuation_correctness_reached",
            )
        )
        or not isinstance(measurement, dict)
        or measurement.get("availability") != "UNAVAILABLE_PRE_DISCARD_HISTORY_VALIDATION_FAILURE"
        or any(
            measurement.get(field) is not None
            for field in (
                "actual_recompute_submitted_tokens",
                "actual_recompute_uncached_tokens",
                "actual_recompute_gpu_seconds",
                "state_discard_seconds",
                "continuation_exact_branch_count",
            )
        )
        or status.get("cleanup", {}).get("in_function_status") != "PASS"
        or status.get("cleanup", {}).get("provider_status") != "PASS"
        or status.get("cleanup", {}).get("provider_artifact_sha256") != ATTEMPT_C_PROVIDER_SHA256
        or not isinstance(accounting, dict)
        or accounting.get("settlement_kind") != "CONSERVATIVE_FULL_BOUND"
        or accounting.get("actual_gpu_seconds") is not None
        or accounting.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
        or accounting.get("charged_gpu_seconds") != 1320.0
        or accounting.get("charged_wall_seconds") != 660.0
        or accounting.get("gpu_count") != 2
        or accounting.get("reservation_id") != "exp004-v11-kill-c-c580c2896a62212fb83f"
        or accounting.get("ledger_reservations_empty") is not True
        or accounting.get("ledger_remaining_gpu_seconds") != ATTEMPT_C_REMAINING_GPU_SECONDS
        or status.get("secondary_local_condition", {}).get("scientific_root_cause") is not False
        or not isinstance(remote_bundle, dict)
        or remote_bundle.get("artifact") != ATTEMPT_C_MANIFEST_REFERENCE
        or remote_bundle.get("sha256") != ATTEMPT_C_MANIFEST_SHA256
        or any(
            remote_bundle.get(field) != 0
            for field in ("missing_files", "extra_files", "hash_or_size_mismatches", "symlinks")
        )
        or remote_bundle.get("declared_files") != 478
        or remote_bundle.get("actual_files") != 478
        or not isinstance(bindings, dict)
        or bindings.get("provider_cleanup")
        != [ATTEMPT_C_PROVIDER_REFERENCE, ATTEMPT_C_PROVIDER_SHA256]
        or bindings.get("conservative_charge")
        != [ATTEMPT_C_CHARGE_REFERENCE, ATTEMPT_C_CHARGE_SHA256]
        or bindings.get("settled_ledger") != [ledger_reference, ATTEMPT_C_SETTLED_LEDGER_SHA256]
    ):
        raise RuntimeError("kill/recompute Attempt-C status evidence is inconsistent")

    provider_apps = provider.get("apps")
    expected_provider_commands = (
        ("app", provider_apps),
        ("container", []),
        ("volume", ["sloforge-branchfabric-results", "sloforge-model-cache"]),
        ("queue", []),
        ("dict", []),
        ("endpoint", []),
    )
    inventory_commands = provider_inventory.get("commands")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-provider-cleanup/v1"
        or provider.get("attempt_id") != ATTEMPT_C_ID
        or provider.get("status") != "PASS"
        or provider.get("failure_boundary")
        != "KILL_TRANSACTION_HISTORY_VALIDATION_BEFORE_STATE_DISCARD"
        or provider.get("function_entry_reached") is not True
        or provider.get("controller_started") is not True
        or provider.get("in_function_cleanup_status") != "PASS"
        or provider.get("provider_cleanup_status") != "PASS"
        or not isinstance(provider_apps, list)
        or len(provider_apps) != 2
        or any(
            not isinstance(app, dict)
            or app.get("description") != APP_NAME
            or app.get("state") != "stopped"
            or app.get("tasks") != "0"
            for app in provider_apps
        )
        or provider.get("active_apps") != []
        or any(
            provider.get(field) != []
            for field in ("containers", "endpoints", "queues", "dicts", "reservations")
        )
        or provider.get("volumes") != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or provider.get("authorized_volumes_only") is not True
        or provider.get("raw_attempt_prefix_present") is not True
        or provider.get("raw_evidence_materialized") is not True
        or provider.get("valid_kill_recompute_measurement_materialized") is not False
        or provider.get("reservation_id") != "exp004-v11-kill-c-c580c2896a62212fb83f"
        or not isinstance(provider_bindings, dict)
        or provider_bindings.get("remote_manifest")
        != {"artifact": ATTEMPT_C_MANIFEST_REFERENCE, "sha256": ATTEMPT_C_MANIFEST_SHA256}
        or provider_bindings.get("conservative_charge")
        != {"artifact": ATTEMPT_C_CHARGE_REFERENCE, "sha256": ATTEMPT_C_CHARGE_SHA256}
        or provider_bindings.get("settled_ledger")
        != {"artifact": ledger_reference, "sha256": ATTEMPT_C_SETTLED_LEDGER_SHA256}
        or provider_bindings.get("provider_inventory_capture")
        != {
            "artifact": ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE,
            "sha256": ATTEMPT_C_PROVIDER_INVENTORY_SHA256,
        }
        or provider_inventory.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-provider-inventory-capture/v1"
        or provider_inventory.get("attempt_id") != ATTEMPT_C_ID
        or provider_inventory.get("modal_client_version") != "1.5.3"
        or provider_inventory.get("ledger_reservations") != []
        or not isinstance(inventory_commands, list)
        or len(inventory_commands) != len(expected_provider_commands)
    ):
        raise RuntimeError("kill/recompute Attempt-C provider evidence is inconsistent")
    for command, (noun, expected_rows) in zip(
        inventory_commands, expected_provider_commands, strict=True
    ):
        rows = command.get("rows") if isinstance(command, dict) else None
        if (
            not isinstance(command, dict)
            or command.get("argv") != ["modal", noun, "list", "--json"]
            or command.get("returncode") != 0
            or not isinstance(rows, list)
        ):
            raise RuntimeError("kill/recompute Attempt-C provider command evidence changed")
        if noun == "app":
            if rows != provider_apps:
                raise RuntimeError("kill/recompute Attempt-C provider app rows changed")
        elif noun == "volume":
            if [row.get("name") for row in rows if isinstance(row, dict)] != expected_rows:
                raise RuntimeError("kill/recompute Attempt-C provider volume rows changed")
        elif rows != expected_rows:
            raise RuntimeError("kill/recompute Attempt-C provider zero-resource rows changed")

    if (
        charge.get("schema_version") != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or charge.get("status") != "CONSERVATIVELY_CHARGED"
        or charge.get("attempt_id") != ATTEMPT_C_ID
        or charge.get("reservation_id") != "exp004-v11-kill-c-c580c2896a62212fb83f"
        or charge.get("failure_stage") != "result-validation"
        or charge.get("charged_wall_seconds") != 660.0
        or charge.get("charged_gpu_seconds") != 1320.0
        or charge.get("actual_gpu_seconds") is not None
        or charge.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
        or charge.get("verified_remote_failure") is not None
    ):
        raise RuntimeError("kill/recompute Attempt-C charge evidence is inconsistent")

    final_fix = review.get("final_fix_bindings")
    if (
        review.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-attempt-c-kill-history-boundary-review/v1"
        or review.get("status") != "PASS"
        or review.get("reviewer") != "agent11"
        or review.get("attempt_id") != ATTEMPT_C_ID
        or review.get("classification") != "KILL_HISTORY_GATE_FIELD_CONFLATION_BUG"
        or review.get("scientific_validity") is not False
        or review.get("evidence_scope", {}).get("destructive_discard_reached") is not False
        or review.get("evidence_scope", {}).get("recompute_reached") is not False
        or review.get("root_cause", {}).get("root_cause_class") != "GATE_BUG"
        or review.get("root_cause", {}).get("not_allocator_or_lifecycle_corruption") is not True
        or review.get("semantic_derivation", {}).get("computed_tokens_per_branch") != 16640
        or review.get("semantic_derivation", {}).get("uncomputed_sampled_tail_tokens_per_branch")
        != 1
        or review.get("semantic_derivation", {}).get("live_history_tokens_per_branch") != 16641
        or review.get("semantic_derivation", {}).get("full_live_history_tokens_across_branches")
        != 133128
        or review.get("semantic_derivation", {}).get(
            "attempt_c_recompute_tokens_actually_submitted"
        )
        != 0
        or review.get("attempt_c_evidence", {}).get("remote_manifest", {}).get("sha256")
        != ATTEMPT_C_MANIFEST_SHA256
        or review.get("attempt_c_evidence", {}).get("status", {}).get("sha256")
        != ATTEMPT_C_STATUS_SHA256
        or review.get("attempt_c_evidence", {}).get("provider_cleanup", {}).get("sha256")
        != ATTEMPT_C_PROVIDER_SHA256
        or review.get("attempt_c_evidence", {}).get("provider_inventory_capture", {}).get("sha256")
        != ATTEMPT_C_PROVIDER_INVENTORY_SHA256
        or review.get("attempt_c_evidence", {}).get("settled_ledger", {}).get("sha256")
        != ATTEMPT_C_SETTLED_LEDGER_SHA256
        or not isinstance(final_fix, dict)
        or final_fix.get("worker")
        != {
            "artifact": SOURCE_BINDINGS["worker"],
            "sha256": "38bc1a074655da0c0d9a24ed0c06700f787c5e263d3ec6349785ee16bfb930d3",
        }
        or final_fix.get("controller")
        != {
            "artifact": SOURCE_BINDINGS["controller"],
            "sha256": "e6d6f414b8c20d7f1beb814665c77294cc79fae1a859adfb4546b542c76d5ea1",
        }
        or final_fix.get("methodology")
        != {
            "artifact": SOURCE_BINDINGS["methodology"],
            "sha256": "23b8c3b2b6521c0dea1982c903f1a32c62a3e96d72712b7d35b08c82b20d1368",
        }
        or review.get("authorization_recommendation")
        != (
            "The history-boundary root cause and fix are sufficient evidence for an exact "
            "fresh-ID D seal after a fresh repository-wide make check, current-ledger/budget "
            "review, immutable C evidence binding, and ordinary collision checks. This review "
            "does not itself authorize or launch a GPU invocation."
        )
    ):
        raise RuntimeError("kill/recompute Attempt-C forensic review is inconsistent")

    expected_evidence_bindings = {
        "attempt_c_status": {
            "artifact": ATTEMPT_C_STATUS_REFERENCE,
            "sha256": ATTEMPT_C_STATUS_SHA256,
        },
        "attempt_c_provider_cleanup": {
            "artifact": ATTEMPT_C_PROVIDER_REFERENCE,
            "sha256": ATTEMPT_C_PROVIDER_SHA256,
        },
        "attempt_c_provider_inventory": {
            "artifact": ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE,
            "sha256": ATTEMPT_C_PROVIDER_INVENTORY_SHA256,
        },
        "attempt_c_remote_manifest": {
            "artifact": ATTEMPT_C_MANIFEST_REFERENCE,
            "sha256": ATTEMPT_C_MANIFEST_SHA256,
        },
        "attempt_c_conservative_charge": {
            "artifact": ATTEMPT_C_CHARGE_REFERENCE,
            "sha256": ATTEMPT_C_CHARGE_SHA256,
        },
        "attempt_c_forensic_review": {
            "artifact": ATTEMPT_C_REVIEW_REFERENCE,
            "sha256": ATTEMPT_C_REVIEW_SHA256,
        },
        "attempt_d_status": {
            "artifact": ATTEMPT_D_STATUS_REFERENCE,
            "sha256": ATTEMPT_D_STATUS_SHA256,
        },
        "attempt_d_terminal_capture": {
            "artifact": ATTEMPT_D_CAPTURE_REFERENCE,
            "sha256": ATTEMPT_D_CAPTURE_SHA256,
        },
        "attempt_e_status": {
            "artifact": ATTEMPT_E_STATUS_REFERENCE,
            "sha256": ATTEMPT_E_STATUS_SHA256,
        },
        "attempt_e_provider_cleanup": {
            "artifact": ATTEMPT_E_PROVIDER_REFERENCE,
            "sha256": ATTEMPT_E_PROVIDER_SHA256,
        },
        "attempt_e_provider_inventory": {
            "artifact": ATTEMPT_E_PROVIDER_INVENTORY_REFERENCE,
            "sha256": ATTEMPT_E_PROVIDER_INVENTORY_SHA256,
        },
        "attempt_e_remote_manifest": {
            "artifact": ATTEMPT_E_MANIFEST_REFERENCE,
            "sha256": ATTEMPT_E_MANIFEST_SHA256,
        },
        "attempt_e_conservative_charge": {
            "artifact": ATTEMPT_E_CHARGE_REFERENCE,
            "sha256": ATTEMPT_E_CHARGE_SHA256,
        },
        "attempt_e_function_call_reconciliation": {
            "artifact": ATTEMPT_E_RECONCILIATION_REFERENCE,
            "sha256": ATTEMPT_E_RECONCILIATION_SHA256,
        },
        "attempt_f_status": {
            "artifact": ATTEMPT_F_STATUS_REFERENCE,
            "sha256": ATTEMPT_F_STATUS_SHA256,
        },
        "attempt_f_provider_cleanup": {
            "artifact": ATTEMPT_F_PROVIDER_REFERENCE,
            "sha256": ATTEMPT_F_PROVIDER_SHA256,
        },
        "attempt_f_provider_inventory": {
            "artifact": ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE,
            "sha256": ATTEMPT_F_PROVIDER_INVENTORY_SHA256,
        },
        "attempt_f_remote_manifest": {
            "artifact": ATTEMPT_F_MANIFEST_REFERENCE,
            "sha256": ATTEMPT_F_MANIFEST_SHA256,
        },
        "attempt_f_conservative_charge": {
            "artifact": ATTEMPT_F_CHARGE_REFERENCE,
            "sha256": ATTEMPT_F_CHARGE_SHA256,
        },
        "settled_ledger": {"artifact": ledger_reference, "sha256": ledger_sha256},
    }
    if (
        continuation.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-continuation-authorization/v5"
        or continuation.get("status") != "AUTHORIZED"
        or continuation.get("authorized_attempt_id") != ATTEMPT_ID
        or continuation.get("maximum_fresh_invocations") != 1
        or continuation.get("maximum_wall_seconds") != 660.0
        or continuation.get("maximum_gpu_seconds") != 1320.0
        or continuation.get("source")
        != "USER_COMPLETION_CONTRACT_CONTINUE_AFTER_ATTEMPT_F_INVALID_CONTROL_STABILITY_GATE"
        or continuation.get("valid_kill_recompute_measurements_before_g") != 0
        or continuation.get("prior_attempts")
        != {
            "a": {"status": "INVALID_PRE_RESERVATION", "charged_gpu_seconds": 0.0},
            "b": {
                "status": "INVALID_INFRASTRUCTURE_BEFORE_HANDLER",
                "charged_gpu_seconds": 1320.0,
                "scientific_arm_executed": False,
            },
            "c": {
                "status": "INVALID_PRE_DISCARD_HISTORY_GATE_BUG",
                "charged_gpu_seconds": 1320.0,
                "destructive_discard_reached": False,
                "valid_measurement_materialized": False,
            },
            "d": {
                "status": "INVALID_PRE_RESERVATION_HISTORICAL_PREFIX_POLICY_BUG",
                "charged_gpu_seconds": 0.0,
                "prebuild_started": False,
                "reservation_created": False,
                "gpu_or_function_invoked": False,
            },
            "e": {
                "status": (
                    "INVALID_PAID_INFRASTRUCTURE_CLIENT_POLL_TIMEOUT_CANCELED_REMOTE_STARTUP"
                ),
                "charged_gpu_seconds": 1320.0,
                "both_engines_started": True,
                "retained_engine_readiness_passed": False,
                "transaction_started": False,
                "destructive_discard_reached": False,
                "recompute_reached": False,
                "valid_measurement_materialized": False,
                "in_function_cleanup": "PASS",
                "provider_cleanup": "PASS",
            },
            "f": {
                "status": "INVALID_SCIENTIFIC_CONTROL_STABILITY_GATE",
                "charged_gpu_seconds": 1320.0,
                "retained_engine_readiness_passed": True,
                "transaction_started": True,
                "destructive_discard_reached": True,
                "recompute_reached": True,
                "strict_nine_rps_control_interval_passed": False,
                "valid_measurement_materialized": False,
                "partial_diagnostic_recompute_evidence_materialized": True,
                "in_function_cleanup": "PASS",
                "provider_cleanup": "PASS",
            },
        }
        or continuation.get("evidence_bindings") != expected_evidence_bindings
        or continuation.get("budget")
        != {
            "settled_ledger_sha256": ledger_sha256,
            "remaining_gpu_seconds": remaining_gpu_seconds,
            "required_gpu_seconds_with_15_percent_margin": 1518.0,
            "margin_gate_pass": True,
        }
        or continuation.get("retrieval_fix_contract")
        != {
            "first_poll_timeout_seconds": 840.0,
            "same_function_call_repolled": True,
            "repoll_timeout_seconds": 720.0,
            "function_timeout_seconds_unchanged": 660.0,
            "maximum_gpu_seconds_unchanged": 1320.0,
            "result_publication_grace_seconds": 60.0,
            "outer_coordinator_timeout_seconds": 1620.0,
            "function_call_id_persisted_and_fsynced_before_first_poll": True,
            "typed_first_and_terminal_timeout_evidence_required": True,
        }
        or continuation.get("control_gate_fix_contract")
        != {
            "scope": "KILL_RECOMPUTE_ONLY",
            "frozen_integrated_worker_unchanged": True,
            "policy_schema": "sloforge.branchfabric.kill-recompute-control-drift-policy/v1",
            "material_persistent_drift_predicate": {
                "slope_strictly_greater_than": 0.1,
                "last_quarter_median_strictly_greater_than_first_plus": 2.0,
                "quarter_count": "max(2,n//4)",
                "both_conditions_required": True,
            },
            "unchanged_gates": {
                "exact_cadence_request_ids_timestamps_and_outputs": True,
                "eventual_accounting": True,
                "minimum_completion_fraction": 0.9,
                "maximum_p95_ttft_seconds": 2.0,
                "maximum_waiting_requests_exclusive": 20,
                "maximum_total_outstanding_requests_exclusive": 20,
            },
            "attempt_f_offline_promotion_permitted": False,
            "fresh_attempt_g_required": True,
            "function_call_config_digest_uses_strict_validated_model": True,
        }
        or continuation.get("required_prelaunch_policy_review")
        != {
            "artifact": CONTROL_GATE_POLICY_REVIEW_REFERENCE,
            "required_schema": (
                "sloforge.branchfabric.experiment-004-v11-attempt-f-"
                "kill-control-gate-methodology-review/v1"
            ),
            "required_status": "PASS",
            "required_before": "MAKE_CHECK_AND_SEAL",
            "content_hash_must_be_bound_by_final_seal": True,
        }
        or continuation.get("restrictions")
        != [
            (
                "No Modal or GPU invocation before a fresh post-fix make check and "
                "content-addressed Attempt-G seal pass independent review."
            ),
            (
                "No change to frozen integrated K, allocator, movement, serving, trigger, "
                "discard, recompute, correctness, Function 660-second bound, or 1320 "
                "GPU-second reservation gates."
            ),
            (
                "Attempt-G must use the same persisted FunctionCall ID across both bounded "
                "polls and must fail closed after the second timeout."
            ),
            (
                "Attempt-F remains scientifically invalid and must not be promoted by offline "
                "reinterpretation of its partial diagnostic bundle."
            ),
            "No invocation after Attempt G under this authorization.",
        ]
    ):
        raise RuntimeError("kill/recompute Attempt-G continuation evidence is inconsistent")


def _load_kill_config(
    config_path: Path,
    *,
    repository_root: Path,
    ledger_path: Path,
    seal_verifier: Any = None,
) -> tuple[Any, dict[str, Any]]:
    del seal_verifier
    preimport_verified = _validate_preimport_closure(repository_root)
    from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
        Experiment004V11KillRecomputeConfig,
    )

    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("kill/recompute config must be a regular file")
    resolved_root = repository_root.resolve(strict=True)
    resolved = config_path.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError("kill/recompute config must remain inside the repository")
    config = Experiment004V11KillRecomputeConfig.model_validate_json(
        resolved.read_text(), strict=True
    )
    _require_function_call_evidence_fresh(resolved_root)
    ledger_payload = json.loads(ledger_path.resolve(strict=True).read_text())
    remaining_gpu_seconds = (
        ledger_payload.get("hard_additional_gpu_seconds", 0.0)
        - ledger_payload.get("consumed_additional_gpu_seconds", 0.0)
        if isinstance(ledger_payload, dict)
        else -1.0
    )
    attempt_history_valid = _kill_attempt_history_is_valid(ledger_payload)
    if (
        config.attempt_id != ATTEMPT_ID
        or config.gpu_count != GPU_COUNT
        or config.maximum_wall_seconds != FUNCTION_WALL_SECONDS
        or _sha256(ledger_path.resolve(strict=True)) != config.ledger_sha256_before_reservation
        or config.ledger_snapshot_sha256 != config.ledger_sha256_before_reservation
        or not attempt_history_valid
        or not isinstance(remaining_gpu_seconds, float)
        or not math.isfinite(remaining_gpu_seconds)
        or remaining_gpu_seconds < 1.15 * 1320.0
    ):
        raise RuntimeError("kill/recompute config is stale or outside its sole paid envelope")
    if SEAL_PATH.is_symlink() or not SEAL_PATH.is_file():
        raise FileNotFoundError("kill/recompute content-addressed seal is absent")
    seal = json.loads(SEAL_PATH.read_text())
    config_reference = str(resolved.relative_to(resolved_root))
    source_rows = seal.get("source_bindings") if isinstance(seal, dict) else None
    make_check_path = repository_root / MAKE_CHECK_REFERENCE
    make_check_row = seal.get("make_check") if isinstance(seal, dict) else None
    if (
        not isinstance(seal, dict)
        or set(seal)
        != {
            "schema_version",
            "status",
            "attempt_id",
            "maximum_invocations",
            "maximum_wall_seconds",
            "maximum_gpu_seconds",
            "config",
            "current_ledger",
            "optimized_v11_freeze",
            "make_check",
            "prior_attempt_a",
            "prior_attempt_b",
            "prior_attempt_c",
            "prior_attempt_d",
            "prior_attempt_e",
            "prior_attempt_f",
            "continuation_authorization",
            "control_gate_policy_review_requirement",
            "source_bindings",
            "preimport_bindings",
            "freshness_contract",
        }
        or seal.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-seal/v1"
        or seal.get("status") != "PASS"
        or seal.get("attempt_id") != ATTEMPT_ID
        or seal.get("maximum_invocations") != 1
        or seal.get("maximum_wall_seconds") != 660.0
        or seal.get("maximum_gpu_seconds") != 1320.0
        or seal.get("config") != {"artifact": config_reference, "sha256": _sha256(resolved)}
        or seal.get("current_ledger")
        != {
            "artifact": str(ledger_path.resolve().relative_to(resolved_root)),
            "sha256": config.ledger_sha256_before_reservation,
            "reservations_empty": True,
        }
        or seal.get("optimized_v11_freeze")
        != {
            "artifact": config.optimized_v11_freeze_artifact,
            "sha256": config.optimized_v11_freeze_sha256,
            "tag": config.optimized_v11_freeze_tag,
            "tag_object": config.optimized_v11_freeze_tag_object,
        }
        or not isinstance(make_check_row, dict)
        or set(make_check_row) != {"artifact", "sha256", "status"}
        or make_check_row.get("artifact") != MAKE_CHECK_REFERENCE
        or re.fullmatch(r"[0-9a-f]{64}", str(make_check_row.get("sha256"))) is None
        or make_check_row.get("status") != "PASS"
        or seal.get("prior_attempt_a")
        != {
            "attempt_id": ATTEMPT_A_ID,
            "failure_classification": ("INVALID_PRE_RESERVATION_MISSING_PREFIX_PARSER_ASSUMPTION"),
            "status": {
                "artifact": PRIOR_ATTEMPT_STATUS_REFERENCE,
                "sha256": PRIOR_ATTEMPT_STATUS_SHA256,
            },
            "cli_capture": {
                "artifact": PRIOR_ATTEMPT_CAPTURE_REFERENCE,
                "sha256": PRIOR_ATTEMPT_CAPTURE_SHA256,
            },
            "ledger_unchanged": True,
            "consumed_authorized_arm": False,
        }
        or seal.get("prior_attempt_b")
        != {
            "attempt_id": ATTEMPT_B_ID,
            "failure_classification": (
                "INVALID_PAID_INFRASTRUCTURE_REMOTE_MODULE_IMPORT_ROOT_RESOLUTION"
            ),
            "status": {
                "artifact": ATTEMPT_B_STATUS_REFERENCE,
                "sha256": ATTEMPT_B_STATUS_SHA256,
            },
            "provider_cleanup": {
                "artifact": ATTEMPT_B_PROVIDER_REFERENCE,
                "sha256": ATTEMPT_B_PROVIDER_SHA256,
            },
            "conservative_charge": {
                "artifact": ATTEMPT_B_CHARGE_REFERENCE,
                "sha256": ATTEMPT_B_CHARGE_SHA256,
                "charged_gpu_seconds": 1320.0,
            },
            "scientific_arm_executed": False,
        }
        or seal.get("prior_attempt_c")
        != {
            "attempt_id": ATTEMPT_C_ID,
            "failure_classification": ("INVALID_KILL_BRANCH_HISTORY_LIVE_KV_BOUNDARY_PRE_DISCARD"),
            "status": {
                "artifact": ATTEMPT_C_STATUS_REFERENCE,
                "sha256": ATTEMPT_C_STATUS_SHA256,
            },
            "provider_cleanup": {
                "artifact": ATTEMPT_C_PROVIDER_REFERENCE,
                "sha256": ATTEMPT_C_PROVIDER_SHA256,
            },
            "provider_inventory_capture": {
                "artifact": ATTEMPT_C_PROVIDER_INVENTORY_REFERENCE,
                "sha256": ATTEMPT_C_PROVIDER_INVENTORY_SHA256,
            },
            "remote_manifest": {
                "artifact": ATTEMPT_C_MANIFEST_REFERENCE,
                "sha256": ATTEMPT_C_MANIFEST_SHA256,
                "verified_artifact_count": 478,
            },
            "conservative_charge": {
                "artifact": ATTEMPT_C_CHARGE_REFERENCE,
                "sha256": ATTEMPT_C_CHARGE_SHA256,
                "charged_gpu_seconds": 1320.0,
            },
            "forensic_review": {
                "artifact": ATTEMPT_C_REVIEW_REFERENCE,
                "sha256": ATTEMPT_C_REVIEW_SHA256,
                "classification": "KILL_HISTORY_GATE_FIELD_CONFLATION_BUG",
            },
            "destructive_discard_reached": False,
            "valid_measurement_materialized": False,
        }
        or seal.get("prior_attempt_d")
        != {
            "attempt_id": ATTEMPT_D_ID,
            "failure_classification": (
                "INVALID_PRE_RESERVATION_HISTORICAL_C_REMOTE_PREFIX_POLICY_BUG"
            ),
            "status": {
                "artifact": ATTEMPT_D_STATUS_REFERENCE,
                "sha256": ATTEMPT_D_STATUS_SHA256,
            },
            "terminal_capture": {
                "artifact": ATTEMPT_D_CAPTURE_REFERENCE,
                "sha256": ATTEMPT_D_CAPTURE_SHA256,
            },
            "ledger_unchanged": True,
            "consumed_authorized_arm": False,
        }
        or seal.get("prior_attempt_e")
        != {
            "attempt_id": ATTEMPT_E_ID,
            "failure_classification": (
                "INVALID_CLIENT_FUNCTION_CALL_POLL_TIMEOUT_CANCELED_REMOTE_STARTUP"
            ),
            "status": {
                "artifact": ATTEMPT_E_STATUS_REFERENCE,
                "sha256": ATTEMPT_E_STATUS_SHA256,
            },
            "provider_cleanup": {
                "artifact": ATTEMPT_E_PROVIDER_REFERENCE,
                "sha256": ATTEMPT_E_PROVIDER_SHA256,
            },
            "provider_inventory_capture": {
                "artifact": ATTEMPT_E_PROVIDER_INVENTORY_REFERENCE,
                "sha256": ATTEMPT_E_PROVIDER_INVENTORY_SHA256,
            },
            "remote_manifest": {
                "artifact": ATTEMPT_E_MANIFEST_REFERENCE,
                "sha256": ATTEMPT_E_MANIFEST_SHA256,
                "verified_artifact_count": 16,
            },
            "conservative_charge": {
                "artifact": ATTEMPT_E_CHARGE_REFERENCE,
                "sha256": ATTEMPT_E_CHARGE_SHA256,
                "charged_gpu_seconds": 1320.0,
            },
            "function_call_reconciliation": {
                "artifact": ATTEMPT_E_RECONCILIATION_REFERENCE,
                "sha256": ATTEMPT_E_RECONCILIATION_SHA256,
                "function_call_id": "fc-01M152JAGCBMQSBQ1TD55Y9HXV",
                "reconciliation_result": "REMOTE_ERROR_EMPTY_MESSAGE",
            },
            "remote_function_entered": True,
            "scientific_arm_executed": False,
            "destructive_discard_reached": False,
            "valid_measurement_materialized": False,
        }
        or seal.get("prior_attempt_f")
        != {
            "attempt_id": ATTEMPT_F_ID,
            "failure_classification": "INVALID_SCIENTIFIC_CONTROL_STABILITY_GATE",
            "status": {
                "artifact": ATTEMPT_F_STATUS_REFERENCE,
                "sha256": ATTEMPT_F_STATUS_SHA256,
            },
            "provider_cleanup": {
                "artifact": ATTEMPT_F_PROVIDER_REFERENCE,
                "sha256": ATTEMPT_F_PROVIDER_SHA256,
            },
            "provider_inventory_capture": {
                "artifact": ATTEMPT_F_PROVIDER_INVENTORY_REFERENCE,
                "sha256": ATTEMPT_F_PROVIDER_INVENTORY_SHA256,
            },
            "remote_manifest": {
                "artifact": ATTEMPT_F_MANIFEST_REFERENCE,
                "sha256": ATTEMPT_F_MANIFEST_SHA256,
                "verified_artifact_count": 508,
            },
            "conservative_charge": {
                "artifact": ATTEMPT_F_CHARGE_REFERENCE,
                "sha256": ATTEMPT_F_CHARGE_SHA256,
                "charged_gpu_seconds": 1320.0,
            },
            "strict_nine_rps_control_interval_passed": False,
            "partial_diagnostic_recompute_evidence_materialized": True,
            "valid_measurement_materialized": False,
        }
        or seal.get("continuation_authorization")
        != {
            "artifact": CONTINUATION_AUTHORIZATION_REFERENCE,
            "sha256": CONTINUATION_AUTHORIZATION_SHA256,
            "authorized_attempt_id": ATTEMPT_ID,
            "maximum_fresh_invocations": 1,
        }
        or seal.get("control_gate_policy_review_requirement")
        != {
            "artifact": CONTROL_GATE_POLICY_REVIEW_REFERENCE,
            "required_schema": (
                "sloforge.branchfabric.experiment-004-v11-attempt-f-"
                "kill-control-gate-methodology-review/v1"
            ),
            "required_status": "PASS",
            "required_before": "MAKE_CHECK_AND_SEAL",
            "content_hash_must_be_bound_by_final_seal": True,
        }
        or seal.get("freshness_contract")
        != {
            "local_final_absent": True,
            "local_function_call_evidence_absent": True,
            "remote_final_staging_inflight_absent_before_reservation": True,
            "maximum_active_reservations": 1,
            "exactly_once_spawn": True,
            "minimum_remaining_gpu_seconds_with_15_percent_margin": 1518.0,
            "actual_remaining_gpu_seconds": remaining_gpu_seconds,
        }
        or not isinstance(source_rows, dict)
        or set(source_rows) != set(SOURCE_BINDINGS)
        or seal.get("preimport_bindings") != preimport_verified
    ):
        raise RuntimeError("kill/recompute seal differs from its exact contract")
    if (
        make_check_path.is_symlink()
        or not make_check_path.is_file()
        or _sha256(make_check_path) != make_check_row["sha256"]
    ):
        raise RuntimeError("kill/recompute fresh make-check evidence changed")
    make_check = json.loads(make_check_path.read_text())
    if (
        not isinstance(make_check, dict)
        or set(make_check)
        != {
            "schema_version",
            "status",
            "attempt_id",
            "command",
            "gpu_or_cloud_invoked",
            "completed_at_utc",
            "exit_code",
            "python",
            "rust",
            "ui",
            "provenance",
        }
        or make_check.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-kill-recompute-make-check/v1"
        or make_check.get("status") != "PASS"
        or make_check.get("attempt_id") != ATTEMPT_ID
        or make_check.get("command") != "make check"
        or make_check.get("gpu_or_cloud_invoked") is not False
        or make_check.get("exit_code") != 0
        or not isinstance(make_check.get("completed_at_utc"), str)
        or not make_check["completed_at_utc"]
        or not isinstance(make_check.get("python"), dict)
        or set(make_check["python"]) != {"ruff_format_check", "ruff_check", "mypy", "pytest"}
        or re.fullmatch(r"PASS; [1-9][0-9]* files", make_check["python"]["ruff_format_check"])
        is None
        or make_check["python"]["ruff_check"] != "PASS"
        or re.fullmatch(r"PASS; [1-9][0-9]* source files", make_check["python"]["mypy"]) is None
        or re.fullmatch(
            r"PASS; [1-9][0-9]* passed, [0-9]+ skipped, [0-9]+ warnings",
            make_check["python"]["pytest"],
        )
        is None
        or make_check.get("rust")
        != {
            "cargo_fmt": "PASS",
            "cargo_clippy_all_targets_all_features": "PASS",
            "cargo_test_workspace_all_features": "PASS; unit, integration, and doc tests",
        }
        or make_check.get("ui")
        != {
            "typecheck": "PASS",
            "lint": "PASS",
            "tests": "PASS; 37 passed, 1 skipped",
            "build": "PASS",
        }
        or not isinstance(make_check.get("provenance"), str)
        or not make_check["provenance"]
    ):
        raise RuntimeError("kill/recompute make-check evidence is not PASS-complete")
    prior_status_path = repository_root / PRIOR_ATTEMPT_STATUS_REFERENCE
    prior_capture_path = repository_root / PRIOR_ATTEMPT_CAPTURE_REFERENCE
    if any(
        path.is_symlink() or not path.is_file() for path in (prior_status_path, prior_capture_path)
    ) or (
        _sha256(prior_status_path) != PRIOR_ATTEMPT_STATUS_SHA256
        or _sha256(prior_capture_path) != PRIOR_ATTEMPT_CAPTURE_SHA256
    ):
        raise RuntimeError("kill/recompute Attempt-A pre-reservation evidence changed")
    prior_status = json.loads(prior_status_path.read_text())
    prior_capture = json.loads(prior_capture_path.read_text())
    expected_command = [
        "modal",
        "volume",
        "ls",
        RESULTS_VOLUME,
        REMOTE_PREFIX,
        "--json",
    ]
    if (
        prior_status.get("status") != "INVALID"
        or prior_status.get("scientifically_valid") is not False
        or prior_status.get("attempt_id") != ATTEMPT_A_ID
        or prior_status.get("failure_classification")
        != "INVALID_PRE_RESERVATION_MISSING_PREFIX_PARSER_ASSUMPTION"
        or prior_status.get("failure_stage") != "REMOTE_PREFIX_FRESHNESS_BEFORE_PREBUILD"
        or any(
            prior_status.get(field) is not False
            for field in (
                "scientific_work_started",
                "prebuild_started",
                "reservation_created",
                "function_invoked",
                "gpu_invoked",
                "controller_started",
                "source_state_discarded",
                "recompute_started",
            )
        )
        or prior_capture.get("status") != "FAIL_CLOSED"
        or prior_capture.get("attempt_id") != ATTEMPT_A_ID
        or prior_capture.get("failure_stage") != "REMOTE_PREFIX_FRESHNESS_BEFORE_PREBUILD"
        or prior_capture.get("command") != expected_command
        or prior_capture.get("returncode") != 1
        or prior_capture.get("stdout") != ""
        or prior_capture.get("stderr") != "No such file or directory"
        or prior_capture.get("failure_classification")
        != "INVALID_PRE_RESERVATION_MISSING_PREFIX_PARSER_ASSUMPTION"
        or any(
            prior_capture.get(field) is not False
            for field in (
                "prebuild_started",
                "reservation_created",
                "gpu_or_function_invoked",
                "provider_resource_created",
            )
        )
        or prior_capture.get("ledger_sha256_before") != ATTEMPT_A_LEDGER_SHA256
        or prior_capture.get("ledger_sha256_after") != ATTEMPT_A_LEDGER_SHA256
    ):
        raise RuntimeError("kill/recompute Attempt-A pre-reservation evidence is inconsistent")
    attempt_b_paths = {
        "config": repository_root
        / (
            "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
            "kill-recompute/exp004-v11-kill-recompute-s41-b-config.json"
        ),
        "seal": repository_root
        / (
            "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
            "kill-recompute/authorization/seal-verification-attempt-b.json"
        ),
        "make_check": repository_root
        / (
            "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/"
            "kill-recompute/authorization/make-check-attempt-b.json"
        ),
        "prebuild": repository_root
        / (
            "artifacts/branchfabric/gpu-validation/experiment-004/"
            "v11-kill-recompute-image-prebuild/exp004-v11-kill-recompute-s41-b.json"
        ),
        "post_freshness": repository_root
        / (
            "artifacts/branchfabric/gpu-validation/experiment-004/"
            "v11-kill-recompute-image-prebuild/"
            "exp004-v11-kill-recompute-s41-b-pre-reservation-freshness.json"
        ),
        "status": repository_root / ATTEMPT_B_STATUS_REFERENCE,
        "provider": repository_root / ATTEMPT_B_PROVIDER_REFERENCE,
        "charge": repository_root / ATTEMPT_B_CHARGE_REFERENCE,
        "continuation": repository_root / CONTINUATION_AUTHORIZATION_REFERENCE,
    }
    expected_attempt_b_hashes = {
        "config": "a75e054d5c06026abdaa26ad296e53eb4e47e527d15f8f1eed4607ad30100b56",
        "seal": "6467467203d35e24207fea849f38de4af4bd2634d122b4103622326a0ae831fd",
        "make_check": "4951bdf257f9fa2ef2532cfddaf1f7290dbcd20b7adde842ebf9869a0f01de12",
        "prebuild": "936025a11299cec392c92e04a92c2d8097f98df694d55695567f272909771a47",
        "post_freshness": ("eb25b0cf262cbbf60462e21dfc914b31f580acce6bb3b9800fe8970b1ef70c90"),
        "status": ATTEMPT_B_STATUS_SHA256,
        "provider": ATTEMPT_B_PROVIDER_SHA256,
        "charge": ATTEMPT_B_CHARGE_SHA256,
        "continuation": CONTINUATION_AUTHORIZATION_SHA256,
    }
    if any(
        path.is_symlink() or not path.is_file() or _sha256(path) != expected_attempt_b_hashes[label]
        for label, path in attempt_b_paths.items()
    ):
        raise RuntimeError("kill/recompute Attempt-B/continuation evidence changed")
    attempt_b_status = json.loads(attempt_b_paths["status"].read_text())
    attempt_b_provider = json.loads(attempt_b_paths["provider"].read_text())
    attempt_b_charge = json.loads(attempt_b_paths["charge"].read_text())
    attempt_b_false_fields = (
        "scientific_work_started",
        "function_entry_reached",
        "handler_started",
        "controller_started",
        "model_loaded",
        "source_state_discarded",
        "recompute_started",
        "kill_recompute_measurement_available",
        "raw_attempt_prefix_present",
    )
    provider_apps = attempt_b_provider.get("apps")
    if (
        attempt_b_status.get("status") != "INVALID"
        or attempt_b_status.get("scientifically_valid") is not False
        or attempt_b_status.get("attempt_id") != ATTEMPT_B_ID
        or attempt_b_status.get("failure_classification")
        != "INVALID_PAID_INFRASTRUCTURE_REMOTE_MODULE_IMPORT_ROOT_RESOLUTION"
        or attempt_b_status.get("failure_stage") != "REMOTE_MODULE_IMPORT_BEFORE_FUNCTION_ENTRY"
        or any(attempt_b_status.get(field) is not False for field in attempt_b_false_fields)
        or attempt_b_status.get("paid_function_invocation_submitted") is not True
        or attempt_b_status.get("gpu_allocation_status") != "UNKNOWN_PROVIDER_MAY_HAVE_ALLOCATED"
        or attempt_b_status.get("cleanup")
        != {
            "in_function": "UNAVAILABLE_MODULE_IMPORT_FAILED_BEFORE_HANDLER",
            "provider": "PASS",
            "provider_resources_remaining": 0,
            "authorized_volumes_only": True,
            "ledger_reservation_released": True,
        }
        or attempt_b_status.get("reservation_id") != "exp004-v11-kill-b-b2ed62946125171fbd29"
        or attempt_b_status.get("canonical_config_sha256")
        != "afdd02aa5783e1c1387460cf7c594bdb22a0c6bbbbf8fee792902411e124e467"
        or attempt_b_status.get("gpu_accounting")
        != {
            "policy": "CONSERVATIVE_FULL_AUTHORIZED_BOUND_NO_FABRICATED_ACTUAL",
            "measured_gpu_seconds": None,
            "charged_gpu_seconds": 1320.0,
            "charged_wall_seconds": 660.0,
        }
        or attempt_b_status.get("evidence_bindings", {}).get("provider_cleanup")
        != {
            "artifact": ATTEMPT_B_PROVIDER_REFERENCE,
            "sha256": ATTEMPT_B_PROVIDER_SHA256,
        }
        or attempt_b_status.get("evidence_bindings", {}).get("conservative_charge")
        != {
            "artifact": ATTEMPT_B_CHARGE_REFERENCE,
            "sha256": ATTEMPT_B_CHARGE_SHA256,
        }
        or attempt_b_status.get("evidence_bindings", {}).get("settled_ledger")
        != {
            "artifact": LEDGER_REFERENCE,
            "sha256": "7ebbe3e61f13124ca04d2c288ac7665961807d5d62a116530f5334155f5abc30",
        }
        or attempt_b_provider.get("status") != "PASS"
        or attempt_b_provider.get("attempt_id") != ATTEMPT_B_ID
        or attempt_b_provider.get("failure_boundary")
        != "REMOTE_MODULE_IMPORT_BEFORE_FUNCTION_ENTRY"
        or attempt_b_provider.get("function_entry_reached") is not False
        or attempt_b_provider.get("controller_started") is not False
        or attempt_b_provider.get("in_function_cleanup_status")
        != "UNAVAILABLE_MODULE_IMPORT_FAILED_BEFORE_HANDLER"
        or attempt_b_provider.get("provider_cleanup_status") != "PASS"
        or provider_apps
        != [
            {
                "app_id": "ap-5NiGIGi2bFaFVNpFIXaBhy",
                "description": APP_NAME,
                "created_at": "2026-08-27T22:14:01-05:00",
                "stopped_at": "2026-08-27T22:28:22-05:00",
                "state": "stopped",
                "tasks": "0",
            },
            {
                "app_id": "ap-bq9hPG9XYEKLlBJoFi1Jet",
                "description": APP_NAME,
                "created_at": "2026-08-27T22:11:29-05:00",
                "stopped_at": "2026-08-27T22:13:59-05:00",
                "state": "stopped",
                "tasks": "0",
            },
        ]
        or any(
            attempt_b_provider.get(field) != []
            for field in ("containers", "endpoints", "queues", "dicts", "reservations")
        )
        or attempt_b_provider.get("volumes")
        != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or attempt_b_provider.get("authorized_volumes_only") is not True
        or attempt_b_provider.get("raw_attempt_prefix_present") is not False
        or attempt_b_provider.get("scientific_evidence_materialized") is not False
        or attempt_b_provider.get("reservation_id") != "exp004-v11-kill-b-b2ed62946125171fbd29"
        or attempt_b_provider.get("faulty_modal_source_sha256")
        != "fa07b4fc075b3e36f7ce5802a9a5e658128a98db9ed54d222dc4edc431bdff89"
        or attempt_b_provider.get("evidence_bindings", {}).get("conservative_charge")
        != {
            "artifact": ATTEMPT_B_CHARGE_REFERENCE,
            "sha256": ATTEMPT_B_CHARGE_SHA256,
        }
        or attempt_b_provider.get("evidence_bindings", {}).get("settled_ledger")
        != {
            "artifact": LEDGER_REFERENCE,
            "sha256": "7ebbe3e61f13124ca04d2c288ac7665961807d5d62a116530f5334155f5abc30",
        }
        or attempt_b_charge.get("status") != "CONSERVATIVELY_CHARGED"
        or attempt_b_charge.get("attempt_id") != ATTEMPT_B_ID
        or attempt_b_charge.get("reservation_id") != "exp004-v11-kill-b-b2ed62946125171fbd29"
        or attempt_b_charge.get("failure_stage") != "modal-launch"
        or attempt_b_charge.get("charged_wall_seconds") != 660.0
        or attempt_b_charge.get("charged_gpu_seconds") != 1320.0
        or attempt_b_charge.get("actual_gpu_seconds") is not None
        or attempt_b_charge.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
    ):
        raise RuntimeError("kill/recompute Attempt-B evidence is inconsistent")
    _validate_attempt_c_d_e_and_f_continuation(
        repository_root=resolved_root,
        ledger_reference=LEDGER_REFERENCE,
        ledger_sha256=config.ledger_sha256_before_reservation,
        remaining_gpu_seconds=remaining_gpu_seconds,
    )
    for label, reference in SOURCE_BINDINGS.items():
        path = resolved_root / reference
        row = source_rows[label]
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(row, dict)
            or row != {"artifact": reference, "sha256": _sha256(path)}
        ):
            raise RuntimeError(f"kill/recompute sealed source changed: {label}")
    experiment_modules = resolved_root / "experiments/branchfabric"
    sys.path.insert(0, str(experiment_modules))
    try:
        from gpu_reclamation_kill_recompute_controller_v11 import (
            verify_kill_recompute_evidence,
        )

        verified = verify_kill_recompute_evidence(config, resolved_root)
    finally:
        sys.path.remove(str(experiment_modules))
    if verified.get("passed") is not True:
        raise RuntimeError("kill/recompute K/freeze evidence verification failed")
    binding_path = resolved_root / config.optimized_v11_freeze_tag_binding_artifact
    binding = json.loads(binding_path.read_text())
    tag_object = subprocess.run(
        ["git", "rev-parse", config.optimized_v11_freeze_tag],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    peeled = subprocess.run(
        ["git", "rev-parse", f"{config.optimized_v11_freeze_tag}^{{}}"],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tag_body = subprocess.run(
        ["git", "cat-file", "tag", tag_object],
        cwd=resolved_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if (
        tag_object != config.optimized_v11_freeze_tag_object
        or peeled != binding.get("peeled_commit")
        or config.optimized_v11_freeze_sha256 not in tag_body
        or config.integrated_k_status_sha256 not in tag_body
    ):
        raise RuntimeError("live optimized-v11 annotated tag differs from the freeze binding")
    return config, seal


def _invoke_kill_modal(
    config_path: Path,
    *,
    wall_seconds: float,
    reservation_id: str,
    budget_usd: float,
    runner: Any = None,
) -> tuple[subprocess.CompletedProcess[str], float]:
    base = _BASE
    if (
        wall_seconds != FUNCTION_WALL_SECONDS
        or base.FUNCTION_STARTUP_SECONDS != 180.0
        or base.COORDINATOR_PROCESS_GRACE_SECONDS != COORDINATOR_PROCESS_GRACE_SECONDS
        or KILL_MODAL_RETRIEVAL_TIMEOUT_SECONDS != 1620.0
    ):
        raise RuntimeError("kill/recompute FunctionCall retrieval envelope changed")
    runner = base._run_bounded_process_group if runner is None else runner
    ledger = base._load_ledger(base._LEDGER)
    reservations = [row for row in ledger.reservations if row.reservation_id == reservation_id]
    if len(reservations) != 1 or len(ledger.reservations) != 1:
        raise RuntimeError("kill/recompute sole reservation is absent or duplicated")
    commitment = hashlib.sha256(_BASE.canonical_json_bytes(reservations[0])).hexdigest()
    launch_token = secrets.token_hex(32)
    environment = dict(os.environ)
    environment.update(
        {
            "SLOFORGE_GPU_BUDGET_USD": format(budget_usd, ".17g"),
            "SLOFORGE_EXP004_RESERVATION_ID": reservation_id,
            "SLOFORGE_EXP004_RESERVATION_COMMITMENT_SHA256": commitment,
            "SLOFORGE_MODAL_PREFLIGHT_TOKEN": launch_token,
        }
    )
    started = time.monotonic()
    completed = runner(
        [
            "/usr/bin/caffeinate",
            "-dimsu",
            "modal",
            "run",
            str(base._APP),
            "--config-path",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=KILL_MODAL_RETRIEVAL_TIMEOUT_SECONDS,
        env=environment,
    )
    try:
        _validate_completed_function_call_evidence(
            config=json.loads(config_path.read_text()),
            reservation_id=reservation_id,
            reservation_commitment_sha256=commitment,
            completed=completed,
        )
    except Exception as error:
        # Return a typed local-validation failure as a completed process so the
        # inherited coordinator retains the original stdout/stderr and call-ID
        # capture in its conservative failure charge.  Raising here would leave
        # its ``completed`` variable unset and discard that provenance.
        validation_failure = json.dumps(
            {
                "schema_version": (
                    "sloforge.branchfabric.function-call-evidence-validation-failure/v1"
                ),
                "type": type(error).__name__,
                "message": str(error),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        completed = subprocess.CompletedProcess(
            args=getattr(completed, "args", ()),
            returncode=completed.returncode if completed.returncode != 0 else 86,
            stdout=completed.stdout,
            stderr=(
                completed.stderr
                + ("" if completed.stderr.endswith("\n") or not completed.stderr else "\n")
                + validation_failure
                + "\n"
            ),
        )
    return completed, time.monotonic() - started


def _read_existing_remote_directory(
    prefix: str,
    *,
    runner: Any,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Read one known-existing directory with an exact Modal 1.5.3 schema."""

    command = ["modal", "volume", "ls", RESULTS_VOLUME, prefix, "--json"]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Modal results-volume existing-parent inventory failed: "
            f"prefix={prefix!r}, returncode={completed.returncode}, "
            f"stderr={completed.stderr!r}"
        )
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("Modal results-volume existing-parent inventory is not JSON") from error
    expected_fields = {"filename", "type", "created_modified", "size"}
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or set(row) != expected_fields
        or any(not isinstance(row[field], str) or not row[field] for field in expected_fields)
        or row["type"] not in {"dir", "file"}
        for row in rows
    ):
        raise ValueError("Modal results-volume existing-parent inventory schema is invalid")
    filenames = [row["filename"] for row in rows]
    canonical_prefix = prefix.strip("/")
    if prefix != canonical_prefix or any(
        filename != filename.strip("/")
        or not filename.startswith(f"{canonical_prefix}/")
        or "/" in filename.removeprefix(f"{canonical_prefix}/")
        or "\\" in filename
        or "\x00" in filename
        or any(part in {"", ".", ".."} for part in filename.split("/"))
        for filename in filenames
    ):
        raise ValueError("Modal results-volume existing-parent inventory paths are invalid")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Modal results-volume existing-parent inventory paths are duplicated")
    capture = {
        "listed_prefix": prefix,
        "command": command,
        "returncode": completed.returncode,
        "stdout_rows": rows,
        "stderr": completed.stderr,
    }
    return rows, capture


def _require_hierarchical_remote_attempt_prefix_fresh(
    config: Any,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Descend only through observed directories before checking the leaf.

    Modal 1.5.3 returns rc=1 for a missing leaf.  Absence is therefore proven
    from the nearest known-existing parent instead of interpreting a failed
    leaf listing as an empty directory.  Attempts C, E, and F are exact isolated
    historical namespaces: the current run consumes only their locally
    rehashed manifest closures, never their persistent descendants.
    """

    if getattr(config, "attempt_id", None) != ATTEMPT_ID:
        raise ValueError("Modal results-volume freshness attempt identity changed")
    historical_c_prefix = f"{REMOTE_PREFIX}/{ATTEMPT_C_ID}"
    historical_e_prefix = f"{REMOTE_PREFIX}/{ATTEMPT_E_ID}"
    historical_f_prefix = f"{REMOTE_PREFIX}/{ATTEMPT_F_ID}"
    attempt_prefixes = tuple(
        f"{REMOTE_PREFIX}/{attempt_id}"
        for attempt_id in (ATTEMPT_A_ID, ATTEMPT_B_ID, ATTEMPT_D_ID, config.attempt_id)
    )
    forbidden_paths = (
        *(
            path
            for attempt_prefix in attempt_prefixes
            for path in (
                attempt_prefix,
                f"{attempt_prefix}.staging",
                f"{attempt_prefix}.inflight",
            )
        ),
        f"{historical_c_prefix}.staging",
        f"{historical_c_prefix}.inflight",
        f"{historical_e_prefix}.staging",
        f"{historical_e_prefix}.inflight",
        f"{historical_f_prefix}.staging",
        f"{historical_f_prefix}.inflight",
    )
    steps: list[dict[str, Any]] = []
    current = REMOTE_INVENTORY_ROOT
    for expected_child in ("experiment-004/v11/kill-recompute", REMOTE_PREFIX):
        rows, capture = _read_existing_remote_directory(current, runner=runner)
        steps.append(capture)
        matching = [row for row in rows if row["filename"] == expected_child]
        if not matching:
            raise FileNotFoundError(
                "Modal required historical kill/recompute hierarchy is absent: " + expected_child
            )
        if matching[0]["type"] != "dir":
            raise ValueError(
                f"Modal results-volume hierarchy component is not a directory: {expected_child}"
            )
        current = expected_child

    leaf_rows, capture = _read_existing_remote_directory(REMOTE_PREFIX, runner=runner)
    steps.append(capture)
    historical_c_rows = [row for row in leaf_rows if row["filename"] == historical_c_prefix]
    historical_e_rows = [row for row in leaf_rows if row["filename"] == historical_e_prefix]
    historical_f_rows = [row for row in leaf_rows if row["filename"] == historical_f_prefix]
    observed = {row["filename"] for row in leaf_rows}
    collisions = sorted(
        path
        for path in observed
        if any(
            path == forbidden or path.startswith(f"{forbidden}/") for forbidden in forbidden_paths
        )
    )
    if collisions:
        raise FileExistsError(
            "fresh kill/recompute attempt remote prefix exists before reservation: "
            + ", ".join(collisions)
        )
    if len(historical_c_rows) != 1:
        raise FileNotFoundError(
            "Modal required historical Attempt-C prefix is absent: " + historical_c_prefix
        )
    if historical_c_rows[0]["type"] != "dir":
        raise ValueError("Modal historical Attempt-C prefix is not a directory")
    if len(historical_e_rows) != 1:
        raise FileNotFoundError(
            "Modal required historical Attempt-E prefix is absent: " + historical_e_prefix
        )
    if historical_e_rows[0]["type"] != "dir":
        raise ValueError("Modal historical Attempt-E prefix is not a directory")
    if len(historical_f_rows) != 1:
        raise FileNotFoundError(
            "Modal required historical Attempt-F prefix is absent: " + historical_f_prefix
        )
    if historical_f_rows[0]["type"] != "dir":
        raise ValueError("Modal historical Attempt-F prefix is not a directory")
    return {
        "schema_version": "sloforge.branchfabric.remote-prefix-freshness/v5",
        "status": "PASS",
        "volume_name": RESULTS_VOLUME,
        "inventory_root": REMOTE_INVENTORY_ROOT,
        "listed_prefix": REMOTE_PREFIX,
        "attempt_id": config.attempt_id,
        "forbidden_paths": list(forbidden_paths),
        "allowed_historical_prefixes": [
            {
                "attempt_id": ATTEMPT_C_ID,
                "path": historical_c_prefix,
                "type": "dir",
                "manifest": {
                    "artifact": ATTEMPT_C_MANIFEST_REFERENCE,
                    "sha256": ATTEMPT_C_MANIFEST_SHA256,
                    "verified_artifact_count": 478,
                },
            },
            {
                "attempt_id": ATTEMPT_E_ID,
                "path": historical_e_prefix,
                "type": "dir",
                "manifest": {
                    "artifact": ATTEMPT_E_MANIFEST_REFERENCE,
                    "sha256": ATTEMPT_E_MANIFEST_SHA256,
                    "verified_artifact_count": 16,
                },
            },
            {
                "attempt_id": ATTEMPT_F_ID,
                "path": historical_f_prefix,
                "type": "dir",
                "manifest": {
                    "artifact": ATTEMPT_F_MANIFEST_REFERENCE,
                    "sha256": ATTEMPT_F_MANIFEST_SHA256,
                    "verified_artifact_count": 508,
                },
            },
        ],
        "historical_contents_used_for_current_attempt": False,
        "expected_present_paths": [
            historical_c_prefix,
            historical_e_prefix,
            historical_f_prefix,
        ],
        "observed_expected_paths": [
            historical_c_prefix,
            historical_e_prefix,
            historical_f_prefix,
        ],
        "missing_expected_count": 0,
        "inventory_steps": steps,
        "observed_entries": steps,
        "collision_count": 0,
    }


def _cleanup_is_pass_complete(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _BASE._IN_FUNCTION_CLEANUP_FIELDS:
        return False
    lifecycle = value.get("lifecycle")
    empty_fields = (
        "leaked_threads",
        "leaked_ipc_resources",
        "forced_kills",
        "surviving_children",
        "surviving_process_groups",
        "profiler_processes_after",
        "serving_workers_after",
        "rollout_workers_after",
        "resource_tracker_processes_after",
        "zombie_processes_after",
        "compute_processes_after",
        "cleanup_errors",
    )
    affirmative = (
        "parent_reaped_all_owned_children",
        "owned_ipc_resources_released",
        "pipes_closed",
        "cuda_released",
    )
    return bool(
        value.get("schema_version") == "sloforge.branchfabric.in-function-cleanup/v1"
        and value.get("status") == "PASS"
        and value.get("pass") is True
        and value.get("forced_kill_required") is False
        and value.get("required_lifecycle") == list(_BASE._PROCESS_LIFECYCLE_PHASES)
        and isinstance(lifecycle, list)
        and all(
            isinstance(row, dict)
            and set(row) == {"phase", "observed_at_monotonic_ns", "observed_at_utc"}
            and type(row.get("observed_at_monotonic_ns")) is int
            and row["observed_at_monotonic_ns"] > 0
            and isinstance(row.get("observed_at_utc"), str)
            and bool(row["observed_at_utc"])
            and row.get("phase") == phase
            for row, phase in zip(lifecycle, _BASE._PROCESS_LIFECYCLE_PHASES, strict=True)
        )
        and all(
            current["observed_at_monotonic_ns"] < following["observed_at_monotonic_ns"]
            for current, following in pairwise(lifecycle)
        )
        and type(value.get("recorded_at_monotonic_ns")) is int
        and value["recorded_at_monotonic_ns"] >= lifecycle[-1]["observed_at_monotonic_ns"]
        and isinstance(value.get("recorded_at_utc"), str)
        and bool(value["recorded_at_utc"])
        and type(value.get("parent_pid")) is int
        and value["parent_pid"] > 0
        and type(value.get("parent_pgid")) is int
        and value["parent_pgid"] > 0
        and type(value.get("parent_sid")) is int
        and value["parent_sid"] > 0
        and isinstance(value.get("initial_threads"), list)
        and value["initial_threads"] == value.get("final_threads")
        and isinstance(value.get("child_subreaper"), dict)
        and value["child_subreaper"].get("restored_before_return") is True
        and value["child_subreaper"].get("error") is None
        and all(value.get(field) == [] for field in empty_fields)
        and all(value.get(field) is True for field in affirmative)
    )


def _post_worker_cleanup_ownership_is_exact(
    controller: Mapping[str, Any], cleanup: Mapping[str, Any]
) -> bool:
    """Bind cleanup rows and signals to the two controller-owned workers.

    Zero-survivor evidence is insufficient on its own: it is only meaningful
    when the cleanup record proves that the serving and rollout process trees
    it inspected are the exact trees the controller launched.
    """

    roles = {"serving", "rollout"}
    worker_pids = controller.get("worker_pids")
    worker_groups = controller.get("worker_process_groups")
    worker_sessions = controller.get("worker_session_ids")
    owned_children = cleanup.get("owned_children")
    actions = cleanup.get("termination_actions")
    forced_kills = cleanup.get("forced_kills")
    if not (
        isinstance(worker_pids, dict)
        and set(worker_pids) == roles
        and isinstance(worker_groups, dict)
        and set(worker_groups) == roles
        and isinstance(worker_sessions, dict)
        and set(worker_sessions) == roles
        and isinstance(owned_children, list)
        and bool(owned_children)
        and all(isinstance(row, dict) for row in owned_children)
        and isinstance(actions, list)
        and all(isinstance(row, dict) for row in actions)
        and isinstance(forced_kills, list)
    ):
        return False

    required_child_fields = {
        "pid",
        "ppid",
        "pgid",
        "sid",
        "role",
        "process_kind",
        "termination_signal",
        "exit_status",
        "reap_timestamp_monotonic_ns",
        "reap_method",
    }
    child_pids: set[int] = set()
    children_by_pid: dict[int, Mapping[str, Any]] = {}
    for row in owned_children:
        pid = row.get("pid")
        if not (
            required_child_fields <= set(row)
            and type(pid) is int
            and pid > 0
            and pid not in child_pids
            and type(row.get("ppid")) is int
            and row["ppid"] > 0
            and type(row.get("pgid")) is int
            and row["pgid"] > 0
            and type(row.get("sid")) is int
            and row["sid"] > 0
            and row.get("role") in roles
            and row.get("process_kind") in {"worker", "helper"}
            and row.get("termination_signal") in {None, "SIGTERM", "SIGKILL"}
            and row.get("reap_method")
        ):
            return False
        child_pids.add(pid)
        children_by_pid[pid] = row

    for role in roles:
        pid = worker_pids.get(role)
        group = worker_groups.get(role)
        session = worker_sessions.get(role)
        worker = children_by_pid.get(pid) if type(pid) is int else None
        if not (
            type(pid) is int
            and pid > 0
            and type(group) is int
            and group > 0
            and type(session) is int
            and session > 0
            and isinstance(worker, Mapping)
            and worker.get("pgid") == group
            and worker.get("sid") == session
            and worker.get("role") == role
            and worker.get("process_kind") == "worker"
            and worker.get("exit_status") == controller["worker_returncodes"][role]
        ):
            return False
        if any(
            row.get("pgid") != group or row.get("sid") != session
            for row in owned_children
            if row.get("role") == role
        ):
            return False

    action_base_fields = {"pid", "process_group", "signal", "forced", "target", "at_utc"}
    for action in actions:
        pid = action.get("pid")
        group = action.get("process_group")
        target = action.get("target")
        if not (
            (set(action) == action_base_fields or set(action) == action_base_fields | {"role"})
            and type(pid) is int
            and pid > 0
            and type(group) is int
            and group > 0
            and action.get("signal") in {"SIGTERM", "SIGKILL"}
            and action.get("forced") is (action.get("signal") == "SIGKILL")
            and target in {"worker-process-group", "exact-owned-pid"}
            and isinstance(action.get("at_utc"), str)
            and bool(action["at_utc"])
        ):
            return False
        if target == "worker-process-group":
            if pid != group or group not in set(worker_groups.values()):
                return False
        else:
            child = children_by_pid.get(pid)
            if not (
                isinstance(child, Mapping)
                and child.get("pgid") == group
                and ("role" not in action or action.get("role") == child.get("role"))
            ):
                return False

    if forced_kills != [row for row in actions if row.get("signal") == "SIGKILL"]:
        return False
    if cleanup.get("forced_kill_required") is not bool(forced_kills):
        return False

    for child in owned_children:
        matching_actions = [
            action
            for action in actions
            if action.get("pid") == child.get("pid")
            or action.get("process_group") == child.get("pgid")
        ]
        expected_signal = matching_actions[-1]["signal"] if matching_actions else None
        if child.get("termination_signal") != expected_signal:
            return False

    return bool(
        cleanup.get("parent_pid") == controller.get("controller_pid")
        and controller.get("cleanup_scope") is None
        and controller.get("cleanup_actions") == cleanup.get("termination_actions")
        and isinstance(controller.get("compute_processes_after"), (list, tuple))
        and list(controller["compute_processes_after"]) == cleanup.get("compute_processes_after")
        and isinstance(cleanup.get("initial_ipc_resources"), list)
        and cleanup.get("initial_ipc_resources") == cleanup.get("final_ipc_resources")
    )


def _post_worker_engine_evidence_is_exact(controller: Mapping[str, Any]) -> bool:
    roles = {"serving", "rollout"}
    role_devices = {"serving": "gpu0", "rollout": "gpu1"}
    role_indices = {"serving": 0, "rollout": 1}
    inventory = controller.get("inventory_before")
    worker_pids = controller.get("worker_pids")
    engine_rows = controller.get("engine_start_evidence")
    readiness_rows = controller.get("readiness_evidence")
    if not (
        isinstance(inventory, list)
        and len(inventory) == 2
        and all(isinstance(row, dict) for row in inventory)
        and isinstance(worker_pids, dict)
        and set(worker_pids) == roles
        and isinstance(engine_rows, list)
        and len(engine_rows) == 2
        and all(isinstance(row, dict) for row in engine_rows)
        and isinstance(readiness_rows, list)
        and len(readiness_rows) == 2
        and all(isinstance(row, dict) for row in readiness_rows)
    ):
        return False
    inventory_by_index = {row.get("index"): row for row in inventory}
    engine_by_role = {row.get("role"): row for row in engine_rows}
    readiness_by_role = {row.get("role"): row for row in readiness_rows}
    readiness_fields = {
        "adapter_object_identity",
        "compilation_observation",
        "compile_warmup_ended_ns",
        "compile_warmup_started_ns",
        "cudagraph_coverage_pass",
        "device",
        "engine_nonce",
        "engine_object_identity",
        "engine_reloaded",
        "engine_started_ns",
        "hbm",
        "model_ready_ns",
        "observed_cudagraph_capture_sizes",
        "passed",
        "physical_gpu_uuid",
        "pid",
        "probe",
        "queue_empty_pass",
        "required_cudagraph_batch_sizes",
        "role",
        "rollouts_ready",
        "runtime_evidence",
        "runtime_state",
        "schema_version",
        "verification_batches",
        "verification_output_target_verified",
        "verification_request_count",
        "warmup_batches",
        "warmup_output_target_verified",
        "warmup_request_count",
    }
    runtime_fields = {
        "health",
        "health_source",
        "metrics_available",
        "metrics_source",
        "prefix_cache_reset",
        "queue_depth",
        "request_count",
        "running_requests",
        "skipped_waiting_requests",
        "waiting_requests",
    }
    compilation_fields = {
        "engine_initialization_completed_before_warmup",
        "fail_closed_limitation",
        "measured_probe_compilation_events",
        "no_active_compilation_event",
        "source",
        "warmup_compilation_events",
    }
    if (
        set(inventory_by_index) != {0, 1}
        or set(engine_by_role) != roles
        or set(readiness_by_role) != roles
    ):
        return False
    for role in roles:
        device = role_devices[role]
        inventory_row = inventory_by_index[role_indices[role]]
        engine = engine_by_role[role]
        readiness = readiness_by_role[role]
        runtime_state = readiness.get("runtime_state")
        compilation = readiness.get("compilation_observation")
        hbm = readiness.get("hbm")
        if not (
            set(engine)
            == {
                "schema_version",
                "role",
                "device",
                "pid",
                "physical_gpu_uuid",
                "engine_started_ns",
            }
            and engine.get("schema_version") == "sloforge.branchfabric.retained-engine-start/v1"
            and engine.get("device") == device
            and engine.get("pid") == worker_pids[role]
            and engine.get("physical_gpu_uuid") == inventory_row.get("uuid")
            and type(engine.get("engine_started_ns")) is int
            and controller["started_ns"] <= engine["engine_started_ns"] <= controller["ended_ns"]
            and readiness.get("schema_version")
            == "sloforge.branchfabric.engine-readiness-evidence/v1"
            and set(readiness) == readiness_fields
            and readiness.get("role") == role
            and readiness.get("device") == device
            and readiness.get("pid") == worker_pids[role]
            and readiness.get("physical_gpu_uuid") == inventory_row.get("uuid")
            and readiness.get("engine_started_ns") == engine["engine_started_ns"]
            and type(readiness.get("model_ready_ns")) is int
            and type(readiness.get("compile_warmup_started_ns")) is int
            and type(readiness.get("compile_warmup_ended_ns")) is int
            and engine["engine_started_ns"]
            <= readiness["model_ready_ns"]
            <= readiness["compile_warmup_started_ns"]
            < readiness["compile_warmup_ended_ns"]
            <= controller["ended_ns"]
            and readiness.get("engine_reloaded") is False
            and readiness.get("queue_empty_pass") is True
            and readiness.get("cudagraph_coverage_pass") is True
            and readiness.get("verification_output_target_verified") is True
            and readiness.get("warmup_output_target_verified") is True
            and isinstance(runtime_state, dict)
            and set(runtime_state) == runtime_fields
            and runtime_state.get("health") == "healthy"
            and runtime_state.get("metrics_available") is True
            and runtime_state.get("prefix_cache_reset") is True
            and all(
                type(runtime_state.get(field)) is int and runtime_state[field] == 0
                for field in (
                    "queue_depth",
                    "request_count",
                    "running_requests",
                    "skipped_waiting_requests",
                    "waiting_requests",
                )
            )
            and isinstance(compilation, dict)
            and set(compilation) == compilation_fields
            and compilation.get("engine_initialization_completed_before_warmup") is True
            and compilation.get("no_active_compilation_event") is True
            and compilation.get("measured_probe_compilation_events") == []
            and isinstance(hbm, dict)
            and hbm.get("stable") is True
            and readiness.get("passed") is True
        ):
            return False
    return True


def _no_preservation_proof_is_pass(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "scan_scope",
            "scanned_file_count",
            "scanned_relative_paths_sha256",
            "forbidden_path_tokens",
            "forbidden_matches",
            "checkpoint_materialized",
            "optimized_export_executed",
            "passed",
        }
        and value.get("schema_version")
        == "sloforge.branchfabric.kill-recompute-no-preservation-outputs/v1"
        and value.get("scan_scope")
        == "controller work root after both workers and before function return"
        and type(value.get("scanned_file_count")) is int
        and value["scanned_file_count"] >= 0
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("scanned_relative_paths_sha256")))
        is not None
        and value.get("forbidden_path_tokens") == list(_FORBIDDEN_PRESERVATION_TOKENS)
        and value.get("forbidden_matches") == []
        and value.get("checkpoint_materialized") is False
        and value.get("optimized_export_executed") is False
        and value.get("passed") is True
    )


def _pre_worker_failure_is_exact(controller: Any, cleanup: Any) -> bool:
    if not isinstance(controller, dict) or not isinstance(cleanup, dict):
        return False
    times = tuple(
        controller.get(field)
        for field in (
            "controller_pid",
            "started_ns",
            "ended_ns",
            "operation_deadline_ns",
            "cleanup_deadline_ns",
        )
    )
    error = controller.get("controller_error")
    audits = controller.get("cuda_clean_import_audits")
    lifecycle = cleanup.get("lifecycle")
    audit = audits[0] if isinstance(audits, list) and len(audits) == 1 else None
    empty_cleanup = (
        "leaked_threads",
        "initial_ipc_resources",
        "final_ipc_resources",
        "leaked_ipc_resources",
        "owned_children",
        "termination_actions",
        "forced_kills",
        "surviving_children",
        "surviving_process_groups",
        "profiler_processes_after",
        "serving_workers_after",
        "rollout_workers_after",
        "resource_tracker_processes_after",
        "zombie_processes_after",
        "compute_processes_after",
        "cleanup_errors",
    )
    return bool(
        controller.get("failure_stage") in _BASE._PRE_WORKER_FAILURE_STAGES
        and controller.get("sealed_evidence") is None
        and controller.get("stable_physical_gpu_identity") is False
        and controller.get("readiness_deadline_ns") is None
        and controller.get("sanity_guard_pair") is None
        and controller.get("cleanup_error") is None
        and len(times) == 5
        and all(type(value) is int and value > 0 for value in times)
        and times[1] < times[2] <= times[4]
        and times[1] < times[3] < times[4]
        and isinstance(error, dict)
        and set(error) == {"type", "message"}
        and all(isinstance(error.get(field), str) and bool(error[field]) for field in error)
        and isinstance(audit, dict)
        and set(audit)
        == {
            "schema_version",
            "stage",
            "pid",
            "observed_at_monotonic_ns",
            "observed_at_utc",
            "loaded_forbidden_modules",
            "cuda_clean",
        }
        and audit.get("schema_version") == "sloforge.branchfabric.cuda-clean-import-audit/v1"
        and audit.get("stage") == "v11-integrated-pre-worker-failure-cleanup"
        and audit.get("pid") == controller.get("controller_pid")
        and type(audit.get("observed_at_monotonic_ns")) is int
        and times[1] <= audit["observed_at_monotonic_ns"] <= times[4]
        and isinstance(audit.get("observed_at_utc"), str)
        and bool(audit["observed_at_utc"])
        and audit.get("loaded_forbidden_modules") == []
        and audit.get("cuda_clean") is True
        and _cleanup_is_pass_complete(cleanup)
        and cleanup.get("parent_pid") == controller.get("controller_pid")
        and isinstance(lifecycle, list)
        and lifecycle[0]["observed_at_monotonic_ns"] >= times[1]
        and cleanup.get("recorded_at_monotonic_ns") <= times[4]
        and all(cleanup.get(field) == [] for field in empty_cleanup)
        and _no_preservation_proof_is_pass(controller.get("no_preservation_output_artifacts"))
    )


def _post_worker_controller_is_exact(controller: Any, cleanup: Any, *, status: str) -> bool:
    if not isinstance(controller, dict) or status not in {"succeeded", "failed"}:
        return False
    times = tuple(
        controller.get(field)
        for field in (
            "controller_pid",
            "started_ns",
            "ended_ns",
            "operation_deadline_ns",
            "cleanup_deadline_ns",
        )
    )
    roles = {"serving", "rollout"}
    worker_results = controller.get("worker_results")
    worker_returncodes = controller.get("worker_returncodes")
    if not isinstance(worker_results, list) or not isinstance(worker_returncodes, dict):
        return False
    result_roles: list[str] = []
    for result in worker_results:
        role = result.get("role") if isinstance(result, dict) else None
        if (
            role not in roles
            or role in result_roles
            or result.get("status") != "succeeded"
            or result.get("attempt_id") != ATTEMPT_ID
            or result.get("mode") != "KILL_AND_RECOMPUTE"
        ):
            return False
        result_roles.append(role)
    successful_roles = {role for role, returncode in worker_returncodes.items() if returncode == 0}
    worker_outcome_is_exact = (
        len(worker_results) == 2 and set(result_roles) == roles and successful_roles == roles
        if status == "succeeded"
        else worker_results == [] and bool(roles - successful_roles)
    )
    return bool(
        set(controller) == _POST_WORKER_CONTROLLER_FIELDS
        and controller.get("schema_version")
        == "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        and controller.get("status") == status
        and controller.get("attempt_id") == ATTEMPT_ID
        and controller.get("execution_mode") == "integrated-kill-recompute-v11"
        and controller.get("terminal_phase") == "INTEGRATED_TRANSACTION"
        and controller.get("absolute_wall_seconds") == 660.0
        and len(times) == 5
        and all(type(value) is int and value > 0 for value in times)
        and times[1] < times[2] <= times[4]
        and times[1] < times[3] < times[4]
        and controller.get("stable_physical_gpu_identity") is True
        and isinstance(controller.get("inventory_before"), list)
        and isinstance(controller.get("inventory_after"), list)
        and len(controller["inventory_before"]) == len(controller["inventory_after"]) == 2
        and isinstance(controller.get("worker_pids"), dict)
        and set(controller["worker_pids"]) == roles
        and all(type(value) is int and value > 0 for value in controller["worker_pids"].values())
        and isinstance(controller.get("worker_process_groups"), dict)
        and set(controller["worker_process_groups"]) == roles
        and isinstance(controller.get("worker_session_ids"), dict)
        and set(controller["worker_session_ids"]) == roles
        and set(worker_returncodes) == roles
        and all(type(value) is int for value in controller["worker_returncodes"].values())
        and _post_worker_engine_evidence_is_exact(controller)
        and worker_outcome_is_exact
        and _cleanup_is_pass_complete(cleanup)
        and controller.get("in_function_cleanup") == cleanup
        and _post_worker_cleanup_ownership_is_exact(controller, cleanup)
        and _no_preservation_proof_is_pass(controller.get("no_preservation_output_artifacts"))
    )


def _validate_kill_remote_result(
    envelope: Mapping[str, Any],
    *,
    config: Any,
    reservation_id: str,
    reservation_commitment_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], dict[str, Any]] | None,
]:
    if set(envelope) != {"result", "materialized"}:
        raise ValueError("kill/recompute result envelope differs from its exact schema")
    remote = envelope["result"]
    materialized = envelope["materialized"]
    expected_fields = {
        "schema_version",
        "status",
        "scientific_status",
        "attempt_id",
        "reservation_id",
        "reservation_commitment_sha256",
        "config_sha256",
        "function_call_id",
        "requested_gpu",
        "gpu_count",
        "controller_and_analysis_interval_seconds",
        "gpu_allocation_seconds_status",
        "hardware_comparability",
        "bound_k_evidence",
        "absolute_deadlines",
        "controller",
        "in_function_cleanup",
        "run_error",
        "completed_at_utc",
        "gpu_allocation_seconds",
        "gpu_seconds",
        "gpu_hours",
        "remote_prefix",
        "remote_manifest_sha256",
    }
    if not isinstance(remote, dict) or set(remote) != expected_fields:
        raise ValueError("kill/recompute completion differs from its exact schema")
    if not isinstance(materialized, dict) or set(materialized) != {"remote_path", "volume_name"}:
        raise ValueError("kill/recompute materialization differs from its exact schema")
    allocation = _BASE._bounded_number(
        remote["gpu_allocation_seconds"], upper=660.0, label="kill allocation time"
    )
    gpu_seconds = _BASE._bounded_number(remote["gpu_seconds"], upper=1320.0, label="kill GPU time")
    interval = _BASE._bounded_number(
        remote["controller_and_analysis_interval_seconds"],
        upper=660.0,
        label="kill controller interval",
    )
    expected_remote = f"{REMOTE_PREFIX}/{ATTEMPT_ID}"
    expected_bound = {
        "integrated_k_status": config.integrated_k_status_artifact,
        "integrated_k_status_sha256": config.integrated_k_status_sha256,
        "integrated_k_remote_manifest": config.integrated_k_remote_manifest,
        "integrated_k_remote_manifest_sha256": config.integrated_k_remote_manifest_sha256,
        "integrated_k_provider_cleanup": config.integrated_k_provider_cleanup_artifact,
        "integrated_k_provider_cleanup_sha256": config.integrated_k_provider_cleanup_sha256,
        "optimized_v11_freeze": config.optimized_v11_freeze_artifact,
        "optimized_v11_freeze_sha256": config.optimized_v11_freeze_sha256,
        "optimized_v11_freeze_tag_binding": config.optimized_v11_freeze_tag_binding_artifact,
        "optimized_v11_freeze_tag_binding_sha256": (config.optimized_v11_freeze_tag_binding_sha256),
        "ledger_snapshot": config.ledger_snapshot_artifact,
        "ledger_snapshot_sha256": config.ledger_snapshot_sha256,
    }
    deadlines = remote.get("absolute_deadlines")
    cleanup = remote.get("in_function_cleanup")
    controller = remote.get("controller")
    hardware = remote.get("hardware_comparability")
    controller_inventory = (
        controller.get("inventory_before") if isinstance(controller, dict) else None
    )
    success_hardware = bool(
        isinstance(hardware, dict)
        and set(hardware)
        == {
            "modal_request",
            "modal_interconnect_selection_available",
            "observed_gpu_names",
            "observed_gpu_uuids",
            "frozen_v10_gpu_name",
            "direct_v10_timing_comparable",
            "timing_comparability_reason",
            "byte_and_amplification_comparable",
            "direct_k_comparison_ready",
        }
        and hardware.get("modal_request") == "A100-80GB:2"
        and hardware.get("modal_interconnect_selection_available") is False
        and isinstance(controller_inventory, list)
        and len(controller_inventory) == 2
        and hardware.get("observed_gpu_names") == [row.get("name") for row in controller_inventory]
        and hardware.get("observed_gpu_uuids") == [row.get("uuid") for row in controller_inventory]
        and hardware.get("direct_k_comparison_ready")
        is hardware.get("direct_v10_timing_comparable")
        and hardware.get("direct_k_comparison_ready") is True
        and hardware.get("byte_and_amplification_comparable") is True
    )
    failure_hardware = hardware == {
        "modal_request": "A100-80GB:2",
        "observed_gpu_names": [],
        "observed_gpu_uuids": [],
        "direct_k_comparison_ready": False,
    }
    success = (
        remote.get("status") == "provisional"
        and remote.get("scientific_status")
        == "pending-local-budget-settlement-and-provider-cleanup"
        and remote.get("run_error") is None
        and isinstance(controller, dict)
        and controller.get("status") == "succeeded"
        and success_hardware
        and _cleanup_is_pass_complete(cleanup)
        and controller.get("in_function_cleanup") == cleanup
        and _post_worker_controller_is_exact(controller, cleanup, status="succeeded")
        and controller.get("controller_error") is None
        and controller.get("cleanup_error") is None
        and all(value == 0 for value in controller["worker_returncodes"].values())
    )
    failed = (
        remote.get("status") == "failed"
        and remote.get("scientific_status") in {"invalid", "invalid-pre-controller"}
        and isinstance(remote.get("run_error"), dict)
        and isinstance(cleanup, dict)
    )
    pre_controller_failed = bool(
        failed
        and remote.get("scientific_status") == "invalid-pre-controller"
        and controller is None
        and set(cleanup)
        == {
            "schema_version",
            "status",
            "phase",
            "attempt_id",
            "function_call_id",
            "controller_started",
            "controller_owned_children",
            "engines_created",
            "cuda_owning_children",
            "ipc_resources_created",
            "primary_error_preserved",
        }
        and cleanup.get("schema_version")
        == "sloforge.branchfabric.kill-recompute-pre-controller-cleanup/v1"
        and cleanup.get("status") == "PASS"
        and cleanup.get("phase") == "PRE_CONTROLLER"
        and cleanup.get("attempt_id") == ATTEMPT_ID
        and cleanup.get("function_call_id") == remote.get("function_call_id")
        and cleanup.get("controller_started") is False
        and all(
            type(cleanup.get(field)) is int and cleanup[field] == 0
            for field in (
                "controller_owned_children",
                "engines_created",
                "cuda_owning_children",
                "ipc_resources_created",
            )
        )
        and cleanup.get("primary_error_preserved") is True
        and hardware
        == {
            "modal_request": "A100-80GB:2",
            "observed_gpu_names": [],
            "observed_gpu_uuids": [],
            "direct_k_comparison_ready": False,
        }
    )
    pre_worker_failed = bool(
        failed
        and remote.get("scientific_status") == "invalid"
        and isinstance(controller, dict)
        and set(controller)
        == _BASE._PRE_WORKER_CONTROLLER_FIELDS | {"no_preservation_output_artifacts"}
        and controller.get("schema_version")
        == "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        and controller.get("status") == "failed"
        and controller.get("attempt_id") == ATTEMPT_ID
        and controller.get("execution_mode") == "integrated-kill-recompute-v11"
        and controller.get("terminal_phase") == "INTEGRATED_TRANSACTION"
        and controller.get("cleanup_scope") == "PRE_WORKER_PREFLIGHT"
        and controller.get("absolute_wall_seconds") == 660.0
        and all(
            controller.get(field) in ([], {})
            for field in (
                "inventory_before",
                "inventory_after",
                "worker_pids",
                "worker_process_groups",
                "worker_session_ids",
                "worker_returncodes",
                "engine_start_evidence",
                "readiness_evidence",
                "worker_results",
                "cleanup_actions",
                "compute_processes_after",
            )
        )
        and _cleanup_is_pass_complete(cleanup)
        and controller.get("in_function_cleanup") == cleanup
        and failure_hardware
        and _pre_worker_failure_is_exact(controller, cleanup)
    )
    post_controller_error = (
        controller.get("controller_error") if isinstance(controller, dict) else None
    )
    controller_times = (
        tuple(
            controller.get(field)
            for field in (
                "controller_pid",
                "started_ns",
                "ended_ns",
                "operation_deadline_ns",
                "cleanup_deadline_ns",
            )
        )
        if isinstance(controller, dict)
        else ()
    )
    post_controller_failed = bool(
        failed
        and remote.get("scientific_status") == "invalid"
        and isinstance(controller, dict)
        and set(controller) == _POST_WORKER_CONTROLLER_FIELDS
        and controller.get("schema_version")
        == "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        and controller.get("status") == "failed"
        and controller.get("attempt_id") == ATTEMPT_ID
        and controller.get("execution_mode") == "integrated-kill-recompute-v11"
        and controller.get("terminal_phase") == "INTEGRATED_TRANSACTION"
        and controller.get("absolute_wall_seconds") == 660.0
        and len(controller_times) == 5
        and all(type(value) is int and value > 0 for value in controller_times)
        and controller_times[1] < controller_times[2]
        and controller_times[2] <= controller_times[4]
        and controller_times[1] < controller_times[3] < controller_times[4]
        and isinstance(post_controller_error, dict)
        and set(post_controller_error) == {"type", "message"}
        and all(
            isinstance(post_controller_error.get(field), str) and bool(post_controller_error[field])
            for field in ("type", "message")
        )
        and (
            controller.get("cleanup_error") is None
            or (
                isinstance(controller.get("cleanup_error"), dict)
                and set(controller["cleanup_error"]) == {"type", "message"}
                and all(
                    isinstance(controller["cleanup_error"].get(field), str)
                    and bool(controller["cleanup_error"][field])
                    for field in ("type", "message")
                )
            )
        )
        and failure_hardware
        and _cleanup_is_pass_complete(cleanup)
        and controller.get("in_function_cleanup") == cleanup
        and _post_worker_controller_is_exact(controller, cleanup, status="failed")
    )
    run_error = remote.get("run_error")
    launcher_failure = bool(
        isinstance(run_error, dict)
        and run_error.get("type") == "RuntimeError"
        and run_error.get("message") == "kill/recompute controller failed closed"
        and isinstance(run_error.get("traceback"), str)
        and "kill/recompute controller failed closed" in run_error["traceback"]
    )
    if (
        not (success or pre_controller_failed or pre_worker_failed or post_controller_failed)
        or (not isinstance(run_error, dict) and not success)
        or (
            not success
            and (
                set(run_error) != {"type", "message", "traceback"}
                or any(
                    not isinstance(run_error.get(key), str) or not run_error[key]
                    for key in run_error
                )
            )
        )
        or ((pre_worker_failed or post_controller_failed) and not launcher_failure)
        or remote.get("schema_version") != "sloforge.branchfabric.kill-recompute-completion/v1"
        or remote.get("attempt_id") != ATTEMPT_ID
        or remote.get("reservation_id") != reservation_id
        or remote.get("reservation_commitment_sha256") != reservation_commitment_sha256
        or remote.get("config_sha256")
        != hashlib.sha256(_BASE.canonical_json_bytes(config)).hexdigest()
        or not isinstance(remote.get("function_call_id"), str)
        or not remote["function_call_id"]
        or remote.get("requested_gpu") != "A100-80GB:2"
        or remote.get("gpu_count") != 2
        or remote.get("gpu_allocation_seconds_status") != "pending-final-function-return"
        or interval > allocation
        or not math.isclose(gpu_seconds, 2 * allocation, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(remote["gpu_hours"], gpu_seconds / 3600.0, rel_tol=0.0, abs_tol=1e-9)
        or remote.get("bound_k_evidence") != expected_bound
        or not isinstance(deadlines, dict)
        or deadlines.get("controller_deadline_monotonic_ns")
        - deadlines.get("function_entry_monotonic_ns")
        != 650_000_000_000
        or deadlines.get("function_deadline_monotonic_ns")
        - deadlines.get("function_entry_monotonic_ns")
        != 660_000_000_000
        or deadlines.get("post_controller_reserve_seconds") != 10.0
        or materialized != {"remote_path": expected_remote, "volume_name": RESULTS_VOLUME}
        or remote.get("remote_prefix") != expected_remote
        or re.fullmatch(r"[0-9a-f]{64}", str(remote.get("remote_manifest_sha256"))) is None
    ):
        raise ValueError("kill/recompute result failed identity/resource/status validation")
    inventory = _BASE._exact_inventory(remote) if success or post_controller_failed else None
    return dict(remote), dict(materialized), inventory


def _verify_kill_manifest(
    local_root: Path,
    *,
    remote: Mapping[str, Any],
    materialized: Mapping[str, Any],
) -> Path:
    manifest = _BASE_VERIFY_MANIFEST(local_root, remote=remote, materialized=materialized)
    paths = [path.relative_to(local_root).as_posix().lower() for path in local_root.rglob("*")]
    if any(any(token in path for token in _FORBIDDEN_PRESERVATION_TOKENS) for path in paths):
        raise ValueError("kill/recompute immutable bundle contains a forbidden export artifact")
    for path in local_root.rglob("*.json"):
        payload = json.loads(path.read_text())
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if (
                    value.get("export_started") is True
                    or value.get("checkpoint_materialized") is True
                ):
                    raise ValueError("kill/recompute bundle claims checkpoint/export execution")
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    completion = json.loads((local_root / "function-completion.json").read_text())
    if completion.get("scientific_status") == "invalid-pre-controller":
        failure_path = local_root / "pre-controller-failure.json"
        if (
            not failure_path.is_file()
            or failure_path.is_symlink()
            or json.loads(failure_path.read_text()) != completion.get("run_error")
            or not (local_root / "in_function_cleanup.json").is_file()
            or json.loads((local_root / "in_function_cleanup.json").read_text())
            != completion.get("in_function_cleanup")
        ):
            raise ValueError("kill/recompute pre-controller failure bundle is incomplete")
    if completion.get("status") == "failed" and completion.get("scientific_status") == "invalid":
        controller = completion.get("controller")
        cleanup = completion.get("in_function_cleanup")
        run_error = completion.get("run_error")
        failure_path = local_root / "function-failure.json"
        finally_path = local_root / "function-finally-cleanup.json"
        cleanup_path = local_root / "in_function_cleanup.json"
        kill_controller_path = local_root / "kill-controller-result.json"
        inherited_controller_path = local_root / "controller-result.json"
        finally_cleanup = (
            json.loads(finally_path.read_text())
            if finally_path.is_file() and not finally_path.is_symlink()
            else None
        )
        inherited_controller = (
            json.loads(inherited_controller_path.read_text())
            if inherited_controller_path.is_file() and not inherited_controller_path.is_symlink()
            else None
        )
        expected_inherited_controller = dict(controller) if isinstance(controller, dict) else None
        if isinstance(expected_inherited_controller, dict):
            expected_inherited_controller.pop("no_preservation_output_artifacts", None)
        if (
            not isinstance(controller, dict)
            or not _cleanup_is_pass_complete(cleanup)
            or not failure_path.is_file()
            or failure_path.is_symlink()
            or json.loads(failure_path.read_text()) != run_error
            or not cleanup_path.is_file()
            or cleanup_path.is_symlink()
            or json.loads(cleanup_path.read_text()) != cleanup
            or not kill_controller_path.is_file()
            or kill_controller_path.is_symlink()
            or json.loads(kill_controller_path.read_text()) != controller
            or inherited_controller != expected_inherited_controller
            or not isinstance(finally_cleanup, dict)
            or set(finally_cleanup)
            != {
                "schema_version",
                "attempt_id",
                "controller_status",
                "controller_cleanup_evidence_present",
                "in_function_cleanup_pass",
                "in_function_cleanup_artifact",
                "provider_cleanup_status",
                "recorded_at_utc",
            }
            or finally_cleanup.get("schema_version")
            != "sloforge.branchfabric.kill-recompute-function-cleanup/v1"
            or finally_cleanup.get("attempt_id") != ATTEMPT_ID
            or finally_cleanup.get("controller_status") != "failed"
            or finally_cleanup.get("controller_cleanup_evidence_present") is not True
            or finally_cleanup.get("in_function_cleanup_pass") is not True
            or finally_cleanup.get("in_function_cleanup_artifact") != "in_function_cleanup.json"
            or finally_cleanup.get("provider_cleanup_status")
            != "pending-function-return-and-local-postflight"
            or not isinstance(finally_cleanup.get("recorded_at_utc"), str)
            or not finally_cleanup["recorded_at_utc"]
        ):
            raise ValueError("kill/recompute failed controller bundle is incomplete")
        proof = controller.get("no_preservation_output_artifacts")
        proof_path = local_root / "kill-no-preservation-output-artifacts.json"
        if (
            not _no_preservation_proof_is_pass(proof)
            or not proof_path.is_file()
            or proof_path.is_symlink()
            or json.loads(proof_path.read_text()) != proof
        ):
            raise ValueError("kill/recompute failed no-preservation output proof is invalid")
    if completion.get("status") == "provisional":
        controller = completion.get("controller")
        proof = (
            controller.get("no_preservation_output_artifacts")
            if isinstance(controller, dict)
            else None
        )
        proof_path = local_root / "kill-no-preservation-output-artifacts.json"
        kill_controller_path = local_root / "kill-controller-result.json"
        if (
            not isinstance(controller, dict)
            or not _no_preservation_proof_is_pass(proof)
            or not proof_path.is_file()
            or proof_path.is_symlink()
            or json.loads(proof_path.read_text()) != proof
            or not kill_controller_path.is_file()
            or kill_controller_path.is_symlink()
            or json.loads(kill_controller_path.read_text()) != controller
        ):
            raise ValueError("kill/recompute no-preservation output proof is invalid")
        cleanup_path = local_root / "in_function_cleanup.json"
        if (
            not _cleanup_is_pass_complete(controller.get("in_function_cleanup"))
            or not cleanup_path.is_file()
            or cleanup_path.is_symlink()
            or json.loads(cleanup_path.read_text()) != controller["in_function_cleanup"]
        ):
            raise ValueError("kill/recompute downloaded in-function cleanup is incomplete")
        worker_pids = controller.get("worker_pids")
        inventory = controller.get("inventory_before")
        results = controller.get("worker_results")
        if (
            not isinstance(worker_pids, dict)
            or set(worker_pids) != {"serving", "rollout"}
            or not isinstance(inventory, list)
            or len(inventory) != 2
            or not isinstance(results, list)
        ):
            raise ValueError("kill/recompute downloaded worker evidence is incomplete")
        experiment_modules = ROOT / "experiments/branchfabric"
        sys.path.insert(0, str(experiment_modules))
        try:
            from gpu_reclamation_kill_recompute_controller_v11 import (
                _validate_kill_worker_results,
            )

            _validate_kill_worker_results(
                results,
                expected={
                    "serving": (worker_pids["serving"], inventory[0]["uuid"]),
                    "rollout": (worker_pids["rollout"], inventory[1]["uuid"]),
                },
                attempt_id=ATTEMPT_ID,
                targeted=False,
            )
        finally:
            sys.path.remove(str(experiment_modules))
    return manifest


def _settle_kill_failure(
    *,
    ledger_path: Path,
    reservation_id: str,
    attempt_id: str,
    failure_root: Path,
    remote: Mapping[str, Any],
    inventory: tuple[dict[str, Any], dict[str, Any]] | None,
    manifest_path: Path,
    client_elapsed_seconds: float,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    if inventory is not None:
        return _BASE_SETTLE_FAILURE(
            ledger_path=ledger_path,
            reservation_id=reservation_id,
            attempt_id=attempt_id,
            failure_root=failure_root,
            remote=remote,
            inventory=inventory,
            manifest_path=manifest_path,
            client_elapsed_seconds=client_elapsed_seconds,
            completed=completed,
        )
    controller = remote.get("controller")
    pre_worker = bool(
        isinstance(controller, dict) and controller.get("cleanup_scope") == "PRE_WORKER_PREFLIGHT"
    )
    error = controller.get("controller_error") if pre_worker else remote.get("run_error")
    if not isinstance(error, dict):
        raise ValueError("kill/recompute inventory-free failure omits its primary error")
    return _BASE._charge_active_failure(
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        attempt_id=attempt_id,
        failure_root=failure_root,
        stage=(
            "verified-pre-worker-failure-conservative-settlement"
            if pre_worker
            else "verified-pre-controller-failure-conservative-settlement"
        ),
        error=RuntimeError(f"{error['type']}: {error['message']}"),
        client_elapsed_seconds=client_elapsed_seconds,
        completed=completed,
        verified_remote=remote,
        verified_manifest_path=manifest_path,
    )


def _install(base: Any) -> None:
    base.ATTEMPT_ID = ATTEMPT_ID
    base.APP_NAME = APP_NAME
    base.REMOTE_PREFIX = REMOTE_PREFIX
    base.FUNCTION_WALL_SECONDS = FUNCTION_WALL_SECONDS
    base._APP = ROOT / "experiments/branchfabric/modal_gpu_reclamation_kill_recompute_v11.py"
    base._LOCAL_RESULT_PARENT = EXPERIMENT_ROOT / "v11-final/kill-recompute/raw"
    base._FAILURE_ROOT = EXPERIMENT_ROOT / "v11-final/kill-recompute/failures"
    base._PROVIDER_INPUTS_PARENT = EXPERIMENT_ROOT / "v11-final/kill-recompute"
    base._PREBUILD_ROOT = EXPERIMENT_ROOT / "v11-kill-recompute-image-prebuild"
    base._load_sealed_config = _load_kill_config
    base._function_wall_seconds = lambda _config: FUNCTION_WALL_SECONDS
    base._maximum_estimated_cost_usd = lambda _config: (
        FUNCTION_WALL_SECONDS
        * (
            2 * base.GPU_PRICE_PER_HOUR_USD / 3600.0
            + 16.0 * base.CPU_PRICE_PER_CORE_SECOND_USD
            + 64.0 * base.MEMORY_PRICE_PER_GIB_SECOND_USD
        )
    )
    base._invoke_modal = _invoke_kill_modal
    base._require_remote_attempt_prefix_fresh = _require_hierarchical_remote_attempt_prefix_fresh
    base._validate_remote_result = _validate_kill_remote_result
    base._verify_downloaded_manifest = _verify_kill_manifest
    base._conservatively_settle_verified_failure = _settle_kill_failure


_PREIMPORT_VERIFIED = _validate_preimport_closure()
_BASE = _load_base()
_BASE_VERIFY_MANIFEST = _BASE._verify_downloaded_manifest
_BASE_SETTLE_FAILURE = _BASE._conservatively_settle_verified_failure


def main() -> int:
    _install(_BASE)
    return int(_BASE.main())


if __name__ == "__main__":
    raise SystemExit(main())
