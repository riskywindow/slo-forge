"""CUDA-clean, fail-closed controller for the one integrated v11 transaction.

This module deliberately does not import CUDA, PyTorch, vLLM, the v10 worker,
or the v10 serving implementation.  It retains the frozen experiment's
two-process and immutable-file protocol while binding the paid launch to the
strict v11 configuration and its content-addressed scientific evidence.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from gpu_reclamation_integrated_worker_v11 import (
    _control_interval_evidence,
    build_live_v10_config,
    expanded_runtime_config,
    validate_sanity_guard_pair,
)

from sloforge.helix.characterization.gpu_reclamation_v11_methodology import (
    V11_INTEGRATED_RESERVATION_WALL_SECONDS,
    Experiment004V11IntegratedConfig,
    Experiment004V11TargetedSourceIdentityConfig,
    validate_bound_artifact,
)

FORBIDDEN_GPU_MODULE_PREFIXES = (
    "torch",
    "vllm",
    "triton",
    "cupy",
    "pynvml",
    "cuda",
    "numba.cuda",
)
ROLES = ("serving", "rollout")
DEVICES = ("gpu0", "gpu1")
SANITY_RATES_RPS = (12.0, 15.0)
ABSOLUTE_WALL_SECONDS = 588.0
TARGETED_SERVING_SANITY_CLASSIFICATIONS = {
    (True, True): "CURRENT_RETAINED_ENVELOPE_STABLE",
    (False, False): "SUSTAINED_WITHIN_ALLOCATION_FAILURE",
    (False, True): "ORDER_DEPENDENT_RECOVERY_SIGNAL",
    (True, False): "UNSTABLE_ENVELOPE",
}
ORIGINAL_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-a"
PRIOR_INVALID_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-b"
REPLACEMENT_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-c"
POST_TARGET_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-d"
PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-e"
PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID = POST_TARGET_INTEGRATED_ATTEMPT_ID
BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-f"
BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID = PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID
QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-g"
QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID = BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID
CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-h"
CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID = QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID
IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-i"
IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID = CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID
REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-j"
REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID = IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID
RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID = "exp004-v11-integrated-s41-k"
RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID = REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID
INITIAL_TARGETED_IDENTITY_ATTEMPT_ID = "exp004-v11-targeted-identity-s41-a"
RETRY_TARGETED_IDENTITY_ATTEMPT_ID = "exp004-v11-targeted-identity-s41-b"
POST_TARGET_LEDGER_SHA256 = "d02df172dff078cedee9da8bb57ed1c811b63b7854ad988a8873d62867672a8b"
PREFUNCTION_FAILED_LEDGER_SHA256 = (
    "8e8011ca993f82f4de7553fb0b2e1ff464d7a65b1c4148410194826895ad6e7d"
)
PREFUNCTION_RETRY_LEDGER_SHA256 = "ae4624645fb2fd57ef098a56826191e78d659742cb1ad0910e9d54cf841aed5d"
PREFUNCTION_RETRY_BUDGET_AUTHORIZATION = (
    "artifacts/branchfabric/gpu-validation/experiment-004/"
    "budget-authorization-v11-continuation.json"
)
PREFUNCTION_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-d.json"
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "provider-cleanup-attempt-d.json"
    ),
    "failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "failures/exp004-v11-integrated-s41-d-conservative-charge.json"
    ),
}
BUNDLED_LEDGER_RETRY_LEDGER_SHA256 = (
    "d9d8e40909b1ffa67f3182545527dcc051c676b7164f6315b18fa5e69ad9ead4"
)
BUNDLED_LEDGER_SNAPSHOT_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/ledger-snapshot-before-attempt-f.json",
    BUNDLED_LEDGER_RETRY_LEDGER_SHA256,
)
BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-e.json",
        "0f163e5cf5db5f07ee85e2fbe13baa74df969e18f4598c9ab83d864f49c339ea",
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "provider-cleanup-attempt-e.json",
        "9ce05c1dc289c67bab147e6aa30deadd87366ec24382c4472fb00c4f8a2c7b20",
    ),
    "failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "failures/exp004-v11-integrated-s41-e-conservative-charge.json",
        "3efbc040ee92364e93606bc550fe93c37fac57ee3c1950e30196d1e96b1d2bc7",
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "raw/exp004-v11-integrated-s41-e/REMOTE_MANIFEST.json",
        "9d8890e16b75e2b612861a85deaebf81874d782f97d1bdd8dd366891a3e46747",
    ),
    "function_completion": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "raw/exp004-v11-integrated-s41-e/function-completion.json",
        "219f62c577d27fdb599ea55a5f8abbfadff8b473da6d0037f437ba4f7a7d1c8b",
    ),
    "function_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "raw/exp004-v11-integrated-s41-e/function-failure.json",
        "6565fca972ee3dcefa8fb33b646439c6cd2a6a3228b0f9c2246d1c651a8d5f89",
    ),
    "function_finally_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "raw/exp004-v11-integrated-s41-e/function-finally-cleanup.json",
        "ea38385b840a5e7fee13bf9fecd552d3b562d8ca88a689d55e1a07d0172b8143",
    ),
    "pre_gpu_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/attempt-e-post-prefunction-pre-gpu-manifest.json",
        "b66879ec7cc3cd5606a24288ff33f929a8a21be3021f81da17fa366969cac8da",
    ),
    "seal_verification": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/seal-verification-attempt-e.json",
        "0b9eb102c7797d8390cbd2d8936ffa9b07d8b8ed3751fef8d89c180b4bc37fb5",
    ),
    "resource_audit": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/attempt-e-pre-controller-resource-audit.json",
        "0a58441af229738533a05cef1f0a1852d3097ce22990647467329f7b9d63738c",
    ),
    "independent_resource_review": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-attempt-e-pre-controller-resource-audit.json",
        "115c55f44cfe318927dafa538a61cbf1d20f091bbc6c369e4629b9f2969a732a",
    ),
}
BUNDLED_LEDGER_TARGET_B_MANIFEST_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/attempt-d-post-target-pre-gpu-manifest.json",
    "4dfa87328919e87f5d39c0754b43f35d5efb7c277ec41303e3eb078f349e57ad",
)
QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256 = (
    "d42b760e3421d82478c00c1d8d85737c008c7c395aae8d7b7a4418a15c03725f"
)
QUEUE_ACCOUNTING_RETRY_LEDGER_SNAPSHOT_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/ledger-snapshot-before-attempt-g.json",
    QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256,
)
QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-f.json",
        "ee43aa48c6cb33410c6ec4cfab8587eaad299571a803c8bbf9a7ad445715d252",
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "provider-cleanup-attempt-f.json",
        "1a9f5ae7475cc77b075efb3c77c4f977b89bd4c2603191d74f7e5124898a101f",
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/REMOTE_MANIFEST.json",
        "994bd867e8b93dcc78cdba12355a1da9971e5460a5bfc18eb5f60ab40dbd6dfe",
    ),
    "function_completion": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/function-completion.json",
        "379a5565874fe876e04c11fbd16d216b321dd58e0d61ffd9bddcaa1ec07d8b65",
    ),
    "function_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/function-failure.json",
        "dd8960358c34ba76684f02cf757f386dca829431b2e048b23c05b3eb4a00e437",
    ),
    "function_finally_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/function-finally-cleanup.json",
        "8381df0f82d2bf5ac8d9ee486987be38beb4facf4ceb963249c3f8888cc26b94",
    ),
    "controller_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/controller-result.json",
        "493a3f9782dd0e3b875a6bbe6cb4371d47e33c421fcb8537329d4ebe4b9f4415",
    ),
    "in_function_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/in_function_cleanup.json",
        "4b5b0f2c4fe865ae1f8ba7ff004bc81722c69dbea2c9f3543342398c54aaa156",
    ),
    "serving_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/serving/failure.json",
        "a5f4441ea6ba6f8698463e97562382da18d925c8a977071f3a72597f820263c1",
    ),
    "partial_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v10-gpu0-partial-failure.json",
        "b7faf5e742b48be42704feea91a878e0bd6b386fa74b1dc70b6e428b6ebd4530",
    ),
    "trigger": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v10-reclaim-trigger.json",
        "81a1e57fecf5da8e0225adb2e9844c17a0ec8ba30072ce44b1197247b581dd28",
    ),
    "source_capture_commit": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v11-source-capture-commit.json",
        "51f0b772b789e15b0d25bb0a3cb68440a96e59e91d30318caf78452dae719384",
    ),
    "pre_export_identity": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v11-source-identity-pre-export-pass.json",
        "636aff8b05caf5a6f68795785a9aca9bd634e1e1e1377b0cb5b861c859d22ef6",
    ),
    "post_export_identity": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v11-source-identity-post-export-pass.json",
        "8631501a58c8286a5c838ff8093c1d7f84fc0036f4e200fdc09a0cc4c460df63",
    ),
    "gpu1_telemetry_000000": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v10-gpu1-telemetry/000000.json",
        "4dfeb8295df42ad7ce4d3ccb4e5eb68f26ae13df8753c1193469ffa4241654ee",
    ),
    "gpu1_telemetry_000001": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v10-gpu1-telemetry/000001.json",
        "800fe9bc6b5694fd473a8acba8be6b5b97ca3c889a847f97e839d276537fbf9b",
    ),
    "gpu1_telemetry_000002": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-f/barriers/v10-gpu1-telemetry/000002.json",
        "b7c7322090247f5ab779724784869ea3387e5bd69304850a6a277038c7244639",
    ),
}
QUEUE_ACCOUNTING_FIX_BINDINGS = {
    "trigger": (
        "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py",
        "9e81ab90ed6d7cb8f7f77e9fd63f2a0b7f3001422c53bcef84eb01344c83d2cc",
    ),
    "trigger_test": (
        "tests/python/test_gpu_reclamation_integrated_trigger_v11.py",
        "fb54ba57cce6dfe8f9641ca637c93e057bd6b91af4f18de1e378601b641b4d44",
    ),
}
QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS = {
    "v10_serving": (
        "experiments/branchfabric/gpu_reclamation_v10_serving.py",
        "d53b007be4fc72d760ae7a83223773a206e0a0eed52137a834ce5b601bcad3ba",
    ),
    "optimized_state_worker": (
        "experiments/branchfabric/gpu_reclamation_worker_v11.py",
        "08449e5af89759847add7433054b2383eb9b1659785fb7e1184f25dc5e02de11",
    ),
    "allocator_epochs": (
        "python/sloforge/continuum/adapters/vllm_allocator_epochs.py",
        "db543a8c4e7c2c7628e78f7dd0c4b04bcab60a4234bc2829787cca21214e35ce",
    ),
    "source_identity": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py",
        "0ca553fbe21b431bc315553e792b7b4de523da3c44f84f36e95ff469ba3e6de0",
    ),
}
QUEUE_ACCOUNTING_RETRY_REVIEW_BINDINGS = {
    12: (
        "runtime-synchronization-and-queue-accounting-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent12-attempt-f-queue-accounting-review.json",
        "38e8722971dde6125b24c8c341f57d0bbfedbd846d69763bc72045c1b4957db8",
    ),
    13: (
        "independent-scientific-methodology-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-attempt-f-serving-methodology-review.json",
        "910fc5658a2ba119dfff56a7de25a811ddb30c38de2f8fddd2aca0e2c1fcaf17",
    ),
}
CONTROL_GATE_RETRY_LEDGER_SHA256 = (
    "4259181994cde63bb783fc42e6533ea79fc6f88cb7f1ebbc555baee40d2f8bf4"
)
CONTROL_GATE_RETRY_LEDGER_SNAPSHOT_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/ledger-snapshot-before-attempt-h.json",
    CONTROL_GATE_RETRY_LEDGER_SHA256,
)
CONTROL_GATE_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-g.json",
        "c322bf734297cf623e6b0828cf738477f5121b28695fff2bc315376ac89ec4ef",
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "provider-cleanup-attempt-g.json",
        "6983acd188e961ee2194f9157ff80b7eb9f031035f79a4aa6b77eff4faaf2604",
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/REMOTE_MANIFEST.json",
        "295d7acb987edc0d450a5ed6004663ecf541e1277d467516669ecc706cc56541",
    ),
    "controller_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/controller-result.json",
        "42fbc59f8aa579b53f22d7455b32a6604b541a61dbe7df28af77fb5d0745004d",
    ),
    "function_completion": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/function-completion.json",
        "eba7d8cc6e58f6fccfb7ccfae5977ed0e7b3978eda5a6738b6b35510ca031701",
    ),
    "function_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/function-failure.json",
        "dd8960358c34ba76684f02cf757f386dca829431b2e048b23c05b3eb4a00e437",
    ),
    "function_finally_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/function-finally-cleanup.json",
        "0b7cb7ce4006bc86204091c3113c890fcafe67ab3715027db6f90c4de08c4c30",
    ),
    "in_function_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/in_function_cleanup.json",
        "508a342ae6bd6a790f42b45a64dc110e07d5d254c6af9c9c3b1dee9b0fb3571c",
    ),
    "serving_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/serving/failure.json",
        "9f3d074020145814c3a8838a686bb9634fbdb45401cefa391c4b4d5dd9b69d9f",
    ),
    "rollout_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/rollout/result.json",
        "eba8eb44644b40ffaf1075771f62e1e1f0479f4e30bed9aa029800cb86e17807",
    ),
    "sanity_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/sanity/sanity-result.json",
        "81ca139220fb0320049398acce101764cfabd5688df208485a6a6ee85cf0990a",
    ),
    "trigger": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v10-reclaim-trigger.json",
        "6a13fa69d2e32118eb537a28d5dca419385fa7851cfd1ac299fcfcae873a8398",
    ),
    "admission_stop": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v11-rollout-admission-stop.json",
        "283f830e60b4f8e9dbe7f6334de159923c996c74e0f3f315f99cd7bea1104762",
    ),
    "source_capture_commit": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v11-source-capture-commit.json",
        "e80792a648f29e1d2d38e5b06426cb842b4fa80fb373370981d6b0ea2f804db3",
    ),
    "pre_export_identity": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v11-source-identity-pre-export-pass.json",
        "8a8575965aa52ba89c87416a1179a55f781b2ef8169b2dec4ebcc0277c34f71e",
    ),
    "post_export_identity": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v11-source-identity-post-export-pass.json",
        "b099a28798922264b0c4ce6d8b4051bd437eece6c1e807dd5036586e86411c53",
    ),
    "gpu1_serving_ready": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v10-gpu1-serving-ready.json",
        "c0e692bf918f5a144e318dc93c980dfb854424c4be56cac54d542ed5ef9a6a9f",
    ),
    "gpu1_first_useful": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v10-gpu1-first-useful.json",
        "6cd8534a0f2f8870d03d9cf9ec8dcd1e0137be27d8dc728224a6dbc96105d7fb",
    ),
    "serving_recovery": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v10-serving-recovery.json",
        "6bb8ee4da1384487225743e543cb7a27c4f50f12b60e826f4fb0533539c5fb89",
    ),
    "restore_start": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/v10-restore-start.json",
        "20cab5a0fa6a6b786ebc224840bd1089f20af1b68d6916aa269f052e1760b76b",
    ),
    "restore_complete": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-g/barriers/rollout-restore-complete.json",
        "4680b54a7fd2ce6ff0e96021eaa5f042f47d7201680f6b262cf96b5694bc32e3",
    ),
}
CONTROL_GATE_FROZEN_V10_BINDINGS = {
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
        "exp004-v10-naive-s41-v7/REMOTE_MANIFEST.json",
        "ada4d36309500d9e02d13cb1f31ce741248fdf54edddfc801155d13a3175ae04",
    ),
    "serving_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
        "exp004-v10-naive-s41-v7/serving/result.json",
        "aa20776b88e0c0757059ccdafcdf82415357f024b2c3daff8c13f7e225576477",
    ),
    "scientific_validity": (
        "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/"
        "exp004-v10-naive-s41-v7/analysis/scientific-validity.json",
        "1e0ac4601f658bcdc7e451cb07105ed836bd3f67e54deaf3a75e8b49b48bd1d8",
    ),
}
CONTROL_GATE_FIX_BINDINGS = {
    "worker": (
        "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
        "fbf9ae25a0f80bf3d108e936420c66c9f2a2762d1661ee34b68b7d89bd02f600",
    ),
    "worker_test": (
        "tests/python/test_gpu_reclamation_integrated_worker_v11.py",
        "3a9492840d70b1bbf53f53f9b19dcf140e1c67cbaffa21c354bc76a413d4e9b8",
    ),
}
CONTROL_GATE_UNCHANGED_RUNTIME_BINDINGS = {
    "optimized_state_worker": QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS["optimized_state_worker"],
    "allocator_epochs": QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS["allocator_epochs"],
    "source_identity": QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS["source_identity"],
}
CONTROL_GATE_RETRY_REVIEW_BINDINGS = {
    12: (
        "runtime-synchronization-and-control-methodology-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent12-attempt-g-control-gate-review.json",
        "a697efaf90894a36b43d47aa7ba294e1db0dcd7b33b89c7a6af875108314d0d6",
    ),
    13: (
        "independent-scientific-methodology-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-attempt-g-serving-methodology-review.json",
        "cc099bfa910270c1364f6e05c024047f679b93063c50a93776735392580596c0",
    ),
}
CONTROL_GATE_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS = {
    "optimized_state_worker": (
        "experiments/branchfabric/gpu_reclamation_worker_v11.py",
        "08449e5af89759847add7433054b2383eb9b1659785fb7e1184f25dc5e02de11",
    ),
    "allocator_epochs": (
        "python/sloforge/continuum/adapters/vllm_allocator_epochs.py",
        "db543a8c4e7c2c7628e78f7dd0c4b04bcab60a4234bc2829787cca21214e35ce",
    ),
    "source_identity": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py",
        "0ca553fbe21b431bc315553e792b7b4de523da3c44f84f36e95ff469ba3e6de0",
    ),
    "allocator_epoch_tests": (
        "tests/python/test_vllm_allocator_epochs.py",
        "7c437b43bfcd3f697b7c580c297829c30561753c85a4bcd300a86e9cb4240b33",
    ),
    "source_identity_tests": (
        "tests/python/test_vllm_reclamation_v11_ownership.py",
        "0480204b2a716170a5582caa40b52bd014b28260a9fc04d623a7cbb2b902cc68",
    ),
}
IMAGE_CLOSURE_RETRY_LEDGER_SHA256 = (
    "37cdf269d92611f947684d1d0f79152446d87a6753b98b87d5e732237dd8d00c"
)
IMAGE_CLOSURE_RETRY_LEDGER_SNAPSHOT_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/ledger-snapshot-before-attempt-i.json",
    IMAGE_CLOSURE_RETRY_LEDGER_SHA256,
)
IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-h.json",
        "cfc4ebc883e6c8fd2ec7f5989c99ccf1248f5101d1a2ba2cc43ac4b2c2050d4a",
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "provider-cleanup-attempt-h.json",
        "44ac116033e8f9e10bed74ff195b24dc3a43c074759a3a99b865227ff11c649f",
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-h/REMOTE_MANIFEST.json",
        "07abf45fb974d9e73b881fea990de98aeb1bef5ea4818c3c9672eb918336dcfb",
    ),
    "controller_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-h/controller-result.json",
        "92a668d364ff62dfa7f803f24abdc3fc2c2109c2b3679cca87a1abbd4f76c5b3",
    ),
    "function_completion": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-h/function-completion.json",
        "80355ea6f71b96d3ff62cfc2a817bee54374549f61c06ad76e8061cc91c75a41",
    ),
    "function_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-h/function-failure.json",
        "dd8960358c34ba76684f02cf757f386dca829431b2e048b23c05b3eb4a00e437",
    ),
    "function_finally_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-h/function-finally-cleanup.json",
        "4fc275276efb40fcab584d5823ac3ece736e7d8d8ed3c0ac998e85f91f9239e9",
    ),
    "in_function_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-h/in_function_cleanup.json",
        "18c60202722a2dfbb4f55ec2cf25e9c7bec74187422d4311d6de76974d5e6f8f",
    ),
    "conservative_failure_charge": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/failures/"
        "exp004-v11-integrated-s41-h-conservative-charge.json",
        "77bd380d72a4afe2d9e644262545e1313ed89a5830615fa7120adb4840fc7394",
    ),
    "config": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "exp004-v11-integrated-s41-h-config.json",
        "89f3db25c2cfc2ed1a222da75cdad0aba3dc2a2c44c8befac5ba5b8aba27b97d",
    ),
    "authorization_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/attempt-h-post-control-gate-pre-gpu-manifest.json",
        "84de6be77b13ebacec0019441a63fad2ffe7f87971ed2c4f0aa83083c5a9826d",
    ),
    "seal_verification": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/seal-verification-attempt-h.json",
        "6a8a1ec407e8d01fde3fdaaa5e77065f0b86fee1e74aa1d35fc95ce0847ecb83",
    ),
    "image_prebuild": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-image-prebuild/"
        "exp004-v11-integrated-s41-h.json",
        "ffe7db3d3a0106ea092d1b3cd357c0494cea0e38fb91774be6ff8f4c37a30d5f",
    ),
}
IMAGE_CLOSURE_FIX_BINDINGS = {
    "sole_coordinator": (
        "tools/branchfabric-experiment-004-v11-final.py",
        "35d54a05c64f7f9c82d107e045e421d6e3139c4a976787c924963c22d604449a",
    ),
    "modal_launcher": (
        "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py",
        "638555b54abb6804ced704f06fc6016c176c4f5f44abe7e32544e3b6dbc32cde",
    ),
    "sole_coordinator_test": (
        "tests/python/test_experiment_004_v11_final_coordinator.py",
        "db0e6a8459c3830b6b66ea8d4348fb991434df885224f9e7a052e43fba0a2b8c",
    ),
    "modal_launcher_test": (
        "tests/python/test_modal_gpu_reclamation_integrated_v11.py",
        "02f2d10d2f00b43c10bccb7d8ee89f7fd7d361be56aa9bc0c1da156971d025d3",
    ),
}
ATTEMPT_I_SEALED_COORDINATOR_SHA256 = (
    "bf2a5d5a327a7d5ba2df5fb796ddc91224e72b1cb0ee2c854e6ef5b2e284b6e4"
)
IMAGE_CLOSURE_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS = (
    CONTROL_GATE_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS
)
IMAGE_CLOSURE_TARGET_B_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/"
        "status-attempt-b.json",
        "1e4d116b85ed805c9a821ddcb5bd40667d78769d46c78cbae8bd5b6216338a8e",
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/"
        "provider-cleanup-attempt-b.json",
        "f2682ac3d57404b242e02185380da588527649870cb8cc554ce9f08f85fc507f",
    ),
    "identity_gate": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/barriers/v11-source-identity-pre-export-pass.json",
        "da83a650b404cb9048990f6e16ed491ac2d1672d95ed5b686100dfb9da9ec9c0",
    ),
    "source_capture_commit": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/barriers/v11-source-capture-commit.json",
        "2a643c6010fb5754e9cb87fe8196734210045b16614b2c21b9cad2c8e9b8715e",
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/REMOTE_MANIFEST.json",
        "ba4ebb51909a6bb6f640920771dff40c6418b3252d4cf5d7cb6b9a01ad0418e1",
    ),
    "in_function_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/in_function_cleanup.json",
        "3f05994a0ef613453ffc6bc16ae841855f264413c47028f5c45e9806763f51a2",
    ),
}
IMAGE_CLOSURE_RETRY_REVIEW_BINDINGS = {
    12: (
        "runtime-synchronization-and-image-packaging-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent12-attempt-h-runtime-packaging-review.json",
        "563fcf19a9667b2d5a8e92817ec6a0bc717dacef4cd307a739162a3e894e00b4",
    ),
    13: (
        "independent-scientific-methodology-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-attempt-h-scientific-methodology-review.json",
        "37298093e8d97138b709db33a8fd6923d7855bc8f42fc193c88c88f05b204a7f",
    ),
}
REMOTE_PREFIX_RETRY_LEDGER_SHA256 = (
    "37cdf269d92611f947684d1d0f79152446d87a6753b98b87d5e732237dd8d00c"
)
REMOTE_PREFIX_RETRY_LEDGER_SNAPSHOT_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/ledger-snapshot-before-attempt-j.json",
    REMOTE_PREFIX_RETRY_LEDGER_SHA256,
)
REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-i.json",
        "d400a7bf92f8a45a9404384df86fb03fab8dba7d2170fa71d98cf338b9609d87",
    ),
    "remote_prefix_inventory": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "preflight/attempt-i-remote-prefix-inventory.json",
        "20eeb978342cfaf4d59448066dc1259e367e874f60840abd2497b111ca7c332d",
    ),
    "config": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "exp004-v11-integrated-s41-i-config.json",
        "a28f5f86d62706c8054e2534234458f7e2cbd3263bd07da87044ad6e7d157a64",
    ),
    "authorization_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/attempt-i-post-image-closure-pre-gpu-manifest.json",
        "71d68e6f30c5edc26f905afb30f9806ae6574c60bfbb64f9f112ac2b72ef8ee8",
    ),
    "seal_verification": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "authorization/seal-verification-attempt-i.json",
        "5c185fa63f7bd01237c48761743cee8f09adcbf99609a461aa87868305e41d4c",
    ),
}
REMOTE_PREFIX_PARSER_FIX_BINDINGS = {
    "sole_coordinator": (
        "tools/branchfabric-experiment-004-v11-final.py",
        "35d54a05c64f7f9c82d107e045e421d6e3139c4a976787c924963c22d604449a",
    ),
    "sole_coordinator_test": (
        "tests/python/test_experiment_004_v11_final_coordinator.py",
        "db0e6a8459c3830b6b66ea8d4348fb991434df885224f9e7a052e43fba0a2b8c",
    ),
}
REMOTE_PREFIX_UNCHANGED_RUNTIME_BINDINGS = {
    "methodology": (
        "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py",
        "4d6c2f00d5f97b6d245bceee1942a30e312dd85cf7d5a26a07153316590ae3c3",
    ),
    "integrated_worker": (
        "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
        "fbf9ae25a0f80bf3d108e936420c66c9f2a2762d1661ee34b68b7d89bd02f600",
    ),
    "frozen_calibration_worker": (
        "experiments/branchfabric/gpu_reclamation_worker.py",
        "e6858e445ddbca912877fc67155dd9422c82d6f6fd4dcbf71871831dee15008a",
    ),
    "optimized_state_worker": (
        "experiments/branchfabric/gpu_reclamation_worker_v11.py",
        "08449e5af89759847add7433054b2383eb9b1659785fb7e1184f25dc5e02de11",
    ),
    "allocator_epochs": (
        "python/sloforge/continuum/adapters/vllm_allocator_epochs.py",
        "db543a8c4e7c2c7628e78f7dd0c4b04bcab60a4234bc2829787cca21214e35ce",
    ),
    "source_identity": (
        "python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py",
        "0ca553fbe21b431bc315553e792b7b4de523da3c44f84f36e95ff469ba3e6de0",
    ),
    "trigger": (
        "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py",
        "9e81ab90ed6d7cb8f7f77e9fd63f2a0b7f3001422c53bcef84eb01344c83d2cc",
    ),
    "modal_launcher": (
        "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py",
        "638555b54abb6804ced704f06fc6016c176c4f5f44abe7e32544e3b6dbc32cde",
    ),
}
RETAINED_CAPACITY_RETRY_LEDGER_SHA256 = (
    "a6209f1cd59e8471f342dccae5bf467b65e78ce17dd21d74165cb4c3ee845670"
)
RETAINED_CAPACITY_RETRY_LEDGER_SNAPSHOT_BINDING = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
    "authorization/ledger-snapshot-before-attempt-k.json",
    RETAINED_CAPACITY_RETRY_LEDGER_SHA256,
)
RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "status-attempt-j.json",
        "641bc33ce3975d43722d692da22f459eb5079803536581818f4f65aeda86a4cb",
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/"
        "provider-cleanup-attempt-j.json",
        "67cdf01cdd33aff305d126bc1481dfefe9d038292dfc0a1ec5487d6e5ce7b56c",
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/REMOTE_MANIFEST.json",
        "5167a6542aa250e7f1de3b6009603cb17acdb43312507be2e25fc370cd4f10de",
    ),
    "readiness": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/readiness/both-engines-ready.json",
        "e304f2317d30d067d912866c8deff75873729202867829012f597989857e051b",
    ),
    "serving_readiness": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/serving.ready.json",
        "e3587cffa4fcf56a7c4687a00406218c7135fcc09e38467b1b5e70da2bf571c4",
    ),
    "rollout_readiness": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/rollout.ready.json",
        "5cf944dc6323ccfdbf0d9495d4ee7ff32cc7ed1c095d9c6b755dcb5783b8d3b9",
    ),
    "probe_command": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/capacity/probe-00.command.json",
        "1070bbd9ff14b8ae74be57929c5f1ca8ae527fb4faec23395f64e84c87f871e7",
    ),
    "probe_gpu0_raw": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/capacity/probe-00.gpu0.raw.json",
        "5c3e1da4ffee429bc724cd61aafee8f30e05024c82b417355571f695c50d9c0e",
    ),
    "probe_gpu0_reset": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/capacity/probe-00.gpu0.reset.json",
        "aa1534e9e9f9b7d4b1faffb70611fd1ac1e459606e07a181ee16d3d4ad416224",
    ),
    "probe_gpu1_raw": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/capacity/probe-00.gpu1.raw.json",
        "49b759021cbb50977683fd0a5d0ce750790a62c2110040017dc278c48b6d131e",
    ),
    "probe_gpu1_reset": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/barriers/capacity/probe-00.gpu1.reset.json",
        "b98c9a5e608fc6fd85da52093444ecc515843a94bc986123ca81b44fb7350392",
    ),
    "controller_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/controller-result.json",
        "015e6350dbe25ecb1a97c072c5c6259bebe0f8cd8b63b878e409c8975a02ed0e",
    ),
    "in_function_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/in_function_cleanup.json",
        "9243ce948af5875057e3af836a2df82662c73720fd576e11f1ac0834e4c9fb57",
    ),
    "function_completion": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/function-completion.json",
        "5b3c8f088625d1e8c790108bd165616a6e44b110ede4a97d66a202cfbf135826",
    ),
    "function_failure": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/function-failure.json",
        "8aad1a23ece1287ec95a9a86a001da3742e9f57986cb7d7c4f035e7107a70f5f",
    ),
    "function_finally_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        "exp004-v11-integrated-s41-j/function-finally-cleanup.json",
        "4340731863e6271574e4dd00faac1a86dc8659f946ffc31a4b20cf7f4c192140",
    ),
    "failure_charge": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/failures/"
        "exp004-v11-integrated-s41-j-conservative-charge.json",
        "cf6ca51e3a3dddf1a892f0383f622abee50598aaf89e1f75b5c9497a7f39dc49",
    ),
}
RETAINED_CAPACITY_FIX_BINDING_PATHS = {
    "same_allocation_controller": (
        "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py"
    ),
    "three_probe_scoped_worker_wrapper": (
        "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py"
    ),
    "controller_and_readiness_tests": (
        "tests/python/test_gpu_reclamation_integrated_controller_v11.py"
    ),
    "same_allocation_reproduction_tests": (
        "tests/python/test_gpu_reclamation_integrated_sanity_reproduction_v11.py"
    ),
}
RETAINED_CAPACITY_RUNTIME_PROJECTION_SOURCES = {
    "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py": (
        "ROLES",
        "DEVICES",
        "SANITY_RATES_RPS",
        "TARGETED_SERVING_SANITY_CLASSIFICATIONS",
        "_retained_engine_readiness_assessment_components",
        "_validate_retained_engine_readiness_comparability",
        "_assess_retained_engine_readiness_comparability",
        "_classify_targeted_serving_sanity",
        "_validate_targeted_probe_runtime_evidence",
        "_run_targeted_serving_sanity_probe",
        "_validate_selected_load_gate_commitment",
        "run_integrated_v11_controller",
    ),
    "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py": (
        "_V11_POST_J_SANITY_PROBE_COUNT",
        "_run_v11_integrated_calibration_phase",
    ),
    "tests/python/test_gpu_reclamation_integrated_controller_v11.py": (
        "_retained_readiness_replay",
        "test_retained_engine_readiness_comparability_accepts_measured_replays",
        "test_retained_engine_readiness_comparability_rejects_attempt_j_replay",
        "test_retained_engine_readiness_assessment_persists_recomputed_attempt_j_failure",
        "test_retained_engine_readiness_comparability_rejects_invalid_floor",
        "test_retained_engine_readiness_comparability_rejects_non_exact_finite_float",
        "test_retained_engine_readiness_comparability_requires_exactly_one_batch16",
        "test_retained_engine_readiness_comparability_requires_exact_batch_matrix",
        "test_retained_engine_readiness_comparability_requires_exact_float_for_every_batch",
        "test_retained_engine_readiness_comparability_rejects_non_exact_batch16_timing",
        "test_retained_engine_readiness_comparability_rejects_batch16_timestamp_order",
        "test_retained_engine_readiness_comparability_binds_batch16_throughput_to_timing",
        "test_retained_engine_readiness_comparability_rejects_batch_accounting_tamper",
        "test_retained_engine_readiness_comparability_rejects_identity_tamper",
        "test_retained_engine_readiness_comparability_rejects_continuity_tamper",
        "test_retained_engine_readiness_comparability_requires_disjoint_engine_identity",
        "test_retained_engine_readiness_comparability_requires_exact_pair",
        "test_retained_readiness_comparator_does_not_replace_frozen_sanity_assessor",
        "test_integrated_readiness_comparability_precedes_all_serving_probes",
    ),
}
RETAINED_CAPACITY_UNCHANGED_RUNTIME_BINDINGS = {
    **{
        name: binding
        for name, binding in REMOTE_PREFIX_UNCHANGED_RUNTIME_BINDINGS.items()
        if name != "integrated_worker"
    },
    "frozen_sanity_assessor": (
        "experiments/branchfabric/gpu_reclamation_controller.py",
        "c91ccd8b3aa4b4c5a9034a5137ccc33257e3471528a96dcc2ca7920c8b7eefc9",
    ),
    "capacity_controller": (
        "experiments/branchfabric/gpu_capacity_calibration_controller.py",
        "31fb70855e49d3f59493592de84120fc82f6a950e5b5edd0e775d5ba47978aa2",
    ),
    "capacity_worker": (
        "experiments/branchfabric/gpu_capacity_calibration_worker.py",
        "2c41499a847879b813ce01470a7e32625d772fc07233753404296125c4afc225",
    ),
    "capacity_model": (
        "python/sloforge/helix/characterization/gpu_capacity_calibration.py",
        "7ffb993004fbc9b595cc3c5fc2e503ffd9dd2370befdeb8714d017badc4e0b3d",
    ),
}
RETAINED_CAPACITY_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS = (
    CONTROL_GATE_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS
)
RETAINED_CAPACITY_TARGET_B_EVIDENCE_BINDINGS = IMAGE_CLOSURE_TARGET_B_EVIDENCE_BINDINGS
RETAINED_CAPACITY_RETRY_REVIEW_BINDINGS = {
    12: (
        "agent12-runtime-synchronization-and-capacity-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent12-attempt-j-runtime-capacity-review.json",
    ),
    13: (
        "independent-scientific-methodology-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-attempt-j-scientific-methodology-review.json",
    ),
}
SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS = frozenset(
    {
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
        "bound_artifacts",
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
)
PRIOR_TARGETED_RETRY_STATUS = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/status.json"
)
PRIOR_TARGETED_RETRY_CLEANUP = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/"
    "provider-cleanup.json"
)
PRIOR_TARGETED_RETRY_IN_FUNCTION_CLEANUP = (
    "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
    "exp004-v11-targeted-identity-s41-a/in_function_cleanup.json"
)
POST_TARGET_EVIDENCE_BINDINGS = {
    "status": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/"
        "status-attempt-b.json"
    ),
    "provider_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/"
        "provider-cleanup-attempt-b.json"
    ),
    "identity_gate": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/barriers/"
        "v11-source-identity-pre-export-pass.json"
    ),
    "source_capture_commit": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/barriers/v11-source-capture-commit.json"
    ),
    "remote_manifest": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/REMOTE_MANIFEST.json"
    ),
    "in_function_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/in_function_cleanup.json"
    ),
    "trigger": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/barriers/v10-reclaim-trigger.json"
    ),
    "admission_stop": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/barriers/v11-rollout-admission-stop.json"
    ),
    "controller_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/controller-result.json"
    ),
    "serving_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/serving/result.json"
    ),
    "rollout_result": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        "exp004-v11-targeted-identity-s41-b/rollout/result.json"
    ),
}
POST_TARGET_RUNTIME_INSTANCE_ID = (
    "adapter:46767681563072:scheduler:46771086498592:manager:46791730063904"
)
POST_TARGET_REVIEW_BINDINGS = {
    11: (
        "allocation-identity-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent11-targeted-allocation-identity-review.json",
    ),
    12: (
        "runtime-synchronization-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent12-targeted-runtime-synchronization-review.json",
    ),
    13: (
        "scientific-methodology-reviewer",
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/reviews/"
        "agent13-targeted-scientific-methodology-review.json",
    ),
}
INTEGRATED_PRELAUNCH_BINDINGS = {
    "methodology": "python/sloforge/helix/characterization/gpu_reclamation_v11_methodology.py",
    "worker": "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
    "frozen_calibration_worker": "experiments/branchfabric/gpu_reclamation_worker.py",
    "capacity_worker": "experiments/branchfabric/gpu_capacity_calibration_worker.py",
    "optimized_state_worker": "experiments/branchfabric/gpu_reclamation_worker_v11.py",
    "allocator_epochs": "python/sloforge/continuum/adapters/vllm_allocator_epochs.py",
    "source_identity": ("python/sloforge/continuum/adapters/vllm_reclamation_v11_ownership.py"),
    "controller": "experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py",
    "launcher": "experiments/branchfabric/modal_gpu_reclamation_integrated_v11.py",
    "paid_coordinator": "tools/branchfabric-experiment-004-v11-final.py",
    "trigger": "experiments/branchfabric/gpu_reclamation_integrated_trigger_v11.py",
    "worker_test": "tests/python/test_gpu_reclamation_integrated_worker_v11.py",
    "frozen_calibration_worker_test": "tests/python/test_gpu_reclamation_worker_integrated.py",
    "capacity_worker_test": "tests/python/test_gpu_capacity_calibration_harness.py",
    "optimized_state_worker_test": "tests/python/test_gpu_reclamation_runtime_v11.py",
    "allocator_epochs_test": "tests/python/test_vllm_allocator_epochs.py",
    "source_identity_test": "tests/python/test_vllm_reclamation_v11_ownership.py",
    "controller_test": "tests/python/test_gpu_reclamation_integrated_controller_v11.py",
    "launcher_test": "tests/python/test_modal_gpu_reclamation_integrated_v11.py",
    "paid_coordinator_test": "tests/python/test_experiment_004_v11_final_coordinator.py",
    "trigger_test": "tests/python/test_gpu_reclamation_integrated_trigger_v11.py",
    "trigger_race_test": "tests/python/test_gpu_reclamation_integrated_trigger_v11_races.py",
    "cross_runtime_review": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11/reviews/"
        "agent03-integrated-cross-runtime-audit.json"
    ),
    "make_check": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/offline-fixes/"
        "make-check.json"
    ),
}
REPLACEMENT_EVIDENCE_BINDINGS = {
    "prior_attempt_cleanup": (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11/runtime/"
        "exp004-v11-integrated-s41-b/postflight-cleanup-and-settlement.json"
    ),
}
FINAL_PRE_GPU_REVIEW_ROLES = {
    1: "early-trigger-implementation",
    2: "trigger-race-review",
    3: "process-cleanup-implementation",
    4: "process-lifecycle-review",
    5: "classification-logic-review",
    6: "scientific-validity-review",
    7: "gpu-budget-review",
    8: "integrated-methodology-review",
}
PROCESS_LIFECYCLE_PHASES = (
    "RUNNING",
    "QUIESCE",
    "ENGINE_STOP",
    "WORKER_STOP",
    "CHILD_REAP",
    "PGID_EMPTY",
    "CUDA_RELEASED",
    "FUNCTION_RETURN",
)
PROFILER_COMMAND_TOKENS = ("nsys", "ncu", "nvprof")
RESOURCE_TRACKER_COMMAND_TOKEN = "multiprocessing.resource_tracker"
MAX_RECLAIM_TRIGGER_REACTION_NS = 100_000_000
MAX_SCIENTIFIC_TRIGGER_DEPTH = 30
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        value = dump(mode="json")

    def default(item: Any) -> Any:
        nested = getattr(item, "model_dump", None)
        if callable(nested):
            return nested(mode="json")
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=default,
        )
        + "\n"
    ).encode()


def _write_new(path: Path, value: Any) -> None:
    """Publish one immutable barrier using an exclusive hard link."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retained_capacity_runtime_projection(repository_root: Path) -> dict[str, Any]:
    """Hash only the reviewed retained-allocation runtime and its direct tests."""

    sources = {
        reference: list(symbols)
        for reference, symbols in RETAINED_CAPACITY_RUNTIME_PROJECTION_SOURCES.items()
    }
    projected_nodes: list[dict[str, str]] = []
    for reference, symbol_names in RETAINED_CAPACITY_RUNTIME_PROJECTION_SOURCES.items():
        source_path = repository_root / reference
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(f"Attempt-K runtime projection source is invalid: {reference}")
        try:
            module = ast.parse(source_path.read_text(), filename=reference)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise RuntimeError(
                f"Attempt-K runtime projection source cannot be parsed: {reference}"
            ) from exc
        candidates: dict[str, list[ast.AST]] = {}
        for node in module.body:
            names: tuple[str, ...]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = (node.name,)
            elif isinstance(node, ast.Assign):
                names = tuple(target.id for target in node.targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = (node.target.id,)
            else:
                names = ()
            for name in names:
                candidates.setdefault(name, []).append(node)
        for name in symbol_names:
            matches = candidates.get(name, [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"Attempt-K runtime projection symbol is not unique: {reference}:{name}"
                )
            projected_nodes.append(
                {
                    "artifact": reference,
                    "symbol": name,
                    "ast": ast.dump(matches[0], annotate_fields=True, include_attributes=False),
                }
            )
    projection = {
        "schema_version": ("sloforge.branchfabric.v11-retained-capacity-runtime-projection/v1"),
        "algorithm": "python-ast-dump-no-attributes/v1",
        "sources": sources,
        "nodes": projected_nodes,
    }
    return {
        "schema_version": projection["schema_version"],
        "algorithm": projection["algorithm"],
        "sources": sources,
        "sha256": hashlib.sha256(_canonical_bytes(projection)).hexdigest(),
    }


def cuda_clean_import_audit(stage: str) -> dict[str, Any]:
    loaded = sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in FORBIDDEN_GPU_MODULE_PREFIXES
        )
    )
    return {
        "schema_version": "sloforge.branchfabric.cuda-clean-import-audit/v1",
        "stage": stage,
        "observed_at_utc": _utc_now(),
        "observed_at_monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "loaded_forbidden_modules": loaded,
        "cuda_clean": not loaded,
    }


def _parse_inventory(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 7:
            raise RuntimeError("nvidia-smi inventory returned an unexpected field count")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
                "memory_used_mib": int(fields[5]),
                "utilization_percent": int(fields[6]),
            }
        )
    if len(rows) != 2:
        raise RuntimeError(f"integrated v11 requires exactly two visible GPUs, got {len(rows)}")
    if len({row["uuid"] for row in rows}) != 2:
        raise RuntimeError("integrated v11 received duplicate physical GPU UUIDs")
    if len({row["index"] for row in rows}) != 2:
        raise RuntimeError("integrated v11 received duplicate visible GPU indices")
    for row in rows:
        if "A100" not in str(row["name"]) or int(row["memory_total_mib"]) < 79_000:
            raise RuntimeError(f"integrated v11 requires A100-80GB, observed {row}")
    return rows[0], rows[1]


def _subprocess_timeout(*, deadline_ns: int | None, ceiling_seconds: float) -> float:
    if deadline_ns is None:
        return ceiling_seconds
    remaining = (deadline_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0.0:
        raise TimeoutError("integrated v11 exhausted its absolute deadline")
    return min(ceiling_seconds, remaining)


def _inventory(*, deadline_ns: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_subprocess_timeout(deadline_ns=deadline_ns, ceiling_seconds=15.0),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi inventory failed: {completed.stderr.strip()}")
    return _parse_inventory(completed.stdout)


def _parse_compute_processes(stdout: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        if not raw.strip() or raw.lower().startswith("no running processes"):
            continue
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi compute query returned an unexpected row")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mib": int(fields[3]),
            }
        )
    return tuple(rows)


def _compute_processes(*, deadline_ns: int | None = None) -> tuple[dict[str, Any], ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=_subprocess_timeout(deadline_ns=deadline_ns, ceiling_seconds=15.0),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute query failed: {completed.stderr.strip()}")
    return _parse_compute_processes(completed.stdout)


def _require_no_compute_processes(processes: tuple[dict[str, Any], ...], *, phase: str) -> None:
    if processes:
        raise RuntimeError(f"unexpected GPU compute processes during {phase}: {processes}")


def _stable_inventory_identity(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> bool:
    keys = ("index", "uuid", "name", "driver_version", "memory_total_mib")
    return len(before) == len(after) == 2 and all(
        all(left.get(key) == right.get(key) for key in keys)
        for left, right in zip(before, after, strict=True)
    )


def _wait_for_files(
    paths: tuple[Path, ...],
    *,
    deadline_ns: int,
    phase: str,
    processes: tuple[tuple[str, subprocess.Popen[bytes]], ...] = (),
    observe_processes: Any | None = None,
) -> None:
    while not all(path.is_file() for path in paths):
        if observe_processes is not None:
            observe_processes(phase)
        exited = {
            role: returncode
            for role, process in processes
            if (returncode := process.poll()) is not None
        }
        if exited:
            raise RuntimeError(f"workers exited while waiting for {phase}: {exited}")
        if time.monotonic_ns() >= deadline_ns:
            absent = [str(path) for path in paths if not path.is_file()]
            raise TimeoutError(f"timed out waiting for {phase}: {absent}")
        time.sleep(0.05)


def _process_tree_procfs() -> tuple[dict[str, Any], ...]:
    root = Path("/proc")
    if not root.is_dir():
        raise FileNotFoundError("/proc is unavailable")
    rows: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            close = raw.rfind(")")
            if close < 0:
                continue
            fields = raw[close + 2 :].split()
            command = " ".join(
                item.decode("utf-8", errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            )
            rows.append(
                {
                    "pid": int(entry.name),
                    "ppid": int(fields[1]),
                    "pgid": int(fields[2]),
                    "sid": int(fields[3]),
                    "state": fields[0],
                    "start_token": fields[19],
                    "command": command,
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
    return tuple(sorted(rows, key=lambda item: int(item["pid"])))


def _process_tree_ps() -> tuple[dict[str, Any], ...]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,sess=,state=,command="],
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ps process-tree query failed: {completed.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        fields = raw.strip().split(maxsplit=5)
        if len(fields) < 5:
            continue
        try:
            rows.append(
                {
                    "pid": int(fields[0]),
                    "ppid": int(fields[1]),
                    "pgid": int(fields[2]),
                    "sid": int(fields[3]),
                    "state": fields[4],
                    "start_token": None,
                    "command": fields[5] if len(fields) == 6 else "",
                }
            )
        except ValueError:
            continue
    return tuple(sorted(rows, key=lambda item: int(item["pid"])))


def _process_tree() -> tuple[dict[str, Any], ...]:
    try:
        return _process_tree_procfs()
    except FileNotFoundError:
        return _process_tree_ps()


def _descendant_pids(processes: Sequence[Mapping[str, Any]], root_pid: int) -> set[int]:
    by_parent: dict[int, set[int]] = {}
    for process in processes:
        by_parent.setdefault(int(process["ppid"]), set()).add(int(process["pid"]))
    descendants: set[int] = set()
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in by_parent.get(parent, set()):
            if child in descendants:
                continue
            descendants.add(child)
            frontier.append(child)
    return descendants


def _thread_snapshot() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": thread.name,
            "ident": thread.ident,
            "native_id": thread.native_id,
            "daemon": thread.daemon,
            "alive": thread.is_alive(),
        }
        for thread in threading.enumerate()
    )


def _ipc_snapshot() -> tuple[str, ...]:
    root = Path("/dev/shm")
    if not root.is_dir():
        return ()
    try:
        return tuple(sorted(item.name for item in root.iterdir()))
    except PermissionError:
        return ()


class _ChildSubreaper:
    """Temporarily adopt orphaned worker helpers so the controller can reap them."""

    def __init__(self) -> None:
        self.supported = sys.platform.startswith("linux")
        self.initially_enabled: bool | None = None
        self.enabled = False
        self.restored = False
        self.error: str | None = None
        self._libc: Any | None = None
        if not self.supported:
            return
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            current = ctypes.c_int()
            if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(current), 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
            self.initially_enabled = bool(current.value)
            if not self.initially_enabled and libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
            self._libc = libc
            self.enabled = True
        except OSError as caught:
            self.error = str(caught)

    def restore(self) -> None:
        if not self.supported:
            self.restored = True
            return
        if not self.enabled or self._libc is None or self.initially_enabled is None:
            return
        if not self.initially_enabled and self._libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "restoring child-subreaper state failed")
        self.restored = True

    def evidence(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "initially_enabled": self.initially_enabled,
            "enabled_for_experiment": self.enabled,
            "restored_before_return": self.restored,
            "error": self.error,
        }


class _ProcessLifecycleAudit:
    """Track experiment-owned processes without importing a CUDA runtime."""

    def __init__(self) -> None:
        self.subreaper = _ChildSubreaper()
        self.parent_pid = os.getpid()
        self.parent_pgid = os.getpgid(0)
        self.parent_sid = os.getsid(0)
        self.initial_processes = _process_tree()
        self.initial_descendants = _descendant_pids(self.initial_processes, self.parent_pid)
        self.initial_identities = {
            int(item["pid"]): item.get("start_token") for item in self.initial_processes
        }
        self.initial_threads = _thread_snapshot()
        self.initial_ipc = _ipc_snapshot()
        self.worker_roots: dict[int, dict[str, Any]] = {}
        self.owned: dict[int, dict[str, Any]] = {}
        self.reaped: dict[int, dict[str, Any]] = {}
        self.transitions: list[dict[str, Any]] = []
        self.last_observed_ns = 0

    def register_worker(self, role: str, process: subprocess.Popen[bytes]) -> None:
        self.worker_roots[process.pid] = {
            "role": role,
            "pid": process.pid,
            "pgid": os.getpgid(process.pid),
            "sid": os.getsid(process.pid),
        }
        self.observe("worker-spawn", force=True)

    def transition(self, phase: str) -> None:
        index = len(self.transitions)
        if index >= len(PROCESS_LIFECYCLE_PHASES) or PROCESS_LIFECYCLE_PHASES[index] != phase:
            raise RuntimeError(f"invalid process lifecycle transition to {phase}")
        self.transitions.append(
            {
                "phase": phase,
                "observed_at_utc": _utc_now(),
                "observed_at_monotonic_ns": time.monotonic_ns(),
            }
        )

    def observe(self, stage: str, *, force: bool = False) -> tuple[dict[str, Any], ...]:
        now = time.monotonic_ns()
        if not force and now - self.last_observed_ns < 250_000_000:
            return ()
        self.last_observed_ns = now
        processes = _process_tree()
        descendants = _descendant_pids(processes, self.parent_pid)
        worker_descendants = {pid: _descendant_pids(processes, pid) for pid in self.worker_roots}
        worker_groups = {int(item["pgid"]) for item in self.worker_roots.values()}
        worker_sessions = {int(item["sid"]) for item in self.worker_roots.values()}
        observed: list[dict[str, Any]] = []
        for process in processes:
            pid = int(process["pid"])
            start_token = process.get("start_token")
            tracked = self.owned.get(pid)
            if (
                tracked is not None
                and tracked.get("start_token") is not None
                and start_token is not None
                and tracked["start_token"] != start_token
            ):
                continue
            new_controller_descendant = pid in descendants and (
                pid not in self.initial_descendants
                or (
                    self.initial_identities.get(pid) is not None
                    and start_token is not None
                    and self.initial_identities[pid] != start_token
                )
            )
            candidate = (
                pid in self.worker_roots
                or any(pid in values for values in worker_descendants.values())
                or int(process["pgid"]) in worker_groups
                or int(process["sid"]) in worker_sessions
                or new_controller_descendant
                or tracked is not None
            )
            if not candidate or pid == self.parent_pid:
                continue
            role = next(
                (
                    str(root["role"])
                    for root_pid, root in self.worker_roots.items()
                    if pid == root_pid
                    or pid in worker_descendants[root_pid]
                    or int(process["pgid"]) == int(root["pgid"])
                    or int(process["sid"]) == int(root["sid"])
                ),
                "helper",
            )
            command = str(process.get("command", ""))
            process_kind = (
                "profiler"
                if _profiler_process(process)
                else (
                    "multiprocessing_resource_tracker"
                    if RESOURCE_TRACKER_COMMAND_TOKEN in command
                    else ("worker" if pid in self.worker_roots else "helper")
                )
            )
            first_seen = now if tracked is None else int(tracked["first_seen_ns"])
            record = {
                **dict(process),
                "role": role,
                "process_kind": process_kind,
                "cuda_owning_candidate": role in ROLES,
                "first_seen_ns": first_seen,
                "last_seen_ns": now,
                "last_observed_stage": stage,
            }
            self.owned[pid] = record
            observed.append(record)
        return tuple(observed)

    def mark_reaped(
        self,
        pid: int,
        *,
        exit_status: int | None,
        method: str = "popen-waitpid",
    ) -> None:
        self.reaped[pid] = {
            "exit_status": exit_status,
            "reap_method": method,
            "reap_timestamp_utc": _utc_now(),
            "reap_timestamp_monotonic_ns": time.monotonic_ns(),
        }

    def currently_owned(self, *, stage: str) -> tuple[dict[str, Any], ...]:
        self.observe(stage, force=True)
        current = {int(item["pid"]): item for item in _process_tree()}
        rows: list[dict[str, Any]] = []
        for pid, tracked in self.owned.items():
            process = current.get(pid)
            if process is None:
                if pid not in self.reaped:
                    self.mark_reaped(
                        pid,
                        exit_status=None,
                        method="process-absence-confirmed-parent-or-worker-reaped",
                    )
                continue
            if (
                tracked.get("start_token") is not None
                and process.get("start_token") is not None
                and tracked["start_token"] != process["start_token"]
            ):
                continue
            rows.append({**tracked, **process})
        return tuple(sorted(rows, key=lambda item: int(item["pid"])))


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_group_exit(process_group: int, *, deadline_ns: int) -> bool:
    while _process_group_exists(process_group):
        if time.monotonic_ns() >= deadline_ns:
            return False
        time.sleep(0.02)
    return True


def _poll_and_reap_workers(
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    lifecycle: _ProcessLifecycleAudit | None = None,
) -> None:
    """Reap leaders before testing PGID emptiness.

    A dead-but-unreaped session leader keeps ``killpg(pgid, 0)`` observable on
    Linux.  The previous ordering therefore reported a surviving group after
    SIGKILL even though the sole remaining member was a zombie.
    """

    for process in processes:
        returncode = process.poll()
        if returncode is not None and lifecycle is not None and process.pid not in lifecycle.reaped:
            lifecycle.mark_reaped(process.pid, exit_status=returncode)


def _terminate_group(
    process: subprocess.Popen[bytes],
    actions: list[dict[str, Any]],
    *,
    deadline_ns: int,
) -> None:
    """Bound teardown by the same absolute paid-allocation deadline."""

    process_group = process.pid
    _poll_and_reap_workers((process,))
    if process.poll() is not None and not _process_group_exists(process_group):
        return
    actions.append(
        {
            "pid": process.pid,
            "process_group": process_group,
            "signal": "SIGTERM",
            "forced": False,
            "target": "worker-process-group",
            "at_utc": _utc_now(),
        }
    )
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGTERM)
    term_deadline = min(deadline_ns, time.monotonic_ns() + 5_000_000_000)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, (term_deadline - time.monotonic_ns()) / 1e9))
    _poll_and_reap_workers((process,))
    if _wait_for_group_exit(process_group, deadline_ns=term_deadline):
        return
    actions.append(
        {
            "pid": process.pid,
            "process_group": process_group,
            "signal": "SIGKILL",
            "forced": True,
            "target": "worker-process-group",
            "at_utc": _utc_now(),
        }
    )
    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGKILL)
    kill_deadline = min(deadline_ns, time.monotonic_ns() + 3_000_000_000)
    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=max(0.0, (kill_deadline - time.monotonic_ns()) / 1e9))
    _poll_and_reap_workers((process,))
    if not _wait_for_group_exit(process_group, deadline_ns=kill_deadline):
        raise RuntimeError(f"worker process group {process_group} survived SIGKILL")


def _terminate_groups(
    processes: Sequence[subprocess.Popen[bytes]],
    actions: list[dict[str, Any]],
    *,
    deadline_ns: int,
    lifecycle: _ProcessLifecycleAudit | None = None,
) -> None:
    """Signal all role groups in parallel so one child cannot consume teardown."""

    _poll_and_reap_workers(processes, lifecycle=lifecycle)
    groups = {
        process.pid: process
        for process in processes
        if process.poll() is None or _process_group_exists(process.pid)
    }
    for process_group in groups:
        actions.append(
            {
                "pid": process_group,
                "process_group": process_group,
                "signal": "SIGTERM",
                "forced": False,
                "target": "worker-process-group",
                "at_utc": _utc_now(),
            }
        )
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
    term_deadline = min(deadline_ns, time.monotonic_ns() + 5_000_000_000)
    while any(_process_group_exists(group) for group in groups):
        _poll_and_reap_workers(tuple(groups.values()), lifecycle=lifecycle)
        if time.monotonic_ns() >= term_deadline:
            break
        time.sleep(0.02)
    _poll_and_reap_workers(tuple(groups.values()), lifecycle=lifecycle)
    survivors = [group for group in groups if _process_group_exists(group)]
    for process_group in survivors:
        actions.append(
            {
                "pid": process_group,
                "process_group": process_group,
                "signal": "SIGKILL",
                "forced": True,
                "target": "worker-process-group",
                "at_utc": _utc_now(),
            }
        )
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGKILL)
    while any(_process_group_exists(group) for group in survivors):
        _poll_and_reap_workers(tuple(groups.values()), lifecycle=lifecycle)
        if time.monotonic_ns() >= deadline_ns:
            remaining = [group for group in survivors if _process_group_exists(group)]
            raise RuntimeError(f"worker process groups survived SIGKILL: {remaining}")
        time.sleep(0.02)
    for process in groups.values():
        if process.poll() is None:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.0, (deadline_ns - time.monotonic_ns()) / 1e9))
        _poll_and_reap_workers((process,), lifecycle=lifecycle)


def _signal_exact_owned(
    processes: Sequence[Mapping[str, Any]],
    sent_signal: signal.Signals,
    actions: list[dict[str, Any]],
) -> None:
    """Signal tracked escaped helpers by exact, still-matching PID identity."""

    current = {int(item["pid"]): item for item in _process_tree()}
    for tracked in sorted(processes, key=lambda item: int(item["pid"]), reverse=True):
        pid = int(tracked["pid"])
        observed = current.get(pid)
        if observed is None:
            continue
        if (
            tracked.get("start_token") is not None
            and observed.get("start_token") is not None
            and tracked["start_token"] != observed["start_token"]
        ):
            continue
        actions.append(
            {
                "pid": pid,
                "process_group": int(observed["pgid"]),
                "signal": sent_signal.name,
                "forced": sent_signal == signal.SIGKILL,
                "target": "exact-owned-pid",
                "role": tracked.get("role", "helper"),
                "at_utc": _utc_now(),
            }
        )
        with suppress(ProcessLookupError):
            os.kill(pid, sent_signal)


def _wait_for_owned_exit(
    lifecycle: _ProcessLifecycleAudit,
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    deadline_ns: int,
    stage: str,
) -> tuple[dict[str, Any], ...]:
    while True:
        _poll_and_reap_workers(processes, lifecycle=lifecycle)
        survivors = lifecycle.currently_owned(stage=stage)
        if not survivors or time.monotonic_ns() >= deadline_ns:
            return survivors
        time.sleep(0.02)


def _reap_owned_children(
    lifecycle: _ProcessLifecycleAudit,
    processes: Sequence[subprocess.Popen[bytes]],
    *,
    deadline_ns: int,
) -> tuple[dict[str, Any], ...]:
    """Reap worker leaders and subreaper-adopted descendants within the bound."""

    leaders = {process.pid: process for process in processes}
    while time.monotonic_ns() < deadline_ns:
        _poll_and_reap_workers(processes, lifecycle=lifecycle)
        progress = False
        for pid in sorted(lifecycle.owned):
            if pid in leaders or pid in lifecycle.reaped:
                continue
            try:
                reaped_pid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if reaped_pid:
                lifecycle.mark_reaped(
                    reaped_pid,
                    exit_status=os.waitstatus_to_exitcode(status),
                    method="controller-waitpid-adopted-child",
                )
                progress = True
        survivors = lifecycle.currently_owned(stage="child-reap")
        if not survivors:
            return ()
        if not progress:
            time.sleep(0.02)
    return lifecycle.currently_owned(stage="child-reap-deadline")


def _profiler_process(process: Mapping[str, Any]) -> bool:
    words = str(process.get("command", "")).split()
    executables = {Path(word).name.lower() for word in words}
    return bool(executables & set(PROFILER_COMMAND_TOKENS))


def _owned_child_evidence(
    lifecycle: _ProcessLifecycleAudit,
    actions: Sequence[Mapping[str, Any]],
    processes: Sequence[subprocess.Popen[bytes]],
) -> list[dict[str, Any]]:
    leaders = {process.pid: process for process in processes}
    rows: list[dict[str, Any]] = []
    for pid, tracked in sorted(lifecycle.owned.items()):
        matching_actions = [
            action
            for action in actions
            if int(action.get("pid", -1)) == pid
            or int(action.get("process_group", -1)) == int(tracked["pgid"])
        ]
        last_signal = matching_actions[-1]["signal"] if matching_actions else None
        reap = lifecycle.reaped.get(pid, {})
        leader = leaders.get(pid)
        rows.append(
            {
                "pid": pid,
                "ppid": int(tracked["ppid"]),
                "pgid": int(tracked["pgid"]),
                "sid": int(tracked["sid"]),
                "role": tracked["role"],
                "process_kind": tracked["process_kind"],
                "command": tracked["command"],
                "cuda_owning_candidate": tracked["cuda_owning_candidate"],
                "termination_signal": last_signal,
                "exit_status": (
                    leader.returncode if leader is not None else reap.get("exit_status")
                ),
                "reap_timestamp_utc": reap.get("reap_timestamp_utc"),
                "reap_timestamp_monotonic_ns": reap.get("reap_timestamp_monotonic_ns"),
                "reap_method": reap.get("reap_method"),
            }
        )
    return rows


def _build_in_function_cleanup(
    *,
    lifecycle: _ProcessLifecycleAudit,
    actions: Sequence[Mapping[str, Any]],
    processes: Sequence[subprocess.Popen[bytes]],
    surviving_children: Sequence[Mapping[str, Any]],
    surviving_groups: Sequence[int],
    compute_processes_after: Sequence[Mapping[str, Any]],
    cuda_release_verified: bool,
    pipes_closed: bool,
    cleanup_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_threads = _thread_snapshot()
    initial_thread_keys = {
        (item["ident"], item["native_id"], item["name"]) for item in lifecycle.initial_threads
    }
    leaked_threads = [
        item
        for item in final_threads
        if item["alive"]
        and (item["ident"], item["native_id"], item["name"]) not in initial_thread_keys
    ]
    final_ipc = _ipc_snapshot()
    leaked_ipc = sorted(set(final_ipc) - set(lifecycle.initial_ipc))
    profilers = [item for item in surviving_children if _profiler_process(item)]
    resource_trackers = [
        item
        for item in surviving_children
        if RESOURCE_TRACKER_COMMAND_TOKEN in str(item.get("command", ""))
    ]
    zombies = [item for item in surviving_children if "Z" in str(item.get("state", ""))]
    forced_kills = [item for item in actions if item.get("signal") == "SIGKILL"]
    phases = [str(item["phase"]) for item in lifecycle.transitions]
    subreaper = getattr(lifecycle, "subreaper", None)
    subreaper_evidence = (
        subreaper.evidence()
        if subreaper is not None
        else {
            "supported": False,
            "initially_enabled": None,
            "enabled_for_experiment": False,
            "restored_before_return": True,
            "error": None,
        }
    )
    subreaper_pass = bool(
        subreaper_evidence["error"] is None
        and (
            not subreaper_evidence["supported"]
            or (
                subreaper_evidence["enabled_for_experiment"]
                and subreaper_evidence["restored_before_return"]
            )
        )
    )
    passed = bool(
        phases == list(PROCESS_LIFECYCLE_PHASES)
        and not surviving_children
        and not surviving_groups
        and cuda_release_verified
        and not compute_processes_after
        and not profilers
        and not resource_trackers
        and not zombies
        and not leaked_threads
        and not leaked_ipc
        and subreaper_pass
        and pipes_closed
        and not cleanup_errors
        and all(process.returncode is not None for process in processes)
    )
    return {
        "schema_version": "sloforge.branchfabric.in-function-cleanup/v1",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "parent_pid": lifecycle.parent_pid,
        "parent_pgid": lifecycle.parent_pgid,
        "parent_sid": lifecycle.parent_sid,
        "child_subreaper": subreaper_evidence,
        "lifecycle": lifecycle.transitions,
        "required_lifecycle": list(PROCESS_LIFECYCLE_PHASES),
        "owned_children": _owned_child_evidence(lifecycle, actions, processes),
        "termination_actions": list(actions),
        "forced_kills": forced_kills,
        "forced_kill_required": bool(forced_kills),
        "surviving_children": list(surviving_children),
        "surviving_process_groups": list(surviving_groups),
        "profiler_processes_after": profilers,
        "serving_workers_after": [
            item for item in surviving_children if item.get("role") == "serving"
        ],
        "rollout_workers_after": [
            item for item in surviving_children if item.get("role") == "rollout"
        ],
        "resource_tracker_processes_after": resource_trackers,
        "zombie_processes_after": zombies,
        "parent_reaped_all_owned_children": not surviving_children
        and all(process.returncode is not None for process in processes),
        "owned_ipc_resources_released": not leaked_ipc,
        "initial_ipc_resources": list(lifecycle.initial_ipc),
        "final_ipc_resources": list(final_ipc),
        "leaked_ipc_resources": leaked_ipc,
        "initial_threads": list(lifecycle.initial_threads),
        "final_threads": list(final_threads),
        "leaked_threads": leaked_threads,
        "pipes_closed": pipes_closed,
        "compute_processes_after": list(compute_processes_after),
        "cuda_released": cuda_release_verified and not compute_processes_after,
        "cleanup_errors": list(cleanup_errors),
        "recorded_at_utc": _utc_now(),
        "recorded_at_monotonic_ns": time.monotonic_ns(),
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _status_is_pass(value: Any) -> bool:
    return isinstance(value, str) and (
        value.upper() in {"PASS", "PASSED", "MICRO_VALIDATION_PASS"}
        or value.upper().startswith("PASS_")
    )


def _review_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = manifest.get("reviews", manifest.get("post_micro_reviews"))
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise RuntimeError("post-micro manifest must contain a review list")
    return tuple(raw)


def _agent_number(entry: Mapping[str, Any]) -> int | None:
    raw = entry.get("agent", entry.get("agent_id", entry.get("reviewer")))
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        digits = "".join(character for character in raw if character.isdigit())
        if digits:
            return int(digits)
    return None


def _verify_final_pre_gpu_gate(
    manifest: Mapping[str, Any], repository_root: Path, *, attempt_id: str
) -> dict[str, Any]:
    """Verify the outer, post-fix eight-review gate for the fresh retry.

    Review artifacts cannot hash a manifest that already contains their own
    hashes without creating a cycle.  The config therefore seals this outer
    post-micro manifest, which in turn seals each review and the fresh
    ``make check`` evidence.  No review is accepted by filename alone.
    """

    gate = manifest.get("final_integrated_pre_gpu_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("status") != "PASS"
        or gate.get("attempt_id") != attempt_id
    ):
        raise RuntimeError("fresh integrated retry lacks its final pre-GPU PASS gate")
    reviews = gate.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(FINAL_PRE_GPU_REVIEW_ROLES):
        raise RuntimeError("fresh integrated retry does not seal all eight pre-GPU reviews")
    by_agent: dict[int, Mapping[str, Any]] = {}
    verified: list[dict[str, Any]] = []
    for entry in reviews:
        if not isinstance(entry, dict):
            raise RuntimeError("fresh integrated retry review binding is malformed")
        agent = _agent_number(entry)
        if agent is None or agent in by_agent:
            raise RuntimeError("fresh integrated retry review agents are invalid or duplicated")
        by_agent[agent] = entry
    if set(by_agent) != set(FINAL_PRE_GPU_REVIEW_ROLES):
        raise RuntimeError("fresh integrated retry review coverage is incomplete")
    for agent, expected_role in FINAL_PRE_GPU_REVIEW_ROLES.items():
        entry = by_agent[agent]
        reference = entry.get("artifact")
        expected_hash = entry.get("sha256")
        if (
            entry.get("role") != expected_role
            or entry.get("status") != "PASS"
            or not isinstance(reference, str)
            or not isinstance(expected_hash, str)
        ):
            raise RuntimeError(f"fresh integrated retry Agent {agent} binding is not PASS")
        path = validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_hash,
        )
        evidence = _load_json_object(path, label=f"fresh pre-GPU Agent {agent} evidence")
        expected_role_tokens = {
            token for token in expected_role.split("-") if token not in {"implementation", "review"}
        }
        evidence_role_tokens = {
            token
            for token in "".join(
                character.lower() if character.isascii() and character.isalnum() else " "
                for character in (
                    str(evidence.get("role", "")) + " " + str(evidence.get("schema_version", ""))
                )
            ).split()
        }
        if (
            evidence.get("status") != "PASS"
            or _agent_number(evidence) != agent
            or not expected_role_tokens <= evidence_role_tokens
            or evidence.get("gpu_or_cloud_invoked") is True
            or evidence.get("gpu_resources_consumed") is True
        ):
            raise RuntimeError(f"fresh integrated retry Agent {agent} evidence is not PASS offline")
        verified.append(
            {
                "agent": agent,
                "role": expected_role,
                "artifact": reference,
                "sha256": expected_hash,
            }
        )
    make_check = gate.get("make_check")
    if (
        not isinstance(make_check, dict)
        or make_check.get("status") != "PASS"
        or make_check.get("command") != "make check"
        or not isinstance(make_check.get("artifact"), str)
        or not isinstance(make_check.get("sha256"), str)
    ):
        raise RuntimeError("fresh integrated retry lacks fresh make check PASS evidence")
    make_check_path = validate_bound_artifact(
        repository_root,
        reference=str(make_check["artifact"]),
        expected_sha256=str(make_check["sha256"]),
    )
    make_check_evidence = _load_json_object(
        make_check_path, label="fresh integrated retry make-check evidence"
    )
    if (
        make_check_evidence.get("status") != "PASS"
        or make_check_evidence.get("command") != "make check"
    ):
        raise RuntimeError("fresh integrated retry make check payload is not PASS")
    return {
        "schema_version": "sloforge.branchfabric.exp004-v11-final-pre-gpu-verification/v1",
        "status": "PASS",
        "attempt_id": attempt_id,
        "reviews": verified,
        "make_check": {
            "artifact": make_check["artifact"],
            "sha256": make_check["sha256"],
        },
    }


def _exact_zero_int(value: Any) -> bool:
    return type(value) is int and value == 0


def _source_capture_semantic_sha256(commit: Mapping[str, Any]) -> str:
    branches = commit.get("branches")
    allocations = commit.get("allocations")
    if not isinstance(branches, list) or not isinstance(allocations, list):
        raise RuntimeError("post-target SourceCaptureCommit tables are malformed")
    branch_rows: list[tuple[Any, ...]] = []
    for branch in branches:
        if not isinstance(branch, dict):
            raise RuntimeError("post-target SourceCaptureCommit branch is malformed")
        branch_rows.append(
            (
                branch.get("logical_branch_id"),
                branch.get("parent_logical_branch_id"),
                branch.get("runtime_request_id"),
                branch.get("computed_tokens"),
                branch.get("token_history_sha256"),
                branch.get("logical_page_ids"),
            )
        )
    allocation_rows: list[tuple[Any, ...]] = []
    for allocation in allocations:
        if not isinstance(allocation, dict):
            raise RuntimeError("post-target SourceCaptureCommit allocation is malformed")
        allocation_rows.append(
            (
                allocation.get("logical_page_id"),
                allocation.get("block_table_slot"),
                allocation.get("logical_token_start"),
                allocation.get("logical_token_end"),
                allocation.get("valid_tokens"),
                allocation.get("physical_block_id"),
                allocation.get("allocation_epoch"),
                allocation.get("logical_owner_set"),
                allocation.get("runtime_owner_set"),
                allocation.get("expected_refcount"),
                allocation.get("shared"),
                allocation.get("physical_bytes"),
            )
        )
    payload = {
        "schema_version": commit.get("schema_version"),
        "runtime_instance_id": commit.get("runtime_instance_id"),
        "runtime_model_identity": commit.get("runtime_model_identity"),
        "device": commit.get("device"),
        "root_session_id": commit.get("root_session_id"),
        "branch_group": commit.get("branch_group"),
        "branches": branch_rows,
        "allocations": allocation_rows,
        "logical_state_bytes": commit.get("logical_state_bytes"),
        "physical_source_bytes": commit.get("physical_source_bytes"),
        "gate_binding_id": commit.get("gate_binding_id"),
        "gate_acquisition_generation": commit.get("gate_acquisition_generation"),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _verify_post_target_source_commit(
    envelope: Mapping[str, Any],
    *,
    expected_semantic_sha256: str,
    config: Any,
    expected_attempt_id: str = RETRY_TARGETED_IDENTITY_ATTEMPT_ID,
    expected_runtime_instance_id: str = POST_TARGET_RUNTIME_INSTANCE_ID,
    identity_field: str = "identity_validation",
    expected_gate_suffix: str = "targeted-source-identity",
) -> dict[str, Any]:
    commit = envelope.get("commit")
    validation = envelope.get(identity_field)
    notifications = envelope.get("post_commit_allocation_notifications")
    lifecycle = envelope.get("allocator_lifecycle")
    pre_commit_history = envelope.get("pre_commit_allocation_history")
    if (
        envelope.get("schema_version") != "sloforge.branchfabric.v11-source-capture-commit/v1"
        or envelope.get("event") != "CAPTURE_COMMIT_END"
        or envelope.get("passed") is not True
        or not isinstance(commit, dict)
        or not isinstance(validation, dict)
        or not isinstance(notifications, dict)
        or not isinstance(lifecycle, dict)
        or not isinstance(pre_commit_history, dict)
    ):
        raise RuntimeError("post-target SourceCaptureCommit envelope is inconsistent")
    allocations = commit.get("allocations")
    branches = commit.get("branches")
    branch_group = commit.get("branch_group")
    runtime_identity = commit.get("runtime_model_identity")
    quiescence = commit.get("allocator_quiescence")
    if (
        commit.get("schema_version") != "sloforge.continuum.vllm-v11-source-capture-commit/v1"
        or commit.get("semantic_sha256") != expected_semantic_sha256
        or _source_capture_semantic_sha256(commit) != expected_semantic_sha256
        or commit.get("runtime_instance_id") != expected_runtime_instance_id
        or commit.get("device") != "cuda:0"
        or commit.get("root_session_id") != f"{expected_attempt_id}-root"
        or not isinstance(allocations, list)
        or len(allocations) != 1152
        or not isinstance(branches, list)
        or len(branches) != 8
        or not isinstance(branch_group, list)
        or branch_group != [f"branch.{index}" for index in range(8)]
        or not isinstance(runtime_identity, list)
        or not isinstance(quiescence, dict)
    ):
        raise RuntimeError("post-target SourceCaptureCommit semantics are inconsistent")
    runtime_identity_sha = hashlib.sha256(_canonical_bytes(runtime_identity)).hexdigest()
    expected_runtime_identity = {
        "adapter_version": "1.0.0",
        "device": "cuda:0",
        "dtype": config.dtype,
        "model_id": config.model,
        "model_revision": config.model_revision,
        "policy_epoch": "branchfabric-modal-exp004-policy-v1",
        "runtime": config.runtime,
        "runtime_version": config.runtime_version,
        "tokenizer_id": config.model,
        "tokenizer_revision": config.tokenizer_revision,
    }
    if (
        commit.get("runtime_model_identity_sha256") != runtime_identity_sha
        or len(runtime_identity) != len(expected_runtime_identity)
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(item, str) for item in row)
            for row in runtime_identity
        )
        or dict(runtime_identity) != expected_runtime_identity
        or {row[0] for row in runtime_identity} != set(expected_runtime_identity)
    ):
        raise RuntimeError("post-target runtime model identity hash is inconsistent")
    runtime_identity_map = dict(runtime_identity)
    if runtime_identity_map.get("device") != commit.get("device"):
        raise RuntimeError("post-target runtime model device identity is inconsistent")

    branch_by_id: dict[str, Mapping[str, Any]] = {}
    runtime_request_ids: set[str] = set()
    shared_page_ids = [f"logical-page-{index:06d}" for index in range(1024)]
    for branch_index, branch in enumerate(branches):
        if not isinstance(branch, dict) or type(branch.get("computed_tokens")) is not int:
            raise RuntimeError("post-target SourceCaptureCommit branch semantics are malformed")
        branch_id = branch.get("logical_branch_id")
        request_id = branch.get("runtime_request_id")
        page_ids = branch.get("logical_page_ids")
        if (
            not isinstance(branch_id, str)
            or branch_id != f"branch.{branch_index}"
            or branch_id in branch_by_id
            or not isinstance(request_id, str)
            or not request_id
            or request_id in runtime_request_ids
            or branch.get("parent_logical_branch_id") != f"{expected_attempt_id}-root"
            or branch.get("computed_tokens") != 16640
            or not isinstance(page_ids, list)
            or len(page_ids) != 1040
            or len(page_ids) != len(set(page_ids))
            or page_ids
            != shared_page_ids
            + [f"logical-page-{1024 + branch_index + 8 * index:06d}" for index in range(16)]
        ):
            raise RuntimeError("post-target SourceCaptureCommit branch mapping is malformed")
        branch_by_id[branch_id] = branch
        runtime_request_ids.add(request_id)
    if set(branch_by_id) != set(branch_group):
        raise RuntimeError("post-target SourceCaptureCommit branch coverage is incomplete")

    expected_logical_pages = {f"logical-page-{index:06d}" for index in range(1152)}
    seen_logical_pages: set[str] = set()
    seen_blocks: set[int] = set()
    seen_block_epochs: set[tuple[int, int]] = set()
    physical_bytes = 0
    for allocation in allocations:
        if not isinstance(allocation, dict):
            raise RuntimeError("post-target SourceCaptureCommit allocation is malformed")
        logical_page = allocation.get("logical_page_id")
        physical_block = allocation.get("physical_block_id")
        epoch = allocation.get("allocation_epoch")
        slot = allocation.get("block_table_slot")
        token_start = allocation.get("logical_token_start")
        token_end = allocation.get("logical_token_end")
        valid_tokens = allocation.get("valid_tokens")
        bytes_for_page = allocation.get("physical_bytes")
        logical_owners = allocation.get("logical_owner_set")
        runtime_owners = allocation.get("runtime_owner_set")
        refcount = allocation.get("expected_refcount")
        shared = allocation.get("shared")
        integer_values = (
            physical_block,
            epoch,
            slot,
            token_start,
            token_end,
            valid_tokens,
            bytes_for_page,
            refcount,
        )
        if (
            not isinstance(logical_page, str)
            or logical_page in seen_logical_pages
            or any(type(value) is not int for value in integer_values)
            or int(physical_block) < 0
            or int(epoch) <= 0
            or int(slot) < 0
            or int(valid_tokens) <= 0
            or int(token_start) != int(slot) * 16
            or int(token_end) != int(token_start) + int(valid_tokens)
            or int(bytes_for_page) <= 0
            or not isinstance(logical_owners, list)
            or not isinstance(runtime_owners, list)
            or type(shared) is not bool
        ):
            raise RuntimeError("post-target SourceCaptureCommit allocation semantics are invalid")
        expected_owners = sorted(
            branch_id
            for branch_id, branch in branch_by_id.items()
            if logical_page in branch["logical_page_ids"]
        )
        expected_runtime_owners = sorted(
            str(branch_by_id[branch_id]["runtime_request_id"]) for branch_id in expected_owners
        )
        slots = {
            list(branch_by_id[branch_id]["logical_page_ids"]).index(logical_page)
            for branch_id in expected_owners
        }
        if (
            not expected_owners
            or slots != {slot}
            or logical_owners != expected_owners
            or runtime_owners != expected_runtime_owners
            or refcount != len(expected_owners)
            or shared is not (len(expected_owners) > 1)
        ):
            raise RuntimeError("post-target SourceCaptureCommit owner/refcount proof failed")
        if physical_block in seen_blocks or (physical_block, epoch) in seen_block_epochs:
            raise RuntimeError("post-target SourceCaptureCommit allocation identity is aliased")
        seen_logical_pages.add(logical_page)
        seen_blocks.add(physical_block)
        seen_block_epochs.add((physical_block, epoch))
        physical_bytes += bytes_for_page
    if (
        seen_logical_pages != expected_logical_pages
        or physical_bytes != commit.get("logical_state_bytes")
        or physical_bytes != commit.get("physical_source_bytes")
    ):
        raise RuntimeError("post-target SourceCaptureCommit allocation coverage is incomplete")

    semantic_true_fields = (
        "same_engine_step_witness",
        "exact_logical_mapping",
        "exact_block_epoch_identity",
        "exact_owner_sets",
        "exact_refcounts",
        "all_allocations_live",
        "no_post_commit_mutation",
        "passed",
    )
    if (
        validation.get("schema_version")
        != "sloforge.continuum.vllm-v11-source-identity-validation/v1"
        or validation.get("semantic_sha256") != expected_semantic_sha256
        or type(validation.get("allocation_count")) is not int
        or validation.get("allocation_count") != 1152
        or any(validation.get(field) is not True for field in semantic_true_fields)
    ):
        raise RuntimeError("post-target SourceCaptureCommit identity validation failed")
    if (
        notifications.get("schema_version")
        != "sloforge.branchfabric.v11-post-commit-allocation-notification/v1"
        or notifications.get("source_capture_commit_sha256") != expected_semantic_sha256
        or notifications.get("queue_empty") is not True
        or notifications.get("passed") is not True
        or notifications.get("semantic_identity_claimed") is not False
        or not _exact_zero_int(notifications.get("observed_event_count"))
        or notifications.get("observed_block_ids") != []
    ):
        raise RuntimeError("post-target SourceCaptureCommit has post-commit mutations")
    history_block_ids = pre_commit_history.get("history_block_ids")
    history_sha = pre_commit_history.get("history_block_ids_sha256")
    if (
        pre_commit_history.get("schema_version")
        != "sloforge.branchfabric.v11-allocation-history-retirement/v1"
        or pre_commit_history.get("passed") is not True
        or pre_commit_history.get("destructive_drain_performed") is not True
        or pre_commit_history.get("retired_under_export_capture") is not True
        or pre_commit_history.get("semantic_identity_claimed") is not False
        or pre_commit_history.get("needs_kv_cache_zeroing") is not False
        or pre_commit_history.get("capture_manifest_sha256") is not None
        or pre_commit_history.get("gate_operation") != "EXPORT_CAPTURE"
        or pre_commit_history.get("gate_id") != f"{expected_attempt_id}:{expected_gate_suffix}"
        or pre_commit_history.get("gate_binding_id") != commit.get("gate_binding_id")
        or not isinstance(history_block_ids, list)
        or len(history_block_ids) != 2210
        or history_block_ids != list(range(1, 2211))
        or type(pre_commit_history.get("history_event_count")) is not int
        or pre_commit_history.get("history_event_count") != 2210
        or type(pre_commit_history.get("observed_count")) is not int
        or pre_commit_history.get("observed_count") != 2210
        or type(pre_commit_history.get("unique_block_count")) is not int
        or pre_commit_history.get("unique_block_count") != 2210
        or not _exact_zero_int(pre_commit_history.get("duplicate_event_count"))
        or not _exact_zero_int(pre_commit_history.get("post_commit_event_count"))
        or pre_commit_history.get("queue_empty_after_retirement") is not True
        or not isinstance(history_sha, str)
        or history_sha != hashlib.sha256(_canonical_bytes(history_block_ids)).hexdigest()
        or any(
            not isinstance(pre_commit_history.get(field), str)
            or len(pre_commit_history[field]) != 64
            for field in ("ownership_snapshot_sha256", "source_allocation_lifetime_sha256")
        )
    ):
        raise RuntimeError("post-target allocation-history retirement proof failed")
    if (
        quiescence.get("passed") is not True
        or quiescence.get("engine_step_excluded") is not True
        or quiescence.get("allocator_mutations_excluded") is not True
        or quiescence.get("asynchronous_scheduling") is not False
        or not _exact_zero_int(quiescence.get("scheduler_waiting_count"))
        or not _exact_zero_int(quiescence.get("scheduler_skipped_waiting_count"))
        or type(quiescence.get("paused_branch_count")) is not int
        or quiescence.get("paused_branch_count") != 8
        or quiescence.get("gate_binding_id") != commit.get("gate_binding_id")
        or quiescence.get("gate_acquisition_generation")
        != commit.get("gate_acquisition_generation")
    ):
        raise RuntimeError("post-target allocator quiescence proof failed")
    events = lifecycle.get("events")
    start_sequence = lifecycle.get("start_sequence")
    rollout_ready_sequence = lifecycle.get("rollout_ready_sequence")
    commit_sequence = lifecycle.get("capture_commit_sequence")
    lifecycle_without_sha = {key: value for key, value in lifecycle.items() if key != "sha256"}
    if (
        lifecycle.get("schema_version")
        != "sloforge.branchfabric.v11-allocator-lifecycle-journal/v1"
        or type(lifecycle.get("bounded_maximum_event_count")) is not int
        or lifecycle.get("bounded_maximum_event_count") != 262144
        or not isinstance(events, list)
        or not events
        or type(start_sequence) is not int
        or type(rollout_ready_sequence) is not int
        or type(commit_sequence) is not int
        or type(lifecycle.get("event_count")) is not int
        or lifecycle.get("event_count") != len(events)
        or type(lifecycle.get("source_lifetime_event_count")) is not int
        or any(not isinstance(event, dict) for event in events)
        or lifecycle.get("source_lifetime_event_count")
        != sum(event.get("logical_page_id") is not None for event in events)
        or start_sequence != events[0].get("sequence")
        or rollout_ready_sequence != commit_sequence
        or start_sequence + len(events) != commit_sequence
        or [event.get("sequence") for event in events]
        != list(range(start_sequence, commit_sequence))
        or any(type(event.get("observed_at_monotonic_ns")) is not int for event in events)
        or any(
            left.get("observed_at_monotonic_ns") >= right.get("observed_at_monotonic_ns")
            for left, right in pairwise(events)
        )
        or lifecycle.get("sha256")
        != hashlib.sha256(_canonical_bytes(lifecycle_without_sha)).hexdigest()
        or Counter(event.get("event") for event in events)
        != Counter(
            {
                "BIND_BLOCK_TABLE": 9344,
                "OWNER_ADD": 9344,
                "INC_REF": 9344,
                "EPOCH_ISSUE": 1152,
                "ALLOC": 1152,
                "UNBIND_BLOCK_TABLE": 1024,
                "OWNER_REMOVE": 1024,
                "DEC_REF": 1024,
                "CACHE_RELEASE_PENDING": 1024,
                "REQUEST_FINISH": 1,
            }
        )
        or Counter(event.get("phase") for event in events)
        != Counter({"ROLLOUT_CREATE_TO_READY": 34433})
    ):
        raise RuntimeError("post-target allocator journal crosses SourceCaptureCommit")
    return {
        "semantic_sha256": expected_semantic_sha256,
        "allocation_count": 1152,
        "allocation_history_sha256": history_sha,
        "zero_post_commit_events": True,
    }


def _zero_runtime_queue_state(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "queue_depth",
            "request_count",
            "running_requests",
            "skipped_waiting_requests",
            "waiting_requests",
        }
        and all(_exact_zero_int(item) for item in value.values())
    )


def _verify_post_target_runtime_evidence(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    commit_envelope: Mapping[str, Any],
    gate: Mapping[str, Any],
    in_function_cleanup: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    trigger = evidence["trigger"]
    admission_stop = evidence["admission_stop"]
    controller = evidence["controller_result"]
    serving = evidence["serving_result"]
    rollout = evidence["rollout_result"]
    trigger_events = trigger.get("events")
    if (
        trigger.get("schema_version") != "sloforge.branchfabric.reclamation-trigger-evidence/v2"
        or trigger.get("controller") != "integrated-v11-early-trigger-overlay"
        or trigger.get("overload_confirmed") is not True
        or trigger.get("material_service_deficit") is not True
        or trigger.get("positive_queue_slope") is not True
        or trigger.get("offered_rate_exceeds_completed_rate") is not True
        or type(trigger.get("queue_trigger")) is not int
        or trigger.get("queue_trigger") != config.overload_queue_trigger
        or type(trigger.get("queue_abort")) is not int
        or trigger.get("queue_abort") != config.overload_queue_abort
        or type(trigger.get("queue_depth_at_trigger")) is not int
        or trigger.get("queue_depth_at_trigger") != 21
        or type(trigger.get("emergency_ceiling_headroom_requests")) is not int
        or trigger.get("emergency_ceiling_headroom_requests") != 43
        or not isinstance(trigger_events, dict)
    ):
        raise RuntimeError("post-target raw reclaim trigger is inconsistent")
    overload_ns = trigger_events.get("OVERLOAD_DETECTED")
    predicate_ns = trigger_events.get("TRIGGER_PREDICATE_SATISFIED")
    emitted_ns = trigger_events.get("RECLAIM_TRIGGER_EMITTED")
    if (
        any(type(value) is not int for value in (overload_ns, predicate_ns, emitted_ns))
        or not overload_ns < predicate_ns <= emitted_ns
        or trigger.get("overload_detected_ns") != overload_ns
        or trigger.get("trigger_predicate_satisfied_ns") != predicate_ns
        or trigger.get("reclaim_trigger_emitted_ns") != emitted_ns
        or trigger.get("triggered_ns") != emitted_ns
        or type(trigger.get("controller_reaction_latency_ns")) is not int
        or trigger.get("controller_reaction_latency_ns") != emitted_ns - predicate_ns
        or trigger.get("controller_reaction_latency_ns") > MAX_RECLAIM_TRIGGER_REACTION_NS
    ):
        raise RuntimeError("post-target raw reclaim trigger timestamps are inconsistent")
    admission_ns = admission_stop.get("observed_ns")
    if (
        admission_stop.get("schema_version")
        != "sloforge.branchfabric.v11-rollout-admission-stop/v1"
        or admission_stop.get("event") != "ROLLOUT_ADMISSION_STOP"
        or admission_stop.get("reclaim_trigger_emitted_ns") != emitted_ns
        or type(admission_ns) is not int
        or admission_ns <= emitted_ns
    ):
        raise RuntimeError("post-target rollout admission-stop evidence is inconsistent")

    common_result = {
        "status": "succeeded",
        "attempt_id": RETRY_TARGETED_IDENTITY_ATTEMPT_ID,
        "terminal_phase": "SOURCE_ALLOCATION_IDENTITY_GATE",
        "optimized_export_started": False,
        "transaction_source_release_started": False,
    }
    if (
        serving.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-targeted-gpu0-result/v1"
        or serving.get("role") != "serving"
        or any(serving.get(key) != value for key, value in common_result.items())
        or serving.get("producer_stopped") is not True
        or serving.get("reclamation_trigger_evidence") != trigger
        or serving.get("source_identity_terminal") != gate
        or type(serving.get("trigger_backlog_requests")) is not int
        or serving.get("trigger_backlog_requests") != 21
        or type(serving.get("trigger_emergency_ceiling_headroom_requests")) is not int
        or serving.get("trigger_emergency_ceiling_headroom_requests") != 43
        or type(serving.get("maximum_outstanding_requests")) is not int
        or serving.get("maximum_outstanding_requests") != 29
        or type(serving.get("minimum_emergency_ceiling_headroom_requests")) is not int
        or serving.get("minimum_emergency_ceiling_headroom_requests") != 35
        or not _zero_runtime_queue_state(serving.get("final_runtime_queue_state"))
    ):
        raise RuntimeError("post-target raw serving result is inconsistent")
    serving_trace = serving.get("targeted_serving_trace")
    queue_drain = None if not isinstance(serving_trace, dict) else serving_trace.get("queue_drain")
    trace_requests = None if not isinstance(serving_trace, dict) else serving_trace.get("requests")
    if (
        not isinstance(serving_trace, dict)
        or serving_trace.get("schema_version")
        != "sloforge.branchfabric.v11-targeted-gpu0-source-identity/v1"
        or serving_trace.get("passed") is not True
        or serving_trace.get("reclamation_trigger_evidence") != trigger
        or serving_trace.get("source_identity_gate") != gate
        or not isinstance(trace_requests, list)
        or not any(request.get("phase") == "control" for request in trace_requests)
        or not any(request.get("phase") == "gpu0-overload" for request in trace_requests)
        or not isinstance(queue_drain, dict)
        or queue_drain.get("passed") is not True
        or not _exact_zero_int(queue_drain.get("driver_active_requests"))
        or not _exact_zero_int(queue_drain.get("external_gpu0_queue_size"))
        or not _zero_runtime_queue_state(queue_drain.get("runtime_queue_state"))
    ):
        raise RuntimeError("post-target raw serving lifecycle is inconsistent")

    commit = commit_envelope["commit"]
    identity_validation = commit_envelope["identity_validation"]
    history = commit_envelope["pre_commit_allocation_history"]
    lifecycle = commit_envelope["allocator_lifecycle"]
    notifications = commit_envelope["post_commit_allocation_notifications"]
    trigger_timeline = rollout.get("trigger_timeline")
    engine_binding = rollout.get("engine_step_binding")
    topology = rollout.get("topology")
    timings = rollout.get("timings")
    if (
        rollout.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-targeted-gpu1-result/v1"
        or rollout.get("role") != "rollout"
        or any(rollout.get(key) != value for key, value in common_result.items())
        or rollout.get("source_capture_commit") != commit
        or rollout.get("source_identity_gate") != gate
        or rollout.get("source_identity_validation") != identity_validation
        or rollout.get("allocation_history_retirement") != history
        or rollout.get("allocator_lifecycle") != lifecycle
        or rollout.get("post_commit_allocation_notifications") != notifications
        or not isinstance(topology, dict)
        or topology
        != {
            "branch_count": 8,
            "logical_state_bytes": 1056964608,
            "private_blocks": 128,
            "shared_blocks": 1024,
            "total_blocks": 1152,
        }
        or not isinstance(trigger_timeline, dict)
        or trigger_timeline.get("passed") is not True
        or trigger_timeline.get("trigger_precedes_state_quiescence") is not True
        or not isinstance(engine_binding, dict)
        or engine_binding.get("active") is not True
        or engine_binding.get("wrappers_intact") is not True
        or engine_binding.get("asynchronous_scheduling") is not False
        or engine_binding.get("runtime_version") != config.runtime_version
        or engine_binding.get("binding_id") != commit.get("gate_binding_id")
        or not isinstance(timings, dict)
    ):
        raise RuntimeError("post-target raw rollout result is inconsistent")
    timeline_events = trigger_timeline.get("events")
    state_quiescence_ns = (
        None if not isinstance(timeline_events, dict) else timeline_events.get("STATE_QUIESCENCE")
    )
    allocator_quiescence_ns = commit["allocator_quiescence"].get("observed_at_monotonic_ns")
    commit_ns = commit.get("committed_at_monotonic_ns")
    validation_ns = identity_validation.get("validated_at_monotonic_ns")
    identity_gate_ns = gate.get("observed_at_monotonic_ns")
    binding_installed_ns = engine_binding.get("installed_at_monotonic_ns")
    if (
        not isinstance(timeline_events, dict)
        or timeline_events.get("OVERLOAD_DETECTED") != overload_ns
        or timeline_events.get("TRIGGER_PREDICATE_SATISFIED") != predicate_ns
        or timeline_events.get("RECLAIM_TRIGGER_EMITTED") != emitted_ns
        or timeline_events.get("ROLLOUT_ADMISSION_STOP") != admission_ns
        or timings.get("reclaim_trigger_ns") != emitted_ns
        or timings.get("rollout_admission_stop_ns") != admission_ns
        or any(
            type(value) is not int
            for value in (
                binding_installed_ns,
                state_quiescence_ns,
                allocator_quiescence_ns,
                commit_ns,
                validation_ns,
                identity_gate_ns,
            )
        )
        or not (
            emitted_ns
            < admission_ns
            < binding_installed_ns
            < state_quiescence_ns
            < allocator_quiescence_ns
            < commit_ns
            < validation_ns
            < identity_gate_ns
        )
    ):
        raise RuntimeError("post-target raw lifecycle ordering is inconsistent")

    sanity = controller.get("sanity_guard_pair")
    assessments = None if not isinstance(sanity, dict) else sanity.get("assessments")
    if (
        controller.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-targeted-controller/v1"
        or controller.get("status") != "succeeded"
        or controller.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or controller.get("execution_mode") != "targeted-source-identity-v11"
        or controller.get("terminal_phase") != "SOURCE_ALLOCATION_IDENTITY_GATE"
        or controller.get("controller_error") is not None
        or controller.get("cleanup_error") is not None
        or controller.get("stable_physical_gpu_identity") is not True
        or controller.get("worker_returncodes") != {"rollout": 0, "serving": 0}
        or controller.get("worker_results") != [serving, rollout]
        or controller.get("in_function_cleanup") != in_function_cleanup
        or not isinstance(sanity, dict)
        or sanity.get("passed") is not True
        or sanity.get("broad_capacity_calibration_performed") is not False
        or not isinstance(assessments, list)
        or len(assessments) != 2
    ):
        raise RuntimeError("post-target raw controller result is inconsistent")
    by_guard = {
        assessment.get("guard"): assessment
        for assessment in assessments
        if isinstance(assessment, dict)
    }
    if (
        set(by_guard) != {"sanity_12rps_stable", "sanity_15rps_overload"}
        or any(assessment.get("passed") is not True for assessment in by_guard.values())
        or by_guard["sanity_12rps_stable"].get("expected_rate_rps") != config.lambda_1_rps
        or by_guard["sanity_15rps_overload"].get("expected_rate_rps") != config.lambda_spike_rps
        or by_guard["sanity_15rps_overload"].get("signals", {}).get("positive_queue_slope")
        is not True
        or by_guard["sanity_15rps_overload"].get("signals", {}).get("completed_rate_below_offered")
        is not True
    ):
        raise RuntimeError("post-target raw sanity guards are inconsistent")
    return {
        "ordering_monotonic_ns": {
            "overload_detected": overload_ns,
            "trigger_predicate_satisfied": predicate_ns,
            "reclaim_trigger_emitted": emitted_ns,
            "rollout_admission_stop": admission_ns,
            "production_binding_installed": binding_installed_ns,
            "state_quiescence": state_quiescence_ns,
            "allocator_quiescence": allocator_quiescence_ns,
            "source_capture_commit": commit_ns,
            "semantic_validation": validation_ns,
            "identity_gate": identity_gate_ns,
        },
        "trigger_backlog_requests": 21,
        "trigger_emergency_headroom_requests": 43,
        "maximum_outstanding_requests": 29,
        "minimum_emergency_headroom_requests": 35,
        "final_queues_zero": True,
        "live_rollouts": True,
        "sanity_pair_passed": True,
    }


def _verify_post_target_integrated_authorization(
    manifest: Mapping[str, Any], repository_root: Path, *, config: Any
) -> dict[str, Any]:
    authorization = manifest.get("post_target_integrated_authorization")
    if (
        not isinstance(authorization, dict)
        or authorization.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-post-target-integrated-authorization/v1"
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("integrated_attempt_id") != POST_TARGET_INTEGRATED_ATTEMPT_ID
        or authorization.get("targeted_attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or authorization.get("ledger_sha256_before_reservation") != POST_TARGET_LEDGER_SHA256
        or config.attempt_id != POST_TARGET_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != POST_TARGET_LEDGER_SHA256
    ):
        raise RuntimeError("post-target integrated authorization is inconsistent")
    raw_bindings = authorization.get("targeted_attempt_evidence")
    if not isinstance(raw_bindings, dict) or set(raw_bindings) != set(
        POST_TARGET_EVIDENCE_BINDINGS
    ):
        raise RuntimeError("post-target integrated evidence bindings are incomplete")
    evidence: dict[str, dict[str, Any]] = {}
    verified_bindings: dict[str, dict[str, str]] = {}
    for label, expected_reference in POST_TARGET_EVIDENCE_BINDINGS.items():
        binding = raw_bindings[label]
        if (
            not isinstance(binding, dict)
            or binding.get("artifact") != expected_reference
            or not isinstance(binding.get("sha256"), str)
        ):
            raise RuntimeError(f"post-target {label} binding is malformed")
        path = validate_bound_artifact(
            repository_root,
            reference=expected_reference,
            expected_sha256=str(binding["sha256"]),
        )
        evidence[label] = _load_json_object(path, label=f"post-target {label}")
        verified_bindings[label] = {
            "artifact": expected_reference,
            "sha256": str(binding["sha256"]),
        }

    status = evidence["status"]
    provider = evidence["provider_cleanup"]
    gate = evidence["identity_gate"]
    commit_envelope = evidence["source_capture_commit"]
    remote_manifest = evidence["remote_manifest"]
    in_function = evidence["in_function_cleanup"]
    semantic_fields = (
        "exact_logical_mapping",
        "exact_block_epoch_identity",
        "exact_owner_sets",
        "exact_refcounts",
        "device_identity_pass",
        "all_allocations_live",
        "no_post_commit_mutation",
    )
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-status/v1"
        or status.get("status") != "PASS"
        or status.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or status.get("execution_mode") != "targeted-source-identity-v11"
        or status.get("terminal_phase") != "SOURCE_ALLOCATION_IDENTITY_GATE"
        or status.get("source_identity_gate_reached") is not True
        or status.get("source_identity_gate_passed") is not True
        or type(status.get("expected_allocation_count")) is not int
        or status.get("expected_allocation_count") != 1152
        or type(status.get("observed_allocation_count")) is not int
        or status.get("observed_allocation_count") != 1152
        or any(status.get(field) is not True for field in semantic_fields)
        or not _exact_zero_int(status.get("post_commit_allocation_event_count"))
        or status.get("optimized_export_started") is not False
        or status.get("transaction_source_release_started") is not False
        or status.get("optimized_export_authorized") is not False
        or status.get("source_release_authorized") is not False
        or status.get("integrated_v11_retry_authorized") is not True
    ):
        raise RuntimeError("post-target status does not prove 1,152/1,152 identity PASS")
    status_ledger = status.get("ledger")
    if (
        not isinstance(status_ledger, dict)
        or status_ledger.get("authoritative_sha256") != POST_TARGET_LEDGER_SHA256
        or not _exact_zero_int(status_ledger.get("active_reservations"))
        or status_ledger.get("integrated_budget_sufficient") is not True
    ):
        raise RuntimeError("post-target status ledger is inconsistent")
    status_cleanup = status.get("cleanup")
    if (
        not isinstance(status_cleanup, dict)
        or status_cleanup.get("in_function") != "PASS"
        or status_cleanup.get("provider") != "PASS"
        or status_cleanup.get("provider_artifact") != "provider-cleanup-attempt-b.json"
        or any(
            not _exact_zero_int(status_cleanup.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "provider_reservations",
            )
        )
    ):
        raise RuntimeError("post-target status cleanup is inconsistent")
    invocation = status.get("measured_invocation")
    if not isinstance(invocation, dict) or not isinstance(invocation.get("function_call_id"), str):
        raise RuntimeError("post-target status lacks a function-call identity")

    immutable = status.get("immutable_artifacts")
    targeted_repro_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/"
    )
    immutable_expected = {
        label: {
            "artifact": verified_bindings[label]["artifact"].removeprefix(targeted_repro_prefix),
            "sha256": verified_bindings[label]["sha256"],
        }
        for label in ("remote_manifest", "identity_gate", "source_capture_commit")
    }
    if not isinstance(immutable, dict) or any(
        immutable.get(label) != binding for label, binding in immutable_expected.items()
    ):
        raise RuntimeError("post-target status immutable hashes are inconsistent")

    source_sha = status.get("source_capture_commit_sha256")
    validation_sha = status.get("source_identity_validation_sha256")
    if (
        not isinstance(source_sha, str)
        or not isinstance(validation_sha, str)
        or gate.get("schema_version") != "sloforge.branchfabric.v11-source-identity-gate/v1"
        or gate.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or gate.get("event") != "SOURCE_ALLOCATION_IDENTITY_GATE"
        or gate.get("passed") is not True
        or type(gate.get("expected_allocation_count")) is not int
        or gate.get("expected_allocation_count") != 1152
        or type(gate.get("observed_allocation_count")) is not int
        or gate.get("observed_allocation_count") != 1152
        or any(gate.get(field) is not True for field in semantic_fields)
        or not _exact_zero_int(gate.get("post_commit_allocation_event_count"))
        or gate.get("optimized_export_started") is not False
        or gate.get("transaction_source_release_started") is not False
        or gate.get("source_capture_commit_sha256") != source_sha
        or gate.get("source_identity_validation_sha256") != validation_sha
    ):
        raise RuntimeError("post-target source identity gate is inconsistent")
    commit_result = _verify_post_target_source_commit(
        commit_envelope,
        expected_semantic_sha256=source_sha,
        config=config,
    )
    identity_validation = commit_envelope.get("identity_validation")
    if (
        not isinstance(identity_validation, dict)
        or hashlib.sha256(_canonical_bytes(identity_validation)).hexdigest() != validation_sha
    ):
        raise RuntimeError("post-target identity validation hash is inconsistent")
    if gate.get("allocation_history_sha256") != commit_result["allocation_history_sha256"]:
        raise RuntimeError("post-target allocation-history hash does not join identity gate")

    provider_observation = provider.get("provider_observation")
    provider_in_function = provider.get("in_function_cleanup")
    provider_ledger = provider.get("ledger")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-targeted-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or provider.get("function_call_id") != invocation.get("function_call_id")
        or not isinstance(provider_observation, dict)
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(provider_in_function, dict)
        or provider_in_function.get("status") != "PASS"
        or provider_in_function.get("artifact")
        != "raw/exp004-v11-targeted-identity-s41-b/in_function_cleanup.json"
        or provider_in_function.get("sha256") != verified_bindings["in_function_cleanup"]["sha256"]
        or provider_in_function.get("forced_kill_required") is not False
        or any(
            not _exact_zero_int(provider_in_function.get(field))
            for field in ("surviving_children", "zombies", "ipc_leaks", "profiler_processes")
        )
        or not isinstance(provider_ledger, dict)
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_settlement") != POST_TARGET_LEDGER_SHA256
    ):
        raise RuntimeError("post-target provider cleanup is inconsistent")
    if (
        in_function.get("schema_version") != "sloforge.branchfabric.in-function-cleanup/v1"
        or in_function.get("status") != "PASS"
        or in_function.get("pass") is not True
        or in_function.get("forced_kill_required") is not False
        or in_function.get("cuda_released") is not True
        or in_function.get("parent_reaped_all_owned_children") is not True
        or in_function.get("owned_ipc_resources_released") is not True
        or in_function.get("pipes_closed") is not True
        or any(
            in_function.get(field) != []
            for field in (
                "surviving_children",
                "surviving_process_groups",
                "zombie_processes_after",
                "profiler_processes_after",
                "resource_tracker_processes_after",
                "compute_processes_after",
                "leaked_ipc_resources",
                "leaked_threads",
                "cleanup_errors",
            )
        )
    ):
        raise RuntimeError("post-target in-function cleanup is inconsistent")

    manifest_rows = remote_manifest.get("artifacts")
    if (
        remote_manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or remote_manifest.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or remote_manifest.get("remote_prefix")
        != (f"experiment-004/v11/integrated/modal/{RETRY_TARGETED_IDENTITY_ATTEMPT_ID}")
        or not isinstance(manifest_rows, list)
    ):
        raise RuntimeError("post-target remote manifest is inconsistent")
    manifest_by_path: dict[str, Mapping[str, Any]] = {}
    for row in manifest_rows:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            raise RuntimeError("post-target remote manifest row is malformed")
        relative_path = str(row["relative_path"])
        if relative_path in manifest_by_path:
            raise RuntimeError("post-target remote manifest paths are duplicated")
        manifest_by_path[relative_path] = row
    raw_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/targeted-repro/raw/"
        f"{RETRY_TARGETED_IDENTITY_ATTEMPT_ID}/"
    )
    remote_evidence_labels = set(POST_TARGET_EVIDENCE_BINDINGS) - {
        "status",
        "provider_cleanup",
        "remote_manifest",
    }
    for label in sorted(remote_evidence_labels):
        reference = verified_bindings[label]["artifact"]
        if not reference.startswith(raw_prefix):
            raise RuntimeError("post-target evidence path is outside the immutable remote bundle")
        relative_path = reference.removeprefix(raw_prefix)
        row = manifest_by_path.get(relative_path)
        bound_path = repository_root / reference
        if (
            not isinstance(row, dict)
            or row.get("sha256") != verified_bindings[label]["sha256"]
            or type(row.get("bytes")) is not int
            or row.get("bytes") != bound_path.stat().st_size
        ):
            raise RuntimeError("post-target remote manifest child hash is inconsistent")

    runtime_result = _verify_post_target_runtime_evidence(
        evidence,
        commit_envelope=commit_envelope,
        gate=gate,
        in_function_cleanup=in_function,
        config=config,
    )

    raw_reviews = authorization.get("target_reviews")
    if not isinstance(raw_reviews, list) or len(raw_reviews) != len(POST_TARGET_REVIEW_BINDINGS):
        raise RuntimeError("post-target integrated authorization lacks three reviews")
    reviews_by_agent: dict[int, Mapping[str, Any]] = {}
    for binding in raw_reviews:
        if not isinstance(binding, dict):
            raise RuntimeError("post-target review binding is malformed")
        agent = _agent_number(binding)
        if agent is None or agent in reviews_by_agent:
            raise RuntimeError("post-target review agents are invalid or duplicated")
        reviews_by_agent[agent] = binding
    if set(reviews_by_agent) != set(POST_TARGET_REVIEW_BINDINGS):
        raise RuntimeError("post-target review coverage is incomplete")
    verified_reviews: list[dict[str, Any]] = []
    for agent, (role, expected_reference) in POST_TARGET_REVIEW_BINDINGS.items():
        binding = reviews_by_agent[agent]
        if (
            binding.get("role") != role
            or binding.get("status") != "PASS"
            or binding.get("artifact") != expected_reference
            or not isinstance(binding.get("sha256"), str)
        ):
            raise RuntimeError(f"post-target Agent {agent} review binding is malformed")
        review_path = validate_bound_artifact(
            repository_root,
            reference=expected_reference,
            expected_sha256=str(binding["sha256"]),
        )
        review = _load_json_object(review_path, label=f"post-target Agent {agent} review")
        if (
            review.get("schema_version")
            != "sloforge.branchfabric.v11-targeted-independent-review/v1"
            or review.get("status") != "PASS"
            or review.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
            or _agent_number(review) != agent
            or review.get("role") != role
            or review.get("integrated_retry_recommendation") != "AUTHORIZE_EXACTLY_ONE"
        ):
            raise RuntimeError(f"post-target Agent {agent} review is not PASS")
        if agent == 11:
            review_manifest = review.get("immutable_manifest")
            identity = review.get("identity")
            safety = review.get("safety")
            if (
                not isinstance(review_manifest, dict)
                or review_manifest.get("artifact")
                != verified_bindings["remote_manifest"]["artifact"]
                or review_manifest.get("sha256") != verified_bindings["remote_manifest"]["sha256"]
                or type(review_manifest.get("verified_entries")) is not int
                or review_manifest.get("verified_entries") != len(manifest_rows)
                or not _exact_zero_int(review_manifest.get("missing_entries"))
                or not _exact_zero_int(review_manifest.get("hash_or_size_drift"))
                or not isinstance(identity, dict)
                or any(
                    type(identity.get(field)) is not int or identity.get(field) != 1152
                    for field in (
                        "expected_allocations",
                        "observed_allocations",
                        "logical_mapping_passed",
                        "block_epoch_passed",
                        "owner_set_passed",
                        "refcount_passed",
                        "device_passed",
                        "live_status_passed",
                    )
                )
                or type(identity.get("shared_allocations")) is not int
                or identity.get("shared_allocations") != 1024
                or type(identity.get("private_allocations")) is not int
                or identity.get("private_allocations") != 128
                or not _exact_zero_int(identity.get("post_commit_allocator_mutations"))
                or identity.get("source_capture_commit_semantic_sha256") != source_sha
                or identity.get("source_identity_validation_sha256") != validation_sha
                or not isinstance(safety, dict)
                or safety.get("optimized_export_started") is not False
                or safety.get("transaction_source_release_started") is not False
                or safety.get("in_function_cleanup") != "PASS"
                or safety.get("provider_cleanup") != "PASS"
            ):
                raise RuntimeError("post-target Agent 11 allocation review is inconsistent")
        elif agent == 12:
            ordering = review.get("ordering_monotonic_ns")
            witness = review.get("engine_step_witness")
            review_quiescence = review.get("allocator_quiescence")
            safety = review.get("terminal_safety")
            if (
                not isinstance(ordering, dict)
                or ordering != runtime_result["ordering_monotonic_ns"]
                or not all(type(value) is int for value in ordering.values())
                or list(ordering.values()) != sorted(ordering.values())
                or not isinstance(witness, dict)
                or witness.get("binding_id") != commit_envelope["commit"].get("gate_binding_id")
                or witness.get("acquisition_generation")
                != commit_envelope["commit"].get("gate_acquisition_generation")
                or witness.get("same_witness") is not True
                or witness.get("async_scheduling") is not False
                or witness.get("engine_step_excluded") is not True
                or witness.get("allocator_mutations_excluded") is not True
                or not isinstance(review_quiescence, dict)
                or type(review_quiescence.get("paused_branches")) is not int
                or review_quiescence.get("paused_branches") != 8
                or type(review_quiescence.get("expected_paused_branches")) is not int
                or review_quiescence.get("expected_paused_branches") != 8
                or not _exact_zero_int(review_quiescence.get("waiting_requests"))
                or not _exact_zero_int(review_quiescence.get("skipped_waiting_requests"))
                or not _exact_zero_int(review_quiescence.get("post_commit_allocation_events"))
                or review_quiescence.get("passed") is not True
                or not isinstance(safety, dict)
                or safety.get("optimized_export_started") is not False
                or safety.get("transaction_source_release_started") is not False
                or safety.get("workers_returned_zero") is not True
                or safety.get("in_function_cleanup") != "PASS"
                or safety.get("provider_cleanup") != "PASS"
            ):
                raise RuntimeError("post-target Agent 12 synchronization review is inconsistent")
        else:
            lifecycle_review = review.get("lifecycle")
            terminal_contract = review.get("terminal_contract")
            review_cleanup = review.get("cleanup")
            review_budget = review.get("budget")
            if (
                not isinstance(lifecycle_review, dict)
                or lifecycle_review.get("live_rollouts") is not runtime_result["live_rollouts"]
                or lifecycle_review.get("trigger_backlog_requests")
                != runtime_result["trigger_backlog_requests"]
                or lifecycle_review.get("trigger_emergency_headroom_requests")
                != runtime_result["trigger_emergency_headroom_requests"]
                or lifecycle_review.get("maximum_outstanding_requests")
                != runtime_result["maximum_outstanding_requests"]
                or lifecycle_review.get("minimum_emergency_headroom_requests")
                != runtime_result["minimum_emergency_headroom_requests"]
                or any(
                    lifecycle_review.get(field) is not True
                    for field in (
                        "live_rollouts",
                        "control_serving",
                        "twelve_rps_sanity_pass",
                        "fifteen_rps_overload_pass",
                        "allocator_quiescence_pass",
                        "source_capture_commit_pass",
                        "identity_gate_pass",
                    )
                )
                or type(lifecycle_review.get("expected_allocations")) is not int
                or lifecycle_review.get("expected_allocations") != 1152
                or type(lifecycle_review.get("observed_allocations")) is not int
                or lifecycle_review.get("observed_allocations") != 1152
                or not _exact_zero_int(lifecycle_review.get("post_commit_mutations"))
                or not isinstance(terminal_contract, dict)
                or terminal_contract.get("terminal_phase") != "SOURCE_ALLOCATION_IDENTITY_GATE"
                or terminal_contract.get("optimized_export_started") is not False
                or terminal_contract.get("transaction_source_release_started") is not False
                or terminal_contract.get("final_serving_queues_zero") is not True
                or terminal_contract.get("final_serving_queues_zero")
                is not runtime_result["final_queues_zero"]
                or not isinstance(review_cleanup, dict)
                or review_cleanup.get("in_function") != "PASS"
                or review_cleanup.get("provider") != "PASS"
                or not _exact_zero_int(review_cleanup.get("active_provider_resources"))
                or not _exact_zero_int(review_cleanup.get("active_ledger_reservations"))
                or not isinstance(review_budget, dict)
                or review_budget.get("ledger_sha256") != POST_TARGET_LEDGER_SHA256
                or review_budget.get("sufficient") is not True
                or review.get("prelaunch_requirement")
                != (
                    "Use a fresh content-addressed attempt-D authorization; the already "
                    "consumed attempt-C config is forbidden."
                )
            ):
                raise RuntimeError("post-target Agent 13 methodology review is inconsistent")
        verified_reviews.append(
            {
                "agent": agent,
                "role": role,
                "artifact": expected_reference,
                "sha256": binding["sha256"],
            }
        )
    return {
        **authorization,
        "targeted_attempt_evidence": verified_bindings,
        "target_reviews": verified_reviews,
        "source_capture_commit": commit_result,
        "function_call_id": invocation["function_call_id"],
        "verified": True,
    }


def _verify_prefunction_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit Attempt E only from Attempt D's proven pre-function failure."""

    authorization = manifest.get("post_prefunction_failure_integrated_authorization")
    authorization_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "prior_attempt_evidence",
        "expected_successful_integrated_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != authorization_fields
        or authorization.get("schema_version")
        != ("sloforge.branchfabric.exp004-v11-post-prefunction-failure-integrated-authorization/v1")
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id") != PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != PREFUNCTION_FAILED_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation") != PREFUNCTION_RETRY_LEDGER_SHA256
        or config.attempt_id != PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != PREFUNCTION_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-E pre-function retry authorization is inconsistent")

    expected_contract = {
        "config_schema_version": ("sloforge.branchfabric.experiment-004-v11-integrated-config/v1"),
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_contract:
        raise RuntimeError("Attempt-E successful integrated result contract is inconsistent")

    raw_bindings = authorization.get("prior_attempt_evidence")
    if not isinstance(raw_bindings, dict) or set(raw_bindings) != set(
        PREFUNCTION_RETRY_EVIDENCE_BINDINGS
    ):
        raise RuntimeError("Attempt-E prior-attempt evidence bindings are incomplete")
    verified_bindings: dict[str, dict[str, str]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    evidence_paths: dict[str, Path] = {}
    for label, expected_reference in PREFUNCTION_RETRY_EVIDENCE_BINDINGS.items():
        binding = raw_bindings.get(label)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"artifact", "sha256"}
            or binding.get("artifact") != expected_reference
            or not isinstance(binding.get("sha256"), str)
        ):
            raise RuntimeError(f"Attempt-E prior {label} binding is malformed")
        evidence_path = validate_bound_artifact(
            repository_root,
            reference=expected_reference,
            expected_sha256=str(binding["sha256"]),
        )
        evidence_paths[label] = evidence_path
        evidence[label] = _load_json_object(
            evidence_path, label=f"Attempt-D terminal {label} evidence"
        )
        verified_bindings[label] = {
            "artifact": expected_reference,
            "sha256": str(binding["sha256"]),
        }

    status = evidence["status"]
    status_fields = {
        "schema_version",
        "observed_at_utc",
        "status",
        "scientifically_valid",
        "attempt_id",
        "execution_mode",
        "failure_stage",
        "function_entered",
        "gpu_allocation_observed",
        "controller_started",
        "source_identity_gate_reached",
        "optimized_export_started",
        "transaction_source_release_started",
        "hbm_reclaimed",
        "gpu1_serving_enabled",
        "restore_started",
        "branch_continuation_started",
        "blocker",
        "accounting",
        "cleanup",
        "integrated_retry_authorized",
        "kill_and_recompute_authorized",
        "terminal_classification",
    }
    blocker = status.get("blocker")
    accounting = status.get("accounting")
    cleanup = status.get("cleanup")
    if (
        set(status) != status_fields
        or status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-status/v1"
        or status.get("status") != "FAIL_PRE_FUNCTION_IMAGE_BUILD_CLIENT_DISCONNECT"
        or status.get("scientifically_valid") is not False
        or status.get("attempt_id") != PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("execution_mode") != "integrated-reclamation-v11"
        or status.get("failure_stage") != "modal-launch"
        or any(
            status.get(field) is not False
            for field in (
                "function_entered",
                "gpu_allocation_observed",
                "controller_started",
                "source_identity_gate_reached",
                "optimized_export_started",
                "transaction_source_release_started",
                "hbm_reclaimed",
                "gpu1_serving_enabled",
                "restore_started",
                "branch_continuation_started",
                "integrated_retry_authorized",
                "kill_and_recompute_authorized",
            )
        )
        or status.get("terminal_classification") != "HARDWARE_GATE_NOT_REACHED"
        or not isinstance(status.get("observed_at_utc"), str)
        or not status["observed_at_utc"]
        or not isinstance(blocker, dict)
        or set(blocker) != {"code", "client_exception", "provider_terminal", "semantic_consequence"}
        or blocker.get("code")
        != "MODAL_IMAGE_BUILD_EXTERNAL_SHUTDOWN_AFTER_CLIENT_HEARTBEAT_FAILURE"
        or blocker.get("client_exception")
        != "AttributeError: Connection object has no attribute _transport"
        or blocker.get("provider_terminal") != "Image build terminated due to external shut-down"
        or blocker.get("semantic_consequence")
        != "No integrated runtime lifecycle or scientific measurement was produced."
        or not isinstance(accounting, dict)
        or set(accounting)
        != {
            "actual_gpu_seconds",
            "actual_gpu_seconds_status",
            "conservative_charged_gpu_seconds",
            "conservative_charged_wall_seconds",
            "ledger_sha256_after_settlement",
            "remaining_gpu_seconds",
            "active_reservations",
        }
        or accounting.get("actual_gpu_seconds") is not None
        or accounting.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
        or accounting.get("conservative_charged_gpu_seconds") != 1176.0
        or accounting.get("conservative_charged_wall_seconds") != 588.0
        or accounting.get("ledger_sha256_after_settlement") != PREFUNCTION_FAILED_LEDGER_SHA256
        or not _exact_zero_int(accounting.get("active_reservations"))
        or not isinstance(accounting.get("remaining_gpu_seconds"), (int, float))
        or isinstance(accounting.get("remaining_gpu_seconds"), bool)
        or not math.isfinite(float(accounting["remaining_gpu_seconds"]))
        or float(accounting["remaining_gpu_seconds"]) < 0.0
        or not isinstance(cleanup, dict)
        or set(cleanup)
        != {
            "in_function",
            "provider",
            "provider_artifact",
            "active_apps",
            "running_tasks",
            "running_containers",
            "endpoints",
            "provider_reservations",
        }
        or cleanup.get("in_function") != "NOT_APPLICABLE_FUNCTION_NOT_ENTERED"
        or cleanup.get("provider") != "PASS"
        or cleanup.get("provider_artifact") != "provider-cleanup-attempt-d.json"
        or any(
            not _exact_zero_int(cleanup.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "provider_reservations",
            )
        )
    ):
        raise RuntimeError("Attempt-D terminal status is inconsistent")

    provider = evidence["provider_cleanup"]
    provider_fields = {
        "schema_version",
        "observed_at_utc",
        "status",
        "attempt_id",
        "function_call_id",
        "app_name",
        "provider",
        "provider_observation",
        "in_function_cleanup",
        "failure_evidence",
        "commands",
        "ledger",
        "provenance",
    }
    provider_observation = provider.get("provider_observation")
    in_function = provider.get("in_function_cleanup")
    failure_binding = provider.get("failure_evidence")
    provider_ledger = provider.get("ledger")
    expected_commands = {
        "modal app list --json",
        "modal container list --json",
        "modal endpoint list --json",
        "modal queue list --json",
        "modal dict list --json",
        "modal volume list --json",
    }
    if (
        set(provider) != provider_fields
        or provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
        or provider.get("function_call_id") is not None
        or provider.get("app_name") != "sloforge-branchfabric-gpu-reclamation-004-v11-integrated"
        or provider.get("provider") != "Modal"
        or not isinstance(provider.get("observed_at_utc"), str)
        or not provider["observed_at_utc"]
        or not isinstance(provider.get("provenance"), str)
        or not provider["provenance"]
        or not isinstance(provider_observation, dict)
        or set(provider_observation)
        != {
            "active_apps",
            "stopped_apps",
            "running_tasks",
            "running_containers",
            "endpoints",
            "queues",
            "dicts",
            "provider_reservations",
            "authorized_persistent_volumes",
            "unexpected_persistent_volumes",
        }
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or type(provider_observation.get("stopped_apps")) is not int
        or provider_observation["stopped_apps"] < 0
        or provider_observation.get("authorized_persistent_volumes")
        != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(in_function, dict)
        or set(in_function) != {"status", "artifact", "reason"}
        or in_function.get("status") != "NOT_APPLICABLE_FUNCTION_NOT_ENTERED"
        or in_function.get("artifact") is not None
        or not isinstance(in_function.get("reason"), str)
        or not in_function["reason"]
        or not isinstance(failure_binding, dict)
        or set(failure_binding)
        != {
            "artifact",
            "sha256",
            "stage",
            "actual_gpu_seconds",
            "conservative_charged_gpu_seconds",
        }
        or failure_binding.get("artifact")
        != "failures/exp004-v11-integrated-s41-d-conservative-charge.json"
        or failure_binding.get("sha256") != verified_bindings["failure"]["sha256"]
        or failure_binding.get("stage") != "modal-launch"
        or failure_binding.get("actual_gpu_seconds") is not None
        or failure_binding.get("conservative_charged_gpu_seconds") != 1176.0
        or not isinstance(provider.get("commands"), list)
        or set(provider["commands"]) != expected_commands
        or len(provider["commands"]) != len(expected_commands)
        or not isinstance(provider_ledger, dict)
        or set(provider_ledger)
        != {
            "active_reservations",
            "sha256_after_settlement",
            "conservative_consumed_gpu_seconds",
            "remaining_gpu_seconds",
        }
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_settlement") != PREFUNCTION_FAILED_LEDGER_SHA256
        or provider_ledger.get("conservative_consumed_gpu_seconds") != 19397.975013231975
        or provider_ledger.get("remaining_gpu_seconds") != 2202.024986768025
    ):
        raise RuntimeError("Attempt-D provider cleanup is inconsistent")

    failure = evidence["failure"]
    failure_fields = {
        "schema_version",
        "status",
        "attempt_id",
        "reservation_id",
        "failure_stage",
        "error",
        "client_elapsed_seconds",
        "actual_gpu_seconds",
        "actual_gpu_seconds_status",
        "charged_wall_seconds",
        "charged_gpu_seconds",
        "accounting_policy",
        "stdout_tail",
        "stderr_tail",
    }
    failure_error = failure.get("error")
    elapsed = failure.get("client_elapsed_seconds")
    stdout_tail = failure.get("stdout_tail")
    stderr_tail = failure.get("stderr_tail")
    if (
        set(failure) != failure_fields
        or failure.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or failure.get("status") != "CONSERVATIVELY_CHARGED"
        or failure.get("attempt_id") != PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
        or not isinstance(failure.get("reservation_id"), str)
        or not failure["reservation_id"].startswith("exp004-v11-d-")
        or failure.get("failure_stage") != "modal-launch"
        or not isinstance(failure_error, dict)
        or set(failure_error) != {"type", "message"}
        or failure_error.get("type") != "RuntimeError"
        or failure_error.get("message") != "integrated v11 Modal process returned failure"
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or not 0.0 < float(elapsed) <= 900.0
        or failure.get("actual_gpu_seconds") is not None
        or failure.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
        or failure.get("charged_wall_seconds") != 588.0
        or failure.get("charged_gpu_seconds") != 1176.0
        or failure.get("accounting_policy")
        != "charge-full-preflight-bound-without-fabricated-measurement"
        or not isinstance(stdout_tail, str)
        or "Stopping app - local client disconnected" not in stdout_tail
        or '"result"' in stdout_tail
        or "function_call_id" in stdout_tail
        or not isinstance(stderr_tail, str)
        or "object has no attribute '_transport'" not in stderr_tail
        or "APP_STATE_STOPPED" not in stderr_tail
        or "Image build for " not in stderr_tail
        or "terminated due to external" not in stderr_tail
    ):
        raise RuntimeError("Attempt-D conservative failure evidence is inconsistent")

    if (
        status["accounting"]["ledger_sha256_after_settlement"]
        != provider_ledger["sha256_after_settlement"]
    ):
        raise RuntimeError("Attempt-D terminal and provider ledger evidence disagree")

    budget_fields = {
        "schema_version",
        "experiment",
        "authorized_at_utc",
        "authorization_source",
        "authorization_text",
        "authorized_gpu_budget_usd",
        "money_budget_semantics",
        "authorized_cumulative_a100_hours",
        "authorized_cumulative_gpu_seconds",
        "ceiling_semantics",
        "consumed_gpu_seconds_at_authorization",
        "remaining_gpu_seconds_at_authorization",
        "remaining_gpu_hours_at_authorization",
        "active_reservations_at_authorization",
        "ledger_path",
        "previous_ledger_sha256",
        "authorized_ledger_sha256",
        "authorized_settled_intervals_sha256",
        "authorized_failure_charges_sha256",
        "ledger_mutation",
        "money_budget_mutation",
        "authorized_scope",
        "per_invocation_constraints",
        "does_not_execute",
    }
    ledger_mutation = budget.get("ledger_mutation")
    money_mutation = budget.get("money_budget_mutation")
    constraints = budget.get("per_invocation_constraints")
    if (
        set(budget) != budget_fields
        or budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("experiment") != "branchfabric-gpu-validation-experiment-004-v11"
        or budget.get("authorization_source") != "explicit-user-instruction-current-thread"
        or budget.get("authorization_text")
        != [
            "yeah just do that, jack up the gpu budget",
            "just keep going until we get a successful run",
        ]
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or budget.get("authorized_cumulative_a100_hours") != 12.0
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("consumed_gpu_seconds_at_authorization") != 19397.975013231975
        or budget.get("remaining_gpu_seconds_at_authorization") != 23802.024986768025
        or budget.get("remaining_gpu_hours_at_authorization") != 6.611673607435563
        or not _exact_zero_int(budget.get("active_reservations_at_authorization"))
        or budget.get("ledger_path")
        != "artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
        or budget.get("previous_ledger_sha256") != PREFUNCTION_FAILED_LEDGER_SHA256
        or budget.get("authorized_ledger_sha256") != PREFUNCTION_RETRY_LEDGER_SHA256
        or not isinstance(ledger_mutation, dict)
        or set(ledger_mutation)
        != {
            "hard_additional_gpu_seconds_before",
            "hard_additional_gpu_seconds_after",
            "settled_interval_count_before",
            "settled_interval_count_after",
            "settled_intervals_changed",
            "conservative_failure_charge_count_before",
            "conservative_failure_charge_count_after",
            "conservative_failure_charges_changed",
            "reservation_count_before",
            "reservation_count_after",
        }
        or ledger_mutation.get("hard_additional_gpu_seconds_before") != 21600.0
        or ledger_mutation.get("hard_additional_gpu_seconds_after") != 43200.0
        or ledger_mutation.get("settled_interval_count_before") != 23
        or ledger_mutation.get("settled_interval_count_after") != 23
        or ledger_mutation.get("settled_intervals_changed") is not False
        or ledger_mutation.get("conservative_failure_charge_count_before") != 5
        or ledger_mutation.get("conservative_failure_charge_count_after") != 5
        or ledger_mutation.get("conservative_failure_charges_changed") is not False
        or not _exact_zero_int(ledger_mutation.get("reservation_count_before"))
        or not _exact_zero_int(ledger_mutation.get("reservation_count_after"))
        or not isinstance(money_mutation, dict)
        or money_mutation
        != {"v11_configured_budget_usd_before": 40.0, "v11_authorized_budget_usd_after": 80.0}
        or budget.get("authorized_scope")
        != [
            "sequential fresh-ID v11 integrated retries until one scientifically valid success",
            "one kill-and-recompute comparison only after integrated v11 passes and freezes",
            "mandatory post-run analysis and cleanup validation",
        ]
        or not isinstance(constraints, dict)
        or constraints
        != {
            "sole_paid_coordinator": True,
            "maximum_active_reservations": 1,
            "integrated_gpu_count": 2,
            "integrated_maximum_wall_seconds": 588.0,
            "integrated_maximum_gpu_seconds": 1176.0,
            "fresh_attempt_id_and_immutable_prefix_required": True,
            "provider_cleanup_required_before_next_attempt": True,
        }
        or budget.get("does_not_execute")
        != ["Modal invocation", "GPU allocation", "deployment", "FPGA implementation"]
    ):
        raise RuntimeError("Attempt-E expanded budget authorization is inconsistent")

    ledger_path = validate_bound_artifact(
        repository_root,
        reference=str(budget["ledger_path"]),
        expected_sha256=PREFUNCTION_RETRY_LEDGER_SHA256,
    )
    ledger = _load_json_object(ledger_path, label="Attempt-E pre-reservation ledger")
    intervals = ledger.get("intervals")
    charges = ledger.get("conservative_failure_charges")
    reservations = ledger.get("reservations")
    if (
        ledger.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or ledger.get("hard_additional_gpu_seconds") != 43200.0
        or ledger.get("consumed_additional_gpu_seconds") != 19397.975013231975
        or reservations != []
        or not isinstance(intervals, list)
        or len(intervals) != 23
        or not isinstance(charges, list)
        or len(charges) != 5
        or budget.get("authorized_settled_intervals_sha256")
        != hashlib.sha256(_canonical_bytes(intervals)).hexdigest()
        or budget.get("authorized_failure_charges_sha256")
        != hashlib.sha256(_canonical_bytes(charges)).hexdigest()
        or 43200.0 - 19397.975013231975 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-E pre-reservation ledger or budget headroom is inconsistent")
    d_intervals = [
        item
        for item in intervals
        if isinstance(item, dict)
        and item.get("invocation_id") == PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    d_charges = [
        item
        for item in charges
        if isinstance(item, dict)
        and item.get("invocation_id") == PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    d_charge = d_charges[0] if len(d_charges) == 1 else None
    charge_evidence = None if not isinstance(d_charge, dict) else d_charge.get("failure_evidence")
    charge_reference = (
        None if not isinstance(charge_evidence, dict) else charge_evidence.get("artifact_reference")
    )
    expected_charge_reference = PREFUNCTION_RETRY_EVIDENCE_BINDINGS["failure"]
    if (
        d_intervals
        or len(d_charges) != 1
        or not isinstance(d_charge, dict)
        or d_charge.get("reservation_id") != failure.get("reservation_id")
        or d_charge.get("requested_gpu") != "A100-80GB"
        or d_charge.get("gpu_count") != 2
        or d_charge.get("charged_wall_seconds") != 588.0
        or not isinstance(charge_evidence, dict)
        or set(charge_evidence) != {"artifact_reference", "artifact_sha256", "sample_selector"}
        or not isinstance(charge_reference, str)
        or not Path(charge_reference).as_posix().endswith(expected_charge_reference)
        or charge_evidence.get("artifact_sha256") != verified_bindings["failure"]["sha256"]
        or charge_evidence.get("sample_selector") != "$"
        or d_charge.get("failure_stage") != "modal-launch"
    ):
        raise RuntimeError("Attempt-D failure charge is not preserved exactly in the ledger")

    prior_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / PREFUNCTION_FAILED_INTEGRATED_ATTEMPT_ID
    )
    if prior_raw_root.exists() or prior_raw_root.is_symlink():
        raise RuntimeError("Attempt-D unexpectedly has remote function or allocator evidence")

    return {
        **authorization,
        "prior_attempt_evidence": verified_bindings,
        "prior_function_entered": False,
        "prior_in_function_cleanup": "NOT_APPLICABLE_FUNCTION_NOT_ENTERED",
        "prior_provider_cleanup": "PASS",
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "verified": True,
    }


def _verify_bundled_ledger_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit Attempt F from exact Attempt-E evidence and a bundled ledger snapshot."""

    authorization = manifest.get("post_remote_precontroller_failure_integrated_authorization")
    authorization_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "immutable_ledger_snapshot",
        "prior_attempt_evidence",
        "unchanged_target_b_authorization",
        "expected_successful_integrated_contract",
        "required_pre_worker_cleanup_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != authorization_fields
        or authorization.get("schema_version")
        != (
            "sloforge.branchfabric.exp004-v11-post-remote-precontroller-failure-"
            "integrated-authorization/v1"
        )
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id")
        != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != BUNDLED_LEDGER_RETRY_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation")
        != BUNDLED_LEDGER_RETRY_LEDGER_SHA256
        or config.attempt_id != BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != BUNDLED_LEDGER_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-F bundled-ledger retry authorization is inconsistent")

    expected_success_contract = {
        "config_schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    expected_cleanup_contract = {
        "cleanup_schema_version": "sloforge.branchfabric.in-function-cleanup/v1",
        "cleanup_scope": "PRE_WORKER_PREFLIGHT",
        "allowed_failure_stages": [
            "SEALED_EVIDENCE",
            "LIVE_CONTRACT",
            "CPU_CONTROL_IMPORTS",
            "PRE_WORKER_ARTIFACT_INITIALIZATION",
            "PRE_INVENTORY_CUDA_CLEAN_AUDIT",
            "GPU_INVENTORY_BEFORE_WORKERS",
            "ZERO_COMPUTE_BEFORE_WORKERS",
        ],
        "cleanup_pass_required": True,
        "empty_worker_ownership_required": True,
        "empty_runtime_evidence_required": True,
        "zero_compute_processes_required": True,
        "cuda_clean_parent_required": True,
        "controller_error_required": True,
        "cleanup_error": None,
        "cleanup_actions": [],
        "stable_physical_gpu_identity": False,
        "readiness_deadline_ns": None,
        "sanity_guard_pair": None,
        "cuda_clean_import_audit_count": 1,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_success_contract:
        raise RuntimeError("Attempt-F successful integrated result contract is inconsistent")
    if authorization.get("required_pre_worker_cleanup_contract") != expected_cleanup_contract:
        raise RuntimeError("Attempt-F pre-worker cleanup contract is inconsistent")

    snapshot_binding = authorization.get("immutable_ledger_snapshot")
    snapshot_reference, snapshot_sha256 = BUNDLED_LEDGER_SNAPSHOT_BINDING
    if (
        not isinstance(snapshot_binding, dict)
        or set(snapshot_binding) != {"artifact", "sha256"}
        or snapshot_binding.get("artifact") != snapshot_reference
        or snapshot_binding.get("sha256") != snapshot_sha256
    ):
        raise RuntimeError("Attempt-F immutable ledger snapshot binding is malformed")
    snapshot_path = validate_bound_artifact(
        repository_root,
        reference=snapshot_reference,
        expected_sha256=snapshot_sha256,
    )
    snapshot = _load_json_object(snapshot_path, label="Attempt-F immutable ledger snapshot")

    raw_bindings = authorization.get("prior_attempt_evidence")
    if not isinstance(raw_bindings, dict) or set(raw_bindings) != set(
        BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS
    ):
        raise RuntimeError("Attempt-F prior-attempt evidence bindings are incomplete")
    evidence: dict[str, dict[str, Any]] = {}
    verified_bindings: dict[str, dict[str, str]] = {}
    for label, (
        expected_reference,
        expected_sha256,
    ) in BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS.items():
        binding = raw_bindings.get(label)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"artifact", "sha256"}
            or binding.get("artifact") != expected_reference
            or binding.get("sha256") != expected_sha256
        ):
            raise RuntimeError(f"Attempt-F prior {label} binding is malformed")
        path = validate_bound_artifact(
            repository_root,
            reference=expected_reference,
            expected_sha256=expected_sha256,
        )
        evidence[label] = _load_json_object(path, label=f"Attempt-E {label} evidence")
        verified_bindings[label] = {
            "artifact": expected_reference,
            "sha256": expected_sha256,
        }

    target_binding = authorization.get("unchanged_target_b_authorization")
    target_reference, target_sha256 = BUNDLED_LEDGER_TARGET_B_MANIFEST_BINDING
    if (
        not isinstance(target_binding, dict)
        or set(target_binding) != {"artifact", "sha256"}
        or target_binding.get("artifact") != target_reference
        or target_binding.get("sha256") != target_sha256
    ):
        raise RuntimeError("Attempt-F unchanged Target-B binding is malformed")
    target_manifest_path = validate_bound_artifact(
        repository_root,
        reference=target_reference,
        expected_sha256=target_sha256,
    )
    target_manifest = _load_json_object(
        target_manifest_path, label="Attempt-D post-target authorization manifest"
    )
    target_config = config.model_copy(
        update={
            "attempt_id": POST_TARGET_INTEGRATED_ATTEMPT_ID,
            "ledger_sha256_before_reservation": POST_TARGET_LEDGER_SHA256,
        }
    )
    target_result = _verify_post_target_integrated_authorization(
        target_manifest,
        repository_root,
        config=target_config,
    )

    status = evidence["status"]
    status_fields = {
        "schema_version",
        "observed_at_utc",
        "status",
        "scientifically_valid",
        "attempt_id",
        "execution_mode",
        "failure_stage",
        "function_entered",
        "function_call_id",
        "gpu_allocation_observed",
        "controller_python_invoked",
        "controller_transaction_started",
        "source_identity_gate_reached",
        "optimized_export_started",
        "transaction_source_release_started",
        "hbm_reclaimed",
        "gpu1_serving_enabled",
        "restore_started",
        "branch_continuation_started",
        "blocker",
        "remote_evidence",
        "accounting",
        "cleanup",
        "integrated_retry_authorized",
        "kill_and_recompute_authorized",
        "terminal_classification",
    }
    blocker = status.get("blocker")
    remote_status = status.get("remote_evidence")
    accounting = status.get("accounting")
    status_cleanup = status.get("cleanup")
    false_status_fields = (
        "scientifically_valid",
        "controller_transaction_started",
        "source_identity_gate_reached",
        "optimized_export_started",
        "transaction_source_release_started",
        "hbm_reclaimed",
        "gpu1_serving_enabled",
        "restore_started",
        "branch_continuation_started",
        "integrated_retry_authorized",
        "kill_and_recompute_authorized",
    )
    if (
        set(status) != status_fields
        or status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-status/v1"
        or status.get("status") != "FAIL_PRE_CONTROLLER_MISSING_BUNDLED_LEDGER"
        or status.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("execution_mode") != "integrated-reclamation-v11"
        or status.get("failure_stage") != "remote-pre-controller-sealed-evidence"
        or status.get("function_entered") is not True
        or status.get("gpu_allocation_observed") is not True
        or status.get("controller_python_invoked") is not True
        or any(status.get(field) is not False for field in false_status_fields)
        or status.get("function_call_id") != "fc-01M104ZGT6J8YPBJ50BN0BMT9H"
        or status.get("terminal_classification") != "HARDWARE_GATE_NOT_REACHED"
        or not isinstance(blocker, dict)
        or blocker.get("code") != "REMOTE_VERIFIER_LEDGER_ARTIFACT_NOT_BUNDLED"
        or blocker.get("missing_path")
        != "/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004/gpu-hours.json"
        or blocker.get("exception") != "FileNotFoundError"
        or not isinstance(remote_status, dict)
        or remote_status.get("remote_manifest_sha256")
        != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["remote_manifest"][1]
        or remote_status.get("function_completion_sha256")
        != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["function_completion"][1]
        or remote_status.get("manifest_artifact_count") != 3
        or remote_status.get("manifest_integrity") != "PASS"
        or not isinstance(accounting, dict)
        or accounting.get("conservative_charged_gpu_seconds") != 1176.0
        or accounting.get("conservative_charged_wall_seconds") != 588.0
        or accounting.get("failure_evidence_sha256")
        != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["failure"][1]
        or accounting.get("ledger_sha256_after_settlement") != BUNDLED_LEDGER_RETRY_LEDGER_SHA256
        or not _exact_zero_int(accounting.get("active_reservations"))
        or not isinstance(status_cleanup, dict)
        or status_cleanup.get("in_function") != "FAIL_MISSING_PRE_CONTROLLER_CLEANUP_EVIDENCE"
        or status_cleanup.get("provider") != "PASS"
        or any(
            not _exact_zero_int(status_cleanup.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
    ):
        raise RuntimeError("Attempt-E terminal status is inconsistent")

    provider = evidence["provider_cleanup"]
    provider_observation = provider.get("provider_observation")
    provider_in_function = provider.get("in_function_cleanup")
    provider_remote = provider.get("remote_evidence")
    provider_failure = provider.get("failure_evidence")
    provider_ledger = provider.get("ledger")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or provider.get("function_call_id") != status.get("function_call_id")
        or provider.get("provider") != "Modal"
        or not isinstance(provider_observation, dict)
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(provider_in_function, dict)
        or provider_in_function.get("status") != "FAIL_MISSING_PRE_CONTROLLER_CLEANUP_EVIDENCE"
        or provider_in_function.get("artifact") is not None
        or not isinstance(provider_remote, dict)
        or provider_remote.get("remote_manifest_sha256")
        != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["remote_manifest"][1]
        or provider_remote.get("function_completion_sha256")
        != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["function_completion"][1]
        or not isinstance(provider_failure, dict)
        or provider_failure.get("sha256") != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["failure"][1]
        or provider_failure.get("conservative_charged_gpu_seconds") != 1176.0
        or not isinstance(provider_ledger, dict)
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_settlement") != BUNDLED_LEDGER_RETRY_LEDGER_SHA256
        or provider_ledger.get("conservative_consumed_gpu_seconds") != 20573.975013231975
    ):
        raise RuntimeError("Attempt-E provider cleanup is inconsistent")

    failure = evidence["failure"]
    failure_error = failure.get("error")
    if (
        failure.get("schema_version") != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or failure.get("status") != "CONSERVATIVELY_CHARGED"
        or failure.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or failure.get("reservation_id") != "exp004-v11-e-28c798cbedbf2232eee7"
        or failure.get("failure_stage") != "result-validation"
        or failure.get("actual_gpu_seconds") is not None
        or failure.get("actual_gpu_seconds_status") != "unavailable-or-not-trusted"
        or failure.get("charged_wall_seconds") != 588.0
        or failure.get("charged_gpu_seconds") != 1176.0
        or failure.get("accounting_policy")
        != "charge-full-preflight-bound-without-fabricated-measurement"
        or not isinstance(failure_error, dict)
        or failure_error
        != {
            "type": "ValueError",
            "message": "integrated result omits exactly two GPU inventory rows",
        }
    ):
        raise RuntimeError("Attempt-E conservative failure charge is inconsistent")

    remote_manifest = evidence["remote_manifest"]
    expected_remote_rows = [
        {
            "bytes": 4295,
            "relative_path": "function-completion.json",
            "sha256": BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["function_completion"][1],
        },
        {
            "bytes": 1903,
            "relative_path": "function-failure.json",
            "sha256": BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["function_failure"][1],
        },
        {
            "bytes": 513,
            "relative_path": "function-finally-cleanup.json",
            "sha256": BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["function_finally_cleanup"][1],
        },
    ]
    if (
        set(remote_manifest) != {"schema_version", "attempt_id", "remote_prefix", "artifacts"}
        or remote_manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or remote_manifest.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or remote_manifest.get("remote_prefix")
        != (f"experiment-004/v11/integrated/modal/{BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID}")
        or remote_manifest.get("artifacts") != expected_remote_rows
    ):
        raise RuntimeError("Attempt-E remote manifest is inconsistent")

    completion = evidence["function_completion"]
    run_error = completion.get("run_error")
    hardware = completion.get("hardware_comparability")
    if (
        completion.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        or completion.get("status") != "failed"
        or completion.get("scientific_status") != "invalid"
        or completion.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or completion.get("function_call_id") != status.get("function_call_id")
        or completion.get("reservation_id") != failure.get("reservation_id")
        or completion.get("config_sha256")
        != "f70232bdd6af527a87e633ae5777c355073bd54bb0d6d83adf0c97aaa8cd40cc"
        or completion.get("gpu_count") != 2
        or completion.get("requested_gpu") != "A100-80GB:2"
        or completion.get("in_function_cleanup") is not None
        or completion.get("controller")
        != {
            "schema_version": (
                "sloforge.branchfabric.experiment-004-v11-integrated-controller-failure/v1"
            ),
            "status": "failed",
        }
        or not isinstance(run_error, dict)
        or run_error.get("type") != "FileNotFoundError"
        or run_error.get("message")
        != (
            "[Errno 2] No such file or directory: '/opt/sloforge/artifacts/branchfabric/"
            "gpu-validation/experiment-004/gpu-hours.json'"
        )
        or "_verify_prefunction_retry_integrated_authorization"
        not in str(run_error.get("traceback"))
        or not isinstance(hardware, dict)
        or hardware.get("observed_gpu_names") != []
        or hardware.get("observed_gpu_uuids") != []
        or hardware.get("direct_v10_timing_comparable") is not False
    ):
        raise RuntimeError("Attempt-E function completion is inconsistent")

    function_failure = evidence["function_failure"]
    if (
        set(function_failure) != {"type", "message", "traceback"}
        or function_failure.get("type") != run_error.get("type")
        or function_failure.get("message") != run_error.get("message")
        or function_failure.get("traceback") != run_error.get("traceback")
    ):
        raise RuntimeError("Attempt-E function failure is inconsistent")

    finally_cleanup = evidence["function_finally_cleanup"]
    if (
        finally_cleanup.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-function-cleanup/v1"
        or finally_cleanup.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or finally_cleanup.get("controller_status") != "failed"
        or finally_cleanup.get("controller_cleanup_evidence_present") is not False
        or finally_cleanup.get("in_function_cleanup_pass") is not False
        or finally_cleanup.get("owned_child_processes_after") is not None
        or finally_cleanup.get("profiler_processes_after") is not None
        or finally_cleanup.get("forced_kills") is not None
    ):
        raise RuntimeError("Attempt-E function-finally cleanup is inconsistent")

    resource_audit = evidence["resource_audit"]
    audit_cleanup = resource_audit.get("cleanup_interpretation")
    audit_effect = resource_audit.get("authorization_effect")
    ordering = resource_audit.get("ordering_proof")
    findings = resource_audit.get("resource_findings")
    if (
        resource_audit.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-pre-controller-resource-audit/v1"
        or resource_audit.get("status")
        != "PASS_NO_CONTROLLER_TRANSACTION_RESOURCES_CREATED_WITH_CLEANUP_GAP_PRESERVED"
        or resource_audit.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or resource_audit.get("scientifically_valid") is not False
        or not isinstance(ordering, dict)
        or ordering.get("controller_python_invoked") is not True
        or ordering.get("controller_transaction_started") is not False
        or any(
            ordering.get(field) is not False
            for field in (
                "gpu_inventory_reached",
                "model_runtime_load_reached",
                "engine_creation_reached",
                "worker_process_launch_reached",
                "allocator_lifecycle_reached",
                "source_capture_reached",
                "optimized_export_reached",
                "source_release_reached",
            )
        )
        or not isinstance(findings, dict)
        or any(
            not _exact_zero_int(findings.get(field))
            for field in (
                "controller_owned_child_processes_created",
                "controller_owned_process_groups_created",
                "controller_cuda_contexts_created",
                "controller_allocator_instances_created",
                "controller_ipc_resources_created",
                "controller_profiler_processes_created",
                "provider_active_apps_after",
                "provider_running_tasks_after",
                "provider_running_containers_after",
                "provider_endpoints_after",
                "provider_queues_after",
                "provider_dicts_after",
                "provider_reservations_after",
                "ledger_reservations_after",
            )
        )
        or not isinstance(audit_cleanup, dict)
        or audit_cleanup.get("attempt_e_in_function_cleanup_status")
        != "FAIL_MISSING_PRE_CONTROLLER_CLEANUP_EVIDENCE"
        or audit_cleanup.get("attempt_e_provider_cleanup_status") != "PASS"
        or audit_cleanup.get("audit_is_not_in_function_cleanup_pass") is not True
        or not isinstance(audit_effect, dict)
        or audit_effect.get("gpu_invocation_authorized") is not False
        or audit_effect.get("attempt_f_seal_authorized") is not False
    ):
        raise RuntimeError("Attempt-E scoped resource audit is inconsistent")

    independent_review = evidence["independent_resource_review"]
    review_cleanup = independent_review.get("cleanup_classification")
    review_findings = independent_review.get("controller_owned_resource_findings")
    if (
        independent_review.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-attempt-e-pre-controller-resource-audit/v1"
        or independent_review.get("status") != "PASS_SCOPED_CONTROLLER_RESOURCE_NON_CREATION"
        or independent_review.get("attempt_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or independent_review.get("function_call_id") != status.get("function_call_id")
        or independent_review.get("review_decision")
        != "SCOPED_PROOF_ACCEPTED_WITH_CLEANUP_FAILURE_PRESERVED"
        or not isinstance(review_cleanup, dict)
        or review_cleanup.get("in_function_cleanup")
        != "FAIL_MISSING_PRE_CONTROLLER_CLEANUP_EVIDENCE"
        or review_cleanup.get("in_function_cleanup_pass") is not False
        or review_cleanup.get("retrospective_pass_forbidden") is not True
        or not isinstance(review_findings, dict)
        or review_findings.get("worker_processes_created") is not False
        or review_findings.get("controller_allocator_lifecycle_started") is not False
        or review_findings.get("source_capture_started") is not False
        or review_findings.get("optimized_export_started") is not False
        or review_findings.get("transaction_source_release_started") is not False
    ):
        raise RuntimeError("Attempt-E independent resource review is inconsistent")

    intervals = snapshot.get("intervals")
    charges = snapshot.get("conservative_failure_charges")
    reservations = snapshot.get("reservations")
    if (
        snapshot.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or snapshot.get("hard_additional_gpu_seconds") != 43200.0
        or snapshot.get("consumed_additional_gpu_seconds") != 20573.975013231975
        or reservations != []
        or not isinstance(intervals, list)
        or len(intervals) != 23
        or not isinstance(charges, list)
        or len(charges) != 6
        or budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or budget.get("authorized_ledger_sha256") != PREFUNCTION_RETRY_LEDGER_SHA256
        or budget.get("authorized_settled_intervals_sha256")
        != hashlib.sha256(_canonical_bytes(intervals)).hexdigest()
        or budget.get("authorized_failure_charges_sha256")
        != hashlib.sha256(_canonical_bytes(charges[:5])).hexdigest()
        or 43200.0 - 20573.975013231975 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-F immutable ledger snapshot or budget is inconsistent")
    e_charge = charges[-1]
    e_charge_evidence = e_charge.get("failure_evidence") if isinstance(e_charge, dict) else None
    if (
        not isinstance(e_charge, dict)
        or e_charge.get("invocation_id") != BUNDLED_LEDGER_FAILED_INTEGRATED_ATTEMPT_ID
        or e_charge.get("reservation_id") != failure.get("reservation_id")
        or e_charge.get("config_sha256") != completion.get("config_sha256")
        or e_charge.get("requested_gpu") != "A100-80GB"
        or e_charge.get("gpu_count") != 2
        or e_charge.get("charged_wall_seconds") != 588.0
        or e_charge.get("failure_stage") != "result-validation"
        or not isinstance(e_charge_evidence, dict)
        or e_charge_evidence.get("artifact_sha256")
        != BUNDLED_LEDGER_RETRY_EVIDENCE_BINDINGS["failure"][1]
        or e_charge_evidence.get("sample_selector") != "$"
    ):
        raise RuntimeError("Attempt-E conservative charge is not preserved in the snapshot")

    retry_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID
    )
    if retry_raw_root.exists() or retry_raw_root.is_symlink():
        raise RuntimeError("Attempt-F unexpectedly has pre-existing remote evidence")

    return {
        **authorization,
        "prior_attempt_evidence": verified_bindings,
        "immutable_ledger_snapshot_verified": True,
        "unchanged_target_b_verified": target_result["verified"],
        "target_b_identity_gate_passed": 1152,
        "prior_controller_transaction_started": False,
        "prior_in_function_cleanup": "FAIL_MISSING_PRE_CONTROLLER_CLEANUP_EVIDENCE",
        "prior_provider_cleanup": "PASS",
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "pre_worker_cleanup_contract_verified": True,
        "verified": True,
    }


def _verify_attempt_f_queue_accounting(
    partial: Mapping[str, Any], telemetry_rows: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Recompute Attempt F's coherent backlog without trusting its terminal summary."""

    offered = partial.get("global_offered_requests")
    gpu0 = partial.get("gpu0_requests")
    stale_gpu1_completed = partial.get("gpu1_incremental_completed_requests")
    queue_state = partial.get("queue_state")
    failure_ns = partial.get("observed_ns")
    if (
        partial.get("schema_version") != "sloforge.branchfabric.v10-gpu0-partial-failure/v1"
        or not isinstance(offered, list)
        or not isinstance(gpu0, list)
        or stale_gpu1_completed != []
        or partial.get("gpu1_telemetry_next_sequence") != 2
        or not isinstance(queue_state, dict)
        or type(failure_ns) is not int
        or any(not isinstance(row, dict) for row in (*offered, *gpu0))
    ):
        raise RuntimeError("Attempt-F partial serving failure is malformed")
    offered_by_id = {row.get("request_id"): row for row in offered}
    gpu0_by_id = {row.get("request_id"): row for row in gpu0}
    if (
        len(offered_by_id) != 187
        or None in offered_by_id
        or len(gpu0_by_id) != 167
        or None in gpu0_by_id
        or not set(gpu0_by_id).issubset(offered_by_id)
        or queue_state.get("total_outstanding") != 64
        or queue_state.get("driver_active_requests") != 44
    ):
        raise RuntimeError("Attempt-F raw queue-accounting inputs are inconsistent")
    gpu0_completed = {
        request_id: row
        for request_id, row in gpu0_by_id.items()
        if type(row.get("completed_ns")) is int
    }
    raw_stale_outstanding = len(offered_by_id) - len(gpu0_completed)
    if len(gpu0_completed) != 123 or raw_stale_outstanding != 64:
        raise RuntimeError("Attempt-F stale emergency value does not recompute to 64")

    if len(telemetry_rows) != 3:
        raise RuntimeError("Attempt-F queue replay requires telemetry sequences zero through two")
    gpu1_completed: dict[str, Mapping[str, Any]] = {}
    watermark: Mapping[str, Any] | None = None
    previous_observed_ns = -1
    for sequence, telemetry in enumerate(telemetry_rows):
        completed = telemetry.get("completed_requests")
        active = telemetry.get("active_requests")
        observed_ns = telemetry.get("observed_ns")
        if (
            telemetry.get("schema_version")
            != "sloforge.branchfabric.v10-gpu1-telemetry-incremental/v1"
            or telemetry.get("sequence") != sequence
            or type(observed_ns) is not int
            or observed_ns <= previous_observed_ns
            or not isinstance(completed, list)
            or not isinstance(active, list)
            or any(not isinstance(row, dict) for row in (*completed, *active))
        ):
            raise RuntimeError("Attempt-F GPU1 telemetry replay is malformed")
        previous_observed_ns = observed_ns
        for row in completed:
            request_id = row.get("request_id")
            completed_ns = row.get("completed_ns")
            if (
                not isinstance(request_id, str)
                or request_id in gpu1_completed
                or request_id not in offered_by_id
                or type(completed_ns) is not int
                or completed_ns > observed_ns
            ):
                raise RuntimeError("Attempt-F GPU1 completion replay is inconsistent")
            gpu1_completed[request_id] = row
        if telemetry.get("cumulative_completed_requests") != len(gpu1_completed):
            raise RuntimeError("Attempt-F GPU1 cumulative completion count is inconsistent")
        watermark = telemetry
    assert watermark is not None
    if set(gpu0_completed) & set(gpu1_completed):
        raise RuntimeError("Attempt-F request completion ownership is not disjoint")
    watermark_ns = watermark["observed_ns"]
    offered_at_watermark = {
        request_id
        for request_id, row in offered_by_id.items()
        if type(row.get("offered_ns")) is int and row["offered_ns"] <= watermark_ns
    }
    gpu0_completed_at_watermark = {
        request_id
        for request_id, row in gpu0_completed.items()
        if row["completed_ns"] <= watermark_ns
    }
    gpu1_completed_at_watermark = {
        request_id
        for request_id, row in gpu1_completed.items()
        if row["completed_ns"] <= watermark_ns
    }
    synchronized_outstanding = (
        len(offered_at_watermark)
        - len(gpu0_completed_at_watermark)
        - len(gpu1_completed_at_watermark)
    )
    if (
        watermark_ns != 199864068274
        or len(offered_at_watermark) != 179
        or len(gpu0_completed_at_watermark) != 119
        or len(gpu1_completed_at_watermark) != 8
        or synchronized_outstanding != 52
    ):
        raise RuntimeError("Attempt-F synchronized queue watermark does not recompute to 52")

    replay_events: list[tuple[int, int, str]] = []
    for request_id, row in offered_by_id.items():
        offered_ns = row.get("offered_ns")
        if type(offered_ns) is not int or offered_ns > failure_ns:
            raise RuntimeError("Attempt-F offered-request timestamp is invalid")
        replay_events.append((offered_ns, 1, str(request_id)))
    for request_id, row in {**gpu0_completed, **gpu1_completed}.items():
        completed_ns = row.get("completed_ns")
        if type(completed_ns) is not int or completed_ns > failure_ns:
            raise RuntimeError("Attempt-F completion timestamp is invalid")
        replay_events.append((completed_ns, -1, str(request_id)))
    outstanding = 0
    maximum = 0
    maximum_ns = 0
    for observed_ns, delta, _request_id in sorted(
        replay_events, key=lambda row: (row[0], 0 if row[1] < 0 else 1, row[2])
    ):
        outstanding += delta
        if outstanding < 0:
            raise RuntimeError("Attempt-F queue replay became negative")
        if outstanding > maximum:
            maximum = outstanding
            maximum_ns = observed_ns
    if maximum != 57 or maximum_ns != 198858810179 or maximum >= 64:
        raise RuntimeError("Attempt-F coherent queue replay does not prove true maximum 57")
    return {
        "raw_stale_outstanding": raw_stale_outstanding,
        "synchronized_outstanding": synchronized_outstanding,
        "true_maximum_outstanding": maximum,
        "true_maximum_observed_ns": maximum_ns,
    }


def _verify_prior_integrated_identity_evidence(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    config: Any,
    expected_attempt_id: str,
    expected_runtime_instance_id: str,
    label: str,
) -> dict[str, Any]:
    """Recompute both semantic identity gates for an immutable prior attempt."""

    capture = evidence["source_capture_commit"]
    pre_gate = evidence["pre_export_identity"]
    post_gate = evidence["post_export_identity"]
    commit = capture.get("commit")
    if not isinstance(commit, dict):
        raise RuntimeError(f"{label} SourceCaptureCommit is absent")
    semantic_sha = commit.get("semantic_sha256")
    if not isinstance(semantic_sha, str):
        raise RuntimeError(f"{label} SourceCaptureCommit semantic hash is absent")
    commit_result = _verify_post_target_source_commit(
        capture,
        expected_semantic_sha256=semantic_sha,
        config=config,
        expected_attempt_id=expected_attempt_id,
        expected_runtime_instance_id=expected_runtime_instance_id,
        identity_field="pre_export_identity",
        expected_gate_suffix="integrated-export",
    )
    semantic_true_fields = (
        "exact_logical_mapping",
        "exact_block_epoch_identity",
        "exact_owner_sets",
        "exact_refcounts",
        "all_allocations_live",
        "no_post_commit_mutation",
    )
    if (
        pre_gate.get("schema_version") != "sloforge.branchfabric.v11-source-identity-gate/v1"
        or pre_gate.get("event") != "SOURCE_ALLOCATION_IDENTITY_GATE"
        or pre_gate.get("attempt_id") != expected_attempt_id
        or pre_gate.get("source_capture_commit_sha256") != semantic_sha
        or pre_gate.get("expected_allocation_count") != 1152
        or pre_gate.get("observed_allocation_count") != 1152
        or pre_gate.get("post_commit_allocation_event_count") != 0
        or pre_gate.get("passed") is not True
        or any(pre_gate.get(field) is not True for field in semantic_true_fields)
        or pre_gate.get("optimized_export_started") is not False
        or pre_gate.get("transaction_source_release_started") is not False
    ):
        raise RuntimeError(f"{label} pre-export identity gate is inconsistent")
    before = post_gate.get("identity_before_export")
    after = post_gate.get("identity_after_export_read")
    pre_notifications = post_gate.get("post_commit_allocation_notifications")
    post_notifications = post_gate.get("post_export_allocation_notifications")
    if (
        post_gate.get("schema_version") != "sloforge.branchfabric.v11-source-identity-gate/v1"
        or post_gate.get("event") != "POST_EXPORT_READ_IDENTITY_GATE"
        or post_gate.get("source_capture_commit_sha256") != semantic_sha
        or post_gate.get("allocation_count") != 1152
        or post_gate.get("zero_post_commit_semantic_mutations") is not True
        or post_gate.get("passed") is not True
        or not isinstance(before, dict)
        or not isinstance(after, dict)
        or before.get("semantic_sha256") != semantic_sha
        or after.get("semantic_sha256") != semantic_sha
        or any(
            before.get(field) is not True
            for field in (*semantic_true_fields, "same_engine_step_witness", "passed")
        )
        or any(
            after.get(field) is not True
            for field in (*semantic_true_fields, "same_engine_step_witness", "passed")
        )
        or before.get("allocation_count") != 1152
        or after.get("allocation_count") != 1152
        or not isinstance(pre_notifications, dict)
        or not isinstance(post_notifications, dict)
        or not _exact_zero_int(pre_notifications.get("observed_event_count"))
        or not _exact_zero_int(post_notifications.get("observed_event_count"))
        or pre_notifications.get("queue_empty") is not True
        or post_notifications.get("queue_empty") is not True
    ):
        raise RuntimeError(f"{label} post-export identity gate is inconsistent")
    return {
        **commit_result,
        "pre_export_identity_gate_passed": 1152,
        "post_export_identity_gate_passed": 1152,
    }


def _verify_attempt_f_identity_evidence(
    evidence: Mapping[str, Mapping[str, Any]], *, config: Any
) -> dict[str, Any]:
    return _verify_prior_integrated_identity_evidence(
        evidence,
        config=config,
        expected_attempt_id=QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID,
        expected_runtime_instance_id=(
            "adapter:47685710773520:scheduler:47686101609952:manager:47687684112848"
        ),
        label="Attempt-F",
    )


def _verify_queue_accounting_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit Attempt G only from independently replayed immutable Attempt-F evidence."""

    authorization = manifest.get("post_queue_accounting_failure_integrated_authorization")
    expected_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "immutable_ledger_snapshot",
        "prior_attempt_evidence",
        "queue_accounting_fix_bindings",
        "unchanged_runtime_bindings",
        "independent_reviews",
        "expected_successful_integrated_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != expected_fields
        or authorization.get("schema_version")
        != (
            "sloforge.branchfabric.exp004-v11-post-queue-accounting-failure-"
            "integrated-authorization/v1"
        )
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id")
        != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation")
        != QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256
        or config.attempt_id != QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-G queue-accounting retry authorization is inconsistent")
    expected_success_contract = {
        "config_schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_success_contract:
        raise RuntimeError("Attempt-G successful integrated result contract is inconsistent")

    snapshot_binding = authorization.get("immutable_ledger_snapshot")
    snapshot_reference, snapshot_sha = QUEUE_ACCOUNTING_RETRY_LEDGER_SNAPSHOT_BINDING
    if (
        not isinstance(snapshot_binding, dict)
        or set(snapshot_binding) != {"artifact", "sha256"}
        or snapshot_binding.get("artifact") != snapshot_reference
        or snapshot_binding.get("sha256") != snapshot_sha
    ):
        raise RuntimeError("Attempt-G immutable ledger snapshot binding is malformed")
    snapshot = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=snapshot_reference,
            expected_sha256=snapshot_sha,
        ),
        label="Attempt-G immutable ledger snapshot",
    )

    raw_bindings = authorization.get("prior_attempt_evidence")
    if not isinstance(raw_bindings, dict) or set(raw_bindings) != set(
        QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS
    ):
        raise RuntimeError("Attempt-G prior-attempt evidence bindings are incomplete")
    evidence: dict[str, dict[str, Any]] = {}
    verified_bindings: dict[str, dict[str, str]] = {}
    for label, (
        expected_reference,
        expected_sha,
    ) in QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS.items():
        binding = raw_bindings.get(label)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"artifact", "sha256"}
            or binding.get("artifact") != expected_reference
            or binding.get("sha256") != expected_sha
        ):
            raise RuntimeError(f"Attempt-G prior {label} binding is malformed")
        path = validate_bound_artifact(
            repository_root,
            reference=expected_reference,
            expected_sha256=expected_sha,
        )
        evidence[label] = _load_json_object(path, label=f"Attempt-F {label} evidence")
        verified_bindings[label] = {"artifact": expected_reference, "sha256": expected_sha}

    remote_manifest = evidence["remote_manifest"]
    remote_rows = remote_manifest.get("artifacts")
    if (
        remote_manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or remote_manifest.get("attempt_id") != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or remote_manifest.get("remote_prefix")
        != (f"experiment-004/v11/integrated/modal/{QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID}")
        or not isinstance(remote_rows, list)
        or len(remote_rows) != 478
        or any(not isinstance(row, dict) for row in remote_rows)
    ):
        raise RuntimeError("Attempt-F remote manifest is inconsistent")
    remote_by_path = {row.get("relative_path"): row for row in remote_rows}
    if len(remote_by_path) != 478 or None in remote_by_path:
        raise RuntimeError("Attempt-F remote manifest contains duplicate paths")
    raw_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        f"{QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID}/"
    )
    for label, (reference, expected_sha) in QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS.items():
        if label in {"status", "provider_cleanup", "remote_manifest"}:
            continue
        if not reference.startswith(raw_prefix):
            raise RuntimeError(f"Attempt-G prior {label} is outside the immutable raw prefix")
        row = remote_by_path.get(reference.removeprefix(raw_prefix))
        if not isinstance(row, dict) or row.get("sha256") != expected_sha:
            raise RuntimeError(f"Attempt-F manifest does not bind {label}")

    def _verify_named_bindings(
        candidate: Any,
        expected: Mapping[str, tuple[str, str]],
        *,
        label: str,
    ) -> dict[str, dict[str, str]]:
        if not isinstance(candidate, dict) or set(candidate) != set(expected):
            raise RuntimeError(f"Attempt-G {label} bindings are incomplete")
        result: dict[str, dict[str, str]] = {}
        for name, (reference, digest) in expected.items():
            binding = candidate.get(name)
            if (
                not isinstance(binding, dict)
                or set(binding) != {"artifact", "sha256"}
                or binding.get("artifact") != reference
                or binding.get("sha256") != digest
            ):
                raise RuntimeError(f"Attempt-G {label} {name} binding is malformed")
            validate_bound_artifact(repository_root, reference=reference, expected_sha256=digest)
            result[name] = {"artifact": reference, "sha256": digest}
        return result

    verified_fix = _verify_named_bindings(
        authorization.get("queue_accounting_fix_bindings"),
        QUEUE_ACCOUNTING_FIX_BINDINGS,
        label="queue-accounting fix",
    )
    verified_unchanged = _verify_named_bindings(
        authorization.get("unchanged_runtime_bindings"),
        QUEUE_ACCOUNTING_UNCHANGED_RUNTIME_BINDINGS,
        label="unchanged runtime",
    )

    review_bindings = authorization.get("independent_reviews")
    if not isinstance(review_bindings, list) or len(review_bindings) != 2:
        raise RuntimeError("Attempt-G requires exactly two independent Attempt-F reviews")
    verified_reviews: list[dict[str, Any]] = []
    seen_reviewers: set[int] = set()
    for binding in review_bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "agent",
            "role",
            "status",
            "artifact",
            "sha256",
        }:
            raise RuntimeError("Attempt-G independent review binding is malformed")
        agent = binding.get("agent")
        expected_review = QUEUE_ACCOUNTING_RETRY_REVIEW_BINDINGS.get(agent)
        if (
            type(agent) is not int
            or agent in seen_reviewers
            or expected_review is None
            or binding.get("role") != expected_review[0]
            or binding.get("status") != "PASS"
            or binding.get("artifact") != expected_review[1]
            or binding.get("sha256") != expected_review[2]
        ):
            raise RuntimeError("Attempt-G independent review identity is inconsistent")
        review = _load_json_object(
            validate_bound_artifact(
                repository_root,
                reference=expected_review[1],
                expected_sha256=expected_review[2],
            ),
            label=f"Attempt-F agent {agent} review",
        )
        common_review_valid = bool(
            review.get("agent") == agent
            and review.get("role") == expected_review[0]
            and review.get("attempt_id") == QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
            and review.get("terminal_classification") == "HARDWARE_GATE_NOT_REACHED"
        )
        agent_review_valid = bool(
            (
                agent == 12
                and review.get("review_status") == "PASS"
                and review.get("attempt_f_scientifically_valid") is False
                and review.get("attempt_f_scientific_status") == "INVALID_QUEUE_ACCOUNTING_BUG"
                and review.get("decision") == "GO_FOR_SCOPED_QUEUE_FIX_CONDITIONAL_ATTEMPT_G"
            )
            or (
                agent == 13
                and review.get("review_status") == "PASS_RECONSTRUCTION"
                and review.get("function_call_id") == "fc-01M10B0TF4F8T8M300EA972H9M"
                and review.get("attempt_scientifically_valid") is False
                and review.get("attempt_classification") == "INVALID_QUEUE_ACCOUNTING_BUG"
            )
        )
        if not common_review_valid or not agent_review_valid:
            raise RuntimeError("Attempt-G independent review conclusion is inconsistent")
        seen_reviewers.add(agent)
        verified_reviews.append(dict(binding))
    if seen_reviewers != set(QUEUE_ACCOUNTING_RETRY_REVIEW_BINDINGS):
        raise RuntimeError("Attempt-G independent reviewer set is incomplete")

    identity_result = _verify_attempt_f_identity_evidence(evidence, config=config)
    telemetry = [evidence[f"gpu1_telemetry_{sequence:06d}"] for sequence in range(3)]
    queue_result = _verify_attempt_f_queue_accounting(evidence["partial_failure"], telemetry)

    status = evidence["status"]
    status_queue = status.get("queue_accounting")
    status_identity = status.get("source_identity")
    status_cleanup = status.get("cleanup")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-status/v1"
        or status.get("status") != "INVALID_QUEUE_ACCOUNTING_BUG"
        or status.get("scientifically_valid") is not False
        or status.get("attempt_id") != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("function_call_id") != "fc-01M10B0TF4F8T8M300EA972H9M"
        or status.get("controller_transaction_started") is not True
        or status.get("source_identity_gate_reached") is not True
        or status.get("optimized_export_read_completed") is not True
        or status.get("source_release_completed") is not True
        or status.get("hbm_reclaimed") is not True
        or status.get("gpu1_first_useful_reached") is not True
        or status.get("serving_slo_restored") is not False
        or status.get("restore_started") is not False
        or status.get("branch_continuation_started") is not False
        or status.get("terminal_classification") != "HARDWARE_GATE_NOT_REACHED"
        or not isinstance(status_queue, dict)
        or status_queue.get("raw_stale_total_outstanding") != queue_result["raw_stale_outstanding"]
        or status_queue.get("synchronized_total_outstanding")
        != queue_result["synchronized_outstanding"]
        or status_queue.get("event_reconciled_true_maximum_outstanding")
        != queue_result["true_maximum_outstanding"]
        or not isinstance(status_identity, dict)
        or status_identity.get("pre_export", {}).get("allocation_count") != 1152
        or status_identity.get("post_export", {}).get("allocation_count") != 1152
        or not isinstance(status_cleanup, dict)
        or status_cleanup.get("in_function") != "PASS"
        or status_cleanup.get("provider") != "PASS"
        or status.get("integrated_retry_authorized") is not False
        or status.get("kill_and_recompute_authorized") is not False
    ):
        raise RuntimeError("Attempt-F terminal status is inconsistent")

    provider = evidence["provider_cleanup"]
    provider_observation = provider.get("provider_observation")
    provider_in_function = provider.get("in_function_cleanup")
    provider_ledger = provider.get("ledger")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or provider.get("function_call_id") != status.get("function_call_id")
        or provider.get("provider") != "Modal"
        or not isinstance(provider_observation, dict)
        or provider_observation.get("stopped_apps") != 4
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("unexpected_persistent_volumes") != []
        or set(provider_observation.get("authorized_persistent_volumes", []))
        != {"sloforge-branchfabric-results", "sloforge-model-cache"}
        or not isinstance(provider_in_function, dict)
        or provider_in_function.get("status") != "PASS"
        or provider_in_function.get("sha256")
        != QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS["in_function_cleanup"][1]
        or not isinstance(provider_ledger, dict)
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_settlement") != QUEUE_ACCOUNTING_RETRY_LEDGER_SHA256
        or provider_ledger.get("attempt_actual_gpu_seconds") != 422.970453434
    ):
        raise RuntimeError("Attempt-F provider cleanup is inconsistent")

    cleanup = evidence["in_function_cleanup"]
    if (
        cleanup.get("schema_version") != "sloforge.branchfabric.in-function-cleanup/v1"
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or cleanup.get("cleanup_errors") != []
        or cleanup.get("compute_processes_after") != []
        or cleanup.get("cuda_released") is not True
        or cleanup.get("parent_reaped_all_owned_children") is not True
        or cleanup.get("surviving_children") != []
        or cleanup.get("surviving_process_groups") != []
        or cleanup.get("zombie_processes_after") != []
        or cleanup.get("profiler_processes_after") != []
        or cleanup.get("resource_tracker_processes_after") != []
        or cleanup.get("leaked_ipc_resources") != []
        or cleanup.get("forced_kills") != []
    ):
        raise RuntimeError("Attempt-F in-function cleanup is inconsistent")

    controller = evidence["controller_result"]
    completion = evidence["function_completion"]
    function_failure = evidence["function_failure"]
    finally_cleanup = evidence["function_finally_cleanup"]
    serving_failure = evidence["serving_failure"]
    if (
        controller.get("status") != "failed"
        or controller.get("attempt_id") != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or controller.get("terminal_phase") != "INTEGRATED_TRANSACTION"
        or controller.get("in_function_cleanup") != cleanup
        or controller.get("cleanup_error") is not None
        or controller.get("compute_processes_after") != []
        or completion.get("status") != "failed"
        or completion.get("scientific_status") != "invalid"
        or completion.get("attempt_id") != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or completion.get("function_call_id") != status.get("function_call_id")
        or completion.get("controller") != controller
        or completion.get("in_function_cleanup") != cleanup
        or function_failure.get("message") != "integrated v11 controller failed closed"
        or finally_cleanup.get("attempt_id") != QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
        or finally_cleanup.get("in_function_cleanup_pass") is not True
        or finally_cleanup.get("forced_kills") != []
        or finally_cleanup.get("owned_child_processes_after") != []
        or serving_failure.get("status") != "failed"
        or serving_failure.get("error", {}).get("message")
        != "integrated v11 backlog reached its emergency ceiling during two-gpu-recovery"
    ):
        raise RuntimeError("Attempt-F failure and cleanup lifecycle is inconsistent")

    intervals = snapshot.get("intervals")
    charges = snapshot.get("conservative_failure_charges")
    reservations = snapshot.get("reservations")
    if (
        snapshot.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or snapshot.get("hard_additional_gpu_seconds") != 43200.0
        or snapshot.get("consumed_additional_gpu_seconds") != 20996.945466665977
        or reservations != []
        or not isinstance(intervals, list)
        or not isinstance(charges, list)
        or budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or 43200.0 - 20996.945466665977 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-G immutable ledger snapshot or budget is inconsistent")
    interval_gpu_seconds = sum(
        float(row.get("accounted_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in intervals
        if isinstance(row, dict)
    )
    charge_gpu_seconds = sum(
        float(row.get("charged_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in charges
        if isinstance(row, dict)
    )
    if not math.isclose(
        interval_gpu_seconds + charge_gpu_seconds,
        snapshot["consumed_additional_gpu_seconds"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Attempt-G immutable ledger arithmetic is inconsistent")
    f_intervals = [
        row
        for row in intervals
        if isinstance(row, dict)
        and row.get("invocation_id") == QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    f_charges = [
        row
        for row in charges
        if isinstance(row, dict)
        and row.get("invocation_id") == QUEUE_ACCOUNTING_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    if len(f_intervals) != 1 or f_charges:
        raise RuntimeError(
            "Attempt-F must have exactly one measured interval and no failure charge"
        )
    f_interval = f_intervals[0]
    raw_manifest_ref, raw_manifest_sha = QUEUE_ACCOUNTING_RETRY_EVIDENCE_BINDINGS["remote_manifest"]
    if (
        f_interval.get("accounted_wall_seconds") != 211.485226717
        or f_interval.get("function_call_id") != status.get("function_call_id")
        or f_interval.get("gpu_count") != 2
        or f_interval.get("requested_gpu") != "A100-80GB"
        or f_interval.get("actual_gpu_models") != ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"]
        or f_interval.get("gpu_uuids")
        != [
            "GPU-9dd9d2fd-b7da-91cb-7f87-fe1376300384",
            "GPU-f2d1295d-3d0f-aac9-3165-8fe5f841b6b0",
        ]
        or f_interval.get("raw_manifest", {}).get("artifact_sha256") != raw_manifest_sha
        or not str(f_interval.get("raw_manifest", {}).get("artifact_reference", "")).endswith(
            raw_manifest_ref
        )
        or f_interval.get("raw_manifest", {}).get("sample_selector") != "$"
    ):
        raise RuntimeError("Attempt-F measured ledger interval is inconsistent")

    retry_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID
    )
    if retry_raw_root.exists() or retry_raw_root.is_symlink():
        raise RuntimeError("Attempt-G unexpectedly has pre-existing remote evidence")
    return {
        **authorization,
        "prior_attempt_evidence": verified_bindings,
        "queue_accounting_fix_bindings": verified_fix,
        "unchanged_runtime_bindings": verified_unchanged,
        "independent_reviews": verified_reviews,
        "immutable_ledger_snapshot_verified": True,
        "attempt_f_identity": identity_result,
        "attempt_f_queue_accounting": queue_result,
        "prior_in_function_cleanup": "PASS",
        "prior_provider_cleanup": "PASS",
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "verified": True,
    }


def _verify_control_gate_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit Attempt H only from immutable, independently checked Attempt-G evidence."""

    authorization = manifest.get("post_control_gate_failure_integrated_authorization")
    expected_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "immutable_ledger_snapshot",
        "prior_attempt_evidence",
        "frozen_v10_control_bindings",
        "control_gate_fix_bindings",
        "unchanged_runtime_bindings",
        "independent_reviews",
        "expected_successful_integrated_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != expected_fields
        or authorization.get("schema_version")
        != (
            "sloforge.branchfabric.exp004-v11-post-control-gate-failure-integrated-authorization/v1"
        )
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != CONTROL_GATE_RETRY_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation") != CONTROL_GATE_RETRY_LEDGER_SHA256
        or config.attempt_id != CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != CONTROL_GATE_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-H control-gate retry authorization is inconsistent")
    expected_success_contract = {
        "config_schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_success_contract:
        raise RuntimeError("Attempt-H successful integrated result contract is inconsistent")

    snapshot_binding = authorization.get("immutable_ledger_snapshot")
    snapshot_reference, snapshot_sha = CONTROL_GATE_RETRY_LEDGER_SNAPSHOT_BINDING
    if snapshot_binding != {"artifact": snapshot_reference, "sha256": snapshot_sha}:
        raise RuntimeError("Attempt-H immutable ledger snapshot binding is malformed")
    snapshot = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=snapshot_reference,
            expected_sha256=snapshot_sha,
        ),
        label="Attempt-H immutable ledger snapshot",
    )

    def verify_binding_group(
        candidate: Any,
        expected: Mapping[str, tuple[str, str]],
        *,
        label: str,
        load: bool = False,
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
        if not isinstance(candidate, dict) or set(candidate) != set(expected):
            raise RuntimeError(f"Attempt-H {label} bindings are incomplete")
        verified: dict[str, dict[str, str]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for name, (reference, digest) in expected.items():
            binding = candidate.get(name)
            if binding != {"artifact": reference, "sha256": digest}:
                raise RuntimeError(f"Attempt-H {label} {name} binding is malformed")
            path = validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=digest,
            )
            verified[name] = {"artifact": reference, "sha256": digest}
            if load:
                payloads[name] = _load_json_object(path, label=f"Attempt-H {label} {name}")
        return verified, payloads

    verified_evidence, evidence = verify_binding_group(
        authorization.get("prior_attempt_evidence"),
        CONTROL_GATE_RETRY_EVIDENCE_BINDINGS,
        label="prior-attempt evidence",
        load=True,
    )
    verified_baseline, baseline = verify_binding_group(
        authorization.get("frozen_v10_control_bindings"),
        CONTROL_GATE_FROZEN_V10_BINDINGS,
        label="frozen-v10 control",
        load=True,
    )
    verified_fix, _ = verify_binding_group(
        authorization.get("control_gate_fix_bindings"),
        CONTROL_GATE_FIX_BINDINGS,
        label="control-gate fix",
    )
    verified_unchanged, _ = verify_binding_group(
        authorization.get("unchanged_runtime_bindings"),
        CONTROL_GATE_UNCHANGED_RUNTIME_BINDINGS,
        label="unchanged runtime",
    )

    remote_manifest = evidence["remote_manifest"]
    remote_rows = remote_manifest.get("artifacts")
    if (
        remote_manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or remote_manifest.get("attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or remote_manifest.get("remote_prefix")
        != f"experiment-004/v11/integrated/modal/{CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID}"
        or not isinstance(remote_rows, list)
        or len(remote_rows) != 475
        or any(not isinstance(row, dict) for row in remote_rows)
    ):
        raise RuntimeError("Attempt-G remote manifest is inconsistent")
    remote_by_path = {row.get("relative_path"): row for row in remote_rows}
    if len(remote_by_path) != 475 or None in remote_by_path:
        raise RuntimeError("Attempt-G remote manifest contains duplicate paths")
    raw_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        f"{CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID}/"
    )
    raw_root = repository_root / raw_prefix
    for relative_path, row in remote_by_path.items():
        relative = Path(str(relative_path))
        artifact = raw_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or artifact.is_symlink()
            or not artifact.is_file()
            or type(row.get("bytes")) is not int
            or row.get("bytes") != artifact.stat().st_size
            or not isinstance(row.get("sha256"), str)
            or row.get("sha256") != hashlib.sha256(artifact.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Attempt-G remote manifest child failed exact verification")
    actual_paths = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*")
        if path.is_file() and path.name != "REMOTE_MANIFEST.json"
    }
    if actual_paths != set(remote_by_path):
        raise RuntimeError("Attempt-G remote manifest does not exactly cover raw evidence")
    for label, (reference, digest) in CONTROL_GATE_RETRY_EVIDENCE_BINDINGS.items():
        if label in {"status", "provider_cleanup", "remote_manifest"}:
            continue
        if not reference.startswith(raw_prefix):
            raise RuntimeError(f"Attempt-G {label} is outside immutable raw evidence")
        row = remote_by_path.get(reference.removeprefix(raw_prefix))
        if not isinstance(row, dict) or row.get("sha256") != digest:
            raise RuntimeError(f"Attempt-G remote manifest does not bind {label}")

    identity_result = _verify_prior_integrated_identity_evidence(
        evidence,
        config=config,
        expected_attempt_id=CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID,
        expected_runtime_instance_id=(
            "adapter:46856817715072:scheduler:46857487288128:manager:46876680338944"
        ),
        label="Attempt-G",
    )
    _validate_gpu1_scientific_result(evidence["rollout_result"])

    status = evidence["status"]
    status_identity = status.get("source_identity")
    status_release = status.get("source_release")
    status_recovery = status.get("serving_recovery")
    status_continuation = status.get("continuation")
    status_movement = status.get("movement")
    late_gate = status.get("late_control_gate")
    status_cleanup = status.get("cleanup")
    required_completed_phases = (
        "controller_transaction_started",
        "source_identity_gate_reached",
        "optimized_export_started",
        "optimized_export_read_completed",
        "transaction_source_release_started",
        "source_release_completed",
        "hbm_reclaimed",
        "gpu1_serving_enabled",
        "gpu1_first_useful_reached",
        "serving_slo_restored_before_restore",
        "restore_started",
        "restore_completed",
        "branch_continuation_started",
        "branch_continuation_completed",
    )
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-status/v1"
        or status.get("status") != "INVALID_POST_RESTORE_CONTROL_GATE_BUG"
        or status.get("scientifically_valid") is not False
        or status.get("attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("function_call_id") != "fc-01M10ESWY3XPGP145XTYNMEJT5"
        or any(status.get(field) is not True for field in required_completed_phases)
        or status.get("post_restore_control_validation_passed") is not False
        or status.get("failure_stage") != "post-restore-9-rps-control-validation"
        or status.get("terminal_classification") != "HARDWARE_GATE_NOT_REACHED"
        or status.get("v11_frozen") is not False
        or status.get("integrated_retry_authorized") is not False
        or status.get("kill_and_recompute_authorized") is not False
        or not isinstance(status_identity, dict)
        or status_identity.get("pre_export", {}).get("allocation_count") != 1152
        or status_identity.get("post_export", {}).get("allocation_count") != 1152
        or not isinstance(status_release, dict)
        or status_release.get("passed") is not True
        or status_release.get("post_free_record_count") != 1152
        or status_release.get("all_allocator_epochs_tombstoned") is not True
        or status_release.get("all_blocks_allocator_available") is not True
        or status_release.get("all_post_release_owner_sets_empty") is not True
        or status_release.get("all_post_release_refcounts_zero") is not True
        or status_release.get("full_free_pool_recovered") is not True
        or not isinstance(status_recovery, dict)
        or status_recovery.get("slo_restoration_passed") is not True
        or status_recovery.get("stable_window_count") != 5
        or not isinstance(status_continuation, dict)
        or status_continuation.get("exact_first_resumed_tokens") != 8
        or status_continuation.get("all_branches_resumed") is not True
        or status_continuation.get("gpu0_active_during_restore_protocol_pass") is not True
        or status_continuation.get("independent_recompute_first_tokens")
        != status_continuation.get("observed_first_tokens")
        or not isinstance(status_movement, dict)
        or status_movement.get("accounting_complete") is not True
        or status_movement.get("logical_state_bytes") != 1_056_964_608
        or status_movement.get("full_physical_bytes") != 22_197_063_040
        or status_movement.get("full_physical_amplification") != 21.000762818351625
        or not isinstance(late_gate, dict)
        or late_gate.get("gate_bug_proven") is not True
        or late_gate.get("attempt_g_result_artifact_present") is not False
        or late_gate.get("attempt_g_serialized_control_samples_present") is not False
        or not isinstance(status_cleanup, dict)
        or status_cleanup.get("in_function") != "PASS"
        or status_cleanup.get("provider") != "PASS"
    ):
        raise RuntimeError("Attempt-G terminal evidence is inconsistent")

    provider = evidence["provider_cleanup"]
    provider_observation = provider.get("provider_observation")
    provider_cleanup = provider.get("in_function_cleanup")
    provider_ledger = provider.get("ledger")
    if (
        provider.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-provider-cleanup/v1"
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or provider.get("function_call_id") != status.get("function_call_id")
        or not isinstance(provider_observation, dict)
        or provider_observation.get("stopped_apps") != 4
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(provider_cleanup, dict)
        or provider_cleanup.get("status") != "PASS"
        or provider_cleanup.get("sha256")
        != CONTROL_GATE_RETRY_EVIDENCE_BINDINGS["in_function_cleanup"][1]
        or not isinstance(provider_ledger, dict)
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_settlement") != CONTROL_GATE_RETRY_LEDGER_SHA256
        or provider_ledger.get("attempt_actual_gpu_seconds") != 484.434146276
    ):
        raise RuntimeError("Attempt-G provider cleanup is inconsistent")
    cleanup = evidence["in_function_cleanup"]
    if (
        cleanup.get("schema_version") != "sloforge.branchfabric.in-function-cleanup/v1"
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or cleanup.get("cuda_released") is not True
        or cleanup.get("parent_reaped_all_owned_children") is not True
        or any(
            cleanup.get(field) != []
            for field in (
                "cleanup_errors",
                "compute_processes_after",
                "surviving_children",
                "surviving_process_groups",
                "zombie_processes_after",
                "profiler_processes_after",
                "resource_tracker_processes_after",
                "leaked_ipc_resources",
                "forced_kills",
            )
        )
    ):
        raise RuntimeError("Attempt-G in-function cleanup is inconsistent")

    controller = evidence["controller_result"]
    completion = evidence["function_completion"]
    function_failure = evidence["function_failure"]
    finally_cleanup = evidence["function_finally_cleanup"]
    serving_failure = evidence["serving_failure"]
    if (
        controller.get("status") != "failed"
        or controller.get("attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or controller.get("terminal_phase") != "INTEGRATED_TRANSACTION"
        or controller.get("worker_returncodes") != {"serving": 1, "rollout": 0}
        or controller.get("worker_results") != []
        or controller.get("in_function_cleanup") != cleanup
        or controller.get("cleanup_error") is not None
        or controller.get("compute_processes_after") != []
        or completion.get("status") != "failed"
        or completion.get("scientific_status") != "invalid"
        or completion.get("attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or completion.get("function_call_id") != status.get("function_call_id")
        or completion.get("controller") != controller
        or completion.get("in_function_cleanup") != cleanup
        or function_failure.get("message") != "integrated v11 controller failed closed"
        or finally_cleanup.get("attempt_id") != CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        or finally_cleanup.get("in_function_cleanup_pass") is not True
        or finally_cleanup.get("forced_kills") != []
        or finally_cleanup.get("owned_child_processes_after") != []
        or serving_failure.get("status") != "failed"
        or serving_failure.get("error", {}).get("message")
        != "integrated v11 9-rps control interval was not stable"
    ):
        raise RuntimeError("Attempt-G late failure lifecycle is inconsistent")

    baseline_manifest = baseline["remote_manifest"]
    baseline_rows = baseline_manifest.get("artifacts")
    baseline_result = baseline["serving_result"]
    baseline_validity = baseline["scientific_validity"]
    recomputed_control = _control_interval_evidence(
        baseline_result,
        expected_rate_rps=9.0,
        expected_duration_seconds=5.0,
        slo_ttft_seconds=2.0,
        warmup_seconds=1.0,
        evaluation_seconds=1.0,
        minimum_completion_fraction=0.90,
    )
    baseline_result_ref, baseline_result_sha = CONTROL_GATE_FROZEN_V10_BINDINGS["serving_result"]
    baseline_raw_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/raw/modal/exp004-v10-naive-s41-v7/"
    )
    baseline_by_path = {
        row.get("relative_path"): row for row in baseline_rows or [] if isinstance(row, dict)
    }
    if (
        baseline_manifest.get("attempt_id") != "exp004-v10-naive-s41-v7"
        or not isinstance(baseline_rows, list)
        or len(baseline_by_path) != len(baseline_rows)
        or baseline_by_path.get(baseline_result_ref.removeprefix(baseline_raw_prefix), {}).get(
            "sha256"
        )
        != baseline_result_sha
        or baseline_result.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v10-serving-raw/v1"
        or baseline_result.get("status") != "succeeded"
        or baseline_validity.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v10-scientific-validity/v2"
        or baseline_validity.get("scientifically_valid") is not True
        or baseline_validity.get("invalid_reasons") != []
        or any(
            baseline_validity.get(field) is not True
            for field in (
                "bounded_backlog_pass",
                "gpu0_overload_pass",
                "queue_drain_pass",
                "slo_restoration_pass",
                "slo_stability_pass",
                "gpu0_active_during_restore_pass",
            )
        )
        or recomputed_control.get("passed") is not True
        or recomputed_control.get("full_arrivals") != 45
        or recomputed_control.get("arrivals") != 36
        or recomputed_control.get("completions_in_interval") != 36
        or recomputed_control.get("eventual_full_cohort_completions") != 45
        or recomputed_control.get("full_cohort_output_tokens_exact") is not True
        or recomputed_control.get("minimum_completion_fraction") != 0.90
    ):
        raise RuntimeError("Attempt-H frozen v10 control replay is inconsistent")

    review_bindings = authorization.get("independent_reviews")
    if not isinstance(review_bindings, list) or len(review_bindings) != 2:
        raise RuntimeError("Attempt-H requires exactly two independent Attempt-G reviews")
    reviews_by_agent = {
        binding.get("agent"): binding for binding in review_bindings if isinstance(binding, dict)
    }
    if len(reviews_by_agent) != 2 or set(reviews_by_agent) != set(
        CONTROL_GATE_RETRY_REVIEW_BINDINGS
    ):
        raise RuntimeError("Attempt-H independent reviewer set is incomplete")
    verified_reviews: list[dict[str, Any]] = []
    for agent, (role, reference, digest) in CONTROL_GATE_RETRY_REVIEW_BINDINGS.items():
        binding = reviews_by_agent[agent]
        if binding != {
            "agent": agent,
            "role": role,
            "status": "PASS",
            "artifact": reference,
            "sha256": digest,
        }:
            raise RuntimeError("Attempt-H independent review binding is inconsistent")
        review = _load_json_object(
            validate_bound_artifact(repository_root, reference=reference, expected_sha256=digest),
            label=f"Attempt-G Agent {agent} review",
        )
        common = (
            review.get("agent") == agent
            and review.get("role") == role
            and review.get("attempt_id") == CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
        )
        if agent == 12:
            invariant = review.get("corrected_control_invariant")
            valid = (
                review.get("schema_version")
                == "sloforge.branchfabric.experiment-004-v11-attempt-g-control-gate-review/v1"
                and review.get("review_status") == "PASS"
                and review.get("attempt_g_scientifically_valid") is False
                and review.get("attempt_g_scientific_status") == "INVALID_POSTHOC_CONTROL_GATE_BUG"
                and review.get("decision")
                == "CORRECT_SCOPED_CONTROL_GATE_AND_REQUIRE_COMPLETE_FAIL_CLOSED_EVIDENCE"
                and isinstance(invariant, dict)
                and invariant.get("name") == "BOUNDARY_CORRECTED_STABLE_CONTROL_V2"
                and review.get("final_verdict", {}).get("kill_and_recompute_authorized") is False
            )
        else:
            valid = (
                review.get("schema_version")
                == (
                    "sloforge.branchfabric.experiment-004-v11-attempt-g-"
                    "serving-methodology-review/v1"
                )
                and review.get("status") == "PASS"
                and review.get("function_call_id") == status.get("function_call_id")
                and review.get("attempt_g_scientifically_valid") is False
                and review.get("failure_classification") == "INVALID_POST_RESTORE_CONTROL_GATE_BUG"
                and review.get("preservation_transaction_evidence_complete") is True
                and review.get("gate_bug_proven") is True
                and review.get("fix_bindings") == authorization.get("control_gate_fix_bindings")
                and review.get("recommendation") == "AUTHORIZE_EXACTLY_ONE_INTEGRATED_RETRY_H"
                and review.get("kill_and_recompute_authorized") is False
            )
        if not common or not valid:
            raise RuntimeError("Attempt-H independent review conclusion is inconsistent")
        verified_reviews.append(dict(binding))

    intervals = snapshot.get("intervals")
    charges = snapshot.get("conservative_failure_charges")
    if (
        snapshot.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or snapshot.get("hard_additional_gpu_seconds") != 43200.0
        or snapshot.get("consumed_additional_gpu_seconds") != 21481.37961294198
        or snapshot.get("reservations") != []
        or not isinstance(intervals, list)
        or not isinstance(charges, list)
        or budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or 43200.0 - 21481.37961294198 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-H immutable ledger snapshot or budget is inconsistent")
    interval_gpu_seconds = sum(
        float(row.get("accounted_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in intervals
        if isinstance(row, dict)
    )
    charge_gpu_seconds = sum(
        float(row.get("charged_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in charges
        if isinstance(row, dict)
    )
    g_intervals = [
        row
        for row in intervals
        if isinstance(row, dict)
        and row.get("invocation_id") == CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    g_charges = [
        row
        for row in charges
        if isinstance(row, dict)
        and row.get("invocation_id") == CONTROL_GATE_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    if (
        not math.isclose(
            interval_gpu_seconds + charge_gpu_seconds,
            snapshot["consumed_additional_gpu_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or len(g_intervals) != 1
        or g_charges
        or g_intervals[0].get("accounted_wall_seconds") != 242.217073138
        or g_intervals[0].get("function_call_id") != status.get("function_call_id")
        or g_intervals[0].get("gpu_count") != 2
        or g_intervals[0].get("raw_manifest", {}).get("artifact_sha256")
        != CONTROL_GATE_RETRY_EVIDENCE_BINDINGS["remote_manifest"][1]
    ):
        raise RuntimeError("Attempt-G settled ledger interval is inconsistent")

    retry_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID
    )
    if retry_raw_root.exists() or retry_raw_root.is_symlink():
        raise RuntimeError("Attempt-H unexpectedly has pre-existing remote evidence")
    return {
        **authorization,
        "prior_attempt_evidence": verified_evidence,
        "frozen_v10_control_bindings": verified_baseline,
        "control_gate_fix_bindings": verified_fix,
        "unchanged_runtime_bindings": verified_unchanged,
        "independent_reviews": verified_reviews,
        "immutable_ledger_snapshot_verified": True,
        "attempt_g_identity": identity_result,
        "attempt_g_preservation_transaction_verified": True,
        "frozen_v10_control_replay": recomputed_control,
        "prior_in_function_cleanup": "PASS",
        "prior_provider_cleanup": "PASS",
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "verified": True,
    }


def _verify_image_closure_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit Attempt I only from exact H failure, cleanup, fix, and Target-B proof."""

    authorization = manifest.get("post_image_closure_failure_integrated_authorization")
    expected_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "immutable_ledger_snapshot",
        "prior_attempt_evidence",
        "image_closure_fix_bindings",
        "unchanged_allocator_movement_bindings",
        "target_b_identity_evidence",
        "independent_reviews",
        "expected_successful_integrated_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != expected_fields
        or authorization.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-post-image-closure-failure-integrated-authorization/v1"
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id")
        != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != IMAGE_CLOSURE_RETRY_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation")
        != IMAGE_CLOSURE_RETRY_LEDGER_SHA256
        or config.attempt_id != IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != IMAGE_CLOSURE_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-I image-closure retry authorization is inconsistent")
    expected_success_contract = {
        "config_schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_success_contract:
        raise RuntimeError("Attempt-I successful integrated result contract is inconsistent")

    def verify_binding_group(
        candidate: Any,
        expected: Mapping[str, tuple[str, str]],
        *,
        label: str,
        load: bool = False,
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
        if not isinstance(candidate, dict) or set(candidate) != set(expected):
            raise RuntimeError(f"Attempt-I {label} bindings are incomplete")
        verified: dict[str, dict[str, str]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for name, (reference, digest) in expected.items():
            binding = candidate.get(name)
            if binding != {"artifact": reference, "sha256": digest}:
                raise RuntimeError(f"Attempt-I {label} {name} binding is malformed")
            path = validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=digest,
            )
            verified[name] = dict(binding)
            if load:
                payloads[name] = _load_json_object(path, label=f"Attempt-I {label} {name}")
        return verified, payloads

    snapshot_reference, snapshot_sha = IMAGE_CLOSURE_RETRY_LEDGER_SNAPSHOT_BINDING
    if authorization.get("immutable_ledger_snapshot") != {
        "artifact": snapshot_reference,
        "sha256": snapshot_sha,
    }:
        raise RuntimeError("Attempt-I immutable ledger snapshot binding is malformed")
    snapshot = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=snapshot_reference,
            expected_sha256=snapshot_sha,
        ),
        label="Attempt-I immutable ledger snapshot",
    )
    verified_evidence, evidence = verify_binding_group(
        authorization.get("prior_attempt_evidence"),
        IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS,
        label="prior-attempt evidence",
        load=True,
    )
    verified_fix, _ = verify_binding_group(
        authorization.get("image_closure_fix_bindings"),
        IMAGE_CLOSURE_FIX_BINDINGS,
        label="image-closure fix",
    )
    verified_unchanged, _ = verify_binding_group(
        authorization.get("unchanged_allocator_movement_bindings"),
        IMAGE_CLOSURE_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS,
        label="unchanged allocator/movement",
    )
    verified_target_b, target_b = verify_binding_group(
        authorization.get("target_b_identity_evidence"),
        IMAGE_CLOSURE_TARGET_B_EVIDENCE_BINDINGS,
        label="Target-B identity",
        load=True,
    )

    remote_manifest = evidence["remote_manifest"]
    remote_rows = remote_manifest.get("artifacts")
    raw_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        f"{IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID}/"
    )
    raw_root = repository_root / raw_prefix
    if (
        remote_manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or remote_manifest.get("attempt_id") != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or remote_manifest.get("remote_prefix")
        != f"experiment-004/v11/integrated/modal/{IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID}"
        or not isinstance(remote_rows, list)
        or len(remote_rows) != 5
        or any(not isinstance(row, dict) for row in remote_rows)
    ):
        raise RuntimeError("Attempt-H remote manifest is inconsistent")
    remote_by_path = {row.get("relative_path"): row for row in remote_rows}
    if len(remote_by_path) != 5 or None in remote_by_path:
        raise RuntimeError("Attempt-H remote manifest contains duplicate paths")
    actual_paths = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*")
        if path.is_file() and path.name != "REMOTE_MANIFEST.json"
    }
    if actual_paths != set(remote_by_path):
        raise RuntimeError("Attempt-H remote manifest does not exactly cover raw evidence")
    for relative_path, row in remote_by_path.items():
        relative = Path(str(relative_path))
        artifact = raw_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or artifact.is_symlink()
            or not artifact.is_file()
            or type(row.get("bytes")) is not int
            or row.get("bytes") != artifact.stat().st_size
            or row.get("sha256") != hashlib.sha256(artifact.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Attempt-H remote manifest child failed exact verification")
    for name in (
        "controller_result",
        "function_completion",
        "function_failure",
        "function_finally_cleanup",
        "in_function_cleanup",
    ):
        reference, digest = IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS[name]
        row = remote_by_path.get(reference.removeprefix(raw_prefix))
        if not isinstance(row, dict) or row.get("sha256") != digest:
            raise RuntimeError(f"Attempt-H remote manifest does not bind {name}")

    status = evidence["status"]
    provider = evidence["provider_cleanup"]
    controller = evidence["controller_result"]
    completion = evidence["function_completion"]
    failure = evidence["function_failure"]
    finally_cleanup = evidence["function_finally_cleanup"]
    cleanup = evidence["in_function_cleanup"]
    charge = evidence["conservative_failure_charge"]
    terminal = status.get("terminal_decision")
    status_accounting = status.get("accounting")
    provider_observation = provider.get("provider_observation")
    provider_ledger = provider.get("ledger")
    empty_controller_fields = (
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
    if (
        status.get("status") != "INVALID_PRE_WORKER_MISSING_FROZEN_V10_REPLAY_IMAGE_CLOSURE"
        or status.get("scientifically_valid") is not False
        or status.get("attempt_id") != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("failure_stage") != "SEALED_EVIDENCE"
        or status.get("cleanup_scope") != "PRE_WORKER_PREFLIGHT"
        or any(
            status.get(field) is not False
            for field in (
                "controller_transaction_started",
                "worker_processes_started",
                "gpu_inventory_observed",
                "source_identity_gate_reached",
                "source_capture_commit_created",
                "optimized_export_started",
                "source_release_started",
                "restore_started",
                "branch_continuation_started",
            )
        )
        or not isinstance(terminal, dict)
        or terminal.get("classification") != "HARDWARE_GATE_NOT_REACHED"
        or terminal.get("v11_frozen") is not False
        or terminal.get("kill_and_recompute_authorized") is not False
        or not isinstance(status_accounting, dict)
        or status_accounting.get("ledger_sha256") != IMAGE_CLOSURE_RETRY_LEDGER_SHA256
        or status_accounting.get("attempt_charged_gpu_seconds") != 1176.0
        or status_accounting.get("failure_charge_sha256")
        != IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS["conservative_failure_charge"][1]
    ):
        raise RuntimeError("Attempt-H terminal status is inconsistent")
    if (
        provider.get("status") != "PASS"
        or provider.get("attempt_id") != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or provider.get("function_call_id") != status.get("function_call_id")
        or not isinstance(provider_observation, dict)
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("container_list") != []
        or provider_observation.get("endpoint_list") != []
        or provider_observation.get("queue_list") != []
        or provider_observation.get("dict_list") != []
        or provider_observation.get("authorized_persistent_volumes")
        != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(provider_ledger, dict)
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_charge") != IMAGE_CLOSURE_RETRY_LEDGER_SHA256
    ):
        raise RuntimeError("Attempt-H provider cleanup is inconsistent")
    controller_error = controller.get("controller_error")
    if (
        controller.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        or controller.get("status") != "failed"
        or controller.get("attempt_id") != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or controller.get("execution_mode") != "integrated-reclamation-v11"
        or controller.get("terminal_phase") != "INTEGRATED_TRANSACTION"
        or controller.get("cleanup_scope") != "PRE_WORKER_PREFLIGHT"
        or controller.get("failure_stage") != "SEALED_EVIDENCE"
        or controller.get("sealed_evidence") is not None
        or controller.get("stable_physical_gpu_identity") is not False
        or controller.get("readiness_deadline_ns") is not None
        or controller.get("sanity_guard_pair") is not None
        or controller.get("cleanup_error") is not None
        or any(controller.get(field) not in ([], {}) for field in empty_controller_fields)
        or not isinstance(controller_error, dict)
        or controller_error.get("type") != "FileNotFoundError"
        or controller_error.get("message")
        != (
            "[Errno 2] No such file or directory: "
            "'/opt/sloforge/artifacts/branchfabric/gpu-validation/experiment-004/raw'"
        )
        or controller.get("in_function_cleanup") != cleanup
        or completion.get("controller") != controller
        or completion.get("in_function_cleanup") != cleanup
    ):
        raise RuntimeError("Attempt-H PRE_WORKER controller failure is inconsistent")
    if (
        cleanup.get("schema_version") != "sloforge.branchfabric.in-function-cleanup/v1"
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or cleanup.get("owned_children") != []
        or cleanup.get("surviving_children") != []
        or cleanup.get("surviving_process_groups") != []
        or cleanup.get("profiler_processes_after") != []
        or cleanup.get("zombie_processes_after") != []
        or cleanup.get("leaked_ipc_resources") != []
        or cleanup.get("compute_processes_after") != []
        or cleanup.get("cleanup_errors") != []
        or cleanup.get("forced_kills") != []
        or cleanup.get("cuda_released") is not True
        or failure != completion.get("run_error")
        or finally_cleanup.get("attempt_id") != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or finally_cleanup.get("controller_status") != "failed"
        or finally_cleanup.get("in_function_cleanup_pass") is not True
    ):
        raise RuntimeError("Attempt-H failure cleanup chain is inconsistent")
    if (
        charge.get("schema_version") != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or charge.get("status") != "CONSERVATIVELY_CHARGED"
        or charge.get("attempt_id") != IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        or charge.get("failure_stage") != "result-validation"
        or charge.get("actual_gpu_seconds") is not None
        or charge.get("charged_wall_seconds") != 588.0
        or charge.get("charged_gpu_seconds") != 1176.0
        or charge.get("accounting_policy")
        != "charge-full-preflight-bound-without-fabricated-measurement"
    ):
        raise RuntimeError("Attempt-H conservative charge is inconsistent")

    intervals = snapshot.get("intervals")
    charges = snapshot.get("conservative_failure_charges")
    if (
        snapshot.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or snapshot.get("hard_additional_gpu_seconds") != 43200.0
        or snapshot.get("consumed_additional_gpu_seconds") != 22657.37961294198
        or snapshot.get("reservations") != []
        or not isinstance(intervals, list)
        or not isinstance(charges, list)
        or budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or 43200.0 - 22657.37961294198 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-I immutable ledger snapshot or budget is inconsistent")
    interval_gpu_seconds = sum(
        float(row.get("accounted_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in intervals
        if isinstance(row, dict)
    )
    charge_gpu_seconds = sum(
        float(row.get("charged_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in charges
        if isinstance(row, dict)
    )
    h_intervals = [
        row
        for row in intervals
        if isinstance(row, dict)
        and row.get("invocation_id") == IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    h_charges = [
        row
        for row in charges
        if isinstance(row, dict)
        and row.get("invocation_id") == IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    if (
        not math.isclose(
            interval_gpu_seconds + charge_gpu_seconds,
            snapshot["consumed_additional_gpu_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or h_intervals
        or len(h_charges) != 1
        or h_charges[0].get("charged_wall_seconds") != 588.0
        or h_charges[0].get("gpu_count") != 2
        or h_charges[0].get("failure_evidence", {}).get("artifact_sha256")
        != IMAGE_CLOSURE_RETRY_EVIDENCE_BINDINGS["conservative_failure_charge"][1]
    ):
        raise RuntimeError("Attempt-H ledger charge is inconsistent")

    target_status = target_b["status"]
    target_gate = target_b["identity_gate"]
    target_commit = target_b["source_capture_commit"]
    target_cleanup = target_b["in_function_cleanup"]
    target_identity = _verify_post_target_source_commit(
        target_commit,
        expected_semantic_sha256="28168c11daf69cdc89e1ddf369a8c828fc274315d37c234c480ba63c5ab67086",
        config=config,
    )
    if (
        target_status.get("status") != "PASS"
        or target_status.get("attempt_id") != RETRY_TARGETED_IDENTITY_ATTEMPT_ID
        or target_status.get("observed_allocation_count") != 1152
        or target_status.get("source_identity_gate_passed") is not True
        or target_status.get("optimized_export_started") is not False
        or target_status.get("transaction_source_release_started") is not False
        or target_gate.get("event") != "SOURCE_ALLOCATION_IDENTITY_GATE"
        or target_gate.get("passed") is not True
        or target_gate.get("observed_allocation_count") != 1152
        or target_gate.get("post_commit_allocation_event_count") != 0
        or target_gate.get("optimized_export_started") is not False
        or target_gate.get("transaction_source_release_started") is not False
        or target_cleanup.get("status") != "PASS"
        or target_cleanup.get("pass") is not True
        or target_identity.get("allocation_count") != 1152
        or target_identity.get("zero_post_commit_events") is not True
    ):
        raise RuntimeError("Target-B identity evidence is inconsistent")

    review_bindings = authorization.get("independent_reviews")
    if not isinstance(review_bindings, list) or len(review_bindings) != 2:
        raise RuntimeError("Attempt-I independent review bindings are incomplete")
    by_agent = {row.get("agent"): row for row in review_bindings if isinstance(row, dict)}
    if set(by_agent) != set(IMAGE_CLOSURE_RETRY_REVIEW_BINDINGS):
        raise RuntimeError("Attempt-I independent review agent set is inconsistent")
    verified_reviews: list[dict[str, Any]] = []
    for agent, (role, reference, digest) in IMAGE_CLOSURE_RETRY_REVIEW_BINDINGS.items():
        binding = by_agent[agent]
        if binding != {
            "agent": agent,
            "role": role,
            "status": "PASS",
            "artifact": reference,
            "sha256": digest,
        }:
            raise RuntimeError("Attempt-I independent review binding is inconsistent")
        review = _load_json_object(
            validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=digest,
            ),
            label=f"Attempt-I Agent {agent} review",
        )
        common = review.get("attempt_id") == IMAGE_CLOSURE_FAILED_INTEGRATED_ATTEMPT_ID
        if agent == 12:
            valid = (
                review.get("review_status") == "PASS"
                and review.get("attempt_h_scientifically_valid") is False
                and review.get("decision")
                == "GO_FOR_SCOPED_IMAGE_CLOSURE_AND_FAILURE_ADMISSION_FIX_CONDITIONAL_ATTEMPT_I"
            )
        else:
            valid = (
                review.get("status") == "PASS"
                and review.get("attempt_h_scientifically_valid") is False
                and review.get("scientific_work_started") is False
                and review.get("exact_pre_worker_failure_proven") is True
                and review.get("image_closure_bug_proven") is True
                and review.get("recommendation") == "AUTHORIZE_EXACTLY_ONE_INTEGRATED_RETRY_I"
                and review.get("kill_and_recompute_authorized") is False
            )
        if not common or not valid:
            raise RuntimeError("Attempt-I independent review conclusion is inconsistent")
        verified_reviews.append(dict(binding))

    retry_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID
    )
    if retry_raw_root.exists() or retry_raw_root.is_symlink():
        raise RuntimeError("Attempt-I unexpectedly has pre-existing remote evidence")
    return {
        **authorization,
        "prior_attempt_evidence": verified_evidence,
        "image_closure_fix_bindings": verified_fix,
        "unchanged_allocator_movement_bindings": verified_unchanged,
        "target_b_identity_evidence": verified_target_b,
        "independent_reviews": verified_reviews,
        "immutable_ledger_snapshot_verified": True,
        "attempt_h_pre_worker_failure_verified": True,
        "target_b_identity": target_identity,
        "prior_in_function_cleanup": "PASS",
        "prior_provider_cleanup": "PASS",
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "verified": True,
    }


def _verify_remote_prefix_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit Attempt J only from exact pre-reservation I failure and parser repair."""

    authorization = manifest.get("post_remote_prefix_parser_failure_integrated_authorization")
    expected_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "immutable_ledger_snapshot",
        "prior_attempt_evidence",
        "remote_prefix_parser_fix_bindings",
        "unchanged_scientific_runtime_bindings",
        "target_b_identity_evidence",
        "expected_successful_integrated_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != expected_fields
        or authorization.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-post-remote-prefix-parser-failure-integrated-authorization/v1"
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id")
        != REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != REMOTE_PREFIX_RETRY_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation")
        != REMOTE_PREFIX_RETRY_LEDGER_SHA256
        or config.attempt_id != REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != REMOTE_PREFIX_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-J remote-prefix retry authorization is inconsistent")

    expected_success_contract = {
        "config_schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_success_contract:
        raise RuntimeError("Attempt-J successful integrated result contract is inconsistent")

    def verify_group(
        candidate: Any,
        expected: Mapping[str, tuple[str, str]],
        *,
        label: str,
        load: bool = False,
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
        if not isinstance(candidate, dict) or set(candidate) != set(expected):
            raise RuntimeError(f"Attempt-J {label} bindings are incomplete")
        verified: dict[str, dict[str, str]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for name, (reference, digest) in expected.items():
            binding = candidate.get(name)
            if binding != {"artifact": reference, "sha256": digest}:
                raise RuntimeError(f"Attempt-J {label} {name} binding is malformed")
            path = validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=digest,
            )
            verified[name] = dict(binding)
            if load:
                payloads[name] = _load_json_object(path, label=f"Attempt-J {label} {name}")
        return verified, payloads

    snapshot_reference, snapshot_sha = REMOTE_PREFIX_RETRY_LEDGER_SNAPSHOT_BINDING
    if authorization.get("immutable_ledger_snapshot") != {
        "artifact": snapshot_reference,
        "sha256": snapshot_sha,
    }:
        raise RuntimeError("Attempt-J immutable ledger snapshot binding is malformed")
    snapshot = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=snapshot_reference,
            expected_sha256=snapshot_sha,
        ),
        label="Attempt-J immutable ledger snapshot",
    )
    verified_evidence, evidence = verify_group(
        authorization.get("prior_attempt_evidence"),
        REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS,
        label="prior-attempt evidence",
        load=True,
    )
    verified_fix, _ = verify_group(
        authorization.get("remote_prefix_parser_fix_bindings"),
        REMOTE_PREFIX_PARSER_FIX_BINDINGS,
        label="remote-prefix parser fix",
    )
    verified_unchanged, _ = verify_group(
        authorization.get("unchanged_scientific_runtime_bindings"),
        REMOTE_PREFIX_UNCHANGED_RUNTIME_BINDINGS,
        label="unchanged scientific/runtime",
    )
    verified_target_b, target_b = verify_group(
        authorization.get("target_b_identity_evidence"),
        IMAGE_CLOSURE_TARGET_B_EVIDENCE_BINDINGS,
        label="Target-B identity",
        load=True,
    )

    status = evidence["status"]
    inventory = evidence["remote_prefix_inventory"]
    prior_config = evidence["config"]
    prior_manifest = evidence["authorization_manifest"]
    prior_seal = evidence["seal_verification"]
    lifecycle = status.get("lifecycle")
    accounting = status.get("accounting")
    cleanup = status.get("cleanup")
    terminal = status.get("terminal_decision")
    sealed_inputs = status.get("sealed_inputs")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-terminal-status/v1"
        or status.get("status") != "INVALID_PRE_RESERVATION_REMOTE_PREFIX_SCHEMA_PARSER"
        or status.get("scientifically_valid") is not False
        or status.get("attempt_id") != REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("failure_stage") != "REMOTE_PREFIX_FRESHNESS"
        or status.get("observed_at_utc") is not None
        or not isinstance(lifecycle, dict)
        or lifecycle.get("coordinator_loader_passed") is not True
        or lifecycle.get("remote_read_only_inventory_invoked") is not True
        or lifecycle.get("remote_attempt_i_prefix_present") is not False
        or any(
            lifecycle.get(field) is not False
            for field in (
                "image_prebuild_started",
                "provider_app_created",
                "gpu_reservation_created",
                "gpu_function_invoked",
                "controller_python_invoked",
                "controller_transaction_started",
                "worker_processes_started",
                "gpu_inventory_observed",
                "source_identity_gate_reached",
                "optimized_export_started",
                "source_release_started",
                "restore_started",
                "branch_continuation_started",
            )
        )
        or not isinstance(accounting, dict)
        or accounting.get("ledger_sha256_before") != REMOTE_PREFIX_RETRY_LEDGER_SHA256
        or accounting.get("ledger_sha256_after") != REMOTE_PREFIX_RETRY_LEDGER_SHA256
        or not _exact_zero_int(accounting.get("active_reservations_before"))
        or not _exact_zero_int(accounting.get("active_reservations_after"))
        or not _exact_zero_int(accounting.get("attempt_gpu_seconds"))
        or accounting.get("attempt_cost_usd") != 0.0
        or accounting.get("ledger_mutated") is not False
        or cleanup
        != {
            "in_function_cleanup": "NOT_APPLICABLE_GPU_FUNCTION_NOT_INVOKED",
            "provider_cleanup": "NOT_APPLICABLE_PREBUILD_NOT_STARTED_AND_APP_NOT_CREATED",
            "provider_resources_created": False,
        }
        or not isinstance(terminal, dict)
        or terminal.get("classification") != "HARDWARE_GATE_NOT_REACHED"
        or terminal.get("v11_frozen") is not False
        or terminal.get("kill_and_recompute_authorized") is not False
        or terminal.get("attempt_i_reusable") is not False
        or terminal.get("next_attempt_id") != REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID
        or not isinstance(sealed_inputs, dict)
    ):
        raise RuntimeError("Attempt-I pre-reservation terminal status is inconsistent")
    for name in (
        "config",
        "authorization_manifest",
        "seal_verification",
        "remote_prefix_inventory",
    ):
        reference, digest = REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS[name]
        if sealed_inputs.get(name) != {"artifact": reference, "sha256": digest}:
            raise RuntimeError("Attempt-I sealed-input join is inconsistent")
    if sealed_inputs.get("coordinator") != {
        "artifact": "tools/branchfabric-experiment-004-v11-final.py",
        "sha256_at_attempt_i": ATTEMPT_I_SEALED_COORDINATOR_SHA256,
    }:
        raise RuntimeError("Attempt-I consumed coordinator identity is inconsistent")

    expected_inventory_fields = {
        "schema_version",
        "status",
        "attempt_id",
        "command",
        "returncode",
        "stdout_rows",
        "stderr",
        "attempt_prefix_present",
        "attempt_staging_prefix_present",
        "attempt_inflight_prefix_present",
        "observed_at_utc",
        "provenance",
    }
    rows = inventory.get("stdout_rows")
    expected_filenames = (
        "exp004-v11-integrated-s41-a",
        "exp004-v11-integrated-s41-b",
        "exp004-v11-integrated-s41-c",
        "exp004-v11-targeted-identity-s41-a",
        "exp004-v11-targeted-identity-s41-b",
        "exp004-v11-integrated-s41-e",
        "exp004-v11-integrated-s41-f",
        "exp004-v11-integrated-s41-g",
        "exp004-v11-integrated-s41-h",
    )
    expected_paths = [f"experiment-004/v11/integrated/modal/{name}" for name in expected_filenames]
    if (
        set(inventory) != expected_inventory_fields
        or inventory.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-remote-prefix-inventory/v1"
        or inventory.get("status") != "PASS_NO_ATTEMPT_PREFIX"
        or inventory.get("attempt_id") != REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID
        or inventory.get("command")
        != [
            "modal",
            "volume",
            "ls",
            "sloforge-branchfabric-results",
            "experiment-004/v11/integrated/modal",
            "--json",
        ]
        or not _exact_zero_int(inventory.get("returncode"))
        or inventory.get("stderr") != ""
        or inventory.get("observed_at_utc") is not None
        or any(
            inventory.get(field) is not False
            for field in (
                "attempt_prefix_present",
                "attempt_staging_prefix_present",
                "attempt_inflight_prefix_present",
            )
        )
        or not isinstance(rows, list)
        or len(rows) != len(expected_paths)
        or any(
            not isinstance(row, dict)
            or set(row) != {"filename", "type", "created_modified", "size"}
            or row.get("filename") != path
            or row.get("type") != "dir"
            or not isinstance(row.get("created_modified"), str)
            or not row.get("created_modified")
            or not isinstance(row.get("size"), str)
            or not row.get("size")
            for row, path in zip(rows, expected_paths, strict=True)
        )
    ):
        raise RuntimeError("Attempt-I remote-prefix inventory is inconsistent")

    if (
        prior_config.get("attempt_id") != REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID
        or prior_config.get("execution_mode") != "integrated-reclamation-v11"
        or prior_config.get("post_micro_review_manifest")
        != REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS["authorization_manifest"][0]
        or prior_config.get("post_micro_review_manifest_sha256")
        != REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS["authorization_manifest"][1]
        or prior_config.get("ledger_sha256_before_reservation") != REMOTE_PREFIX_RETRY_LEDGER_SHA256
        or prior_manifest.get("status") != "PASS"
        or prior_manifest.get("integrated_v11_admission") != "APPROVED_EXACTLY_ONCE"
        or not isinstance(
            prior_manifest.get("post_image_closure_failure_integrated_authorization"), dict
        )
        or prior_seal.get("status") != "PASS"
        or prior_seal.get("attempt_id") != REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID
        or prior_seal.get("config")
        != {
            "artifact": REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS["config"][0],
            "sha256": REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS["config"][1],
        }
        or prior_seal.get("authorization_manifest")
        != {
            "artifact": REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS["authorization_manifest"][0],
            "sha256": REMOTE_PREFIX_RETRY_EVIDENCE_BINDINGS["authorization_manifest"][1],
        }
        or prior_seal.get("paid_resources_invoked") is not False
    ):
        raise RuntimeError("Attempt-I sealed authorization chain is inconsistent")

    intervals = snapshot.get("intervals")
    charges = snapshot.get("conservative_failure_charges")
    if (
        snapshot.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or snapshot.get("hard_additional_gpu_seconds") != 43200.0
        or snapshot.get("consumed_additional_gpu_seconds") != 22657.37961294198
        or snapshot.get("reservations") != []
        or not isinstance(intervals, list)
        or not isinstance(charges, list)
        or any(
            isinstance(row, dict)
            and row.get("invocation_id") == REMOTE_PREFIX_FAILED_INTEGRATED_ATTEMPT_ID
            for row in (*intervals, *charges)
        )
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or 43200.0 - 22657.37961294198 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-J immutable ledger snapshot or budget is inconsistent")

    target_status = target_b["status"]
    target_gate = target_b["identity_gate"]
    target_commit = target_b["source_capture_commit"]
    target_identity = _verify_post_target_source_commit(
        target_commit,
        expected_semantic_sha256="28168c11daf69cdc89e1ddf369a8c828fc274315d37c234c480ba63c5ab67086",
        config=config,
    )
    if (
        target_status.get("status") != "PASS"
        or target_status.get("observed_allocation_count") != 1152
        or target_status.get("source_identity_gate_passed") is not True
        or target_gate.get("passed") is not True
        or target_gate.get("observed_allocation_count") != 1152
        or target_gate.get("post_commit_allocation_event_count") != 0
        or target_identity.get("allocation_count") != 1152
        or target_identity.get("zero_post_commit_events") is not True
    ):
        raise RuntimeError("Attempt-J Target-B identity evidence is inconsistent")

    retry_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID
    )
    if retry_raw_root.exists() or retry_raw_root.is_symlink():
        raise RuntimeError("Attempt-J unexpectedly has pre-existing remote evidence")
    return {
        **authorization,
        "prior_attempt_evidence": verified_evidence,
        "remote_prefix_parser_fix_bindings": verified_fix,
        "unchanged_scientific_runtime_bindings": verified_unchanged,
        "target_b_identity_evidence": verified_target_b,
        "target_b_identity": target_identity,
        "prior_attempt_pre_reservation_failure_verified": True,
        "immutable_ledger_snapshot_verified": True,
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "verified": True,
    }


def _verify_retained_capacity_retry_integrated_authorization(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    config: Experiment004V11IntegratedConfig,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit K only from exact J capacity drift and the same-allocation repair."""

    authorization = manifest.get("post_retained_capacity_failure_integrated_authorization")
    expected_fields = {
        "schema_version",
        "status",
        "retry_attempt_id",
        "prior_failed_attempt_id",
        "prior_settled_ledger_sha256",
        "ledger_sha256_before_reservation",
        "immutable_ledger_snapshot",
        "prior_attempt_evidence",
        "readiness_and_sanity_fix_bindings",
        "unchanged_scientific_runtime_bindings",
        "unchanged_allocator_movement_bindings",
        "target_b_identity_evidence",
        "independent_reviews",
        "expected_successful_integrated_contract",
    }
    if (
        not isinstance(authorization, dict)
        or set(authorization) != expected_fields
        or authorization.get("schema_version")
        != "sloforge.branchfabric.exp004-v11-post-retained-capacity-failure-integrated-authorization/v1"
        or authorization.get("status") != "APPROVED_EXACTLY_ONCE"
        or authorization.get("retry_attempt_id") != RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_failed_attempt_id")
        != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or authorization.get("prior_settled_ledger_sha256") != RETAINED_CAPACITY_RETRY_LEDGER_SHA256
        or authorization.get("ledger_sha256_before_reservation")
        != RETAINED_CAPACITY_RETRY_LEDGER_SHA256
        or config.attempt_id != RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID
        or config.ledger_sha256_before_reservation != RETAINED_CAPACITY_RETRY_LEDGER_SHA256
        or config.budget_authorization != PREFUNCTION_RETRY_BUDGET_AUTHORIZATION
    ):
        raise RuntimeError("Attempt-K retained-capacity retry authorization is inconsistent")

    expected_success_contract = {
        "config_schema_version": "sloforge.branchfabric.experiment-004-v11-integrated-config/v1",
        "execution_mode": "integrated-reclamation-v11",
        "completion_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-completion/v1"
        ),
        "remote_manifest_schema_version": (
            "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        ),
        "terminal_phase": "INTEGRATED_TRANSACTION",
        "remote_status": "provisional",
        "scientific_status": "pending-local-budget-settlement-and-provider-cleanup",
        "result_envelope_fields": ["materialized", "result"],
        "remote_completion_fields": sorted(SUCCESSFUL_INTEGRATED_COMPLETION_FIELDS),
        "materialized_fields": ["remote_path", "volume_name"],
        "run_error": None,
        "in_function_cleanup": "PASS",
        "full_export_authorized": True,
        "source_release_authorized": True,
    }
    if authorization.get("expected_successful_integrated_contract") != expected_success_contract:
        raise RuntimeError("Attempt-K successful integrated result contract is inconsistent")

    def verify_exact_group(
        candidate: Any,
        expected: Mapping[str, tuple[str, str]],
        *,
        label: str,
        load: bool = False,
    ) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
        if not isinstance(candidate, dict) or set(candidate) != set(expected):
            raise RuntimeError(f"Attempt-K {label} bindings are incomplete")
        verified: dict[str, dict[str, str]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for name, (reference, digest) in expected.items():
            binding = candidate.get(name)
            if binding != {"artifact": reference, "sha256": digest}:
                raise RuntimeError(f"Attempt-K {label} {name} binding is malformed")
            path = validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=digest,
            )
            verified[name] = dict(binding)
            if load:
                payloads[name] = _load_json_object(path, label=f"Attempt-K {label} {name}")
        return verified, payloads

    fix_candidate = authorization.get("readiness_and_sanity_fix_bindings")
    if not isinstance(fix_candidate, dict) or set(fix_candidate) != set(
        RETAINED_CAPACITY_FIX_BINDING_PATHS
    ):
        raise RuntimeError("Attempt-K readiness/sanity fix bindings are incomplete")
    verified_fix: dict[str, dict[str, str]] = {}
    for name, reference in RETAINED_CAPACITY_FIX_BINDING_PATHS.items():
        binding = fix_candidate.get(name)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"artifact", "sha256"}
            or binding.get("artifact") != reference
            or not isinstance(binding.get("sha256"), str)
            or len(binding["sha256"]) != 64
        ):
            raise RuntimeError(f"Attempt-K readiness/sanity fix {name} binding is malformed")
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=str(binding["sha256"]),
        )
        verified_fix[name] = dict(binding)

    snapshot_reference, snapshot_sha = RETAINED_CAPACITY_RETRY_LEDGER_SNAPSHOT_BINDING
    if authorization.get("immutable_ledger_snapshot") != {
        "artifact": snapshot_reference,
        "sha256": snapshot_sha,
    }:
        raise RuntimeError("Attempt-K immutable ledger snapshot binding is malformed")
    snapshot = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=snapshot_reference,
            expected_sha256=snapshot_sha,
        ),
        label="Attempt-K immutable ledger snapshot",
    )
    verified_evidence, evidence = verify_exact_group(
        authorization.get("prior_attempt_evidence"),
        RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS,
        label="prior-attempt evidence",
        load=True,
    )
    verified_runtime, _ = verify_exact_group(
        authorization.get("unchanged_scientific_runtime_bindings"),
        RETAINED_CAPACITY_UNCHANGED_RUNTIME_BINDINGS,
        label="unchanged scientific/runtime",
    )
    verified_movement, _ = verify_exact_group(
        authorization.get("unchanged_allocator_movement_bindings"),
        RETAINED_CAPACITY_UNCHANGED_ALLOCATOR_MOVEMENT_BINDINGS,
        label="unchanged allocator/movement",
    )
    verified_target_b, target_b = verify_exact_group(
        authorization.get("target_b_identity_evidence"),
        RETAINED_CAPACITY_TARGET_B_EVIDENCE_BINDINGS,
        label="Target-B identity",
        load=True,
    )

    raw_prefix = (
        "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw/"
        f"{RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID}/"
    )
    raw_root = repository_root / raw_prefix
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("Attempt-J remote manifest root is invalid")
    remote_manifest = evidence["remote_manifest"]
    remote_rows = remote_manifest.get("artifacts")
    if (
        remote_manifest.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-remote-manifest/v1"
        or remote_manifest.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or remote_manifest.get("remote_prefix")
        != (f"experiment-004/v11/integrated/modal/{RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID}")
        or not isinstance(remote_rows, list)
        or len(remote_rows) != 430
        or any(not isinstance(row, dict) for row in remote_rows)
    ):
        raise RuntimeError("Attempt-J remote manifest is inconsistent")
    remote_by_path = {row.get("relative_path"): row for row in remote_rows}
    raw_entries = tuple(raw_root.rglob("*"))
    if any(path.is_symlink() for path in raw_entries):
        raise RuntimeError("Attempt-J remote manifest tree contains a symlink")
    actual_paths = {
        path.relative_to(raw_root).as_posix()
        for path in raw_entries
        if path.is_file() and path.name != "REMOTE_MANIFEST.json"
    }
    if len(remote_by_path) != 430 or None in remote_by_path or actual_paths != set(remote_by_path):
        raise RuntimeError("Attempt-J remote manifest set equality failed")
    for relative_path, row in remote_by_path.items():
        relative = Path(str(relative_path))
        artifact = raw_root / relative
        if (
            set(row) != {"relative_path", "bytes", "sha256"}
            or relative.is_absolute()
            or ".." in relative.parts
            or artifact.is_symlink()
            or not artifact.is_file()
            or type(row.get("bytes")) is not int
            or row.get("bytes") != artifact.stat().st_size
            or row.get("sha256") != hashlib.sha256(artifact.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Attempt-J remote manifest child failed exact verification")
    for name, (reference, digest) in RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS.items():
        if name == "remote_manifest" or not reference.startswith(raw_prefix):
            continue
        row = remote_by_path.get(reference.removeprefix(raw_prefix))
        if not isinstance(row, dict) or row.get("sha256") != digest:
            raise RuntimeError(f"Attempt-J remote manifest does not bind {name}")

    readiness = evidence["readiness"]
    readiness_pair = readiness.get("engine_evidence")
    if (
        readiness.get("schema_version") != "sloforge.branchfabric.both-engines-ready/v1"
        or readiness.get("BOTH_ENGINES_READY") is not True
        or readiness.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or not isinstance(readiness_pair, list)
        or len(readiness_pair) != 2
        or readiness_pair[0] != evidence["serving_readiness"]
        or readiness_pair[1] != evidence["rollout_readiness"]
    ):
        raise RuntimeError("Attempt-J retained-engine readiness join is inconsistent")
    readiness_assessment = _assess_retained_engine_readiness_comparability(readiness_pair)
    if (
        readiness_assessment.get("status") != "BELOW_FLOOR"
        or readiness_assessment.get("readiness_comparable") is not False
        or [row.get("device") for row in readiness_assessment.get("devices", ())]
        != ["gpu0", "gpu1"]
        or [row.get("reported_request_throughput_rps") for row in readiness_assessment["devices"]]
        != [13.837854042338916, 12.97720271683092]
        or any(row.get("passed") is not False for row in readiness_assessment["devices"])
    ):
        raise RuntimeError("Attempt-J retained-engine capacity divergence is inconsistent")

    command = evidence["probe_command"]
    plan = command.get("plan")
    arrivals = plan.get("arrivals") if isinstance(plan, dict) else None
    gpu0 = evidence["probe_gpu0_raw"]
    gpu1 = evidence["probe_gpu1_raw"]
    observations = gpu0.get("observations")
    if (
        command.get("schema_version") != "sloforge.branchfabric.capacity-probe-command/v1"
        or command.get("reason") != "v11 preauthorized short stale-calibration sanity guard"
        or not isinstance(plan, dict)
        or plan.get("schema_version") != "sloforge.branchfabric.capacity-probe-plan/v1"
        or plan.get("probe_id") != "v11-sanity-12rps"
        or plan.get("seed") != 41
        or plan.get("topology") != "gpu0-only"
        or plan.get("configured_rate_rps") != 12.0
        or plan.get("output_tokens") != 64
        or type(plan.get("warmup_start_ns")) is not int
        or plan.get("measurement_start_ns") != plan["warmup_start_ns"] + 1_000_000_000
        or plan.get("measurement_end_ns") != plan["measurement_start_ns"] + 3_000_000_000
        or not isinstance(arrivals, list)
        or len(arrivals) != 48
        or any(
            not isinstance(row, dict)
            or row.get("global_sequence") != index
            or row.get("request_id") != f"v11-sanity-12rps-{index:05d}"
            or row.get("assigned_device") != "gpu0"
            or type(row.get("scheduled_arrival_ns")) is not int
            for index, row in enumerate(arrivals)
        )
        or gpu0.get("schema_version") != "sloforge.branchfabric.capacity-worker-probe/v1"
        or gpu0.get("probe_id") != "v11-sanity-12rps"
        or gpu0.get("device") != "gpu0"
        or not isinstance(observations, list)
        or len(observations) != 48
        or gpu1.get("schema_version") != "sloforge.branchfabric.capacity-worker-probe/v1"
        or gpu1.get("probe_id") != "v11-sanity-12rps"
        or gpu1.get("device") != "gpu1"
        or gpu1.get("observations") != []
    ):
        raise RuntimeError("Attempt-J 12-rps probe structure is inconsistent")
    for arrival, observation in zip(arrivals, observations, strict=True):
        if (
            observation.get("global_sequence") != arrival["global_sequence"]
            or observation.get("request_id") != arrival["request_id"]
            or observation.get("scheduled_arrival_ns") != arrival["scheduled_arrival_ns"]
            or observation.get("device") != "gpu0"
            or observation.get("requested_output_tokens") != 64
            or observation.get("terminal_state") not in {"completed", "aborted"}
        ):
            raise RuntimeError("Attempt-J 12-rps observation identity is inconsistent")
    measurement_completions = sum(
        type(row.get("completed_ns")) is int
        and plan["measurement_start_ns"] <= row["completed_ns"] <= plan["measurement_end_ns"]
        for row in observations
    )
    terminal_counts = Counter(row.get("terminal_state") for row in observations)
    if (
        measurement_completions != 20
        or terminal_counts != Counter({"completed": 31, "aborted": 17})
        or gpu0.get("engine_continuity")
        != {
            field: readiness_pair[0].get(field)
            for field in (
                "adapter_object_identity",
                "device",
                "engine_nonce",
                "engine_object_identity",
                "engine_reloaded",
                "engine_started_ns",
                "physical_gpu_uuid",
                "pid",
            )
        }
        or gpu1.get("engine_continuity")
        != {
            field: readiness_pair[1].get(field)
            for field in (
                "adapter_object_identity",
                "device",
                "engine_nonce",
                "engine_object_identity",
                "engine_reloaded",
                "engine_started_ns",
                "physical_gpu_uuid",
                "pid",
            )
        }
    ):
        raise RuntimeError("Attempt-J 12-rps probe replay is inconsistent")
    for device, reset, retained in zip(
        DEVICES,
        (evidence["probe_gpu0_reset"], evidence["probe_gpu1_reset"]),
        readiness_pair,
        strict=True,
    ):
        if (
            reset.get("schema_version") != "sloforge.branchfabric.capacity-request-reset/v1"
            or reset.get("probe_id") != "v11-sanity-12rps"
            or reset.get("device") != device
            or reset.get("passed") is not True
            or reset.get("prefix_cache_reset") is not True
            or reset.get("engine_reloaded") is not False
            or any(
                reset.get(field) != retained.get(field)
                for field in (
                    "pid",
                    "physical_gpu_uuid",
                    "engine_nonce",
                    "engine_object_identity",
                    "adapter_object_identity",
                    "engine_started_ns",
                )
            )
            or any(
                reset.get(state_name, {}).get(field) != 0
                for state_name in ("state_before", "state_after")
                for field in (
                    "request_count",
                    "running_requests",
                    "waiting_requests",
                    "skipped_waiting_requests",
                    "queue_depth",
                )
            )
        ):
            raise RuntimeError("Attempt-J 12-rps reset/continuity evidence is inconsistent")

    status = evidence["status"]
    provider = evidence["provider_cleanup"]
    controller = evidence["controller_result"]
    cleanup = evidence["in_function_cleanup"]
    completion = evidence["function_completion"]
    failure = evidence["function_failure"]
    finally_cleanup = evidence["function_finally_cleanup"]
    charge = evidence["failure_charge"]
    status_flags = {
        "function_entered": True,
        "gpu_allocation_observed": True,
        "controller_python_invoked": True,
        "worker_processes_started": True,
        "gpu_inventory_observed": True,
        "both_engines_ready": True,
        "sanity_12rps_started": True,
        "sanity_12rps_passed": False,
        "sanity_15rps_started": False,
        "integrated_transaction_command_published": False,
        "live_rollouts_created": False,
        "source_identity_gate_reached": False,
        "source_capture_commit_created": False,
        "optimized_export_started": False,
        "source_release_started": False,
        "hbm_reclaimed": False,
        "gpu1_serving_enabled": False,
        "restore_started": False,
        "branch_continuation_started": False,
    }
    function_call_id = status.get("function_call_id")
    controller_error = controller.get("controller_error")
    if (
        status.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-status/v1"
        or status.get("status") != "INVALID_PRE_TRANSACTION_12RPS_SANITY_GUARD"
        or status.get("scientifically_valid") is not False
        or status.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or status.get("failure_stage") != "SANITY_GUARD_12RPS"
        or any(status.get(field) is not expected for field, expected in status_flags.items())
        or status.get("blocker", {}).get("code") != "GPU0_ONLY_12RPS_SHORT_SANITY_GUARD_FAILED"
        or status.get("blocker", {}).get("metrics", {}).get("measurement_window_completions")
        != measurement_completions
        or status.get("blocker", {}).get("metrics", {}).get("terminal_completed_requests")
        != terminal_counts["completed"]
        or status.get("blocker", {}).get("metrics", {}).get("terminal_aborted_requests")
        != terminal_counts["aborted"]
        or controller.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        or controller.get("status") != "failed"
        or controller.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or controller.get("readiness_evidence") != readiness_pair
        or controller.get("sanity_guard_pair") is not None
        or controller.get("worker_results") != []
        or not isinstance(controller_error, dict)
        or controller_error.get("type") != "RuntimeError"
        or "v11 sanity guard failed" not in str(controller_error.get("message"))
        or completion.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or completion.get("status") != "failed"
        or completion.get("scientific_status") != "invalid"
        or completion.get("function_call_id") != function_call_id
        or completion.get("controller") != controller
        or completion.get("in_function_cleanup") != cleanup
        or completion.get("run_error") != failure
        or finally_cleanup.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or finally_cleanup.get("controller_status") != "failed"
        or finally_cleanup.get("in_function_cleanup_pass") is not True
    ):
        raise RuntimeError("Attempt-J terminal runtime chain is inconsistent")
    provider_observation = provider.get("provider_observation")
    provider_ledger = provider.get("ledger")
    if (
        cleanup.get("schema_version") != "sloforge.branchfabric.in-function-cleanup/v1"
        or cleanup.get("status") != "PASS"
        or cleanup.get("pass") is not True
        or cleanup.get("surviving_children") != []
        or cleanup.get("surviving_process_groups") != []
        or cleanup.get("zombie_processes_after") != []
        or cleanup.get("profiler_processes_after") != []
        or cleanup.get("resource_tracker_processes_after") != []
        or cleanup.get("leaked_threads") != []
        or cleanup.get("leaked_ipc_resources") != []
        or cleanup.get("compute_processes_after") != []
        or cleanup.get("cleanup_errors") != []
        or cleanup.get("forced_kills") != []
        or cleanup.get("cuda_released") is not True
        or provider.get("status") != "PASS"
        or provider.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or provider.get("function_call_id") != function_call_id
        or not isinstance(provider_observation, dict)
        or any(
            not _exact_zero_int(provider_observation.get(field))
            for field in (
                "active_apps",
                "running_tasks",
                "running_containers",
                "endpoints",
                "queues",
                "dicts",
                "provider_reservations",
            )
        )
        or provider_observation.get("container_list") != []
        or provider_observation.get("endpoint_list") != []
        or provider_observation.get("queue_list") != []
        or provider_observation.get("dict_list") != []
        or provider_observation.get("authorized_persistent_volumes")
        != ["sloforge-branchfabric-results", "sloforge-model-cache"]
        or provider_observation.get("unexpected_persistent_volumes") != []
        or not isinstance(provider_ledger, dict)
        or not _exact_zero_int(provider_ledger.get("active_reservations"))
        or provider_ledger.get("sha256_after_charge") != RETAINED_CAPACITY_RETRY_LEDGER_SHA256
    ):
        raise RuntimeError("Attempt-J cleanup/provider evidence is inconsistent")
    if (
        charge.get("schema_version") != "sloforge.branchfabric.exp004-v11-final-failure-charge/v1"
        or charge.get("status") != "CONSERVATIVELY_CHARGED"
        or charge.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or charge.get("failure_stage") != "verified-post-worker-failure-conservative-settlement"
        or charge.get("actual_gpu_seconds") is not None
        or charge.get("charged_wall_seconds") != 588.0
        or charge.get("charged_gpu_seconds") != 1176.0
        or charge.get("accounting_policy")
        != "charge-full-preflight-bound-without-fabricated-measurement"
        or charge.get("verified_remote_failure", {}).get("function_call_id") != function_call_id
        or charge.get("verified_remote_failure", {}).get("remote_manifest_sha256")
        != RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS["remote_manifest"][1]
    ):
        raise RuntimeError("Attempt-J conservative failure charge is inconsistent")

    intervals = snapshot.get("intervals")
    charges = snapshot.get("conservative_failure_charges")
    if (
        snapshot.get("schema_version") != "sloforge.branchfabric.experiment-004-gpu-hours/v1"
        or snapshot.get("hard_additional_gpu_seconds") != 43200.0
        or snapshot.get("consumed_additional_gpu_seconds") != 23833.37961294198
        or snapshot.get("reservations") != []
        or not isinstance(intervals, list)
        or not isinstance(charges, list)
        or budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds") != 43200.0
        or budget.get("authorized_gpu_budget_usd") != 80.0
        or 43200.0 - 23833.37961294198 < 1176.0 * 1.15
    ):
        raise RuntimeError("Attempt-K immutable ledger snapshot or budget is inconsistent")
    interval_gpu_seconds = sum(
        float(row.get("accounted_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in intervals
        if isinstance(row, dict)
    )
    charge_gpu_seconds = sum(
        float(row.get("charged_wall_seconds", -1)) * int(row.get("gpu_count", -1))
        for row in charges
        if isinstance(row, dict)
    )
    j_charges = [
        row
        for row in charges
        if isinstance(row, dict)
        and row.get("invocation_id") == RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
    ]
    if (
        not math.isclose(
            interval_gpu_seconds + charge_gpu_seconds,
            snapshot["consumed_additional_gpu_seconds"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or len(j_charges) != 1
        or j_charges[0].get("charged_wall_seconds") != 588.0
        or j_charges[0].get("gpu_count") != 2
        or j_charges[0].get("failure_evidence", {}).get("artifact_sha256")
        != RETAINED_CAPACITY_RETRY_EVIDENCE_BINDINGS["failure_charge"][1]
        or any(
            isinstance(row, dict)
            and row.get("invocation_id") == RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID
            for row in (*intervals, *charges)
        )
    ):
        raise RuntimeError("Attempt-J settled ledger accounting is inconsistent")

    review_candidate = authorization.get("independent_reviews")
    if not isinstance(review_candidate, list) or len(review_candidate) != 2:
        raise RuntimeError("Attempt-K independent reviews are incomplete")
    reviews_by_agent = {
        item.get("agent"): item for item in review_candidate if isinstance(item, dict)
    }
    if set(reviews_by_agent) != set(RETAINED_CAPACITY_RETRY_REVIEW_BINDINGS):
        raise RuntimeError("Attempt-K independent review agents are inconsistent")
    verified_reviews: list[dict[str, Any]] = []
    loaded_reviews: dict[int, dict[str, Any]] = {}
    for agent, (role, reference) in RETAINED_CAPACITY_RETRY_REVIEW_BINDINGS.items():
        candidate = reviews_by_agent[agent]
        digest = candidate.get("sha256")
        expected_binding = {
            "agent": agent,
            "role": role,
            "artifact": reference,
            "sha256": digest,
            "status": "PASS",
        }
        if (
            candidate != expected_binding
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"Attempt-K agent{agent} review binding is malformed")
        loaded_reviews[agent] = _load_json_object(
            validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=digest,
            ),
            label=f"Attempt-K agent{agent} review",
        )
        verified_reviews.append(expected_binding)
    agent12 = loaded_reviews[12]
    next_constraints = agent12.get("required_next_run_constraints")
    agent13 = loaded_reviews[13]
    same_allocation_fix = agent13.get("same_allocation_fix")
    agent13_fix_rows = agent13.get("fix_bindings")
    approved_projection = agent13.get("approved_runtime_projection")
    expected_agent13_fix_rows = [
        {
            "label": "same_allocation_controller",
            "artifact": ("experiments/branchfabric/gpu_reclamation_integrated_controller_v11.py"),
            "sha256": ("59f6eed10855f5957618a054f0cb089f2b443c1eb5d9764d882bdf7661bdcb44"),
        },
        {
            "label": "three_probe_scoped_worker_wrapper",
            "artifact": "experiments/branchfabric/gpu_reclamation_integrated_worker_v11.py",
            "sha256": ("a9312a017116ffc86ac311b578da67f2afb97198815f30369c8ee3e0159c2002"),
        },
        {
            "label": "controller_and_readiness_tests",
            "artifact": "tests/python/test_gpu_reclamation_integrated_controller_v11.py",
            "sha256": ("a4c5acdad7377e6fdf0fcfbd87129ff0d7cc75213d8a1b99ec724d8f68d885c2"),
        },
        {
            "label": "same_allocation_reproduction_tests",
            "artifact": ("tests/python/test_gpu_reclamation_integrated_sanity_reproduction_v11.py"),
            "sha256": ("18d086cd5f204aeca66a0c2894a9ce5dc5f20e61715579058777a6eadc648abe"),
        },
    ]
    current_projection = _retained_capacity_runtime_projection(repository_root)
    if (
        agent12.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-agent12-attempt-j-runtime-capacity-review/v1"
        or agent12.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or agent12.get("scientifically_valid") is not False
        or agent12.get("verdict") != "REAL_RUNTIME_CAPACITY_DRIFT"
        or agent12.get("terminal_classification") != "HARDWARE_GATE_NOT_REACHED"
        or not isinstance(next_constraints, dict)
        or next_constraints.get("readiness_comparator_minimum_batch_16_rps_each_engine") != 15.0
        or next_constraints.get("readiness_comparator_must_recompute_from_timestamps") is not True
        or next_constraints.get("only_pp_may_continue") is not True
        or next_constraints.get("original_15rps_overload_guard_still_required") is not True
        or agent13.get("schema_version")
        != "sloforge.branchfabric.experiment-004-v11-attempt-j-scientific-methodology-review/v1"
        or agent13.get("status") != "PASS"
        or agent13.get("agent") != 13
        or agent13.get("attempt_id") != RETAINED_CAPACITY_FAILED_INTEGRATED_ATTEMPT_ID
        or agent13.get("attempt_j_scientifically_valid") is not False
        or agent13.get("failure_classification")
        != "INVALID_PRE_TRANSACTION_REAL_RUNTIME_CAPACITY_DRIFT"
        or agent13.get("terminal_classification") != "HARDWARE_GATE_NOT_REACHED"
        or not isinstance(same_allocation_fix, dict)
        or same_allocation_fix.get("status") != "PASS_CPU_VALIDATED"
        or same_allocation_fix.get("readiness_batch16_minimum_rps_each_engine") != 15.0
        or same_allocation_fix.get("readiness_recomputed_from_timestamps") is not True
        or same_allocation_fix.get("readiness_assessment_persisted_before_gate") is not True
        or same_allocation_fix.get("only_pass_pass_continues") is not True
        or same_allocation_fix.get("unchanged_15rps_guard_runs_after_pass_pass") is not True
        or agent13_fix_rows != expected_agent13_fix_rows
        or approved_projection != current_projection
        or verified_fix["three_probe_scoped_worker_wrapper"]
        != {
            "artifact": expected_agent13_fix_rows[1]["artifact"],
            "sha256": expected_agent13_fix_rows[1]["sha256"],
        }
        or verified_fix["same_allocation_reproduction_tests"]
        != {
            "artifact": expected_agent13_fix_rows[3]["artifact"],
            "sha256": expected_agent13_fix_rows[3]["sha256"],
        }
        or agent13.get("recommendation")
        != "AUTHORIZE_EXACTLY_ONE_INTEGRATED_RETRY_K_AFTER_CONTENT_ADDRESSED_RESEAL"
        or agent13.get("kill_and_recompute_authorized") is not False
    ):
        raise RuntimeError("Attempt-K independent review conclusions are inconsistent")

    target_status = target_b["status"]
    target_gate = target_b["identity_gate"]
    target_identity = _verify_post_target_source_commit(
        target_b["source_capture_commit"],
        expected_semantic_sha256="28168c11daf69cdc89e1ddf369a8c828fc274315d37c234c480ba63c5ab67086",
        config=config,
    )
    if (
        target_status.get("status") != "PASS"
        or target_status.get("observed_allocation_count") != 1152
        or target_status.get("source_identity_gate_passed") is not True
        or target_gate.get("passed") is not True
        or target_gate.get("observed_allocation_count") != 1152
        or target_gate.get("post_commit_allocation_event_count") != 0
        or target_identity.get("allocation_count") != 1152
        or target_identity.get("zero_post_commit_events") is not True
    ):
        raise RuntimeError("Attempt-K Target-B identity evidence is inconsistent")

    retry_raw_root = (
        repository_root
        / "artifacts/branchfabric/gpu-validation/experiment-004/v11-final/integrated/raw"
        / RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID
    )
    if retry_raw_root.exists() or retry_raw_root.is_symlink():
        raise RuntimeError("Attempt-K unexpectedly has pre-existing remote evidence")
    return {
        **authorization,
        "prior_attempt_evidence": verified_evidence,
        "readiness_and_sanity_fix_bindings": verified_fix,
        "unchanged_scientific_runtime_bindings": verified_runtime,
        "unchanged_allocator_movement_bindings": verified_movement,
        "target_b_identity_evidence": verified_target_b,
        "target_b_identity": target_identity,
        "independent_reviews": verified_reviews,
        "attempt_j_readiness_assessment": readiness_assessment,
        "attempt_j_measurement_window_completions": measurement_completions,
        "attempt_j_terminal_counts": dict(terminal_counts),
        "approved_runtime_projection": current_projection,
        "prior_in_function_cleanup": "PASS",
        "prior_provider_cleanup": "PASS",
        "immutable_ledger_snapshot_verified": True,
        "expanded_budget_verified": True,
        "successful_integrated_contract_verified": True,
        "verified": True,
    }


def verify_sealed_evidence(
    config: Experiment004V11IntegratedConfig, repository_root: Path
) -> dict[str, Any]:
    """Verify every content-addressed prerequisite before GPU inventory."""

    references = {
        "offline": (config.offline_gate_manifest, config.offline_gate_manifest_sha256),
        "micro": (config.micro_validation_artifact, config.micro_validation_sha256),
        "post_micro": (
            config.post_micro_review_manifest,
            config.post_micro_review_manifest_sha256,
        ),
        "budget": (config.budget_authorization, config.budget_authorization_sha256),
    }
    paths = {
        name: validate_bound_artifact(
            repository_root, reference=reference, expected_sha256=expected_hash
        )
        for name, (reference, expected_hash) in references.items()
    }
    offline = _load_json_object(paths["offline"], label="offline gate manifest")
    micro = _load_json_object(paths["micro"], label="micro-validation artifact")
    post_micro = _load_json_object(paths["post_micro"], label="post-micro review manifest")
    budget = _load_json_object(paths["budget"], label="budget authorization")

    gates = offline.get("production_integration_gates")
    gate_values: list[Any] = []
    if isinstance(gates, dict):
        gate_values = list(gates.values())
    elif isinstance(gates, list):
        gate_values = [item.get("status") if isinstance(item, dict) else item for item in gates]
    if (
        offline.get("status") != "PASS"
        or len(gate_values) != 4
        or not all(
            _status_is_pass(item.get("status") if isinstance(item, dict) else item)
            for item in gate_values
        )
    ):
        raise RuntimeError("offline manifest does not seal all four production gates PASS")
    if isinstance(gates, list):
        for item in gates:
            if not isinstance(item, dict):
                raise RuntimeError("offline production gate commitment is malformed")
            reference = item.get("artifact")
            expected_hash = item.get("sha256")
            if not isinstance(reference, str) or not isinstance(expected_hash, str):
                raise RuntimeError("offline production gate lacks artifact/hash commitment")
            validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=expected_hash,
            )

    correctness = micro.get("correctness")
    required_correctness = (
        "allocator_epoch_pass",
        "post_free_ownership_pass",
        "fresh_destination_allocations_pass",
        "destination_mapping_commitment_pass",
        "integrity_pass",
        "movement_accounting_pass",
        "all_branches_resumed",
    )
    movement = micro.get("movement")
    if (
        micro.get("status") != "MICRO_VALIDATION_PASS"
        or micro.get("scientifically_admissible_measurement") is not True
        or micro.get("integrated_v11_allowed") is not True
        or not isinstance(correctness, dict)
        or any(correctness.get(name) is not True for name in required_correctness)
        or correctness.get("first_resumed_token_exact_matches") != 8
        or correctness.get("first_resumed_token_total") != 8
        or not isinstance(movement, dict)
        or not isinstance(movement.get("full_physical_amplification"), (int, float))
        or isinstance(movement.get("full_physical_amplification"), bool)
        or not math.isfinite(float(movement["full_physical_amplification"]))
        or float(movement["full_physical_amplification"]) > 25.0
    ):
        raise RuntimeError("micro-validation artifact does not admit integrated v11")

    micro_attempt = micro.get("attempt_id")
    reviews = _review_entries(post_micro)
    agents = {_agent_number(item) for item in reviews}
    if (
        post_micro.get("status") != "PASS"
        or post_micro.get("integrated_v11_admission") != "APPROVED_EXACTLY_ONCE"
        or post_micro.get("attempt_id") != micro_attempt
        or len(reviews) != 4
        or agents != {9, 10, 11, 12}
        or any(not _status_is_pass(item.get("status")) for item in reviews)
    ):
        raise RuntimeError("post-micro manifest does not seal PASS reviews 9-12")
    for entry in reviews:
        reference = entry.get("artifact", entry.get("path"))
        expected_hash = entry.get("sha256")
        if not isinstance(reference, str) or not isinstance(expected_hash, str):
            raise RuntimeError("post-micro review entry lacks artifact/hash commitment")
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_hash,
        )

    prelaunch = post_micro.get("integrated_prelaunch")
    required_prelaunch_labels = set(INTEGRATED_PRELAUNCH_BINDINGS)
    if not isinstance(prelaunch, dict) or prelaunch.get("status") != "PASS":
        raise RuntimeError("post-micro manifest does not seal integrated prelaunch PASS")
    prelaunch_bindings = prelaunch.get("bindings")
    if not isinstance(prelaunch_bindings, list):
        raise RuntimeError("integrated prelaunch bindings are malformed")
    by_label: dict[str, dict[str, Any]] = {}
    for entry in prelaunch_bindings:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            raise RuntimeError("integrated prelaunch binding is malformed")
        label = str(entry["label"])
        if label in by_label:
            raise RuntimeError("integrated prelaunch binding labels are not unique")
        by_label[label] = entry
    if set(by_label) != required_prelaunch_labels:
        raise RuntimeError("integrated prelaunch bindings are incomplete")
    for label, entry in by_label.items():
        reference = entry.get("artifact")
        expected_hash = entry.get("sha256")
        if not isinstance(reference, str) or not isinstance(expected_hash, str):
            raise RuntimeError("integrated prelaunch binding lacks artifact/hash commitment")
        if reference != INTEGRATED_PRELAUNCH_BINDINGS[label]:
            raise RuntimeError("integrated prelaunch binding path substitution")
        validate_bound_artifact(
            repository_root,
            reference=reference,
            expected_sha256=expected_hash,
        )
    make_check = _load_json_object(
        validate_bound_artifact(
            repository_root,
            reference=str(by_label["make_check"]["artifact"]),
            expected_sha256=str(by_label["make_check"]["sha256"]),
        ),
        label="integrated prelaunch make-check evidence",
    )
    if make_check.get("status") != "PASS" or make_check.get("command") != "make check":
        raise RuntimeError("integrated prelaunch make check is not PASS")

    targeted = config.execution_mode == "targeted-source-identity-v11"
    replacement = post_micro.get("integrated_replacement_authorization")
    final_pre_gpu_gate: dict[str, Any] | None = None
    targeted_authorization: dict[str, Any] | None = None
    post_target_authorization: dict[str, Any] | None = None
    prefunction_retry_authorization: dict[str, Any] | None = None
    bundled_ledger_retry_authorization: dict[str, Any] | None = None
    queue_accounting_retry_authorization: dict[str, Any] | None = None
    control_gate_retry_authorization: dict[str, Any] | None = None
    image_closure_retry_authorization: dict[str, Any] | None = None
    remote_prefix_retry_authorization: dict[str, Any] | None = None
    retained_capacity_retry_authorization: dict[str, Any] | None = None
    if targeted:
        candidate = post_micro.get("targeted_source_identity_authorization")
        expected_prior_attempt = {
            INITIAL_TARGETED_IDENTITY_ATTEMPT_ID: REPLACEMENT_INTEGRATED_ATTEMPT_ID,
            RETRY_TARGETED_IDENTITY_ATTEMPT_ID: INITIAL_TARGETED_IDENTITY_ATTEMPT_ID,
        }.get(config.attempt_id)
        if (
            not isinstance(candidate, dict)
            or expected_prior_attempt is None
            or candidate.get("status") != "APPROVED_EXACTLY_ONCE"
            or candidate.get("attempt_id") != config.attempt_id
            or candidate.get("prior_failed_attempt_id") != expected_prior_attempt
            or candidate.get("root_cause_class") != "GATE_BUG"
            or candidate.get("terminal_phase") != "SOURCE_ALLOCATION_IDENTITY_GATE"
            or candidate.get("full_export_authorized") is not False
            or candidate.get("source_release_authorized") is not False
            or candidate.get("ledger_sha256_before_reservation")
            != config.ledger_sha256_before_reservation
        ):
            raise RuntimeError("targeted source-identity authorization is inconsistent")
        bindings = candidate.get("offline_bindings")
        if not isinstance(bindings, list) or len(bindings) < 4:
            raise RuntimeError("targeted source-identity authorization lacks offline bindings")
        verified_bindings: list[dict[str, str]] = []
        for binding in bindings:
            if (
                not isinstance(binding, dict)
                or not isinstance(binding.get("artifact"), str)
                or not isinstance(binding.get("sha256"), str)
            ):
                raise RuntimeError("targeted source-identity binding is malformed")
            validate_bound_artifact(
                repository_root,
                reference=str(binding["artifact"]),
                expected_sha256=str(binding["sha256"]),
            )
            verified_bindings.append(
                {"artifact": str(binding["artifact"]), "sha256": str(binding["sha256"])}
            )
        retry_evidence: dict[str, Any] | None = None
        if config.attempt_id == RETRY_TARGETED_IDENTITY_ATTEMPT_ID:
            retry_candidate = candidate.get("prior_targeted_attempt_evidence")
            if not isinstance(retry_candidate, dict):
                raise RuntimeError("targeted retry lacks prior targeted-attempt evidence")
            status_binding = retry_candidate.get("status")
            cleanup_binding = retry_candidate.get("cleanup")
            for label, binding, expected_reference in (
                ("status", status_binding, PRIOR_TARGETED_RETRY_STATUS),
                ("cleanup", cleanup_binding, PRIOR_TARGETED_RETRY_CLEANUP),
            ):
                if (
                    not isinstance(binding, dict)
                    or binding.get("artifact") != expected_reference
                    or not isinstance(binding.get("sha256"), str)
                ):
                    raise RuntimeError(f"targeted retry prior {label} binding is malformed")
                validate_bound_artifact(
                    repository_root,
                    reference=expected_reference,
                    expected_sha256=str(binding["sha256"]),
                )
            prior_status = _load_json_object(
                repository_root / PRIOR_TARGETED_RETRY_STATUS,
                label="prior targeted-attempt status",
            )
            prior_cleanup = _load_json_object(
                repository_root / PRIOR_TARGETED_RETRY_CLEANUP,
                label="prior targeted-attempt cleanup",
            )
            cleanup_summary = prior_status.get("cleanup")
            blocker = prior_status.get("blocker")
            measured_invocation = prior_status.get("measured_invocation")
            provider = prior_cleanup.get("provider_observation")
            ledger = prior_cleanup.get("ledger")
            in_function = prior_cleanup.get("in_function_cleanup")
            if (
                prior_status.get("attempt_id") != INITIAL_TARGETED_IDENTITY_ATTEMPT_ID
                or prior_status.get("status") != "FAIL_PRE_IDENTITY_COMMAND_SCHEMA"
                or prior_status.get("source_identity_gate_reached") is not False
                or prior_status.get("source_identity_gate_passed") is not None
                or prior_status.get("optimized_export_started") is not False
                or prior_status.get("transaction_source_release_started") is not False
                or not isinstance(blocker, dict)
                or blocker.get("code")
                != "TARGETED_COMMAND_SCHEMA_REJECTED_BY_SHARED_WORKER_VALIDATOR"
                or not isinstance(cleanup_summary, dict)
                or cleanup_summary.get("in_function") != "PASS"
                or cleanup_summary.get("provider") != "PASS"
                or not isinstance(measured_invocation, dict)
                or prior_cleanup.get("status") != "PASS"
                or prior_cleanup.get("attempt_id") != INITIAL_TARGETED_IDENTITY_ATTEMPT_ID
                or measured_invocation.get("function_call_id")
                != prior_cleanup.get("function_call_id")
                or not isinstance(provider, dict)
                or any(
                    type(provider.get(field)) is not int or provider.get(field) != 0
                    for field in (
                        "active_apps",
                        "running_tasks",
                        "running_containers",
                        "endpoints",
                        "provider_reservations",
                    )
                )
                or not isinstance(ledger, dict)
                or type(ledger.get("active_reservations")) is not int
                or ledger.get("active_reservations") != 0
                or ledger.get("sha256_after_settlement") != config.ledger_sha256_before_reservation
                or not isinstance(in_function, dict)
                or in_function.get("status") != "PASS"
                or in_function.get("artifact")
                != "raw/exp004-v11-targeted-identity-s41-a/in_function_cleanup.json"
                or not isinstance(in_function.get("sha256"), str)
            ):
                raise RuntimeError("prior targeted-attempt evidence is inconsistent")
            validate_bound_artifact(
                repository_root,
                reference=PRIOR_TARGETED_RETRY_IN_FUNCTION_CLEANUP,
                expected_sha256=str(in_function["sha256"]),
            )
            retry_evidence = {
                "status": status_binding,
                "cleanup": cleanup_binding,
                "verified": True,
            }
        targeted_authorization = {
            **candidate,
            "offline_bindings": verified_bindings,
            "prior_targeted_attempt_evidence": retry_evidence,
            "verified": True,
        }
    elif config.attempt_id == POST_TARGET_INTEGRATED_ATTEMPT_ID:
        post_target_authorization = _verify_post_target_integrated_authorization(
            post_micro,
            repository_root,
            config=config,
        )
    elif config.attempt_id == PREFUNCTION_RETRY_INTEGRATED_ATTEMPT_ID:
        prefunction_retry_authorization = _verify_prefunction_retry_integrated_authorization(
            post_micro,
            repository_root,
            config=config,
            budget=budget,
        )
    elif config.attempt_id == BUNDLED_LEDGER_RETRY_INTEGRATED_ATTEMPT_ID:
        bundled_ledger_retry_authorization = _verify_bundled_ledger_retry_integrated_authorization(
            post_micro,
            repository_root,
            config=config,
            budget=budget,
        )
    elif config.attempt_id == QUEUE_ACCOUNTING_RETRY_INTEGRATED_ATTEMPT_ID:
        queue_accounting_retry_authorization = (
            _verify_queue_accounting_retry_integrated_authorization(
                post_micro,
                repository_root,
                config=config,
                budget=budget,
            )
        )
    elif config.attempt_id == CONTROL_GATE_RETRY_INTEGRATED_ATTEMPT_ID:
        control_gate_retry_authorization = _verify_control_gate_retry_integrated_authorization(
            post_micro,
            repository_root,
            config=config,
            budget=budget,
        )
    elif config.attempt_id == IMAGE_CLOSURE_RETRY_INTEGRATED_ATTEMPT_ID:
        image_closure_retry_authorization = _verify_image_closure_retry_integrated_authorization(
            post_micro,
            repository_root,
            config=config,
            budget=budget,
        )
    elif config.attempt_id == REMOTE_PREFIX_RETRY_INTEGRATED_ATTEMPT_ID:
        remote_prefix_retry_authorization = _verify_remote_prefix_retry_integrated_authorization(
            post_micro,
            repository_root,
            config=config,
            budget=budget,
        )
    elif config.attempt_id == RETAINED_CAPACITY_RETRY_INTEGRATED_ATTEMPT_ID:
        retained_capacity_retry_authorization = (
            _verify_retained_capacity_retry_integrated_authorization(
                post_micro,
                repository_root,
                config=config,
                budget=budget,
            )
        )
    elif replacement is None and config.attempt_id != ORIGINAL_INTEGRATED_ATTEMPT_ID:
        raise RuntimeError("replacement config lacks replacement-specific authorization")
    elif replacement is not None:
        if (
            not isinstance(replacement, dict)
            or replacement.get("status") != "APPROVED_EXACTLY_ONCE"
            or config.attempt_id != REPLACEMENT_INTEGRATED_ATTEMPT_ID
            or replacement.get("replacement_attempt_id") != config.attempt_id
            or replacement.get("prior_attempt_id") != PRIOR_INVALID_INTEGRATED_ATTEMPT_ID
            or replacement.get("prior_attempt_id") == config.attempt_id
            or replacement.get("prior_attempt_scientific_status")
            != "INVALID_PRE_RECLAIM_BOUNDED_BACKLOG_ABORT"
            or replacement.get("ledger_sha256_before_reservation")
            != config.ledger_sha256_before_reservation
        ):
            raise RuntimeError("integrated replacement authorization is inconsistent")
        final_pre_gpu_gate = _verify_final_pre_gpu_gate(
            post_micro,
            repository_root,
            attempt_id=config.attempt_id,
        )
        for label in ("prior_attempt_cleanup",):
            binding = replacement.get(label)
            if not isinstance(binding, dict) or binding.get("status") != "PASS":
                raise RuntimeError(f"integrated replacement {label} is not PASS")
            reference = binding.get("artifact")
            expected_hash = binding.get("sha256")
            if not isinstance(reference, str) or not isinstance(expected_hash, str):
                raise RuntimeError(f"integrated replacement {label} lacks artifact/hash")
            if reference != REPLACEMENT_EVIDENCE_BINDINGS[label]:
                raise RuntimeError(f"integrated replacement {label} path substitution")
            evidence_path = validate_bound_artifact(
                repository_root,
                reference=reference,
                expected_sha256=expected_hash,
            )
            evidence = _load_json_object(evidence_path, label=f"replacement {label}")
            if label == "prior_attempt_cleanup":
                provider = evidence.get("provider_cleanup")
                settlement = evidence.get("settlement")
                remote = evidence.get("remote_evidence")
                if (
                    evidence.get("schema_version")
                    != "sloforge.branchfabric.experiment-004-v11-integrated-postflight/v1"
                    or evidence.get("status")
                    != "PASS_CLEANUP_SCIENTIFICALLY_INVALID_BUDGET_CHARGED"
                    or evidence.get("attempt_id") != PRIOR_INVALID_INTEGRATED_ATTEMPT_ID
                    or evidence.get("scientific_status")
                    != "INVALID_PRE_RECLAIM_BOUNDED_BACKLOG_ABORT"
                    or evidence.get("scientifically_valid_integrated_transaction") is not False
                    or evidence.get("failure_stage")
                    != "PRE_RECLAIM_GPU0_OVERLOAD_BACKLOG_SAFETY_ABORT"
                    or evidence.get("reclaim_trigger_reached") is not False
                    or evidence.get("v11_export_started") is not False
                    or evidence.get("v11_release_started") is not False
                    or evidence.get("v11_import_started") is not False
                    or evidence.get("cleanup_pass") is not True
                    or not isinstance(provider, dict)
                    or provider.get("pass") is not True
                    or any(
                        provider.get(field) != 0
                        for field in (
                            "running_tasks",
                            "running_containers",
                            "active_apps",
                            "endpoints",
                            "provider_reservations",
                            "owned_child_processes",
                            "profilers",
                        )
                    )
                    or not isinstance(settlement, dict)
                    or settlement.get("active_ledger_reservations_after") != 0
                    or settlement.get("ledger_sha256_after_settlement")
                    != config.ledger_sha256_before_reservation
                    or not isinstance(remote, dict)
                    or remote.get("manifest_verification_pass") is not True
                ):
                    raise RuntimeError("integrated replacement cleanup payload is inconsistent")

    scopes = budget.get("authorized_scope")
    if (
        budget.get("schema_version")
        != "sloforge.branchfabric.experiment-004-budget-authorization/v1"
        or budget.get("authorized_cumulative_gpu_seconds", 0)
        < 2 * float(config.maximum_wall_seconds)
        or not isinstance(scopes, list)
        or not any(
            "integrated reclamation" in str(item) or "integrated retries" in str(item)
            for item in scopes
        )
    ):
        raise RuntimeError("budget authorization does not cover integrated v11")

    return {
        "schema_version": "sloforge.branchfabric.exp004-v11-sealed-evidence-verification/v1",
        "status": "PASS",
        "config_attempt_id": config.attempt_id,
        "micro_attempt_id": micro_attempt,
        "ledger_sha256_before_reservation": config.ledger_sha256_before_reservation,
        "micro_full_physical_amplification": movement["full_physical_amplification"],
        "micro_exact_first_token_matches": 8,
        "review_agents": sorted(agents),
        "integrated_prelaunch_labels": sorted(by_label),
        "replacement_authorization_verified": (
            replacement is not None and config.attempt_id == REPLACEMENT_INTEGRATED_ATTEMPT_ID
        ),
        "targeted_source_identity_authorization": targeted_authorization,
        "post_target_integrated_authorization": post_target_authorization,
        "post_prefunction_failure_integrated_authorization": (prefunction_retry_authorization),
        "post_remote_precontroller_failure_integrated_authorization": (
            bundled_ledger_retry_authorization
        ),
        "post_queue_accounting_failure_integrated_authorization": (
            queue_accounting_retry_authorization
        ),
        "post_control_gate_failure_integrated_authorization": (control_gate_retry_authorization),
        "post_image_closure_failure_integrated_authorization": (image_closure_retry_authorization),
        "post_remote_prefix_parser_failure_integrated_authorization": (
            remote_prefix_retry_authorization
        ),
        "post_retained_capacity_failure_integrated_authorization": (
            retained_capacity_retry_authorization
        ),
        "final_pre_gpu_gate": final_pre_gpu_gate,
        "artifacts": {
            name: {"reference": references[name][0], "sha256": references[name][1]}
            for name in references
        },
        "verified_at_monotonic_ns": time.monotonic_ns(),
    }


def _validate_engine_start(
    payloads: Sequence[Mapping[str, Any]],
    *,
    children: Sequence[tuple[str, subprocess.Popen[bytes], Any, Any]],
    inventory: Sequence[Mapping[str, Any]],
    launch_started_ns: int,
    observed_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2 or len({item.get("role") for item in payloads}) != 2:
        raise RuntimeError("retained-engine start evidence must cover two distinct roles")
    by_role = {str(item.get("role")): item for item in payloads}
    for (role, child, *_), gpu in zip(children, inventory, strict=True):
        item = by_role.get(role)
        timestamp = None if item is None else item.get("engine_started_ns")
        if (
            item is None
            or item.get("schema_version") != "sloforge.branchfabric.retained-engine-start/v1"
            or item.get("pid") != child.pid
            or item.get("physical_gpu_uuid") != gpu.get("uuid")
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or not launch_started_ns <= timestamp <= observed_ns
        ):
            raise RuntimeError(f"invalid retained-engine start evidence for {role}")
    return dict(by_role["serving"]), dict(by_role["rollout"])


def _retained_engine_readiness_assessment_components(
    readiness: Sequence[Mapping[str, Any]],
    *,
    minimum_batch16_request_throughput_rps: float = 15.0,
) -> tuple[dict[str, Mapping[str, Any]], tuple[dict[str, Any], ...], float]:
    """Validate the retained pair and recompute its batch-16 observations."""

    if (
        type(minimum_batch16_request_throughput_rps) is not float
        or not math.isfinite(minimum_batch16_request_throughput_rps)
        or minimum_batch16_request_throughput_rps <= 0.0
    ):
        raise ValueError(
            "retained-engine batch-16 throughput floor must be an exact finite positive float"
        )
    if len(readiness) != 2 or any(not isinstance(item, Mapping) for item in readiness):
        raise RuntimeError("retained-engine readiness requires exactly two payloads")

    expected_devices = {"serving": "gpu0", "rollout": "gpu1"}
    by_role: dict[str, Mapping[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    continuity_values: dict[str, set[Any]] = {
        "physical_gpu_uuid": set(),
        "engine_nonce": set(),
        "pid": set(),
        "engine_object_identity": set(),
        "adapter_object_identity": set(),
    }
    for payload in readiness:
        role = payload.get("role")
        if not isinstance(role, str) or role not in expected_devices or role in by_role:
            raise RuntimeError("retained-engine readiness role coverage is invalid")
        device = payload.get("device")
        if device != expected_devices[role]:
            raise RuntimeError("retained-engine readiness role/device binding is invalid")
        physical_gpu_uuid = payload.get("physical_gpu_uuid")
        if (
            not isinstance(physical_gpu_uuid, str)
            or not physical_gpu_uuid.startswith("GPU-")
            or physical_gpu_uuid in continuity_values["physical_gpu_uuid"]
        ):
            raise RuntimeError("retained-engine readiness physical GPU UUID is invalid")
        if (
            payload.get("schema_version") != "sloforge.branchfabric.engine-readiness-evidence/v1"
            or payload.get("passed") is not True
            or payload.get("queue_empty_pass") is not True
        ):
            raise RuntimeError(f"retained-engine readiness structure failed for {device}")
        compilation = payload.get("compilation_observation")
        runtime_state = payload.get("runtime_state")
        if (
            not isinstance(compilation, Mapping)
            or compilation.get("no_active_compilation_event") is not True
            or tuple(compilation.get("measured_probe_compilation_events", ()))
            or not isinstance(runtime_state, Mapping)
            or any(
                runtime_state.get(field) != 0
                for field in (
                    "request_count",
                    "running_requests",
                    "waiting_requests",
                    "skipped_waiting_requests",
                    "queue_depth",
                )
            )
        ):
            raise RuntimeError(f"retained-engine readiness structure failed for {device}")

        engine_nonce = payload.get("engine_nonce")
        integer_continuity = {
            field: payload.get(field)
            for field in (
                "pid",
                "engine_object_identity",
                "adapter_object_identity",
                "engine_started_ns",
            )
        }
        if (
            payload.get("engine_reloaded") is not False
            or not isinstance(engine_nonce, str)
            or len(engine_nonce) != 64
            or any(character not in "0123456789abcdef" for character in engine_nonce)
            or any(type(value) is not int or value <= 0 for value in integer_continuity.values())
        ):
            raise RuntimeError(f"retained-engine readiness continuity failed for {device}")
        for field in ("engine_nonce", "pid", "engine_object_identity", "adapter_object_identity"):
            value = payload.get(field)
            if value in continuity_values[field]:
                raise RuntimeError(f"retained-engine readiness continuity duplicates {field}")
            continuity_values[field].add(value)
        continuity_values["physical_gpu_uuid"].add(physical_gpu_uuid)

        verification = payload.get("verification_batches")
        if not isinstance(verification, list) or len(verification) != 5:
            raise RuntimeError(
                f"retained-engine readiness requires exact batch-1/2/4/8/16 evidence for {device}"
            )
        by_batch_size: dict[int, Mapping[str, Any]] = {}
        prior_completed_ns: int | None = None
        for batch in verification:
            if not isinstance(batch, Mapping):
                raise RuntimeError(
                    f"retained-engine readiness batch evidence is invalid for {device}"
                )
            batch_size = batch.get("batch_size")
            if type(batch_size) is not int or batch_size not in {1, 2, 4, 8, 16}:
                raise RuntimeError(
                    f"retained-engine readiness requires exact batch-1/2/4/8/16 evidence for "
                    f"{device}"
                )
            if batch_size in by_batch_size:
                raise RuntimeError(
                    f"retained-engine readiness duplicates batch-{batch_size} evidence for {device}"
                )
            throughput = batch.get("request_throughput_rps")
            if type(throughput) is not float or not math.isfinite(throughput) or throughput <= 0.0:
                raise RuntimeError(
                    f"retained-engine batch-{batch_size} throughput is not an exact finite float "
                    f"for {device}"
                )
            started_ns = batch.get("started_ns")
            first_token_ns = batch.get("first_token_ns")
            completed_ns = batch.get("completed_ns")
            request_ids = batch.get("request_ids")
            output_lengths = batch.get("output_lengths")
            if (
                type(started_ns) is not int
                or type(first_token_ns) is not int
                or type(completed_ns) is not int
                or not 0 < started_ns < first_token_ns <= completed_ns
                or (prior_completed_ns is not None and started_ns < prior_completed_ns)
            ):
                raise RuntimeError(
                    f"retained-engine batch-{batch_size} timing/order is invalid for {device}"
                )
            if (
                not isinstance(request_ids, list)
                or len(request_ids) != batch_size
                or len(set(request_ids)) != batch_size
                or any(
                    not isinstance(request_id, str) or not request_id for request_id in request_ids
                )
                or not isinstance(output_lengths, list)
                or len(output_lengths) != batch_size
                or any(type(length) is not int or length != 64 for length in output_lengths)
            ):
                raise RuntimeError(
                    f"retained-engine batch-{batch_size} request/output accounting is invalid "
                    f"for {device}"
                )
            by_batch_size[batch_size] = batch
            prior_completed_ns = completed_ns
        if set(by_batch_size) != {1, 2, 4, 8, 16}:
            raise RuntimeError(
                f"retained-engine readiness requires exact batch-1/2/4/8/16 evidence for {device}"
            )

        batch16 = by_batch_size[16]
        elapsed_ns = batch16["completed_ns"] - batch16["started_ns"]
        recomputed_throughput = 16.0 / (elapsed_ns / 1_000_000_000.0)
        reported_throughput = batch16["request_throughput_rps"]
        if not math.isclose(
            reported_throughput,
            recomputed_throughput,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"retained-engine batch-16 throughput/timing commitment mismatch for {device}"
            )
        observations.append(
            {
                "device": device,
                "role": role,
                "physical_gpu_uuid": physical_gpu_uuid,
                "engine_nonce": engine_nonce,
                "pid": integer_continuity["pid"],
                "engine_object_identity": integer_continuity["engine_object_identity"],
                "adapter_object_identity": integer_continuity["adapter_object_identity"],
                "engine_started_ns": integer_continuity["engine_started_ns"],
                "batch_size": 16,
                "started_ns": batch16["started_ns"],
                "completed_ns": batch16["completed_ns"],
                "elapsed_ns": elapsed_ns,
                "reported_request_throughput_rps": reported_throughput,
                "recomputed_request_throughput_rps": recomputed_throughput,
                "minimum_request_throughput_rps": minimum_batch16_request_throughput_rps,
                "passed": reported_throughput >= minimum_batch16_request_throughput_rps,
            }
        )
        by_role[role] = payload

    observations.sort(key=lambda item: DEVICES.index(str(item["device"])))
    return by_role, tuple(observations), minimum_batch16_request_throughput_rps


def _validate_retained_engine_readiness_comparability(
    readiness: Sequence[Mapping[str, Any]],
    *,
    minimum_batch16_request_throughput_rps: float = 15.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require comparable retained engines before exercising either sanity probe."""

    assessment = _assess_retained_engine_readiness_comparability(
        readiness,
        minimum_batch16_request_throughput_rps=minimum_batch16_request_throughput_rps,
    )
    if assessment.get("readiness_comparable") is not True:
        failed_devices = ", ".join(
            str(item.get("device"))
            for item in assessment.get("devices", ())
            if isinstance(item, Mapping) and item.get("passed") is not True
        )
        raise RuntimeError(
            "retained-engine batch-16 throughput is below the comparable floor for "
            f"{failed_devices or 'unknown device'}"
        )
    by_role = {str(payload["role"]): payload for payload in readiness}
    return dict(by_role["serving"]), dict(by_role["rollout"])


def _validate_handoffs(
    payloads: Sequence[Mapping[str, Any]],
    *,
    continuity: Mapping[str, Mapping[str, Any]],
    selection_sha256: str,
    sanity_result_sha256: str,
    authorization_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2:
        raise RuntimeError("transaction handoff requires exactly two workers")
    by_device = {str(item.get("device")): item for item in payloads}
    if set(by_device) != set(DEVICES):
        raise RuntimeError("transaction handoff device coverage is invalid")
    for device in DEVICES:
        item = by_device[device]
        hashes = item.get("pre_gpu_evidence_hashes")
        if (
            item.get("schema_version") != "sloforge.branchfabric.v11-transaction-ready/v1"
            or item.get("passed") is not True
            or item.get("selected_load_sha256") != selection_sha256
            or not isinstance(hashes, dict)
            or hashes.get("sanity_result_sha256") != sanity_result_sha256
            or hashes.get("authorization_artifact_hash") != authorization_sha256
            or item.get("authorization_artifact_hash") != authorization_sha256
            or any(item.get(key) != continuity[device].get(key) for key in continuity[device])
        ):
            raise RuntimeError(f"transaction handoff identity/commitment failed for {device}")
    return dict(by_device["gpu0"]), dict(by_device["gpu1"])


def _validate_worker_results(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected: Mapping[str, tuple[int, str]],
    attempt_id: str,
    targeted: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payloads) != 2:
        raise RuntimeError("integrated v11 results do not cover both worker roles")
    validated: list[dict[str, Any]] = []
    for role, payload in zip(ROLES, payloads, strict=True):
        pid, uuid = expected[role]
        compilation = payload.get("measured_transaction_compilation_observation")
        interval_start = (
            compilation.get("interval_start_ns") if isinstance(compilation, dict) else None
        )
        interval_end = compilation.get("interval_end_ns") if isinstance(compilation, dict) else None
        expected_schema = (
            (
                "sloforge.branchfabric.experiment-004-v11-targeted-gpu0-result/v1"
                if role == "serving"
                else "sloforge.branchfabric.experiment-004-v11-targeted-gpu1-result/v1"
            )
            if targeted
            else (
                "sloforge.branchfabric.experiment-004-v11-gpu0-result/v1"
                if role == "serving"
                else "sloforge.branchfabric.experiment-004-v11-gpu1-result/v1"
            )
        )
        if (
            payload.get("schema_version") != expected_schema
            or payload.get("status") != "succeeded"
            or payload.get("role") != role
            or payload.get("attempt_id") != attempt_id
            or payload.get("pid") != pid
            or payload.get("physical_gpu_uuid") != uuid
            or not isinstance(compilation, dict)
            or compilation.get("schema_version")
            != "sloforge.branchfabric.measured-transaction-compilation-observation/v1"
            or compilation.get("source") != "bounded-python-logging-handler"
            or compilation.get("role") != role
            or isinstance(interval_start, bool)
            or not isinstance(interval_start, int)
            or isinstance(interval_end, bool)
            or not isinstance(interval_end, int)
            or interval_end <= interval_start
            or compilation.get("capture_buffer_valid") is not True
            or compilation.get("passed") is not True
            or compilation.get("events") != []
            or compilation.get("no_deferred_compilation_event") is not True
        ):
            raise RuntimeError(f"integrated v11 {role} result failed identity/status gates")
        if targeted:
            _validate_targeted_worker_result(payload, role=role)
        elif role == "serving":
            _validate_gpu0_scientific_result(payload)
        else:
            _validate_gpu1_scientific_result(payload)
        validated.append(dict(payload))
    if targeted:
        _validate_targeted_cross_role_consistency(validated[0], validated[1])
    else:
        _validate_cross_role_scientific_consistency(validated[0], validated[1])
    return validated[0], validated[1]


def _validate_targeted_worker_result(payload: Mapping[str, Any], *, role: str) -> None:
    if (
        payload.get("terminal_phase") != "SOURCE_ALLOCATION_IDENTITY_GATE"
        or payload.get("optimized_export_started") is not False
        or payload.get("transaction_source_release_started") is not False
    ):
        raise RuntimeError(f"targeted v11 {role} exceeded its authorized terminal phase")
    if role == "serving":
        trigger = payload.get("reclamation_trigger_evidence")
        terminal = payload.get("source_identity_terminal")
        queue = payload.get("final_runtime_queue_state")
        required_queue_fields = {
            "request_count",
            "running_requests",
            "waiting_requests",
            "skipped_waiting_requests",
            "queue_depth",
        }
        if (
            not isinstance(trigger, dict)
            or _validate_emitted_trigger_for_controller(trigger).get("overload_confirmed")
            is not True
            or not isinstance(terminal, dict)
            or terminal.get("passed") is not True
            or not isinstance(queue, dict)
            or set(queue) != required_queue_fields
            or any(
                isinstance(queue[name], bool)
                or not isinstance(queue[name], int)
                or queue[name] != 0
                for name in required_queue_fields
            )
            or payload.get("producer_stopped") is not True
        ):
            raise RuntimeError("targeted v11 serving result failed trigger/terminal/queue gates")
        return
    gate = payload.get("source_identity_gate")
    validation = payload.get("source_identity_validation")
    history = payload.get("allocation_history_retirement")
    notifications = payload.get("post_commit_allocation_notifications")
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("expected_allocation_count") != 1152
        or gate.get("observed_allocation_count") != 1152
        or any(
            gate.get(name) is not True
            for name in (
                "exact_logical_mapping",
                "exact_block_epoch_identity",
                "exact_owner_sets",
                "exact_refcounts",
                "all_allocations_live",
                "device_identity_pass",
                "no_post_commit_mutation",
            )
        )
        or gate.get("post_commit_allocation_event_count") != 0
        or not isinstance(validation, dict)
        or validation.get("passed") is not True
        or not isinstance(history, dict)
        or history.get("semantic_identity_claimed") is not False
        or history.get("passed") is not True
        or not isinstance(notifications, dict)
        or notifications.get("queue_empty") is not True
        or payload.get("cleanup_release_only") is not True
    ):
        raise RuntimeError("targeted v11 rollout source identity evidence failed closed")


def _validate_emitted_trigger_for_controller(trigger: Mapping[str, Any]) -> dict[str, Any]:
    events = trigger.get("events")
    depth = trigger.get("queue_depth_at_trigger")
    required_events = (
        "OVERLOAD_DETECTED",
        "TRIGGER_PREDICATE_SATISFIED",
        "RECLAIM_TRIGGER_EMITTED",
    )
    if (
        trigger.get("schema_version") != "sloforge.branchfabric.reclamation-trigger-evidence/v2"
        or trigger.get("overload_confirmed") is not True
        or trigger.get("positive_queue_slope") is not True
        or trigger.get("offered_rate_exceeds_completed_rate") is not True
        or not isinstance(depth, int)
        or isinstance(depth, bool)
        or not 20 <= depth < 64
        or trigger.get("queue_trigger") != 20
        or trigger.get("queue_abort") != 64
        or trigger.get("emergency_ceiling_headroom_requests") != 64 - depth
        or not isinstance(events, dict)
        or any(
            isinstance(events.get(name), bool) or not isinstance(events.get(name), int)
            for name in required_events
        )
        or not events["OVERLOAD_DETECTED"]
        <= events["TRIGGER_PREDICATE_SATISFIED"]
        <= events["RECLAIM_TRIGGER_EMITTED"]
        or trigger.get("triggered_ns") != events["RECLAIM_TRIGGER_EMITTED"]
        or trigger.get("reclaim_trigger_emitted_ns") != events["RECLAIM_TRIGGER_EMITTED"]
        or trigger.get("trigger_predicate_satisfied_ns") != events["TRIGGER_PREDICATE_SATISFIED"]
        or trigger.get("overload_detected_ns") != events["OVERLOAD_DETECTED"]
        or trigger.get("controller_reaction_latency_ns")
        != events["RECLAIM_TRIGGER_EMITTED"] - events["TRIGGER_PREDICATE_SATISFIED"]
    ):
        raise RuntimeError("targeted v11 early trigger evidence failed closed")
    return dict(trigger)


def _validate_targeted_cross_role_consistency(
    serving: Mapping[str, Any], rollout: Mapping[str, Any]
) -> None:
    left = serving.get("source_identity_terminal")
    right = rollout.get("source_identity_gate")
    if not isinstance(left, dict) or not isinstance(right, dict) or left != right:
        raise RuntimeError("targeted v11 source identity barrier differs across workers")


def _all_true_fields(value: Any, required: Sequence[str], *, label: str) -> None:
    if not isinstance(value, dict) or any(value.get(field) is not True for field in required):
        raise RuntimeError(f"integrated v11 {label} evidence is incomplete or failed")


def _validate_gpu0_scientific_result(payload: Mapping[str, Any]) -> None:
    _all_true_fields(
        payload.get("scientific_gates"),
        (
            "control_stability_pass",
            "gpu0_overload_pass",
            "bounded_backlog_pass",
            "two_gpu_service_gt_offered_pass",
            "queue_drain_pass",
            "slo_restoration_pass",
            "slo_stability_pass",
            "gpu0_active_during_restore_pass",
        ),
        label="GPU0 scientific gate",
    )
    interference = payload.get("gpu0_restore_interference")
    control = payload.get("gpu0_control_interval")
    if not isinstance(interference, dict):
        raise RuntimeError("integrated v11 GPU0 restore interference evidence is absent")
    arrivals = interference.get("arrivals")
    completions = interference.get("completions")
    emitted = interference.get("emitted_tokens")
    live_samples = interference.get("live_vllm_queue_depth_samples")
    offered_samples = interference.get("offered_outstanding_samples")
    nested_control = interference.get("control_interval")
    restore_interval = interference.get("measured_restore_interval")
    serving = payload.get("serving")
    try:
        recomputed_control = (
            _control_interval_evidence(
                serving,
                expected_rate_rps=9.0,
                expected_duration_seconds=5.0,
                slo_ttft_seconds=2.0,
                warmup_seconds=1.0,
                evaluation_seconds=1.0,
                minimum_completion_fraction=0.90,
            )
            if isinstance(serving, dict)
            else None
        )
    except RuntimeError as error:
        raise RuntimeError("integrated v11 GPU0 control-interval evidence is invalid") from error
    control_diagnostic = (
        control.get("total_outstanding_diagnostic") if isinstance(control, dict) else None
    )
    control_samples = (
        control_diagnostic.get("samples") if isinstance(control_diagnostic, dict) else None
    )
    if (
        not isinstance(control, dict)
        or control != nested_control
        or control != recomputed_control
        or control.get("schema_version") != "sloforge.branchfabric.v11-control-interval/v2"
        or control.get("passed") is not True
        or control.get("full_duration_seconds") != 5.0
        or control.get("warmup_seconds") != 1.0
        or control.get("duration_seconds") != 4.0
        or control.get("expected_full_arrivals") != 45
        or control.get("full_arrivals") != 45
        or control.get("expected_arrivals") != 36
        or control.get("arrivals") != 36
        or control.get("eventual_cohort_completions") != 36
        or isinstance(control.get("completions_in_interval"), bool)
        or not isinstance(control.get("completions_in_interval"), int)
        or not 33 <= int(control["completions_in_interval"]) <= 45
        or control.get("offered_rate_per_second") != 9.0
        or not isinstance(control.get("completed_rate_per_second"), (int, float))
        or isinstance(control.get("completed_rate_per_second"), bool)
        or float(control["completed_rate_per_second"]) < 9.0 * 0.90
        or control.get("minimum_completion_fraction") != 0.90
        or not isinstance(control.get("p95_ttft_ns"), (int, float))
        or isinstance(control.get("p95_ttft_ns"), bool)
        or float(control["p95_ttft_ns"]) > 2_000_000_000
        or control.get("unique_control_request_ids") is not True
        or control.get("unique_scheduled_arrivals") is not True
        or control.get("scheduled_arrival_cadence_exact") is not True
        or control.get("monotonic_control_timestamps") is not True
        or control.get("eventual_full_cohort_completions") != 45
        or control.get("full_cohort_output_tokens_exact") is not True
        or control.get("complete_full_request_accounting") is not True
        or control.get("complete_request_accounting") is not True
        or control.get("offered_rate_matches_config") is not True
        or control.get("completion_tracks_offer") is not True
        or control.get("p95_ttft_below_slo") is not True
        or control.get("outstanding_queue_non_positive_trend") is not True
        or control.get("in_service_requests_not_treated_as_waiting_queue") is not True
        or isinstance(control.get("waiting_queue_maximum_depth"), bool)
        or not isinstance(control.get("waiting_queue_maximum_depth"), int)
        or int(control["waiting_queue_maximum_depth"]) < 0
        or control.get("waiting_queue_bounded_below_normal_trigger") is not True
        or control.get("total_outstanding_bounded_below_normal_trigger") is not True
        or not isinstance(control_diagnostic, dict)
        or control_diagnostic.get("sample_interval_ns") != 1_000_000_000
        or control_diagnostic.get("sustained_positive") is not False
        or isinstance(control_diagnostic.get("maximum_depth"), bool)
        or not isinstance(control_diagnostic.get("maximum_depth"), int)
        or int(control_diagnostic["maximum_depth"]) >= 20
        or not isinstance(control_samples, (tuple, list))
        or len(control_samples) != 5
        or any(
            not isinstance(sample, dict)
            or isinstance(sample.get("timestamp_ns"), bool)
            or not isinstance(sample.get("timestamp_ns"), int)
            or isinstance(sample.get("total_outstanding"), bool)
            or not isinstance(sample.get("total_outstanding"), int)
            or int(sample["total_outstanding"]) < 0
            for sample in (control_samples or ())
        )
    ):
        raise RuntimeError("integrated v11 GPU0 control-interval evidence is invalid")
    interval_start = (
        restore_interval.get("start_ns") if isinstance(restore_interval, dict) else None
    )
    interval_end = restore_interval.get("end_ns") if isinstance(restore_interval, dict) else None
    interval_duration = (
        restore_interval.get("duration_seconds") if isinstance(restore_interval, dict) else None
    )
    interval_arrivals = (
        restore_interval.get("arrivals") if isinstance(restore_interval, dict) else None
    )
    interval_completions = (
        restore_interval.get("completions") if isinstance(restore_interval, dict) else None
    )
    interval_tokens = (
        restore_interval.get("emitted_tokens") if isinstance(restore_interval, dict) else None
    )
    interval_samples = (
        restore_interval.get("live_vllm_queue_depth_samples")
        if isinstance(restore_interval, dict)
        else None
    )
    if (
        not isinstance(restore_interval, dict)
        or restore_interval.get("schema_version")
        != "sloforge.branchfabric.v11-gpu0-restore-interference/v1"
        or restore_interval.get("passed") is not True
        or restore_interval.get("methodology_stall_absent") is not True
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (
                interval_start,
                interval_end,
                interval_arrivals,
                interval_completions,
                interval_tokens,
            )
        )
        or interval_end <= interval_start
        or not isinstance(interval_duration, (int, float))
        or isinstance(interval_duration, bool)
        or not math.isclose(
            float(interval_duration), (interval_end - interval_start) / 1e9, abs_tol=1e-12
        )
        or interval_arrivals <= 0
        or interval_completions <= 0
        or interval_tokens <= 0
        or restore_interval.get("ttft_sample_count") != interval_arrivals
        or not math.isclose(
            float(restore_interval.get("observed_arrival_rate_per_second", math.nan)),
            interval_arrivals / float(interval_duration),
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(restore_interval.get("completion_rate_per_second", math.nan)),
            interval_completions / float(interval_duration),
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(restore_interval.get("token_rate_per_second", math.nan)),
            interval_tokens / float(interval_duration),
            abs_tol=1e-12,
        )
        or not isinstance(interval_samples, (tuple, list))
        or not interval_samples
        or any(
            not isinstance(item, dict)
            or not interval_start <= int(item.get("observed_ns", -1)) <= interval_end
            for item in interval_samples
        )
    ):
        raise RuntimeError("integrated v11 GPU0 actual-restore interference is invalid")
    if (
        interference.get("active_during_restore_pass") is not True
        or isinstance(arrivals, bool)
        or not isinstance(arrivals, int)
        or arrivals <= 0
        or completions != arrivals
        or emitted != arrivals * 64
        or interference.get("offered_rps") != 9.0
        or not isinstance(live_samples, (tuple, list))
        or not live_samples
        or not all(isinstance(item, dict) for item in live_samples)
        or not isinstance(offered_samples, (tuple, list))
        or not offered_samples
    ):
        raise RuntimeError("integrated v11 GPU0 was not measurably active during restore")
    serving = payload.get("serving")
    if not isinstance(serving, dict):
        raise RuntimeError("integrated v11 GPU0 serving trace is absent")
    trigger = serving.get("reclamation_trigger_evidence")
    recovery = serving.get("serving_recovery_evidence")
    events = trigger.get("events") if isinstance(trigger, dict) else None
    trigger_depth = trigger.get("queue_depth_at_trigger") if isinstance(trigger, dict) else None
    emitted_ns = events.get("RECLAIM_TRIGGER_EMITTED") if isinstance(events, dict) else None
    predicate_ns = events.get("TRIGGER_PREDICATE_SATISFIED") if isinstance(events, dict) else None
    overload_ns = events.get("OVERLOAD_DETECTED") if isinstance(events, dict) else None
    reaction_ns = (
        trigger.get("controller_reaction_latency_ns") if isinstance(trigger, dict) else None
    )
    if (
        not isinstance(trigger, dict)
        or trigger.get("schema_version") != "sloforge.branchfabric.reclamation-trigger-evidence/v2"
        or trigger.get("overload_confirmed") is not True
        or trigger.get("positive_queue_slope") is not True
        or trigger.get("offered_rate_exceeds_completed_rate") is not True
        or trigger.get("queue_trigger") != 20
        or trigger.get("queue_abort") != 64
        or isinstance(trigger_depth, bool)
        or not isinstance(trigger_depth, int)
        or not 20 <= trigger_depth <= MAX_SCIENTIFIC_TRIGGER_DEPTH
        or trigger.get("emergency_ceiling_headroom_requests") != 64 - trigger_depth
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (overload_ns, predicate_ns, emitted_ns, reaction_ns)
        )
        or not overload_ns <= predicate_ns <= emitted_ns
        or reaction_ns != emitted_ns - predicate_ns
        or not 0 <= reaction_ns <= MAX_RECLAIM_TRIGGER_REACTION_NS
        or trigger.get("triggered_ns") != emitted_ns
    ):
        raise RuntimeError("integrated v11 GPU0 early-trigger evidence is invalid")
    windows = recovery.get("stability_windows") if isinstance(recovery, dict) else None
    if (
        not isinstance(recovery, dict)
        or recovery.get("schema_version") != "sloforge.branchfabric.serving-recovery-evidence/v1"
        or recovery.get("completed_rate_per_second", 0.0)
        <= recovery.get("offered_rate_per_second", math.inf)
        or recovery.get("queue_depth_slope_per_second", 0.0) >= 0.0
        or not isinstance(windows, list)
        or len(windows) != 5
        or any(
            not isinstance(window, dict)
            or window.get("passed") is not True
            or window.get("p95_ttft_ns") is None
            or float(window["p95_ttft_ns"]) > 2_000_000_000
            or int(window.get("queue_depth_end", 5)) > 4
            or int(window.get("ttft_sample_count", 0)) <= 0
            or int(window.get("end_ns", 0)) - int(window.get("start_ns", 0)) != 1_000_000_000
            for window in windows
        )
        or any(int(left["end_ns"]) != int(right["start_ns"]) for left, right in pairwise(windows))
    ):
        raise RuntimeError("integrated v11 GPU0 recovery/SLO evidence is invalid")


def _validate_non_overlapping_partition(
    value: Any, *, chain: str, expected_stages: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"integrated v11 {chain} critical path is absent")
    stages = value.get("stages")
    if (
        value.get("schema_version") != "sloforge.branchfabric.v11-non-overlapping-critical-path/v1"
        or value.get("chain") != chain
        or value.get("passed") is not True
        or value.get("overlap_double_count_ns") != 0
        or not isinstance(stages, list)
        or [item.get("stage") for item in stages if isinstance(item, dict)] != list(expected_stages)
    ):
        raise RuntimeError(f"integrated v11 {chain} critical path is malformed")
    previous_end: int | None = None
    total = 0
    for item in stages:
        start = item.get("start_ns")
        end = item.get("end_ns")
        duration = item.get("wall_time_ns")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or end < start
            or duration != end - start
            or (previous_end is not None and start != previous_end)
        ):
            raise RuntimeError(f"integrated v11 {chain} critical path overlaps or has gaps")
        total += duration
        previous_end = end
    if (
        not stages
        or value.get("start_ns") != stages[0]["start_ns"]
        or value.get("end_ns") != stages[-1]["end_ns"]
        or value.get("wall_time_ns") != total
        or total != int(value["end_ns"]) - int(value["start_ns"])
    ):
        raise RuntimeError(f"integrated v11 {chain} critical path does not conserve time")
    return value


def _validate_gpu1_scientific_result(payload: Mapping[str, Any]) -> None:
    topology = payload.get("topology")
    if not isinstance(topology, dict) or topology != {
        "branch_count": 8,
        "shared_blocks": 1024,
        "private_blocks": 128,
        "total_blocks": 1152,
        "logical_state_bytes": 1_056_964_608,
    }:
        raise RuntimeError("integrated v11 GPU1 topology differs from exact workload")
    _all_true_fields(
        payload.get("correctness"),
        (
            "allocator_epoch_pass",
            "ownership_release_pass",
            "engine_step_binding_pass",
            "integrity_pass",
            "destination_mapping_commitment_pass",
            "fresh_destination_allocations_pass",
            "all_branches_resumed",
            "first_token_exact_8_of_8",
            "movement_accounting_pass",
            "gpu0_active_during_restore_protocol_pass",
        ),
        label="GPU1 correctness gate",
    )
    continuation = payload.get("continuation")
    movement = payload.get("movement")
    passes = payload.get("state_passes")
    if (
        not isinstance(continuation, dict)
        or continuation.get("exact_matches") != 8
        or not isinstance(continuation.get("minimum_tokens_per_branch"), int)
        or isinstance(continuation.get("minimum_tokens_per_branch"), bool)
        or continuation["minimum_tokens_per_branch"] < 8
        or not isinstance(movement, dict)
        or movement.get("movement_accounting_complete") is not True
        or movement.get("logical_state_bytes") != 1_056_964_608
        or not isinstance(movement.get("full_physical_bytes"), int)
        or isinstance(movement.get("full_physical_bytes"), bool)
        or movement["full_physical_bytes"] <= 0
        or not isinstance(passes, list)
        or not passes
        or not all(isinstance(item, dict) for item in passes)
    ):
        raise RuntimeError("integrated v11 GPU1 measured continuation/movement evidence failed")
    timings = payload.get("timings")
    timeline = payload.get("trigger_timeline")
    temporary = payload.get("temporary_memory")
    required_timing_fields = (
        "reclaim_trigger_ns",
        "rollout_admission_stop_ns",
        "state_quiescence_ns",
        "source_authentication_ended_ns",
        "source_pipeline_ended_ns",
        "state_publish_ended_ns",
        "source_release_ended_ns",
        "export_started_ns",
        "export_ended_ns",
        "hbm_reclaim_confirmed_ns",
        "gpu1_serving_ready_ns",
        "restore_trigger_ns",
        "restore_started_ns",
        "checkpoint_authentication_started_ns",
        "checkpoint_authentication_ended_ns",
        "destination_staging_ended_ns",
        "native_restore_pipeline_ended_ns",
        "restore_native_complete_ns",
        "continuation_started_ns",
        "first_resumed_token_ns",
        "all_branches_resumed_ns",
        "continuation_complete_ns",
    )
    if (
        not isinstance(timings, dict)
        or any(
            isinstance(timings.get(field), bool) or not isinstance(timings.get(field), int)
            for field in required_timing_fields
        )
        or not (
            timings["reclaim_trigger_ns"]
            <= timings["rollout_admission_stop_ns"]
            <= timings["export_started_ns"]
            <= timings["state_quiescence_ns"]
            <= timings["source_authentication_ended_ns"]
            <= timings["source_pipeline_ended_ns"]
            <= timings["state_publish_ended_ns"]
            <= timings["source_release_ended_ns"]
            <= timings["export_ended_ns"]
            <= timings["hbm_reclaim_confirmed_ns"]
            <= timings["gpu1_serving_ready_ns"]
            <= timings["restore_trigger_ns"]
            <= timings["restore_started_ns"]
            <= timings["checkpoint_authentication_started_ns"]
            <= timings["checkpoint_authentication_ended_ns"]
            <= timings["destination_staging_ended_ns"]
            <= timings["native_restore_pipeline_ended_ns"]
            <= timings["restore_native_complete_ns"]
            <= timings["continuation_started_ns"]
            <= timings["first_resumed_token_ns"]
            <= timings["all_branches_resumed_ns"]
            <= timings["continuation_complete_ns"]
        )
        or not isinstance(timeline, dict)
        or timeline.get("passed") is not True
        or timeline.get("trigger_precedes_state_quiescence") is not True
        or not isinstance(temporary, dict)
        or any(
            isinstance(temporary.get(field), bool)
            or not isinstance(temporary.get(field), int)
            or temporary[field] < 0
            for field in (
                "peak_pinned_host_temporary_bytes",
                "peak_pageable_host_temporary_bytes",
                "peak_gpu_temporary_bytes",
            )
        )
    ):
        raise RuntimeError("integrated v11 GPU1 timing/temporary-memory evidence failed")
    critical_paths = payload.get("critical_paths")
    if not isinstance(critical_paths, dict):
        raise RuntimeError("integrated v11 critical-path evidence is absent")
    reclaim_partition = _validate_non_overlapping_partition(
        critical_paths.get("reclamation"),
        chain="RECLAIM_TRIGGER_TO_GPU1_SERVING_READY",
        expected_stages=(
            "branch_quiesce",
            "source_authentication",
            "pre_commit_allocation_history_retirement",
            "source_capture_commit",
            "pre_export_identity_validation_and_commit_publish",
            "fused_gather_repack_d2h",
            "post_export_read_identity_validation",
            "state_publish_and_allocation_history_retirement",
            "source_release",
            "hbm_reclaim_confirmation",
            "gpu1_serving_ready",
        ),
    )
    restore_partition = _validate_non_overlapping_partition(
        critical_paths.get("restore"),
        chain="RESTORE_TRIGGER_TO_FIRST_RESUMED_TOKEN",
        expected_stages=(
            "restore_preconditions",
            "checkpoint_authentication",
            "destination_allocation_staging",
            "fused_h2d_direct_scatter_raw_validation_and_admission",
            "canonical_pass_validation_and_commit",
            "first_resumed_token",
        ),
    )
    if (
        reclaim_partition["start_ns"] != timings["reclaim_trigger_ns"]
        or reclaim_partition["end_ns"] != timings["gpu1_serving_ready_ns"]
        or restore_partition["start_ns"] != timings["restore_trigger_ns"]
        or restore_partition["end_ns"] != timings["first_resumed_token_ns"]
        or critical_paths.get("residual_source_chain_ns")
        != timings["source_pipeline_ended_ns"] - timings["state_quiescence_ns"]
        or critical_paths.get("residual_restore_chain_ns")
        != timings["restore_native_complete_ns"] - timings["restore_started_ns"]
        or not isinstance(critical_paths.get("stage_overlap_policy"), str)
        or not critical_paths["stage_overlap_policy"]
    ):
        raise RuntimeError("integrated v11 critical-path boundaries are inconsistent")
    pass_ids: set[str] = set()
    capture_count = 0
    restore_count = 0
    for record in passes:
        pass_id = record.get("pass_id")
        wall_start = record.get("wall_start_ns")
        wall_end = record.get("wall_end_ns")
        if (
            record.get("schema_version") != "sloforge.branchfabric.state-pass-record/v11"
            or not isinstance(pass_id, str)
            or not pass_id
            or pass_id in pass_ids
            or isinstance(wall_start, bool)
            or not isinstance(wall_start, int)
            or isinstance(wall_end, bool)
            or not isinstance(wall_end, int)
            or wall_end < wall_start
            or isinstance(record.get("physical_touch_bytes"), bool)
            or not isinstance(record.get("physical_touch_bytes"), int)
            or record["physical_touch_bytes"] < 0
        ):
            raise RuntimeError("integrated v11 GPU1 canonical StatePassRecord is invalid")
        pass_ids.add(pass_id)
        if pass_id.startswith("capture:"):
            capture_count += 1
            if (
                not timings["export_started_ns"]
                <= wall_start
                <= wall_end
                <= timings["export_ended_ns"]
            ):
                raise RuntimeError("integrated v11 capture pass escaped its measured wall chain")
        elif pass_id.startswith("restore:"):
            restore_count += 1
            if (
                not timings["restore_started_ns"]
                <= wall_start
                <= wall_end
                <= timings["restore_native_complete_ns"]
            ):
                raise RuntimeError("integrated v11 restore pass escaped its measured wall chain")
        else:
            raise RuntimeError("integrated v11 StatePassRecord has an unknown chain prefix")
    movement_by_phase = payload.get("movement_by_phase")
    if not isinstance(movement_by_phase, dict) or set(movement_by_phase) != {
        "source_export",
        "rollout_restore",
    }:
        raise RuntimeError("integrated v11 per-phase movement accounting is absent")
    export_movement = movement_by_phase["source_export"]
    restore_movement = movement_by_phase["rollout_restore"]
    phase_movements = (export_movement, restore_movement)
    movement_integer_fields = (
        "full_physical_bytes",
        "external_movement_bytes",
        "avoidable_physical_bytes",
        "required_physical_bytes",
        "diagnostic_physical_bytes",
        "record_count",
    )
    if any(
        not isinstance(item, dict)
        or item.get("movement_accounting_complete") is not True
        or item.get("logical_state_bytes") != 1_056_964_608
        or any(
            isinstance(item.get(field), bool)
            or not isinstance(item.get(field), int)
            or item[field] < 0
            for field in movement_integer_fields
        )
        or not math.isclose(
            float(item.get("full_physical_amplification", math.nan)),
            item["full_physical_bytes"] / 1_056_964_608,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(item.get("external_movement_amplification", math.nan)),
            item["external_movement_bytes"] / 1_056_964_608,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(item.get("avoidable_amplification", math.nan)),
            item["avoidable_physical_bytes"] / 1_056_964_608,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for item in phase_movements
    ):
        raise RuntimeError("integrated v11 per-phase movement accounting is invalid")
    logical = movement["logical_state_bytes"]
    if (
        capture_count == 0
        or restore_count == 0
        or export_movement["record_count"] != capture_count
        or restore_movement["record_count"] != restore_count
        or any(
            export_movement[field] + restore_movement[field] != movement.get(field)
            for field in movement_integer_fields
        )
        or movement.get("record_count") != len(passes)
        or sum(int(record["physical_touch_bytes"]) for record in passes)
        != movement.get("full_physical_bytes")
        or any(
            isinstance(movement.get(field), bool)
            or not isinstance(movement.get(field), int)
            or movement[field] < 0
            for field in (
                "full_physical_bytes",
                "external_movement_bytes",
                "avoidable_physical_bytes",
                "required_physical_bytes",
            )
        )
        or not math.isclose(
            float(movement.get("full_physical_amplification", math.nan)),
            movement["full_physical_bytes"] / logical,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(movement.get("external_movement_amplification", math.nan)),
            movement["external_movement_bytes"] / logical,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(movement.get("avoidable_amplification", math.nan)),
            movement["avoidable_physical_bytes"] / logical,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("integrated v11 movement totals are incomplete or inconsistent")


def _validate_cross_role_scientific_consistency(
    gpu0: Mapping[str, Any], gpu1: Mapping[str, Any]
) -> None:
    """Bind cross-process monotonic evidence into one causal transaction."""

    gpu0_serving = gpu0["serving"]
    trigger = gpu0_serving["reclamation_trigger_evidence"]
    recovery = gpu0_serving["serving_recovery_evidence"]
    gpu1_serving = gpu1.get("serving")
    timings = gpu1["timings"]
    gpu1_requests = gpu1_serving.get("requests") if isinstance(gpu1_serving, dict) else None
    if not isinstance(gpu1_requests, (tuple, list)) or not gpu1_requests:
        raise RuntimeError("integrated v11 GPU1 useful-serving request evidence is absent")
    useful_tokens = [
        row.get("first_token_ns")
        for row in gpu1_requests
        if isinstance(row, dict) and row.get("first_token_ns") is not None
    ]
    if not useful_tokens or any(
        isinstance(item, bool) or not isinstance(item, int) for item in useful_tokens
    ):
        raise RuntimeError("integrated v11 GPU1 useful-serving timestamps are invalid")
    first_useful_ns = recovery.get("gpu1_first_useful_ns")
    windows = recovery["stability_windows"]
    slo_restoration_ns = int(windows[0]["start_ns"])
    gpu0_restore = gpu0_serving.get("restore_start")
    if (
        trigger["events"]["RECLAIM_TRIGGER_EMITTED"] != timings["reclaim_trigger_ns"]
        or isinstance(first_useful_ns, bool)
        or not isinstance(first_useful_ns, int)
        or first_useful_ns not in useful_tokens
        or not isinstance(gpu0_restore, dict)
        or gpu0_restore.get("observed_ns") != timings["restore_trigger_ns"]
        or not (
            timings["hbm_reclaim_confirmed_ns"]
            <= first_useful_ns
            <= slo_restoration_ns
            <= timings["restore_trigger_ns"]
        )
    ):
        raise RuntimeError("integrated v11 cross-role timestamp/accounting evidence conflicts")


def _worker_command(
    *,
    worker_path: Path,
    role: str,
    config_path: Path,
    model_snapshot: Path,
    role_root: Path,
    barrier_root: Path,
    physical_gpu_uuid: str,
) -> list[str]:
    if role not in ROLES:
        raise ValueError("integrated v11 worker role is invalid")
    return [
        sys.executable,
        str(worker_path),
        "--role",
        role,
        "--config",
        str(config_path),
        "--model-snapshot",
        str(model_snapshot),
        "--work-root",
        str(role_root),
        "--barrier-root",
        str(barrier_root),
        "--physical-gpu-uuid",
        physical_gpu_uuid,
    ]


def _transaction_command(
    *,
    config: Experiment004V11IntegratedConfig,
    effective_config: Mapping[str, Any],
    selection_sha256: str,
    sanity_result_sha256: str,
) -> dict[str, Any]:
    expected = expanded_runtime_config(config)
    if dict(effective_config) != expected:
        raise RuntimeError("v11 transaction effective config differs from expanded mapping")
    targeted = config.execution_mode == "targeted-source-identity-v11"
    return {
        "schema_version": (
            "sloforge.branchfabric.targeted-source-identity-command/v1"
            if targeted
            else "sloforge.branchfabric.integrated-transaction-command/v1"
        ),
        "effective_config": expected,
        "selection_sha256": selection_sha256,
        "authorization_artifact_hash": config.budget_authorization_sha256,
        "sanity_result_sha256": sanity_result_sha256,
        "issued_at_monotonic_ns": time.monotonic_ns(),
        **(
            {
                "terminal_phase": config.terminal_phase,
                "full_export_authorized": config.full_export_authorized,
                "source_release_authorized": config.source_release_authorized,
            }
            if targeted
            else {}
        ),
    }


def _validate_selected_load_gate_commitment(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    readiness_comparability_path: Path,
    targeted_sanity_path: Path,
    targeted_probe_roots: Sequence[Path],
) -> dict[str, Any]:
    """Revalidate both same-allocation gates before worker authorization."""

    if _sha256(selection_path) != expected_selection_sha256:
        raise RuntimeError("selected-load commitment changed before worker handoff")
    selected = _load_json_object(selection_path, label="selected load")
    expected_bindings = {
        "readiness_comparability_sha256": _sha256(readiness_comparability_path),
        "targeted_12rps_reproduction_sha256": _sha256(targeted_sanity_path),
    }
    for field, observed_sha256 in expected_bindings.items():
        committed_sha256 = selected.get(field)
        if (
            not isinstance(committed_sha256, str)
            or len(committed_sha256) != 64
            or any(character not in "0123456789abcdef" for character in committed_sha256)
            or committed_sha256 != observed_sha256
        ):
            raise RuntimeError(f"selected-load {field} changed before worker handoff")
    classification = _load_json_object(
        targeted_sanity_path,
        label="targeted serving sanity classification",
    )
    probe_rows = classification.get("probe_results")
    if (
        classification.get("schema_version")
        != "sloforge.branchfabric.targeted-serving-sanity-classification/v1"
        or not isinstance(probe_rows, list)
        or len(probe_rows) != 2
        or len(targeted_probe_roots) != 2
    ):
        raise RuntimeError("targeted serving sanity commitment chain is malformed")
    artifact_names = {
        "plan.json",
        "raw.json",
        "worker-shards.json",
        "result.json",
        "assessment.json",
        "request-state-reset.json",
    }
    for expected_probe, row, probe_root in zip(
        ("A", "B"), probe_rows, targeted_probe_roots, strict=True
    ):
        if not isinstance(row, Mapping) or row.get("probe") != expected_probe:
            raise RuntimeError("targeted serving sanity commitment probe ordering changed")
        commitment_path = probe_root / "evidence-commitment.json"
        committed_commitment_sha256 = row.get("evidence_commitment_sha256")
        if not isinstance(
            committed_commitment_sha256, str
        ) or committed_commitment_sha256 != _sha256(commitment_path):
            raise RuntimeError("targeted serving sanity evidence commitment changed")
        commitment = _load_json_object(
            commitment_path,
            label=f"targeted serving sanity probe {expected_probe} commitment",
        )
        artifacts = commitment.get("artifacts")
        if (
            commitment.get("schema_version")
            != "sloforge.branchfabric.targeted-sanity-probe-commitment/v1"
            or commitment.get("probe") != expected_probe
            or not isinstance(commitment.get("probe_id"), str)
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != artifact_names
        ):
            raise RuntimeError("targeted serving sanity evidence commitment is malformed")
        for artifact_name in artifact_names:
            artifact_sha256 = artifacts.get(artifact_name)
            if not isinstance(artifact_sha256, str) or artifact_sha256 != _sha256(
                probe_root / artifact_name
            ):
                raise RuntimeError(
                    f"targeted serving sanity {expected_probe} {artifact_name} changed"
                )
    return selected


def _wait_for_postflight_zero_compute(*, deadline_ns: int) -> tuple[dict[str, Any], ...]:
    """Tolerate bounded NVML process-list lag without extending the paid cap."""

    while True:
        processes = _compute_processes(deadline_ns=deadline_ns)
        if not processes:
            return processes
        if time.monotonic_ns() >= deadline_ns:
            _require_no_compute_processes(processes, phase="integrated v11 postflight")
        time.sleep(0.1)


def _partition_controller_deadlines(
    *,
    started_ns: int,
    caller_deadline_ns: int,
    cleanup_reserve_seconds: int,
    absolute_wall_seconds: float = ABSOLUTE_WALL_SECONDS,
) -> tuple[int, int]:
    if cleanup_reserve_seconds <= 0 or cleanup_reserve_seconds >= absolute_wall_seconds:
        raise ValueError("integrated v11 cleanup reserve is outside the absolute wall bound")
    controller_deadline_ns = min(
        caller_deadline_ns,
        started_ns + round(absolute_wall_seconds * 1e9),
    )
    operation_deadline_ns = controller_deadline_ns - cleanup_reserve_seconds * 1_000_000_000
    if operation_deadline_ns <= started_ns:
        raise TimeoutError("integrated v11 caller left no bounded operation window")
    return operation_deadline_ns, controller_deadline_ns


def _assess_retained_engine_readiness_comparability(
    readiness: Sequence[Mapping[str, Any]],
    *,
    minimum_batch16_request_throughput_rps: float = 15.0,
) -> dict[str, Any]:
    """Assess the strict batch-16 floor without selecting a provider allocation."""

    _by_role, observations, floor = _retained_engine_readiness_assessment_components(
        readiness,
        minimum_batch16_request_throughput_rps=minimum_batch16_request_throughput_rps,
    )
    comparable = all(item["passed"] for item in observations)
    return {
        "schema_version": "sloforge.branchfabric.retained-engine-readiness-comparability/v1",
        "status": "PASS" if comparable else "BELOW_FLOOR",
        "minimum_batch16_request_throughput_rps": floor,
        "devices": list(observations),
        "readiness_comparable": comparable,
        "selection_authority": False,
        "interpretation": "same-allocation hard preflight; never select a provider allocation",
    }


def _classify_targeted_serving_sanity(
    assessments: Sequence[Mapping[str, Any]],
    *,
    readiness_comparability: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared two-probe classification without averaging or retries."""

    if len(assessments) != 2:
        raise RuntimeError("targeted serving sanity requires exactly two probe assessments")
    passes: list[bool] = []
    rows: list[dict[str, Any]] = []
    for index, assessment in enumerate(assessments):
        passed = assessment.get("passed")
        if (
            assessment.get("schema_version")
            != "sloforge.branchfabric.v10-sanity-guard-assessment/v1"
            or assessment.get("guard") != "sanity_12rps_stable"
            or assessment.get("expected_rate_rps") != 12.0
            or not isinstance(passed, bool)
        ):
            raise RuntimeError("targeted serving sanity assessment differs from the exact gate")
        evidence_sha256 = assessment.get("targeted_probe_evidence_commitment_sha256")
        if (
            not isinstance(evidence_sha256, str)
            or len(evidence_sha256) != 64
            or any(character not in "0123456789abcdef" for character in evidence_sha256)
        ):
            raise RuntimeError("targeted serving sanity assessment lacks its evidence commitment")
        passes.append(passed)
        rows.append(
            {
                "probe": "A" if index == 0 else "B",
                "passed": passed,
                "evidence_commitment_sha256": evidence_sha256,
            }
        )
    pattern = (passes[0], passes[1])
    classification = TARGETED_SERVING_SANITY_CLASSIFICATIONS[pattern]
    readiness_comparable = readiness_comparability.get("readiness_comparable")
    if not isinstance(readiness_comparable, bool):
        raise RuntimeError("targeted serving sanity readiness assessment is malformed")
    return {
        "schema_version": "sloforge.branchfabric.targeted-serving-sanity-classification/v1",
        "status": "PASS",
        "terminal_phase": "TARGETED_SERVING_SANITY_CLASSIFICATION",
        "classification": classification,
        "probe_results": rows,
        "probe_pass_pattern": [*pattern],
        "readiness_comparable": readiness_comparable,
        "integrated_retry_authorized": (
            classification == "CURRENT_RETAINED_ENVELOPE_STABLE" and readiness_comparable
        ),
        "adaptive_retry_performed": False,
        "broad_capacity_calibration_performed": False,
        "full_export_authorized": False,
        "source_release_authorized": False,
    }


def _validate_targeted_probe_runtime_evidence(
    *,
    raw_payloads: Sequence[Mapping[str, Any]],
    resets: Sequence[Mapping[str, Any]],
    readiness: Sequence[Mapping[str, Any]],
    probe_id: str,
    expected_request_count: int,
    expected_cumulative_salt_count: int,
) -> None:
    """Bind each probe/reset to the same retained engines and uncached request cohort."""

    ready_by_device = {str(item.get("device")): item for item in readiness}
    raw_by_device = {str(item.get("device")): item for item in raw_payloads}
    reset_by_device = {str(item.get("device")): item for item in resets}
    if any(
        set(group) != set(DEVICES) for group in (ready_by_device, raw_by_device, reset_by_device)
    ):
        raise RuntimeError("targeted serving sanity runtime evidence lacks gpu0/gpu1")
    identity_fields = (
        "pid",
        "engine_nonce",
        "engine_started_ns",
        "engine_object_identity",
        "adapter_object_identity",
        "physical_gpu_uuid",
    )
    for device in DEVICES:
        ready = ready_by_device[device]
        raw = raw_by_device[device]
        reset = reset_by_device[device]
        continuity = raw.get("engine_continuity")
        compilation = raw.get("compilation_observation")
        salt = raw.get("cache_salt_evidence")
        policy = raw.get("prefix_cache_policy")
        observations = raw.get("observations")
        if (
            raw.get("schema_version") != "sloforge.branchfabric.capacity-worker-probe/v1"
            or raw.get("probe_id") != probe_id
            or raw.get("device") != device
            or raw.get("physical_gpu_uuid") != ready.get("physical_gpu_uuid")
            or not isinstance(continuity, Mapping)
            or any(continuity.get(field) != ready.get(field) for field in identity_fields)
            or any(reset.get(field) != ready.get(field) for field in identity_fields)
            or continuity.get("engine_reloaded") is not False
            or reset.get("engine_reloaded") is not False
            or reset.get("probe_id") != probe_id
            or not isinstance(compilation, Mapping)
            or compilation.get("no_deferred_compilation_event") is not True
            or tuple(compilation.get("events", ()))
            or not isinstance(salt, Mapping)
            or salt.get("passed") is not True
            or not isinstance(policy, Mapping)
            or policy.get("calibration_reuse_prevented") is not True
            or policy.get("method") != "per-request-vllm-cache-salt-sha256"
            or not isinstance(observations, list)
        ):
            raise RuntimeError(
                f"targeted serving sanity retained-engine evidence failed for {device}"
            )
        device_expected_count = expected_request_count if device == "gpu0" else 0
        if len(observations) != device_expected_count:
            raise RuntimeError(f"targeted serving sanity observation count failed for {device}")
        if device == "gpu0" and (
            salt.get("request_count") != expected_cumulative_salt_count
            or salt.get("unique_request_count") != expected_cumulative_salt_count
            or salt.get("unique_salt_count") != expected_cumulative_salt_count
        ):
            raise RuntimeError("targeted serving sanity cache-salt cohort is incomplete")


def _run_targeted_serving_sanity_probe(
    *,
    probe_index: int,
    probe_name: str,
    config: Experiment004V11IntegratedConfig,
    work_root: Path,
    capacity_root: Path,
    operation_deadline_ns: int,
    processes: Sequence[tuple[str, subprocess.Popen[bytes]]],
    lifecycle: _ProcessLifecycleAudit,
    readiness: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Run and persist one exact 12-rps probe without interpreting its verdict."""

    from gpu_capacity_calibration_controller import _validate_reset_payloads
    from gpu_reclamation_controller import _assess_sanity_guard

    from sloforge.helix.characterization.gpu_capacity_calibration import (
        CapacityProbeRaw,
        CapacityRequestObservation,
        ProbeTopology,
        build_probe_plan,
        evaluate_probe,
    )

    if probe_index not in {0, 1} or probe_name not in {"A", "B"}:
        raise ValueError("targeted serving sanity admits only predeclared probes A and B")
    probe_id = "v11-sanity-12rps" if probe_name == "A" else "v11-sanity-12rps-mandatory-replication"
    plan = build_probe_plan(
        probe_id=probe_id,
        seed=config.seed,
        topology=ProbeTopology.GPU0_ONLY,
        configured_rate_rps=12.0,
        start_ns=time.monotonic_ns() + 250_000_000,
        warmup_seconds=config.warmup_seconds,
        measurement_seconds=config.sanity_guard_measurement_seconds,
    )
    _write_new(
        capacity_root / f"probe-{probe_index:02d}.command.json",
        {
            "schema_version": "sloforge.branchfabric.capacity-probe-command/v1",
            "reason": "v11 targeted exact Attempt-J 12-rps sanity reproduction",
            "probe_name": probe_name,
            "plan": plan,
        },
    )
    raw_paths = tuple(
        capacity_root / f"probe-{probe_index:02d}.{device}.raw.json" for device in DEVICES
    )
    reset_paths = tuple(
        capacity_root / f"probe-{probe_index:02d}.{device}.reset.json" for device in DEVICES
    )
    _wait_for_files(
        raw_paths + reset_paths,
        deadline_ns=min(
            operation_deadline_ns,
            time.monotonic_ns() + round(config.probe_timeout_seconds * 1e9),
        ),
        phase=f"targeted serving sanity probe {probe_name}",
        processes=processes,
        observe_processes=lifecycle.observe,
    )
    raw_payloads = tuple(
        _load_json_object(path, label="targeted capacity raw shard") for path in raw_paths
    )
    if {item.get("device") for item in raw_payloads} != set(DEVICES):
        raise RuntimeError("targeted serving sanity raw shards do not cover gpu0/gpu1")
    resets = _validate_reset_payloads(
        payloads=tuple(
            _load_json_object(path, label="targeted capacity reset") for path in reset_paths
        ),
        expected_probe_id=plan.probe_id,
    )
    if {item.get("device") for item in resets} != set(DEVICES):
        raise RuntimeError("targeted serving sanity reset shards do not cover gpu0/gpu1")
    _validate_targeted_probe_runtime_evidence(
        raw_payloads=raw_payloads,
        resets=resets,
        readiness=readiness,
        probe_id=plan.probe_id,
        expected_request_count=len(plan.arrivals),
        expected_cumulative_salt_count=(probe_index + 1) * len(plan.arrivals),
    )
    observations = tuple(row for payload in raw_payloads for row in payload.get("observations", ()))
    typed_observations = tuple(
        CapacityRequestObservation.model_validate_json(_canonical_bytes(row), strict=True)
        for row in observations
    )
    raw = CapacityProbeRaw(
        plan=plan,
        observations=typed_observations,
        tail_drain_end_ns=plan.measurement_end_ns + round(config.tail_drain_seconds * 1e9),
        probe_end_ns=max(int(payload["probe_end_ns"]) for payload in raw_payloads),
        tail_drain_seconds=config.tail_drain_seconds,
    )
    result = evaluate_probe(raw, slo_ttft_seconds=config.serving_slo_ttft_seconds)
    assessment = _assess_sanity_guard(result, expected_rate_rps=12.0)
    probe_root = work_root / "sanity" / f"12-rps-probe-{probe_name.lower()}"
    # Persist the complete cohort before the caller may interpret PASS/FAIL.
    _write_new(probe_root / "plan.json", plan)
    _write_new(probe_root / "raw.json", raw)
    _write_new(probe_root / "worker-shards.json", raw_payloads)
    _write_new(probe_root / "result.json", result)
    _write_new(probe_root / "assessment.json", assessment)
    _write_new(probe_root / "request-state-reset.json", resets)
    evidence_paths = {
        name: probe_root / name
        for name in (
            "plan.json",
            "raw.json",
            "worker-shards.json",
            "result.json",
            "assessment.json",
            "request-state-reset.json",
        )
    }
    commitment_path = probe_root / "evidence-commitment.json"
    _write_new(
        commitment_path,
        {
            "schema_version": "sloforge.branchfabric.targeted-sanity-probe-commitment/v1",
            "probe": probe_name,
            "probe_id": probe_id,
            "artifacts": {name: _sha256(path) for name, path in sorted(evidence_paths.items())},
        },
    )
    return (
        {
            **assessment,
            "targeted_probe_evidence_commitment_sha256": _sha256(commitment_path),
        },
        len(plan.arrivals),
    )


def _finalize_pre_worker_failure(
    *,
    config: Experiment004V11IntegratedConfig | Experiment004V11TargetedSourceIdentityConfig,
    work_root: Path,
    lifecycle: _ProcessLifecycleAudit,
    started_ns: int,
    operation_deadline_ns: int,
    cleanup_deadline_ns: int,
    failure_stage: str,
    error: BaseException,
) -> dict[str, Any]:
    """Prove cleanup when sealed preflight fails before either worker exists."""

    cleanup_actions: list[dict[str, Any]] = []
    cleanup_errors: list[dict[str, Any]] = []
    surviving_children: tuple[dict[str, Any], ...] = ()
    processes_after: tuple[dict[str, Any], ...] = ()
    try:
        lifecycle.transition("QUIESCE")
        lifecycle.observe("pre-worker-quiesce", force=True)
        lifecycle.transition("ENGINE_STOP")
        lifecycle.transition("WORKER_STOP")
        escaped = lifecycle.currently_owned(stage="pre-worker-child-check")
        if escaped:
            _signal_exact_owned(escaped, signal.SIGTERM, cleanup_actions)
            escaped = _wait_for_owned_exit(
                lifecycle,
                (),
                deadline_ns=min(cleanup_deadline_ns, time.monotonic_ns() + 1_000_000_000),
                stage="pre-worker-child-term-wait",
            )
        if escaped:
            _signal_exact_owned(escaped, signal.SIGKILL, cleanup_actions)
        lifecycle.transition("CHILD_REAP")
        surviving_children = _reap_owned_children(
            lifecycle,
            (),
            deadline_ns=cleanup_deadline_ns,
        )
        if lifecycle.owned:
            raise RuntimeError("pre-worker preflight created an experiment-owned child")
    except BaseException as caught:
        cleanup_errors.append(
            {
                "stage": "pre-worker-child-cleanup",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
    try:
        lifecycle.subreaper.restore()
    except BaseException as caught:
        cleanup_errors.append(
            {
                "stage": "child-subreaper-restore",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
    try:
        lifecycle.transition("PGID_EMPTY")
        surviving_children = lifecycle.currently_owned(stage="pre-worker-pgid-empty")
        if surviving_children:
            raise RuntimeError("pre-worker preflight children survived cleanup")
    except BaseException as caught:
        cleanup_errors.append(
            {
                "stage": "pgid-empty",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
    try:
        processes_after = _wait_for_postflight_zero_compute(deadline_ns=cleanup_deadline_ns)
    except BaseException as caught:
        cleanup_errors.append(
            {
                "stage": "cuda-released",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
    lifecycle.transition("CUDA_RELEASED")
    after_audit = cuda_clean_import_audit("v11-integrated-pre-worker-failure-cleanup")
    if not after_audit["cuda_clean"]:
        cleanup_errors.append(
            {
                "stage": "cuda-clean-parent",
                "type": "RuntimeError",
                "message": "controller imported a CUDA-owning module",
            }
        )
    ended_ns = time.monotonic_ns()
    if ended_ns > cleanup_deadline_ns:
        cleanup_errors.append(
            {
                "stage": "absolute-deadline",
                "type": "TimeoutError",
                "message": "integrated v11 pre-worker cleanup exceeded its deadline",
            }
        )
    lifecycle.transition("FUNCTION_RETURN")
    cleanup = _build_in_function_cleanup(
        lifecycle=lifecycle,
        actions=cleanup_actions,
        processes=(),
        surviving_children=surviving_children,
        surviving_groups=(),
        compute_processes_after=processes_after,
        cuda_release_verified=not processes_after,
        pipes_closed=True,
        cleanup_errors=cleanup_errors,
    )
    work_root.mkdir(parents=True, exist_ok=True)
    _write_new(work_root / "in_function_cleanup.json", cleanup)
    targeted = config.execution_mode == "targeted-source-identity-v11"
    result = {
        "schema_version": (
            "sloforge.branchfabric.experiment-004-v11-targeted-controller/v1"
            if targeted
            else "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        ),
        "status": "failed",
        "attempt_id": config.attempt_id,
        "controller_pid": os.getpid(),
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "operation_deadline_ns": operation_deadline_ns,
        "cleanup_deadline_ns": cleanup_deadline_ns,
        "absolute_wall_seconds": config.maximum_wall_seconds,
        "execution_mode": config.execution_mode,
        "terminal_phase": (
            "SOURCE_ALLOCATION_IDENTITY_GATE" if targeted else "INTEGRATED_TRANSACTION"
        ),
        "cleanup_scope": "PRE_WORKER_PREFLIGHT",
        "failure_stage": failure_stage,
        "sealed_evidence": None,
        "inventory_before": [],
        "inventory_after": [],
        "stable_physical_gpu_identity": False,
        "worker_pids": {},
        "worker_process_groups": {},
        "worker_session_ids": {},
        "worker_returncodes": {},
        "engine_start_evidence": [],
        "readiness_evidence": [],
        "readiness_deadline_ns": None,
        "sanity_guard_pair": None,
        "worker_results": [],
        "cleanup_actions": cleanup_actions,
        "in_function_cleanup": cleanup,
        "compute_processes_after": processes_after,
        "cuda_clean_import_audits": [after_audit],
        "controller_error": {"type": type(error).__name__, "message": str(error)},
        "cleanup_error": (
            None
            if cleanup["pass"] is True
            else {
                "type": "RuntimeError",
                "message": "pre-worker in-function cleanup gate failed",
            }
        ),
    }
    _write_new(work_root / "controller-result.json", result)
    return result


def run_integrated_v11_controller(
    *,
    config_path: Path,
    work_root: Path,
    worker_path: Path,
    model_snapshot: Path,
    repository_root: Path,
    absolute_deadline_ns: int,
) -> dict[str, Any]:
    """Run exactly one two-A100 v11 integrated transaction under a 588s cap."""

    started_ns = time.monotonic_ns()
    if (
        isinstance(absolute_deadline_ns, bool)
        or not isinstance(absolute_deadline_ns, int)
        or absolute_deadline_ns <= started_ns
    ):
        raise ValueError("integrated v11 requires a future absolute deadline")
    if V11_INTEGRATED_RESERVATION_WALL_SECONDS != ABSOLUTE_WALL_SECONDS:
        raise RuntimeError("integrated v11 methodology wall bound drifted from 588 seconds")
    if not config_path.is_file() or not worker_path.is_file() or not model_snapshot.is_dir():
        raise FileNotFoundError("integrated v11 controller input path is absent")
    root = repository_root.resolve(strict=True)
    raw_config = json.loads(config_path.read_text())
    targeted = (
        raw_config.get("schema_version")
        == "sloforge.branchfabric.experiment-004-v11-targeted-source-identity-config/v1"
    )
    config = (
        Experiment004V11TargetedSourceIdentityConfig.model_validate(raw_config, strict=True)
        if targeted
        else Experiment004V11IntegratedConfig.model_validate(raw_config, strict=True)
    )
    operation_deadline_ns, cleanup_deadline_ns = _partition_controller_deadlines(
        started_ns=started_ns,
        caller_deadline_ns=absolute_deadline_ns,
        cleanup_reserve_seconds=config.cleanup_timeout_seconds,
        absolute_wall_seconds=config.maximum_wall_seconds,
    )
    lifecycle = _ProcessLifecycleAudit()
    lifecycle.transition("RUNNING")
    preflight_stage = "SEALED_EVIDENCE"
    try:
        sealed = verify_sealed_evidence(config, root)
        # Constructing the typed live contract is an additional exact-methodology
        # check.  The worker protocol itself consumes the expanded JSON mapping.
        preflight_stage = "LIVE_CONTRACT"
        build_live_v10_config(config)
        effective_config = expanded_runtime_config(config)
        if time.monotonic_ns() >= operation_deadline_ns:
            raise TimeoutError("integrated v11 sealed preflight exhausted its operation deadline")

        # Load CPU-only capacity contracts after the controller's own sealed-input
        # checks.  These modules define JSON models; neither owns a CUDA context.
        preflight_stage = "CPU_CONTROL_IMPORTS"
        from gpu_capacity_calibration_controller import (
            _validate_readiness_payloads,
            _validate_reset_payloads,
        )
        from gpu_reclamation_controller import _assess_sanity_guard

        from sloforge.helix.characterization.gpu_capacity_calibration import (
            CapacityProbeRaw,
            ProbeTopology,
            build_probe_plan,
            evaluate_probe,
        )

        preflight_stage = "PRE_WORKER_ARTIFACT_INITIALIZATION"
        work_root.mkdir(parents=True, exist_ok=False)
        barrier_root = work_root / "barriers"
        capacity_root = barrier_root / "capacity"
        log_root = work_root / "logs"
        capacity_root.mkdir(parents=True)
        log_root.mkdir()
        base_config_path = work_root / "base-config.json"
        _write_new(base_config_path, config)
        _write_new(work_root / "authorization/sealed-evidence.json", sealed)

        preflight_stage = "PRE_INVENTORY_CUDA_CLEAN_AUDIT"
        before_audit = cuda_clean_import_audit("v11-integrated-before-inventory")
        if not before_audit["cuda_clean"]:
            raise RuntimeError("integrated v11 controller imported a CUDA-owning module")
        preflight_stage = "GPU_INVENTORY_BEFORE_WORKERS"
        inventory_before = _inventory(deadline_ns=operation_deadline_ns)
        preflight_stage = "ZERO_COMPUTE_BEFORE_WORKERS"
        _require_no_compute_processes(
            _compute_processes(deadline_ns=operation_deadline_ns),
            phase="integrated v11 preflight",
        )
    except BaseException as preflight_error:
        return _finalize_pre_worker_failure(
            config=config,
            work_root=work_root,
            lifecycle=lifecycle,
            started_ns=started_ns,
            operation_deadline_ns=operation_deadline_ns,
            cleanup_deadline_ns=cleanup_deadline_ns,
            failure_stage=preflight_stage,
            error=preflight_error,
        )

    children: list[tuple[str, subprocess.Popen[bytes], Any, Any]] = []
    cleanup_actions: list[dict[str, Any]] = []
    cleanup_errors: list[dict[str, Any]] = []
    controller_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    readiness_deadline_ns: int | None = None
    engine_start_evidence: tuple[dict[str, Any], ...] = ()
    ready_payloads: tuple[dict[str, Any], ...] = ()
    readiness_comparability: dict[str, Any] | None = None
    targeted_sanity_classification: dict[str, Any] | None = None
    sanity_pair: dict[str, Any] | None = None
    result_payloads: tuple[dict[str, Any], ...] = ()
    inventory_after: tuple[dict[str, Any], ...] = ()
    processes_after: tuple[dict[str, Any], ...] = ()
    surviving_children: tuple[dict[str, Any], ...] = ()
    surviving_groups: tuple[int, ...] = ()
    cuda_release_verified = False
    in_function_cleanup: dict[str, Any] | None = None
    try:
        launch_started_ns = time.monotonic_ns()
        for role, gpu in zip(ROLES, inventory_before, strict=True):
            role_root = work_root / role
            role_root.mkdir()
            stdout_handle = (log_root / f"{role}.stdout.log").open("xb")
            stderr_handle = (log_root / f"{role}.stderr.log").open("xb")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_VISIBLE_GPU_INDEX"] = str(gpu["index"])
            environment["SLOFORGE_EXP004_PHYSICAL_GPU_UUID"] = str(gpu["uuid"])
            command = _worker_command(
                worker_path=worker_path,
                role=role,
                config_path=config_path,
                model_snapshot=model_snapshot,
                role_root=role_root,
                barrier_root=barrier_root,
                physical_gpu_uuid=str(gpu["uuid"]),
            )
            try:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=environment,
                    start_new_session=True,
                )
            except BaseException:
                stdout_handle.close()
                stderr_handle.close()
                raise
            children.append((role, child, stdout_handle, stderr_handle))
            lifecycle.register_worker(role, child)
        processes = tuple((role, child) for role, child, *_ in children)

        engine_paths = tuple(barrier_root / f"{role}.engine-started.json" for role in ROLES)
        _wait_for_files(
            engine_paths,
            deadline_ns=operation_deadline_ns,
            phase="retained-engine startup",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        engine_start_evidence = _validate_engine_start(
            tuple(_load_json_object(path, label="engine-start") for path in engine_paths),
            children=children,
            inventory=inventory_before,
            launch_started_ns=launch_started_ns,
            observed_ns=time.monotonic_ns(),
        )
        readiness_origin_ns = min(item["engine_started_ns"] for item in engine_start_evidence)
        readiness_deadline_ns = min(
            operation_deadline_ns,
            readiness_origin_ns + config.initialization_timeout_seconds * 1_000_000_000,
        )
        ready_paths = tuple(barrier_root / f"{role}.ready.json" for role in ROLES)
        _wait_for_files(
            ready_paths,
            deadline_ns=readiness_deadline_ns,
            phase="integrated v11 readiness",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        ready_payloads = tuple(
            _validate_readiness_payloads(
                payloads=tuple(
                    _load_json_object(path, label="engine readiness") for path in ready_paths
                ),
                expected_inventory=inventory_before,
            )
        )
        _write_new(
            work_root / "readiness/both-engines-ready.json",
            {
                "schema_version": "sloforge.branchfabric.both-engines-ready/v1",
                "BOTH_ENGINES_READY": True,
                "attempt_id": config.attempt_id,
                "engine_evidence": ready_payloads,
                "observed_at_monotonic_ns": time.monotonic_ns(),
            },
        )

        # Attempt J exposed a whole-allocation slowdown before the normal 12/15
        # stale-calibration guards.  The next integrated attempt therefore
        # performs one predeclared A/B reproduction on these same retained
        # engines.  This is not a provider-selection probe: failure terminates
        # this allocation, and success continues on this allocation only.
        readiness_comparability = _assess_retained_engine_readiness_comparability(ready_payloads)
        readiness_comparability_path = work_root / "readiness/retained-engine-comparability.json"
        _write_new(readiness_comparability_path, readiness_comparability)
        _validate_retained_engine_readiness_comparability(ready_payloads)
        reproduction_assessments: list[dict[str, Any]] = []
        reproduction_request_counts: list[int] = []
        for probe_index, probe_name in enumerate(("A", "B")):
            probe_assessment, probe_request_count = _run_targeted_serving_sanity_probe(
                probe_index=probe_index,
                probe_name=probe_name,
                config=config,
                work_root=work_root,
                capacity_root=capacity_root,
                operation_deadline_ns=operation_deadline_ns,
                processes=processes,
                lifecycle=lifecycle,
                readiness=ready_payloads,
            )
            reproduction_assessments.append(probe_assessment)
            reproduction_request_counts.append(probe_request_count)
        targeted_sanity_classification = _classify_targeted_serving_sanity(
            reproduction_assessments,
            readiness_comparability=readiness_comparability,
        )
        targeted_sanity_path = work_root / "sanity/targeted-12rps-reproduction.json"
        _write_new(targeted_sanity_path, targeted_sanity_classification)
        if targeted_sanity_classification.get("integrated_retry_authorized") is not True:
            raise RuntimeError(
                "Attempt-J 12-rps A/B reproduction did not authorize the integrated transaction"
            )

        # Probe A is the unchanged canonical 12-rps guard.  Probe B is its
        # mandatory replication, while the sole remaining legacy probe is 15 rps.
        assessments: list[dict[str, Any]] = [reproduction_assessments[0]]
        for probe_index, rate in enumerate(SANITY_RATES_RPS[1:], start=2):
            plan = build_probe_plan(
                probe_id=f"v11-sanity-{int(rate)}rps",
                seed=config.seed,
                topology=ProbeTopology.GPU0_ONLY,
                configured_rate_rps=rate,
                start_ns=time.monotonic_ns() + 250_000_000,
                warmup_seconds=config.warmup_seconds,
                measurement_seconds=config.sanity_guard_measurement_seconds,
            )
            _write_new(
                capacity_root / f"probe-{probe_index:02d}.command.json",
                {
                    "schema_version": "sloforge.branchfabric.capacity-probe-command/v1",
                    "reason": "v11 preauthorized short stale-calibration sanity guard",
                    "plan": plan,
                },
            )
            raw_paths = tuple(
                capacity_root / f"probe-{probe_index:02d}.{device}.raw.json" for device in DEVICES
            )
            reset_paths = tuple(
                capacity_root / f"probe-{probe_index:02d}.{device}.reset.json" for device in DEVICES
            )
            _wait_for_files(
                raw_paths + reset_paths,
                deadline_ns=min(
                    operation_deadline_ns,
                    time.monotonic_ns() + round(config.probe_timeout_seconds * 1e9),
                ),
                phase=f"v11 sanity probe {probe_index}",
                processes=processes,
                observe_processes=lifecycle.observe,
            )
            raw_payloads = tuple(
                _load_json_object(path, label="capacity raw shard") for path in raw_paths
            )
            if {item.get("device") for item in raw_payloads} != set(DEVICES):
                raise RuntimeError("v11 sanity raw shards do not cover gpu0/gpu1")
            resets = _validate_reset_payloads(
                payloads=tuple(
                    _load_json_object(path, label="capacity reset") for path in reset_paths
                ),
                expected_probe_id=plan.probe_id,
            )
            if {item.get("device") for item in resets} != set(DEVICES):
                raise RuntimeError("v11 sanity reset shards do not cover gpu0/gpu1")
            _validate_targeted_probe_runtime_evidence(
                raw_payloads=raw_payloads,
                resets=resets,
                readiness=ready_payloads,
                probe_id=plan.probe_id,
                expected_request_count=len(plan.arrivals),
                expected_cumulative_salt_count=(
                    sum(reproduction_request_counts) + len(plan.arrivals)
                ),
            )
            observations = tuple(
                row for payload in raw_payloads for row in payload.get("observations", ())
            )
            encoded_rows = _canonical_bytes(list(observations))
            from sloforge.helix.characterization.gpu_capacity_calibration import (
                CapacityRequestObservation,
            )

            typed_observations = tuple(
                CapacityRequestObservation.model_validate_json(_canonical_bytes(row), strict=True)
                for row in json.loads(encoded_rows)
            )
            expected_tail_end_ns = plan.measurement_end_ns + round(config.tail_drain_seconds * 1e9)
            raw = CapacityProbeRaw(
                plan=plan,
                observations=typed_observations,
                tail_drain_end_ns=expected_tail_end_ns,
                probe_end_ns=max(int(payload["probe_end_ns"]) for payload in raw_payloads),
                tail_drain_seconds=config.tail_drain_seconds,
            )
            result = evaluate_probe(raw, slo_ttft_seconds=config.serving_slo_ttft_seconds)
            gpu1_payload = next(item for item in raw_payloads if item.get("device") == "gpu1")
            if gpu1_payload.get("observations") != []:
                raise RuntimeError("GPU1 received work during a GPU0-only sanity guard")
            assessment = _assess_sanity_guard(result, expected_rate_rps=rate)
            probe_root = work_root / "sanity" / f"{int(rate)}-rps"
            _write_new(probe_root / "plan.json", plan)
            _write_new(probe_root / "raw.json", raw)
            _write_new(probe_root / "worker-shards.json", raw_payloads)
            _write_new(probe_root / "result.json", result)
            _write_new(probe_root / "assessment.json", assessment)
            _write_new(probe_root / "request-state-reset.json", resets)
            if assessment.get("passed") is not True:
                raise RuntimeError(f"v11 sanity guard failed: {assessment}")
            assessments.append(assessment)
        sanity_pair = validate_sanity_guard_pair(assessments)
        sanity_path = work_root / "sanity/sanity-result.json"
        _write_new(sanity_path, sanity_pair)
        sanity_sha256 = _sha256(sanity_path)

        selection_path = work_root / "calibration/selected-load.json"
        _write_new(
            selection_path,
            {
                "schema_version": "sloforge.branchfabric.v11-stale-calibration-selection/v1",
                "lambda_1_rps": config.lambda_1_rps,
                "lambda_spike_rps": config.lambda_spike_rps,
                "lambda_2_rps": config.lambda_2_rps,
                "sanity_result_sha256": sanity_sha256,
                "readiness_comparability_sha256": _sha256(readiness_comparability_path),
                "targeted_12rps_reproduction_sha256": _sha256(targeted_sanity_path),
                "broad_capacity_calibration_performed": False,
            },
        )
        selection_sha256 = _sha256(selection_path)
        _validate_selected_load_gate_commitment(
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
            readiness_comparability_path=readiness_comparability_path,
            targeted_sanity_path=targeted_sanity_path,
            targeted_probe_roots=tuple(
                work_root / "sanity" / f"12-rps-probe-{probe_name}" for probe_name in ("a", "b")
            ),
        )
        _write_new(
            capacity_root / "final-reset.command.json",
            {
                "schema_version": "sloforge.branchfabric.integrated-final-reset-command/v1",
                "selected_load_sha256": selection_sha256,
                "authorization_artifact_hash": config.budget_authorization_sha256,
                "issued_at_monotonic_ns": time.monotonic_ns(),
            },
        )
        final_reset_paths = tuple(
            capacity_root / f"final-reset.{device}.json" for device in DEVICES
        )
        _wait_for_files(
            final_reset_paths,
            deadline_ns=operation_deadline_ns,
            phase="v11 final request reset",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        final_resets = _validate_reset_payloads(
            payloads=tuple(
                _load_json_object(path, label="final reset") for path in final_reset_paths
            ),
            expected_probe_id="final-calibration-reset",
        )
        if {item.get("device") for item in final_resets} != set(DEVICES):
            raise RuntimeError("v11 final reset shards do not cover gpu0/gpu1")
        continuity = {
            str(payload["device"]): {
                "pid": int(payload["pid"]),
                "engine_nonce": str(payload["engine_nonce"]),
                "physical_gpu_uuid": str(payload["physical_gpu_uuid"]),
            }
            for payload in ready_payloads
        }
        if set(continuity) != set(DEVICES):
            raise RuntimeError("readiness identities do not cover gpu0/gpu1")
        for payload in final_resets:
            device = str(payload["device"])
            if (
                payload.get("selected_load_sha256") != selection_sha256
                or payload.get("authorization_artifact_hash") != config.budget_authorization_sha256
                or any(payload.get(key) != continuity[device][key] for key in continuity[device])
            ):
                raise RuntimeError("worker identity changed before final reset")

        effective_path = work_root / "effective-config.json"
        _write_new(effective_path, effective_config)
        _validate_selected_load_gate_commitment(
            selection_path=selection_path,
            expected_selection_sha256=selection_sha256,
            readiness_comparability_path=readiness_comparability_path,
            targeted_sanity_path=targeted_sanity_path,
            targeted_probe_roots=tuple(
                work_root / "sanity" / f"12-rps-probe-{probe_name}" for probe_name in ("a", "b")
            ),
        )
        _write_new(
            barrier_root / "transaction.command.json",
            _transaction_command(
                config=config,
                effective_config=effective_config,
                selection_sha256=selection_sha256,
                sanity_result_sha256=sanity_sha256,
            ),
        )
        handoff_paths = tuple(barrier_root / f"{role}.transaction-ready.json" for role in ROLES)
        _wait_for_files(
            handoff_paths,
            deadline_ns=operation_deadline_ns,
            phase="v11 transaction handoff",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        _validate_handoffs(
            tuple(_load_json_object(path, label="transaction handoff") for path in handoff_paths),
            continuity=continuity,
            selection_sha256=selection_sha256,
            sanity_result_sha256=sanity_sha256,
            authorization_sha256=config.budget_authorization_sha256,
        )
        start_ns = time.monotonic_ns() + 1_000_000_000
        _write_new(
            barrier_root / "start.json",
            {
                "schema_version": "sloforge.branchfabric.experiment-004-start/v1",
                "start_monotonic_ns": start_ns,
                "issued_at_monotonic_ns": time.monotonic_ns(),
                "issued_at_utc": _utc_now(),
            },
        )
        while any(child.poll() is None for _, child, *_ in children):
            lifecycle.observe("integrated-v11-transaction")
            if time.monotonic_ns() >= operation_deadline_ns:
                raise TimeoutError("integrated v11 transaction exceeded the 588-second bound")
            failed = [
                (role, child.returncode)
                for role, child, *_ in children
                if child.poll() not in {None, 0}
            ]
            if failed:
                raise RuntimeError(f"an integrated v11 worker failed: {failed}")
            time.sleep(0.1)
        returncodes = {role: child.returncode for role, child, *_ in children}
        if set(returncodes.values()) != {0}:
            raise RuntimeError(f"integrated v11 worker return codes failed: {returncodes}")
        result_paths = tuple(work_root / role / "result.json" for role in ROLES)
        _wait_for_files(
            result_paths,
            deadline_ns=operation_deadline_ns,
            phase="integrated v11 worker results",
            processes=processes,
            observe_processes=lifecycle.observe,
        )
        expected = {
            role: (child.pid, str(gpu["uuid"]))
            for (role, child, *_), gpu in zip(children, inventory_before, strict=True)
        }
        result_payloads = _validate_worker_results(
            tuple(_load_json_object(path, label="worker result") for path in result_paths),
            expected=expected,
            attempt_id=config.attempt_id,
            targeted=targeted,
        )
        manifest_paths = sorted(
            {
                base_config_path,
                effective_path,
                sanity_path,
                selection_path,
                readiness_comparability_path,
                targeted_sanity_path,
                *(
                    work_root / "sanity" / f"12-rps-probe-{probe_name}" / artifact_name
                    for probe_name in ("a", "b")
                    for artifact_name in (
                        "plan.json",
                        "raw.json",
                        "worker-shards.json",
                        "result.json",
                        "assessment.json",
                        "request-state-reset.json",
                        "evidence-commitment.json",
                    )
                ),
                barrier_root / "transaction.command.json",
                *(work_root / role / "result.json" for role in ROLES),
            },
            key=str,
        )
        _write_new(
            work_root
            / ("targeted-run-manifest.json" if targeted else "integrated-run-manifest.json"),
            {
                "schema_version": (
                    "sloforge.branchfabric.v11-targeted-source-identity-run-manifest/v1"
                    if targeted
                    else "sloforge.branchfabric.v11-integrated-run-manifest/v1"
                ),
                "attempt_id": config.attempt_id,
                "single_worker_pair": True,
                "engine_reload_count": 0,
                "sanity_and_transaction_same_allocation": True,
                "broad_capacity_calibration_performed": False,
                "absolute_wall_seconds": config.maximum_wall_seconds,
                "sealed_evidence": sealed,
                "artifacts": [
                    {
                        "relative_path": str(path.relative_to(work_root)),
                        "sha256": _sha256(path),
                    }
                    for path in manifest_paths
                ],
            },
        )
    except BaseException as caught:
        controller_error = caught
        abort = barrier_root / "abort.json"
        if not abort.exists():
            with suppress(FileExistsError):
                _write_new(
                    abort,
                    {
                        "schema_version": "sloforge.branchfabric.integrated-abort/v1",
                        "error": type(caught).__name__,
                        "issued_at_monotonic_ns": time.monotonic_ns(),
                    },
                )
    finally:
        process_objects = tuple(child for _role, child, *_ in children)
        try:
            lifecycle.transition("QUIESCE")
            lifecycle.observe("quiesce", force=True)
            lifecycle.transition("ENGINE_STOP")
            graceful_deadline_ns = min(
                cleanup_deadline_ns,
                time.monotonic_ns() + 1_000_000_000,
            )
            _wait_for_owned_exit(
                lifecycle,
                process_objects,
                deadline_ns=graceful_deadline_ns,
                stage="engine-stop-grace",
            )
            lifecycle.transition("WORKER_STOP")
            _terminate_groups(
                process_objects,
                cleanup_actions,
                deadline_ns=cleanup_deadline_ns,
                lifecycle=lifecycle,
            )
        except BaseException as caught:
            cleanup_errors.append(
                {
                    "stage": "worker-stop",
                    "type": type(caught).__name__,
                    "message": str(caught),
                }
            )
            if cleanup_error is None:
                cleanup_error = caught
        finally:
            try:
                escaped = lifecycle.currently_owned(stage="escaped-helper-term")
                grouped = {int(item["pgid"]) for item in lifecycle.worker_roots.values()}
                escaped = tuple(item for item in escaped if int(item["pgid"]) not in grouped)
                if escaped:
                    _signal_exact_owned(escaped, signal.SIGTERM, cleanup_actions)
                    escaped = _wait_for_owned_exit(
                        lifecycle,
                        process_objects,
                        deadline_ns=min(
                            cleanup_deadline_ns,
                            time.monotonic_ns() + 1_000_000_000,
                        ),
                        stage="escaped-helper-term-wait",
                    )
                    escaped = tuple(item for item in escaped if int(item["pgid"]) not in grouped)
                if escaped:
                    _signal_exact_owned(escaped, signal.SIGKILL, cleanup_actions)
                lifecycle.transition("CHILD_REAP")
                surviving_children = _reap_owned_children(
                    lifecycle,
                    process_objects,
                    deadline_ns=cleanup_deadline_ns,
                )
            except BaseException as caught:
                cleanup_errors.append(
                    {
                        "stage": "child-reap",
                        "type": type(caught).__name__,
                        "message": str(caught),
                    }
                )
                if cleanup_error is None:
                    cleanup_error = caught
            for _role, _child, stdout_handle, stderr_handle in children:
                stdout_handle.close()
                stderr_handle.close()
            try:
                lifecycle.subreaper.restore()
            except BaseException as caught:
                cleanup_errors.append(
                    {
                        "stage": "child-subreaper-restore",
                        "type": type(caught).__name__,
                        "message": str(caught),
                    }
                )
                if cleanup_error is None:
                    cleanup_error = caught
            try:
                lifecycle.transition("PGID_EMPTY")
                _poll_and_reap_workers(process_objects, lifecycle=lifecycle)
                surviving_children = lifecycle.currently_owned(stage="pgid-empty")
                surviving_groups = tuple(
                    sorted(
                        int(item["pgid"])
                        for item in lifecycle.worker_roots.values()
                        if _process_group_exists(int(item["pgid"]))
                    )
                )
                if surviving_children or surviving_groups:
                    raise RuntimeError(
                        "experiment-owned processes remain after bounded cleanup: "
                        f"pids={[item['pid'] for item in surviving_children]}, "
                        f"pgids={list(surviving_groups)}"
                    )
            except BaseException as caught:
                cleanup_errors.append(
                    {
                        "stage": "pgid-empty",
                        "type": type(caught).__name__,
                        "message": str(caught),
                    }
                )
                if cleanup_error is None:
                    cleanup_error = caught

    try:
        processes_after = _wait_for_postflight_zero_compute(deadline_ns=cleanup_deadline_ns)
        inventory_after = _inventory(deadline_ns=cleanup_deadline_ns)
        if not _stable_inventory_identity(inventory_before, inventory_after):
            raise RuntimeError("physical GPU identity changed during integrated v11")
        cuda_release_verified = True
    except BaseException as caught:
        cleanup_errors.append(
            {
                "stage": "cuda-released",
                "type": type(caught).__name__,
                "message": str(caught),
            }
        )
        if cleanup_error is None:
            cleanup_error = caught
    lifecycle.transition("CUDA_RELEASED")
    after_audit = cuda_clean_import_audit("v11-integrated-after-worker-cleanup")
    if not after_audit["cuda_clean"]:
        audit_error = RuntimeError("controller imported a CUDA-owning module")
        cleanup_errors.append(
            {
                "stage": "cuda-clean-parent",
                "type": type(audit_error).__name__,
                "message": str(audit_error),
            }
        )
        if cleanup_error is None:
            cleanup_error = audit_error
    ended_ns = time.monotonic_ns()
    if ended_ns > cleanup_deadline_ns:
        deadline_error = TimeoutError("integrated v11 controller exceeded its absolute deadline")
        cleanup_errors.append(
            {
                "stage": "absolute-deadline",
                "type": type(deadline_error).__name__,
                "message": str(deadline_error),
            }
        )
        if cleanup_error is None:
            cleanup_error = deadline_error
    lifecycle.transition("FUNCTION_RETURN")
    in_function_cleanup = _build_in_function_cleanup(
        lifecycle=lifecycle,
        actions=cleanup_actions,
        processes=tuple(child for _role, child, *_ in children),
        surviving_children=surviving_children,
        surviving_groups=surviving_groups,
        compute_processes_after=processes_after,
        cuda_release_verified=cuda_release_verified,
        pipes_closed=all(
            stdout_handle.closed and stderr_handle.closed
            for _role, _child, stdout_handle, stderr_handle in children
        ),
        cleanup_errors=cleanup_errors,
    )
    _write_new(work_root / "in_function_cleanup.json", in_function_cleanup)
    if in_function_cleanup["pass"] is not True and cleanup_error is None:
        cleanup_error = RuntimeError("integrated v11 in-function cleanup gate failed")
    status = (
        "succeeded"
        if controller_error is None
        and cleanup_error is None
        and in_function_cleanup["pass"] is True
        else "failed"
    )
    controller_result = {
        "schema_version": (
            "sloforge.branchfabric.experiment-004-v11-targeted-controller/v1"
            if targeted
            else "sloforge.branchfabric.experiment-004-v11-integrated-controller/v1"
        ),
        "status": status,
        "attempt_id": config.attempt_id,
        "controller_pid": os.getpid(),
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "operation_deadline_ns": operation_deadline_ns,
        "cleanup_deadline_ns": cleanup_deadline_ns,
        "absolute_wall_seconds": config.maximum_wall_seconds,
        "execution_mode": config.execution_mode,
        "terminal_phase": (
            "SOURCE_ALLOCATION_IDENTITY_GATE" if targeted else "INTEGRATED_TRANSACTION"
        ),
        "sealed_evidence": sealed,
        "inventory_before": inventory_before,
        "inventory_after": inventory_after,
        "stable_physical_gpu_identity": _stable_inventory_identity(
            inventory_before, inventory_after
        ),
        "worker_pids": {role: child.pid for role, child, *_ in children},
        "worker_process_groups": {
            str(item["role"]): int(item["pgid"]) for item in lifecycle.worker_roots.values()
        },
        "worker_session_ids": {
            str(item["role"]): int(item["sid"]) for item in lifecycle.worker_roots.values()
        },
        "worker_returncodes": {role: child.returncode for role, child, *_ in children},
        "engine_start_evidence": engine_start_evidence,
        "readiness_evidence": ready_payloads,
        "readiness_comparability": readiness_comparability,
        "readiness_deadline_ns": readiness_deadline_ns,
        "targeted_12rps_reproduction": targeted_sanity_classification,
        "sanity_guard_pair": sanity_pair,
        "worker_results": result_payloads,
        "cleanup_actions": cleanup_actions,
        "in_function_cleanup": in_function_cleanup,
        "compute_processes_after": processes_after,
        "cuda_clean_import_audits": (before_audit, after_audit),
        "controller_error": None
        if controller_error is None
        else {"type": type(controller_error).__name__, "message": str(controller_error)},
        "cleanup_error": None
        if cleanup_error is None
        else {"type": type(cleanup_error).__name__, "message": str(cleanup_error)},
    }
    _write_new(work_root / "controller-result.json", controller_result)
    return controller_result


__all__ = [
    "ABSOLUTE_WALL_SECONDS",
    "cuda_clean_import_audit",
    "run_integrated_v11_controller",
    "verify_sealed_evidence",
]
