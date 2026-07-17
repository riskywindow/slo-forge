"""Capability-gated PyTorch and PEFT trainer adapters.

The optional adapters import external frameworks only after their version and
symbol probes pass.  Normal CI never downloads a model.  PEFT accepts only a
caller-provided local model directory and a provenance-complete Helix batch.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sloforge.continuum.adapters.external import IntegrationStatus
from sloforge.continuum.adapters.pytorch import probe_pytorch
from sloforge.helix.datasets import ReferenceTrainingBatchManifest
from sloforge.helix.policy import DeterministicPolicy
from sloforge.helix.trainers.reference import ReferenceTrainer, TrainingAlgorithm


class _OptionalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AdapterCapability(_OptionalModel):
    adapter: Literal["pytorch", "peft_lora"]
    available: bool
    version: str | None
    exercised: bool
    reason: str
    official_api_evidence: tuple[str, ...]


class OptionalTrainingResult(_OptionalModel):
    schema_version: Literal["sloforge.helix.optional-training/v1"] = (
        "sloforge.helix.optional-training/v1"
    )
    adapter: Literal["pytorch", "peft_lora"]
    base_policy_epoch_id: str
    candidate_policy_epoch_id: str
    algorithm: TrainingAlgorithm
    batch_id: str
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    steps: Annotated[int, Field(gt=0, le=10_000)]
    checkpoint_path: str
    checkpoint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    framework_version: str
    final_loss: float
    candidate: DeterministicPolicy | None = None
    validation_class: Literal["hardware-backed", "local-cpu-framework"]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not math.isfinite(self.final_loss):
            raise ValueError("optional trainer loss must be finite")
        if self.adapter == "pytorch":
            if self.candidate is None:
                raise ValueError("PyTorch result must carry its inspectable candidate")
            if self.candidate.policy_epoch_id != self.candidate_policy_epoch_id:
                raise ValueError("candidate policy identity disagrees with trainer result")
        elif self.candidate is not None:
            raise ValueError("PEFT result cannot claim a categorical policy candidate")
        return self


class PeftTrainingExample(_OptionalModel):
    sample_id: str
    prompt: Annotated[str, Field(min_length=1, max_length=65_536)]
    completion: Annotated[str, Field(min_length=1, max_length=65_536)]


class PeftTrainingRequest(_OptionalModel):
    model_directory: Path
    base_policy_epoch_id: str
    candidate_policy_epoch_id: str
    batch: ReferenceTrainingBatchManifest
    examples: Annotated[tuple[PeftTrainingExample, ...], Field(min_length=1, max_length=65_536)]
    algorithm: Literal[TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION] = (
        TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION
    )
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)]
    steps: Annotated[int, Field(gt=0, le=10_000)] = 1
    learning_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 0.0001
    lora_rank: Annotated[int, Field(gt=0, le=256)] = 8
    maximum_sequence_tokens: Annotated[int, Field(gt=1, le=16_384)] = 512
    controlled_environment: bool = False

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if not self.model_directory.is_absolute() or not self.model_directory.is_dir():
            raise ValueError("PEFT requires an existing absolute local model directory")
        if self.batch.learner_policy_epoch_id != self.base_policy_epoch_id:
            raise ValueError("PEFT base policy differs from the batch learner policy")
        if self.batch.algorithm is not self.algorithm:
            raise ValueError("PEFT request algorithm differs from the sealed training batch")
        if self.candidate_policy_epoch_id == self.base_policy_epoch_id:
            raise ValueError("PEFT candidate epoch must differ from the immutable base epoch")
        if not math.isfinite(self.learning_rate):
            raise ValueError("PEFT learning rate must be finite")
        expected = set(self.batch.training_sample_ids)
        observed = {item.sample_id for item in self.examples}
        if len(observed) != len(self.examples) or observed != expected:
            raise ValueError("PEFT examples must cover each training sample exactly once")
        return self


def probe_optional_pytorch() -> AdapterCapability:
    probe = probe_pytorch()
    available = probe.status is IntegrationStatus.READY
    return AdapterCapability(
        adapter="pytorch",
        available=available,
        version=probe.runtime_version,
        exercised=False,
        reason=(
            "version and required public symbols are available"
            if available
            else "; ".join(probe.missing_requirements) or "PyTorch is not installed"
        ),
        official_api_evidence=probe.evidence,
    )


def probe_peft_lora() -> AdapterCapability:
    evidence = (
        "https://huggingface.co/docs/peft/package_reference/lora",
        "https://huggingface.co/docs/peft/package_reference/peft_model",
        "https://huggingface.co/docs/transformers/peft",
    )
    missing: list[str] = []
    versions: list[str] = []
    for distribution, module, symbols in (
        ("peft", "peft", ("LoraConfig", "get_peft_model")),
        ("transformers", "transformers", ("AutoModelForCausalLM", "AutoTokenizer")),
    ):
        try:
            version = importlib.metadata.version(distribution)
            loaded = importlib.import_module(module)
        except (importlib.metadata.PackageNotFoundError, ImportError):
            missing.append(f"{distribution} is not installed")
            continue
        versions.append(f"{distribution}={version}")
        for symbol in symbols:
            if not hasattr(loaded, symbol):
                missing.append(f"{module}:{symbol}")
    torch_capability = probe_optional_pytorch()
    if not torch_capability.available:
        missing.append("compatible PyTorch runtime")
    return AdapterCapability(
        adapter="peft_lora",
        available=not missing,
        version=",".join(versions) or None,
        exercised=False,
        reason="public PEFT/Transformers APIs are available" if not missing else "; ".join(missing),
        official_api_evidence=evidence,
    )


def _require_pytorch() -> Any:
    capability = probe_optional_pytorch()
    if not capability.available:
        raise RuntimeError(f"PyTorch trainer unavailable: {capability.reason}")
    return importlib.import_module("torch")


class PyTorchTinyTrainerAdapter:
    """Real autograd adapter for the bounded categorical Helix policy."""

    def train(
        self,
        *,
        base: DeterministicPolicy,
        batch: ReferenceTrainingBatchManifest,
        candidate_policy_epoch_id: str,
        output: Path,
        seed: int,
        steps: int = 8,
        learning_rate: float = 0.1,
    ) -> OptionalTrainingResult:
        if batch.behavior_policy_epoch_id != base.policy_epoch_id:
            raise ValueError("PyTorch trainer rejects a batch from another behavior policy")
        if batch.learner_policy_epoch_id != base.policy_epoch_id:
            raise ValueError("PyTorch base policy differs from the batch learner policy")
        if candidate_policy_epoch_id == base.policy_epoch_id:
            raise ValueError("candidate policy epoch must differ from the immutable base epoch")
        if seed < 0 or seed > 2**64 - 1:
            raise ValueError("seed must fit an unsigned 64-bit value")
        if not 1 <= steps <= 10_000:
            raise ValueError("PyTorch trainer steps must be in 1..10000")
        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ValueError("PyTorch learning rate must be in (0, 1]")
        if output.exists():
            raise FileExistsError("PyTorch checkpoint output already exists")
        torch = _require_pytorch()
        torch.manual_seed(seed)
        logits = torch.tensor(base.logits, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD((logits,), lr=learning_rate)
        reference = torch.softmax(
            torch.tensor(base.logits, dtype=torch.float64) / base.temperature, dim=0
        )
        accepted = tuple(
            item
            for item in batch.trainer_samples()
            if ReferenceTrainer._effective_sample(item, batch.algorithm)
        )
        if not accepted:
            raise ValueError("PyTorch trainer received no eligible samples")
        if any(item.action not in base.actions for item in accepted):
            raise ValueError("PyTorch training sample references an unknown action")
        if any(
            not math.isclose(
                item.behavior_log_probability,
                base.log_probability(item.action),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for item in accepted
        ):
            raise ValueError("behavior log probability disagrees with the immutable policy epoch")
        final_loss = math.nan
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            probabilities = torch.softmax(logits / base.temperature, dim=0)
            log_probabilities = torch.log_softmax(logits / base.temperature, dim=0)
            loss = torch.zeros((), dtype=torch.float64)
            for sample in accepted:
                index = base.actions.index(sample.action)
                if batch.algorithm is TrainingAlgorithm.SUCCESSFUL_BRANCH_DISTILLATION:
                    weight = sample.token_weight * max(sample.reward_margin, sample.advantage)
                    loss = loss - weight * log_probabilities[index]
                elif batch.algorithm is TrainingAlgorithm.PAIRWISE_PREFERENCE:
                    if sample.chosen_action is None or sample.rejected_action is None:
                        raise ValueError("pairwise sample lacks chosen/rejected provenance")
                    if (
                        sample.chosen_action not in base.actions
                        or sample.rejected_action not in base.actions
                    ):
                        raise ValueError("pairwise sample references an unknown action")
                    chosen = base.actions.index(sample.chosen_action)
                    rejected = base.actions.index(sample.rejected_action)
                    loss = (
                        loss
                        - sample.token_weight
                        * sample.reward_margin
                        * torch.nn.functional.logsigmoid(
                            (logits[chosen] - logits[rejected]) / base.temperature
                        )
                    )
                else:
                    ratio = torch.exp(log_probabilities[index] - sample.behavior_log_probability)
                    clipped = torch.clamp(ratio, 0.8, 1.2)
                    weight = (
                        sample.token_weight
                        if batch.algorithm is TrainingAlgorithm.BRANCH_RELATIVE
                        else 1.0
                    )
                    loss = loss - weight * torch.minimum(
                        ratio * sample.advantage, clipped * sample.advantage
                    )
            loss = loss / len(accepted)
            loss = loss + 0.02 * torch.sum(
                reference * (torch.log(reference) - torch.log(probabilities))
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("PyTorch trainer produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((logits,), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            final_loss = float(loss.detach().item())
        candidate_logits = tuple(float(value) for value in logits.detach().cpu().tolist())
        candidate = DeterministicPolicy(
            policy_epoch_id=candidate_policy_epoch_id,
            actions=base.actions,
            logits=candidate_logits,
            temperature=base.temperature,
        )
        buffer = io.BytesIO()
        provenance = {
            "schema_version": "sloforge.helix.pytorch-checkpoint/v2",
            "tenant_id": batch.tenant_id,
            "batch_id": batch.batch_id,
            "batch_data_hash": batch.data_hash,
            "algorithm": batch.algorithm.value,
            "base_policy_epoch_id": base.policy_epoch_id,
            "base_weights_hash": base.weights_hash,
            "candidate_policy_epoch_id": candidate_policy_epoch_id,
            "seed": seed,
            "steps": steps,
            "learning_rate": learning_rate,
            "temperature": base.temperature,
        }
        torch.save({"provenance": provenance, "logits": logits.detach().cpu()}, buffer)
        payload = buffer.getvalue()
        digest = sha256(payload).hexdigest()
        loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if loaded.get("provenance") != provenance:
            raise ValueError("PyTorch checkpoint provenance validation failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staging_name = tempfile.mkstemp(prefix=".helix-pytorch-", dir=output.parent)
        os.close(descriptor)
        staging = Path(staging_name)
        try:
            with staging.open("wb") as checkpoint:
                checkpoint.write(payload)
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
            if sha256(staging.read_bytes()).hexdigest() != digest:
                raise OSError("PyTorch checkpoint failed post-write integrity verification")
            os.link(staging, output)
        finally:
            staging.unlink(missing_ok=True)
        return OptionalTrainingResult(
            adapter="pytorch",
            base_policy_epoch_id=base.policy_epoch_id,
            candidate_policy_epoch_id=candidate_policy_epoch_id,
            algorithm=batch.algorithm,
            batch_id=batch.batch_id,
            seed=seed,
            steps=steps,
            checkpoint_path=output.as_posix(),
            checkpoint_sha256=digest,
            framework_version=str(torch.__version__),
            final_loss=final_loss,
            candidate=candidate,
            validation_class="local-cpu-framework",
        )


def _directory_digest(root: Path) -> str:
    identity: list[dict[str, str | int]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        identity.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not identity:
        raise ValueError("PEFT did not publish any adapter files")
    return sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validation_class_for_device(
    device_type: str,
) -> Literal["hardware-backed", "local-cpu-framework"]:
    return "hardware-backed" if device_type == "cuda" else "local-cpu-framework"


class PeftLoraTrainerAdapter:
    """Local-only LoRA SFT path using the official PEFT adapter APIs."""

    def train(self, request: PeftTrainingRequest, *, output: Path) -> OptionalTrainingResult:
        capability = probe_peft_lora()
        if not capability.available:
            raise RuntimeError(f"PEFT trainer unavailable: {capability.reason}")
        if not request.controlled_environment:
            raise PermissionError("PEFT execution requires an explicitly controlled environment")
        if output.exists():
            raise FileExistsError("PEFT candidate output already exists")
        torch = _require_pytorch()
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
        torch.manual_seed(request.seed)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(request.model_directory), local_files_only=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(request.model_directory), local_files_only=True
        )
        config = peft.LoraConfig(
            r=request.lora_rank,
            lora_alpha=request.lora_rank,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = peft.get_peft_model(model, config)
        model_device_type = str(next(model.parameters()).device.type)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
        by_id = {item.sample_id: item for item in request.examples}
        ordered = tuple(by_id[sample_id] for sample_id in request.batch.training_sample_ids)
        final_loss = math.nan
        for step in range(request.steps):
            example = ordered[step % len(ordered)]
            prompt_ids = tokenizer(
                example.prompt,
                add_special_tokens=True,
            )["input_ids"]
            completion_ids = tokenizer(
                example.completion,
                add_special_tokens=False,
            )["input_ids"]
            eos_token_id = tokenizer.eos_token_id
            if eos_token_id is not None and (
                not completion_ids or completion_ids[-1] != eos_token_id
            ):
                completion_ids.append(eos_token_id)
            if not completion_ids:
                raise ValueError("PEFT completion produced no trainable tokens")
            prompt_ids = prompt_ids[: request.maximum_sequence_tokens - 1]
            completion_ids = completion_ids[: request.maximum_sequence_tokens - len(prompt_ids)]
            input_ids = torch.tensor((prompt_ids + completion_ids,), dtype=torch.long)
            encoded = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
            labels = input_ids.clone()
            labels[:, : len(prompt_ids)] = -100
            optimizer.zero_grad(set_to_none=True)
            loss = model(**encoded, labels=labels).loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("PEFT trainer produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".helix-peft-", dir=output.parent))
        try:
            model.save_pretrained(staging, safe_serialization=True)
            checkpoint_provenance = {
                "schema_version": "sloforge.helix.peft-checkpoint/v1",
                "tenant_id": request.batch.tenant_id,
                "batch_id": request.batch.batch_id,
                "batch_data_hash": request.batch.data_hash,
                "algorithm": request.algorithm.value,
                "base_policy_epoch_id": request.base_policy_epoch_id,
                "candidate_policy_epoch_id": request.candidate_policy_epoch_id,
                "seed": request.seed,
                "steps": request.steps,
                "learning_rate": request.learning_rate,
                "lora_rank": request.lora_rank,
                "maximum_sequence_tokens": request.maximum_sequence_tokens,
            }
            (staging / "helix_provenance.json").write_text(
                json.dumps(
                    checkpoint_provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            digest = _directory_digest(staging)
            os.replace(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return OptionalTrainingResult(
            adapter="peft_lora",
            base_policy_epoch_id=request.base_policy_epoch_id,
            candidate_policy_epoch_id=request.candidate_policy_epoch_id,
            algorithm=request.algorithm,
            batch_id=request.batch.batch_id,
            seed=request.seed,
            steps=request.steps,
            checkpoint_path=output.as_posix(),
            checkpoint_sha256=digest,
            framework_version=capability.version or "unknown",
            final_loss=final_loss,
            candidate=None,
            validation_class=_validation_class_for_device(model_device_type),
        )


__all__ = [
    "AdapterCapability",
    "OptionalTrainingResult",
    "PeftLoraTrainerAdapter",
    "PeftTrainingExample",
    "PeftTrainingRequest",
    "PyTorchTinyTrainerAdapter",
    "probe_optional_pytorch",
    "probe_peft_lora",
]
