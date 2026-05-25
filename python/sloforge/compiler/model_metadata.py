from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from sloforge.ir import LicenseMetadata, ModelArchitecture


class ModelProfileMetadata(BaseModel):
    """Validated model identity carried from profiling into plan compilation."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_is_mock: bool
    architecture: ModelArchitecture
    license: LicenseMetadata
    resolved_snapshot: str | None = None
    maximum_sequence_length: int = Field(ge=128)
    dtype: str | None = None


def mock_qwen3_metadata(
    *,
    requested_model: str,
    parameter_count: int = 600_000_000,
    maximum_sequence_length: int = 32_768,
) -> ModelProfileMetadata:
    return ModelProfileMetadata(
        model_id="sloforge/mock-qwen3-0.6b-shape",
        requested_model=requested_model,
        revision="cpu-demo-v1",
        checksum_sha256=hashlib.sha256(
            b"sloforge explicit mock with Qwen3-0.6B architecture shape v1"
        ).hexdigest(),
        model_is_mock=True,
        architecture=ModelArchitecture(
            family="qwen3-compatible-mock",
            parameter_count=parameter_count,
            hidden_size=1024,
            num_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            vocabulary_size=151_936,
        ),
        license=LicenseMetadata(
            spdx_id="Apache-2.0",
            name="Apache License 2.0",
            url="https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE",
            redistribution_allowed=True,
            verified_at=datetime.now(UTC),
        ),
        maximum_sequence_length=maximum_sequence_length,
        dtype="float32",
    )
