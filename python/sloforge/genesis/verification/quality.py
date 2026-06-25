"""Quality-contract evaluation kept separate from synthesis scoring."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from .model import EvidenceStatus, QualityContract, QualityEvidence, VerificationError

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _probabilities(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return cast(FloatArray, exponent / np.sum(exponent, axis=-1, keepdims=True))


def evaluate_quality(
    reference_logits: FloatArray,
    candidate_logits: FloatArray,
    reference_tokens: IntArray,
    candidate_tokens: IntArray,
    contract: QualityContract,
) -> QualityEvidence:
    if reference_logits.shape != candidate_logits.shape or reference_logits.ndim != 2:
        raise VerificationError("quality logits must be matching [samples, vocabulary] arrays")
    if reference_tokens.shape != candidate_tokens.shape or reference_tokens.ndim != 1:
        raise VerificationError("quality token arrays must be matching vectors")
    if reference_logits.shape[0] != reference_tokens.shape[0] or not reference_tokens.size:
        raise VerificationError("quality dataset must be non-empty and aligned")
    if not np.issubdtype(reference_logits.dtype, np.floating) or not np.issubdtype(
        candidate_logits.dtype, np.floating
    ):
        raise VerificationError("quality logits must use floating-point dtypes")
    if not np.all(np.isfinite(reference_logits)) or not np.all(np.isfinite(candidate_logits)):
        raise VerificationError("quality logits must be finite")
    if not np.issubdtype(reference_tokens.dtype, np.integer) or not np.issubdtype(
        candidate_tokens.dtype, np.integer
    ):
        raise VerificationError("quality tokens must use integer dtypes")
    vocabulary = reference_logits.shape[1]
    if (
        np.any(reference_tokens < 0)
        or np.any(reference_tokens >= vocabulary)
        or np.any(candidate_tokens < 0)
        or np.any(candidate_tokens >= vocabulary)
    ):
        raise VerificationError("quality tokens must lie inside the logits vocabulary")
    agreement_thresholds = (
        contract.exact_token_match_minimum,
        contract.top1_agreement_minimum,
        contract.topk_agreement_minimum,
    )
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in agreement_thresholds):
        raise VerificationError("quality agreement thresholds must be finite probabilities")
    divergence_bounds = (
        contract.maximum_kl_divergence,
        contract.maximum_js_divergence,
        contract.maximum_absolute_error,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in divergence_bounds):
        raise VerificationError("quality error bounds must be finite and non-negative")
    if not 1 <= contract.topk <= reference_logits.shape[1]:
        raise VerificationError("top-k quality domain is invalid")
    reference_probability = _probabilities(reference_logits)
    candidate_probability = _probabilities(candidate_logits)
    epsilon = np.finfo(np.float64).tiny
    kl = np.mean(
        np.sum(
            reference_probability
            * np.log((reference_probability + epsilon) / (candidate_probability + epsilon)),
            axis=-1,
        )
    )
    midpoint = (reference_probability + candidate_probability) / 2
    js = 0.5 * np.mean(
        np.sum(
            reference_probability
            * np.log((reference_probability + epsilon) / (midpoint + epsilon)),
            axis=-1,
        )
        + np.sum(
            candidate_probability
            * np.log((candidate_probability + epsilon) / (midpoint + epsilon)),
            axis=-1,
        )
    )
    reference_top1 = np.argmax(reference_logits, axis=-1)
    candidate_top1 = np.argmax(candidate_logits, axis=-1)
    reference_topk = np.argpartition(reference_logits, -contract.topk, axis=-1)[:, -contract.topk :]
    candidate_topk = np.argpartition(candidate_logits, -contract.topk, axis=-1)[:, -contract.topk :]
    topk_agreement = np.mean(
        [
            len(set(reference_row.tolist()) & set(candidate_row.tolist())) / contract.topk
            for reference_row, candidate_row in zip(reference_topk, candidate_topk, strict=True)
        ]
    )
    exact_tokens = float(np.mean(reference_tokens == candidate_tokens))
    top1 = float(np.mean(reference_top1 == candidate_top1))
    maximum_error = float(np.max(np.abs(reference_logits - candidate_logits)))
    violations: list[str] = []
    if exact_tokens < contract.exact_token_match_minimum:
        violations.append("exact_token_match")
    if top1 < contract.top1_agreement_minimum:
        violations.append("top1_agreement")
    if topk_agreement < contract.topk_agreement_minimum:
        violations.append("topk_agreement")
    if kl > contract.maximum_kl_divergence:
        violations.append("kl_divergence")
    if js > contract.maximum_js_divergence:
        violations.append("js_divergence")
    if maximum_error > contract.maximum_absolute_error:
        violations.append("maximum_absolute_error")
    return QualityEvidence(
        status=EvidenceStatus.FAILED if violations else EvidenceStatus.PASSED,
        sample_count=reference_tokens.size,
        exact_token_match=exact_tokens,
        top1_agreement=top1,
        topk_agreement=float(topk_agreement),
        kl_divergence=float(kl),
        js_divergence=float(js),
        maximum_absolute_error=maximum_error,
        violations=tuple(violations),
    )
