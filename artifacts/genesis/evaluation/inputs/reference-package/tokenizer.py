"""Synthetic deterministic token interface for the HybridDecoder fixture."""

from __future__ import annotations


def encode(text: str) -> list[int]:
    if not text:
        raise ValueError("text cannot be empty")
    return [(ord(character) % 31) + 1 for character in text]


def decode_token(token_id: int) -> str:
    if not 0 <= token_id < 32:
        raise ValueError("token identifier is outside the synthetic vocabulary")
    return chr(96 + token_id) if token_id else "_"
