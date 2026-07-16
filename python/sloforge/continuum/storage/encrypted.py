"""Authenticated encryption wrapper for tenant-scoped state chunks.

The wrapper intentionally keeps encryption keys out of content-store metadata.  It
stores ciphertext in an ordinary :class:`ContentStore` and returns a separate,
authenticated reference containing only a key identifier and nonce.  Importing
this module does not require ``cryptography``; constructing the wrapper fails
closed when the optional dependency is unavailable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Mapping
from typing import Annotated, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sloforge.continuum.storage.content_store import ChunkRef, ContentStore

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
HexNonce = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{24}$")]
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_AES_GCM_TAG_BYTES = 16
_MAX_PLAINTEXT_BYTES = _MAX_CHUNK_BYTES - _AES_GCM_TAG_BYTES
_AAD_SCHEMA: Literal["sloforge.continuum.encrypted-chunk/v1"] = (
    "sloforge.continuum.encrypted-chunk/v1"
)


class EncryptedStoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class CryptoUnavailable(RuntimeError):
    """The requested authenticated-encryption implementation is unavailable."""


class KeyUnavailable(KeyError):
    """An encryption key identifier cannot be resolved in the current trust domain."""


class AuthenticationFailed(PermissionError):
    """Ciphertext, metadata, or authorization context failed authentication."""


class ReplayRejected(PermissionError):
    """A stale encrypted state reference was rejected by caller policy."""


class EncryptionKeyProvider(Protocol):
    """Resolve key material without serializing it into state metadata."""

    def resolve(self, key_id: str) -> bytes: ...


class _AuthenticatedCipher(Protocol):
    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes: ...

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes | None) -> bytes: ...


class _AuthenticatedCipherFactory(Protocol):
    def __call__(self, key: bytes) -> _AuthenticatedCipher: ...


class StaticEncryptionKeyProvider:
    """Small local key provider for tests and offline deployments.

    Production deployments should implement :class:`EncryptionKeyProvider` with a
    KMS or secret manager.  The mapping is copied so later caller mutation cannot
    silently rotate keys.
    """

    def __init__(self, keys: Mapping[str, bytes]) -> None:
        if not keys:
            raise ValueError("at least one encryption key is required")
        copied: dict[str, bytes] = {}
        for key_id, key in keys.items():
            if not key_id or len(key_id) > 256:
                raise ValueError("key identifiers must contain 1..256 characters")
            if len(key) != 32:
                raise ValueError("AES-256-GCM keys must contain exactly 32 bytes")
            copied[key_id] = bytes(key)
        self._keys = copied

    def resolve(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError as exc:
            raise KeyUnavailable("state encryption key is unavailable") from exc


class EncryptedChunkRef(EncryptedStoreModel):
    """Portable metadata for one authenticated ciphertext chunk.

    ``plaintext_digest`` is an integrity identity, not an authorization token.  It
    is authenticated as AAD together with tenant, capsule, state version, and key
    identity, preventing cross-context chunk substitution.
    """

    schema_version: Literal["sloforge.continuum.encrypted-chunk/v1"] = _AAD_SCHEMA
    tenant_id: NonEmpty
    capsule_id: NonEmpty
    state_version: Annotated[int, Field(ge=0)]
    key_id: NonEmpty
    algorithm: Literal["AES-256-GCM"] = "AES-256-GCM"
    nonce_hex: HexNonce
    plaintext_digest: Sha256
    plaintext_size: Annotated[int, Field(ge=0, le=_MAX_PLAINTEXT_BYTES)]
    ciphertext: ChunkRef

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.ciphertext.tenant_id != self.tenant_id:
            raise ValueError("ciphertext reference crosses a tenant boundary")
        if self.ciphertext.compression != "none":
            raise ValueError("encrypted ciphertext must not be store-compressed")
        return self


class EncryptedReadAuthorization(EncryptedStoreModel):
    """Replay and authorization policy supplied by the transaction layer."""

    tenant_id: NonEmpty
    capsule_id: NonEmpty
    minimum_state_version: Annotated[int, Field(ge=0)]
    allowed_key_ids: Annotated[tuple[NonEmpty, ...], Field(min_length=1, max_length=64)]


def cryptography_available() -> bool:
    """Return whether the optional AEAD dependency can be imported."""

    try:
        return importlib.util.find_spec("cryptography.hazmat.primitives.ciphers.aead") is not None
    except (ImportError, AttributeError):
        return False


def _load_aesgcm() -> _AuthenticatedCipherFactory:
    if not cryptography_available():
        raise CryptoUnavailable(
            "authenticated state encryption requires the optional 'cryptography' package"
        )
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - protects optional installations
        raise CryptoUnavailable("AES-256-GCM support is unavailable") from exc
    return cast(_AuthenticatedCipherFactory, AESGCM)


def _aad(
    *,
    tenant_id: str,
    capsule_id: str,
    state_version: int,
    key_id: str,
    plaintext_digest: str,
    plaintext_size: int,
) -> bytes:
    value = {
        "capsule_id": capsule_id,
        "key_id": key_id,
        "plaintext_digest": plaintext_digest,
        "plaintext_size": plaintext_size,
        "schema": _AAD_SCHEMA,
        "state_version": state_version,
        "tenant_id": tenant_id,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class AuthenticatedEncryptedStore:
    """Encrypt state chunks before storing them in a tenant-scoped content store.

    AES-GCM nonces are generated with the operating system CSPRNG.  Keys are
    obtained on every operation from a separate provider and never placed in a
    :class:`ChunkRef`, exception, event, or log message.
    """

    def __init__(self, underlying: ContentStore, key_provider: EncryptionKeyProvider) -> None:
        if not cryptography_available():
            raise CryptoUnavailable(
                "authenticated state encryption requires the optional 'cryptography' package"
            )
        self._underlying = underlying
        self._key_provider = key_provider
        self._aesgcm_factory = _load_aesgcm()

    def put(
        self,
        *,
        tenant_id: str,
        capsule_id: str,
        state_version: int,
        key_id: str,
        plaintext: bytes,
        expires_at_ms: int | None = None,
    ) -> EncryptedChunkRef:
        if not tenant_id or not capsule_id or not key_id:
            raise ValueError("tenant, capsule, and key identifiers are required")
        if state_version < 0:
            raise ValueError("state version must be non-negative")
        if len(plaintext) > _MAX_PLAINTEXT_BYTES:
            raise ValueError(
                f"encrypted state chunk exceeds {_MAX_PLAINTEXT_BYTES} plaintext bytes"
            )
        key = self._key_provider.resolve(key_id)
        if len(key) != 32:
            raise KeyUnavailable("resolved AES-256-GCM key has an invalid length")
        digest = hashlib.sha256(plaintext).hexdigest()
        associated_data = _aad(
            tenant_id=tenant_id,
            capsule_id=capsule_id,
            state_version=state_version,
            key_id=key_id,
            plaintext_digest=digest,
            plaintext_size=len(plaintext),
        )
        nonce = os.urandom(12)
        aesgcm = self._aesgcm_factory(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        stored = self._underlying.put(
            tenant_id,
            ciphertext,
            compression="none",
            expires_at_ms=expires_at_ms,
        )
        return EncryptedChunkRef(
            tenant_id=tenant_id,
            capsule_id=capsule_id,
            state_version=state_version,
            key_id=key_id,
            nonce_hex=nonce.hex(),
            plaintext_digest=digest,
            plaintext_size=len(plaintext),
            ciphertext=stored,
        )

    def read(
        self,
        reference: EncryptedChunkRef,
        *,
        authorization: EncryptedReadAuthorization,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        if reference.tenant_id != authorization.tenant_id:
            raise AuthenticationFailed("encrypted state tenant authorization failed")
        if reference.capsule_id != authorization.capsule_id:
            raise AuthenticationFailed("encrypted state capsule authorization failed")
        if reference.state_version < authorization.minimum_state_version:
            raise ReplayRejected("encrypted state version is older than the accepted watermark")
        if reference.key_id not in authorization.allowed_key_ids:
            raise AuthenticationFailed("encrypted state key is outside the authorization policy")
        if offset < 0 or offset > reference.plaintext_size:
            raise ValueError("chunk read offset is outside the plaintext chunk")
        if length is not None and length < 0:
            raise ValueError("chunk read length must be non-negative")

        key = self._key_provider.resolve(reference.key_id)
        if len(key) != 32:
            raise KeyUnavailable("resolved AES-256-GCM key has an invalid length")
        associated_data = _aad(
            tenant_id=reference.tenant_id,
            capsule_id=reference.capsule_id,
            state_version=reference.state_version,
            key_id=reference.key_id,
            plaintext_digest=reference.plaintext_digest,
            plaintext_size=reference.plaintext_size,
        )
        ciphertext = self._underlying.read(reference.tenant_id, reference.ciphertext)
        aesgcm = self._aesgcm_factory(key)
        try:
            plaintext = aesgcm.decrypt(
                bytes.fromhex(reference.nonce_hex), ciphertext, associated_data
            )
        except Exception as exc:
            # Do not include metadata, ciphertext, plaintext, or key material in errors.
            if exc.__class__.__name__ != "InvalidTag":
                raise
            raise AuthenticationFailed("encrypted state authentication failed") from exc
        if len(plaintext) != reference.plaintext_size:
            raise AuthenticationFailed("encrypted state size authentication failed")
        if hashlib.sha256(plaintext).hexdigest() != reference.plaintext_digest:
            raise AuthenticationFailed("encrypted state digest authentication failed")
        return plaintext[offset:] if length is None else plaintext[offset : offset + length]
