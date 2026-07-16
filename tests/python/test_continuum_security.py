from __future__ import annotations

import hashlib

import pytest

from sloforge.continuum.storage import (
    AuthenticatedEncryptedStore,
    AuthenticationFailed,
    CryptoUnavailable,
    EncryptedReadAuthorization,
    KeyUnavailable,
    MemoryContentStore,
    ReplayRejected,
    StaticEncryptionKeyProvider,
    cryptography_available,
)
from sloforge.continuum.storage import encrypted as encrypted_module

requires_cryptography = pytest.mark.skipif(
    not cryptography_available(), reason="optional cryptography package is unavailable"
)


def _store() -> tuple[MemoryContentStore, AuthenticatedEncryptedStore]:
    underlying = MemoryContentStore()
    keys = StaticEncryptionKeyProvider(
        {
            "key-2026-a": bytes(range(32)),
            "key-2026-b": bytes(reversed(range(32))),
        }
    )
    return underlying, AuthenticatedEncryptedStore(underlying, keys)


def _authorization(*, state_version: int = 4) -> EncryptedReadAuthorization:
    return EncryptedReadAuthorization(
        tenant_id="tenant-a",
        capsule_id="capsule-001",
        minimum_state_version=state_version,
        allowed_key_ids=("key-2026-a",),
    )


@requires_cryptography
def test_authenticated_encryption_separates_keys_and_binds_context() -> None:
    underlying, store = _store()
    plaintext = b"private recurrent and attention state" * 8
    reference = store.put(
        tenant_id="tenant-a",
        capsule_id="capsule-001",
        state_version=4,
        key_id="key-2026-a",
        plaintext=plaintext,
    )

    ciphertext = underlying.read("tenant-a", reference.ciphertext)
    assert ciphertext != plaintext
    assert plaintext not in ciphertext
    assert "key-2026-a" in reference.model_dump_json()
    assert bytes(range(32)).hex() not in reference.model_dump_json()
    assert store.read(reference, authorization=_authorization()) == plaintext
    assert (
        store.read(reference, authorization=_authorization(), offset=8, length=11)
        == plaintext[8:19]
    )


@requires_cryptography
def test_authenticated_encryption_rejects_tamper_substitution_and_stale_replay() -> None:
    _underlying, store = _store()
    reference = store.put(
        tenant_id="tenant-a",
        capsule_id="capsule-001",
        state_version=4,
        key_id="key-2026-a",
        plaintext=b"state-v4",
    )

    tampered = reference.model_copy(
        update={"plaintext_digest": hashlib.sha256(b"substitute").hexdigest()}
    )
    with pytest.raises(AuthenticationFailed, match="authentication failed"):
        store.read(tampered, authorization=_authorization())

    other_reference = store.put(
        tenant_id="tenant-a",
        capsule_id="capsule-001",
        state_version=4,
        key_id="key-2026-a",
        plaintext=b"different-state-v4",
    )
    substituted = reference.model_copy(update={"ciphertext": other_reference.ciphertext})
    with pytest.raises(AuthenticationFailed, match="authentication failed"):
        store.read(substituted, authorization=_authorization())

    wrong_capsule = _authorization().model_copy(update={"capsule_id": "capsule-other"})
    with pytest.raises(AuthenticationFailed, match="capsule authorization"):
        store.read(reference, authorization=wrong_capsule)

    stale_policy = _authorization(state_version=5)
    with pytest.raises(ReplayRejected, match="older than the accepted watermark"):
        store.read(reference, authorization=stale_policy)

    wrong_tenant = _authorization().model_copy(update={"tenant_id": "tenant-b"})
    with pytest.raises(AuthenticationFailed, match="tenant authorization"):
        store.read(reference, authorization=wrong_tenant)


@requires_cryptography
def test_authenticated_encryption_fails_closed_for_missing_key() -> None:
    underlying, store = _store()
    reference = store.put(
        tenant_id="tenant-a",
        capsule_id="capsule-001",
        state_version=4,
        key_id="key-2026-a",
        plaintext=b"state",
    )
    missing_key_store = AuthenticatedEncryptedStore(
        underlying, StaticEncryptionKeyProvider({"different-key": b"x" * 32})
    )
    with pytest.raises(KeyUnavailable, match="unavailable"):
        missing_key_store.read(reference, authorization=_authorization())


def test_authenticated_encryption_fails_closed_when_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(encrypted_module.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(CryptoUnavailable, match="optional 'cryptography'"):
        AuthenticatedEncryptedStore(
            MemoryContentStore(), StaticEncryptionKeyProvider({"key": b"y" * 32})
        )
