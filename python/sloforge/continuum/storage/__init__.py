"""Tenant-scoped content-addressed state storage."""

from .content_store import (
    ChunkCorrupt,
    ChunkMissing,
    ChunkRef,
    ContentStore,
    FileContentStore,
    MemoryContentStore,
    StoredManifest,
)
from .encrypted import (
    AuthenticatedEncryptedStore,
    AuthenticationFailed,
    CryptoUnavailable,
    EncryptedChunkRef,
    EncryptedReadAuthorization,
    EncryptionKeyProvider,
    KeyUnavailable,
    ReplayRejected,
    StaticEncryptionKeyProvider,
    cryptography_available,
)

__all__ = [
    "AuthenticatedEncryptedStore",
    "AuthenticationFailed",
    "ChunkCorrupt",
    "ChunkMissing",
    "ChunkRef",
    "ContentStore",
    "CryptoUnavailable",
    "EncryptedChunkRef",
    "EncryptedReadAuthorization",
    "EncryptionKeyProvider",
    "FileContentStore",
    "KeyUnavailable",
    "MemoryContentStore",
    "ReplayRejected",
    "StaticEncryptionKeyProvider",
    "StoredManifest",
    "cryptography_available",
]
