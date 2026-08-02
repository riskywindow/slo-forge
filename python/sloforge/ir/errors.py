"""IR-specific error types."""


class IRValidationError(ValueError):
    """Raised when a document cannot be migrated or parsed as canonical IR."""
