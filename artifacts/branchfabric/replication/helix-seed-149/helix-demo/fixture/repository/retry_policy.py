def retry_delay(status: int, retry_after: str | None) -> int:
    """Return seconds before retrying a response."""
    if status >= 500:
        return 2
    return 0
