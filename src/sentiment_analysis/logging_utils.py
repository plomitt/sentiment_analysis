import logging


def setup_logging(name: str | None = None, level: int = logging.INFO, format_string: str | None = None) -> logging.Logger:
    """
    Standardized logging setup with optional custom format.

    Args:
        name: Logger name (defaults to __name__ of calling module).
        level: Logging level (defaults to INFO).
        format_string: Custom format string (defaults to standard format).

    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(message)s"

    # Only configure basic logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=format_string)

    logger_name = name if name else __name__
    return logging.getLogger(logger_name)

__all__ = ["setup_logging"]
