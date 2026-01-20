"""Central logging configuration."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str = "INFO", logger_name: Optional[str] = None) -> logging.Logger:
    """Configure application-wide logging with a standard formatter."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logging.getLogger().setLevel(level)
    return logger
