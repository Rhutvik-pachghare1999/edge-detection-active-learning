"""Structured logging setup for the benchmark."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def configure_logging(level: str, log_file: str, fmt: str, console: bool = True) -> None:
    """Configure loguru with a file sink and optional console sink."""
    logger.remove()
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=level,
        format=fmt,
        rotation="10 MB",
        retention=5,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    if console:
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            colorize=True,
        )
