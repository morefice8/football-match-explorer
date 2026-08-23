"""Project logging configuration for normal Match Analysis runtime.

Normal execution defaults to WARNING so routine callbacks and intermediate
DataFrames do not flood the terminal. Set MATCH_ANALYSIS_LOG_LEVEL to INFO or
DEBUG when diagnostics are needed.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


DEFAULT_LOG_LEVEL = "WARNING"
LOG_LEVEL_ENV = "MATCH_ANALYSIS_LOG_LEVEL"
_HANDLER_MARKER = "_match_analysis_handler"


def _resolve_level(level: Optional[str | int]) -> int:
    if level is None:
        level = os.getenv(
            LOG_LEVEL_ENV,
            DEFAULT_LOG_LEVEL,
        )

    if isinstance(level, int):
        return level

    level_name = str(level).strip().upper()

    resolved = logging.getLevelName(
        level_name
    )

    if not isinstance(resolved, int):
        return logging.WARNING

    return resolved


def configure_logging(
    level: Optional[str | int] = None,
) -> int:
    """Configure project loggers without changing third-party logging."""
    resolved_level = _resolve_level(
        level
    )

    formatter = logging.Formatter(
        "%(levelname)s %(name)s: %(message)s"
    )

    for logger_name in (
        "__main__",
        "app",
        "src",
    ):
        project_logger = logging.getLogger(
            logger_name
        )
        project_logger.setLevel(
            resolved_level
        )
        project_logger.propagate = False

        handler = next(
            (
                existing
                for existing
                in project_logger.handlers
                if getattr(
                    existing,
                    _HANDLER_MARKER,
                    False,
                )
            ),
            None,
        )

        if handler is None:
            handler = logging.StreamHandler()
            setattr(
                handler,
                _HANDLER_MARKER,
                True,
            )
            project_logger.addHandler(
                handler
            )

        handler.setLevel(
            resolved_level
        )
        handler.setFormatter(
            formatter
        )

    return resolved_level
