"""Sportmonks Football API integration."""

from typing import Any

__all__ = ["SportmonksAPIError", "SportmonksClient"]


def __getattr__(name: str) -> Any:
    # Keep data-normalisation utilities importable without initializing the
    # HTTP stack (useful for offline processing and unit tests).
    if name in __all__:
        from .client import SportmonksAPIError, SportmonksClient

        return {
            "SportmonksAPIError": SportmonksAPIError,
            "SportmonksClient": SportmonksClient,
        }[name]
    raise AttributeError(name)
