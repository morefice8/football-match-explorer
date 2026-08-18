"""HTTP client for Sportmonks Football API v3."""

from __future__ import annotations

from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api.sportmonks.com/v3/football"


class SportmonksAPIError(RuntimeError):
    def __init__(self, status_code: int | None, message: str):
        self.status_code = status_code
        prefix = f"HTTP {status_code}" if status_code is not None else "Network error"
        super().__init__(f"{prefix}: {message}")


class SportmonksClient:
    """Small authenticated client with retry and pagination support.

    Authentication uses the Authorization header so the token does not appear
    in request URLs, logs, exceptions, or cached JSON files.
    """

    def __init__(self, token: str, timeout: int = 45):
        token = token.strip()
        if not token:
            raise ValueError("Sportmonks API token is empty")

        self._token = token
        self._timeout = timeout
        self._session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.headers.update(
            {
                "Accept": "application/json",
                # Sportmonks API v3 expects the persistent API token as the
                # raw Authorization header value (not an OAuth Bearer token).
                "Authorization": token,
                "User-Agent": "football-match-explorer/1.0",
            }
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "SportmonksClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _sanitise(self, value: Any) -> str:
        return str(value).replace(self._token, "***")

    @staticmethod
    def _error_message(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in ("message", "error"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"]
        return None

    def get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        query = {key: value for key, value in params.items() if value is not None}
        try:
            response = self._session.get(url, params=query, timeout=self._timeout)
        except requests.RequestException as exc:
            raise SportmonksAPIError(None, self._sanitise(exc)) from None

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if not response.ok:
            message = self._error_message(payload) or response.reason or "request failed"
            raise SportmonksAPIError(response.status_code, self._sanitise(message))
        if not isinstance(payload, dict):
            raise SportmonksAPIError(response.status_code, "unexpected JSON response")
        return payload

    def iter_pages(
        self,
        endpoint: str,
        *,
        per_page: int = 50,
        max_pages: int = 100,
        **params: Any,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        page = 1
        while page <= max_pages:
            payload = self.get(endpoint, page=page, per_page=per_page, **params)
            yield page, payload

            pagination = payload.get("pagination")
            if not isinstance(pagination, dict):
                meta = payload.get("meta")
                pagination = meta.get("pagination", {}) if isinstance(meta, dict) else {}

            next_page = pagination.get("next_page")
            has_more = bool(pagination.get("has_more"))
            if not next_page and not has_more:
                return
            page = next_page if isinstance(next_page, int) else page + 1

        raise SportmonksAPIError(None, f"pagination exceeded {max_pages} pages for {endpoint}")
