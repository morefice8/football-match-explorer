r"""Audit minimo della copertura Sportmonks Football API v3.

Eseguire dalla radice del progetto, dove si trova il file .env:

    python .\sportmonks_audit.py

Il token deve essere salvato nel file .env come:

    SPORTMONKS_API_TOKEN=...

Lo script non stampa e non salva mai il token.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_URL = "https://api.sportmonks.com/v3/football"
TARGET_SEASON_START = 2025
TARGET_SEASON_END = 2026
OUTPUT_DIR = Path("data") / "raw" / "sportmonks" / "audit"


@dataclass(frozen=True)
class TargetLeague:
    label: str
    accepted_names: tuple[str, ...]
    accepted_countries: tuple[str, ...]


TARGET_LEAGUES = (
    TargetLeague("Premier League", ("premier league",), ("england",)),
    TargetLeague("La Liga", ("la liga", "primera division"), ("spain",)),
    TargetLeague("Bundesliga", ("bundesliga",), ("germany",)),
    TargetLeague("Serie A", ("serie a",), ("italy",)),
    TargetLeague("Ligue 1", ("ligue 1",), ("france",)),
)


class SportmonksError(RuntimeError):
    """Errore API privo di token o URL sensibili."""

    def __init__(self, status_code: int | None, message: str):
        self.status_code = status_code
        prefix = f"HTTP {status_code}" if status_code is not None else "Errore di rete"
        super().__init__(f"{prefix}: {message}")


class SportmonksClient:
    def __init__(self, token: str, timeout: int = 30):
        self._token = token
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "football-match-explorer/1.0",
            }
        )

    def close(self) -> None:
        self._session.close()

    def _sanitize(self, value: str) -> str:
        return value.replace(self._token, "***")

    def get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        query = {"api_token": self._token, **params}

        try:
            response = self._session.get(url, params=query, timeout=self._timeout)
        except requests.RequestException as exc:
            raise SportmonksError(None, self._sanitize(str(exc))) from None

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if not response.ok:
            message = _extract_error_message(payload) or response.reason or "richiesta fallita"
            raise SportmonksError(response.status_code, self._sanitize(message))

        if not isinstance(payload, dict):
            raise SportmonksError(response.status_code, "risposta JSON inattesa")

        return payload

    def get_all(self, endpoint: str, **params: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Legge tutte le pagine disponibili, con un limite di sicurezza."""
        items: list[dict[str, Any]] = []
        raw_pages: list[dict[str, Any]] = []
        page = 1

        while page <= 50:
            payload = self.get(endpoint, page=page, per_page=100, **params)
            raw_pages.append(payload)
            page_items = _as_list(payload.get("data"))
            items.extend(item for item in page_items if isinstance(item, dict))

            pagination = payload.get("pagination")
            if not isinstance(pagination, dict):
                meta = payload.get("meta")
                pagination = meta.get("pagination", {}) if isinstance(meta, dict) else {}

            has_more = bool(pagination.get("has_more"))
            next_page = pagination.get("next_page")

            if not has_more and not next_page:
                break

            if isinstance(next_page, int):
                page = next_page
            else:
                page += 1
        else:
            raise SportmonksError(None, "superato il limite di 50 pagine")

        return items, raw_pages


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        nested = value.get("data")
        if isinstance(nested, list):
            return nested
        return [value]
    return []


def _extract_error_message(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    for key in ("message", "error"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = value.get("message")
            if isinstance(nested, str):
                return nested

    return None


def _normalise(value: Any) -> str:
    text = str(value or "").casefold().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _country_name(league: dict[str, Any]) -> str:
    country = league.get("country")
    if isinstance(country, dict):
        country_data = country.get("data", country)
        if isinstance(country_data, dict):
            return str(country_data.get("name", ""))
    return ""


def _find_league(leagues: list[dict[str, Any]], target: TargetLeague) -> dict[str, Any] | None:
    names = {_normalise(name) for name in target.accepted_names}
    countries = {_normalise(country) for country in target.accepted_countries}

    exact_candidates = [league for league in leagues if _normalise(league.get("name")) in names]
    for league in exact_candidates:
        country = _normalise(_country_name(league))
        if not country or country in countries:
            return league

    return exact_candidates[0] if len(exact_candidates) == 1 else None


def _season_candidates(league_payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = league_payload.get("data", {})
    if not isinstance(data, dict):
        return []
    return [item for item in _as_list(data.get("seasons")) if isinstance(item, dict)]


def _find_season(seasons: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted_names = {
        "2025 2026",
        "2025 26",
        "2025",
    }

    for season in seasons:
        if _normalise(season.get("name")) in accepted_names:
            return season

    for season in seasons:
        start = str(season.get("starting_at", ""))
        end = str(season.get("ending_at", ""))
        if start.startswith(str(TARGET_SEASON_START)) and end.startswith(str(TARGET_SEASON_END)):
            return season

    return None


def _save_json(filename: str, payload: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _test_endpoint(client: SportmonksClient, endpoint: str) -> dict[str, Any]:
    try:
        payload = client.get(endpoint)
        data = _as_list(payload.get("data"))
        return {
            "ok": True,
            "records": len(data),
            "payload": payload,
        }
    except SportmonksError as exc:
        return {
            "ok": False,
            "status_code": exc.status_code,
            "error": str(exc),
        }


def _print_header() -> None:
    print("\nSPORTMONKS COVERAGE AUDIT")
    print(f"Stagione richiesta: {TARGET_SEASON_START}-{TARGET_SEASON_END}")
    print("-" * 68)


def _print_result(label: str, result: dict[str, Any]) -> None:
    if result.get("ok"):
        print(f"    [OK] {label}: {result.get('records', 0)} record")
    else:
        print(f"    [NO] {label}: {result.get('error', 'errore sconosciuto')}")


def run_audit() -> int:
    load_dotenv()
    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()

    if not token:
        print("ERRORE: SPORTMONKS_API_TOKEN non trovato nel file .env.", file=sys.stderr)
        return 2

    _print_header()
    client = SportmonksClient(token)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": f"{TARGET_SEASON_START}-{TARGET_SEASON_END}",
        "leagues": [],
    }

    try:
        print("[1/3] Verifica autenticazione e catalogo leghe...")
        leagues, league_pages = client.get_all("leagues", include="country")
        _save_json("leagues_raw.json", league_pages)
        report["accessible_leagues_count"] = len(leagues)
        print(f"      Token valido. Leghe visibili: {len(leagues)}")

        print("[2/3] Ricerca delle cinque leghe e della stagione...")
        for target in TARGET_LEAGUES:
            league = _find_league(leagues, target)
            league_report: dict[str, Any] = {"requested_name": target.label}

            if league is None:
                league_report.update({"league_found": False, "season_found": False})
                report["leagues"].append(league_report)
                print(f"  [NO] {target.label}: non visibile nel piano")
                continue

            league_id = league.get("id")
            league_report.update(
                {
                    "league_found": True,
                    "league_id": league_id,
                    "api_name": league.get("name"),
                    "country": _country_name(league),
                }
            )

            try:
                league_payload = client.get(f"leagues/{league_id}", include="seasons")
                _save_json(f"league_{league_id}_seasons_raw.json", league_payload)
                season = _find_season(_season_candidates(league_payload))
            except SportmonksError as exc:
                league_report.update({"season_found": False, "season_error": str(exc)})
                report["leagues"].append(league_report)
                print(f"  [NO] {target.label}: {exc}")
                continue

            if season is None:
                league_report["season_found"] = False
                report["leagues"].append(league_report)
                print(f"  [NO] {target.label}: stagione 2025-2026 non trovata")
                continue

            season_id = season.get("id")
            league_report.update(
                {
                    "season_found": True,
                    "season_id": season_id,
                    "season_name": season.get("name"),
                }
            )
            print(
                f"  [OK] {target.label}: league_id={league_id}, "
                f"season_id={season_id} ({season.get('name')})"
            )

            standings = _test_endpoint(client, f"standings/seasons/{season_id}")
            teams = _test_endpoint(client, f"teams/seasons/{season_id}")
            league_report["standings"] = {key: value for key, value in standings.items() if key != "payload"}
            league_report["teams"] = {key: value for key, value in teams.items() if key != "payload"}

            if standings.get("ok"):
                _save_json(f"season_{season_id}_standings_raw.json", standings["payload"])
            if teams.get("ok"):
                _save_json(f"season_{season_id}_teams_raw.json", teams["payload"])

            _print_result("classifica", standings)
            _print_result("squadre", teams)
            report["leagues"].append(league_report)

        report["visible_leagues"] = [
            {
                "id": league.get("id"),
                "name": league.get("name"),
                "country": _country_name(league),
            }
            for league in leagues
        ]

        print("[3/3] Salvataggio report...")
        _save_json("coverage_report.json", report)
        print(f"      Report: {OUTPUT_DIR / 'coverage_report.json'}")
        print("\nAudit completato. Il token non e stato scritto nei file.")
        return 0

    except SportmonksError as exc:
        report["fatal_error"] = str(exc)
        _save_json("coverage_report.json", report)
        print(f"\nAUDIT FALLITO: {exc}", file=sys.stderr)
        print("Controlla token, connessione e stato della prova Sportmonks.", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(run_audit())