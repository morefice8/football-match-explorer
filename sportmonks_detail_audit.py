r"""Audit dettagliato della copertura Sportmonks Football API v3.

Prerequisiti:
    1. sportmonks_audit.py nella stessa cartella
    2. .env con SPORTMONKS_API_TOKEN
    3. data/raw/sportmonks/audit/coverage_report.json generato dal primo audit

Esecuzione dalla radice del progetto:

    python .\sportmonks_detail_audit.py

Lo script usa un solo club, un solo giocatore e una sola partita conclusa per
campionato. Non stampa e non salva mai il token.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from sportmonks_audit import SportmonksClient, SportmonksError


BASE_AUDIT_DIR = Path("data") / "raw" / "sportmonks" / "audit"
INPUT_REPORT = BASE_AUDIT_DIR / "coverage_report.json"
OUTPUT_DIR = Path("data") / "raw" / "sportmonks" / "detail_audit"
OUTPUT_REPORT = OUTPUT_DIR / "detail_coverage_report.json"


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        nested = value.get("data")
        if isinstance(nested, list):
            return nested
        return [value]
    return []


def as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get("data")
    if isinstance(nested, dict):
        return nested
    return value


def relation_list(obj: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = obj.get(key)
    return [item for item in as_list(value) if isinstance(item, dict)]


def relation_dict(obj: dict[str, Any], key: str) -> dict[str, Any]:
    return as_dict(obj.get(key))


def find_relation_key(obj: dict[str, Any], wanted: str) -> str | None:
    wanted_folded = wanted.casefold()
    for key in obj:
        if key.casefold() == wanted_folded:
            return key
    return None


def save_json(filename: str, payload: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def safe_slug(value: Any) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(value or "")]
    return "_".join(part for part in "".join(chars).split("_") if part)


def probe(client: SportmonksClient, endpoint: str, **params: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "payload": client.get(endpoint, **params)}
    except SportmonksError as exc:
        return {
            "ok": False,
            "status_code": exc.status_code,
            "error": str(exc),
        }


def finished_fixture(fixture: dict[str, Any]) -> bool:
    state = relation_dict(fixture, "state")
    labels = {
        str(state.get("short_name", "")).casefold(),
        str(state.get("developer_name", "")).casefold(),
        str(state.get("name", "")).casefold(),
    }
    finished_labels = {
        "ft",
        "aet",
        "pen",
        "finished",
        "after extra time",
        "after penalties",
    }
    return bool(labels & finished_labels) or bool(fixture.get("result_info"))


def select_fixture(fixtures: list[dict[str, Any]]) -> dict[str, Any] | None:
    finished = [fixture for fixture in fixtures if finished_fixture(fixture)]
    candidates = finished or fixtures
    if not candidates:
        return None
    return max(candidates, key=lambda item: str(item.get("starting_at", "")))


def type_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    type_obj = relation_dict(record, "type")
    type_id = record.get("type_id") or type_obj.get("id")
    name = (
        type_obj.get("name")
        or type_obj.get("developer_name")
        or type_obj.get("code")
        or f"type_{type_id}"
    )
    return {
        "type_id": type_id,
        "name": name,
        "code": type_obj.get("code"),
        "developer_name": type_obj.get("developer_name"),
    }


def unique_types(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        descriptor = type_descriptor(record)
        key = str(descriptor.get("type_id") or descriptor.get("name"))
        seen[key] = descriptor
    return sorted(seen.values(), key=lambda item: str(item.get("name", "")).casefold())


def season_stat_details(entity_payload: dict[str, Any]) -> list[dict[str, Any]]:
    entity = as_dict(entity_payload.get("data"))
    details: list[dict[str, Any]] = []
    for statistic in relation_list(entity, "statistics"):
        details.extend(relation_list(statistic, "details"))
    return details


def fixture_team_stats(fixture_payload: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = as_dict(fixture_payload.get("data"))
    return relation_list(fixture, "statistics")


def fixture_lineups(fixture_payload: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = as_dict(fixture_payload.get("data"))
    return relation_list(fixture, "lineups")


def lineup_stat_details(lineups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for lineup in lineups:
        details.extend(relation_list(lineup, "details"))
    return details


def event_types(fixture_payload: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = as_dict(fixture_payload.get("data"))
    return unique_types(relation_list(fixture, "events"))


def choose_team(fixture_payload: dict[str, Any]) -> dict[str, Any] | None:
    fixture = as_dict(fixture_payload.get("data"))
    participants = relation_list(fixture, "participants")
    if not participants:
        return None

    for participant in participants:
        meta = participant.get("meta")
        if isinstance(meta, dict) and str(meta.get("location", "")).casefold() == "home":
            return participant
    return participants[0]


def choose_player(
    lineups: list[dict[str, Any]],
    squad_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    for lineup in lineups:
        player = relation_dict(lineup, "player")
        player_id = lineup.get("player_id") or player.get("id")
        if player_id:
            return {
                "id": player_id,
                "name": player.get("display_name") or player.get("common_name") or player.get("name"),
            }

    if squad_payload:
        for squad_member in [item for item in as_list(squad_payload.get("data")) if isinstance(item, dict)]:
            player = relation_dict(squad_member, "player")
            player_id = squad_member.get("player_id") or player.get("id")
            if player_id:
                return {
                    "id": player_id,
                    "name": player.get("display_name") or player.get("common_name") or player.get("name"),
                }
    return None


def xg_summary(payload: dict[str, Any], relationship: str) -> dict[str, Any]:
    fixture = as_dict(payload.get("data"))
    key = find_relation_key(fixture, relationship)
    records = relation_list(fixture, key) if key else []
    return {
        "relationship_returned": key,
        "records": len(records),
        "types": unique_types(records),
    }


def xg_lineup_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fixture = as_dict(payload.get("data"))
    records: list[dict[str, Any]] = []
    relationship_names: set[str] = set()
    for lineup in relation_list(fixture, "lineups"):
        key = find_relation_key(lineup, "xGLineup")
        if key:
            relationship_names.add(key)
            records.extend(relation_list(lineup, key))
    return {
        "relationships_returned": sorted(relationship_names),
        "records": len(records),
        "types": unique_types(records),
    }


def public_probe_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "payload"}


def print_metric(label: str, value: int | str) -> None:
    print(f"    {label:<31} {value}")


def audit_league(
    client: SportmonksClient,
    league: dict[str, Any],
) -> dict[str, Any]:
    league_name = str(league.get("requested_name") or league.get("api_name") or "league")
    season_id = league["season_id"]
    slug = safe_slug(league_name)
    result: dict[str, Any] = {
        "league_name": league_name,
        "league_id": league.get("league_id"),
        "season_id": season_id,
        "season_name": league.get("season_name"),
    }

    fixtures_probe = probe(
        client,
        "fixtures",
        include="state",
        filters=f"fixtureSeasons:{season_id}",
        page=1,
        per_page=50,
    )
    result["fixtures_endpoint"] = public_probe_result(fixtures_probe)
    if not fixtures_probe.get("ok"):
        print(f"  [NO] {league_name}: {fixtures_probe.get('error')}")
        return result

    fixtures_payload = fixtures_probe["payload"]
    save_json(f"{slug}_fixtures_page_raw.json", fixtures_payload)
    fixtures = [item for item in as_list(fixtures_payload.get("data")) if isinstance(item, dict)]
    result["fixtures_endpoint"]["records_in_sample_page"] = len(fixtures)
    fixture = select_fixture(fixtures)
    if fixture is None:
        result["fixture_sample_error"] = "nessuna partita disponibile"
        print(f"  [NO] {league_name}: nessuna partita disponibile")
        return result

    fixture_id = fixture.get("id")
    result["sample_fixture"] = {
        "id": fixture_id,
        "starting_at": fixture.get("starting_at"),
        "result_info": fixture.get("result_info"),
    }

    core_probe = probe(
        client,
        f"fixtures/{fixture_id}",
        include=(
            "participants;scores;state;statistics.type;events.type;"
            "lineups.player;lineups.details.type"
        ),
    )
    result["fixture_detail_endpoint"] = public_probe_result(core_probe)
    if not core_probe.get("ok"):
        print(f"  [NO] {league_name}: dettaglio partita non disponibile")
        return result

    core_payload = core_probe["payload"]
    save_json(f"{slug}_fixture_{fixture_id}_raw.json", core_payload)
    lineups = fixture_lineups(core_payload)
    team_stats = fixture_team_stats(core_payload)
    player_match_stats = lineup_stat_details(lineups)
    events = relation_list(as_dict(core_payload.get("data")), "events")

    result["fixture_coverage"] = {
        "participants": len(relation_list(as_dict(core_payload.get("data")), "participants")),
        "scores": len(relation_list(as_dict(core_payload.get("data")), "scores")),
        "events": len(events),
        "event_types": event_types(core_payload),
        "lineups": len(lineups),
        "team_stat_records": len(team_stats),
        "team_stat_types": unique_types(team_stats),
        "player_stat_records": len(player_match_stats),
        "player_stat_types": unique_types(player_match_stats),
    }

    team = choose_team(core_payload)
    if team is None:
        result["team_sample_error"] = "partecipanti non presenti"
        print(f"  [NO] {league_name}: partecipanti non presenti")
        return result

    team_id = team.get("id")
    result["sample_team"] = {
        "id": team_id,
        "name": team.get("name") or team.get("short_code"),
    }

    squad_probe = probe(
        client,
        f"squads/seasons/{season_id}/teams/{team_id}",
        include="player;details.type",
    )
    result["squad_endpoint"] = public_probe_result(squad_probe)
    squad_payload = squad_probe.get("payload") if squad_probe.get("ok") else None
    squad_records = as_list(squad_payload.get("data")) if isinstance(squad_payload, dict) else []
    result["squad_endpoint"]["records"] = len(squad_records)
    if isinstance(squad_payload, dict):
        save_json(f"{slug}_team_{team_id}_squad_raw.json", squad_payload)

    team_stats_probe = probe(
        client,
        f"teams/{team_id}",
        include="statistics.details.type",
        filters=f"teamStatisticSeasons:{season_id}",
    )
    result["team_season_statistics_endpoint"] = public_probe_result(team_stats_probe)
    if team_stats_probe.get("ok"):
        team_stats_payload = team_stats_probe["payload"]
        team_season_details = season_stat_details(team_stats_payload)
        result["team_season_statistics_endpoint"].update(
            {
                "detail_records": len(team_season_details),
                "types": unique_types(team_season_details),
            }
        )
        save_json(f"{slug}_team_{team_id}_season_stats_raw.json", team_stats_payload)

    player = choose_player(lineups, squad_payload if isinstance(squad_payload, dict) else None)
    if player is not None:
        player_id = player["id"]
        result["sample_player"] = player
        player_stats_probe = probe(
            client,
            f"players/{player_id}",
            include="statistics.details.type",
            filters=f"playerStatisticSeasons:{season_id}",
        )
        result["player_season_statistics_endpoint"] = public_probe_result(player_stats_probe)
        if player_stats_probe.get("ok"):
            player_stats_payload = player_stats_probe["payload"]
            player_season_details = season_stat_details(player_stats_payload)
            result["player_season_statistics_endpoint"].update(
                {
                    "detail_records": len(player_season_details),
                    "types": unique_types(player_season_details),
                }
            )
            save_json(f"{slug}_player_{player_id}_season_stats_raw.json", player_stats_payload)
    else:
        result["player_sample_error"] = "nessun giocatore in formazione o rosa"

    team_xg_probe = probe(client, f"fixtures/{fixture_id}", include="xGFixture")
    result["team_xg_endpoint"] = public_probe_result(team_xg_probe)
    if team_xg_probe.get("ok"):
        team_xg_payload = team_xg_probe["payload"]
        result["team_xg_endpoint"].update(xg_summary(team_xg_payload, "xGFixture"))
        save_json(f"{slug}_fixture_{fixture_id}_team_xg_raw.json", team_xg_payload)

    player_xg_probe = probe(
        client,
        f"fixtures/{fixture_id}",
        include="lineups.xGLineup",
    )
    result["player_xg_endpoint"] = public_probe_result(player_xg_probe)
    if player_xg_probe.get("ok"):
        player_xg_payload = player_xg_probe["payload"]
        result["player_xg_endpoint"].update(xg_lineup_summary(player_xg_payload))
        save_json(f"{slug}_fixture_{fixture_id}_player_xg_raw.json", player_xg_payload)

    print(f"  [OK] {league_name} — fixture {fixture_id}")
    print_metric("Rosa", result["squad_endpoint"].get("records", 0))
    print_metric(
        "Metriche squadra/stagione",
        result["team_season_statistics_endpoint"].get("detail_records", 0),
    )
    print_metric(
        "Metriche giocatore/stagione",
        result.get("player_season_statistics_endpoint", {}).get("detail_records", 0),
    )
    print_metric("Metriche squadra/partita", len(result["fixture_coverage"]["team_stat_types"]))
    print_metric("Metriche giocatore/partita", len(result["fixture_coverage"]["player_stat_types"]))
    print_metric("Eventi", result["fixture_coverage"]["events"])

    team_xg = result["team_xg_endpoint"]
    player_xg = result["player_xg_endpoint"]
    print_metric(
        "xG squadra",
        f"OK ({team_xg.get('records', 0)})" if team_xg.get("ok") else "NON DISPONIBILE",
    )
    print_metric(
        "xG giocatore",
        f"OK ({player_xg.get('records', 0)})" if player_xg.get("ok") else "NON DISPONIBILE",
    )
    return result


def main() -> int:
    load_dotenv()
    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        print("ERRORE: SPORTMONKS_API_TOKEN non trovato nel file .env.", file=sys.stderr)
        return 2

    if not INPUT_REPORT.exists():
        print(
            f"ERRORE: {INPUT_REPORT} non trovato. Esegui prima sportmonks_audit.py.",
            file=sys.stderr,
        )
        return 2

    try:
        input_report = json.loads(INPUT_REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERRORE: impossibile leggere {INPUT_REPORT}: {exc}", file=sys.stderr)
        return 2

    leagues = [
        item
        for item in input_report.get("leagues", [])
        if isinstance(item, dict) and item.get("league_found") and item.get("season_found")
    ]
    if not leagues:
        print("ERRORE: nessuna lega valida trovata nel primo audit.", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(INPUT_REPORT),
        "method": "one finished fixture, one team and one player per league",
        "leagues": [],
    }

    print("\nSPORTMONKS DETAIL COVERAGE AUDIT")
    print(f"Campionati da verificare: {len(leagues)}")
    print("-" * 72)
    client = SportmonksClient(token)
    try:
        for league in leagues:
            report["leagues"].append(audit_league(client, league))
    finally:
        client.close()

    save_json(OUTPUT_REPORT.name, report)
    print("-" * 72)
    print(f"Report: {OUTPUT_REPORT}")
    print("Audit completato. Il token non e stato scritto nei file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())