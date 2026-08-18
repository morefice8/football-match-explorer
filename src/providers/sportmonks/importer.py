"""Resumable Sportmonks season importer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .client import SportmonksAPIError, SportmonksClient
from .config import LEAGUES, LeagueConfig, season_name_matches
from .normalizer import SeasonNormalizer, coach_from_team, response_data


CORE_FIXTURE_INCLUDE = (
    "participants;scores;state;statistics.type;events.type;"
    "lineups.player;lineups.details.type"
)
FULL_FIXTURE_INCLUDE = f"{CORE_FIXTURE_INCLUDE};xGFixture;lineups.xGLineup"


def _items(payload: Any) -> list[dict[str, Any]]:
    value = response_data(payload)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _finished(fixture: dict[str, Any]) -> bool:
    state = fixture.get("state") if isinstance(fixture.get("state"), dict) else {}
    labels = {
        str(state.get("short_name", "")).casefold(),
        str(state.get("developer_name", "")).casefold(),
        str(state.get("name", "")).casefold(),
        str(state.get("state", "")).casefold(),
    }
    finished = {"ft", "aet", "pen", "finished", "after extra time", "after penalties"}
    return bool(labels & finished) or bool(fixture.get("result_info"))


class JsonCache:
    def __init__(self, root: Path, refresh: bool = False, cache_only: bool = False):
        self.root = root
        self.refresh = refresh
        self.cache_only = cache_only

    def get(
        self,
        relative: str | Path,
        fetch: Callable[[], dict[str, Any]],
        *,
        force_refresh: bool = False,
        allow_fetch_in_cache_only: bool = False,
    ) -> dict[str, Any]:
        path = self.root / relative
        if path.exists() and not self.refresh and not force_refresh:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if self.cache_only and not allow_fetch_in_cache_only:
                    raise RuntimeError(f"Invalid cached Sportmonks payload: {path}") from exc
                # An interrupted write or manual edit should repair itself.
                pass
        if self.cache_only and not allow_fetch_in_cache_only:
            raise RuntimeError(f"Missing cached Sportmonks payload: {path}")
        payload = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return payload


class SportmonksImporter:
    def __init__(
        self,
        client: SportmonksClient,
        project_root: Path,
        season: str,
        *,
        refresh: bool = False,
        skip_xg: bool = False,
        max_fixtures: int | None = None,
        cache_only: bool = False,
        refresh_coaches: bool = False,
        hydrate_coaches: bool = False,
    ):
        if refresh and cache_only:
            raise ValueError("refresh and cache_only cannot be enabled together")
        if refresh_coaches and hydrate_coaches:
            raise ValueError("refresh_coaches cannot be combined with hydrate_coaches")
        if (refresh_coaches or hydrate_coaches) and (refresh or cache_only):
            raise ValueError(
                "refresh_coaches/hydrate_coaches cannot be combined with refresh or cache_only"
            )
        self.client = client
        self.project_root = project_root.resolve()
        self.season = season
        self.raw_root = self.project_root / "data" / "sportmonks" / "raw" / season
        # Coach-only modes are intentionally cache-only for every other resource.
        # Refresh updates one teams payload per league; hydration reuses that payload.
        self.cache = JsonCache(
            self.raw_root,
            refresh=refresh,
            cache_only=cache_only or refresh_coaches or hydrate_coaches,
        )
        # Names and photos are season-independent, so one cached coach profile
        # can be reused by every league and every imported season.
        self.coach_cache = JsonCache(
            self.project_root / "data" / "sportmonks" / "raw" / "_shared",
            cache_only=cache_only or refresh_coaches or hydrate_coaches,
        )
        self.skip_xg = skip_xg
        self.max_fixtures = max_fixtures
        self.cache_only = cache_only
        self.refresh_coaches = refresh_coaches
        self.hydrate_coaches = hydrate_coaches
        self.normalizer = SeasonNormalizer(season)
        self.errors: list[dict[str, Any]] = []

        try:
            start, end = season.split("-", maxsplit=1)
            self.start_year = int(start)
            self.end_year = int(end)
        except (ValueError, TypeError):
            raise ValueError("Season must use the YYYY-YYYY format, for example 2025-2026") from None

    def _record_error(self, league: LeagueConfig, scope: str, exc: Exception) -> None:
        self.errors.append({"league": league.name, "scope": scope, "error": str(exc)})

    def _resolve_season(self, league: LeagueConfig) -> dict[str, Any]:
        payload = self.cache.get(
            Path(league.slug) / "league.json",
            lambda: self.client.get(f"leagues/{league.league_id}", include="seasons"),
        )
        league_data = response_data(payload)
        if not isinstance(league_data, dict):
            raise RuntimeError(f"Unexpected league response for {league.name}")
        seasons = league_data.get("seasons")
        if isinstance(seasons, dict):
            seasons = seasons.get("data", [])
        for season in seasons if isinstance(seasons, list) else []:
            if not isinstance(season, dict):
                continue
            if season_name_matches(season.get("name"), self.start_year, self.end_year):
                return season
            starts = str(season.get("starting_at", ""))
            ends = str(season.get("ending_at", ""))
            if starts.startswith(str(self.start_year)) and ends.startswith(str(self.end_year)):
                return season
        raise RuntimeError(f"Season {self.season} not found for {league.name}")

    def _paginated(
        self,
        league: LeagueConfig,
        cache_directory: str,
        endpoint: str,
        **params: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        page = 1
        while page <= 100:
            payload = self.cache.get(
                Path(league.slug) / cache_directory / f"page_{page:03d}.json",
                lambda page=page: self.client.get(endpoint, page=page, per_page=50, **params),
            )
            pages.append(payload)
            records.extend(_items(payload))
            pagination = payload.get("pagination") if isinstance(payload, dict) else {}
            if not isinstance(pagination, dict):
                meta = payload.get("meta") if isinstance(payload, dict) else {}
                pagination = meta.get("pagination", {}) if isinstance(meta, dict) else {}
            next_page = pagination.get("next_page")
            if not next_page and not pagination.get("has_more"):
                return records, pages
            page = next_page if isinstance(next_page, int) else page + 1
        raise RuntimeError(f"Pagination safety limit exceeded for {endpoint}")

    def _fixture_payload(self, league: LeagueConfig, fixture_id: int) -> dict[str, Any]:
        relative = Path(league.slug) / "fixtures" / f"{fixture_id}.json"

        def fetch() -> dict[str, Any]:
            include = CORE_FIXTURE_INCLUDE if self.skip_xg else FULL_FIXTURE_INCLUDE
            try:
                return self.client.get(f"fixtures/{fixture_id}", include=include)
            except SportmonksAPIError as combined_error:
                # Some subscriptions reject a deep combined include even though
                # its individual relationships are available.
                if self.skip_xg:
                    raise
                try:
                    parts = Path(league.slug) / "fixture_parts" / str(fixture_id)
                    core = self.cache.get(
                        parts / "core.json",
                        lambda: self.client.get(f"fixtures/{fixture_id}", include=CORE_FIXTURE_INCLUDE),
                    )
                    team_xg = self.cache.get(
                        parts / "team_xg.json",
                        lambda: self.client.get(f"fixtures/{fixture_id}", include="xGFixture"),
                    )
                    player_xg = self.cache.get(
                        parts / "player_xg.json",
                        lambda: self.client.get(f"fixtures/{fixture_id}", include="lineups.xGLineup"),
                    )
                except SportmonksAPIError:
                    raise combined_error
                return self._merge_fixture_relations(core, team_xg, player_xg)

        return self.cache.get(relative, fetch)

    @staticmethod
    def _merge_fixture_relations(
        core_payload: dict[str, Any],
        team_xg_payload: dict[str, Any],
        player_xg_payload: dict[str, Any],
    ) -> dict[str, Any]:
        core = response_data(core_payload)
        team_xg = response_data(team_xg_payload)
        player_xg = response_data(player_xg_payload)
        if not isinstance(core, dict):
            return core_payload
        if isinstance(team_xg, dict):
            relation = team_xg.get("xgfixture") or team_xg.get("xGFixture") or []
            core["xgfixture"] = relation
        xg_by_lineup: dict[Any, Any] = {}
        if isinstance(player_xg, dict):
            for lineup in player_xg.get("lineups", []):
                if isinstance(lineup, dict):
                    xg_by_lineup[lineup.get("id")] = lineup.get("xglineup") or lineup.get("xGLineup") or []
        for lineup in core.get("lineups", []):
            if isinstance(lineup, dict) and lineup.get("id") in xg_by_lineup:
                lineup["xglineup"] = xg_by_lineup[lineup["id"]]
        return core_payload

    def _squad_payload(
        self, league: LeagueConfig, season_id: int, team_id: int
    ) -> dict[str, Any]:
        relative = Path(league.slug) / "squads" / f"team_{team_id}.json"

        def fetch() -> dict[str, Any]:
            endpoint = f"squads/seasons/{season_id}/teams/{team_id}"
            try:
                return self.client.get(
                    endpoint,
                    include="player;player.nationality;player.position;details.type",
                )
            except SportmonksAPIError:
                return self.client.get(endpoint, include="player;details.type")

        return self.cache.get(relative, fetch)

    def _coach_profiles(
        self,
        league: LeagueConfig,
        teams: list[dict[str, Any]],
        *,
        as_of: Any,
    ) -> dict[int, dict[str, Any]]:
        """Load only the one season-relevant coach profile for each team."""
        coach_ids: set[int] = set()
        for team in teams:
            selected = coach_from_team(team, as_of=as_of)
            raw_id = selected.get("coach_id")
            try:
                if raw_id is not None:
                    coach_ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue

        profiles: dict[int, dict[str, Any]] = {}
        for coach_id in sorted(coach_ids):
            try:
                payload = self.coach_cache.get(
                    Path("coaches") / f"{coach_id}.json",
                    lambda coach_id=coach_id: self.client.get(f"coaches/{coach_id}"),
                    # Coach profiles are stable and reusable. Refreshing the
                    # team relation must not spend requests on profiles already cached.
                    allow_fetch_in_cache_only=self.refresh_coaches or self.hydrate_coaches,
                )
            except SportmonksAPIError as exc:
                self._record_error(league, f"coach {coach_id}", exc)
                continue
            except RuntimeError:
                # Old raw caches can still be normalised without coach photos.
                if self.cache_only:
                    continue
                raise
            except Exception as exc:
                self._record_error(league, f"coach {coach_id}", exc)
                continue
            profile = response_data(payload)
            if isinstance(profile, dict):
                profiles[coach_id] = profile
        return profiles

    def import_league(self, league: LeagueConfig) -> dict[str, Any]:
        print(f"\n[{league.name}] Resolving season...")
        season = self._resolve_season(league)
        season_id = int(season["id"])
        league_context = {
            "league_id": league.league_id,
            "league": league.name,
            "league_slug": league.slug,
            "country": league.country,
            "season_id": season_id,
            "season_api_name": season.get("name"),
        }
        self.normalizer.add_league(league_context)

        standings = self.cache.get(
            Path(league.slug) / "standings.json",
            lambda: self.client.get(
                f"standings/seasons/{season_id}", include="participant;details.type"
            ),
        )
        self.normalizer.add_standings(standings, league_context)

        teams_payload = self.cache.get(
            Path(league.slug) / "teams.json",
            lambda: self.client.get(f"teams/seasons/{season_id}", include="coaches"),
            force_refresh=self.refresh_coaches,
            allow_fetch_in_cache_only=self.refresh_coaches,
        )
        teams = _items(teams_payload)
        coach_as_of = season.get("ending_at") or f"{self.end_year}-06-30"
        coach_profiles = self._coach_profiles(
            league,
            teams,
            as_of=coach_as_of,
        )
        self.normalizer.add_teams(
            teams_payload,
            league_context,
            coach_profiles=coach_profiles,
            coach_as_of=coach_as_of,
        )
        print(f"[{league.name}] Teams: {len(teams)}")
        if coach_profiles:
            print(f"[{league.name}] Coach profiles: {len(coach_profiles)}")
        for index, team in enumerate(teams, start=1):
            team_id = team.get("id")
            if team_id is None:
                continue
            team_context = {"team_id": team_id, "team_name": team.get("name")}
            try:
                squad = self._squad_payload(league, season_id, int(team_id))
                self.normalizer.add_squad(squad, league_context, team_context)
            except Exception as exc:  # one unavailable squad must not discard the league
                self._record_error(league, f"squad team {team_id}", exc)
            print(f"  squads {index}/{len(teams)}", end="\r", flush=True)
        print(" " * 48, end="\r")

        fixtures, _ = self._paginated(
            league,
            "fixture_index",
            "fixtures",
            filters=f"fixtureSeasons:{season_id}",
            include="participants;scores;state",
        )
        for fixture in fixtures:
            self.normalizer.add_fixture({"data": fixture}, league_context)
        completed = [fixture for fixture in fixtures if _finished(fixture)]
        completed.sort(key=lambda item: str(item.get("starting_at", "")))
        if self.max_fixtures is not None:
            completed = completed[: self.max_fixtures]
        print(f"[{league.name}] Completed fixtures to import: {len(completed)}")

        imported = 0
        for index, fixture in enumerate(completed, start=1):
            fixture_id = fixture.get("id")
            if fixture_id is None:
                continue
            try:
                detail = self._fixture_payload(league, int(fixture_id))
                self.normalizer.add_fixture(detail, league_context)
                imported += 1
            except Exception as exc:
                self._record_error(league, f"fixture {fixture_id}", exc)
            print(f"  fixtures {index}/{len(completed)}", end="\r", flush=True)
        print(" " * 48, end="\r")
        print(f"[{league.name}] Imported: {imported}/{len(completed)}")
        return {
            **league_context,
            "teams": len(teams),
            "fixtures_available": len(fixtures),
            "fixtures_completed_selected": len(completed),
            "fixtures_imported": imported,
        }

    def run(self, leagues: Iterable[LeagueConfig]) -> dict[str, Any]:
        selected = list(leagues)
        report: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "season": self.season,
            "leagues": [],
        }
        for league in selected:
            try:
                report["leagues"].append(self.import_league(league))
            except Exception as exc:
                self._record_error(league, "league", exc)
                print(f"[{league.name}] ERROR: {exc}")
                if self.cache_only:
                    raise RuntimeError(
                        "Cache-only normalization aborted before writing processed datasets. "
                        "The existing processed files were not replaced."
                    ) from exc

        if not report["leagues"]:
            report["errors"] = self.errors
            report_path = self.raw_root / "import_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError("No league was imported; existing processed datasets were not changed")

        print("\nNormalising and writing datasets...")
        frames = self.normalizer.save(self.project_root)
        report["errors"] = self.errors
        report["datasets"] = {name: len(frame) for name, frame in frames.items()}
        report_path = self.raw_root / "import_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {report_path.relative_to(self.project_root)}")
        if self.errors:
            print(f"Completed with {len(self.errors)} recoverable error(s). See the report.")
        else:
            print("Import completed without errors.")
        return report


def select_leagues(values: Iterable[str] | None) -> list[LeagueConfig]:
    if not values:
        return list(LEAGUES.values())
    requested = {
        part.strip().casefold()
        for value in values
        for part in value.split(",")
        if part.strip()
    }
    selected = [
        league
        for slug, league in LEAGUES.items()
        if slug.casefold() in requested or league.name.casefold() in requested
    ]
    aliases = {
        alias
        for league in selected
        for alias in (league.slug.casefold(), league.name.casefold())
    }
    missing = requested - aliases
    if missing:
        choices = ", ".join(LEAGUES)
        raise ValueError(f"Unknown league(s): {', '.join(sorted(missing))}. Choices: {choices}")
    return selected
