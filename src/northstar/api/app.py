from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from northstar.api.store import DEFAULT_PROFILE_PATH, ProfileStore


API_VERSION = "v1"
SEASON = 2025


class Utf8JSONResponse(JSONResponse):
    """JSON response with an explicit UTF-8 charset for older HTTP clients."""

    media_type = "application/json; charset=utf-8"


def _cors_origins() -> list[str]:
    configured = os.getenv(
        "NORTHSTAR_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [
        origin.strip()
        for origin in configured.split(",")
        if origin.strip()
    ]


def _store(request: Request) -> ProfileStore:
    current = request.app.state.profile_store
    if current is not None:
        return current

    profile_path = Path(
        os.getenv(
            "NORTHSTAR_PROFILE_PATH",
            str(DEFAULT_PROFILE_PATH),
        )
    )
    try:
        current = ProfileStore.from_csv(profile_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    request.app.state.profile_store = current
    return current


def create_app(
    *,
    store: ProfileStore | None = None,
) -> FastAPI:
    service = FastAPI(
        title="NorthStar Baseball API",
        default_response_class=Utf8JSONResponse,
        version="0.1.0",
        description=(
            "Read-only API for NorthStar hitter profiles, contextual "
            "diagnostics, Canadian-born views, team views, and comparisons."
        ),
    )
    service.state.profile_store = store

    service.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @service.get("/")
    def root() -> dict[str, object]:
        return {
            "service": "NorthStar Baseball API",
            "api_version": API_VERSION,
            "season": SEASON,
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "api": f"/api/{API_VERSION}",
        }

    @service.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "northstar-baseball-api",
            "api_version": API_VERSION,
            "season": SEASON,
        }

    @service.get(f"/api/{API_VERSION}/meta")
    def meta(request: Request) -> dict[str, object]:
        profile_store = _store(request)
        return {
            "season": SEASON,
            "players": profile_store.size,
            "northstar_axes": [
                "selection_context_percentile",
                "contact_context_percentile",
                "damage_context_percentile",
            ],
            "scope_notes": {
                "canadian": (
                    "Canadian-born means MLB birth_country == Canada; "
                    "citizenship is not inferred."
                ),
                "team": (
                    "Team membership uses 2025 plate-appearance exposure. "
                    "NorthStar diagnostics remain full-season MLB values."
                ),
                "score": (
                    "NorthStar does not calculate a composite player score."
                ),
            },
        }

    @service.get(f"/api/{API_VERSION}/players")
    def players(
        request: Request,
        query: str | None = None,
        team: str | None = None,
        canadian_born: bool | None = None,
        limit: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        items = _store(request).search_players(
            query=query,
            team=team,
            canadian_born=canadian_born,
            limit=limit,
        )
        return {
            "count": len(items),
            "items": items,
        }

    @service.get(f"/api/{API_VERSION}/players/{{batter}}")
    def player(
        batter: int,
        request: Request,
    ) -> dict[str, object]:
        item = _store(request).get_player(batter)
        if item is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown MLBAM batter ID: {batter}",
            )
        return item

    @service.get(f"/api/{API_VERSION}/canada")
    def canada(
        request: Request,
        limit: int = Query(default=100, ge=1, le=250),
    ) -> dict[str, object]:
        items = _store(request).canadian_players(limit=limit)
        return {
            "scope": "Canadian-born",
            "count": len(items),
            "items": items,
        }

    @service.get(f"/api/{API_VERSION}/teams/{{team_code}}")
    def team(
        team_code: str,
        request: Request,
        limit: int = Query(default=100, ge=1, le=250),
    ) -> dict[str, object]:
        code = team_code.strip().upper()
        if not code or len(code) > 4:
            raise HTTPException(
                status_code=422,
                detail="team_code must be a short MLB team code",
            )

        items = _store(request).team_players(
            code,
            limit=limit,
        )
        return {
            "team": code,
            "season": SEASON,
            "metric_scope": "full-season MLB",
            "count": len(items),
            "items": items,
        }

    @service.get(f"/api/{API_VERSION}/leaderboards")
    def leaderboard(
        request: Request,
        scope: str = "all",
        sort_by: str = "re24_per_100_pa",
        direction: str = "desc",
        min_pa: int = Query(default=0, ge=0),
        limit: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        direction_normalized = direction.strip().casefold()
        if direction_normalized not in {"asc", "desc"}:
            raise HTTPException(
                status_code=422,
                detail="direction must be asc or desc",
            )

        try:
            items = _store(request).leaderboard(
                scope=scope,
                sort_by=sort_by,
                ascending=direction_normalized == "asc",
                min_pa=min_pa,
                limit=limit,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown leaderboard field: {sort_by}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        return {
            "scope": scope,
            "sort_by": sort_by,
            "direction": direction_normalized,
            "min_pa": min_pa,
            "count": len(items),
            "items": items,
        }

    @service.get(f"/api/{API_VERSION}/compare")
    def compare(
        request: Request,
        player: list[str] = Query(
            ...,
            description=(
                "Repeat the player query parameter 2-4 times using "
                "MLBAM IDs or exact player names."
            ),
        ),
    ) -> dict[str, object]:
        profile_store = _store(request)
        resolved = profile_store.resolve_players(player)

        if len(resolved) < 2 or len(resolved) > 4:
            raise HTTPException(
                status_code=422,
                detail="Provide between 2 and 4 players.",
            )

        try:
            items = profile_store.compare(resolved)
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        return {
            "count": len(items),
            "players": items,
            "interpretation": (
                "NorthStar axes are contextual diagnostics; "
                "no comparison winner or composite score is calculated."
            ),
        }

    return service


app = create_app()
