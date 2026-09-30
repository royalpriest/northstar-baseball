from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from northstar.api.store import DEFAULT_PROFILE_PATH, ProfileStore
from northstar.db.database import (
    create_engine_from_url,
    create_schema,
    session_factory,
)
from northstar.db.store import DatabaseProfileStore


API_VERSION = "v1"
SEASON = 2025
logger = logging.getLogger(__name__)


def _is_production() -> bool:
    return os.getenv("NORTHSTAR_ENV", "development").strip().casefold() == "production"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _docs_enabled() -> bool:
    return _env_flag("NORTHSTAR_ENABLE_DOCS", default=not _is_production())


def _auto_create_schema() -> bool:
    return _env_flag(
        "NORTHSTAR_AUTO_CREATE_SCHEMA",
        default=not _is_production(),
    )


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


def _store(
    request: Request,
) -> ProfileStore | DatabaseProfileStore:
    current = request.app.state.profile_store
    if current is not None:
        return current

    database_url = os.getenv("NORTHSTAR_DATABASE_URL")
    if database_url:
        try:
            engine = create_engine_from_url(database_url)
            if _auto_create_schema():
                create_schema(engine)
            current = DatabaseProfileStore(
                session_factory(engine),
                season=SEASON,
            )
        except Exception as exc:
            logger.exception("Failed to initialize NorthStar database backend")
            raise HTTPException(
                status_code=503,
                detail="NorthStar database backend is unavailable.",
            ) from exc

        request.app.state.db_engine = engine
        request.app.state.profile_store = current
        return current

    if _is_production():
        raise HTTPException(
            status_code=503,
            detail="NorthStar database backend is required in production.",
        )

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
    docs_enabled = _docs_enabled()
    service = FastAPI(
        title="NorthStar Baseball API",
        default_response_class=Utf8JSONResponse,
        version="0.1.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        description=(
            "Read-only API for NorthStar hitter profiles, contextual "
            "diagnostics, Canadian-born views, team views, and comparisons."
        ),
    )
    service.state.profile_store = store
    service.state.db_engine = None

    service.add_middleware(GZipMiddleware, minimum_size=1024)

    service.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @service.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        return response

    @service.get("/")
    def root() -> dict[str, object]:
        return {
            "service": "NorthStar Baseball API",
            "api_version": API_VERSION,
            "season": SEASON,
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs" if docs_enabled else None,
            "openapi": "/openapi.json" if docs_enabled else None,
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

    @service.get("/ready")
    def ready(request: Request) -> dict[str, object]:
        try:
            profile_store = _store(request)
            players = profile_store.size
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("NorthStar readiness check failed")
            raise HTTPException(
                status_code=503,
                detail="NorthStar data backend is not ready.",
            ) from exc

        if players <= 0:
            raise HTTPException(
                status_code=503,
                detail="NorthStar data backend contains no player profiles.",
            )

        return {
            "status": "ready",
            "service": "northstar-baseball-api",
            "storage_backend": (
                "postgresql"
                if isinstance(profile_store, DatabaseProfileStore)
                else "csv"
            ),
            "players": players,
            "season": SEASON,
        }

    @service.get(f"/api/{API_VERSION}/meta")
    def meta(request: Request) -> dict[str, object]:
        profile_store = _store(request)
        return {
            "season": SEASON,
            "players": profile_store.size,
            "storage_backend": (
                "postgresql"
                if isinstance(profile_store, DatabaseProfileStore)
                else "csv"
            ),
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
                "MLBAM IDs, first names, last names, or full player names."
            ),
        ),
    ) -> dict[str, object]:
        profile_store = _store(request)

        try:
            resolved = profile_store.resolve_players(player)
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

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
