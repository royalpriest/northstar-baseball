from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class HitterProfile(Base):
    __tablename__ = "hitter_profiles"

    batter: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True, default=2025)

    player_name: Mapped[str] = mapped_column(String(160), index=True)
    birth_country: Mapped[str | None] = mapped_column(String(80), nullable=True)
    birth_city: Mapped[str | None] = mapped_column(String(120), nullable=True)
    birth_state_province: Mapped[str | None] = mapped_column(String(120), nullable=True)
    canadian_born: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    primary_team_2025: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    teams_2025: Mapped[str | None] = mapped_column(Text, nullable=True)
    season_plate_appearances: Mapped[int | None] = mapped_column(Integer, nullable=True)

    selection_context_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    contact_context_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    damage_context_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)

    woba: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_woba: Mapped[float | None] = mapped_column(Float, nullable=True)
    re24_per_100_pa: Mapped[float | None] = mapped_column(Float, nullable=True)

    profile_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class HitterTeamMembership(Base):
    __tablename__ = "hitter_team_memberships"

    batter: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("hitter_profiles.batter", ondelete="CASCADE"),
        primary_key=True,
    )
    season: Mapped[int] = mapped_column(Integer, primary_key=True, default=2025)
    team_code: Mapped[str] = mapped_column(String(8), primary_key=True, index=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
