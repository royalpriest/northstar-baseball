from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from northstar.db.models import Base


def create_engine_from_url(database_url: str, *, echo: bool = False) -> Engine:
    kwargs: dict[str, object] = {
        "future": True,
        "echo": echo,
        "pool_pre_ping": True,
    }

    if database_url.startswith("sqlite") and ":memory:" in database_url:
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = StaticPool

    return create_engine(database_url, **kwargs)


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )
