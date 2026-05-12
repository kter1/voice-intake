from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

DATABASE_URL_DEFAULT = "sqlite:///./voice_intake.db"


class Base(DeclarativeBase):
    pass


def get_engine(url: str | None = None) -> Engine:
    resolved = url or os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)
    connect_args: dict = {}
    kwargs: dict = {}
    if resolved.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in resolved:
            kwargs["poolclass"] = StaticPool
    return create_engine(resolved, connect_args=connect_args, **kwargs)


def create_tables(engine: Engine) -> None:
    from . import models_orm  # noqa: F401 - registers ORM classes with Base metadata
    from voice_intake.knowledge import orm as _knowledge_orm  # noqa: F401
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)
