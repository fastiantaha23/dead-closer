"""Runtime settings and time helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: os.getenv("DEADCLOSER_DB", "data/deadcloser.db"))
    checkpoint_path: str = field(default_factory=lambda: os.getenv("DEADCLOSER_CHECKPOINTS", "data/checkpoints.db"))
    admin_key: str = field(default_factory=lambda: os.getenv("DEADCLOSER_ADMIN_KEY", ""))
    llm_backend: str = field(default_factory=lambda: os.getenv("DEADCLOSER_LLM", "auto"))  # auto|rules|claude
    model: str = field(default_factory=lambda: os.getenv("DEADCLOSER_MODEL", "claude-haiku-5-5"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))   # postgres:// (Neon/Supabase) => persistent
    cors_origins: str = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("DEADCLOSER_LLM_TIMEOUT", "20")))
    demo_mode: bool = field(default_factory=lambda: os.getenv("DEADCLOSER_DEMO", "1") == "1")
    serverless: bool = field(default_factory=lambda: bool(os.getenv("VERCEL") or os.getenv("DEADCLOSER_SERVERLESS")))
    first_reply_sla_s: int = 60
    scheduler_interval_s: int = field(default_factory=lambda: int(os.getenv("DEADCLOSER_TICK", "30")))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def from_epoch(ts: int | float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def local(dt: datetime, tz_offset_min: int) -> datetime:
    """Express dt in the tenant's fixed UTC offset."""
    return dt.astimezone(timezone(timedelta(minutes=tz_offset_min)))


def fmt_slot(dt: datetime, tz_offset_min: int) -> str:
    l = local(dt, tz_offset_min)
    hour = l.strftime("%I").lstrip("0")
    return f"{l.strftime('%a %b')} {l.day}, {hour}:{l.strftime('%M')}{l.strftime('%p').lower()}"
