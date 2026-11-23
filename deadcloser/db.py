"""Persistence. SQLite for local/dev and tests, Postgres (Neon / Supabase) for production.

Pass a filesystem path (or ":memory:") for SQLite, or a postgres:// URL for Postgres. All SQL in the code base is
written with `?` placeholders and portable syntax; this module adapts it. Every row is tenant-scoped and all helpers
require a tenant_id so one tenant can never read another tenant's data."""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, vertical TEXT NOT NULL,
  api_key_hash TEXT UNIQUE, config TEXT NOT NULL, created_at INTEGER NOT NULL, is_demo INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS items (
  id TEXT NOT NULL, tenant_id TEXT NOT NULL, title TEXT NOT NULL, price INTEGER NOT NULL,
  attrs TEXT NOT NULL, active INTEGER DEFAULT 1, PRIMARY KEY (tenant_id, id)
);
CREATE TABLE IF NOT EXISTS leads (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT, thread_id TEXT, name TEXT, phone TEXT, email TEXT, source TEXT,
  item_id TEXT, status TEXT NOT NULL DEFAULT 'new', awaiting TEXT, offered TEXT DEFAULT '[]',
  budget_min INTEGER, budget_max INTEGER, timeline_days INTEGER, timeline_text TEXT,
  browsing INTEGER DEFAULT 0, prefs TEXT DEFAULT '{}', score INTEGER DEFAULT 0, qualified INTEGER DEFAULT 0,
  asked TEXT DEFAULT '{}', followups_sent INTEGER DEFAULT 0, opted_out INTEGER DEFAULT 0,
  summary TEXT DEFAULT '', flags TEXT DEFAULT '[]', created_at INTEGER NOT NULL,
  last_in_at INTEGER, last_out_at INTEGER, first_reply_ms INTEGER, last_angle TEXT, used_angles TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_leads_tenant ON leads(tenant_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_leads_phone ON leads(tenant_id, phone) WHERE phone IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_leads_user ON leads(tenant_id, user_id) WHERE user_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, lead_id TEXT NOT NULL, thread_id TEXT,
  direction TEXT NOT NULL, body TEXT NOT NULL, ts INTEGER NOT NULL, meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_msg_lead ON messages(tenant_id, lead_id, id);
CREATE TABLE IF NOT EXISTS memories (
  tenant_id TEXT NOT NULL, lead_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
  kind TEXT NOT NULL, confidence REAL DEFAULT 1.0, ts INTEGER NOT NULL,
  PRIMARY KEY (tenant_id, lead_id, key)
);
CREATE TABLE IF NOT EXISTS playbook (
  tenant_id TEXT NOT NULL, angle TEXT NOT NULL, tries INTEGER DEFAULT 0, wins INTEGER DEFAULT 0,
  PRIMARY KEY (tenant_id, angle)
);
CREATE TABLE IF NOT EXISTS visits (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, lead_id TEXT NOT NULL, item_id TEXT,
  start_ts INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'booked', billable INTEGER DEFAULT 0,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_visits_t ON visits(tenant_id, start_ts);
CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, visit_id TEXT NOT NULL UNIQUE,
  amount_cents INTEGER NOT NULL, status TEXT NOT NULL, note TEXT, created_at INTEGER NOT NULL, settled_at INTEGER
);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, lead_id TEXT NOT NULL,
  kind TEXT NOT NULL, run_at INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', payload TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_jobs_due ON jobs(status, run_at);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, lead_id TEXT, ts INTEGER NOT NULL,
  event TEXT NOT NULL, detail TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS demo_sessions (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, lead_id TEXT NOT NULL, sim_now INTEGER NOT NULL, created_at INTEGER NOT NULL
);
"""

JSON_COLS = {"config", "attrs", "offered", "prefs", "asked", "flags", "meta", "payload", "detail", "used_angles"}
TABLES = ["tenants", "items", "leads", "messages", "memories", "playbook", "visits", "ledger", "jobs", "audit", "demo_sessions"]


def is_postgres_url(s: str) -> bool:
    return s.startswith(("postgres://", "postgresql://"))


def _row(r) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        if isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


class Database:
    def __init__(self, target: str = ":memory:"):
        self.target = target
        self.pg = is_postgres_url(target)
        self.lock = threading.RLock()          # SQLite only: one connection, serialised
        self._tl = threading.local()           # Postgres: the connection pinned by an open transaction
        self.conn = None
        self.pool = None
        self._connect()
        with self._guard():
            if not (self.pg and self._schema_present()):
                for stmt in self._schema_statements():
                    self._raw(stmt)

    # ---------------------------------------------------------------- connection
    def _guard(self):
        """SQLite needs a global lock; Postgres runs different users in parallel on pooled connections."""
        return contextlib.nullcontext() if self.pg else self.lock

    def _connect(self) -> None:
        if self.pg:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
            self.pool = ConnectionPool(self.target, min_size=1, max_size=int(os.getenv("DB_POOL_MAX", "8")), open=True, timeout=30,
                                       check=ConnectionPool.check_connection,
                                       kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None, "connect_timeout": 15})
            self.pool.wait(timeout=30)
        else:
            if self.target != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(self.target)), exist_ok=True)
            self.conn = sqlite3.connect(self.target, check_same_thread=False, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")   # durable across app crashes; avoids an fsync per write

    def _schema_present(self) -> bool:
        row = self._raw("SELECT to_regclass('public.demo_sessions') AS a, to_regclass('public.audit') AS b").fetchone()
        return bool(row["a"] and row["b"])

    def _schema_statements(self) -> list[str]:
        sql = SCHEMA
        if self.pg:
            sql = (sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
                      .replace(" INTEGER", " BIGINT").replace(" REAL", " DOUBLE PRECISION"))
        return [s.strip() for s in sql.split(";") if s.strip()]

    def _tr(self, sql: str) -> str:
        return sql.replace("%", "%%").replace("?", "%s") if self.pg else sql

    def _raw(self, sql: str, params: Iterable[Any] = ()):
        sql = self._tr(sql)
        if not self.pg:
            return self.conn.execute(sql, tuple(params))
        pinned = getattr(self._tl, "conn", None)
        if pinned is not None:                       # inside a transaction: same connection
            return pinned.execute(sql, tuple(params))
        for attempt in (0, 1):
            try:
                with self.pool.connection() as conn:
                    return conn.execute(sql, tuple(params))     # results are buffered client-side, safe after release
            except Exception as e:  # noqa: BLE001
                if attempt == 0 and type(e).__name__ in ("OperationalError", "InterfaceError", "AdminShutdown", "PoolTimeout"):
                    time.sleep(0.2)                  # serverless Postgres drops idle connections: retry once on a fresh one
                    continue
                raise

    # ---------------------------------------------------------------- primitives
    def execute(self, sql: str, params: Iterable[Any] = ()):
        with self._guard():
            return self._raw(sql, params)

    def insert_id(self, sql: str, params: Iterable[Any] = ()) -> int:
        """INSERT and return the new autoincrement id on either backend."""
        with self._guard():
            if self.pg:
                return self._raw(sql + " RETURNING id", params).fetchone()["id"]
            return self._raw(sql, params).lastrowid

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        with self._guard():
            return _row(self._raw(sql, params).fetchone())

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._guard():
            return [_row(r) for r in self._raw(sql, params).fetchall()]

    def transaction(self):
        return _Tx(self)

    def update(self, table: str, tenant_id: str, row_id: str, **fields) -> None:
        """UPDATE one row, always constrained by tenant_id."""
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if k in JSON_COLS else v for k, v in fields.items()]
        self.execute(f"UPDATE {table} SET {cols} WHERE tenant_id=? AND id=?", (*vals, tenant_id, row_id))

    def audit(self, tenant_id: str | None, lead_id: str | None, ts: int, event: str, **detail) -> None:
        self.execute(
            "INSERT INTO audit(tenant_id, lead_id, ts, event, detail) VALUES (?,?,?,?,?)",
            (tenant_id, lead_id, ts, event, json.dumps(detail, default=str)),
        )

    def reset(self) -> None:
        """Drop every table (tests only)."""
        with self._guard():
            for t in TABLES:
                self._raw(f"DROP TABLE IF EXISTS {t}")
            for stmt in self._schema_statements():
                self._raw(stmt)


class _Tx:
    """Atomic block. SQLite: global lock + BEGIN IMMEDIATE. Postgres: one pooled connection pinned to this thread, with a
    global advisory lock so concurrent bookings cannot oversell a slot."""

    def __init__(self, db: Database):
        self.db = db
        self._cm = None

    def __enter__(self):
        db = self.db
        if not db.pg:
            db.lock.acquire()
            db._raw("BEGIN IMMEDIATE")
            return db
        self._cm = db.pool.connection()
        conn = self._cm.__enter__()
        db._tl.conn = conn
        conn.execute("BEGIN")
        conn.execute("SELECT pg_advisory_xact_lock(727001)")
        return db

    def __exit__(self, exc_type, exc, tb):
        db = self.db
        if not db.pg:
            try:
                db._raw("ROLLBACK" if exc_type else "COMMIT")
            finally:
                db.lock.release()
            return
        try:
            db._tl.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            db._tl.conn = None
            self._cm.__exit__(None, None, None)
