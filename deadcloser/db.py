"""SQLite persistence. Every row is tenant-scoped; all helpers require a tenant_id so one
tenant can never read another tenant's data."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
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
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, name TEXT, phone TEXT, email TEXT, source TEXT,
  item_id TEXT, status TEXT NOT NULL DEFAULT 'new', awaiting TEXT, offered TEXT DEFAULT '[]',
  budget_min INTEGER, budget_max INTEGER, timeline_days INTEGER, timeline_text TEXT,
  browsing INTEGER DEFAULT 0, prefs TEXT DEFAULT '{}', score INTEGER DEFAULT 0, qualified INTEGER DEFAULT 0,
  asked TEXT DEFAULT '{}', followups_sent INTEGER DEFAULT 0, opted_out INTEGER DEFAULT 0,
  summary TEXT DEFAULT '', flags TEXT DEFAULT '[]', created_at INTEGER NOT NULL,
  last_in_at INTEGER, last_out_at INTEGER, first_reply_ms INTEGER, last_angle TEXT, used_angles TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_leads_tenant ON leads(tenant_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_leads_phone ON leads(tenant_id, phone) WHERE phone IS NOT NULL;
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, lead_id TEXT NOT NULL,
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


def _row(r: sqlite3.Row | None) -> dict | None:
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
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")   # durable across app crashes; avoids an fsync per write
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, tuple(params))

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        with self.lock:
            return _row(self.conn.execute(sql, tuple(params)).fetchone())

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self.lock:
            return [_row(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

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


class _Tx:
    def __init__(self, db: Database):
        self.db = db

    def __enter__(self):
        self.db.lock.acquire()
        self.db.conn.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        try:
            self.db.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.db.lock.release()
