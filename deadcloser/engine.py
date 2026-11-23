"""DeadCloser: the application façade used by the API, the benchmark and the tests."""
from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime

from . import billing, memory, seed, tools
from .config import Settings, from_epoch, to_epoch, utcnow
from .db import Database
from .graph import Brain
from .llm import RuleLLM, make_llm

_PHONE_RE = re.compile(r"^\+?[\d\-\s\(\)]{7,20}$")
_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


class DeadCloser:
    def __init__(self, settings: Settings | None = None, db: Database | None = None, llm: RuleLLM | None = None,
                 checkpoints: str | None = None):
        self.settings = settings or Settings()
        self.db = db or Database(self.settings.db_path)
        self.llm = llm or make_llm(self.settings.llm_backend, self.settings.model, self.settings.llm_timeout_s)
        cp = checkpoints if checkpoints is not None else self.settings.checkpoint_path
        self.brain = Brain(self.db, self.llm, self._checkpointer(cp))
        self._locks: dict[tuple, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    @staticmethod
    def _checkpointer(path: str):
        if path == ":memory:":
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
        import os
        from langgraph.checkpoint.sqlite import SqliteSaver
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return SqliteSaver(conn)

    def _lead_lock(self, tenant_id: str, lead_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[(tenant_id, lead_id)]

    # ------------------------------------------------------------------ tenants
    def create_tenant(self, name: str, vertical: str, now: datetime | None = None, **kw) -> tuple[str, str]:
        return seed.create_tenant(self.db, name, vertical, to_epoch(now or utcnow()), **kw)

    def authenticate(self, api_key: str) -> dict | None:
        if not api_key:
            return None
        return self.db.one("SELECT * FROM tenants WHERE api_key_hash=?", (seed.hash_key(api_key),))

    # ------------------------------------------------------------------ leads
    def new_lead(self, tenant_id: str, *, name: str | None = None, phone: str | None = None, email: str | None = None,
                 source: str = "web_form", message: str = "", item_id: str | None = None, now: datetime | None = None) -> dict:
        """A new lead arrived. Replies in-line (typically in milliseconds; the SLA is 60 s)."""
        now = now or utcnow()
        tenant = tools.get_tenant(self.db, tenant_id)
        if not tenant:
            raise LookupError("unknown tenant")
        name = (name or "").strip()[:60] or None
        if name and not re.fullmatch(r"[\w .'\-]{1,60}", name):
            name = None
        phone = phone.strip() if phone and _PHONE_RE.match(phone.strip()) else None
        email = email.strip() if email and _EMAIL_RE.match(email.strip()) else None
        if item_id and not tools.get_item(self.db, tenant_id, item_id):
            item_id = None
        message = (message or "")[:2000]
        t0 = time.perf_counter()

        existing = self.db.one("SELECT * FROM leads WHERE tenant_id=? AND phone=?", (tenant_id, phone)) if phone else None
        if existing:
            lead = existing
            if lead["opted_out"]:
                self.db.audit(tenant_id, lead["id"], to_epoch(now), "suppressed_opted_out_resubmission")
                return {"lead_id": lead["id"], "sent": False, "reason": "opted_out", "latency_ms": 0}
            recent = lead["last_out_at"] and to_epoch(now) - lead["last_out_at"] < 7 * 86400 and lead["status"] not in ("lost", "dormant")
            with self._lead_lock(tenant_id, lead["id"]):
                if recent:  # duplicate form submission: treat as a normal message, never double-greet
                    out = self.brain.run(tenant_id, lead["id"], "inbound", message, now) if message else {"sent": False, "reason": "duplicate"}
                else:
                    out = self.brain.run(tenant_id, lead["id"], "returning", message, now)
            return self._finish(tenant_id, lead["id"], out, t0, first=False)

        lead_id = "l_" + uuid.uuid4().hex[:10]
        self.db.execute(
            "INSERT INTO leads(id,tenant_id,name,phone,email,source,item_id,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (lead_id, tenant_id, name, phone, email, (source or "")[:40], item_id, "new", to_epoch(now)))
        if name:
            memory.remember(self.db, tenant_id, lead_id, "name", name, "rapport", to_epoch(now))
        if item_id:
            memory.remember(self.db, tenant_id, lead_id, "interested_in", item_id, "preference", to_epoch(now))
        with self._lead_lock(tenant_id, lead_id):
            out = self.brain.run(tenant_id, lead_id, "new_lead", message, now)
        return self._finish(tenant_id, lead_id, out, t0, first=True)

    def _finish(self, tenant_id: str, lead_id: str, out: dict, t0: float, first: bool) -> dict:
        ms = int((time.perf_counter() - t0) * 1000)
        if first and out.get("sent"):
            self.db.update("leads", tenant_id, lead_id, first_reply_ms=ms)
        return {"lead_id": lead_id, "latency_ms": ms, "within_sla": ms <= self.settings.first_reply_sla_s * 1000, **out}

    def inbound(self, tenant_id: str, lead_id: str, text: str, now: datetime | None = None) -> dict:
        now = now or utcnow()
        if not tools.get_lead(self.db, tenant_id, lead_id):
            raise LookupError("unknown lead")
        t0 = time.perf_counter()
        with self._lead_lock(tenant_id, lead_id):
            out = self.brain.run(tenant_id, lead_id, "inbound", (text or "")[:2000], now)
        return self._finish(tenant_id, lead_id, out, t0, first=False)

    # ------------------------------------------------------------------ scheduler
    def tick(self, now: datetime | None = None, tenant_id: str | None = None, max_jobs: int = 10_000,
             lead_id: str | None = None) -> list[dict]:
        """Run every due job (follow-ups, reminders, nurture) in chronological order up to `now`.
        `lead_id` restricts the run to one lead (used by per-session virtual clocks in the demo)."""
        now = now or utcnow()
        results = []
        for _ in range(max_jobs):
            jobs = tools.due_jobs(self.db, now, tenant_id, lead_id)
            if not jobs:
                break
            job = jobs[0]
            claimed = self.db.execute("UPDATE jobs SET status='running' WHERE id=? AND status='pending'", (job["id"],))
            if claimed.rowcount != 1:
                continue
            at = from_epoch(job["run_at"])
            lead = tools.get_lead(self.db, job["tenant_id"], job["lead_id"])
            if lead:
                try:
                    with self._lead_lock(job["tenant_id"], job["lead_id"]):
                        out = self.brain.run(job["tenant_id"], job["lead_id"], job["kind"], "", at, job["payload"])
                    results.append({"job": job["kind"], "lead_id": job["lead_id"], **out})
                except Exception as e:  # one bad job must not stop the scheduler
                    self.db.audit(job["tenant_id"], job["lead_id"], to_epoch(at), "job_error", error=str(e)[:300])
            self.db.execute("UPDATE jobs SET status='done' WHERE id=? AND status='running'", (job["id"],))
        for t in self.db.all("SELECT * FROM tenants" + (" WHERE id=?" if tenant_id else ""), (tenant_id,) if tenant_id else ()):
            billing.settle(self.db, t, to_epoch(now))
        return results

    # ------------------------------------------------------------------ views
    def lead_view(self, tenant_id: str, lead_id: str) -> dict | None:
        lead = tools.get_lead(self.db, tenant_id, lead_id)
        if not lead:
            return None
        lead.pop("api_key_hash", None)
        return {
            "lead": lead,
            "messages": tools.list_messages(self.db, tenant_id, lead_id),
            "memory": memory.recall(self.db, tenant_id, lead_id),
            "visits": self.db.all("SELECT * FROM visits WHERE tenant_id=? AND lead_id=? ORDER BY start_ts", (tenant_id, lead_id)),
            "jobs": self.db.all("SELECT id,kind,run_at,status FROM jobs WHERE tenant_id=? AND lead_id=? ORDER BY run_at", (tenant_id, lead_id)),
        }

    def stats(self, tenant_id: str) -> dict:
        leads = self.db.all("SELECT status, COUNT(*) n FROM leads WHERE tenant_id=? GROUP BY status", (tenant_id,))
        total = sum(r["n"] for r in leads)
        visits = self.db.one("SELECT COUNT(*) n, SUM(billable) b FROM visits WHERE tenant_id=? AND status!='cancelled'", (tenant_id,))
        lat = self.db.all("SELECT first_reply_ms FROM leads WHERE tenant_id=? AND first_reply_ms IS NOT NULL", (tenant_id,))
        ms = sorted(r["first_reply_ms"] for r in lat)
        return {
            "leads_total": total, "by_status": {r["status"]: r["n"] for r in leads},
            "visits_booked": visits["n"] or 0, "visits_billable": visits["b"] or 0,
            "booking_rate": round((visits["n"] or 0) / total, 3) if total else 0,
            "first_reply_ms_p50": ms[len(ms) // 2] if ms else None,
            "first_reply_ms_p95": ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else None,
            "billing": billing.summary(self.db, tenant_id),
            "playbook": memory.playbook(self.db, tenant_id),
            "llm": self.llm.name,
        }
