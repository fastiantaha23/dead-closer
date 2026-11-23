"""Pay-per-booked-visit billing.

Lifecycle of a fee:  pending (visit booked) -> earned (visit time passed, not cancelled)
                                            -> void   (cancelled / duplicate / unqualified)

Incentive-alignment guard: a visit is only *billable* when the lead is genuinely qualified
(budget fits inventory and a realistic timeline). Unqualified bookings still happen - we never
refuse a real customer - but they earn nothing, so the agent has no reason to pad visits.
The UNIQUE(visit_id) constraint makes charging idempotent (no double billing).
"""
from __future__ import annotations

import sqlite3

from .db import Database


def charge_booking(db: Database, tenant: dict, visit_id: str, billable: bool, now: int) -> dict:
    fee = int(tenant["config"]["fee_cents"]) if billable else 0
    status = "pending" if billable else "void"
    note = "booked visit" if billable else "unqualified lead - not billable"
    try:
        db.execute("INSERT INTO ledger(tenant_id,visit_id,amount_cents,status,note,created_at) VALUES (?,?,?,?,?,?)",
                   (tenant["id"], visit_id, fee, status, note, now))
    except sqlite3.IntegrityError:
        pass  # idempotent
    return db.one("SELECT * FROM ledger WHERE tenant_id=? AND visit_id=?", (tenant["id"], visit_id))


def void_for_cancel(db: Database, tenant_id: str, visit_id: str, now: int, reason: str = "visit cancelled") -> None:
    db.execute("UPDATE ledger SET status='void', note=?, settled_at=? WHERE tenant_id=? AND visit_id=? AND status='pending'",
               (reason, now, tenant_id, visit_id))


def reconsider(db: Database, tenant: dict, lead_id: str, now: int) -> int:
    """The lead became qualified after booking: upgrade not-yet-held, unqualified (void) visits to billable."""
    rows = db.all(
        "SELECT l.id lid, v.id vid FROM ledger l JOIN visits v ON v.id=l.visit_id AND v.tenant_id=l.tenant_id "
        "WHERE l.tenant_id=? AND v.lead_id=? AND l.status='void' AND l.note LIKE 'unqualified%' "
        "AND v.status='booked' AND v.start_ts>?", (tenant["id"], lead_id, now))
    for r in rows:
        db.execute("UPDATE ledger SET status='pending', amount_cents=?, note='booked visit (lead qualified)' WHERE id=?",
                   (int(tenant["config"]["fee_cents"]), r["lid"]))
        db.execute("UPDATE visits SET billable=1 WHERE tenant_id=? AND id=?", (tenant["id"], r["vid"]))
    return len(rows)


def settle(db: Database, tenant: dict, now: int) -> int:
    """Move pending fees to earned once the visit time has passed. Returns number settled."""
    bill_no_show = bool(tenant["config"].get("bill_no_shows"))
    rows = db.all(
        "SELECT l.id lid, v.status vstatus FROM ledger l JOIN visits v ON v.id=l.visit_id AND v.tenant_id=l.tenant_id "
        "WHERE l.tenant_id=? AND l.status='pending' AND v.start_ts<=?", (tenant["id"], now))
    n = 0
    for r in rows:
        if r["vstatus"] == "cancelled":
            db.execute("UPDATE ledger SET status='void', settled_at=?, note='visit cancelled' WHERE id=?", (now, r["lid"]))
        elif r["vstatus"] == "no_show" and not bill_no_show:
            db.execute("UPDATE ledger SET status='void', settled_at=?, note='no-show not billed' WHERE id=?", (now, r["lid"]))
        else:
            db.execute("UPDATE ledger SET status='earned', settled_at=? WHERE id=?", (now, r["lid"]))
        n += 1
    return n


def summary(db: Database, tenant_id: str) -> dict:
    rows = db.all("SELECT status, COUNT(*) n, COALESCE(SUM(amount_cents),0) cents FROM ledger WHERE tenant_id=? GROUP BY status",
                  (tenant_id,))
    out = {"pending_cents": 0, "earned_cents": 0, "void_count": 0, "pending_count": 0, "earned_count": 0}
    for r in rows:
        if r["status"] == "pending":
            out["pending_cents"], out["pending_count"] = r["cents"], r["n"]
        elif r["status"] == "earned":
            out["earned_cents"], out["earned_count"] = r["cents"], r["n"]
        else:
            out["void_count"] = r["n"]
    out["total_cents"] = out["pending_cents"] + out["earned_cents"]
    return out
