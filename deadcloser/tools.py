"""Tools the agents can call. Every tool takes tenant_id explicitly and filters by it - the LLM never
supplies a tenant id, it comes from graph state, so cross-tenant access is structurally impossible."""
from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

from . import billing
from .config import fmt_slot, from_epoch, local, to_epoch
from .db import Database
from .safety import in_send_window, next_window_open


# ---------------------------------------------------------------------------- tool-call recorder
CALLS: ContextVar[list | None] = ContextVar("dc_tool_calls", default=None)


def record(name: str, args: dict, result: dict) -> None:
    """Append to the current request's tool-call log (returned by POST /chat as `tool_calls`)."""
    lst = CALLS.get()
    if lst is not None and len(lst) < 40:
        lst.append({"name": name, "args": args, "result": result})


# ---------------------------------------------------------------------------- records
def get_tenant(db: Database, tenant_id: str) -> dict | None:
    return db.one("SELECT * FROM tenants WHERE id=?", (tenant_id,))


def get_lead(db: Database, tenant_id: str, lead_id: str) -> dict | None:
    return db.one("SELECT * FROM leads WHERE tenant_id=? AND id=?", (tenant_id, lead_id))


def add_message(db: Database, tenant_id: str, lead_id: str, direction: str, body: str, ts: int, meta: dict | None = None,
                thread_id: str | None = None) -> int:
    return db.insert_id("INSERT INTO messages(tenant_id,lead_id,thread_id,direction,body,ts,meta) VALUES (?,?,?,?,?,?,?)",
                        (tenant_id, lead_id, thread_id, direction, body, ts, json.dumps(meta or {})))


def list_messages(db: Database, tenant_id: str, lead_id: str, limit: int = 200, thread_id: str | None = None) -> list[dict]:
    q, p = "SELECT id,direction,body,ts,meta,thread_id FROM messages WHERE tenant_id=? AND lead_id=?", [tenant_id, lead_id]
    if thread_id is not None:
        q += " AND thread_id=?"
        p.append(thread_id)
    return db.all(q + " ORDER BY id DESC LIMIT ?", (*p, limit))[::-1]


# ---------------------------------------------------------------------------- inventory
def list_items(db: Database, tenant_id: str) -> list[dict]:
    return db.all("SELECT id,title,price,attrs FROM items WHERE tenant_id=? AND active=1 ORDER BY price", (tenant_id,))


def get_item(db: Database, tenant_id: str, item_id: str | None) -> dict | None:
    if not item_id:
        return None
    return db.one("SELECT id,title,price,attrs,active FROM items WHERE tenant_id=? AND id=?", (tenant_id, item_id))


def _pref_score(item: dict, prefs: dict) -> int:
    a, s = item["attrs"], 0
    if prefs.get("bedrooms") is not None and a.get("beds") == prefs["bedrooms"]:
        s += 3
    elif prefs.get("bedrooms") is not None and a.get("beds") is not None and abs(a["beds"] - prefs["bedrooms"]) == 1:
        s += 1
    for k in ("body", "fuel", "brand", "area"):
        if prefs.get(k) and str(a.get(k, "")).lower() == str(prefs[k]).lower():
            s += 3
    for f in prefs.get("features", []):
        if any(f.lower() in x.lower() for x in a.get("features", [])):
            s += 1
    return s


def search_inventory(db: Database, tenant_id: str, budget_max: int | None, budget_min: int | None = None,
                     prefs: dict | None = None, limit: int = 3, exclude: tuple = ()) -> list[dict]:
    """Items within budget (5% tolerance), best preference match first, then closest to budget."""
    prefs = prefs or {}
    out = []
    for it in list_items(db, tenant_id):
        if it["id"] in exclude:
            continue
        if budget_max and it["price"] > budget_max * 1.05:
            continue
        if budget_min and it["price"] < budget_min * 0.8:
            continue
        out.append((_pref_score(it, prefs), -abs(it["price"] - (budget_max or it["price"])), it))
    out.sort(key=lambda x: (x[0], x[1]), reverse=True)
    res = [x[2] for x in out[:limit]]
    record("search_inventory", {"budget_max": budget_max, "prefs": prefs, "limit": limit}, {"matches": [i["title"] for i in res]})
    return res


def budget_fit(db: Database, tenant_id: str, budget_max: int | None) -> str:
    if not budget_max:
        return "unknown"
    items = list_items(db, tenant_id)
    if not items:
        return "unknown"
    cheapest = items[0]["price"]
    if cheapest <= budget_max * 1.05:
        return "fit"
    if cheapest <= budget_max * 1.20:
        return "stretch"
    return "none"


def compute_qualification(db: Database, tenant_id: str, lead: dict) -> tuple[int, bool, str]:
    fit = budget_fit(db, tenant_id, lead.get("budget_max"))
    days = lead.get("timeline_days")
    browsing = bool(lead.get("browsing"))
    score = 0
    if days is not None and not browsing:
        score += 40 if days <= 14 else 32 if days <= 30 else 22 if days <= 60 else 14 if days <= 90 else 6
    score += {"fit": 40, "stretch": 20}.get(fit, 0)
    if lead.get("last_in_at"):
        score += 10
    if lead.get("status") in ("booked", "qualified"):
        score += 10
    qualified = fit in ("fit", "stretch") and days is not None and days <= 90 and not browsing
    return min(100, score), qualified, fit


# ---------------------------------------------------------------------------- slots & visits
def generate_slots(tenant: dict, now: datetime, horizon_days: int = 10) -> list[int]:
    cfg = tenant["config"]
    tz = cfg["tz_offset_min"]
    open_h, close_h = cfg["visit_hours"]
    step = int(cfg["slot_minutes"])
    earliest = now + timedelta(hours=cfg["min_notice_hours"])
    lnow = local(now, tz)
    out = []
    for d in range(horizon_days + 1):
        day = (lnow + timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
        if day.weekday() not in cfg["visit_days"]:
            continue
        minute = open_h * 60
        while minute + step <= close_h * 60:
            slot = day + timedelta(minutes=minute)
            if slot >= earliest:
                out.append(to_epoch(slot))
            minute += step
    return out


def slot_taken(db: Database, tenant: dict, ts: int) -> bool:
    row = db.one("SELECT COUNT(*) n FROM visits WHERE tenant_id=? AND start_ts=? AND status='booked'", (tenant["id"], ts))
    return row["n"] >= int(tenant["config"]["slot_capacity"])


def available_slots(db: Database, tenant: dict, now: datetime, pref: dict | None = None, n: int = 3) -> list[int]:
    """Up to n open slots. With a preference (date/hour/part) return the closest matches first,
    otherwise a spread over the next days (earliest, midday, afternoon) so the lead can choose."""
    tz = tenant["config"]["tz_offset_min"]
    free = [s for s in generate_slots(tenant, now) if not slot_taken(db, tenant, s)]
    if not free:
        record("check_availability", {"pref": pref}, {"slots": []})
        return []
    pref = pref or {}
    want_date, want_hour, part = pref.get("date"), pref.get("hour"), pref.get("part")
    if want_hour is None and part:
        want_hour = {"morning": 10, "afternoon": 14, "evening": 17}[part]
    if want_date or want_hour is not None or pref.get("weekend"):
        def dist(ts: int) -> tuple:
            l = local(from_epoch(ts), tz)
            dd = abs((l.date() - datetime.fromisoformat(want_date).date()).days) if want_date else 0
            if pref.get("weekend") and not want_date and l.weekday() < 5:
                dd += 5
            hd = abs(l.hour - want_hour) if want_hour is not None else 0
            return (dd, hd, ts)
        res = sorted(sorted(free, key=dist)[:n])
        record("check_availability", {"pref": pref}, {"slots": [fmt_slot(from_epoch(s), tz) for s in res]})
        return res
    by_day: dict = {}
    for s in free:
        by_day.setdefault(local(from_epoch(s), tz).date(), []).append(s)
    targets = [None, 11, 15]
    picks: list[int] = []
    for i, (_, slots) in enumerate(list(by_day.items())[:n]):
        tgt = targets[i % 3]
        picks.append(slots[0] if tgt is None else min(slots, key=lambda s: abs(local(from_epoch(s), tz).hour - tgt)))
    if len(picks) < n:  # not enough distinct days: fill from remaining slots of the first day
        for s in free:
            if s not in picks:
                picks.append(s)
            if len(picks) == n:
                break
    res = sorted(picks)
    record("check_availability", {"pref": None}, {"slots": [fmt_slot(from_epoch(s), tz) for s in res]})
    return res


def exact_slot(db: Database, tenant: dict, now: datetime, pref: dict) -> int | None:
    """A free slot that exactly satisfies date + hour (or date + part)."""
    tz = tenant["config"]["tz_offset_min"]
    if not pref.get("date") or (pref.get("hour") is None and not pref.get("part")):
        return None
    for s in generate_slots(tenant, now):
        l = local(from_epoch(s), tz)
        if l.date().isoformat() != pref["date"] or slot_taken(db, tenant, s):
            continue
        if pref.get("hour") is not None and l.hour == pref["hour"]:
            return s
        if pref.get("hour") is None:
            lo, hi = {"morning": (0, 12), "afternoon": (12, 17), "evening": (17, 24)}[pref["part"]]
            if lo <= l.hour < hi:
                return s
    return None


def active_visits(db: Database, tenant_id: str, lead_id: str, now: int | None = None) -> list[dict]:
    q = "SELECT * FROM visits WHERE tenant_id=? AND lead_id=? AND status='booked'"
    params: list = [tenant_id, lead_id]
    if now is not None:
        q += " AND start_ts>=?"
        params.append(now)
    return db.all(q + " ORDER BY start_ts", params)


def book_visit(db: Database, tenant: dict, lead: dict, item_id: str | None, slot_ts: int, now: datetime) -> dict:
    """Validated, atomic, idempotent booking. Returns {ok, visit?, error?}."""
    tid = tenant["id"]
    if slot_ts not in set(generate_slots(tenant, now)):
        return {"ok": False, "error": "slot_not_offered"}
    if item_id and not get_item(db, tid, item_id):
        return {"ok": False, "error": "unknown_item"}
    with db.transaction():
        if slot_taken(db, tenant, slot_ts):
            return {"ok": False, "error": "slot_taken"}
        mine = active_visits(db, tid, lead["id"], to_epoch(now))
        if any(v["start_ts"] == slot_ts for v in mine):
            return {"ok": True, "visit": mine[0], "duplicate": True}
        if len(mine) >= 2:
            return {"ok": False, "error": "too_many_active_visits"}
        _, qualified, _ = compute_qualification(db, tid, lead)
        vid = "v_" + uuid.uuid4().hex[:10]
        db.execute("INSERT INTO visits(id,tenant_id,lead_id,item_id,start_ts,status,billable,created_at) VALUES (?,?,?,?,?,'booked',?,?)",
                   (vid, tid, lead["id"], item_id, slot_ts, 1 if qualified else 0, to_epoch(now)))
        billing.charge_booking(db, tenant, vid, qualified, to_epoch(now))
    record("book_visit", {"item_id": item_id, "slot": fmt_slot(from_epoch(slot_ts), tenant["config"]["tz_offset_min"])},
           {"ok": True, "visit_id": vid, "billable": qualified})
    return {"ok": True, "visit": db.one("SELECT * FROM visits WHERE id=?", (vid,)), "billable": qualified}


def cancel_visit(db: Database, tenant: dict, visit_id: str, now: datetime) -> bool:
    v = db.one("SELECT * FROM visits WHERE tenant_id=? AND id=? AND status='booked'", (tenant["id"], visit_id))
    if not v:
        return False
    db.update("visits", tenant["id"], visit_id, status="cancelled")
    record("cancel_visit", {"visit_id": visit_id}, {"ok": True})
    billing.void_for_cancel(db, tenant["id"], visit_id, to_epoch(now))
    db.execute("UPDATE jobs SET status='cancelled' WHERE tenant_id=? AND lead_id=? AND kind LIKE 'reminder%' AND payload LIKE ?",
               (tenant["id"], v["lead_id"], f'%"{visit_id}"%'))
    return True


# ---------------------------------------------------------------------------- jobs
def cancel_jobs(db: Database, tenant_id: str, lead_id: str, kinds: tuple[str, ...]) -> None:
    q = ",".join("?" * len(kinds))
    db.execute(f"UPDATE jobs SET status='cancelled' WHERE tenant_id=? AND lead_id=? AND status='pending' AND kind IN ({q})",
               (tenant_id, lead_id, *kinds))


def schedule_job(db: Database, tenant_id: str, lead_id: str, kind: str, run_at: datetime, payload: dict | None = None) -> None:
    db.execute("INSERT INTO jobs(tenant_id,lead_id,kind,run_at,payload) VALUES (?,?,?,?,?)",
               (tenant_id, lead_id, kind, to_epoch(run_at), json.dumps(payload or {})))


def schedule_followup(db: Database, tenant: dict, lead: dict, now: datetime) -> datetime | None:
    """Queue the next unanswered-lead follow-up, respecting cadence, quiet hours and the max-attempt cap."""
    cfg = tenant["config"]
    gaps = cfg["followup_gaps_h"]
    n = lead["followups_sent"]
    cancel_jobs(db, tenant["id"], lead["id"], ("followup",))
    if lead["opted_out"] or lead["status"] in ("lost", "opted_out", "handoff", "dormant"):
        return None
    if n >= len(gaps):
        return None
    when = now + timedelta(hours=gaps[n])
    when = next_window_open(when, cfg["tz_offset_min"], tuple(cfg["send_window"]))
    schedule_job(db, tenant["id"], lead["id"], "followup", when, {"attempt": n + 1})
    return when


def schedule_reminders(db: Database, tenant: dict, lead_id: str, visit: dict, now: datetime) -> None:
    cfg = tenant["config"]
    start = from_epoch(visit["start_ts"])
    for kind, hrs in (("reminder_24h", 24), ("reminder_2h", 2)):
        when = start - timedelta(hours=hrs)
        if when <= now + timedelta(minutes=30):
            continue
        if not in_send_window(when, cfg["tz_offset_min"], tuple(cfg["send_window"])):
            when = next_window_open(when, cfg["tz_offset_min"], tuple(cfg["send_window"]))
            if when >= start - timedelta(hours=1):
                continue
        schedule_job(db, tenant["id"], lead_id, kind, when, {"visit_id": visit["id"]})


def due_jobs(db: Database, now: datetime, tenant_id: str | None = None, lead_id: str | None = None) -> list[dict]:
    q = "SELECT * FROM jobs WHERE status='pending' AND run_at<=?"
    params: list = [to_epoch(now)]
    if tenant_id:
        q += " AND tenant_id=?"
        params.append(tenant_id)
    if lead_id:
        q += " AND lead_id=?"
        params.append(lead_id)
    return db.all(q + " ORDER BY run_at LIMIT 1", params)
