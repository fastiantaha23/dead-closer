"""Qualifier agent: learns budget + timeline (+ preferences), scores the lead, and decides the next
qualifying question - at most one question per message and at most two asks per field, so we never
interrogate. Facts are written to long-term memory."""
from __future__ import annotations

import re
import zlib

from .. import billing, memory, nlu, tools
from ..config import to_epoch
from .base import Turn
from ..nlg import money


def absorb(t: Turn) -> None:
    """Fold the NLU result into lead state + long-term memory, then re-score."""
    n, lead, now = t.nlu, t.lead, to_epoch(t.now)
    up: dict = {}
    asked = dict(lead.get("asked") or {})

    if n.get("name") and not lead.get("name"):
        up["name"] = n["name"]
        memory.remember(t.db, t.tid, lead["id"], "name", n["name"], "rapport", now)

    # --- budget (with a clarification round-trip for ambiguous bare numbers)
    pending = asked.get("pending_budget")
    if t.awaiting_before == "clarify_budget" and pending:
        said_month = "month" in t.text.lower() or "/mo" in t.text.lower()
        if "decline" not in t.intents and (("accept" in t.intents) or "total" in t.text.lower() or said_month):
            if said_month:
                val = pending * t.cfg["monthly_factor"]
                memory.remember(t.db, t.tid, lead["id"], "budget_monthly", pending, "fact", now)
            else:
                val = pending * 1000
            up["budget_max"] = int(val)
        asked.pop("pending_budget", None)
    elif n.get("budget_ambiguous") and (n.get("budget_max") or n.get("budget_min")):
        asked["pending_budget"] = n.get("budget_max") or n.get("budget_min")
        t.clarify_budget = asked["pending_budget"]
    elif n.get("budget_max") or n.get("budget_min"):
        if n.get("budget_max"):
            up["budget_max"] = n["budget_max"]
        if n.get("budget_min"):
            up["budget_min"] = n["budget_min"]
        if n.get("budget_monthly"):
            memory.remember(t.db, t.tid, lead["id"], "budget_monthly", "yes", "fact", now)
    if "budget_max" in up:
        memory.remember(t.db, t.tid, lead["id"], "budget_max", up["budget_max"], "fact", now)

    # --- timeline
    if n.get("timeline_days") is not None:
        up["timeline_days"], up["timeline_text"], up["browsing"] = n["timeline_days"], n.get("timeline_text"), 1 if n.get("browsing") else 0
        memory.remember(t.db, t.tid, lead["id"], "timeline", n.get("timeline_text") or f"{n['timeline_days']}d", "fact", now)

    # --- preferences
    if n.get("prefs"):
        merged = {**(lead.get("prefs") or {}), **n["prefs"]}
        up["prefs"] = merged
        for k, v in n["prefs"].items():
            memory.remember(t.db, t.tid, lead["id"], f"pref_{k}", v, "preference", now)
    if n.get("item_id") and n["item_id"] != lead.get("item_id"):
        up["item_id"] = n["item_id"]
        memory.remember(t.db, t.tid, lead["id"], "interested_in", n["item_id"], "preference", now)

    _capture_notes(t, now)
    up["asked"] = asked
    lead.update(up)
    t.lead_updates.update(up)

    score, qualified, fit = tools.compute_qualification(t.db, t.tid, lead)
    t.lead_updates.update({"score": score, "qualified": 1 if qualified else 0})
    lead.update({"score": score, "qualified": 1 if qualified else 0})
    t.budget_fit = fit
    if qualified and lead["status"] in ("new", "engaged"):
        t.status = "qualified"
    if qualified:  # a visit booked before we knew the lead was real becomes billable now
        billing.reconsider(t.db, t.tenant, lead["id"], now)


def _capture_notes(t: Turn, now: int) -> None:
    """Keep a few free-text self-descriptions ("I have two kids", "I work nights") as long-term notes."""
    if not t.text or t.flags.get("injection") or t.flags.get("pii") or t.trigger not in ("inbound", "new_lead", "returning"):
        return
    existing = [k for k in (t.mem or {}) if k.startswith("note_")]
    for sent in re.split(r"[.!?\n;]+", t.text):
        s = sent.strip()
        if not 12 <= len(s) <= 140 or "?" in s or len(existing) >= 8:
            continue
        if not nlu.SELF_INFO_RE.search(s) or nlu.NAME_RE.search(s) or nlu.parse_budget(s, None)["budget_max"] or nlu.parse_timeline(s, t.now)[0] is not None:
            continue
        if nlu.BED_RE.search(s) or nlu.OFFTOPIC_RE.search(s):
            continue
        key = "note_" + format(zlib.crc32(s.lower().encode()) & 0xFFFFFF, "06x")
        if key not in existing:
            memory.remember(t.db, t.tid, t.lead["id"], key, s, "note", now)
            existing.append(key)


def known(lead: dict) -> tuple[bool, bool]:
    return bool(lead.get("budget_max") or lead.get("budget_min")), lead.get("timeline_days") is not None


def next_question(t: Turn) -> str | None:
    """Return 'timeline' | 'budget' | None (= enough info, go to scheduling)."""
    has_budget, has_timeline = known(t.lead)
    asked = t.lead.get("asked") or {}
    if not has_timeline and asked.get("timeline", 0) < 2:
        return "timeline"
    if not has_budget and asked.get("budget", 0) < 2:
        return "budget"
    return None


def ack_segment(t: Turn) -> dict | None:
    n = t.nlu
    prefs = n.get("prefs") or {}
    budget = n.get("budget_max") or n.get("budget_min")
    if n.get("budget_ambiguous"):
        budget = None
    monthly = bool(n.get("budget_monthly"))
    if budget and monthly:
        budget = int(budget / max(1, t.cfg["monthly_factor"]))
    timeline = n.get("timeline_text") if n.get("timeline_days") is not None else None
    facts = {k: v for k, v in prefs.items() if k in ("bedrooms", "body", "fuel", "brand")}
    if not (budget or timeline or facts):
        return None
    return {"t": "ack", "budget": budget, "budget_monthly": monthly, "timeline": timeline, "prefs": facts}


def mismatch_segment(t: Turn) -> dict | None:
    """Honest handling when the stated budget is well below everything we have."""
    if getattr(t, "budget_fit", None) != "none":
        return None
    items = tools.list_items(t.db, t.tid)
    if not items:
        return None
    t.lead_updates["item_id"] = t.lead["item_id"] = items[0]["id"]
    return {"t": "budget_mismatch", "lo": items[0]["price"], "title": items[0]["title"]}
