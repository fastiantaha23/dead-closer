"""Closer / scheduling agent: proposes visit slots, resolves which slot the lead means, books
through the validated booking tool, handles reschedule + cancel."""
from __future__ import annotations

import re

from .. import memory, tools
from ..config import fmt_slot, from_epoch, local, to_epoch
from .base import Turn


def propose(t: Turn, pref: dict | None = None, alt: bool = False) -> None:
    slots = tools.available_slots(t.db, t.tenant, t.now, pref, n=3)
    if not slots:
        t.add("no_slots")
        t.status = "handoff"
        t.actions.append("no_slots_handoff")
        return
    t.offered, t.awaiting = slots, "slot"
    t.add("propose_slots_alt" if alt else "propose_slots", slots=slots)
    t.actions.append("propose_slots")


def _asked(t: Turn) -> dict:
    return dict(t.lead.get("asked") or {})


def _set_asked(t: Turn, a: dict) -> None:
    t.lead["asked"] = a
    t.lead_updates["asked"] = a


def _confirm_slot(t: Turn, slot_ts: int) -> bool:
    """A time the lead asked for that we did not offer: confirm before writing a booking."""
    t.offered, t.awaiting = [slot_ts], "slot"
    t.add("confirm_slot", slot=slot_ts)
    t.actions.append("confirm_slot_requested")
    if t.reschedule_of:
        a = _asked(t)
        a["pending_reschedule"] = t.reschedule_of
        _set_asked(t, a)
    return True


def _book(t: Turn, slot_ts: int) -> None:
    item = t.item()
    res = tools.book_visit(t.db, t.tenant, t.lead, item["id"] if item else None, slot_ts, t.now)
    if not res["ok"]:
        t.events.append({"event": "booking_failed", "error": res["error"]})
        propose(t, {"date": local(from_epoch(slot_ts), t.cfg["tz_offset_min"]).date().isoformat()}, alt=True)
        return
    visit = res["visit"]
    t.booked_visit, t.offered, t.awaiting, t.status = visit, [], None, "booked"
    a = _asked(t)
    if a.pop("pending_reschedule", None) is not None:
        _set_asked(t, a)
    t.add("confirm_booking", slot=slot_ts)
    t.actions.append("booked" if not res.get("duplicate") else "booked_duplicate")
    memory.remember(t.db, t.tid, t.lead["id"], "episode_visit",
                    f"{t.cfg['visit_noun']} booked for {fmt_slot(from_epoch(slot_ts), t.cfg['tz_offset_min'])}", "episode", to_epoch(t.now))
    if t.reschedule_of:
        old = t.db.one("SELECT * FROM visits WHERE tenant_id=? AND id=?", (t.tid, t.reschedule_of))
        if old and old["id"] != visit["id"]:
            tools.cancel_visit(t.db, t.tenant, old["id"], t.now)
            t.actions.append("rescheduled")


def handle_selection(t: Turn) -> bool:
    """Return True if the message was a slot choice / time request and was handled."""
    slot = t.nlu.get("slot") or {}
    offered = list(t.lead.get("offered") or [])
    awaiting = t.awaiting_before
    intents = t.intents
    if not t.reschedule_of and _asked(t).get("pending_reschedule"):
        t.reschedule_of = _asked(t)["pending_reschedule"]
    specific = bool(slot.get("date") and (slot.get("hour") is not None or slot.get("part")))
    chosen = None

    if slot.get("index") and awaiting in ("slot", "confirm_slot") and 1 <= slot["index"] <= len(offered):
        chosen = offered[slot["index"] - 1]
    elif awaiting == "slot" and "accept" in intents and not specific and not slot.get("index"):
        if len(offered) == 1:
            chosen = offered[0]
        elif len(offered) > 1:
            t.add("clarify_slot")
            t.offered, t.awaiting = offered, "slot"
            return True
    elif specific:
        # a specific time: prefer one we already offered, otherwise any exactly matching free slot
        tz = t.cfg["tz_offset_min"]
        for s in offered:
            l = local(from_epoch(s), tz)
            if l.date().isoformat() == slot["date"] and (slot.get("hour") is None or l.hour == slot["hour"]):
                chosen = s
                break
        if chosen is None:
            chosen = tools.exact_slot(t.db, t.tenant, t.now, slot)
            if chosen is None:
                propose(t, slot, alt=True)
                return True
            return _confirm_slot(t, chosen)
    elif (slot.get("date") or slot.get("part") or slot.get("hour") is not None or slot.get("weekend")) and (
            awaiting in ("slot", "interest") or "book_request" in intents or "reschedule" in intents):
        propose(t, slot)
        return True

    if chosen is not None:
        _book(t, chosen)
        return True
    return False


def handle_cancel(t: Turn) -> bool:
    """Cancelling is destructive: ask first, act only on an explicit YES."""
    visits = t.upcoming_visits()
    if not visits:
        t.add("no_visit_to_cancel")
        t.awaiting = "interest"
        return True
    t.add("confirm_cancel", slot=visits[0]["start_ts"])
    t.awaiting = "confirm_cancel"
    t.actions.append("confirm_cancel_requested")
    return True


def handle_cancel_reply(t: Turn) -> bool:
    visits = t.upcoming_visits()
    if not visits:
        return False
    v = visits[0]
    i = t.intents
    if "decline" in i or "not_interested" in i and "accept" not in i:
        t.add("keep_visit", slot=v["start_ts"])
        t.actions.append("cancel_declined")
        return True
    if "accept" not in i:
        return False
    if tools.cancel_visit(t.db, t.tenant, v["id"], t.now):
        t.cancelled_visit = v
        t.add("cancel_confirm", slot=v["start_ts"])
        t.awaiting = "interest"
        t.status = "engaged"
        t.actions.append("cancelled")
        memory.remember(t.db, t.tid, t.lead["id"], "episode_cancel",
                        f"cancelled {fmt_slot(from_epoch(v['start_ts']), t.cfg['tz_offset_min'])}", "episode", to_epoch(t.now))
    return True


def handle_reschedule(t: Turn) -> bool:
    visits = t.upcoming_visits()
    if not visits:
        return False
    t.reschedule_of = visits[0]["id"]
    slot = t.nlu.get("slot") or {}
    if slot.get("date") and (slot.get("hour") is not None or slot.get("part")):
        return handle_selection(t)
    propose(t, slot if (slot.get("date") or slot.get("part")) else None)
    return True
