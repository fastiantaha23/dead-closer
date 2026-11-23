"""Supervisor: the deterministic policy that decides what the agent does this turn and which
specialist agents run (qualifier, closer, objections, info, follow-up). Because decisions are plain
code, they are testable, benchmarkable and immune to prompt injection."""
from __future__ import annotations

from .. import memory, tools
from . import closer, followup, info, objections, qualifier
from .base import Turn


def _focus_item(t: Turn) -> None:
    """If the lead has stated needs but no specific item is in play, pick the best match."""
    if t.item():
        return
    lead = t.lead
    opts = tools.search_inventory(t.db, t.tid, lead.get("budget_max"), lead.get("budget_min"), lead.get("prefs"), limit=1)
    if opts and (lead.get("budget_max") or lead.get("prefs")):
        t.lead_updates["item_id"] = lead["item_id"] = opts[0]["id"]


def next_step(t: Turn, wants_visit: bool = False) -> None:
    if t.awaiting or t.status in ("lost", "handoff"):
        return
    visits = t.upcoming_visits()
    if visits and not t.booked_visit:
        q = qualifier.next_question(t)
        if q:
            t.add(f"ask_{q}")
            t.asked_inc.append(q)
            t.awaiting = q
        elif not t.segments:
            t.add("see_you")
        return
    if t.booked_visit:  # just booked: collect what's missing so the team is prepared (and the visit is billable)
        q = qualifier.next_question(t)
        if q:
            t.add(f"ask_{q}")
            t.asked_inc.append(q)
            t.awaiting = q
        return
    _focus_item(t)
    q = None if wants_visit else qualifier.next_question(t)
    if q:
        t.add(f"ask_{q}")
        t.asked_inc.append(q)
        t.awaiting = q
    else:
        closer.propose(t)


def _escalate(t: Turn) -> bool:
    f = t.flags
    if f.get("distress"):
        t.add("distress")
        t.status, t.actions = "handoff", t.actions + ["escalated_distress"]
    elif f.get("threat"):
        t.add("handoff")
        t.status, t.actions = "handoff", t.actions + ["escalated_legal"]
    elif f.get("abuse"):
        t.add("deescalate")
        t.status, t.actions = "handoff", t.actions + ["escalated_abuse"]
    elif f.get("human"):
        t.add("handoff")
        t.status, t.actions = "handoff", t.actions + ["escalated_human_request"]
    else:
        return False
    t.awaiting = None
    return True


def inbound(t: Turn) -> None:
    f, i = t.flags, t.intents
    pre: list[dict] = []
    if f.get("asks_bot"):
        pre.append({"t": "ai_disclosure"})
    if f.get("pii"):
        pre.append({"t": "pii_warning"})
    if _escalate(t):
        t.segments[:0] = pre
        return
    t.segments[:0] = pre

    if f.get("injection"):
        t.add("injection_deflect")
        t.actions.append("injection_deflected")
        next_step(t)
        return

    if "cancel_visit" in i and "reschedule" not in i:
        closer.handle_cancel(t)
        return
    if "reschedule" in i and closer.handle_reschedule(t):
        return
    if objections.handle(t):
        return
    if closer.handle_selection(t):
        _maybe_post_booking_question(t)
        return
    if t.awaiting_before == "interest" and "accept" in i:
        asked = dict(t.lead.get("asked") or {})
        pending = asked.pop("pending_item", None)
        if pending and tools.get_item(t.db, t.tid, pending):
            t.lead_updates["item_id"] = t.lead["item_id"] = pending
            t.lead_updates["asked"] = t.lead["asked"] = asked
        closer.propose(t)
        return
    if t.awaiting_before == "slot" and "decline" in i:
        t.add("ask_slot_pref")
        t.awaiting = "slot"
        return

    ack = qualifier.ack_segment(t)
    mismatch = qualifier.mismatch_segment(t)
    steered = objections.steering(t)
    n_ans = info.answer(t)
    if t.clarify_budget:
        t.segments = [s for s in t.segments if s["t"] not in ("ack",)]
        t.add("clarify_budget", value=t.clarify_budget)
        t.awaiting = "clarify_budget"
        return
    if mismatch:
        t.segments.append(mismatch)
        t.awaiting = "interest"
        return
    body = [s for s in t.segments if s not in pre]
    if ack and (len(body) + 2 <= 3 or not body):
        at = len(pre) + (1 if t.segments[len(pre):len(pre) + 1] and t.segments[len(pre)]["t"] in ("first_contact", "returning") else 0)
        t.segments.insert(at, ack)
    if steered and "objection_partner" in [s["t"] for s in t.segments]:
        return
    if t.status == "nurture":
        return
    wants_visit = "book_request" in i or ("ask_availability" in i)
    next_step(t, wants_visit=wants_visit)


def _maybe_post_booking_question(t: Turn) -> None:
    if t.booked_visit:
        q = qualifier.next_question(t)
        if q:
            t.add(f"ask_{q}")
            t.asked_inc.append(q)
            t.awaiting = q


def new_lead(t: Turn) -> None:
    t.add("first_contact")
    if t.text.strip():
        inbound(t)
        # inbound() may have added nothing actionable (e.g. bare greeting): make sure we ask something
        if len(t.segments) == 1:
            next_step(t)
    else:
        next_step(t)


def returning(t: Turn) -> None:
    mem = t.mem
    rec = None
    if "interested_in" in mem:
        it = tools.get_item(t.db, t.tid, mem["interested_in"]["value"])
        rec = it["title"] if it else None
    t.add("returning", recall=rec)
    if t.text.strip():
        inbound(t)
        if len(t.segments) == 1:
            next_step(t)
    else:
        next_step(t)


def reminder(t: Turn) -> None:
    vid = t.payload.get("visit_id")
    v = t.db.one("SELECT * FROM visits WHERE tenant_id=? AND id=? AND status='booked'", (t.tid, vid))
    if not v or v["start_ts"] <= int(t.now.timestamp()):
        t.send = False
        return
    t.lead["item_id"] = v["item_id"] or t.lead.get("item_id")
    t.add(t.trigger, slot=v["start_ts"])


def nurture(t: Turn) -> None:
    if t.lead["opted_out"] or t.lead["status"] in ("lost", "handoff", "opted_out") or t.upcoming_visits():
        t.send = False
        return
    t.add("fu_nurture")
    next_step(t)


def run_policy(t: Turn) -> None:
    trig = t.trigger
    if trig in ("reminder_24h", "reminder_2h"):
        reminder(t)
    elif trig == "followup":
        followup.plan(t)
    elif trig == "nurture":
        nurture(t)
    elif trig == "returning":
        returning(t)
    elif trig == "new_lead":
        new_lead(t)
    else:
        inbound(t)
    if t.asked_inc:
        asked = dict(t.lead.get("asked") or {})
        for k in t.asked_inc:
            asked[k] = asked.get(k, 0) + 1
        t.lead_updates["asked"] = asked
        t.lead["asked"] = asked
