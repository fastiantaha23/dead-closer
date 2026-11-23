"""Objection agent: price, timing, "need to ask my partner", already-has-agent, not-interested.
It never promises discounts and it exits gracefully - a polite no is final."""
from __future__ import annotations

from .. import tools
from .base import Turn


def _when(days: int | None) -> tuple[str, int]:
    """(spoken phrase, days until we check back)."""
    if not days:
        return "in a few weeks", 21
    back = max(7, days - 14)
    if back < 14:
        return "in a week or so", back
    if back < 45:
        return f"in about {round(back / 7)} weeks", back
    return f"in about {round(back / 30)} months", back


def handle(t: Turn) -> bool:
    """Handle terminal/steering objections. Returns True if the turn is fully handled (no steering needed)."""
    i = t.intents
    if "not_interested" in i or "has_agent" in i:
        t.add("soft_exit", wrong_number="wrong_number" in i)
        t.status = "lost"
        t.awaiting = None
        t.actions.append("closed_lost")
        t.lead_updates["flags"] = list({*(t.lead.get("flags") or []), "has_agent" if "has_agent" in i else "not_interested"})
        return True
    return False


def steering(t: Turn) -> bool:
    """Objections we answer and then keep going (return True if we added a segment)."""
    i = t.intents
    if "objection_price" in i:
        item = t.item()
        cap = t.lead.get("budget_max")
        cheaper = [x for x in tools.search_inventory(t.db, t.tid, cap, None, t.lead.get("prefs"), limit=5,
                                                     exclude=(item["id"],) if item else ())
                   if not item or x["price"] < item["price"]]
        t.add("objection_price", alt=cheaper[0] if cheaper else None)
        t.actions.append("objection_price")
        return True
    if "objection_timing" in i and "ask_price" not in i:
        phrase, back = _when(t.nlu.get("timeline_days") or t.lead.get("timeline_days"))
        t.add("objection_timing", when=phrase)
        t.nurture_days = back
        t.status = "nurture"
        t.awaiting = None
        t.actions.append("nurture_scheduled")
        return True
    if "need_partner" in i:
        t.add("objection_partner")
        t.awaiting = "interest"
        t.actions.append("partner_objection")
        return True
    return False
