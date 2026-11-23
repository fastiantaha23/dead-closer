"""Info agent: answers questions strictly from tool data (inventory, tenant address). Anything it
does not have is escalated to the team instead of guessed."""
from __future__ import annotations

import re

from .. import tools
from .base import Turn

_UNKNOWN = re.compile(r"\b(?:hoa|pets?|schools?|taxes|utilities|mileage|miles on|warranty|carfax|accident|colou?r|insurance|commute|noise|crime|neighbou?rs?|age of|roof|furnace|appliances|history)\b", re.I)


def answer(t: Turn) -> int:
    """Append answer segments for the lead's questions. Returns how many were added."""
    intents, item = t.intents, t.item()
    before = len(t.segments)
    items = tools.list_items(t.db, t.tid)
    if "ask_price" in intents:
        if item:
            t.add("answer_price")
        elif items:
            t.add("answer_price", range=(items[0]["price"], items[-1]["price"]))
    if "ask_availability" in intents and item:
        t.add("answer_availability", available=bool(item.get("active", 1)))
    if "ask_location" in intents:
        t.add("answer_location")
    if "ask_features" in intents:
        if _UNKNOWN.search(t.text):
            t.add("unknown_detail")
            t.events.append({"event": "needs_human_answer", "text": t.text[:200]})
        else:
            t.add("answer_features")
    if "ask_demographics" in intents:
        t.add("fair_housing_decline")
        t.events.append({"event": "fair_housing_question_declined"})
    if "ask_promise" in intents:
        t.add("no_promises")
    if "ask_financing" in intents:
        t.add("answer_financing")
    if "ask_options" in intents:
        opts = tools.search_inventory(t.db, t.tid, t.lead.get("budget_max"), t.lead.get("budget_min"), t.lead.get("prefs"),
                                      limit=2, exclude=(item["id"],) if item else ())
        if opts:
            t.add("show_options", options=opts)
    if "question" in intents and len(t.segments) == before:
        t.add("unknown_detail")
        t.events.append({"event": "needs_human_answer", "text": t.text[:200]})
    return len(t.segments) - before
