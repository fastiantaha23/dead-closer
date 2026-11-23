"""Info agent: answers questions strictly from tool data (inventory, tenant address). Anything it
does not have is escalated to the team instead of guessed."""
from __future__ import annotations

import re

from .. import memory, tools
from .base import Turn

_UNKNOWN = re.compile(r"\b(?:hoa|pets?|schools?|taxes|utilities|mileage|miles on|warranty|carfax|accident|colou?r|insurance|commute|noise|crime|neighbou?rs?|age of|roof|furnace|appliances|history)\b", re.I)


def _clean_quote(s: str) -> str:
    s = re.sub(r"https?://\S+|www\.\S+", "[link]", s)
    s = re.sub(r"\d{7,}", "[number]", s).replace('"', "'")
    return s[:140]


def recall_facts(t: Turn) -> tuple[list[str], list[int]]:
    from .. import nlg
    lead, facts, amounts = t.lead, [], []
    if lead.get("name"):
        facts.append(f"your name is {lead['name']}")
    if lead.get("budget_max"):
        facts.append(f"budget up to {nlg.money(lead['budget_max'])}")
        amounts.append(int(lead["budget_max"]))
    if lead.get("timeline_days") is not None:
        facts.append("timeline: " + ("no firm timeline" if lead.get("browsing") else (lead.get("timeline_text") or f"{lead['timeline_days']} days")))
    p = lead.get("prefs") or {}
    bits = []
    if p.get("bedrooms"):
        bits.append(f"{p['bedrooms']} bedrooms")
    bits += [str(p[k]) for k in ("body", "fuel", "brand", "area") if p.get(k)]
    bits += [str(f) for f in p.get("features", [])]
    if bits:
        facts.append("preferences: " + ", ".join(bits))
    item = tools.get_item(t.db, t.tid, lead.get("item_id"))
    if item:
        facts.append(f"interested in the {item['title']}")
    for v in tools.active_visits(t.db, t.tid, lead["id"], int(t.now.timestamp()))[:2]:
        from ..config import fmt_slot, from_epoch
        facts.append(f"{t.cfg['visit_noun']} booked for {fmt_slot(from_epoch(v['start_ts']), t.cfg['tz_offset_min'])}")
    mem = memory.recall(t.db, t.tid, lead["id"])
    for k, m in mem.items():
        if k.startswith("note_"):
            facts.append(f'you told me "{_clean_quote(m["value"])}"')
    return facts, amounts


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
    if "ask_about_bot" in intents:
        t.add("about_me")
    if "ask_memory" in intents:
        facts, amounts = recall_facts(t)
        tools.record("recall_memory", {}, {"facts": len(facts)})
        t.add("recall_memory", facts=facts, amounts=amounts)
    if "ask_recap" in intents:
        msgs = tools.list_messages(t.db, t.tid, t.lead["id"], 40, thread_id=t.thread_id)
        lead_msgs = [m["body"] for m in msgs if m["direction"] == "in"]
        agent_msgs = [m["body"] for m in msgs if m["direction"] == "out"]
        low = t.text.lower()
        if "first" in low and lead_msgs:
            t.add("recap_chat", kind="first", quote_user=_clean_quote(lead_msgs[0]))
        elif re.search(r"repeat|again|what did you|what have you|what we", low) and agent_msgs:
            t.add("recap_chat", kind="repeat", quote_agent=_clean_quote(agent_msgs[-1]))
        else:
            prev = lead_msgs[-2] if len(lead_msgs) >= 2 else None
            t.add("recap_chat", kind="last", quote_user=_clean_quote(prev) if prev else None)
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
