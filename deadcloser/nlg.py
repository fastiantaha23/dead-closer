"""Deterministic message renderer. Turns a *plan* (list of typed segments produced by the agents)
into a short SMS-style text. It is both the offline "voice" and the guaranteed-safe fallback when
an LLM rewrite fails the output guard.

Segment text only ever contains facts present in the segment data (prices, slots, addresses come
from tools), so nothing here can invent a number."""
from __future__ import annotations

import zlib
from datetime import datetime

from .config import fmt_slot, from_epoch
from .safety import parse_money_amounts as _parse_amounts


def money(n: int | float) -> str:
    return f"${int(round(n)):,}"


def short_money(n: int | float) -> str:
    n = int(round(n))
    if n >= 1_000_000 and n % 100_000 == 0:
        return f"${n / 1_000_000:g}M"
    if n >= 10_000 and n % 1000 == 0:
        return f"${n // 1000}k"
    return money(n)


def pick(options: list[str], seed: str, key: str) -> str:
    return options[zlib.crc32(f"{seed}:{key}".encode()) % len(options)]


def slot_label(ts: int, tz: int) -> str:
    return fmt_slot(from_epoch(ts), tz)


def _item(rc: dict) -> str:
    it = rc.get("item")
    return f"the {it['title']}" if it else f"the {rc['thing']} you asked about"


def _slots_text(slots: list[int], tz: int) -> str:
    return "  ".join(f"{i}) {slot_label(s, tz)}" for i, s in enumerate(slots, 1))


def _facts(seg: dict, rc: dict) -> str:
    bits = []
    p = seg.get("prefs") or {}
    if p.get("bedrooms"):
        bits.append(f"{p['bedrooms']}-bedroom")
    for k in ("body", "fuel", "brand"):
        if p.get(k):
            bits.append(str(p[k]))
    if seg.get("budget"):
        suffix = "/month" if seg.get("budget_monthly") else ""
        bits.append(f"around {short_money(seg['budget'])}{suffix}")
    if seg.get("timeline"):
        bits.append(seg["timeline"])
    return ", ".join(bits)


def render_segment(seg: dict, rc: dict) -> str:  # noqa: C901 - flat dispatch is clearest here
    t, seed = seg["t"], rc["seed"]
    name = rc.get("name") or "there"
    agent, biz, noun, thing = rc["agent"], rc["biz"], rc["noun"], rc["thing"]
    tz = rc["tz"]
    item = _item(rc)
    re_ = rc["vertical"] == "real_estate"

    if t == "first_contact":
        if not rc.get("item"):
            return pick([f"Hi {name}, this is {agent}, {biz}'s virtual assistant. Thanks for reaching out!",
                         f"Hi {name}! {agent} here, {biz}'s virtual assistant. Thanks for your interest!"], seed, t)
        return pick([
            f"Hi {name}, this is {agent}, {biz}'s virtual assistant. Thanks for your interest in {item}!",
            f"Hi {name}! {agent} here, {biz}'s virtual assistant. Thanks for reaching out about {item}.",
        ], seed, t)
    if t == "returning":
        rec = seg.get("recall") or item
        return f"Hi {name}, welcome back! Last time you were looking at {rec}. Still searching?"
    if t == "ack":
        f = _facts(seg, rc)
        return pick([f"Got it - {f}.", f"Thanks, noted: {f}."], seed, t) if f else "Thanks, that helps."
    if t == "ask_timeline":
        return pick([
            "When are you hoping to move in?" if re_ else "When are you hoping to get behind the wheel?",
            "What's your timeline - are you hoping to decide this month, or further out?",
        ], seed, t)
    if t == "ask_budget":
        return pick([
            "What budget range are you working with? A rough number is perfect.",
            "Roughly what budget do you have in mind? That way I only show you things that fit.",
        ], seed, t)
    if t == "clarify_budget":
        v = seg["value"]
        if rc["vertical"] == "auto":
            return f"Just to be sure - did you mean {money(v)} per month, or {short_money(v * 1000)} in total?"
        return f"Just to be sure - did you mean {short_money(v * 1000)} in total?"
    if t == "budget_mismatch":
        return (f"Thanks for sharing. Our {thing}s currently start at {money(seg['lo'])}, which may be above your range. "
                f"Want to see the closest one, {seg['title']}?")
    if t == "propose_slots":
        slots = seg["slots"]
        lead_in = pick([
            f"I can set up a {noun} for {item}. Here are some open times:",
            f"Want to see it in person? I have these {noun} times open:",
        ], seed, t)
        return f"{lead_in} {_slots_text(slots, tz)}. Reply with the number (or tell me a day/time that suits you)."
    if t == "propose_slots_alt":
        return f"That time isn't open, but I have: {_slots_text(seg['slots'], tz)}. Which works best?"
    if t == "no_slots":
        return f"I don't see open {noun} times right now - I'll have the team reach out with options."
    if t == "ask_slot_pref":
        return f"No problem - what day and time would suit you best for a {noun}?"
    if t == "see_you":
        return "Great - we'll see you then!"
    if t == "clarify_slot":
        return "Which one works best for you? Just reply with the number."
    if t == "confirm_booking":
        when = slot_label(seg["slot"], tz)
        where = f", at {rc['address']}" if rc.get("address") else ""
        return (f"You're booked! {noun.capitalize()} for {item} on {when}{where}. "
                f"I'll send the details and a reminder beforehand. Need a change? Reply CHANGE, or CANCEL to cancel.")
    if t == "cancel_confirm":
        return f"Done - your {noun} on {slot_label(seg['slot'], tz)} is cancelled. Want me to find another time?"
    if t == "no_visit_to_cancel":
        return f"I don't see an upcoming {noun} for you. Want me to set one up?"
    if t == "answer_price":
        it = rc.get("item")
        if it:
            return f"The {it['title']} is listed at {money(it['price'])}."
        lo, hi = seg["range"]
        return f"Our {thing}s range from {money(lo)} to {money(hi)}."
    if t == "answer_availability":
        if seg.get("available"):
            return f"Yes, {item} is still available as of now."
        return f"I'm not able to confirm that {thing} is still available."
    if t == "answer_location":
        area = (rc.get("item") or {}).get("attrs", {}).get("area")
        base = f"The {rc['item']['title']} is in {area}. " if area else ""
        return f"{base}We're at {rc['address']}." if rc.get("address") else f"{base}I'll send the exact address with your confirmation."
    if t == "answer_features":
        a = (rc.get("item") or {}).get("attrs", {})
        if not a:
            return "Happy to share details once we know which one you like best."
        if re_:
            core = f"{a.get('beds')} bed / {a.get('baths')} bath, about {a.get('sqft'):,} sqft"
        else:
            core = f"{a.get('body', '')} - {a.get('fuel', '')}".strip(" -")
        feats = ", ".join(a.get("features", [])[:4])
        return f"{core}{', with ' + feats if feats else ''}."
    if t == "unknown_detail":
        return "I don't have that detail on hand, so I'll have the team confirm it for you rather than guess."
    if t == "confirm_cancel":
        return f"Just to confirm - do you want me to cancel your {noun} on {slot_label(seg['slot'], tz)}? Reply YES to cancel or NO to keep it."
    if t == "keep_visit":
        return f"No problem - your {noun} on {slot_label(seg['slot'], tz)} stays booked."
    if t == "confirm_slot":
        return f"Shall I book your {noun} for {item} on {slot_label(seg['slot'], tz)}? Reply YES to confirm."
    if t == "off_topic":
        return (f"I can only help with our {thing}s and booking {noun}s, so I can't help with that one. "
                f"Is there anything about {thing}s, prices or a {noun} I can do for you?")
    if t == "about_me":
        return (f"I'm {agent}, {biz}'s virtual assistant (an AI). I can answer questions about our {thing}s, check availability, "
                f"book or change a {noun}, and remember your preferences for next time.")
    if t == "resubscribed":
        return f"Welcome back - you're subscribed to messages from {biz} again. How can I help?"
    if t == "recall_memory":
        facts = seg.get("facts") or []
        if not facts:
            return "I don't have anything saved about you yet. Tell me your name, budget and timeline and I'll remember them for next time."
        return "Here's what I have on file for you: " + "; ".join(facts) + "."
    if t == "recap_chat":
        kind = seg.get("kind")
        if kind == "repeat" and seg.get("quote_agent"):
            return f"Sure - I said: \"{seg['quote_agent']}\""
        if kind == "first" and seg.get("quote_user"):
            return f"Your first message in this conversation was: \"{seg['quote_user']}\""
        if seg.get("quote_user"):
            return f"Just now you said: \"{seg['quote_user']}\""
        return "We've only just started this conversation."
    if t == "fair_housing_decline":
        return ("I'm not able to describe neighborhoods or the people who live there - that's best judged by visiting and "
                "checking public sources. I'm happy to share objective details about the home itself.")
    if t == "no_promises":
        return "I can't promise or predict prices, values or approvals, but the team can walk you through the facts and options."
    if t == "distress":
        return ("I'm really sorry you're going through this. If you're in immediate danger, please call your local emergency "
                "number. I'm asking a team member to reach out to you.")
    if t == "answer_financing":
        return (f"Our team can walk you through financing options during your {noun}. "
                "I can't quote rates or approvals over text.")
    if t == "show_options":
        opts = seg["options"]
        listing = "; ".join(f"{o['title']} ({money(o['price'])})" for o in opts)
        return f"Based on what you've shared, these may fit: {listing}."
    if t == "objection_price":
        alt = seg.get("alt")
        tail = (f"Within your range I'd look at {alt['title']} ({money(alt['price'])}). " if alt else
                "I'll keep an eye out for options closer to your range. ")
        return ("Totally fair - price matters. " + tail +
                "Pricing is set by the seller, so I can't promise discounts, but the team can talk through options in person.")
    if t == "objection_timing":
        when = seg.get("when") or "in a while"
        return f"No problem at all. I'll check back {when}. If anything changes sooner, just text me."
    if t == "objection_partner":
        return f"Makes sense - feel free to bring them along! Want me to line up a time that works for both of you?"
    if t == "soft_exit":
        if seg.get("wrong_number"):
            return "Sorry about that - I'll make sure we don't contact you again."
        return pick([f"Understood, thanks for letting me know. If anything changes, just message me anytime. Best of luck!",
                     f"No worries at all. I'll stop checking in - text me anytime if you'd like to pick this back up."], seed, t)
    if t == "optout":
        return f"You're unsubscribed and won't receive more messages from {biz}. Reply START to opt back in."
    if t == "handoff":
        return f"Absolutely - I'm letting {biz}'s team know so a real person can reach out to you directly."
    if t == "ai_disclosure":
        return (f"I'm {agent}, {biz}'s virtual assistant (an AI). I can answer questions and book {noun}s, "
                "and a person on the team is always available if you prefer.")
    if t == "deescalate":
        return ("I'm sorry about the frustration - that's not what I want. I'm asking a team member to reach out. "
                "If you'd rather not hear from us, reply STOP.")
    if t == "injection_deflect":
        return f"I can only help with questions about {thing}s and booking {noun}s - happy to do that whenever you're ready."
    if t == "pii_warning":
        return "For your security, please don't share card, ID or password details by text - I don't need them."
    if t == "reminder_24h":
        where = f", at {rc['address']}" if rc.get("address") else ""
        return (f"Reminder: your {noun} for {item} is tomorrow, {slot_label(seg['slot'], tz)}{where}. "
                "Reply CHANGE to reschedule or CANCEL to cancel.")
    if t == "reminder_2h":
        where = f", {rc['address']}" if rc.get("address") else ""
        return f"See you soon! Your {noun} for {item} is at {slot_label(seg['slot'], tz)}{where}. Reply if you're running late."
    if t == "post_booking_qualify_note":
        return "In the meantime"
    if t == "visit_summary_ready":
        return f"I'll have the details ready for you."
    # ---- follow-up angles
    if t == "fu_checkin":
        return pick([f"Hi {name}, just checking in on {item}.", f"Hi {name}, {agent} from {biz} - still interested in {item}?"], seed, t)
    if t == "fu_new_match":
        o = seg["option"]
        return f"Hi {name}, something new came in that fits what you described: {o['title']} ({money(o['price'])}). Want to take a look?"
    if t == "fu_value_add":
        return (f"Hi {name}, quick note: a {noun} usually takes under an hour and there's no obligation - "
                "it's the easiest way to know if it's right.")
    if t == "fu_question":
        return f"Hi {name}, any questions I can answer about {item}? Happy to help, or to line up a time whenever it suits you."
    if t == "fu_easy_choice":
        return f"Hi {name}, easiest next step: pick a time and I'll handle the rest. {_slots_text(seg['slots'], tz)}."
    if t == "fu_availability":
        return f"Hi {name}, a heads-up that {item} is still available as of today."
    if t == "fu_low_pressure":
        return f"Hi {name}, no pressure at all - if the timing's off just say so and I'll stop checking in."
    if t == "fu_breakup":
        return (f"Hi {name}, I haven't heard back so I'll assume the timing isn't right and stop reaching out. "
                "If you'd like to pick this up later, just text me anytime.")
    if t == "fu_revival":
        rec = seg.get("recall") or item
        return f"Hi {name}, it's {agent} from {biz}. It's been a while since you looked at {rec} - still searching? Happy to share what's new."
    if t == "fu_nurture":
        return f"Hi {name}, {agent} from {biz} - you mentioned the timing might work around now. Still interested in {item}?"
    raise ValueError(f"unknown segment {t}")


def render(plan: dict, rc: dict) -> str:
    parts = [render_segment(s, rc) for s in plan["segments"]]
    return " ".join(p.strip() for p in parts if p and p.strip())


def allowed_amounts(rc: dict, plan: dict, extra: set[int] | None = None) -> set[int]:
    """Dollar figures the guard will accept in a message: tenant inventory prices, the lead's own
    stated numbers, and anything a plan segment explicitly carries."""
    amts = set(rc.get("price_catalog", []))
    amts |= set(extra or set())
    for s in plan["segments"]:
        for q in ("quote_user", "quote_agent"):
            if s.get(q):
                amts |= _parse_amounts(s[q])
        amts |= set(s.get("amounts") or [])
        for k in ("budget", "value"):
            if isinstance(s.get(k), (int, float)):
                amts.add(int(s[k]))
                if k == "value":
                    amts.add(int(s[k]) * 1000)
    return amts
