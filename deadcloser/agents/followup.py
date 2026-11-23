"""Follow-up agent: persistent but respectful. Cadence is ~3h, 1d, 3d, 7d, 14d, 28d for an unanswered lead
(the gaps live in tenant config). Each touch uses a different angle; angles 2-4 are chosen by UCB1 from the
tenant's own reply-rate history (long-term playbook memory). The 6th touch is an honest last message."""
from __future__ import annotations

from .. import memory, tools
from . import closer, qualifier
from .base import Turn

ROTATING = ["new_match", "value_add", "easy_choice", "availability", "question_prompt"]


def _steer(t: Turn) -> None:
    q = qualifier.next_question(t)
    if q:
        t.add(f"ask_{q}")
        t.asked_inc.append(q)
        t.awaiting = q
    else:
        closer.propose(t)


def plan(t: Turn) -> None:
    lead = t.lead
    attempt = lead["followups_sent"] + 1
    if t.upcoming_visits() or lead["opted_out"] or lead["status"] in ("lost", "handoff", "opted_out"):
        t.send = False
        t.actions.append("followup_skipped")
        return
    item = t.item()
    t.lead_updates["followups_sent"] = attempt
    used = list(lead.get("used_angles") or [])

    if attempt == 1:
        t.angle = "checkin"
        t.add("fu_checkin")
        _steer(t)
    elif attempt >= len(t.cfg["followup_gaps_h"]):
        t.angle = "breakup"
        t.add("fu_breakup")
        t.status, t.awaiting = "dormant", None
    elif attempt == len(t.cfg["followup_gaps_h"]) - 1:
        t.angle = "low_pressure"
        t.add("fu_low_pressure")
        t.awaiting = "interest"
    else:
        opts = tools.search_inventory(t.db, t.tid, lead.get("budget_max"), lead.get("budget_min"), lead.get("prefs"),
                                      limit=1, exclude=(item["id"],) if item else ())
        slots = tools.available_slots(t.db, t.tenant, t.now, None, n=3)
        eligible = [a for a in ROTATING if a not in used]
        if not opts or not (lead.get("budget_max") or lead.get("prefs")):
            eligible = [a for a in eligible if a != "new_match"]   # "fits what you described" must be true
        if not slots:
            eligible = [a for a in eligible if a != "easy_choice"]
        if not item or not item.get("active", 1):
            eligible = [a for a in eligible if a != "availability"]
        angle = memory.choose_angle(t.db, t.tid, eligible or ["question_prompt"])
        t.angle = angle
        if angle == "new_match":
            t.add("fu_new_match", option=opts[0])
            asked = dict(lead.get("asked") or {})
            asked["pending_item"] = opts[0]["id"]          # becomes the focus item only if the lead says yes
            t.lead_updates["asked"] = asked
            t.awaiting = "interest"
        elif angle == "easy_choice":
            t.offered, t.awaiting = slots, "slot"
            t.add("fu_easy_choice", slots=slots)
        elif angle == "availability":
            t.add("fu_availability")
            closer.propose(t)
        elif angle == "question_prompt":
            t.add("fu_question")
            t.awaiting = None
        else:
            t.add("fu_value_add")
            t.awaiting = "interest"
    if t.angle:
        memory.record_try(t.db, t.tid, t.angle)
        t.lead_updates["last_angle"] = t.angle
        if t.angle not in ("checkin", "breakup", "low_pressure"):
            t.lead_updates["used_angles"] = used + [t.angle]
