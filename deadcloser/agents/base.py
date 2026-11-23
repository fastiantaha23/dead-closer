"""Shared turn context passed between the specialist agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .. import tools
from ..db import Database


@dataclass
class Turn:
    db: Database
    tenant: dict
    lead: dict
    now: datetime
    trigger: str                      # new_lead | inbound | followup | nurture | reminder_24h | reminder_2h
    text: str = ""
    nlu: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    mem: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    # outputs
    segments: list[dict] = field(default_factory=list)
    lead_updates: dict = field(default_factory=dict)
    asked_inc: list[str] = field(default_factory=list)
    awaiting: str | None = None
    offered: list[int] = field(default_factory=list)
    status: str | None = None
    actions: list[str] = field(default_factory=list)
    send: bool = True
    nurture_days: int | None = None
    angle: str | None = None
    booked_visit: dict | None = None
    cancelled_visit: dict | None = None
    reschedule_of: str | None = None
    events: list[dict] = field(default_factory=list)
    awaiting_before: str | None = None
    thread_id: str | None = None
    chat_mode: bool = False
    clarify_budget: int | None = None
    budget_fit: str | None = None

    # -------------------------------------------------------------- helpers
    @property
    def cfg(self) -> dict:
        return self.tenant["config"]

    @property
    def tid(self) -> str:
        return self.tenant["id"]

    @property
    def intents(self) -> set[str]:
        return set(self.nlu.get("intents", []))

    def item(self) -> dict | None:
        iid = (self.nlu.get("item_id") if self.nlu else None) or self.lead.get("item_id")
        return tools.get_item(self.db, self.tid, iid)

    def add(self, t: str, **data: Any) -> None:
        self.segments.append({"t": t, **data})

    def upcoming_visits(self) -> list[dict]:
        return tools.active_visits(self.db, self.tid, self.lead["id"], int(self.now.timestamp()))


def build_rc(db: Database, tenant: dict, lead: dict, item: dict | None) -> dict:
    cfg = tenant["config"]
    catalog = [i["price"] for i in tools.list_items(db, tenant["id"])]
    return {
        "seed": lead["id"], "name": lead.get("name"), "agent": cfg["agent_name"], "biz": tenant["name"],
        "noun": cfg["visit_noun"], "thing": "home" if tenant["vertical"] == "real_estate" else "car",
        "address": cfg["address"], "tz": cfg["tz_offset_min"], "item": item, "vertical": tenant["vertical"],
        "price_catalog": catalog,
    }
