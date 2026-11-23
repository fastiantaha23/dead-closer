"""Tenant creation and demo inventory."""
from __future__ import annotations

import hashlib
import json
import secrets
import uuid

from .db import Database

DEFAULT_CONFIG = {
    "agent_name": "Dana",
    "visit_noun": "site visit",
    "tz_offset_min": -360,          # fixed UTC offset used for quiet hours / slot hours
    "visit_hours": [10, 18],        # local open/close for visits
    "visit_days": [0, 1, 2, 3, 4, 5],  # Mon..Sat
    "slot_minutes": 60,
    "slot_capacity": 2,
    "min_notice_hours": 3,
    "fee_cents": 5000,
    "bill_no_shows": False,
    "monthly_factor": 60,           # monthly payment -> approx total budget
    "address": "",                  # shown to leads in confirmations; empty = "exact address sent before the visit"
    "phone": "",
    "followup_gaps_h": [3, 21, 48, 96, 168, 336],  # gaps between unanswered follow-ups (hours)
    "send_window": [9, 20],         # proactive messages only inside this local window
    "qualify_min_budget_fit": True,
}

REAL_ESTATE_ITEMS = [
    ("RE-101", "3-bed craftsman in Mueller", 465000, {"beds": 3, "baths": 2, "sqft": 1750, "area": "Mueller", "features": ["garage", "yard"]}),
    ("RE-102", "2-bed condo downtown with skyline view", 389000, {"beds": 2, "baths": 2, "sqft": 1100, "area": "Downtown", "features": ["balcony", "gym", "parking"]}),
    ("RE-103", "4-bed family home in Cedar Park", 585000, {"beds": 4, "baths": 3, "sqft": 2650, "area": "Cedar Park", "features": ["pool", "garage", "yard"]}),
    ("RE-104", "1-bed loft in East Austin", 295000, {"beds": 1, "baths": 1, "sqft": 780, "area": "East Austin", "features": ["rooftop", "bike storage"]}),
    ("RE-105", "3-bed townhome in Round Rock", 410000, {"beds": 3, "baths": 2, "sqft": 1620, "area": "Round Rock", "features": ["garage", "patio"]}),
    ("RE-106", "5-bed estate in Westlake", 1250000, {"beds": 5, "baths": 4, "sqft": 4200, "area": "Westlake", "features": ["pool", "office", "garage"]}),
    ("RE-107", "2-bed bungalow in Hyde Park", 520000, {"beds": 2, "baths": 1, "sqft": 1300, "area": "Hyde Park", "features": ["yard", "porch"]}),
    ("RE-108", "3-bed new build in Pflugerville", 375000, {"beds": 3, "baths": 2, "sqft": 1580, "area": "Pflugerville", "features": ["garage", "smart home"]}),
]

AUTO_ITEMS = [
    ("VH-201", "2024 Toyota RAV4 Hybrid XLE", 36900, {"body": "suv", "brand": "Toyota", "fuel": "hybrid", "area": "Showroom A", "features": ["awd", "apple carplay"]}),
    ("VH-202", "2023 Honda Civic Sport", 24500, {"body": "sedan", "brand": "Honda", "fuel": "gas", "area": "Showroom A", "features": ["sunroof"]}),
    ("VH-203", "2024 Ford F-150 XLT", 52800, {"body": "truck", "brand": "Ford", "fuel": "gas", "area": "Lot B", "features": ["4x4", "tow package"]}),
    ("VH-204", "2024 Tesla Model 3 Long Range", 44990, {"body": "sedan", "brand": "Tesla", "fuel": "electric", "area": "Showroom B", "features": ["autopilot", "glass roof"]}),
    ("VH-205", "2022 Mazda CX-5 Touring", 27800, {"body": "suv", "brand": "Mazda", "fuel": "gas", "area": "Lot B", "features": ["awd", "leather"]}),
    ("VH-206", "2024 Kia Telluride SX", 49500, {"body": "suv", "brand": "Kia", "fuel": "gas", "area": "Showroom A", "features": ["third row", "awd"]}),
    ("VH-207", "2021 Toyota Corolla LE", 18900, {"body": "sedan", "brand": "Toyota", "fuel": "gas", "area": "Lot A", "features": ["backup camera"]}),
    ("VH-208", "2023 Hyundai Ioniq 5", 41200, {"body": "suv", "brand": "Hyundai", "fuel": "electric", "area": "Showroom B", "features": ["fast charging", "awd"]}),
]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_tenant(db: Database, name: str, vertical: str, now: int, *, config: dict | None = None,
                  items: list | None = None, demo: bool = False, tenant_id: str | None = None) -> tuple[str, str]:
    """Create a tenant + inventory. Returns (tenant_id, plaintext_api_key). Only the key hash is stored."""
    if vertical not in ("real_estate", "auto"):
        raise ValueError("vertical must be real_estate or auto")
    cfg = dict(DEFAULT_CONFIG)
    if vertical == "auto":
        cfg.update({"visit_noun": "test drive", "fee_cents": 4000, "agent_name": "Riley",
                    "slot_minutes": 60, "monthly_factor": 60})
    else:
        cfg.update({"monthly_factor": 200})
    cfg.update(config or {})
    tid = tenant_id or "t_" + uuid.uuid4().hex[:10]
    api_key = "dc_" + secrets.token_urlsafe(24)
    db.execute(
        "INSERT INTO tenants(id,name,vertical,api_key_hash,config,created_at,is_demo) VALUES (?,?,?,?,?,?,?)",
        (tid, name, vertical, hash_key(api_key), json.dumps(cfg), now, 1 if demo else 0),
    )
    for iid, title, price, attrs in (items if items is not None else (REAL_ESTATE_ITEMS if vertical == "real_estate" else AUTO_ITEMS)):
        db.execute("INSERT INTO items(id,tenant_id,title,price,attrs) VALUES (?,?,?,?,?)",
                   (iid, tid, title, price, json.dumps(attrs)))
    return tid, api_key


def seed_demo(db: Database, now: int) -> dict[str, str]:
    """Idempotently create the two public demo tenants."""
    out = {}
    for tid, name, vertical in (("demo_realty", "Skyline Realty", "real_estate"), ("demo_motors", "Northside Motors", "auto")):
        if not db.one("SELECT id FROM tenants WHERE id=?", (tid,)):
            addr = "500 Congress Ave" if vertical == "real_estate" else "2200 Northside Blvd"
            _, key = create_tenant(db, name, vertical, now, demo=True, tenant_id=tid, config={"address": addr, "slot_capacity": 1000})
            out[tid] = key
    return out
