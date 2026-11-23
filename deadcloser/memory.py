"""Memory layers.

* Short-term: the LangGraph checkpointer persists the rolling conversation window per lead
  thread (see graph.py).
* Long-term (per lead): durable facts / preferences / objections / episodes in the `memories`
  table. They survive the thread window, summarisation and even a lead coming back weeks later.
* Long-term (per tenant): the *playbook* - which follow-up angles actually get replies. The
  follow-up agent picks angles with UCB1 so it keeps learning from outcomes.
"""
from __future__ import annotations

import math
from typing import Any

from .db import Database

ANGLES = ["new_match", "value_add", "easy_choice", "availability", "question_prompt", "low_pressure"]


def remember(db: Database, tenant_id: str, lead_id: str, key: str, value: Any, kind: str, now: int,
             confidence: float = 1.0) -> None:
    db.execute(
        "INSERT INTO memories(tenant_id,lead_id,key,value,kind,confidence,ts) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(tenant_id,lead_id,key) DO UPDATE SET value=excluded.value, kind=excluded.kind, "
        "confidence=excluded.confidence, ts=excluded.ts",
        (tenant_id, lead_id, key, str(value), kind, confidence, now),
    )


def recall(db: Database, tenant_id: str, lead_id: str) -> dict[str, dict]:
    rows = db.all("SELECT key,value,kind,confidence,ts FROM memories WHERE tenant_id=? AND lead_id=? ORDER BY ts",
                  (tenant_id, lead_id))
    return {r["key"]: r for r in rows}


def recall_text(mem: dict[str, dict]) -> str:
    return "; ".join(f"{k}={v['value']}" for k, v in mem.items()) or "nothing yet"


# ---------------------------------------------------------------- tenant playbook
def record_try(db: Database, tenant_id: str, angle: str) -> None:
    db.execute("INSERT INTO playbook(tenant_id,angle,tries,wins) VALUES (?,?,1,0) "
               "ON CONFLICT(tenant_id,angle) DO UPDATE SET tries=tries+1", (tenant_id, angle))


def record_win(db: Database, tenant_id: str, angle: str) -> None:
    db.execute("INSERT INTO playbook(tenant_id,angle,tries,wins) VALUES (?,?,1,1) "
               "ON CONFLICT(tenant_id,angle) DO UPDATE SET wins=wins+1", (tenant_id, angle))


def playbook(db: Database, tenant_id: str) -> dict[str, dict]:
    return {r["angle"]: r for r in db.all("SELECT angle,tries,wins FROM playbook WHERE tenant_id=?", (tenant_id,))}


def choose_angle(db: Database, tenant_id: str, candidates: list[str]) -> str:
    """UCB1 over this tenant's learned reply rates. Untried angles are explored first (in list order)."""
    if not candidates:
        return ANGLES[-1]
    pb = playbook(db, tenant_id)
    total = sum(max(1, pb.get(a, {}).get("tries", 0)) for a in candidates)
    best, best_score = candidates[0], -1.0
    for a in candidates:
        tries = pb.get(a, {}).get("tries", 0)
        if tries == 0:
            return a
        wins = pb.get(a, {}).get("wins", 0)
        score = wins / tries + math.sqrt(2 * math.log(total) / tries)
        if score > best_score:
            best, best_score = a, score
    return best
