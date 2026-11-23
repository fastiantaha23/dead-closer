"""The LangGraph state machine.

START -> ingest -> guard_in -+-> optout ------------------------------+
                              +-> silent (human owns the lead) -> END   |
                              +-> understand -> remember -> decide <----+--(follow-up / reminder triggers skip understand)
                                                              |
                                                          compose -> guard_out -> deliver -> END

Short-term memory : `messages` (+ rolling `summary`) are checkpointed per lead thread by the LangGraph
                    checkpointer, so every turn resumes the same conversation.
Long-term memory  : memory.py (per-lead facts, per-tenant playbook), loaded in `ingest`.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import memory, nlg, safety, tools
from .agents import qualifier, supervisor
from .agents.base import Turn, build_rc
from .config import from_epoch, to_epoch
from .db import Database
from .llm import USAGE, RuleLLM

log = logging.getLogger("deadcloser.graph")
WINDOW = 14          # messages kept verbatim in short-term memory
TERMINAL = ("lost", "opted_out", "dormant")


class State(TypedDict, total=False):
    tenant_id: str
    lead_id: str
    trigger: str
    text: str
    pii: list
    thread_id: str
    chat_mode: bool
    now_ts: int
    payload: dict
    run_id: str
    messages: Annotated[list[AnyMessage], add_messages]   # short-term memory
    summary: str
    route: str
    plan: dict
    reply: str | None
    outcome: dict


class Brain:
    def __init__(self, db: Database, llm: RuleLLM, checkpointer: Any = None):
        self.db, self.llm = db, llm
        self.turns: dict[str, Turn] = {}
        g = StateGraph(State)
        for name, fn in (("ingest", self.ingest), ("guard_in", self.guard_in), ("optout", self.optout),
                         ("resub", self.resub), ("understand", self.understand), ("remember", self.remember), ("decide", self.decide),
                         ("compose", self.compose), ("guard_out", self.guard_out), ("deliver", self.deliver),
                         ("silent", self.silent)):
            g.add_node(name, fn)
        g.add_edge(START, "ingest")
        g.add_edge("ingest", "guard_in")
        g.add_conditional_edges("guard_in", lambda s: s["route"],
                                {"optout": "optout", "silent": "silent", "understand": "understand", "decide": "decide",
                                 "resub": "resub"})
        g.add_edge("optout", "compose")
        g.add_edge("resub", "compose")
        g.add_edge("understand", "remember")
        g.add_edge("remember", "decide")
        g.add_edge("decide", "compose")
        g.add_edge("compose", "guard_out")
        g.add_edge("guard_out", "deliver")
        g.add_edge("deliver", END)
        g.add_edge("silent", END)
        self.graph = g.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------ run
    def run(self, tenant_id: str, lead_id: str, trigger: str, text: str, now, payload: dict | None = None,
            thread_id: str | None = None, chat_mode: bool = False) -> dict:
        run_id = uuid.uuid4().hex
        # Sensitive data is detected and redacted BEFORE it can reach the checkpointer, DB, memory or LLM.
        pii = safety.find_pii(text or "")
        calls, usage = [], {"in": 0, "out": 0}
        tk1, tk2 = tools.CALLS.set(calls), USAGE.set(usage)
        try:
            # short-term memory = one LangGraph checkpoint thread per (lead, conversation thread_id)
            cfg_thread = f"{tenant_id}:{lead_id}" + (f":{thread_id}" if thread_id else "")
            out = self.graph.invoke(
                {"tenant_id": tenant_id, "lead_id": lead_id, "trigger": trigger, "text": safety.redact(text or ""),
                 "pii": pii, "now_ts": to_epoch(now), "payload": payload or {}, "run_id": run_id,
                 "thread_id": thread_id or "", "chat_mode": chat_mode},
                config={"configurable": {"thread_id": cfg_thread}})
        finally:
            tools.CALLS.reset(tk1)
            USAGE.reset(tk2)
            self.turns.pop(run_id, None)
        outcome = dict(out.get("outcome") or {"sent": False})
        outcome["tool_calls"], outcome["tokens"] = calls, usage
        return outcome

    # ------------------------------------------------------------------ nodes
    def ingest(self, s: State) -> dict:
        tenant = tools.get_tenant(self.db, s["tenant_id"])
        lead = tools.get_lead(self.db, s["tenant_id"], s["lead_id"])
        if not tenant or not lead:
            raise LookupError("unknown tenant/lead")
        now = from_epoch(s["now_ts"])
        t = Turn(self.db, tenant, lead, now, s["trigger"], text=s.get("text", ""), payload=s.get("payload", {}))
        t.awaiting_before = lead.get("awaiting")
        t.thread_id, t.chat_mode = s.get("thread_id") or None, bool(s.get("chat_mode"))
        t.mem = memory.recall(self.db, tenant["id"], lead["id"])
        self.turns[s["run_id"]] = t
        update: dict = {"reply": None, "outcome": {}, "plan": {}}
        if t.text and s["trigger"] in ("inbound", "new_lead", "returning"):
            clean = safety.redact(t.text)
            tools.add_message(self.db, tenant["id"], lead["id"], "in", clean, s["now_ts"], {"trigger": s["trigger"]}, t.thread_id)
            update["messages"] = [HumanMessage(content=clean)]
            t.text = clean
            lead_up = {"last_in_at": s["now_ts"]}
            resumable = ("new", "lost", "dormant", "nurture") + (("handoff",) if t.chat_mode else ())
            if lead["status"] in resumable and not lead["opted_out"]:
                lead_up["status"] = "engaged"
                lead["status"] = "engaged"
            # playbook: did this reply answer one of our follow-up angles?
            if lead.get("last_angle") and lead.get("last_out_at") and s["now_ts"] - lead["last_out_at"] < 72 * 3600:
                memory.record_win(self.db, tenant["id"], lead["last_angle"])
                lead_up["last_angle"] = None
            lead.update(lead_up)
            self.db.update("leads", tenant["id"], lead["id"], **lead_up)
            tools.cancel_jobs(self.db, tenant["id"], lead["id"], ("followup", "nurture"))
        return update

    def guard_in(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        if s["trigger"] not in ("inbound", "new_lead", "returning") or not t.text:
            return {"route": "decide"}
        t.flags = safety.scan_inbound(t.text)
        t.flags["pii"] = s.get("pii", [])
        if t.flags["pii"]:
            self.db.audit(t.tid, t.lead["id"], s["now_ts"], "pii_redacted", kinds=t.flags["pii"])
        if t.flags["injection"]:
            self.db.audit(t.tid, t.lead["id"], s["now_ts"], "prompt_injection_flagged", text=t.text[:200])
        if t.flags["opt_out"]:
            return {"route": "optout"}
        if t.lead["opted_out"] and t.text.strip().lower().strip(".!") in ("start", "unstop", "subscribe"):
            return {"route": "resub"}
        burst = self.db.one("SELECT COUNT(*) n FROM messages WHERE tenant_id=? AND lead_id=? AND direction='in' AND ts>=?",
                            (t.tid, t.lead["id"], s["now_ts"] - 60))["n"]
        if burst > 10:   # message flood: stop replying (cost + abuse control), a human can review
            self.db.audit(t.tid, t.lead["id"], s["now_ts"], "flood_throttled", inbound_last_60s=burst)
            return {"route": "silent"}
        if t.lead["opted_out"] or (t.lead["status"] == "handoff" and not t.chat_mode):
            self.db.audit(t.tid, t.lead["id"], s["now_ts"], "inbound_for_human", status=t.lead["status"])
            return {"route": "silent"}
        return {"route": "understand"}

    def silent(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        reason = "opted_out" if t.lead["opted_out"] else "handoff" if t.lead["status"] == "handoff" else "throttled"
        return {"outcome": {"sent": False, "reason": reason, "status": t.lead["status"]}}

    def optout(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        t.add("optout")
        t.status, t.send = "opted_out", True
        t.lead_updates["opted_out"] = 1
        t.actions.append("opted_out")
        tools.cancel_jobs(self.db, t.tid, t.lead["id"], ("followup", "nurture", "reminder_24h", "reminder_2h"))
        return {}

    def resub(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        t.add("resubscribed")
        t.status, t.send, t.awaiting = "engaged", True, None
        t.lead_updates["opted_out"] = 0
        t.lead["opted_out"] = 0
        t.actions.append("resubscribed")
        return {}

    def understand(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        items = tools.list_items(self.db, t.tid)
        offered = list(t.lead.get("offered") or [])
        ctx = {
            "now": t.now, "tz_offset_min": t.cfg["tz_offset_min"], "awaiting": t.lead.get("awaiting"),
            "areas": sorted({i["attrs"].get("area") for i in items if i["attrs"].get("area")}),
            "brands": sorted({i["attrs"].get("brand") for i in items if i["attrs"].get("brand")}),
            "items": items, "monthly_factor": t.cfg["monthly_factor"], "offered": offered,
            "offered_labels": [nlg.slot_label(x, t.cfg["tz_offset_min"]) for x in offered],
            "biz": t.tenant["name"], "vertical": t.tenant["vertical"],
        }
        t.nlu = self.llm.understand(t.text, ctx)
        return {}

    def remember(self, s: State) -> dict:
        qualifier.absorb(self.turns[s["run_id"]])
        return {}

    def decide(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        supervisor.run_policy(t)
        return {"plan": {"segments": t.segments}}

    def _rc(self, t: Turn) -> dict:
        return build_rc(self.db, t.tenant, t.lead, t.item())

    def compose(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        if not t.send or not t.segments:
            return {"reply": None}
        rc = self._rc(t)
        history = tools.list_messages(self.db, t.tid, t.lead["id"], 8, thread_id=t.thread_id)
        mem = memory.recall_text(t.mem)
        text = self.llm.compose({"segments": t.segments}, rc, history, f"{mem}; {s.get('summary', '')}")
        return {"reply": text}

    def guard_out(self, s: State) -> dict:
        t = self.turns[s["run_id"]]
        text = s.get("reply")
        if not text:
            return {}
        rc = self._rc(t)
        plan = {"segments": t.segments}
        extra = {x for x in (t.lead.get("budget_max"), t.lead.get("budget_min")) if x}
        others = [r["name"] for r in self.db.all("SELECT name FROM tenants WHERE id!=?", (t.tid,))]
        kw = dict(allowed_amounts=nlg.allowed_amounts(rc, plan, extra), other_tenant_names=others,
                  tenant_phone=t.cfg.get("phone", ""), vertical=t.tenant["vertical"])
        viol = safety.check_outbound(text, **kw)
        if viol:
            self.db.audit(t.tid, t.lead["id"], s["now_ts"], "output_guard_violation", violations=viol, llm=self.llm.name)
            text = nlg.render(plan, rc)
            viol2 = safety.check_outbound(text, **kw)
            segs = list(t.segments)
            while viol2 == ["too_long"] and len(segs) > 1:      # too long: drop trailing steering segments, keep the answer
                segs = segs[:-1]
                text = nlg.render({"segments": segs}, rc)
                viol2 = safety.check_outbound(text, **kw)
            if viol2:
                self.db.audit(t.tid, t.lead["id"], s["now_ts"], "output_blocked", violations=viol2)
                t.send, t.status = False, "handoff"
                return {"reply": None, "outcome": {"sent": False, "reason": "guard_blocked", "violations": viol2}}
        return {"reply": text}

    def deliver(self, s: State) -> dict:  # noqa: C901
        t, now_ts = self.turns[s["run_id"]], s["now_ts"]
        tid, lead = t.tid, t.lead
        cfg = t.cfg
        reply = s.get("reply")
        proactive = s["trigger"] in ("followup", "nurture", "reminder_24h", "reminder_2h")
        window = tuple(cfg["send_window"])
        out: dict = {"sent": False, "trigger": s["trigger"]}

        if reply and proactive and not safety.in_send_window(t.now, cfg["tz_offset_min"], window):
            when = safety.next_window_open(t.now, cfg["tz_offset_min"], window)
            tools.schedule_job(self.db, tid, lead["id"], s["trigger"] if s["trigger"] != "followup" else "followup", when,
                               {**t.payload, "attempt": t.payload.get("attempt")})
            self.db.audit(tid, lead["id"], now_ts, "deferred_quiet_hours", until=to_epoch(when))
            return {"outcome": {**out, "reason": "quiet_hours_deferred"}}
        if reply and lead["opted_out"] and "opted_out" not in t.actions:
            self.db.audit(tid, lead["id"], now_ts, "blocked_opted_out")
            return {"outcome": {**out, "reason": "opted_out"}}
        if reply:
            recent = self.db.one("SELECT COUNT(*) n FROM messages WHERE tenant_id=? AND lead_id=? AND direction='out' AND ts>?",
                                 (tid, lead["id"], now_ts - 24 * 3600))["n"]
            if recent >= 6 and s["trigger"] in ("followup", "nurture"):
                self.db.audit(tid, lead["id"], now_ts, "rate_capped")
                return {"outcome": {**out, "reason": "rate_capped"}}

        updates = dict(t.lead_updates)
        if t.status:
            updates["status"] = t.status
        if reply:
            first_ever = self.db.one("SELECT COUNT(*) n FROM messages WHERE tenant_id=? AND lead_id=? AND direction='out'",
                                     (tid, lead["id"]))["n"] == 0
            if first_ever and "opted_out" not in t.actions and "STOP" not in reply:
                reply = f"{reply} Reply STOP to opt out."
            updates.update({"last_out_at": now_ts, "awaiting": t.awaiting, "offered": t.offered})
            tools.add_message(self.db, tid, lead["id"], "out", reply, now_ts,
                              {"segments": [x["t"] for x in t.segments], "angle": t.angle, "actions": t.actions,
                               "llm": self.llm.name}, t.thread_id)
            out["sent"] = True
        elif t.status or t.lead_updates:
            updates.setdefault("awaiting", t.awaiting)
        if updates:
            self.db.update("leads", tid, lead["id"], **{k: (int(v) if isinstance(v, bool) else v) for k, v in updates.items()})
        lead.update(updates)

        # ---- scheduling side effects
        if t.booked_visit:
            tools.cancel_jobs(self.db, tid, lead["id"], ("followup", "nurture"))
            tools.schedule_reminders(self.db, t.tenant, lead["id"], t.booked_visit, t.now)
        elif lead["status"] in TERMINAL or lead["opted_out"] or lead["status"] == "handoff":
            tools.cancel_jobs(self.db, tid, lead["id"], ("followup", "nurture"))
        elif t.nurture_days:
            tools.cancel_jobs(self.db, tid, lead["id"], ("followup",))
            tools.schedule_job(self.db, tid, lead["id"], "nurture", t.now + timedelta(days=t.nurture_days))
        elif (reply and s["trigger"] in ("inbound", "new_lead", "returning", "nurture")) or s["trigger"] == "followup":
            if not tools.active_visits(self.db, tid, lead["id"], now_ts):
                tools.schedule_followup(self.db, t.tenant, lead, t.now)
        for ev in t.events:
            self.db.audit(tid, lead["id"], now_ts, ev["event"], **{k: v for k, v in ev.items() if k != "event"})
        if t.actions:
            self.db.audit(tid, lead["id"], now_ts, "actions", actions=t.actions, trigger=s["trigger"])

        out.update({"reply": reply if out["sent"] else None, "actions": t.actions, "status": lead["status"],
                    "angle": t.angle, "booked_visit": t.booked_visit["id"] if t.booked_visit else None})
        result: dict = {"outcome": out}
        msgs: list = [AIMessage(content=reply)] if reply else []
        trim = self._trim(s, msgs)
        msgs = msgs + trim.pop("messages", [])
        if msgs:
            result["messages"] = msgs
        result.update(trim)
        return result

    def _trim(self, s: State, new: list) -> dict:
        """Keep the last WINDOW messages verbatim; fold older ones into a running summary."""
        existing = list(s.get("messages") or [])
        total = existing + new
        if len(total) <= WINDOW:
            return {}
        old = total[:-WINDOW]
        old_existing = [m for m in old if m in existing]
        line = " | ".join(("Lead: " if isinstance(m, HumanMessage) else "Agent: ") + str(m.content)[:90] for m in old_existing)
        summary = (s.get("summary", "") + " " + line).strip()[-900:]
        return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old_existing if getattr(m, "id", None)]}
