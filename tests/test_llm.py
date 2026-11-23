"""The Claude backend is exercised with a fake client (no network): wiring, fallbacks, and the fact that the
deterministic guards still apply to whatever the model says."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from conftest import T0
from deadcloser.config import Settings
from deadcloser.engine import DeadCloser
from deadcloser.llm import ClaudeLLM, make_llm


class FakeMessages:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def create(self, **kw):
        self.calls.append(kw)
        return self.handler(kw)


def tool_use(inp):
    return SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(type="tool_use", name="record_understanding", input=inp)])


def text(t):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=t)])


@pytest.fixture()
def claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = ClaudeLLM("claude-opus-5-5")
    return llm


def ctx():
    return {"now": T0, "tz_offset_min": -360, "awaiting": "budget", "areas": [], "brands": [], "items": [], "offered": [],
            "offered_labels": [], "biz": "Skyline Realty", "vertical": "real_estate", "monthly_factor": 200}


def test_make_llm_selects_backend(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert make_llm("auto").name == "rules"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert make_llm("auto").name.startswith("claude:")
    assert make_llm("rules").name == "rules"


def test_understand_uses_structured_tool_output(claude):
    claude.client = SimpleNamespace(messages=FakeMessages(lambda kw: tool_use(
        {"intents": ["ask_price", "bogus_intent"], "budget_max": 500000, "timeline_days": 30, "browsing": False, "slot_index": 9})))
    out = claude.understand("something the regexes cannot parse", ctx())
    assert out["budget_max"] == 500000 and out["timeline_days"] == 30
    assert out["intents"] == ["ask_price"]                       # unknown intents dropped
    assert out["slot"]["index"] is None                          # out-of-range index rejected
    call = claude.client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"] == {"effort": "low"}
    assert call["tools"][0]["name"] == "record_understanding" and "temperature" not in call
    assert "<lead_message>" in call["messages"][0]["content"]    # lead text is fenced as untrusted data


def test_model_not_called_when_rules_already_understand(claude):
    claude.client = SimpleNamespace(messages=FakeMessages(lambda kw: (_ for _ in ()).throw(AssertionError("model must not be called"))))
    out = claude.understand("my budget is 450k, moving in 2 months", ctx())
    assert out["budget_max"] == 450000 and out["timeline_days"] == 60
    assert claude.understand("What did I just ask you?", ctx())["intents"] == ["ask_recap"]   # stays rule-labelled, no call


def test_claude_fills_gaps_and_rules_win_on_conflict(claude):
    claude.client = SimpleNamespace(messages=FakeMessages(lambda kw: tool_use({"intents": ["ask_price"], "budget_max": 999999})))
    out = claude.understand("yo whats the damage on that mueller place", ctx())      # rules: nothing -> Claude adds ask_price
    assert "ask_price" in out["intents"] and out["budget_max"] == 999999


def test_understand_falls_back_on_error_and_refusal(claude):
    def boom(kw):
        raise RuntimeError("network down")
    text = "hmmm so what is the damage on that mueller place"
    claude.client = SimpleNamespace(messages=FakeMessages(boom))
    assert claude.understand(text, ctx())["intents"] == []                        # falls back to the rule parse
    claude.client = SimpleNamespace(messages=FakeMessages(lambda kw: SimpleNamespace(stop_reason="refusal", content=[])))
    assert claude.understand(text, ctx())["intents"] == []
    assert claude.errors == 2


def test_compose_requires_slot_labels_else_template(claude):
    from deadcloser import nlg
    rc = {"seed": "x", "name": "Priya", "agent": "Dana", "biz": "Skyline Realty", "noun": "site visit", "thing": "home", "address": "",
          "tz": -360, "item": None, "vertical": "real_estate", "price_catalog": []}
    ts = int(T0.timestamp()) + 86400
    plan = {"segments": [{"t": "propose_slots", "slots": [ts]}]}
    label = nlg.slot_label(ts, -360)
    claude.client = SimpleNamespace(messages=FakeMessages(lambda kw: text(f"Happy to show you around! How about {label}?")))
    assert label in claude.compose(plan, rc, [], "")
    claude.client = SimpleNamespace(messages=FakeMessages(lambda kw: text("How about next Tuesday?")))   # dropped the slot
    assert label in claude.compose(plan, rc, [], "")                                                   # template fallback


def test_hallucinating_claude_is_caught_by_output_guard(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = ClaudeLLM("claude-opus-5-5")
    llm.client = SimpleNamespace(messages=FakeMessages(lambda kw: text("Great news - I can guarantee it for $199,000, safe neighborhood!")
                                                       if "record_understanding" not in str(kw.get("tools", "")) else tool_use({"intents": []})))
    dc = DeadCloser(Settings(db_path=":memory:", checkpoint_path=":memory:"), llm=llm)
    tid, _ = dc.create_tenant("Skyline Realty", "real_estate")
    r = dc.new_lead(tid, name="Eve", phone="+15125550199", message="interested", now=T0)
    assert r["sent"] and "199,000" not in r["reply"] and "guarantee" not in r["reply"].lower()
    assert any(a["event"] == "output_guard_violation" for a in dc.db.all("SELECT event FROM audit"))
