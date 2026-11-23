import threading
from datetime import timedelta

from conftest import T0
from deadcloser import tools
from deadcloser.config import from_epoch, local


def say(dc, tid, lid, text, now):
    return dc.inbound(tid, lid, text, now)


def book(dc, tid, lid, text, now):
    """Ask for a specific time; the agent asks for confirmation before writing the booking, so answer YES."""
    r = dc.inbound(tid, lid, text, now)
    assert "confirm_slot_requested" in r["actions"], r
    return dc.inbound(tid, lid, "yes", now + timedelta(seconds=30))


def test_first_reply_is_fast_discloses_ai_and_opt_out(dc, realty):
    r = dc.new_lead(realty, name="Priya", phone="+15125550101", message="Hi, interested in the Mueller home", now=T0)
    assert r["sent"] and r["within_sla"] and r["latency_ms"] < 5000
    assert "virtual assistant" in r["reply"] and "STOP" in r["reply"] and "Priya" in r["reply"]


def test_qualify_then_book_is_billable(dc, realty):
    lid = dc.new_lead(realty, name="Priya", phone="+15125550101", message="Interested in the Mueller home", now=T0)["lead_id"]
    now = T0
    for text in ["moving next month", "budget is around 470k"]:
        now += timedelta(minutes=1)
        r = say(dc, realty, lid, text, now)
    assert "propose_slots" in r["actions"]
    r = say(dc, realty, lid, "1", now + timedelta(minutes=1))
    assert "booked" in r["actions"]
    stats = dc.stats(realty)
    assert stats["visits_booked"] == 1 and stats["billing"]["pending_cents"] == 5000


def test_specific_time_request_books_directly(dc, realty):
    lid = dc.new_lead(realty, name="Sam", phone="+15125550102", message="", now=T0)["lead_id"]
    r = book(dc, realty, lid, "can I see it saturday at 11am?", T0 + timedelta(minutes=1))
    assert "booked" in r["actions"]
    v = dc.lead_view(realty, lid)["visits"][0]
    l = local(from_epoch(v["start_ts"]), -360)
    assert l.weekday() == 5 and l.hour == 11


def test_unqualified_visit_not_billable_until_lead_qualifies(dc, realty):
    lid = dc.new_lead(realty, name="Sam", phone="+15125550102", message="", now=T0)["lead_id"]
    book(dc, realty, lid, "can I see it saturday at 11am?", T0 + timedelta(minutes=1))
    assert dc.stats(realty)["billing"]["pending_cents"] == 0          # incentive guard: nothing owed yet
    say(dc, realty, lid, "probably within 2 months", T0 + timedelta(minutes=3))
    say(dc, realty, lid, "budget 480k", T0 + timedelta(minutes=4))
    assert dc.stats(realty)["billing"]["pending_cents"] == 5000        # now a genuine qualified visit


def test_cancel_voids_fee_and_reschedule_moves_visit(dc, realty):
    lid = dc.new_lead(realty, name="Sam", phone="+15125550102", message="3 bed under 480k, moving next month", now=T0)["lead_id"]
    book(dc, realty, lid, "tomorrow at 2pm", T0 + timedelta(minutes=1))
    assert dc.stats(realty)["billing"]["pending_count"] == 1
    r = book(dc, realty, lid, "can we reschedule to wednesday at 3pm", T0 + timedelta(minutes=2))
    assert "rescheduled" in r["actions"]
    visits = [v for v in dc.lead_view(realty, lid)["visits"] if v["status"] == "booked"]
    assert len(visits) == 1
    r = say(dc, realty, lid, "actually cancel my visit", T0 + timedelta(minutes=3))
    assert "confirm_cancel_requested" in r["actions"] and "cancelled" not in r["actions"]      # asks before destroying
    assert dc.stats(realty)["billing"]["pending_cents"] == 5000
    r = say(dc, realty, lid, "yes", T0 + timedelta(minutes=4))
    assert "cancelled" in r["actions"] and dc.stats(realty)["billing"]["pending_cents"] == 0


def test_no_double_booking_beyond_capacity(dc, realty):
    tenant = tools.get_tenant(dc.db, realty)
    slot = tools.generate_slots(tenant, T0)[3]
    ok = 0
    for i in range(5):
        lead = {"id": f"x{i}", "tenant_id": realty, "budget_max": None, "timeline_days": None, "browsing": 0, "status": "new"}
        dc.db.execute("INSERT INTO leads(id,tenant_id,status,created_at) VALUES (?,?,?,?)", (f"x{i}", realty, "new", 0))
        ok += tools.book_visit(dc.db, tenant, lead, None, slot, T0)["ok"]
    assert ok == tenant["config"]["slot_capacity"]


def test_followups_follow_cadence_and_respect_quiet_hours(dc, realty):
    lid = dc.new_lead(realty, name="Ghost", phone="+15125550103", message="", now=T0)["lead_id"]
    sent = [r for r in dc.tick(T0 + timedelta(days=40)) if r["lead_id"] == lid and r.get("sent")]
    assert len(sent) == 6
    msgs = [m for m in dc.lead_view(realty, lid)["messages"] if m["direction"] == "out"]
    for m in msgs[1:]:
        assert 9 <= local(from_epoch(m["ts"]), -360).hour < 20
    assert dc.lead_view(realty, lid)["lead"]["status"] == "dormant"
    assert len({m["meta"]["angle"] for m in msgs[1:]}) >= 4            # varied angles
    assert dc.tick(T0 + timedelta(days=80)) == []                      # then silence


def test_reply_cancels_followups_and_conversation_resumes(dc, realty):
    lid = dc.new_lead(realty, name="Ghost", phone="+15125550103", message="", now=T0)["lead_id"]
    dc.tick(T0 + timedelta(days=2))
    r = say(dc, realty, lid, "sorry, been busy. budget 450k, want to move in 2 months", T0 + timedelta(days=2, hours=1))
    assert r["sent"] and "propose_slots" in r["actions"]
    pending = [j for j in dc.lead_view(realty, lid)["jobs"] if j["status"] == "pending"]
    assert all(j["kind"] == "followup" for j in pending) and len(pending) == 1


def test_opt_out_is_final(dc, realty):
    lid = dc.new_lead(realty, name="Ann", phone="+15125550104", message="", now=T0)["lead_id"]
    r = say(dc, realty, lid, "STOP", T0 + timedelta(minutes=1))
    assert "unsubscribed" in r["reply"]
    assert dc.tick(T0 + timedelta(days=60)) == []
    r = say(dc, realty, lid, "actually what's the price?", T0 + timedelta(days=61))
    assert r["sent"] is False
    again = dc.new_lead(realty, name="Ann", phone="+15125550104", message="hello again", now=T0 + timedelta(days=70))
    assert again["sent"] is False


def test_handoff_stops_automation(dc, realty):
    lid = dc.new_lead(realty, name="Bob", phone="+15125550105", message="", now=T0)["lead_id"]
    r = say(dc, realty, lid, "I want to talk to a real person", T0 + timedelta(minutes=1))
    assert r["status"] == "handoff"
    assert say(dc, realty, lid, "hello??", T0 + timedelta(minutes=5))["sent"] is False
    assert dc.tick(T0 + timedelta(days=10)) == []


def test_long_term_memory_survives_weeks(dc, realty):
    lid = dc.new_lead(realty, name="Mia", phone="+15125550106", message="Looking at the Cedar Park home, budget 600k", now=T0)["lead_id"]
    mem = dc.lead_view(realty, lid)["memory"]
    assert mem["budget_max"]["value"] == "600000" and "interested_in" in mem
    later = T0 + timedelta(days=45)
    r = dc.new_lead(realty, name="Mia", phone="+15125550106", message="", now=later)
    assert r["lead_id"] == lid and "welcome back" in r["reply"].lower() and "Cedar Park" in r["reply"]


def test_short_term_memory_is_checkpointed(dc, realty):
    lid = dc.new_lead(realty, name="Mia", phone="+15125550106", message="hi", now=T0)["lead_id"]
    say(dc, realty, lid, "budget 450k", T0 + timedelta(minutes=1))
    state = dc.brain.graph.get_state({"configurable": {"thread_id": f"{realty}:{lid}"}})
    assert len(state.values["messages"]) >= 4


def test_short_term_window_is_summarised(dc, realty):
    lid = dc.new_lead(realty, name="Mia", phone="+15125550106", message="hi", now=T0)["lead_id"]
    for i in range(12):
        say(dc, realty, lid, f"how big is it? {i}", T0 + timedelta(minutes=i + 1))
    state = dc.brain.graph.get_state({"configurable": {"thread_id": f"{realty}:{lid}"}})
    assert len(state.values["messages"]) <= 16 and state.values["summary"]


def test_auto_vertical_test_drive(dc, motors):
    lid = dc.new_lead(motors, name="Raj", phone="+15125550107", message="Interested in the RAV4 hybrid, buying this month, budget 40k", now=T0)["lead_id"]
    r = say(dc, motors, lid, "2", T0 + timedelta(minutes=1))
    assert "booked" in r["actions"] and "test drive" in r["reply"].lower()


def test_budget_far_below_inventory_is_honest(dc, realty):
    lid = dc.new_lead(realty, name="Kim", phone="+15125550108", message="budget is $100k, buying next month", now=T0)["lead_id"]
    msgs = [m for m in dc.lead_view(realty, lid)["messages"] if m["direction"] == "out"]
    assert "start at" in msgs[-1]["body"] and "$295,000" in msgs[-1]["body"]


def test_concurrent_leads(dc, realty):
    results, errors = [], []

    def worker(i):
        try:
            results.append(dc.new_lead(realty, name=f"L{i}", phone=f"+1512555{1000 + i}", message="interested, budget 450k", now=T0))
        except Exception as e:  # pragma: no cover
            errors.append(e)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and len(results) == 40 and all(r["sent"] for r in results)
