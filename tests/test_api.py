import pytest
from fastapi.testclient import TestClient

from deadcloser.api import create_app
from deadcloser.config import Settings


@pytest.fixture()
def client():
    s = Settings(db_path=":memory:", checkpoint_path=":memory:", llm_backend="rules", admin_key="adm-secret", demo_mode=True, scheduler_interval_s=3600)
    app = create_app(s)
    with TestClient(app) as c:
        yield c


def mk(client, name, vertical="real_estate"):
    r = client.post("/admin/tenants", json={"name": name, "vertical": vertical}, headers={"X-Admin-Key": "adm-secret"})
    assert r.status_code == 201
    return r.json()["api_key"], r.json()["tenant_id"]


def H(key):
    return {"Authorization": f"Bearer {key}"}


def test_health_and_static_site(client):
    assert client.get("/health").text == "ok"
    assert client.get("/api/status").json()["ok"] is True
    r = client.get("/")
    assert r.status_code == 200 and "Dead Closer" in r.text
    assert client.get("/app.js").status_code == 200


def test_admin_requires_key(client):
    assert client.post("/admin/tenants", json={"name": "Acme Homes", "vertical": "real_estate"}).status_code == 403
    assert client.post("/admin/tenants", json={"name": "Acme Homes", "vertical": "real_estate"}, headers={"X-Admin-Key": "nope"}).status_code == 403


def test_tenant_auth_required(client):
    assert client.get("/v1/leads").status_code == 401
    assert client.get("/v1/leads", headers=H("bogus")).status_code == 401


def test_lead_lifecycle_over_http(client):
    key, tid = mk(client, "Acme Homes")
    r = client.post("/v1/leads", json={"name": "Priya", "phone": "+15125550101", "message": "Interested, budget 450k, moving next month"}, headers=H(key))
    assert r.status_code == 201
    body = r.json()
    assert body["sent"] and body["within_sla"] and "Acme Homes" in body["reply"]
    lid = body["lead_id"]
    r = client.post(f"/v1/leads/{lid}/messages", json={"text": "1"}, headers=H(key))
    assert "booked" in r.json()["actions"]
    assert client.get("/v1/billing", headers=H(key)).json()["summary"]["pending_cents"] == 5000
    assert len(client.get("/v1/outbox", headers=H(key)).json()) >= 2
    assert client.get("/v1/stats", headers=H(key)).json()["visits_booked"] == 1
    assert client.get(f"/v1/leads/{lid}", headers=H(key)).json()["lead"]["status"] == "booked"


def test_cross_tenant_access_is_blocked_over_http(client):
    ka, _ = mk(client, "Acme Homes")
    kb, _ = mk(client, "Bolt Motors", "auto")
    lid = client.post("/v1/leads", json={"name": "A", "phone": "+15125550101", "message": "hi"}, headers=H(ka)).json()["lead_id"]
    assert client.get(f"/v1/leads/{lid}", headers=H(kb)).status_code == 404
    assert client.post(f"/v1/leads/{lid}/messages", json={"text": "hello"}, headers=H(kb)).status_code == 404
    assert client.get("/v1/leads", headers=H(kb)).json() == []
    assert all(i["id"].startswith("VH") for i in client.get("/v1/inventory", headers=H(kb)).json())
    outbox_b = client.get("/v1/outbox", headers=H(kb)).json()
    assert outbox_b == []


def test_input_validation(client):
    key, _ = mk(client, "Acme Homes")
    assert client.post("/v1/leads", json={"message": "x" * 5000}, headers=H(key)).status_code == 422
    assert client.post("/v1/leads/abc/messages", json={"text": ""}, headers=H(key)).status_code == 422
    r = client.post("/v1/leads", json={"name": "<script>", "phone": "not a phone", "message": "hi"}, headers=H(key))
    assert r.status_code == 201   # bad name/phone are dropped, not trusted


def test_cannot_tick_live_tenant_into_future(client):
    key, _ = mk(client, "Acme Homes")
    r = client.post("/v1/scheduler/tick", json={"now": "2099-01-01T00:00:00Z"}, headers=H(key))
    assert r.status_code == 400


def test_demo_session_flow_and_isolation(client):
    a = client.post("/demo/start", json={"vertical": "real_estate", "name": "Alex", "message": ""}).json()
    b = client.post("/demo/start", json={"vertical": "real_estate", "name": "Bea", "message": ""}).json()
    assert a["session"] != b["session"] and a["messages"][0]["direction"] == "out"
    s = client.post(f"/demo/{a['session']}/say", json={"text": "budget 450k, buying next month"}).json()
    assert s["lead"]["qualified"] == 1
    assert client.get(f"/demo/{b['session']}").json()["lead"]["qualified"] == 0     # sessions do not leak
    adv = client.post(f"/demo/{b['session']}/advance", json={"hours": 72}).json()
    assert adv["lead"]["followups_sent"] >= 2
    assert client.get(f"/demo/{a['session']}").json()["lead"]["followups_sent"] == 0  # b's clock did not move a
    assert client.get("/demo/nonexistent").status_code == 404
    # tenant API keys are never exposed through demo
    assert "api_key" not in str(s)


def test_demo_rate_limit(client):
    last = None
    for _ in range(25):
        last = client.post("/demo/start", json={"vertical": "auto", "name": "X", "message": ""})
    assert last.status_code == 429
