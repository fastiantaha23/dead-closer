"""Run tests.json against any Dead Closer deployment (or the in-process app with --local).

  python scripts/run_tests_json.py --url https://your-agent.hf.space
  python scripts/run_tests_json.py --local            # no server needed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_poster(url: str | None):
    if url:
        import httpx
        client = httpx.Client(timeout=60)

        def post(body):
            r = client.post(url.rstrip("/") + "/chat", json=body)
            r.raise_for_status()
            return r.json()
        return post
    from fastapi.testclient import TestClient

    from deadcloser.api import create_app
    from deadcloser.config import Settings
    c = TestClient(create_app(Settings(db_path=":memory:", checkpoint_path=":memory:", llm_backend=os.getenv("DEADCLOSER_LLM", "rules"), scheduler_interval_s=3600)))
    c.__enter__()
    return lambda body: c.post("/chat", json=body).json()


def check(expect: dict, resp: dict) -> list[str]:
    reply, names = resp["reply"], [c["name"] for c in resp["tool_calls"]]
    low, errs = reply.lower(), []
    for s in expect.get("reply_contains", []):
        if s.lower() not in low:
            errs.append(f"reply missing {s!r}")
    any_of = expect.get("reply_contains_any")
    if any_of and not any(s.lower() in low for s in any_of):
        errs.append(f"reply has none of {any_of}")
    for s in expect.get("reply_not_contains", []):
        if s.lower() in low:
            errs.append(f"reply must not contain {s!r}")
    for n in expect.get("tool_called", []):
        if n not in names:
            errs.append(f"tool {n} not called (got {names})")
    for n in expect.get("tool_not_called", []):
        if n in names:
            errs.append(f"tool {n} must not be called")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    if not a.url and not a.local:
        ap.error("pass --url or --local")
    post = make_poster(a.url)
    spec = json.load(open(os.path.join(ROOT, "tests.json"), encoding="utf-8"))
    rand = uuid.uuid4().hex[:6]
    passed = 0
    lat = []
    for t in spec["tests"]:
        errs = []
        for i, st in enumerate(t["steps"], 1):
            body = {k: st[k].replace("{rand}", rand) for k in ("user_id", "thread_id", "message")}
            t0 = time.time()
            try:
                resp = post(body)
            except Exception as e:  # noqa: BLE001
                errs.append(f"step {i}: request failed {e!r}")
                break
            lat.append(time.time() - t0)
            errs += [f"step {i}: {e}" for e in check(st.get("expect", {}), resp)]
        ok = not errs
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {t['id']} {t['name']}")
        for e in errs:
            print("      ", e)
    lat.sort()
    print(f"\n{passed}/{len(spec['tests'])} passed | requests={len(lat)} median={lat[len(lat) // 2] * 1000:.0f}ms p95={lat[int(len(lat) * .95)] * 1000:.0f}ms")
    return 0 if passed == len(spec["tests"]) else 1


if __name__ == "__main__":
    sys.exit(main())
