# Dead Closer

An AI sales closer for **real-estate agencies** (site visits) and **car dealers** (test drives).

It replies to every new lead **inside a minute**, finds out **budget and timeline**, **follows up for weeks** without being annoying, and **books the visit**. You pay **a flat fee per qualified, booked visit** - nothing for chatter.

> Live site: **https://fastiantaha23.github.io/dead-closer/** · Source: **https://github.com/fastiantaha23/dead-closer** · Benchmarks: [RESULTS.md](RESULTS.md)

The site hosts a chat with the agent plus a panel showing what it learned, remembers, scheduled and earned. On GitHub Pages it replays transcripts recorded from the real engine; run the server (below) or deploy it and the same page talks to the live engine in a sandbox with a virtual clock.

## What you asked for, and where it is

| Requirement | Implementation |
|---|---|
| Replies to new leads within a minute | `POST /v1/leads` answers synchronously; first-reply p95 is 54 ms on the offline engine, `within_sla` is returned on every call |
| Qualifies budget and timeline | `nlu.py` / Claude tool call -> `agents/qualifier.py` (score 0-100, max two asks per field, one question per message) |
| Follows up for days | `agents/followup.py` + job scheduler: ~3 h, 1 d, 3 d, 7 d, 14 d, 28 d, quiet hours respected, stops on reply / STOP / booking |
| Books site visits / test drives | `agents/closer.py` + `tools.book_visit` (validated slots, capacity, atomic, idempotent), reminders 24 h and 2 h before |
| Earns per booked visit | `billing.py`: fee **pending** on booking, **earned** when the visit time passes, **void** on cancel; only *qualified* visits are billable |
| LangGraph | `graph.py`: ingest -> guard_in -> understand -> remember -> decide -> compose -> guard_out -> deliver |
| Website | `deadcloser/web` (landing, live demo, benchmark charts, API docs) |
| Tools | inventory search, slot search, booking, cancel, jobs, memory (`tools.py`), all tenant-scoped |
| Short-term memory | LangGraph checkpointer per lead thread (rolling window + running summary) |
| Long-term memory | per-lead facts / preferences / episodes survive weeks (`memory.py`); per-tenant follow-up playbook learned with UCB1 |
| Multiple users | multi-tenant: hashed API keys, every query tenant-scoped, isolation tests over HTTP and in the engine |
| Safety | deterministic guards on input and output + a 100-case adversarial suite |
| Benchmark | `bench/`: speed, load, extraction accuracy, conversion ablations, compliance counters, safety |
| Deploy + GitHub | `Dockerfile`, `render.yaml`, GitHub Pages site in `docs/` |
| Agents | supervisor + qualifier, closer, info, objections, follow-up agents in `deadcloser/agents/` |

## Architecture

```
inbound lead / timer ──► ingest ──► guard_in ──┬─► optout ──────────────────────────┐
                         (redact PII,          ├─► silent (human owns the lead)      │
                          load memory)         └─► understand ─► remember ─► decide ─┤◄─ follow-up / reminder triggers
                                                   (Claude tool   (facts → memory,   │   skip straight to decide
                                                    or rules)      score, billing)   ▼
                                       deliver ◄── guard_out ◄── compose ◄── supervisor policy
                              (quiet hours, caps, STOP footer,    (no invented     (qualifier / closer /
                               persist, schedule next follow-up)   prices, links,    info / objections /
                                                                   steering)         follow-up agents)
```

**Design rule: the model never decides, it only reads and words.** Which action to take, which tool to call, which tenant's data to touch, whether a message may be sent - all plain code. Claude is used to (1) turn the lead's text into structured fields (tool call) and (2) rewrite an already-decided plan in a warmer voice. Every model output passes the same deterministic output guard, and any failure (timeout, refusal, bad output) falls back to the rules engine, so the lead is always answered.

### Safety, concretely

- **Opt-out**: STOP and ~15 phrasings/languages (while not tripping on "stop by at 3"), one confirmation, then permanent silence - follow-ups, reminders, even re-submitted forms are suppressed.
- **Prompt injection**: lead text is fenced as data; injection attempts are detected, logged and deflected; there is nothing to hijack because actions are not model-chosen.
- **PII**: card numbers (Luhn), SSNs, passwords/CVV/PIN are redacted *before* they reach the checkpointer, DB, memory or model.
- **Honesty**: introduces itself as a virtual assistant, admits being an AI when asked, never quotes a price that is not in inventory data, never promises discounts/approvals/appreciation, escalates unknown details instead of guessing.
- **Fair housing**: declines neighbourhood/people questions, output guard blocks steering language.
- **Escalation**: abuse, legal threats, "get me a human" and distress hand the lead to a person and silence automation.
- **Incentive integrity**: a visit is billable only if the lead is qualified (budget fits inventory, timeline <= 90 days, not "just browsing"). Padding visits earns nothing.
- **Compliance**: proactive messages only 9:00-20:00 local, caps per day, flood throttling, STOP footer on the first message.
- **Tenant isolation**: tenant id comes from the API key, never from the model; hashed keys; cross-tenant leakage guard on output.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                                # 55 tests
DEADCLOSER_ADMIN_KEY=change-me uvicorn deadcloser.api:app_factory --factory --port 8000
# open http://localhost:8000
```

Create a tenant and send a lead:

```bash
curl -X POST localhost:8000/admin/tenants -H "X-Admin-Key: change-me" -H "Content-Type: application/json" \
     -d '{"name":"Acme Homes","vertical":"real_estate","fee_cents":7500}'          # -> api_key (shown once)

curl -X POST localhost:8000/v1/leads -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"name":"Priya","phone":"+15125550101","message":"Interested in the Mueller home, budget 450k"}'
```

Endpoints (OpenAPI at `/docs`): `POST /v1/leads`, `GET /v1/leads[/{id}]`, `POST /v1/leads/{id}/messages` (inbound SMS webhook), `GET /v1/outbox` (messages to deliver through your SMS/WhatsApp provider), `GET /v1/visits`, `POST /v1/visits/{id}/status` (completed / no_show), `GET /v1/billing`, `GET /v1/stats`, `POST /v1/scheduler/tick`.

The server runs the scheduler itself every 30 s. Delivery channels are intentionally an adapter you own: poll `/v1/outbox` and post inbound replies to `/v1/leads/{id}/messages` (a Twilio webhook is ~20 lines).

### Use Claude

```bash
export ANTHROPIC_API_KEY=...            # DEADCLOSER_LLM=auto picks Claude when a key is present
export DEADCLOSER_MODEL=claude-opus-5-5 # default; claude-haiku-5-5 is a cheaper, faster choice for SMS
```

Without a key everything runs on the offline rules engine (the default for tests, benchmarks and the public demo).

## Benchmarks

Reproduce everything:

```bash
python -m bench.safety_suite        # 100 adversarial / compliance cases
python -m bench.simulate 400        # speed, load, extraction, conversion ablations (3 seeds, ~10 min)
python -m bench.export_replay       # transcripts for the static site
python scripts/render_results.py    # -> RESULTS.md
```

| What | Result | Kind |
|---|---|---|
| First reply latency (whole engine, offline mode) | p50 **24 ms**, p95 **54 ms**, 100% inside the 60 s SLA | measured |
| Burst of 300 simultaneous leads (SQLite, one instance) | all answered in <= 2.5 s, ~11 leads/s | measured |
| Budget extracted correctly (n=641) / timeline (n=642) | **96.9%** / **94.0%** | measured vs ground truth |
| Safety suite | **100/100** (first run was 92/100; bugs found and fixed, see RESULTS.md) | measured, author-written |
| Messages after opt-out / proactive outside 9-20h | **0 / 0** over 1,200 simulated leads | measured |
| Booking rate: full agent vs no follow-ups vs human-speed (4 h) reply | **30.3%** vs 20.2% vs 12.7% | simulated, assumed lead behaviour |
| No qualification questions | 33.7% bookings but only 28% billable: revenue **$430 vs $1,023** per 100 leads | simulated |

Full tables, assumptions and caveats: **[RESULTS.md](RESULTS.md)**.

## Deploy

- **Static site (GitHub Pages)**: `python scripts/build_site.py --repo <url>` writes `docs/`; Pages serves `main:/docs`. Without a backend it replays real transcripts.
- **Full engine (Render, free tier works)**: Render -> New -> Blueprint -> select this repo (`render.yaml`). Add `ANTHROPIC_API_KEY` to use Claude. Then rebuild the site with `--api https://<your-service>.onrender.com/` and the Pages demo becomes live.
- **Docker**: `docker build -t dead-closer . && docker run -p 8000:8000 -v dc:/data dead-closer`.

## Limits you should know about

- **Simulated conversion numbers are assumptions, not evidence.** They show how variants compare; real booking rates need a pilot with real leads.
- **Claude mode is wired, unit-tested with a fake client, and falls back safely, but it has not been run against the live API or the benchmark in this repo's history.** Run the safety suite with `DEADCLOSER_LLM=claude` before trusting it.
- SQLite + a single worker: about 11 new-lead conversations/s in the burst test on a laptop; for more, shard by tenant or swap the storage layer for Postgres.
- Channel delivery (SMS/WhatsApp/email), CRM sync, calendar sync and real consent records are integration points, not included. You are responsible for TCPA/GDPR consent for the numbers you load.
- Timezones are a fixed UTC offset per tenant (no DST rules). Inventory is managed through the DB/seed, not an admin UI.
- The Docker image and Render blueprint were written but not built in the authoring environment (no Docker daemon / hosting account available).

## Layout

```
deadcloser/   api.py engine.py graph.py  agents/  tools.py memory.py billing.py safety.py nlu.py nlg.py llm.py  web/
bench/        simulate.py safety_suite.py export_replay.py results/
tests/        flow, safety, API, Claude-backend tests
scripts/      build_site.py render_results.py
```

MIT licensed.
