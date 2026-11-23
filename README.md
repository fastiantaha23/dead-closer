# Dead Closer

An AI sales agent for **real-estate agencies** (site visits) and **car dealers** (test drives). It answers every new lead in seconds, qualifies **budget and timeline**, follows up for weeks, books the visit, and earns a flat fee **only when a qualified visit is booked**.

![Demo](bench/results/demo.gif)

| | |
|---|---|
| Website (GitHub Pages, replay mode) | https://fastiantaha23.github.io/dead-closer/ |
| Website (Vercel) + Agent API (Hugging Face / Render) | set after deploy, see [Deploy](#deploy-free-tier) |
| Benchmarks | [RESULTS.md](RESULTS.md) |
| Own test cases | [tests.json](tests.json) (10 cases, runner in `scripts/run_tests_json.py`) |

## Workshop checklist

| Must-have | Where it is |
|---|---|
| Customer page + admin dashboard | `deadcloser/web/chat.html` (customer chat), `admin.html` (users, memory, visits, fees, decision log), `index.html` (landing + demo + benchmarks) |
| LangGraph agent, >= 2 agents/nodes | `deadcloser/graph.py`: 10 nodes (ingest, guard_in, understand, remember, decide, compose, guard_out, deliver, optout, resub) and a supervisor routing to the **qualifier, closer, info, objections and follow-up agents** (`deadcloser/agents/`) |
| >= 2 real tools | `search_inventory`, `check_availability`, `book_visit`, `cancel_visit`, `save_memory`, `recall_memory` (`deadcloser/tools.py`); every call is returned in `tool_calls` |
| Short-term memory by `thread_id` | LangGraph checkpointer, one checkpoint thread per `(user, thread_id)`; rolling window + running summary |
| Long-term memory by `user_id` | per-user facts, preferences, notes and episodes in Postgres (`memories` + `leads`) |
| Users never see each other's data | every query scoped by `user_id`/tenant; tested in the engine, over HTTP and on real Postgres |
| Safety | refuses off-topic and prompt injection, asks YES before cancelling or booking an un-offered time, STOP/START, PII redaction, no invented prices/promises, fair-housing guard |
| Model | `claude-haiku-5-5` (`DEADCLOSER_MODEL`), at most 2 small calls per message; compliance/slot messages use exact templates with no model call |
| `POST /chat`, `GET /health` | open, no login; `/chat` returns exactly `{reply, tool_calls, tokens}`, `/health` returns `ok` |
| Hosted Postgres memory | `DATABASE_URL` (Neon / Supabase); nothing important is in process memory |
| `.env.example`, `tests.json`, benchmark table, memory note | in this repo, below |

## Architecture

```mermaid
flowchart LR
  U[Customer page<br/>Vercel static site] -->|POST /chat| API
  A[Admin dashboard<br/>X-Admin-Key] -->|/admin/*| API
  subgraph API[Agent API - FastAPI on Hugging Face Space / Render]
    direction TB
    G[LangGraph state machine] --> T[Tools]
    G --> S[Safety guards<br/>input + output, plain code]
  end
  API -->|messages, long-term memory,<br/>visits, ledger, checkpoints| PG[(Postgres<br/>Neon / Supabase)]
  API -->|understand + wording only| C[Claude Haiku 5.5]
```

```mermaid
flowchart LR
  I[ingest<br/>redact PII, load memory] --> GI{guard_in}
  GI -->|STOP| O[optout]
  GI -->|START| R[resub]
  GI -->|human owns lead / flood| X[silent]
  GI --> UN[understand<br/>Claude tool call or rules] --> RM[remember<br/>facts to long-term memory,<br/>score, billing] --> D
  O --> CO
  R --> CO
  GI -->|follow-up / reminder jobs| D[decide<br/>supervisor policy]
  D --> Q[qualifier agent]
  D --> CL[closer agent<br/>slots / book / cancel]
  D --> IN[info agent<br/>answers, memory recall]
  D --> OB[objections agent]
  D --> FU[follow-up agent]
  Q & CL & IN & OB & FU --> CO[compose<br/>Claude rewrite or template] --> GO{guard_out<br/>no invented prices,<br/>links, steering, AI denial} --> DE[deliver<br/>quiet hours, caps, persist,<br/>schedule follow-ups]
```

**Design rule: the model never decides, it only reads and words.** Which action to take, which tool to call and whose data to touch are plain code. Claude turns the lead's text into structured fields (one tool call) and rewrites an already-decided plan in a friendlier voice. Every output passes the same deterministic guard, and any model failure (timeout, refusal, bad output) falls back to a template, so the user always gets an answer.

## Memory design note

| | Short-term | Long-term |
|---|---|---|
| **Scope** | one conversation = one `thread_id` | one person = one `user_id`, across all threads |
| **What goes in it** | the rolling message window (last 14 messages) + a running summary of older turns; pending question (`awaiting`), offered visit slots | name, budget, timeline, bedrooms/body/area/feature preferences, the listing they like, booked/cancelled visits, free-text notes ("I have two dogs"), per-business follow-up playbook |
| **Where it is stored** | LangGraph `PostgresSaver` checkpoint (`thread_id` = `tenant:user:thread`), plus the `messages` table tagged with `thread_id` | Postgres tables `leads` (typed fields) and `memories` (key/value facts), written by the `save_memory` tool |
| **Survives restart** | yes (Postgres) | yes (Postgres) |
| **How users are separated** | the checkpoint thread id embeds the user; messages are queried by `lead_id` + `thread_id` | every query has `tenant_id` and the user's lead row; `user_id` is unique per tenant; the model never supplies an id; PII is redacted before storage |
| **Read by** | `ask_recap` ("what did I just ask?"), the model prompt (last 6 turns) | `ask_memory` ("what do you know about me?"), qualification, personalised greeting for a returning user |

A new `thread_id` for the same `user_id` starts a clean conversation (no pending question, no transcript) but the agent greets the user back and still knows their budget, timeline, preferences and notes.

## Setup

```bash
git clone https://github.com/fastiantaha23/dead-closer && cd dead-closer
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                         # fill in ANTHROPIC_API_KEY, DATABASE_URL, DEADCLOSER_ADMIN_KEY
export $(grep -v '^#' .env | xargs)                          # or set the variables in your shell
uvicorn deadcloser.api:app_factory --factory --port 8000    # http://localhost:8000  (site)  /docs (OpenAPI)
```

Without `ANTHROPIC_API_KEY` the agent runs on its offline rules engine; without `DATABASE_URL` it uses local SQLite files.

```bash
curl localhost:8000/health                                   # -> ok
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"sam","thread_id":"t1","message":"Hi, I am Sam, budget 480k, moving in 2 months"}'
# -> {"reply": "...", "tool_calls": [{"name":"save_memory",...}, {"name":"check_availability",...}], "tokens": {"in": 0, "out": 0}}

pytest -q                                                    # 74 tests (includes a real embedded Postgres)
python scripts/run_tests_json.py --local                     # tests.json against an in-process app
python scripts/run_tests_json.py --url https://<your-agent>  # ... or against a deployment
```

## Deploy (free tier)

1. **Postgres (Neon or Supabase):** create a free project, copy the connection string (`postgresql://...?sslmode=require`). Tables and LangGraph checkpoint tables are created automatically on first start.
2. **Agent API on Hugging Face Spaces (Docker):** create a write token, then
   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_...  DATABASE_URL=postgresql://...  ANTHROPIC_API_KEY=sk-ant-...  DEADCLOSER_ADMIN_KEY=$(openssl rand -hex 24)
   python scripts/deploy_hf.py --space YOUR_USER/dead-closer
   ```
   Keys are set as Space *secrets* from your environment, never committed. API base URL: `https://YOUR_USER-dead-closer.hf.space`. (Render alternative: New -> Blueprint -> this repo, then add `DATABASE_URL` and `ANTHROPIC_API_KEY` in the dashboard; `render.yaml` is included.)
3. **Website on Vercel:** build the static site pointing at your API, then deploy the `docs/` folder:
   ```bash
   python scripts/build_site.py --repo https://github.com/YOUR_USER/dead-closer --api https://YOUR_USER-dead-closer.hf.space/
   npx vercel deploy docs --prod --yes        # or: import the repo in Vercel, Framework "Other", Root Directory "docs"
   ```
4. Check: `<api>/health` -> `ok`; open `<site>/chat.html`; `<site>/admin.html` with your admin key; `python scripts/run_tests_json.py --url <api>`.

## Benchmarks

Everything below runs offline on the rules engine (no model calls) - see [RESULTS.md](RESULTS.md) for methods, assumptions and caveats.

| What | Result | Kind |
|---|---|---|
| First-reply latency, whole engine | p50 **15 ms**, p95 **45 ms**, 100% inside the 60 s SLA | measured |
| Burst of 300 simultaneous leads (SQLite, one instance) | all answered in <= 1.6 s | measured |
| Budget / timeline read correctly (n = 639 / 643) | **96.9% / 93.8%** | measured vs ground truth |
| Safety suite (opt-out, injection, PII, fair housing, honesty, escalation, billing, isolation, off-topic, confirmation, robustness) | **118/118** (first run 92/100; bugs fixed) | measured, author-written |
| `tests.json` (own 10 cases) | **10/10** locally | measured |
| Booking rate: full agent vs no follow-ups vs human-speed (4 h) reply | **30.6%** vs 19.5% vs 12.8% | simulated, assumed lead behaviour |
| Without qualification questions | 33.5% bookings but only 28% billable: **$423 vs $1,025** revenue per 100 leads | simulated |

## Safety, concretely

Off-topic requests are refused (and the conversation state is preserved); prompt injection is detected, logged and deflected - and since actions are not model-chosen there is nothing to hijack; **cancelling a visit or booking a time that was never offered requires an explicit YES**; PII (cards, SSNs, passwords) is redacted before it reaches the DB, checkpoint or model; prices are only quoted from inventory data; no promises about discounts, approvals or appreciation; neighbourhood/people questions are declined (fair housing); abuse, legal threats and distress hand over to a human; STOP unsubscribes (START resumes); proactive follow-ups only 9:00-20:00 local; a visit is billable only for a qualified lead, so the agent has no incentive to pad bookings.

## Honest limits

- The simulated conversion numbers rest on assumed lead behaviour - they compare variants, they do not predict real booking rates.
- **The Claude path (Haiku 5.5) is implemented and unit-tested with a fake client, but has not been run against the live API or the benchmarks.** Run `DEADCLOSER_LLM=claude python -m bench.safety_suite` and `scripts/run_tests_json.py` with your key before relying on it. Cost control: at most two calls per message, `effort: low`, trivial messages skip the model.
- Off-topic and "what do you remember" detection are rule-based (explicit phrases + a domain-vocabulary check); unusual phrasings can slip through either way.
- One server process (per-user locks are in-process); SQLite/Postgres are fine for hundreds of leads/minute.
- Channels (SMS/WhatsApp), calendar and CRM sync are integration points, not included. Timezones are a fixed UTC offset per business.

## Layout

```
deadcloser/   api.py engine.py graph.py agents/ tools.py memory.py billing.py safety.py nlu.py nlg.py llm.py db.py web/
bench/        simulate.py safety_suite.py export_replay.py results/
tests/        flow, safety, API, chat/memory, Claude-backend tests (incl. real Postgres)
scripts/      deploy_hf.py build_site.py run_tests_json.py render_results.py
tests.json  .env.example  Dockerfile  render.yaml
```

MIT licensed.
