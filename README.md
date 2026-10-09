# AI Commerce Operations Agent

A FastAPI service running a LangGraph agent that searches a product catalog,
checks inventory, compares options and makes a grounded recommendation, with
persistent sessions, guarded tool execution, tracing, metrics, evals, and
Docker/Kubernetes deployment files.

It extends my AI Commerce Capstone from "LLM endpoints" into an agent that
chooses actions, executes tools under controls, and can be tested and operated.

## Architecture

```
Client ──HTTP──▶ FastAPI  (API-key auth · Pydantic validation · request IDs · error mapping)
                   │  run_in_threadpool (graph + DB are blocking; event loop stays free)
                   ▼
            AgentService  (per-session lock · metrics · structured logs)
                   │
                   ▼
   LangGraph   plan ──call_tool──▶ act ──▶ plan … ──finish──▶ finalize ──▶ END
                │                   │                           │
            Planner              Executor                  grounding check
       (rules | OpenAI)   validate · permission ·          status, history
                           timeout · retry+backoff
                                    │
                                    ▼
          Tools: search_products · check_inventory · compare_products
                 reserve_stock (write; requires human approval, never agent-callable)
                   │
   Checkpoints: SQLite (dev) | PostgreSQL (multi-replica), keyed by session_id
   Ops: /health · /ready · /metrics (Prometheus) · JSON logs · per-request trace spans
```

### Key design decisions

| Decision | Why |
|---|---|
| **LangGraph** over a plain loop | Explicit nodes and routing are unit-testable; checkpointer persists state after every node; runtime context injects per-request planner/tracer without persisting them. A plain `while` loop would work for this graph, but I'd be re-implementing persistence and resumability. |
| **Planner interface** with a deterministic `RulePlanner` and an `OpenAIPlanner` | CI evals and regression tests run without an API key and are repeatable. The same graph, executor and guards apply to the LLM planner. |
| **All tool calls go through `execute_tool`** | Validation, permission checks, timeouts and bounded retries can't be bypassed by any agent path. |
| **Transient vs permanent errors** | Only `TransientToolError` (503, timeout) is retried. Bad arguments and unknown ids fail fast so the planner can react. |
| **Loop guard + iteration cap** | The same tool with the same args twice ends the run (`loop_detected`); `AGENT_MAX_ITERATIONS` bounds cost and latency (`max_iterations`). |
| **Grounding check in `finalize`** | A recommended product must have come back from search and not be reported out of stock; otherwise the status is `ungrounded` and the answer is replaced. This catches hallucinated ids from an LLM planner. |
| **Graceful degradation** | If inventory is down after retries, the agent still answers, says stock is unverified, and the status is `degraded` (not a 500). |
| **Per-session lock** | Concurrent requests on the same `session_id` are serialized so checkpoints don't interleave. |
| **Postgres checkpoints for >1 replica** | SQLite lives inside one pod; with Postgres any pod can resume any session (tested: `tests/test_postgres.py`). |

Run statuses: `completed`, `degraded`, `max_iterations`, `loop_detected`, `ungrounded`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # optional
API_KEY=dev-key uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API docs.

```bash
curl -s -X POST localhost:8000/v1/agent/run \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"query": "compare noise cancelling headphones under 10k", "session_id": "demo-1"}'

curl -s localhost:8000/v1/agent/sessions/demo-1 -H "X-API-Key: dev-key"
curl -s localhost:8000/metrics
```

Use the LLM planner instead of the rules planner:

```bash
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... uvicorn app.main:app
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: process is up |
| GET | `/ready` | Readiness: checkpoint store reachable |
| GET | `/metrics` | Prometheus: runs by status, tool outcomes, tokens, latency histogram |
| POST | `/v1/search` | Direct product search (no agent) |
| POST | `/v1/agent/run` | Run the agent; returns answer, status, steps, spans, tokens, latency |
| GET | `/v1/agent/sessions/{id}` | Persisted session history |

## Tests

```bash
pytest -q                                   # 43 tests, no external services
TEST_POSTGRES_URL=postgresql://agent:agent@localhost:5432/agentdb pytest -q   # +1 Postgres test
```

| File | Covers |
|---|---|
| `tests/test_tools.py` | Each tool in isolation; executor validation, unknown tool, approval, retries, retry exhaustion, timeout, wrapping tool bugs |
| `tests/test_agent.py` | Happy path, out-of-stock handling, no results, inventory outage, loop guard, iteration cap, blocked write tool, hallucinated and out-of-stock recommendations, token accounting, persistence across restarts, session isolation |
| `tests/test_api.py` | HTTP layer: probes, validation (422), auth (401), session endpoints, metrics |
| `tests/test_concurrency.py` | 40 parallel runs across sessions; parallel runs on one session are serialized |
| `tests/test_regressions.py` | One test per real incident (see below) |
| `tests/test_postgres.py` | Session written by one replica, resumed by another |

## Evals

```bash
python -m evals.run_evals                    # writes evals/report.json
python -m evals.run_evals --min-success 0.9  # non-zero exit below threshold (used in CI)
LLM_PROVIDER=openai python -m evals.run_evals
```

`evals/dataset.json` has 10 cases including out-of-stock, no-match, and an
inventory outage (fault injected via `INVENTORY_FAILURE_RATE=1.0`). Metrics:
task success, tool-call correctness, groundedness, p50/p95 latency, tokens.

Current results (rules planner): task success 1.0, tool correctness 1.0,
grounded 1.0, p50 ≈ 23 ms. The LLM planner has **not** been evaluated yet;
that's the next step and the reason the harness supports `LLM_PROVIDER`.

## Debugging incident: wrong "cheapest" recommendation

**Found by:** the first eval run. Task success was 0.9; case `headphones-cheap`
("cheapest bluetooth headphones") recommended Sony WH-CH720N (₹8,990) instead
of boAt Rockerz 450 (₹1,499). Tool sequence was correct and the answer was
grounded, so only the outcome check caught it.

**Reproduction:** called `search_products` directly with the same input. All
four headphones had the same relevance score (0.667), and ties were broken by
rating, so boAt (rating 4.0) came 4th.

**Root cause:** the agent only checks inventory for the top 3 search results.
The "cheapest" intent was applied at the final pick (choosing the cheapest
among compared products), but retrieval had already cut the cheapest product
before it could be compared. The intent was applied one stage too late.

**Fix:** `search_products` gained a `sort` argument (`relevance`, `price_asc`,
`rating_desc`) that reorders only within a relevance tier, and the planner
derives one intent per run and uses it for both retrieval and the final pick.

**Verification:** three regression tests in `tests/test_regressions.py`
(including one proving price sorting never outranks relevance); eval task
success 0.9 → 1.0 with no other case changing.

**Lesson:** check that ranking intent is applied before any truncation step.
The same class of bug applies to RAG pipelines (top-k cut before reranking).

## Deploy

### Docker Compose (API + Postgres)

```bash
docker compose up --build
```

### Kubernetes / EKS

```bash
# 1. build and push
aws ecr create-repository --repository-name commerce-agent
docker build -t <account>.dkr.ecr.ap-south-1.amazonaws.com/commerce-agent:0.1.0 .
docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/commerce-agent:0.1.0

# 2. secrets (use RDS Postgres for CHECKPOINT_URL on EKS)
kubectl create secret generic commerce-agent-secrets \
  --from-literal=API_KEY=... --from-literal=OPENAI_API_KEY=... \
  --from-literal=CHECKPOINT_URL=postgresql://user:pass@<rds-host>:5432/agentdb

# 3. deploy (set the image in k8s/deployment.yaml first)
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/commerce-agent
kubectl port-forward svc/commerce-agent 8000:80

# roll back a bad release
kubectl rollout undo deployment/commerce-agent
```

For a test cluster (kind/minikube), `kubectl apply -f k8s/postgres-dev.yaml`
provides Postgres. The manifest includes rolling updates with
`maxUnavailable: 0`, readiness/liveness probes, resource limits, non-root and
read-only root filesystem, HPA on CPU (2–6 replicas) and a PodDisruptionBudget.

`.github/workflows/ci.yml` runs tests (with a Postgres service), gates on evals,
uploads the eval report and builds the image.

## Known limitations (stated honestly)

- **Not production traffic.** This is a portfolio project. It has not served real users.
- **Deployment status:** the app, tests, evals and Postgres checkpointing were run
  locally. The Docker image and Kubernetes manifests have not yet been built or applied
  in a cluster from this environment (Docker Hub was not reachable there); the
  manifests are validated YAML. Deploying to a real EKS/kind cluster is the next step.
- **Retrieval is keyword scoring** over a 16-item in-memory catalog. A real system would
  use PostgreSQL + pgvector (or OpenSearch) with hybrid search and a reranker.
- **Inventory is simulated** in-process, with configurable latency and failure rate
  for fault-injection drills.
- **Timeouts don't kill threads.** A timed-out tool call returns control to the
  agent, but its worker thread runs until the call finishes. Real upstream clients
  must also set socket timeouts.
- **LLM planner is unevaluated.** It's implemented with structured output and goes
  through the same guards, but no eval numbers exist for it yet.
- **Single-process session lock.** It serializes runs per session inside one pod.
  Across replicas, two simultaneous requests for one session could still race; a
  Postgres advisory lock would close that gap.
- **No rate limiting** at the app level; expected at the ingress / API gateway.
- **Tracing is in-process** (spans returned in the response and logged as JSON).
  OpenTelemetry/LangSmith export would only touch `app/observability.py`.

## Project layout

```
app/
  main.py            FastAPI app, auth, probes, error handling
  config.py          env-based settings
  schemas.py         API models
  tools.py           tool schemas, implementations, registry, error types
  executor.py        validation, permissions, timeouts, retries
  observability.py   JSON logs, Tracer, Prometheus metrics
  data.py            seed catalog + inventory
  agent/
    graph.py         LangGraph workflow, loop guard, grounding check
    planner.py       Decision model, RulePlanner, OpenAIPlanner
    service.py       AgentService: graph + checkpoints + session locks
    checkpoints.py   SQLite / Postgres checkpoint store
evals/               dataset + harness
tests/               unit, agent, API, concurrency, regression, Postgres
k8s/                 Deployment, Service, HPA, PDB, ConfigMap, secret example, dev Postgres
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```
