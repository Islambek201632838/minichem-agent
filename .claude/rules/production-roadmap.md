# Production roadmap (what is missing, in order, and why)

Target level (owner's decision, 2026-11-12): medium production, local. One machine, only free
software, nothing built only to look impressive. Clarified by the owner on 2026-11-13: "no cloud services"
was too strict; what is excluded is paid or closed software the owner cannot get. A free hosted API or
website may be used instead of a heavy local install (see CLAUDE.md, Engineering stance). Items below were
decided and built under the stricter reading; revisit a local component only when a free hosted
replacement is shown to be reliable (status.md lists the candidates).

Status on 2026-11-12 (evening): items 1-7 and 9 are implemented (owner's go-ahead for 1-7 given the same day);
item 8 is not, it needs the owner's choice of list. The README states the scope; update that paragraph as
items are finished. Each item names the alternative that was rejected. Do not start an item without the
owner's go-ahead. What was verified and what was not is in the "Implementation notes" below the table.

Evidence from real use that shaped the order: a stale `uvicorn` from an earlier session was still
bound to port 8000 and served OLD code (no `routes` field) while the repo was already newer; two
parallel retrosynthesis questions took ~188 s each (one alone takes 1-2 min), so the API has no
protection against a few users starting heavy searches at once.

| # | Item | Why | Rejected alternative |
|---|---|---|---|
| 1 (DONE) | A local check script (`scripts/check.sh`: `ruff check`, `ruff format --check`, `pytest -q`) run by a git pre-push hook; a GitHub Actions workflow running the same script is optional, because GitHub is a hosted commercial service | The "definition of done" is only a convention today; the script makes it one command and the hook makes it hard to skip (tests need no network or key) | CI only: it would make the local workflow depend on a hosted service |
| 2 (DONE, image not built) | Dockerfile (uv, non-root user, healthcheck on `/health`) and a compose file for API + UI, tested with an open runtime (Podman or Colima), not only Docker Desktop, whose licence is commercial for larger companies. The AiZynthFinder venv and its 770 MB models are NOT baked in: mounted as a volume | Reproducible local run; removes the "which process is serving which code" problem | One image with AiZynthFinder: needs the older RDKit, doubles size, bakes in models. A plain `launchd` service: not reproducible on another machine |
| 3 (DONE) | API key auth on `/chat` (keys from environment, compared with `secrets.compare_digest`) and a per-key rate limit | Today anyone who reaches the port spends the owner's Gemini quota and can start 2-minute searches | OAuth/JWT: there are no user accounts to justify it |
| 4 (DONE) | Concurrency and time limits: async endpoint, per-request timeout, a semaphore capping simultaneous retrosynthesis searches (they are CPU-bound, 120 s each), clear 429/504 responses | Measured today: parallel searches slow each other down | Unlimited threads: one heavy question starves everyone |
| 5 (DONE) | Persistent conversation memory: `AsyncSqliteSaver` (langgraph-checkpoint-sqlite) for one node, session TTL and a size cap | `MemorySaver` loses all sessions, including the molecule registry, on restart | Redis: extra service with no benefit at this scale. Postgres only when more than one API replica is needed |
| 6 (DONE) | Structured logs with a request id on every trace line; metrics endpoint (request count, latency, tool errors) | `logs/trace.jsonl` is per tool call, not per request, and has no rotation | Langfuse cloud: sends users' chemistry questions to a third party. Self-hosted Langfuse or OpenTelemetry: possible later, extra services to run |
| 7 (DONE) | Secrets and config: fail fast at startup if the key is missing, never log settings, the key comes from `.env` or the container's environment and is never baked into an image | Silent misconfiguration shows up as a 503 at the first user question | Baking keys into the image |
| 8 | Hard safety filter for hazardous and controlled substances (a deterministic pre-check before any LLM call) | Today it is a prompt rule checked by one golden question (`t13`); a prompt can be talked around | More prompt text. Needs the owner to choose the list source before any code |
| 9 (DONE 2026-11-12) | Handle-based tool arguments (tools accept `M3`, code resolves it) | Removes the one failure class the eval keeps finding: the model retyping a SMILES inside a sub-agent (2 of 69 runs in the README run, 1 of 24 in `20261112T164829Z`, question `w03`: `CCCc1...` became `CCCCc1...`) | Only detecting it, which is what exists now |

Decision of 2026-11-12: item 9 is done first, before the infrastructure items, because it fixes a
correctness defect that is already measured; the rest do not change answers.

Open question for the owner: the language model is the one hosted, closed service left (Gemini API,
free tier). If "local" must include the model, the option is an open-weights model served by Ollama
(MIT) or llama.cpp (MIT) on the 24 GB Mac. Cost: weaker tool calling, so the whole golden suite would
have to be re-run and the prompts re-tested before it can replace Gemini.

## Implementation notes (2026-11-12)

Real runs: the API on port 8200 with the real Gemini model and the real MCP server; unit tests for
everything else (210 tests, no network). Not verified items are listed per number.

1. `scripts/check.sh`, `scripts/hooks/pre-push`, `scripts/install-hooks.sh` (sets git's `core.hooksPath` to the
   versioned folder, so nothing is copied into `.git/hooks`). The owner runs the installer once per clone; it
   was NOT run here and the hook was only executed as a script, never through a real `git push`. GitHub Actions
   was left out on purpose (hosted service); the same script can be called from a workflow later.
2. `Dockerfile` (python:3.12-slim-bookworm, uv 0.13.6 copied from its official image, `uv sync --frozen --no-dev`,
   uid 10001, `HEALTHCHECK` on `/health` with the standard library) and `compose.yaml` (API + UI, ports on
   127.0.0.1, `AUTH_REQUIRED=true`, keys via `env_file`, volumes for logs/cache, `cap_drop: ALL`,
   `no-new-privileges`). Checked: `docker compose config` on a copy with a fake `.env`. NOT checked: the image
   build and any container run (the daemon was off; starting Docker Desktop was not done on purpose, its licence
   is commercial for larger companies). I did not set `read_only` on the containers because it could not be
   tested (Streamlit and RDKit may write to the home directory). Retrosynthesis is off in compose: the host's
   AiZynthFinder venv is a macOS build.
3. `app/security.py`: keys from `API_KEYS` (comma separated, at least 16 characters, compared with
   `secrets.compare_digest` against every key, no early exit), header `X-API-Key`, per-key sliding-window limit
   (default 20 per minute, in memory). Default with no keys: no authentication (the UI and local use keep working;
   the README says so and the API logs a warning). `AUTH_REQUIRED=true` forbids that mode (compose sets it).
   Rejected: OAuth/JWT (no user accounts); a limit on failed key attempts (keys are 16+ characters, and nothing is
   exposed beyond 127.0.0.1 by default).
4. The endpoint was already `async`. Added: `asyncio.timeout` per turn (504), a cap on questions running at once
   (429 + `Retry-After`, checked without waiting so the rejection is immediate), one turn per `session_id` at a
   time (409, two parallel messages would interleave one conversation), and `minichem/concurrency.py`: the two
   retrosynthesis tools share `RETRO_MAX_CONCURRENT` slots (default 1) and wait up to 150 s for one, then return an
   error VALUE (tools never raise into the agent loop). Rejected: rejecting the search with HTTP 429 (the model
   would not know why; a waiting search is cheaper than a failed question). Verified only with a fake tool; the
   real retrosynthesis server was not started, and whether cancelling a timed-out request also stops a running
   AiZynthFinder search was not checked.
5. `langgraph-checkpoint-sqlite==3.1.3` (MIT; pulls `aiosqlite` MIT and `sqlite-vec` MIT/Apache-2.0, which this
   project does not use) and `minichem/sessions.py`. `AsyncSqliteSaver.aprune` exists in the installed version
   but raises NotImplementedError, so compaction is plain SQL on `checkpoints` and `writes` (keep the newest
   checkpoint id of the thread; this graph never resumes older ones and has no delta channels). TTL 7 days, 200
   sessions, 200 turns per session, all settings. Only the API uses the file; the CLI and the eval keep
   `MemorySaver` (otherwise eval questions would land in the owner's conversation store). Verified with the real
   model: a molecule handle from before a server restart was found in a new process (`M1` of the old session).
   Not verified: behaviour under many sessions and a database that is several hundred MB.
6. `minichem/jsonl.py` (standard-library `RotatingFileHandler`, 5 MB, 3 backups) for `logs/trace.jsonl` and the new
   `logs/api.jsonl`; `request_id` on every trace line; `GET /metrics` with `prometheus-client==0.26.0`
   (Apache-2.0 / BSD-2, reused instead of hand-written counters). The key never reaches a log, only
   `sha256(key)[:8]`. Metric paths are route templates (`unmatched` for 404s) so a scanner cannot create unlimited
   labels. Rejected again: Langfuse (see table). Not done: shipping metrics anywhere; they are only exposed.
7. `Settings.startup_problems()` / `check_startup()` run first in the API lifespan; messages never contain values.
   An empty `GOOGLE_API_KEY=` counts as missing (it used to pass). No code logs the settings object; the secret
   fields are `SecretStr` (tested). `REQUIRE_LLM_KEY=false` skips only the Gemini-key check, for tests; tests
   also clear the key variables so a developer's `.env` does not leak into them.

Found while testing item 2/7: `scripts/serve.sh stop` left the API and UI running (the pid file held a
background subshell, not the server), which can produce exactly the "stale uvicorn serving old code" problem
named above. Fixed with `exec` in the two start lines; checked: `start` then `stop` on ports 8201/8601 leaves
nothing listening.
