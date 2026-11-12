# Production roadmap (what is missing, in order, and why)

Target level (owner's decision, 2026-11-12): medium production, local. One machine, no cloud
services, only open or free software, nothing built only to look impressive.

Status on 2026-11-12: only item 9 is implemented; the rest is not. The README states the scope and says which of
these are still missing; update that paragraph as items are finished. Each item names the alternative
that was rejected. Do not start an item without the owner's go-ahead.

Evidence from real use that shaped the order: a stale `uvicorn` from an earlier session was still
bound to port 8000 and served OLD code (no `routes` field) while the repo was already newer; two
parallel retrosynthesis questions took ~188 s each (one alone takes 1-2 min), so the API has no
protection against a few users starting heavy searches at once.

| # | Item | Why | Rejected alternative |
|---|---|---|---|
| 1 (must) | A local check script (`scripts/check.sh`: `ruff check`, `ruff format --check`, `pytest -q`) run by a git pre-push hook; a GitHub Actions workflow running the same script is optional, because GitHub is a hosted commercial service | The "definition of done" is only a convention today; the script makes it one command and the hook makes it hard to skip (tests need no network or key) | CI only: it would make the local workflow depend on a hosted service |
| 2 (must) | Dockerfile (uv, non-root user, healthcheck on `/health`) and a compose file for API + UI, tested with an open runtime (Podman or Colima), not only Docker Desktop, whose licence is commercial for larger companies. The AiZynthFinder venv and its 770 MB models are NOT baked in: mounted as a volume | Reproducible local run; removes the "which process is serving which code" problem | One image with AiZynthFinder: needs the older RDKit, doubles size, bakes in models. A plain `launchd` service: not reproducible on another machine |
| 3 (must) | API key auth on `/chat` (keys from environment, compared with `secrets.compare_digest`) and a per-key rate limit | Today anyone who reaches the port spends the owner's Gemini quota and can start 2-minute searches | OAuth/JWT: there are no user accounts to justify it |
| 4 (must) | Concurrency and time limits: async endpoint, per-request timeout, a semaphore capping simultaneous retrosynthesis searches (they are CPU-bound, 120 s each), clear 429/504 responses | Measured today: parallel searches slow each other down | Unlimited threads: one heavy question starves everyone |
| 5 (must) | Persistent conversation memory: `AsyncSqliteSaver` (langgraph-checkpoint-sqlite) for one node, session TTL and a size cap | `MemorySaver` loses all sessions, including the molecule registry, on restart | Redis: extra service with no benefit at this scale. Postgres only when more than one API replica is needed |
| 6 | Structured logs with a request id on every trace line; metrics endpoint (request count, latency, tool errors) | `logs/trace.jsonl` is per tool call, not per request, and has no rotation | Langfuse cloud: sends users' chemistry questions to a third party. Self-hosted Langfuse or OpenTelemetry: possible later, extra services to run |
| 7 | Secrets and config: fail fast at startup if the key is missing, never log settings, the key comes from `.env` or the container's environment and is never baked into an image | Silent misconfiguration shows up as a 503 at the first user question | Baking keys into the image |
| 8 | Hard safety filter for hazardous and controlled substances (a deterministic pre-check before any LLM call) | Today it is a prompt rule checked by one golden question (`t13`); a prompt can be talked around | More prompt text. Needs the owner to choose the list source before any code |
| 9 (DONE 2026-11-12) | Handle-based tool arguments (tools accept `M3`, code resolves it) | Removes the one failure class the eval keeps finding: the model retyping a SMILES inside a sub-agent (2 of 69 runs in the README run, 1 of 24 in `20261112T164829Z`, question `w03`: `CCCc1...` became `CCCCc1...`) | Only detecting it, which is what exists now |

Decision of 2026-11-12: item 9 is done first, before the infrastructure items, because it fixes a
correctness defect that is already measured; the rest do not change answers.

Open question for the owner: the language model is the one hosted, closed service left (Gemini API,
free tier). If "local" must include the model, the option is an open-weights model served by Ollama
(MIT) or llama.cpp (MIT) on the 24 GB Mac. Cost: weaker tool calling, so the whole golden suite would
have to be re-run and the prompts re-tested before it can replace Gemini.
