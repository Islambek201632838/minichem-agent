# One image for the API and the UI (same code, different command; see compose.yaml).
# Not in the image: .env (keys come from the environment at run time), the AiZynthFinder venv and
# its models (770 MB, older RDKit; mounted if retrosynthesis is wanted), logs and caches (volumes).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 minichem \
    && useradd --system --uid 10001 --gid minichem --home-dir /app --shell /usr/sbin/nologin minichem

WORKDIR /app

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY minichem ./minichem
COPY mcp_server ./mcp_server
COPY app ./app
COPY ui ./ui

# Writable places for the non-root user: traces and request logs, PubChem cache, session database.
RUN mkdir -p /app/logs /app/.cache && chown -R minichem:minichem /app/logs /app/.cache
USER minichem

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
