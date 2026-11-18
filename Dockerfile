FROM ghcr.io/astral-sh/uv:0.8.22-python3.12-bookworm-slim@sha256:28df4bbd896cf66a224f2e0cb22240a9a2b9803a3a13519bcadf2e9fdd68c632 AS base
ARG UV_HTTP_TIMEOUT=300
ARG UV_CONCURRENT_DOWNLOADS=2
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

FROM base AS test
COPY tests ./tests
COPY docs ./docs
COPY examples ./examples
COPY scripts ./scripts
COPY AGENTS.md CONTRIBUTING.md CITATION.cff CHANGELOG.md SOURCE.json Dockerfile ./
COPY .gitignore .dockerignore .python-version ./
COPY assets ./assets
COPY .agents ./.agents
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-editable
CMD ["sh", "-c", "uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest"]

FROM test AS test-adapters
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-editable --extra all

FROM test-adapters AS release-check
CMD ["uv", "run", "--no-sync", "python", "scripts/validate_release.py", "--with-adapters"]

FROM base AS gpu
RUN --mount=type=cache,target=/root/.cache/uv test "$(uname -m)" = x86_64 && uv sync --locked --no-dev --no-editable --extra gpu
COPY examples ./examples
ENV JAX_PLATFORMS=cuda
COPY Dockerfile ./
CMD ["uv", "run", "--no-sync", "python", "-c", "import jax; print(jax.devices())"]

FROM test-adapters AS test-gpu
RUN --mount=type=cache,target=/root/.cache/uv test "$(uname -m)" = x86_64 && uv sync --locked --no-editable --extra all --extra gpu
ENV JAX_PLATFORMS=cuda

FROM base AS runtime
CMD ["uv", "run", "--no-sync", "python", "-c", "import jax; from figgie_jax.env import reset; print(reset(jax.random.key(0)).hands)"]
