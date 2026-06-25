FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG SLOFORGE_SOURCE_COMMIT
WORKDIR /src
COPY pyproject.toml README.md uv.lock ./
COPY python ./python
COPY models ./models
COPY schemas ./schemas
RUN printf '%s\n' "$SLOFORGE_SOURCE_COMMIT" > .sloforge-source-commit \
    && python -m pip install --no-cache-dir uv==0.10.2 \
    && uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 genesis
USER genesis
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=73129 \
    PYTHONNOUSERSITE=1 \
    PATH="/src/.venv/bin:$PATH"
ENTRYPOINT []
