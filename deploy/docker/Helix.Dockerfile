FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG SLOFORGE_SOURCE_COMMIT
WORKDIR /src
COPY pyproject.toml README.md uv.lock ./
COPY python ./python
COPY schemas ./schemas
COPY scenarios ./scenarios
RUN printf '%s\n' "$SLOFORGE_SOURCE_COMMIT" > .sloforge-source-commit \
    && python -m pip install --no-cache-dir uv==0.10.2 \
    && uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 helix
USER helix
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=41 \
    PYTHONNOUSERSITE=1 \
    SLOFORGE_HELIX_ALLOW_GPU=0 \
    SLOFORGE_HELIX_ALLOW_MULTI_NODE=0 \
    SLOFORGE_HELIX_ALLOW_EXTERNAL_API=0 \
    SLOFORGE_HELIX_ALLOW_PRODUCTION_CAPTURE=0 \
    SLOFORGE_HELIX_ALLOW_EXTERNAL_SIDE_EFFECTS=0 \
    SLOFORGE_HELIX_ALLOW_EXTERNAL_DEPLOYMENT=0 \
    SLOFORGE_HELIX_ALLOW_LIVE_PROMOTION=0 \
    PATH="/src/.venv/bin:$PATH"
ENTRYPOINT []
