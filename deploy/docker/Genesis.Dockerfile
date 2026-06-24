FROM python:3.12-slim-bookworm

ARG SLOFORGE_SOURCE_COMMIT
WORKDIR /src
COPY pyproject.toml README.md uv.lock ./
COPY python ./python
COPY models ./models
COPY schemas ./schemas
RUN printf '%s\n' "$SLOFORGE_SOURCE_COMMIT" > .sloforge-source-commit \
    && python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 genesis
USER genesis
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=73129 \
    PYTHONNOUSERSITE=1
ENTRYPOINT []
