FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    OPENML_CONFIG_DIR=/home/appuser/.openml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && update-ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

RUN useradd -m appuser

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY . .
RUN chown -R appuser:appuser /workspace

USER appuser

CMD ["bash"]

