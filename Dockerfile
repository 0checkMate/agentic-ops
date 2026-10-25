FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml requirements.txt README.md* ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -c requirements.txt .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANGGRAPH_STRICT_MSGPACK=true
ENV PATH=/opt/venv/bin:$PATH
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=build /opt/venv /opt/venv
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn agentic_ops.api.app:create_default_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]