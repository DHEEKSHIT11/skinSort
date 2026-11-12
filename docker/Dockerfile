# SkinSort model pipeline image — multi-stage, following the course Docker_Labs/Lab2
# pattern (stage 1 trains, stage 2 serves). Build context = repository root.
#
#   docker build -f docker/Dockerfile --target trainer -t skinsort-trainer .
#   docker build -f docker/Dockerfile --target serving -t skinsort-serving .
#   (target serving_base = same API without a baked-in model, used by docker-compose)
#
# trainer: runs src.training.pipeline (fusion head + out-of-fold tau -> bias -> selection ->
#          registry gate -> export). Data, backbone checkpoints and mlflow.db are mounted
#          (docker-compose.yml); the GPU backbone training runs on the GCP VM instead.
# serving: FastAPI /triage with the exported Production model baked in
#          (models/triage_model, made by `python -m src.training.registry export`).

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
COPY requirements-model.txt .
RUN pip install -r requirements-model.txt


# ---------------------------------------------------------------- stage 1: training
FROM base AS trainer
COPY params.yaml ./
COPY Data-Pipeline/dvc.lock Data-Pipeline/dvc.lock
COPY src ./src
ENTRYPOINT ["python", "-m", "src.training.pipeline"]


# ---------------------------------------------------------------- stage 2: serving
# serving_base: the API without a model (docker-compose mounts the model the trainer exported)
FROM base AS serving_base
COPY params.yaml ./
COPY src ./src
ENV MODEL_DIR=/app/models/triage_model
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]

# serving: the deployable image with the exported Production model baked in (Artifact Registry);
# needs models/triage_model from `python -m src.training.registry export`
FROM serving_base AS serving
COPY --chown=app models/triage_model /app/models/triage_model
