# SkinSort model pipeline image — multi-stage, following the course Docker_Labs/Lab2
# pattern (stage 1 trains, stage 2 serves). Build context = repository root.
#
#   docker build -f docker/Dockerfile --target trainer -t skinsort-trainer .
#   docker build -f docker/Dockerfile --target serving -t skinsort-serving .
#   (target serving_base = same API without a baked-in model, used by docker-compose)
#
# trainer: runs any training module; default src.training.pipeline (fusion head + out-of-fold tau
#          -> bias -> selection -> registry gate -> export). Data, backbone checkpoints and mlflow.db
#          are mounted (docker-compose.yml). The image backbones are trained in the same image on a
#          GPU machine (CUDA PyTorch build arg, docker-compose profile "gpu"):
#   docker build -f docker/Dockerfile --target trainer \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/<cuda version, e.g. cu128> -t skinsort-trainer-gpu .
#   docker run --gpus all --shm-size=2g -v "$PWD:$PWD" -w "$PWD" skinsort-trainer-gpu src.training.train
# serving: FastAPI /triage with the exported Production model baked in
#          (models/triage_model, made by `python -m src.training.registry export`).

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# CPU-only PyTorch by default keeps the image small; pass a CUDA wheel index to train backbones on a GPU
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch==2.15.0 torchvision==0.29.2
COPY requirements-model.txt .
RUN pip install -r requirements-model.txt


# ---------------------------------------------------------------- stage 1: training
FROM base AS trainer
COPY params.yaml ./
COPY Data-Pipeline/dvc.lock Data-Pipeline/dvc.lock
COPY src ./src
# `docker run skinsort-trainer` = the model pipeline; `docker run skinsort-trainer src.training.train` = backbones
ENTRYPOINT ["python", "-m"]
CMD ["src.training.pipeline"]


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
