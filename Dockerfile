FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --python python3.11

# Download COMET + its xlm-roberta-large encoder at build time so the container is fully
# air-gapped at runtime. download_model() returns the absolute checkpoint path; persist it.
RUN python -c "from comet import download_model; \
p = download_model('Unbabel/wmt22-comet-da', saving_directory='/opt/comet'); \
open('/opt/comet/CHECKPOINT', 'w').write(p)"
RUN python -c "from transformers import XLMRobertaTokenizerFast, AutoModel; \
XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large'); \
AutoModel.from_pretrained('xlm-roberta-large')"

COPY scripts/ ./scripts/

ENTRYPOINT ["python", "scripts/evaluate.py"]
