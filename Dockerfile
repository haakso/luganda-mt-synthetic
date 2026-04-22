FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download COMET at build time so the container is fully air-gapped at runtime
RUN python3.11 -c "from comet import download_model; download_model('wmt22-comet-da', saving_directory='/opt/comet')"

COPY scripts/ ./scripts/

ENTRYPOINT ["python3.11", "scripts/evaluate.py"]
