# syntax=docker/dockerfile:1.7
# GPU-enabled training image: extends the CPU runtime with a CUDA-capable
# PyTorch build and the optional `lerobot` fine-tuning dependency group.
# Build with: docker build -f docker/training/Dockerfile -t vla-eval-training .
# Run with:   docker run --gpus all vla-eval-training vla-eval train --model openvla --dataset pusht

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/data/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN groupadd --gid 1000 vlaeval && \
    useradd --uid 1000 --gid vlaeval --shell /bin/bash --create-home vlaeval

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt && \
    pip install --no-deps -e ".[lerobot]" || pip install --no-deps -e . && \
    mkdir -p /data/hf_cache /app/data /app/models /app/mlartifacts && \
    chown -R vlaeval:vlaeval /app /data

USER vlaeval

ENTRYPOINT ["vla-eval"]
CMD ["--help"]
