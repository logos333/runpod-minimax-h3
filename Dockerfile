FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.source="https://github.com/logos333/runpod-minimax-h3" \
      org.opencontainers.image.description="Headless RunPod Serverless worker for MiniMax-H3 (MATLOWAI Fused Turbo INT8 ConvRot)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODELS_ROOT=/runpod-volume/models \
    COMFY_DIR=/app/ComfyUI

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Create isolated Python 3.12 virtualenv + fast uv installer (avoids Ubuntu 24.04 system-pip conflicts)
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_LINK_MODE=copy

RUN python3.12 -m venv /opt/venv && \
    pip install --upgrade pip uv

# Install PyTorch cu130 (recommended by Comfy-Org/MiniMax-H3 for int8_convrot kernels)
RUN uv pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Clone ComfyUI (latest master with native MiniMax-H3 support)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    uv pip install -r /app/ComfyUI/requirements.txt

# Install MiniMax-H3 optimization & long-form custom nodes
RUN git clone --depth 1 https://github.com/ethanfel/ComfyUI-PlagueKind-Nodes-only-sparse.git \
      /app/ComfyUI/custom_nodes/ComfyUI-PlagueKind-Nodes-only-sparse && \
    git clone --depth 1 https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia.git \
      /app/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-LongMedia && \
    git clone --depth 1 https://github.com/matlowai/ComfyUI-MAINodes.git \
      /app/ComfyUI/custom_nodes/ComfyUI-MAINodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
      /app/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    uv pip install -r /app/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt

# Worker runtime dependencies (RunPod SDK, HF fast download, async HTTP, S3/R2 optional)
RUN uv pip install \
    "runpod>=1.7.0" \
    "httpx>=0.27.0" \
    "websockets>=12.0" \
    "huggingface-hub>=0.27.0" \
    "hf-transfer>=0.1.8" \
    "pydantic>=2.7.0" \
    "pydantic-settings>=2.3.0" \
    "boto3>=1.35.0"

COPY docker/ /app/docker/
COPY worker/ /app/worker/

RUN chmod +x /app/docker/start.sh

ENTRYPOINT ["/app/docker/start.sh"]
