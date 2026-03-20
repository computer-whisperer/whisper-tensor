# Accuracy validation image for whisper-tensor.
#
# Contains both Python (for reference generation) and Rust (for validation).
# Build and push:
#   docker build -t registry.cjbal.com/whisper-tensor/accuracy:latest -f ci/accuracy/Dockerfile .
#   docker push registry.cjbal.com/whisper-tensor/accuracy:latest

FROM docker.io/library/rust:latest

# System dependencies
RUN apt-get update -qq && apt-get install -y -qq \
    protobuf-compiler \
    libopenblas-dev \
    python3 \
    python3-pip \
    python3-venv \
    git-lfs \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies in a venv (avoids PEP 668 externally-managed issues)
RUN python3 -m venv /opt/accuracy-venv
ENV PATH="/opt/accuracy-venv/bin:$PATH"
RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    safetensors \
    accelerate

# Pre-install Rust components
RUN rustup component add clippy rustfmt

# Git LFS
RUN git lfs install --system

# Default working directory
WORKDIR /builds
