FROM    docker.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

LABEL   maintainer="cengleby86@gmail.com" \
        org.opencontainers.image.source="https://github.com/poc-examples/fastapi-models/tree/main" \
        org.opencontainers.image.description="Container for AI model development with CUDA-enabled PyTorch"

ENV DEBIAN_FRONTEND="noninteractive" \
    CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1

RUN mkdir /ai-models /.local /var/log/nvidia-mps \
    && apt-get update \
    && apt-get install -y \
        python3 \
        build-essential \
        libomp-dev \
        gcc \
        g++ \
        cmake \
        git \
        make \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
COPY start.sh /app/

RUN python3 -m pip install -r /tmp/requirements.txt && \
    chmod +x start.sh && \
    # Grant root group full access to the necessary directories
    chgrp -R 0 /app /ai-models /.local /var/log/nvidia-mps && \
    chmod -R g+rwX /app /ai-models /.local /var/log/nvidia-mps
COPY src/ /app/

USER 1001

# RUN python3 -m pip install --user -r /tmp/requirements.txt

EXPOSE 8080

CMD ["./start.sh"]
