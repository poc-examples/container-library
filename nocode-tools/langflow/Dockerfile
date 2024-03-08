FROM    docker.io/python:3.10-slim

LABEL   maintainer="cengleby86@gmail.com" \
        org.opencontainers.image.source="https://github.com/poc-examples/core-containers/tree/main" \
        org.opencontainers.image.description="Container for AI model development with CUDA-enabled PyTorch"

RUN apt-get update \
        && apt-get install -y \
            gcc \
            g++ \
            git \
            make

ENV HOME=/langflow \
    PATH=/langflow/.local/bin:$PATH

WORKDIR $HOME

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt

COPY start.sh /langflow/

RUN chgrp -R 0 /langflow && \
    chmod -R g+rwX /langflow && \
    chmod +x /langflow/start.sh

USER 1001

EXPOSE 8080

CMD ["./start.sh"]