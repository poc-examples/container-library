FROM    docker.io/python:3.10-slim

LABEL   maintainer="cengleby86@gmail.com" \
        org.opencontainers.image.source="https://github.com/poc-examples/core-containers/tree/main" \
        org.opencontainers.image.description="Container for Rapid Prototyping LangFlow Apps"

RUN apt-get update \
        && apt-get install -y \
            gcc \
            g++ \
            git \
            make

ENV HOME=/langflow \
    PATH=/langflow/.local/bin:$PATH \
    LANGFLOW_CONFIG_DIR=/langflow/.cache/langflow \
    LANGFLOW_SAVE_DB_IN_CONFIG_DIR=true \
    DATABASE_PATH=/langflow/.cache/langflow/langflow.db

WORKDIR $HOME

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt --user

COPY start.sh /langflow/

RUN mkdir -p /langflow/.cache/langflow && \
    chgrp -R 0 /langflow && \
    chmod -R g+rwX /langflow && \
    chmod +x /langflow/start.sh

EXPOSE 8080

USER 1001

CMD ["./start.sh"]