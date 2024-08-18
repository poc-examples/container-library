FROM    docker.io/python:3.10-slim

LABEL   maintainer="cengleby86@gmail.com" \
        org.opencontainers.image.source="https://github.com/poc-examples/container-library/tree/main" \
        org.opencontainers.image.description="Container for running KFP jobs"

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt --user
