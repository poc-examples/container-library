FROM python:3.9-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -U pip && \
    pip install -r requirements.txt

COPY src .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
