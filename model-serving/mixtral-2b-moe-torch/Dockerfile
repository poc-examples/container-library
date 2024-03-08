# Start with a CUDA-enabled PyTorch base image
FROM docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /app

# Install Python and Pip (if not already included in the base image)
RUN mkdir /ai-models \
    && apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Install Hugging Face Transformers
COPY start.sh /app/
RUN chmod +x start.sh

# Copy your script into the container
COPY src/ /app/

# Expose any ports your application might need (optional, depends on your application)
# For example, if your app has a web interface on port 8080:
EXPOSE 8080

# Set the command to run your script
CMD ["python3", "/app/model.py"]
