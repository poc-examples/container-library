# Use the official Python image as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
