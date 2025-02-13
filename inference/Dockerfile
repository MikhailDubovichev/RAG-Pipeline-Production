# inference/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY inference/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference code
COPY inference/ .

# Copy configuration files
COPY config/ config/

# Define environment variables (optional)
# ENV API_KEY=your_api_key_here
# ENV API_BASE=https://api.studio.nebius.ai/v1/

# Expose ports for FastAPI and Gradio
EXPOSE 8080 7860

# Command to run the inference script
CMD ["python", "inference.py"]

