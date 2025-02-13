# data_preparation/Dockerfile

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
COPY data_preparation/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data preparation code
COPY data_preparation/ .

# Copy configuration files
COPY config/ config/

# Define environment variables (optional)
# ENV API_KEY=your_api_key_here
# ENV API_BASE=https://api.studio.nebius.ai/v1/

# Expose ports if necessary (usually not required for batch jobs)
EXPOSE 8080

# Command to run the data preparation script
CMD ["python", "data_preparation.py"]

