# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies needed for compiling certain packages and GIS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces runs with user ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR /home/user/app

# Copy the requirements file and install dependencies
COPY --chown=user:user backend-requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend-requirements.txt

# Copy the backend source code and checkpoint folder
COPY --chown=user:user flood-detection-src ./flood-detection-src
COPY --chown=user:user checkpoint-v3 ./checkpoint-v3

# Create directory for active jobs
RUN mkdir -p /home/user/app/.jobs

# Expose the default port Hugging Face Spaces expects
EXPOSE 7860

# Start uvicorn server pointing to our api
CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "flood-detection-src"]
