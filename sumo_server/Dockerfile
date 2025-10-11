# STEP 1: Start from the stable Ubuntu 22.04 LTS image
FROM ubuntu:22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# STEP 2: Install system dependencies, Python 3.12, SUMO tools, and distutils
RUN apt-get update && \
    apt-get install -y software-properties-common gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-distutils \
        python3.12-venv \
        python3-pip \
        sumo \
        sumo-tools \
        sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# STEP 3: Upgrade pip, setuptools, wheel for Python 3.12
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Environment variables for SUMO tools
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install Python dependencies
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy application code and SUMO maps
COPY . .
COPY Berlin ./Berlin

# Expose server port
EXPOSE 5000

# Command to start server with Gunicorn using eventlet workers
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]
