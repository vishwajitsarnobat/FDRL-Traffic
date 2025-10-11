# sumo_server/Dockerfile

# STEP 1: Start from the stable, full-featured Ubuntu 22.04 LTS image
FROM ubuntu:22.04

# Prevent interactive popups during package installation
ENV DEBIAN_FRONTEND=noninteractive

# STEP 2: Install system dependencies, add PPAs, and install Python/SUMO
RUN apt-get update && \
    apt-get install -y software-properties-common gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    # Install python3.12-dev and python3.12-venv to ensure all headers and tools are available
    apt-get install -y python3.12 python3.12-dev python3-pip sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# STEP 3: Fix the broken pip/distutils issue in Python 3.12
# This is the crucial new step.
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Set the working directory for our application
WORKDIR /app

# Set environment variables so Python can find the 'traci' library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install our Python application's dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the SUMO map files into the image
COPY . .
COPY Berlin ./Berlin

# Expose the port our server will run on
EXPOSE 5000

# Define the command to start our production server
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]