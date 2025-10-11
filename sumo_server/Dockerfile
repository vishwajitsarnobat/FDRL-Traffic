# sumo_server/Dockerfile

# STEP 1: Start from the stable Ubuntu 22.04 image
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# STEP 2: Install Python 3.12 and prerequisites for SUMO PPA
RUN apt-get update && \
    apt-get install -y python3.12 python3-pip software-properties-common gnupg && \
    rm -rf /var/lib/apt/lists/*

# STEP 3: Add the official SUMO PPA and install the latest SUMO
RUN add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Set environment variables so Python can find the traci library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code and the original Berlin map files
COPY . .
COPY Berlin ./Berlin

# Expose the port and run the application with Gunicorn
EXPOSE 5000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]