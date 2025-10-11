# sumo_server/Dockerfile

# STEP 1: Start from the newer Ubuntu 24.04 LTS image
FROM ubuntu:24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# --- UPDATED INSTALLATION BLOCK for 24.04 ---
RUN apt-get update && \
    # 1. Install prerequisites. Python 3.12 is the default `python3` on 24.04, so we install it directly.
    apt-get install -y python3 python3-pip software-properties-common gnupg && \
    \
    # 2. Add the PPA for the latest stable SUMO version.
    #    (This is the step that may fail if the PPA does not yet support Ubuntu 24.04)
    add-apt-repository -y ppa:sumo/stable && \
    \
    # 3. Refresh the package list to include the SUMO PPA
    apt-get update && \
    \
    # 4. Install SUMO from the PPA
    apt-get install -y sumo sumo-tools sumo-doc && \
    \
    # 5. Clean up downloaded package files
    rm -rf /var/lib/apt/lists/*

# Set the working directory for our application
WORKDIR /app

# Set environment variables so Python can find the 'traci' library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install our Python application's dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code and the SUMO map files into the image
COPY . .
COPY Berlin ./Berlin

# Expose the port our server will run on
EXPOSE 5000

# Define the command to start our production server
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]