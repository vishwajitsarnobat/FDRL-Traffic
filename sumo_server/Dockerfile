# sumo_server/Dockerfile

# STEP 1: Start from the stable, full-featured Ubuntu 22.04 LTS image
FROM ubuntu:22.04

# Prevent interactive popups during package installation
ENV DEBIAN_FRONTEND=noninteractive

# --- THE CORRECTED, COMBINED INSTALLATION BLOCK ---
RUN apt-get update && \
    # 1. Install the basic tools needed to add new software repositories
    apt-get install -y software-properties-common gnupg && \
    \
    # 2. Add the PPA for the latest Python versions ("deadsnakes")
    add-apt-repository -y ppa:deadsnakes/ppa && \
    \
    # 3. Add the PPA for the latest stable SUMO version
    add-apt-repository -y ppa:sumo/stable && \
    \
    # 4. Refresh the package list AGAIN to include software from our new PPAs
    apt-get update && \
    \
    # 5. Now, install everything we need: Python 3.12, pip, and SUMO
    apt-get install -y python3.12 python3-pip sumo sumo-tools sumo-doc && \
    \
    # 6. Clean up downloaded package files to keep the final image smaller
    rm -rf /var/lib/apt/lists/*

# Set the working directory for our application
WORKDIR /app

# Set environment variables so Python can find the 'traci' library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install our Python application's dependencies
COPY requirements.txt .
# Use python3.12's pip to install the packages
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy the application code and the SUMO map files into the image
COPY . .
COPY Berlin ./Berlin

# Expose the port our server will run on
EXPOSE 5000

# Define the command to start our production server
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]