# sumo_server/Dockerfile

# STEP 1: Start from the stable, full-featured Ubuntu 22.04 LTS image
FROM ubuntu:22.04

# Prevent interactive popups during package installation
ENV DEBIAN_FRONTEND=noninteractive

# STEP 2: Install system dependencies, add PPAs, and install Python/SUMO
RUN apt-get update && \
    apt-get install -y software-properties-common gnupg python3.12-venv && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y python3.12 sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# STEP 3: Create a virtual environment
# This isolates our app's dependencies from the system's Python packages
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the working directory for our application
WORKDIR /app

# Set environment variables so Python can find the 'traci' library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install our Python application's dependencies INTO THE VIRTUAL ENVIRONMENT
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the SUMO map files into the image
COPY . .
COPY Berlin ./Berlin

# Expose the port our server will run on
EXPOSE 5000

# Define the command to start our production server
# This will use the gunicorn from our virtual environment
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]