# sumo_server/Dockerfile

# STEP 1: Start from the official Python 3.12 image.
# This image is based on Debian Bookworm and is configured to allow pip installs.
FROM python:3.12

# STEP 2: Install prerequisites and add the SUMO PPA.
# We install `software-properties-common` from the Debian testing repository.
RUN apt-get update && \
    apt-get install -y gnupg lsb-release && \
    echo "deb http://deb.debian.org/debian testing main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -t testing -y software-properties-common && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for our application
WORKDIR /app

# Set environment variables so Python can find the 'traci' library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools"

# Copy and install our Python application's dependencies using pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the SUMO map files into the image
COPY . .
COPY Berlin ./Berlin

# Expose the port our server will run on
EXPOSE 5000

# Define the command to start our production server
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]