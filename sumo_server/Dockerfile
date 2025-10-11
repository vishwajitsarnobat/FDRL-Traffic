# sumo_server/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install SUMO and its tools from the Debian repositories
# This is the key step to get SUMO into our environment
RUN apt-get update && \
    apt-get install -y sumo sumo-tools && \
    rm -rf /var/lib/apt/lists/*

# Set the SUMO_HOME environment variable (standard path for Debian install)
ENV SUMO_HOME=/usr/share/sumo

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and SUMO scenario files into the container.
# We must copy the server code and the Berlin map files.
COPY . .
COPY Berlin ./Berlin

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=server.py

# The command to run the application using a production Gunicorn server
# --worker-class eventlet is CRITICAL for Socket.IO to work
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]

# Set the SUMO_HOME environment variable (standard path for Debian install)
ENV SUMO_HOME=/usr/share/sumo

# Add SUMO tools to the Python path so it can find the traci library
ENV PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH}"