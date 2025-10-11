# sumo_server/Dockerfile

FROM python:3.12

WORKDIR /app

# --- THIS IS THE UPDATED BLOCK ---
# Install prerequisites, add the official SUMO PPA, and install the latest stable SUMO
RUN apt-get update && \
    apt-get install -y software-properties-common gnupg && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y sumo sumo-tools sumo-doc && \
    rm -rf /var/lib/apt/lists/*
# --- END OF UPDATED BLOCK ---

# Set environment variables so Python can find the traci library
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH}"

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the original Berlin map files
COPY . .
COPY Berlin ./Berlin

# Expose the port and run the application with Gunicorn
EXPOSE 5000
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "server:app"]