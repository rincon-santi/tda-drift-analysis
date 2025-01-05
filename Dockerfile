# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Copy the current directory contents into the container
COPY src/ /app/tda_data_drift/src
COPY requirements.txt /app/tda_data_drift/requirements.txt
COPY __init__.py /app/tda_data_drift/__init__.py
COPY __main__.py /app/tda_data_drift/__main__.py

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r tda_data_drift/requirements.txt

# Command to run the application
ENTRYPOINT ["python", "-m", "tda_data_drift"]