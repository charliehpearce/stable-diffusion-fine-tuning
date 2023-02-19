FROM python:3.8-slim

# Fixes pip build issues
RUN apt-get update
RUN apt-get --yes --no-install-recommends install gcc python3-dev g++

# Copy dependencies to tmp
COPY requirements.txt /tmp/
WORKDIR /tmp
RUN pip install -r requirements.txt
RUN pip install xformers

WORKDIR /app
COPY ./app/app /app/app
COPY ./app/data_mount /tmp/data_mount