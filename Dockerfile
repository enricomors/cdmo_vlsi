# Pulls an image with minizin
FROM minizinc/minizinc:latest

# Setting the base folder for the container
WORKDIR /project

# Copy all the content of this folder into the container
COPY . .

# Installing python and the required libraries
RUN apt-get update \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && python3 -m pip install -r requirements.txt

# What to do when the container starts