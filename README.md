# Project Name
Convolutional Neural Network and Docker

## Description
This project contains the code of a convolutional neural network for the MNIST Fashion dataset classification. 

## Table of Contents

- Steps taken
- Usage
- Dockerfile and Docker

## Steps taken
1. Write programme code
2. Create requirements.txt file
3. Install Docker
4. Write Dockerfile
5. Build the Docker image:
    - Command: docker build -t neural_network .
6. Test the Docker image:
    - Command: docker run -it --rm neural_network
7. Push into registry:
    - Command: docker tag neural_network ugnem/neural_network
    - Command: docker push ugnem/neural_network

## Usage

To run the project pull the project from the Hub with command: docker pull ugnem/neural_network:latest. After the image has been pulled run a container with command: docker run -it --name container_name ugnem/neural_network:latest. After the neural network goes through 50 epochs the programme will ask you to name the run and whether you want to save the model parameters with a yes/no answer. After this is done, a tensorboard command is launched and you can open the tensorboard with the localhost link provided. If you do not see some of the model results, please wait until it loads. Depending on how many models you have launched, this might take up to an hour. Close the tensorboard by pressing ctrl+c in the terminal.

## Dockerfile
Below is the Dockerfile used for this project:
    FROM python:3.9-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    EXPOSE 6006
    CMD [ "python", "training.py" ]

Docker Hub link: https://hub.docker.com/r/ugnem/neural_network/tags </br>
Pull link: docker pull ugnem/neural_network:latest
