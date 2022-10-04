FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential freeglut3-dev mesa-common-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libfftw3-dev libopenexr-dev

COPY . /usr/src/lumosquad

WORKDIR /usr/src/lumosquad

RUN make