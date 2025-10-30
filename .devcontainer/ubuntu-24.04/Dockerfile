FROM mcr.microsoft.com/devcontainers/cpp:dev-ubuntu-24.04

RUN apt update && apt install -y git curl libcurl4-openssl-dev libopenmpi-dev

WORKDIR /src/hdf5
COPY .devcontainer/scripts/install_hdf5.sh .
RUN ./install_hdf5.sh "/usr/local/hdf5"

# NOTE(cmo): Based on https://github.com/adalundhe/dev-images/blob/main/cpp/Dockerfile.cpp
ARG USERNAME=devcontainer
ARG USER_UID=5000
ARG USER_GID=$USER_UID

# https://happihacking.com/blog/posts/2024/dev-containers-uids/ for 24.04
RUN apt install -y sudo

ENV HDF5_ROOT="/usr/local/hdf5"
ENV HDF5_USE_FILE_LOCKING=FALSE
