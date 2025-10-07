FROM nvcr.io/nvidia/nvhpc:24.5-devel-cuda12.4-ubuntu22.04 AS cuda-base
# NOTE(cmo): Somewhere along the line cmake/nvcc started using --option-file for
# specifying paths and flags, and this doens't seem supported by
# intellisense/compile_commands.json. This is fine in 24.5, which is relatively
# on par with most HPC too.

RUN apt update && apt install -y git curl libcurl4-openssl-dev

WORKDIR /src/hdf5
COPY .devcontainer/scripts/install_hdf5.sh .
RUN ./install_hdf5.sh "/usr/local/hdf5"

WORKDIR /src/kokkos
COPY .devcontainer/scripts/kokkos/*.sh .
RUN ./default_cuda_setup_debug.sh
RUN ./default_cuda_setup.sh

# NOTE(cmo): Based on https://github.com/adalundhe/dev-images/blob/main/cpp/Dockerfile.cpp
ARG USERNAME=devcontainer
ARG USER_UID=5000
ARG USER_GID=$USER_UID

# https://happihacking.com/blog/posts/2024/dev-containers-uids/ for 24.04
# RUN userdel -r ubuntu

RUN apt update && apt install -y sudo

RUN groupadd --gid $USER_GID $USERNAME \
    && adduser --disabled-password --gecos "" --uid $USER_UID --gid $USER_GID $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV HOME=/home/$USERNAME
USER $USERNAME

ENV HDF5_ROOT="/usr/local/hdf5"
ENV Kokkos_ROOT="/usr/local/kokkos"
ENV CUDA_DEVCONTAINER=1
ENV HDF5_USE_FILE_LOCKING=FALSE
