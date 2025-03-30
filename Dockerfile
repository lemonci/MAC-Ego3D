# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# Install necessary dependencies (wget, bzip2, ca-certificates, curl)
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libpcl-dev \
    cmake \
    && apt-get clean

# Download and install Miniconda (latest version)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh

# # Install the NVIDIA package repository for CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda

# Set the PATH to include Conda and avoid needing to use the full path
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda (this is needed to allow conda activate to work)
RUN /opt/conda/bin/conda init bash

# Optional: Create a new conda environment (for example, with Python 3.9)
RUN /opt/conda/bin/conda create -n macego python==3.9 -y

# Activate the conda environment and run commands within it
RUN echo "conda activate macego" >> ~/.bashrc

# Set the shell to use the conda environment for the remaining commands
SHELL ["/opt/conda/bin/conda", "run", "-n", "macego", "/bin/bash", "-c"]

RUN conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
RUN conda install pytorch==2.0.0 torchvision=0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

COPY MAC-Ego3D ./MAC-Ego3D
RUN cd ./MAC-Ego3D && \
    pip install -r ./requirements.txt
RUN pip install pcl

RUN cd ./MAC-Ego3D && pip install ./submodules/diff-gaussian-rasterization
RUN cd ./MAC-Ego3D && pip install ./submodules/simple-knn

RUN cd ./MAC-Ego3D/submodules/fast_gicp && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cd .. && \
    python setup.py install --user

# Default command to run when the container starts
CMD ["bash"]
