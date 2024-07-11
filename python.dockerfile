# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && /opt/miniconda/bin/conda clean --all --yes \
    && ln -s /opt/miniconda/bin/conda /usr/bin/conda

# Update conda and install necessary packages
RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge pyopengl

# Install libGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Update conda
RUN conda update -n base -c defaults conda