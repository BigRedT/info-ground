# FROM nvcr.io/nvidia/pytorch:19.11-py3
# FROM continuumio/miniconda3
ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# Install dev tools mainly for interactive usage
RUN apt-get update && apt-get install -y \
    screen \
    emacs \
    htop \
    python3-dev \
    gcc \
    curl \
    wget \
    unzip \
    rsync \
    less

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH
ENV PATH=/miniconda/envs/context-regions/bin:$PATH

# Setup coding environment
ENV HOME /home
ARG WORKSPACE_DIR=$HOME/workspace
ARG DATASET_DIR=$HOME/dataset
RUN mkdir -p $WORKSPACE_DIR $DATASET_DIR 
COPY docker/.screenrc $HOME

COPY environment.yml $HOME
RUN conda env create -f $HOME/environment.yml

SHELL ["/bin/bash", "-c"]
RUN conda init bash 
RUN source /home/.bashrc