# Use Debian as the base image
FROM debian:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    openjdk-11-jdk \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (Python 3.11)
RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

# Create a new Conda environment with Python 3.11
RUN conda create -n myenv python=3.11 -y && conda clean --all -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install required Python packages
RUN conda install -n myenv -y \
    pyspark \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    pyarrow \
    transformers \
    accelerate \
    duckdb \
    s3fs \
    umap-learn \
    smart-open \
    sqlalchemy \
    pytest \
    && pip install datasets neo4j onnxruntime seqeval gensim numba 

# Set working directory
WORKDIR /app

# Copy the script
COPY . .

# Entry point to execute the script with arguments
ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "script.py"]
