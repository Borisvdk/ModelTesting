FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        curl \
        git \
        build-essential \
        ca-certificates \
        locales \
        sudo \
        bash-completion \
        htop \
        jq \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Create workspace directory
WORKDIR /workspace

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Create directories for the project
RUN mkdir -p /workspace/data/exams /workspace/data/results/responses /workspace/config

# Switch to non-root user
USER vscode

# Set Python path
ENV PATH="/home/vscode/.local/bin:${PATH}"

# Install Python tools globally for the user
RUN pip install --user --upgrade pip setuptools wheel && \
    pip install --user \
        black \
        isort \
        flake8 \
        mypy \
        pytest \
        ipython

# The project dependencies will be installed in post-create script