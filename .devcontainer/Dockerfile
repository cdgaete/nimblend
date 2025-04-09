FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies including Java for Duke
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libopenblas-dev \
    curl \
    vim \
    less \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set working directory
WORKDIR /workspaces/nimblend

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements before other files for better caching
COPY pyproject.toml ./

# Install dependencies (tools needed for development)
RUN pip install pre-commit ruff mypy jupyter

# Switch to non-root user
USER $USERNAME
