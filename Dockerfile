# Use the official Python image from the Docker Hub
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Create a virtual environment
ENV VIRTUAL_ENV=/tmp/.venv
ENV UV_PROJECT_ENVIRONMENT=/tmp/.venv

# Set the virtual environment as the active environment
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends \
 build-essential \
 curl \
 git \
 && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain=nightly --profile=minimal -y
ENV PATH="${PATH}:/root/.cargo/bin"

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync

# Copy the project into the intermediate image
COPY . /app

# Expose Jupyter's default port
EXPOSE 8888

# Define a volume
VOLUME [".:/app"]
