#!/bin/bash
# Docker Run Script - Automatically runs commands in Docker container

set -e

IMAGE="wikipedia-retrieval:latest"

# Check if Docker is running
if ! docker ps > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if image exists (more reliable check)
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "⚠️  Docker image not found. Building..."
    docker build -t "$IMAGE" .
fi

# Get current directory (works on Linux/macOS)
CURRENT_DIR=$(pwd)

# Run command in Docker container
docker run --rm \
    -v "${CURRENT_DIR}/data:/app/data" \
    -v "${CURRENT_DIR}:/app" \
    "$IMAGE" \
    python "$@"

