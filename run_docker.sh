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

# Always rebuild from scratch to ensure dependencies are up to date
echo "🔨 Building Docker image from scratch (--no-cache)..."
docker build --no-cache -t "$IMAGE" .

# Get current directory (works on Linux/macOS)
CURRENT_DIR=$(pwd)

# Run command in Docker container
# CSV files are in the image (/app/data and /app/Data)
# When we mount /app, it overrides image files, so we need to preserve CSV files
# Solution: Copy CSV files from a temporary container before mounting
TEMP_CONTAINER=$(docker create "$IMAGE")
docker cp "$TEMP_CONTAINER:/app/Data" "${CURRENT_DIR}/Data_from_image" 2>/dev/null || true
docker cp "$TEMP_CONTAINER:/app/data" "${CURRENT_DIR}/data_from_image" 2>/dev/null || true
docker rm "$TEMP_CONTAINER" > /dev/null 2>&1 || true

# Copy CSV files to mounted location if they don't exist
if [ ! -z "$(ls ${CURRENT_DIR}/Data_from_image/*.csv 2>/dev/null)" ]; then
  mkdir -p "${CURRENT_DIR}/Data" "${CURRENT_DIR}/data"
  cp "${CURRENT_DIR}/Data_from_image"/*.csv "${CURRENT_DIR}/Data/" 2>/dev/null || true
  cp "${CURRENT_DIR}/data_from_image"/*.csv "${CURRENT_DIR}/data/" 2>/dev/null || true
  rm -rf "${CURRENT_DIR}/Data_from_image" "${CURRENT_DIR}/data_from_image" 2>/dev/null || true
fi

# Run the actual command with mounted filesystem
docker run --rm \
    -v "${CURRENT_DIR}:/app" \
    "$IMAGE" \
    python "$@"

