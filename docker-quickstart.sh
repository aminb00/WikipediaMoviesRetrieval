#!/bin/bash
# Docker Quick Start Script
# Makes it easy to run the Wikipedia Movies Retrieval System in Docker

set -e

echo "=========================================="
echo "Wikipedia Movies Retrieval - Docker Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ docker-compose is not available"
    exit 1
fi

echo ""
echo "Building Docker image..."
docker build -t wikipedia-retrieval:latest .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "=========================================="
echo "Quick Start Commands:"
echo "=========================================="
echo ""
echo "1. Download dataset (first time only):"
echo "   docker run --rm -v \$(pwd)/data:/app/data wikipedia-retrieval:latest python download_dataset.py"
echo ""
echo "2. Run tests:"
echo "   docker run --rm -v \$(pwd)/data:/app/data -v \$(pwd):/app wikipedia-retrieval:latest python test_cli.py"
echo ""
echo "3. Build memory index:"
echo "   docker run --rm -v \$(pwd)/data:/app/data -v \$(pwd):/app wikipedia-retrieval:latest python cli.py build --mode=memory --csv data/"
echo ""
echo "4. Search (example):"
echo "   docker run --rm -v \$(pwd)/data:/app/data -v \$(pwd):/app wikipedia-retrieval:latest python cli.py search --mode=memory --model=ltc.ltc --topk=5 --query \"space adventure\""
echo ""
echo "5. Interactive shell:"
echo "   docker run --rm -it -v \$(pwd)/data:/app/data -v \$(pwd):/app wikipedia-retrieval:latest /bin/bash"
echo ""
echo "=========================================="
echo "Or use docker-compose for easier management:"
echo "=========================================="
echo ""
echo "   $COMPOSE_CMD up -d    # Start container"
echo "   $COMPOSE_CMD exec wikipedia-retrieval bash  # Enter container"
echo "   $COMPOSE_CMD down     # Stop container"
echo ""

