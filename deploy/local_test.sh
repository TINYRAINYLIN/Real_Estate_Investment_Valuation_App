#!/bin/bash

# Local testing script for Docker container
# Run this before deploying to AWS

set -e

echo "🧪 Testing Zillow Property Predictor locally..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if model artifacts exist
if [ ! -f "artifacts/best_ridge.pkl" ]; then
    echo "❌ Model artifacts not found in artifacts/ directory"
    echo "Please ensure model files are present before testing"
    exit 1
fi

echo "✅ Model artifacts found"

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t zillow-predictor:test .

# Stop any existing container
echo "🛑 Stopping existing containers..."
docker stop zillow-test 2>/dev/null || true
docker rm zillow-test 2>/dev/null || true

# Run container
echo "▶️ Starting container..."
docker run -d \
    --name zillow-test \
    -p 8501:8501 \
    -v "$(pwd)/artifacts:/app/artifacts:ro" \
    zillow-predictor:test

# Wait for container to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if container is running
if docker ps | grep -q zillow-test; then
    echo "✅ Container is running!"
    echo ""
    echo "🌐 Application available at: http://localhost:8501"
    echo ""
    echo "📋 View logs: docker logs -f zillow-test"
    echo "🛑 Stop container: docker stop zillow-test"
    echo ""
    
    # Show logs
    echo "Recent logs:"
    docker logs --tail=20 zillow-test
else
    echo "❌ Container failed to start"
    echo "Logs:"
    docker logs zillow-test
    exit 1
fi
