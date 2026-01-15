#!/bin/bash

# Deployment script to run on EC2 instance
# This script builds and runs the Docker container

set -e

echo "🚀 Deploying Zillow Property Predictor to EC2..."

# Navigate to project directory
cd "$(dirname "$0")/.."

# Pull latest code (if using git)
if [ -d ".git" ]; then
    echo "📥 Pulling latest code..."
    git pull origin main
fi

# Download artifacts from S3 (if applicable)
if [ ! -z "$S3_BUCKET" ]; then
    echo "📦 Downloading model artifacts from S3..."
    aws s3 sync s3://$S3_BUCKET/artifacts/ ./artifacts/
fi

# Stop existing container
echo "🛑 Stopping existing containers..."
docker-compose -f deploy/docker-compose.yml down || true

# Build Docker image
echo "🔨 Building Docker image..."
docker-compose -f deploy/docker-compose.yml build

# Start container
echo "▶️ Starting application..."
docker-compose -f deploy/docker-compose.yml up -d

# Show logs
echo "📋 Container logs:"
docker-compose -f deploy/docker-compose.yml logs --tail=50

echo ""
echo "✅ Deployment complete!"
echo "🌐 Application should be accessible at http://<your-ec2-public-ip>:8501"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose -f deploy/docker-compose.yml logs -f"
echo "  - Stop app: docker-compose -f deploy/docker-compose.yml down"
echo "  - Restart app: docker-compose -f deploy/docker-compose.yml restart"
