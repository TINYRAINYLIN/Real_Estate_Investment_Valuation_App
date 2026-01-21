#!/bin/bash

# Upload model artifacts to S3
# Usage: ./upload_to_s3.sh <bucket-name>

set -e

if [ -z "$1" ]; then
    echo "Usage: ./upload_to_s3.sh <bucket-name>"
    exit 1
fi

BUCKET_NAME=$1

echo "📦 Uploading model artifacts to S3..."

# Create bucket if it doesn't exist
aws s3 mb s3://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# Upload artifacts (feature names)
echo "⬆️ Uploading feature metadata..."
aws s3 sync ./artifacts/ s3://$BUCKET_NAME/artifacts/ \
    --exclude "*.csv" \
    --exclude "*.log"

# Upload model files from training output path
echo "⬆️ Uploading models..."
aws s3 sync ./notebook/Best_Models/ s3://$BUCKET_NAME/notebook/Best_Models/ \
    --exclude "*.csv" \
    --exclude "*.log"

echo "✅ Upload complete!"
echo "📍 Artifacts available at: s3://$BUCKET_NAME/artifacts/"
