#!/bin/bash  # use bash shell

# AWS EC2 Setup Script for Zillow Property Predictor
# Run this script on your EC2 instance after SSH connection

set -e  # exit on first error

echo "🚀 Starting AWS EC2 setup for Zillow Property Predictor..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update          # refresh package index
sudo apt-get upgrade -y      # apply upgrades

# Install Docker prereqs
echo "🐳 Installing Docker..."
sudo apt-get install -y \    # install base tools for apt over HTTPS
    ca-certificates \        # CA certs for secure downloads
    curl \                   # HTTP client
    gnupg \                  # key management
    lsb-release              # distro info helper

sudo mkdir -p /etc/apt/keyrings                     # ensure keyring dir exists
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg  # add Docker GPG key

echo \                                             # add Docker apt repo
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \"
  "  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update                                 # refresh with Docker repo
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin  # install Docker engine + plugin

# Let current user run docker without sudo (needs re-login)
sudo usermod -aG docker $USER

# Install Docker Compose standalone
echo "📦 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose  # download binary
sudo chmod +x /usr/local/bin/docker-compose                                                                                                     # make executable

# Install AWS CLI
echo "☁️ Installing AWS CLI..."
sudo apt-get install -y awscli                                                                                                                   # aws s3, etc.

# Install Git
echo "📚 Installing Git..."
sudo apt-get install -y git                                                                                                                      # for pulling repo

echo "✅ EC2 setup complete!"
echo ""
echo "Next steps:"
echo "1. Log out and log back in for Docker group changes to take effect"
echo "2. Clone your repository: git clone <your-repo-url>"
echo "3. Configure AWS credentials: aws configure"
echo "4. Download model artifacts from S3 (if stored there)"
echo "5. Build and run Docker container"
