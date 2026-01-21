# 🚀 AWS Deployment Guide for Zillow Property Predictor

This guide walks you through deploying your ML application to AWS using EC2, S3, and Docker.

## 📋 Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured locally
- SSH key pair for EC2 access
- Docker installed locally (for testing)

## 🏗️ Architecture Overview

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   User      │─────▶│  EC2 Instance│─────▶│  S3 Bucket  │
│  Browser    │      │  (Streamlit) │      │  (Models)   │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                     ┌──────┴──────┐
                     │   Docker    │
                     │  Container  │
                     └─────────────┘
```

## 📦 Step 1: Upload Model Artifacts to S3

### 1.1 Create S3 Bucket
```bash
# Replace with your unique bucket name
export BUCKET_NAME="zillow-ml-models-<your-name>"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region us-east-1
```

### 1.2 Upload Model Files
```bash
# From project root directory
cd deploy
chmod +x upload_to_s3.sh
./upload_to_s3.sh $BUCKET_NAME
```

Or manually (aligns with app paths):
```bash
aws s3 cp ../notebook/Best_Models/best_randomforest.pkl s3://$BUCKET_NAME/notebook/Best_Models/
aws s3 cp ../notebook/Best_Models/best_lightgbm.pkl s3://$BUCKET_NAME/notebook/Best_Models/
aws s3 cp ../artifacts/feature_names.json s3://$BUCKET_NAME/artifacts/
```

## 🖥️ Step 2: Launch EC2 Instance

### 2.1 Launch Instance via AWS Console
1. Go to EC2 Dashboard → Launch Instance
2. **Name:** zillow-property-predictor
3. **AMI:** Ubuntu Server 22.04 LTS
4. **Instance Type:** t2.medium (2 vCPU, 4 GB RAM) or t2.large for better performance
5. **Key Pair:** Select or create new key pair
6. **Security Group:** Create new with rules:
   - SSH (22) - Your IP
   - Custom TCP (8501) - 0.0.0.0/0 (Streamlit)
7. **Storage:** 20 GB gp3
8. **IAM Role:** Create role with S3 read access (AmazonS3ReadOnlyAccess)

### 2.2 Or Launch via AWS CLI
```bash
# Create security group
aws ec2 create-security-group \
    --group-name zillow-ml-sg \
    --description "Security group for Zillow ML app"

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-name zillow-ml-sg \
    --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name zillow-ml-sg \
    --protocol tcp --port 8501 --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t2.medium \
    --key-name your-key-pair \
    --security-groups zillow-ml-sg \
    --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=20} \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=zillow-predictor}]'
```

## 🔧 Step 3: Setup EC2 Instance

### 3.1 Connect to EC2
```bash
# Get public IP from AWS Console
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### 3.2 Run Setup Script
```bash
# Download setup script
wget https://raw.githubusercontent.com/<your-repo>/main/deploy/aws_setup.sh
chmod +x aws_setup.sh
./aws_setup.sh

# Log out and back in for Docker group changes
exit
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

### 3.3 Manual Setup (Alternative)
```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install AWS CLI
sudo apt-get install -y awscli git
```

## 📥 Step 4: Deploy Application

### 4.1 Clone Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 4.2 Download Model Artifacts from S3
```bash
# Configure AWS credentials (if not using IAM role)
aws configure

# Download models and feature names
export S3_BUCKET="zillow-ml-models-<your-name>"
aws s3 sync s3://$S3_BUCKET/notebook/Best_Models/ ./notebook/Best_Models/
aws s3 cp s3://$S3_BUCKET/artifacts/feature_names.json ./artifacts/feature_names.json
```

### 4.3 Build and Run Docker Container
```bash
# Make deploy script executable
chmod +x deploy/deploy_to_ec2.sh

# Deploy
export S3_BUCKET="zillow-ml-models-<your-name>"
./deploy/deploy_to_ec2.sh
```

Or manually:
```bash
# Build image
docker-compose -f deploy/docker-compose.yml build

# Start container
docker-compose -f deploy/docker-compose.yml up -d

# Check logs
docker-compose -f deploy/docker-compose.yml logs -f
```

## 🌐 Step 5: Access Your Application

Your app should now be accessible at:
```
http://<EC2-PUBLIC-IP>:8501
```

## 🔍 Monitoring & Maintenance

### View Logs
```bash
docker-compose -f deploy/docker-compose.yml logs -f
```

### Restart Application
```bash
docker-compose -f deploy/docker-compose.yml restart
```

### Stop Application
```bash
docker-compose -f deploy/docker-compose.yml down
```

### Update Application
```bash
git pull origin main
./deploy/deploy_to_ec2.sh
```

## 💰 Cost Optimization

### Estimated Monthly Costs (us-east-1)
- **t2.medium EC2:** ~$30/month
- **S3 Storage (1 GB):** ~$0.02/month
- **Data Transfer:** ~$0.09/GB

### Cost Saving Tips
1. **Use t2.micro** for testing (free tier eligible)
2. **Stop instance** when not in use
3. **Use Elastic IP** to avoid IP changes (small cost if not attached)
4. **Set up CloudWatch alarms** for cost monitoring

## 🔒 Security Best Practices

1. **Restrict SSH access** to your IP only
2. **Use IAM roles** instead of hardcoded credentials
3. **Enable HTTPS** with Let's Encrypt (optional)
4. **Regular security updates:**
   ```bash
   sudo apt-get update && sudo apt-get upgrade -y
   ```
5. **Use AWS Secrets Manager** for sensitive data

## 🚀 Advanced: Production Setup

### Add Domain Name (Optional)
1. Register domain or use Route 53
2. Create A record pointing to EC2 IP
3. Install Nginx as reverse proxy
4. Setup SSL with Let's Encrypt

### Setup Nginx Reverse Proxy
```bash
sudo apt-get install -y nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/zillow-app

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/zillow-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Setup SSL with Let's Encrypt
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose -f deploy/docker-compose.yml logs

# Check if port is in use
sudo netstat -tulpn | grep 8501

# Rebuild container
docker-compose -f deploy/docker-compose.yml down
docker-compose -f deploy/docker-compose.yml build --no-cache
docker-compose -f deploy/docker-compose.yml up -d
```

### Can't access application
1. Check security group allows port 8501
2. Verify container is running: `docker ps`
3. Check EC2 instance is running
4. Test locally: `curl http://localhost:8501`

### Model files not found
```bash
# Verify files exist
ls -lh artifacts/

# Re-download from S3
aws s3 sync s3://$S3_BUCKET/artifacts/ ./artifacts/
```

## 📚 Additional Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)

## 🎯 AWS Quick Checklist

1. Upload models + feature_names to S3
2. Launch EC2 and run `aws_setup.sh`
3. `aws s3 sync` models and `feature_names.json` down to the instance
4. `./deploy/deploy_to_ec2.sh` (build + up)
5. Verify at `http://<EC2-PUBLIC-IP>:8501`
