# ⚡ Quick Start: Deploy to AWS in 30 Minutes

This is the fastest path to get your Zillow Property Predictor live on AWS.

## 🎯 What You'll Need

- AWS account
- Terminal/Command line
- Your trained model files in `artifacts/` folder
- 30 minutes

## 🚀 5-Step Deployment

### Step 1: Install AWS CLI (2 min)
```bash
# Install AWS CLI
pip install awscli

# Configure with your credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

### Step 2: Upload Models to S3 (3 min)
```bash
# Create bucket (replace 'yourname' with your name)
export BUCKET_NAME="zillow-ml-yourname"
aws s3 mb s3://$BUCKET_NAME

# Upload models
aws s3 cp artifacts/best_ridge.pkl s3://$BUCKET_NAME/artifacts/
aws s3 cp artifacts/best_randomforest.pkl s3://$BUCKET_NAME/artifacts/
aws s3 cp artifacts/best_lightgbm.pkl s3://$BUCKET_NAME/artifacts/

# Verify
aws s3 ls s3://$BUCKET_NAME/artifacts/
```

### Step 3: Launch EC2 Instance (5 min)

**Go to AWS Console → EC2 → Launch Instance:**

1. **Name:** zillow-predictor
2. **AMI:** Ubuntu Server 22.04 LTS
3. **Instance Type:** t2.medium
4. **Key Pair:** Create new → Download .pem file
5. **Security Group:** Create new
   - Add rule: Custom TCP, Port 8501, Source 0.0.0.0/0
   - Add rule: SSH, Port 22, Source My IP
6. **Storage:** 20 GB
7. Click **Launch Instance**
8. **Copy the Public IP address**

### Step 4: Setup EC2 (10 min)
```bash
# SSH into your instance (replace with your key and IP)
ssh -i your-key.pem ubuntu@YOUR-EC2-IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install AWS CLI and Git
sudo apt-get update
sudo apt-get install -y awscli git

# Log out and back in
exit
ssh -i your-key.pem ubuntu@YOUR-EC2-IP
```

### Step 5: Deploy App (10 min)
```bash
# Clone your repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# Download models from S3
export BUCKET_NAME="zillow-ml-yourname"
aws s3 sync s3://$BUCKET_NAME/artifacts/ ./artifacts/

# Build and run
docker build -t zillow-app .
docker run -d -p 8501:8501 --name zillow-predictor zillow-app

# Check if running
docker ps
docker logs zillow-predictor
```

## 🎉 Done!

Your app is now live at: **http://YOUR-EC2-IP:8501**

## 🧪 Test It

Open your browser and go to `http://YOUR-EC2-IP:8501`

Try these sample inputs:
- Living Area: 2000 sqft
- Bedrooms: 3
- Bathrooms: 2
- Year Built: 2000
- County: 6037 (Los Angeles)

## 🛠️ Useful Commands

```bash
# View logs
docker logs -f zillow-predictor

# Restart app
docker restart zillow-predictor

# Stop app
docker stop zillow-predictor

# Update app after code changes
git pull
docker stop zillow-predictor
docker rm zillow-predictor
docker build -t zillow-app .
docker run -d -p 8501:8501 --name zillow-predictor zillow-app
```

## 💰 Cost

- **t2.medium:** ~$30/month (~$1/day)
- **S3:** ~$0.02/month
- **Total:** ~$30/month

**Save money:** Stop the instance when not using it!
```bash
# From your local machine
aws ec2 stop-instances --instance-ids YOUR-INSTANCE-ID
aws ec2 start-instances --instance-ids YOUR-INSTANCE-ID
```

## 🐛 Troubleshooting

**Can't access the app?**
1. Check security group allows port 8501
2. Verify container is running: `docker ps`
3. Check logs: `docker logs zillow-predictor`

**Container won't start?**
```bash
docker logs zillow-predictor
# Look for errors, usually missing model files
```

**Models not found?**
```bash
ls -lh artifacts/
# Should see: best_ridge.pkl, best_randomforest.pkl, best_lightgbm.pkl
```

## 📚 Next Steps

- ✅ Test all 3 models (Ridge, Random Forest, LightGBM)
- ✅ Add your deployment URL to resume
- ✅ Share with recruiters
- ✅ Setup custom domain (optional)
- ✅ Add HTTPS (optional)

For detailed deployment options, see [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

---

**Questions?** Check the full [deployment guide](deploy/README_DEPLOYMENT.md)
