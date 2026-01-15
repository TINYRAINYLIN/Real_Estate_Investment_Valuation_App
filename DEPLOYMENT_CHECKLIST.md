# 🚀 Deployment Checklist for Zillow Property Predictor

## ✅ Pre-Deployment Tasks

### 1. Complete Feature Engineering
- [ ] Run `python scripts/extract_features.py` to get all 213 feature names
- [ ] Update `src/pipeline/predicting_pipeline.py` with proper feature transformation
- [ ] Ensure `CustomData` class creates all 213 features matching training data
- [ ] Handle one-hot encoding for categorical variables (fips, regionidzip, etc.)

### 2. Test Locally
- [ ] Test prediction pipeline: `python scripts/test_prediction.py`
- [ ] Verify all 3 models load correctly (Ridge, Random Forest, LightGBM)
- [ ] Test Streamlit app locally: `streamlit run app.py`
- [ ] Test Docker container locally: `bash deploy/local_test.sh`

### 3. Prepare AWS Account
- [ ] Create AWS account (if not exists)
- [ ] Install AWS CLI: `pip install awscli`
- [ ] Configure AWS credentials: `aws configure`
- [ ] Create IAM user with EC2 and S3 permissions
- [ ] Generate EC2 key pair for SSH access

## 🔧 Deployment Steps

### Step 1: Upload Models to S3 (5 minutes)
```bash
# Create S3 bucket
export BUCKET_NAME="zillow-ml-models-yourname"
aws s3 mb s3://$BUCKET_NAME --region us-east-1

# Upload model artifacts
cd deploy
chmod +x upload_to_s3.sh
./upload_to_s3.sh $BUCKET_NAME
```

**Checklist:**
- [ ] S3 bucket created successfully
- [ ] All 3 model files uploaded (.pkl files)
- [ ] Verify files in S3: `aws s3 ls s3://$BUCKET_NAME/artifacts/`

### Step 2: Launch EC2 Instance (10 minutes)
**Via AWS Console:**
- [ ] Go to EC2 Dashboard → Launch Instance
- [ ] Name: `zillow-property-predictor`
- [ ] AMI: Ubuntu Server 22.04 LTS
- [ ] Instance Type: t2.medium (or t2.large)
- [ ] Key Pair: Select or create new
- [ ] Security Group: Allow ports 22 (SSH) and 8501 (Streamlit)
- [ ] Storage: 20 GB
- [ ] IAM Role: S3 read access
- [ ] Launch instance
- [ ] Note down Public IP address

**Security Group Rules:**
- [ ] SSH (22) from your IP
- [ ] Custom TCP (8501) from 0.0.0.0/0

### Step 3: Setup EC2 Instance (15 minutes)
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>

# Download and run setup script
wget https://raw.githubusercontent.com/yourusername/yourrepo/main/deploy/aws_setup.sh
chmod +x aws_setup.sh
./aws_setup.sh

# Log out and back in
exit
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

**Checklist:**
- [ ] Docker installed and running
- [ ] Docker Compose installed
- [ ] AWS CLI installed
- [ ] Git installed
- [ ] User added to docker group

### Step 4: Deploy Application (10 minutes)
```bash
# Clone repository
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

# Configure AWS (if not using IAM role)
aws configure

# Download model artifacts
export S3_BUCKET="zillow-ml-models-yourname"
aws s3 sync s3://$S3_BUCKET/artifacts/ ./artifacts/

# Deploy with Docker
chmod +x deploy/deploy_to_ec2.sh
./deploy/deploy_to_ec2.sh
```

**Checklist:**
- [ ] Repository cloned successfully
- [ ] Model artifacts downloaded from S3
- [ ] Docker image built successfully
- [ ] Container running: `docker ps`
- [ ] No errors in logs: `docker-compose -f deploy/docker-compose.yml logs`

### Step 5: Verify Deployment (5 minutes)
- [ ] Access app at `http://<EC2-PUBLIC-IP>:8501`
- [ ] Test prediction with sample inputs
- [ ] Verify all 3 models work (Ridge, Random Forest, LightGBM)
- [ ] Check predictions are reasonable
- [ ] Test different input combinations

## 🎯 Post-Deployment Tasks

### Monitoring
- [ ] Setup CloudWatch for EC2 monitoring
- [ ] Create billing alerts
- [ ] Monitor application logs regularly
- [ ] Setup health check endpoint

### Documentation
- [ ] Update README with deployment URL
- [ ] Document any issues encountered
- [ ] Create user guide for the app
- [ ] Add screenshots to portfolio

### Optional Enhancements
- [ ] Add SHAP explanations to predictions
- [ ] Setup custom domain name
- [ ] Configure HTTPS with SSL certificate
- [ ] Add Nginx reverse proxy
- [ ] Setup CI/CD with GitHub Actions
- [ ] Add authentication (if needed)
- [ ] Implement caching for faster predictions
- [ ] Add model performance monitoring

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose -f deploy/docker-compose.yml logs

# Rebuild
docker-compose -f deploy/docker-compose.yml down
docker-compose -f deploy/docker-compose.yml build --no-cache
docker-compose -f deploy/docker-compose.yml up -d
```

### Can't access application
1. Check security group allows port 8501
2. Verify container is running: `docker ps`
3. Check EC2 instance is running
4. Test locally on EC2: `curl http://localhost:8501`

### Model files not found
```bash
# Verify files exist
ls -lh artifacts/

# Re-download from S3
aws s3 sync s3://$S3_BUCKET/artifacts/ ./artifacts/
```

### Out of memory
- Upgrade to t2.large or t2.xlarge
- Optimize model loading (lazy loading)
- Add swap space

## 💰 Cost Estimate

**Monthly costs (us-east-1):**
- EC2 t2.medium: ~$30/month
- S3 storage (1 GB): ~$0.02/month
- Data transfer: ~$0.09/GB
- **Total: ~$30-35/month**

**Cost saving tips:**
- Use t2.micro for testing (free tier)
- Stop instance when not in use
- Use spot instances for non-critical workloads

## 📚 Resources

- [Deployment Guide](deploy/README_DEPLOYMENT.md)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

## ✨ Success Criteria

Your deployment is successful when:
- ✅ Application accessible via public URL
- ✅ All 3 models make predictions
- ✅ Predictions are reasonable (within expected range)
- ✅ No errors in application logs
- ✅ Response time < 3 seconds
- ✅ Application survives EC2 restart

## 🎉 Next Steps After Deployment

1. Share URL with recruiters/portfolio
2. Add to resume and LinkedIn
3. Create demo video
4. Write blog post about the project
5. Monitor usage and costs
6. Gather feedback and iterate

---

**Need help?** Check the [detailed deployment guide](deploy/README_DEPLOYMENT.md) or open an issue.
