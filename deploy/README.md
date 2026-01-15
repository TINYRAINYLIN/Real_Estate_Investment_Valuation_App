# 🚀 Deployment Files

This folder contains all scripts and configurations for deploying the Zillow Property Predictor to AWS.

## 📁 Contents

### Shell Scripts
- **`aws_setup.sh`** - Initial EC2 instance setup (Docker, AWS CLI, etc.)
- **`deploy_to_ec2.sh`** - Deploy application to EC2
- **`upload_to_s3.sh`** - Upload model artifacts to S3
- **`local_test.sh`** - Test Docker container locally

### Docker
- **`docker-compose.yml`** - Docker Compose configuration

### Terraform (Infrastructure as Code)
- **`terraform/main.tf`** - AWS infrastructure definition
- **`terraform/variables.tf`** - Configuration variables
- **`terraform/terraform.tfvars.example`** - Example configuration

### Documentation
- **`README_DEPLOYMENT.md`** - Comprehensive deployment guide
- **`DEPLOYMENT_OPTIONS.md`** - Compare deployment methods

## 🚀 Quick Start

### 1. Upload Models to S3
```bash
chmod +x upload_to_s3.sh
./upload_to_s3.sh your-bucket-name
```

### 2. Setup EC2 Instance
```bash
# SSH to your EC2 instance
ssh -i your-key.pem ubuntu@EC2-IP

# Run setup script
chmod +x aws_setup.sh
./aws_setup.sh
```

### 3. Deploy Application
```bash
# On EC2 instance
chmod +x deploy_to_ec2.sh
export S3_BUCKET="your-bucket-name"
./deploy_to_ec2.sh
```

## 📚 Documentation

For detailed instructions, see:
- **Quick Start (30 min):** [../QUICKSTART_DEPLOYMENT.md](../QUICKSTART_DEPLOYMENT.md)
- **Full Guide:** [README_DEPLOYMENT.md](README_DEPLOYMENT.md)
- **Checklist:** [../DEPLOYMENT_CHECKLIST.md](../DEPLOYMENT_CHECKLIST.md)

## 🛠️ Usage

### Make Scripts Executable
```bash
chmod +x *.sh
```

### Test Locally First
```bash
./local_test.sh
```

### Deploy to AWS
```bash
# 1. Upload models
./upload_to_s3.sh my-bucket

# 2. On EC2: Setup
./aws_setup.sh

# 3. On EC2: Deploy
export S3_BUCKET="my-bucket"
./deploy_to_ec2.sh
```

## 🔧 Terraform Deployment

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform plan
terraform apply
```

## 📝 Notes

- All scripts are designed for Ubuntu 22.04 LTS
- Requires AWS CLI configured with appropriate credentials
- Docker and Docker Compose will be installed by `aws_setup.sh`

## 🆘 Troubleshooting

See the troubleshooting section in [README_DEPLOYMENT.md](README_DEPLOYMENT.md)
