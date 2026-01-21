# 🚀 AWS Deployment Options for Zillow Property Predictor

Choose the deployment method that best fits your needs and experience level.

## 📊 Comparison Table

| Method | Difficulty | Time | Cost | Best For |
|--------|-----------|------|------|----------|
| **Manual (Console)** | Easy | 30 min | $30/mo | Beginners, quick start |
| **Docker + EC2** | Medium | 45 min | $30/mo | Production-ready |
| **Terraform** | Advanced | 60 min | $30/mo | Infrastructure as Code |
| **AWS SageMaker** | Advanced | 90 min | $50+/mo | Enterprise ML |
| **Streamlit Cloud** | Easy | 10 min | Free | Testing, demos |

---

## 1️⃣ Manual Deployment (Recommended for Beginners)

**Best for:** First-time AWS users, quick testing

### Pros
- ✅ No infrastructure code needed
- ✅ Visual AWS Console interface
- ✅ Easy to understand
- ✅ Quick to set up

### Cons
- ❌ Manual steps (not reproducible)
- ❌ Harder to manage multiple environments
- ❌ No version control for infrastructure

### Quick Start
Follow: [DEPLOY_GUIDE.md](../DEPLOY_GUIDE.md)

**Time:** 30 minutes  
**Cost:** ~$30/month (t2.medium)

---

## 2️⃣ Docker + EC2 (Recommended for Production)

**Best for:** Production deployments, portfolio projects

### Pros
- ✅ Containerized (portable)
- ✅ Easy to update and rollback
- ✅ Consistent across environments
- ✅ Industry standard

### Cons
- ❌ Requires Docker knowledge
- ❌ Manual EC2 setup still needed

### Files Provided
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-container orchestration
- `deploy_to_ec2.sh` - Deployment script
- `aws_setup.sh` - EC2 setup script

### Quick Start
```bash
# 1. Setup EC2 and install Docker
ssh -i key.pem ubuntu@EC2-IP
./deploy/aws_setup.sh

# 2. Clone repo and deploy
git clone <your-repo>
cd <your-repo>
./deploy/deploy_to_ec2.sh
```

**Time:** 45 minutes  
**Cost:** ~$30/month

---

## 3️⃣ Terraform (Infrastructure as Code)

**Best for:** Teams, multiple environments, reproducible infrastructure

### Pros
- ✅ Infrastructure as code
- ✅ Version controlled
- ✅ Reproducible
- ✅ Easy to manage multiple environments
- ✅ Automated provisioning

### Cons
- ❌ Requires Terraform knowledge
- ❌ More complex setup
- ❌ Overkill for simple projects

### Files Provided
- `deploy/terraform/main.tf` - Infrastructure definition
- `deploy/terraform/variables.tf` - Configuration variables
- `deploy/terraform/terraform.tfvars.example` - Example values

### Quick Start
```bash
cd deploy/terraform

# Copy and edit variables
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars

# Initialize and deploy
terraform init
terraform plan
terraform apply

# Get outputs
terraform output app_url
```

**Time:** 60 minutes  
**Cost:** ~$30/month

---

## 4️⃣ AWS SageMaker

**Best for:** Enterprise ML deployments, auto-scaling needs

### Pros
- ✅ Managed ML infrastructure
- ✅ Auto-scaling
- ✅ Built-in monitoring
- ✅ A/B testing support
- ✅ Model versioning

### Cons
- ❌ More expensive
- ❌ Complex setup
- ❌ Overkill for simple apps
- ❌ Vendor lock-in

### Setup
```python
# Create SageMaker model
import sagemaker
from sagemaker.sklearn import SKLearnModel

model = SKLearnModel(
    model_data='s3://bucket/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='1.0-1'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

**Time:** 90 minutes  
**Cost:** ~$50+/month

---

## 5️⃣ Streamlit Cloud (Free Tier)

**Best for:** Quick demos, testing, free hosting

### Pros
- ✅ Completely free
- ✅ No AWS setup needed
- ✅ Automatic deployments from GitHub
- ✅ Built-in CI/CD
- ✅ Easy to use

### Cons
- ❌ Limited resources
- ❌ Public apps only (on free tier)
- ❌ Less control
- ❌ May sleep after inactivity

### Quick Start
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Deploy!

**Time:** 10 minutes  
**Cost:** Free (with limitations)

---

## 🎯 Recommendation by Use Case

### For Learning / Portfolio
→ **Manual Deployment** or **Streamlit Cloud**
- Quick to set up
- Low/no cost
- Good for demos

### For Job Applications
→ **Docker + EC2**
- Shows DevOps skills
- Production-ready
- Industry standard

### For Team Projects
→ **Terraform**
- Reproducible
- Version controlled
- Professional approach

### For Enterprise
→ **AWS SageMaker**
- Scalable
- Managed service
- Enterprise features

---

## 💰 Cost Breakdown

### EC2 Deployment (Docker)
- **t2.medium:** $30/month
- **t2.large:** $60/month
- **S3 storage:** $0.02/month
- **Data transfer:** ~$0.09/GB
- **Total:** ~$30-35/month

### Cost Optimization Tips
1. **Use t2.micro** for testing (free tier)
2. **Stop instance** when not in use
3. **Use spot instances** (up to 90% savings)
4. **Set up billing alerts**
5. **Use reserved instances** for long-term (up to 75% savings)

### Free Tier Eligible
- t2.micro: 750 hours/month (first 12 months)
- S3: 5 GB storage
- Data transfer: 15 GB/month

---

## 🛠️ Required Tools

### All Methods
- AWS Account
- AWS CLI: `pip install awscli`
- Git

### Docker Method
- Docker Desktop
- Docker Compose

### Terraform Method
- Terraform: [Download](https://www.terraform.io/downloads)

### SageMaker Method
- Python SDK: `pip install sagemaker`

---

## 📚 Next Steps

1. Choose your deployment method
2. Follow the corresponding guide
3. Test your deployment
4. Add to portfolio/resume
5. Monitor costs and usage

---

## 🆘 Need Help?

- **Streamlit Quick Guide:** [../DEPLOY_GUIDE.md](../DEPLOY_GUIDE.md)
- **Detailed AWS Guide:** [DEPLOY_AWS.md](DEPLOY_AWS.md)
- **AWS Docs:** [docs.aws.amazon.com](https://docs.aws.amazon.com)

---

**Ready to deploy?** Start with the [Streamlit Guide](../DEPLOY_GUIDE.md)!
