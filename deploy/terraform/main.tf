# Terraform configuration for AWS deployment
# Optional: Use this for infrastructure as code

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = var.s3_bucket_name
  
  tags = {
    Name        = "Zillow ML Models"
    Environment = var.environment
    Project     = "zillow-property-predictor"
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Security group for EC2
resource "aws_security_group" "zillow_app_sg" {
  name        = "zillow-app-sg"
  description = "Security group for Zillow Property Predictor"
  
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_cidr_blocks
  }
  
  ingress {
    description = "Streamlit"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "zillow-app-sg"
  }
}

# IAM role for EC2 to access S3
resource "aws_iam_role" "ec2_s3_access_role" {
  name = "zillow-ec2-s3-access"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3_read_only" {
  role       = aws_iam_role.ec2_s3_access_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "zillow-ec2-profile"
  role = aws_iam_role.ec2_s3_access_role.name
}

# EC2 instance
resource "aws_instance" "zillow_app" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name              = var.key_name
  vpc_security_group_ids = [aws_security_group.zillow_app_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  
  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }
  
  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y docker.io docker-compose git awscli
              usermod -aG docker ubuntu
              systemctl enable docker
              systemctl start docker
              EOF
  
  tags = {
    Name        = "zillow-property-predictor"
    Environment = var.environment
    Project     = "zillow-ml"
  }
}

# Elastic IP (optional)
resource "aws_eip" "zillow_app_eip" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.zillow_app.id
  domain   = "vpc"
  
  tags = {
    Name = "zillow-app-eip"
  }
}

# Outputs
output "instance_id" {
  value = aws_instance.zillow_app.id
}

output "public_ip" {
  value = var.use_elastic_ip ? aws_eip.zillow_app_eip[0].public_ip : aws_instance.zillow_app.public_ip
}

output "app_url" {
  value = "http://${var.use_elastic_ip ? aws_eip.zillow_app_eip[0].public_ip : aws_instance.zillow_app.public_ip}:8501"
}

output "s3_bucket" {
  value = aws_s3_bucket.model_artifacts.bucket
}
