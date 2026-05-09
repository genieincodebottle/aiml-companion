terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state (uncomment for production)
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "multi-agents/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "multi-agents-app"
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }
}
