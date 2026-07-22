variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "multi-agents"
}

variable "bedrock_model_id" {
  description = "Bedrock model ID for agent LLM calls"
  type        = string
  default     = "us.anthropic.claude-sonnet-4-6"
}

variable "tavily_api_key" {
  description = "Tavily API key for web search (optional, leave empty for mock search)"
  type        = string
  default     = ""
  sensitive   = true
}
