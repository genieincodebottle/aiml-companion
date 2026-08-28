# ===========================================================================
# Multi-Agent App on AWS - Terraform Configuration
#
# This creates the IAM roles and permissions needed for AgentCore deployment.
# The actual AgentCore agents are deployed via the agentcore CLI.
# ===========================================================================

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ---------------------------------------------------------------------------
# IAM Role for AgentCore Runtime
# ---------------------------------------------------------------------------

resource "aws_iam_role" "agentcore_runtime" {
  name = "${var.project_name}-agentcore-runtime-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock-agentcore.amazonaws.com"
        }
        Action = "sts:AssumeRole"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_name}-agentcore-runtime"
  }
}

# Bedrock model access
resource "aws_iam_role_policy" "bedrock_access" {
  name = "${var.project_name}-bedrock-access"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ]
        Resource = [
          "arn:aws:bedrock:${data.aws_region.current.name}::foundation-model/${var.bedrock_model_id}",
          "arn:aws:bedrock:us-*::foundation-model/*",
        ]
      }
    ]
  })
}

# AgentCore Runtime permissions
resource "aws_iam_role_policy" "agentcore_runtime_access" {
  name = "${var.project_name}-agentcore-runtime-access"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock-agentcore:InvokeAgentRuntime",
          "bedrock-agentcore:CreateSession",
          "bedrock-agentcore:DeleteSession",
        ]
        Resource = "arn:aws:bedrock-agentcore:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })
}

# CloudWatch logging
resource "aws_iam_role_policy" "cloudwatch_logging" {
  name = "${var.project_name}-cloudwatch-logging"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# IAM Role for the human deployer (CI/CD or local dev)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "deployer" {
  name = "${var.project_name}-deployer-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = data.aws_caller_identity.current.arn
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "deployer_bedrock" {
  role       = aws_iam_role.deployer.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"
}

resource "aws_iam_role_policy_attachment" "deployer_agentcore" {
  role       = aws_iam_role.deployer.name
  policy_arn = "arn:aws:iam::aws:policy/BedrockAgentCoreFullAccess"
}

# ---------------------------------------------------------------------------
# Secrets Manager (for API keys)
# ---------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "tavily_api_key" {
  count = var.tavily_api_key != "" ? 1 : 0

  name        = "${var.project_name}/${var.environment}/tavily-api-key"
  description = "Tavily API key for web search tool"

  tags = {
    Name = "${var.project_name}-tavily-key"
  }
}

resource "aws_secretsmanager_secret_version" "tavily_api_key" {
  count = var.tavily_api_key != "" ? 1 : 0

  secret_id     = aws_secretsmanager_secret.tavily_api_key[0].id
  secret_string = var.tavily_api_key
}

# ---------------------------------------------------------------------------
# CloudWatch Log Group (for agent logs)
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "agents" {
  name              = "/agentcore/${var.project_name}/${var.environment}"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-agent-logs"
  }
}
