output "agentcore_runtime_role_arn" {
  description = "IAM role ARN for AgentCore Runtime agents"
  value       = aws_iam_role.agentcore_runtime.arn
}

output "deployer_role_arn" {
  description = "IAM role ARN for deploying agents"
  value       = aws_iam_role.deployer.arn
}

output "log_group_name" {
  description = "CloudWatch log group for agent logs"
  value       = aws_cloudwatch_log_group.agents.name
}

output "region" {
  description = "AWS region"
  value       = data.aws_region.current.name
}

output "next_steps" {
  description = "What to do after terraform apply"
  value       = <<-EOT

    Infrastructure is ready! Next steps:

    1. Deploy all four agents with the AgentCore CLI (npm install -g @aws/agentcore):
       bash scripts/deploy.sh

    2. Test the deployed agents:
       bash scripts/test_agents.sh

  EOT
}
