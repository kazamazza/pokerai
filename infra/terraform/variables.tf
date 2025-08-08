variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "eu-central-1"
}

variable "ami_id" {
  description = "AMI ID to use for EC2 Spot instances"
  type        = string
}

variable "instance_types" {
  description = "Preferred Spot instance types (ordered)."
  type        = list(string)
  default     = ["c5.xlarge", "c5a.xlarge", "c5d.xlarge", "c6i.xlarge", "m6i.xlarge"]
}

variable "ec2_key_pair_name" {
  description = "Name of the EC2 Key Pair to SSH into instances (must exist)"
  type        = string
}


variable "subnet_ids" {
  description = "List of subnet IDs to launch EC2 instances into"
  type        = list(string)
}

variable "security_group_ids" {
  description = "List of security group IDs to attach to EC2 instances"
  type        = list(string)
}

variable "github_token" {
  description = "GitHub token for private repo clone"
  type        = string
  sensitive   = true
}

variable "aws_access_key_id" {
  type        = string
  description = "AWS access key ID for EC2 worker access"
  sensitive   = true
}

variable "aws_secret_access_key" {
  type        = string
  description = "AWS secret access key for EC2 worker access"
  sensitive   = true
}

variable "worker_configs" {
  type = map(object({
    script_to_run         = string
    aws_sqs_queue_url     = string
    aws_sqs_dlq_url       = string
    instance_types        = list(string)
    min_size              = number
    max_size              = number
    desired_capacity      = number
    worker_name           = string
  }))
}

variable "job_configs" {
  type = map(object({
    script_to_run         = string
    aws_sqs_queue_url     = string
    instance_type         = string
    worker_name           = string
    volume_size           = number
    disabled              = optional(bool, false)
  }))
  description = "Configuration for ephemeral one-off jobs"
}