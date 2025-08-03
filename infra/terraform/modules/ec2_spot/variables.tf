variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c5.large"
}

variable "key_name" {
  description = "Name of the EC2 key pair for SSH access"
  type        = string
}

variable "instance_profile_name" {
  description = "IAM instance profile to attach to EC2 instances"
  type        = string
}

variable "min_size" {
  description = "Minimum number of instances in ASG"
  type        = number
  default     = 1
}

variable "max_size" {
  description = "Maximum number of instances in ASG"
  type        = number
  default     = 10
}

variable "desired_capacity" {
  description = "Initial desired instance count"
  type        = number
  default     = 3
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
  description = "GitHub token to clone private repo"
  type        = string
  sensitive   = true
}

variable "script_to_run" {
  type = string
}

variable "aws_sqs_queue_url" {
  type = string
}

variable "aws_sqs_dlq_url" {
  type = string
}