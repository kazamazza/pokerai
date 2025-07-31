# infra/terraform/variables.tf

variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "eu-central-1"
}

variable "ami_id" {
  description = "AMI ID to use for EC2 Spot instances"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type for Spot workers"
  type        = string
  default     = "t3.large"
}

variable "ec2_key_pair_name" {
  description = "Name of the EC2 Key Pair to SSH into instances (must exist)"
  type        = string
}

variable "asg_min_size" {
  description = "Minimum number of instances in Auto Scaling Group"
  type        = number
  default     = 1
}

variable "asg_max_size" {
  description = "Maximum number of instances in Auto Scaling Group"
  type        = number
  default     = 10
}

variable "asg_desired_capacity" {
  description = "Desired capacity of Auto Scaling Group"
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
  description = "GitHub token for private repo clone"
  type        = string
  sensitive   = true
}