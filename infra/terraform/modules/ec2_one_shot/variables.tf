variable "subnet_ids" {
  description = "List of subnet IDs to launch EC2 instances into"
  type        = list(string)
}

variable "worker_name" {
  description = "Name of the job"
  type        = string
}

variable "ami_id" {
  type = string
}

variable "instance_type" {
  type = string
}

variable "key_name" {
  type = string
}

variable "security_group_ids" {
  type = list(string)
}

variable "github_token" {
  type = string
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

variable "instance_profile_name" {
  description = "IAM instance profile to attach to EC2 instances"
  type        = string
}

variable "subnet_id" {
  type        = string
  description = "Subnet ID to launch the EC2 instance into"
}

variable "volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 100
}