variable "ami_id"                { type = string }
variable "instance_type"         { type = string }
variable "key_name"              { type = string }
variable "instance_profile_name" { type = string }

variable "subnet_id" {
  description = "Subnet ID to launch the EC2 instance into"
  type        = string
}

variable "security_group_ids" { type = list(string) }

variable "github_token"      { type = string }
variable "script_to_run"     { type = string }
variable "aws_sqs_queue_url" { type = string }

# optional DLQ (string-or-null); avoid one-line + multiple args syntax error
variable "aws_sqs_dlq_url" {
  type     = string
  default  = null
  nullable = true
}

# optional static creds (string-or-null) – only used if your template expects them
variable "aws_access_key_id" {
  type     = string
  default  = null
  nullable = true
}

variable "aws_secret_access_key" {
  type     = string
  default  = null
  nullable = true
}

variable "worker_name" {
  description = "Name tag / job name"
  type        = string
}

variable "volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 100
}