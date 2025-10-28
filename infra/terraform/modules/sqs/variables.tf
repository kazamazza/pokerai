variable "main_queue_name" {
  description = "Name of the main SQS queue"
  type        = string
  default     = "Postflop"
}

variable "dlq_name" {
  description = "Name of the dead-letter SQS queue"
  type        = string
  default     = "Postflop-DLQ"
}