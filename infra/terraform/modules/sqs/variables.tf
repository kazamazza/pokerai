variable "main_queue_name" {
  description = "Name of the main SQS queue"
  type        = string
  default     = "preflop-chart-queue"
}

variable "dlq_name" {
  description = "Name of the dead-letter SQS queue"
  type        = string
  default     = "preflop-chart-dlq"
}