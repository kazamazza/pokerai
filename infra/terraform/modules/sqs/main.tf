resource "aws_sqs_queue" "dlq" {
  name = var.dlq_name

  message_retention_seconds = 1209600  # 14 days
}

resource "aws_sqs_queue" "main" {
  name = var.main_queue_name

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })

  visibility_timeout_seconds = 300
  receive_wait_time_seconds = 20
}

output "main_queue_url" {
  value = aws_sqs_queue.main.id
}

output "dlq_url" {
  value = aws_sqs_queue.dlq.id
}

output "main_queue_arn" {
  value = aws_sqs_queue.main.arn
}

output "dlq_arn" {
  value = aws_sqs_queue.dlq.arn
}