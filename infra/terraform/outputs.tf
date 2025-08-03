output "main_sqs_queue_url" {
  value = module.sqs.main_queue_url
}

output "dlq_queue_url" {
  value = module.sqs.dlq_url
}

output "main_sqs_queue_arn" {
  value = module.sqs.main_queue_arn
}

output "dlq_queue_arn" {
  value = module.sqs.dlq_arn
}

output "ec2_instance_profile_name" {
  value = module.iam.instance_profile_name
}