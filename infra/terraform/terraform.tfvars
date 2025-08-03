aws_region = "eu-central-1"
ami_id     = "ami-0dc33c9c954b3f073"
ec2_key_pair_name = "mainkey"
github_token = "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"

subnet_ids = [
  "subnet-08a0ac8d2a485dd67",
  "subnet-09a4eded6c53fc0a2"
]

security_group_ids = [
  "sg-0e2945357ff34782a"
]

worker_configs = {
  preflop = {
    script_to_run     = "workers/sqs_worker_preflop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-dlq"
    instance_type     = "c5.large"
    min_size          = 0
    max_size          = 15
    desired_capacity  = 1
  }

  equities = {
    script_to_run     = "workers/equity_worker.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/123/equity-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/123/equity-dlq"
    instance_type     = "c5.large"
    min_size          = 0
    max_size          = 5
    desired_capacity  = 0
  }
}