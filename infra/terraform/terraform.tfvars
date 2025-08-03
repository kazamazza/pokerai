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

job_configs = {
  preflop_producer = {
    script_to_run     = "workers/preflop/sqs_producer.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-queue"
    instance_type     = "c5.large"
    worker_name       = "preflop_producer"
    volume_size       = 100
  }

  equity_producer = {
    script_to_run     = "workers/equity/sqs_producer.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-queue"
    instance_type     = "c5.large"
    worker_name       = "equity_producer"
    volume_size       = 100
  }

  exploit_producer = {
    script_to_run     = "workers/exploit/sqs_producer.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/exploit-logs-queue"
    instance_type     = "c5.large"
    worker_name       = "exploit_producer"
    volume_size       = 100
  }
}

worker_configs = {
  preflop = {
    script_to_run     = "workers/preflop/sqs_worker.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/preflop-chart-dlq"
    instance_type     = "c5.large"
    min_size          = 0
    max_size          = 10
    desired_capacity  = 0
    worker_name = "preflop_worker"
  }

  equities = {
    script_to_run     = "workers/equity/sqs_worker.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-dlq"
    instance_type     = "c5.large"
    min_size          = 0
    max_size          = 10
    desired_capacity  = 0
    worker_name = "equity_worker"
  }

   exploit = {
    script_to_run     = "workers/exploit/sqs_worker.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/exploit-logs-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/exploit-logs-dlq"
    instance_type     = "c5.large"
    min_size          = 0
    max_size          = 10
    desired_capacity  = 0
    worker_name = "exploit_worker"
  }
}