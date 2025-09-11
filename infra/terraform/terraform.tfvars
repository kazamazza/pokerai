aws_region = "eu-central-1"
aws_access_key_id = "AKIATDVYMJ5MRZJASBON"
aws_secret_access_key = "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"
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
    disabled          = true
  }

  equity_producer = {
    script_to_run     = "workers/equity/sqs_producer.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-queue"
    instance_type     = "c5.large"
    worker_name       = "equity_producer"
    volume_size       = 100
    disabled          = true
  }

  exploit_producer = {
    script_to_run     = "workers/exploit/sqs_producer.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/exploit-logs-queue"
    instance_type     = "c5.large"
    worker_name       = "exploit_producer"
    volume_size       = 100
    disabled          = true
  }
}

worker_configs = {
  preflop = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/postflop-chart-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/postflop-chart-dlq"
    instance_types    = [
      # Primary compute families
      "c6i.xlarge", "c6i.2xlarge", "c6i.4xlarge",
      "c6a.xlarge", "c6a.2xlarge", "c6a.4xlarge",
      "c5.xlarge",  "c5.2xlarge",  "c5.4xlarge",
      "c5a.xlarge", "c5a.2xlarge", "c5a.4xlarge",
      "c5n.xlarge", "c5n.2xlarge", "c5n.4xlarge",
      "c6in.xlarge","c6in.2xlarge","c6in.4xlarge",
      # NVMe instance-store variants
      "c6id.xlarge","c6id.2xlarge","c5d.xlarge","c5d.2xlarge",
      # General-purpose fallbacks
      "m6i.xlarge","m6i.2xlarge","m5.xlarge","m5.2xlarge"
    ]
    min_size          = 0
    max_size          = 15
    desired_capacity  = 1
    worker_name       = "preflop_worker"
  }


  equities = {
    script_to_run     = "workers/equity/sqs_worker.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/equity-simulations-dlq"
    instance_types    = [
      "c6i.xlarge", "c6i.2xlarge", "c6i.4xlarge",
      "c6a.xlarge", "c6a.2xlarge", "c6a.4xlarge",
      "c5.xlarge",  "c5.2xlarge",  "c5.4xlarge",
      "c5a.xlarge", "c5a.2xlarge", "c5a.4xlarge",
      "c5n.xlarge", "c5n.2xlarge", "c5n.4xlarge",
      "c6in.xlarge","c6in.2xlarge","c6in.4xlarge",
      "c6id.xlarge","c6id.2xlarge","c5d.xlarge","c5d.2xlarge",
      "m6i.xlarge","m6i.2xlarge","m5.xlarge","m5.2xlarge"
    ]
    min_size          = 0
    max_size          = 10
    desired_capacity  = 0
    worker_name       = "equity_worker"
  }

  exploit = {
    script_to_run     = "workers/exploit/sqs_worker.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/214061305689/exploit-logs-queue"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/214061305689/exploit-logs-dlq"
    instance_types    = [
      "c6i.xlarge", "c6i.2xlarge", "c6i.4xlarge",
      "c6a.xlarge", "c6a.2xlarge", "c6a.4xlarge",
      "c5.xlarge",  "c5.2xlarge",  "c5.4xlarge",
      "c5a.xlarge", "c5a.2xlarge", "c5a.4xlarge",
      "c5n.xlarge", "c5n.2xlarge", "c5n.4xlarge",
      "c6in.xlarge","c6in.2xlarge","c6in.4xlarge",
      "c6id.xlarge","c6id.2xlarge","c5d.xlarge","c5d.2xlarge",
      "m6i.xlarge","m6i.2xlarge","m5.xlarge","m5.2xlarge"
    ]
    min_size          = 0
    max_size          = 10
    desired_capacity  = 0
    worker_name       = "exploit_worker"
  }
}