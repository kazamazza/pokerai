aws_region = "eu-central-1"
aws_access_key_id = "AKIATQO3HPKQVGE3Z3OZ"
aws_secret_access_key = "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"
ami_id     = "ami-0dc33c9c954b3f073"
ec2_key_pair_name = "mainkey"
github_token = "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"

subnet_ids = [
  "subnet-0586ffd6ad14a0d37",
  "subnet-0221dde0eb94927d1"
]

security_group_ids = [
  "sg-0ff95ff712aa7b35e"
]

job_configs = {
  postflop_dataset_shard_0 = {
    script_to_run     = "ml/etl/rangenet/postflop/build_rangenet_postflop_dataset.py --shard-count 6 --shard-index 0 --parts-local-dir data/datasets/postflop_policy_parts --parts-s3-prefix datasets/postflop_policy/parts"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"  # 64 GiB RAM
    volume_size       = 300
    worker_name       = "postflop_dataset_shard_0"
    disabled          = true
  }

  postflop_dataset_shard_1 = {
    script_to_run     = "ml/etl/rangenet/postflop/build_rangenet_postflop_dataset.py --shard-count 6 --shard-index 1 --parts-local-dir data/datasets/postflop_policy_parts --parts-s3-prefix datasets/postflop_policy/parts"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 300
    worker_name       = "postflop_dataset_shard_1"
    disabled          = true
  }

  postflop_dataset_shard_2 = {
    script_to_run     = "ml/etl/rangenet/postflop/build_rangenet_postflop_dataset.py --shard-count 6 --shard-index 2 --parts-local-dir data/datasets/postflop_policy_parts --parts-s3-prefix datasets/postflop_policy/parts"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 300
    worker_name       = "postflop_dataset_shard_2"
    disabled          = true
  }

  postflop_dataset_shard_3 = {
    script_to_run     = "ml/etl/rangenet/postflop/build_rangenet_postflop_dataset.py --shard-count 6 --shard-index 3 --parts-local-dir data/datasets/postflop_policy_parts --parts-s3-prefix datasets/postflop_policy/parts"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 300
    worker_name       = "postflop_dataset_shard_3"
    disabled          = true
  }

  postflop_dataset_shard_4 = {
    script_to_run     = "ml/etl/rangenet/postflop/build_rangenet_postflop_dataset.py --shard-count 6 --shard-index 4 --parts-local-dir data/datasets/postflop_policy_parts --parts-s3-prefix datasets/postflop_policy/parts"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 300
    worker_name       = "postflop_dataset_shard_4"
    disabled          = true
  }

  postflop_dataset_shard_5 = {
    script_to_run     = "ml/etl/rangenet/postflop/build_rangenet_postflop_dataset.py --shard-count 6 --shard-index 5 --parts-local-dir data/datasets/postflop_policy_parts --parts-s3-prefix datasets/postflop_policy/parts"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 300
    worker_name       = "postflop_dataset_shard_5"
    disabled          = true
  }

  postflop_heavy = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"  # ≥64 GiB RAM; pick what you decided
    volume_size       = 150
    worker_name       = "postflop_heavy"
    disabled          = false
  }

  postflop_heavy_1 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_1"
    disabled          = false
  }

  postflop_heavy_2 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_2"
    disabled          = true
  }

  postflop_heavy_3 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_3"
    disabled          = true
  }

  postflop_heavy_4 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_4"
    disabled          = true
  }

   postflop_heavy_5 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_5"
    disabled          = true
  }

   postflop_heavy_6 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_6"
    disabled          = true
  }


  postflop_heavy_7 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
   aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_7"
    disabled          = true
  }

  postflop_heavy_8 = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_type     = "r6i.2xlarge"
    volume_size       = 150
    worker_name       = "postflop_heavy_8"
    disabled          = true
  }
}

worker_configs = {
  preflop = {
    script_to_run     = "tools/rangenet/worker_flop.py"
    aws_sqs_queue_url = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"
    aws_sqs_dlq_url   = "https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
    instance_types    = [
      # Primary compute families
      "m6i.xlarge", "m6i.2xlarge", "r6i.xlarge", "r6i.2xlarge",
  "m5.xlarge",  "m5.2xlarge",  "r5.xlarge",  "r5.2xlarge"
    ]
    min_size          = 0
    max_size          = 15
    desired_capacity  = 0
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