resource "aws_instance" "one_shot" {
  ami                  = var.ami_id
  instance_type        = var.instance_type
  key_name             = var.key_name
  monitoring           = true
  iam_instance_profile = var.instance_profile_name

  subnet_id              = var.subnet_id
  vpc_security_group_ids = var.security_group_ids

user_data = templatefile("${path.module}/cloud-init.sh.tpl", {
  github_token          = var.github_token
  script_to_run         = var.script_to_run
  aws_sqs_queue_url     = var.aws_sqs_queue_url
  aws_sqs_dlq_url       = var.aws_sqs_dlq_url
  worker_name           = var.worker_name
  aws_access_key_id     = coalesce(var.aws_access_key_id, "")
  aws_secret_access_key = coalesce(var.aws_secret_access_key, "")
  worker_threads        = var.worker_threads
})

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    iops                  = 9000
    throughput            = 500
  }

  tags = {
    Name = "${var.worker_name}-job-instance"
  }

  lifecycle {
    create_before_destroy = true
  }
}