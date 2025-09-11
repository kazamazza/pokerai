provider "aws" {
  region = var.aws_region
}

module "sqs" {
  source = "./modules/sqs"
}

module "iam" {
  source = "./modules/iam"
}

# Dynamic multi-worker setup from var.worker_configs
module "spot_workers" {
  for_each = var.worker_configs
  source   = "./modules/ec2_spot"
  instance_profile_name = module.iam.instance_profile_name
  worker_name           = each.key
  ami_id                = var.ami_id
  key_name              = var.ec2_key_pair_name
  subnet_ids            = var.subnet_ids
  security_group_ids    = var.security_group_ids
  github_token          = var.github_token
  aws_access_key_id     = var.aws_access_key_id
  aws_secret_access_key = var.aws_secret_access_key

  script_to_run         = each.value.script_to_run
  aws_sqs_queue_url     = each.value.aws_sqs_queue_url
  aws_sqs_dlq_url       = each.value.aws_sqs_dlq_url

  instance_types        = each.value.instance_types
  min_size              = each.value.min_size
  max_size              = each.value.max_size
  desired_capacity      = each.value.desired_capacity
}

output "spot_worker_names" {
  value = [for k in keys(module.spot_workers) : k]
}

locals {
  enabled_jobs = {
    for job_name, config in var.job_configs :
    job_name => config
    if !lookup(config, "disabled", false)
  }
}

module "ephemeral_jobs" {
  for_each = local.enabled_jobs
  source   = "./modules/ec2_one_shot"

  instance_profile_name = module.iam.instance_profile_name
  worker_name           = each.key
  ami_id                = var.ami_id
  key_name              = var.ec2_key_pair_name
  subnet_ids            = var.subnet_ids
  security_group_ids    = var.security_group_ids
  github_token          = var.github_token

  script_to_run         = each.value.script_to_run
  aws_sqs_queue_url     = each.value.aws_sqs_queue_url
  instance_type         = each.value.instance_type
  subnet_id             = var.subnet_ids[0] # pick one subnet
  aws_sqs_dlq_url       = each.value.aws_sqs_queue_url
}

output "ephemeral_job_names" {
  value = [for k in keys(module.ephemeral_jobs) : k]
}