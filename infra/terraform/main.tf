provider "aws" {
  region = var.aws_region
}

module "sqs" {
  source = "./modules/sqs"
}

module "iam" {
  source = "./modules/iam"
}

module "ec2_spot" {
  source = "./modules/ec2_spot"

  instance_profile_name = module.iam.instance_profile_name
  ami_id                = var.ami_id
  instance_type         = var.instance_type
  key_name              = var.ec2_key_pair_name
  min_size              = var.asg_min_size
  max_size              = var.asg_max_size
  desired_capacity      = var.asg_desired_capacity
  subnet_ids            = var.subnet_ids
  security_group_ids    = var.security_group_ids
  github_token        = var.github_token
}