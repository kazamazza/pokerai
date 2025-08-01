aws_region = "eu-central-1"
ami_id     = "ami-0dc33c9c954b3f073"
instance_type = "t3.large"
ec2_key_pair_name = "mainkey" # You'll create this in EC2
github_token = "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"

subnet_ids = [
  "subnet-08a0ac8d2a485dd67",
  "subnet-09a4eded6c53fc0a2"
]

security_group_ids = [
  "sg-0e2945357ff34782a"
]

asg_min_size         = 5
asg_max_size         = 15
asg_desired_capacity = 10

