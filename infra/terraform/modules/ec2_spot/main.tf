resource "aws_launch_template" "worker_template" {
  name_prefix   = "preflop-worker-"
  image_id      = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name

  iam_instance_profile {
    name = var.instance_profile_name
  }

  user_data = var.user_data_script

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "spot_asg" {
  name                      = "preflop-worker-asg"
  max_size                 = var.max_size
  min_size                 = var.min_size
  desired_capacity         = var.desired_capacity
  vpc_zone_identifier      = var.subnet_ids
  health_check_type        = "EC2"
  force_delete             = true

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 0
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "lowest-price"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.worker_template.id
        version             = "$Latest"
      }
    }
  }

  tag {
    key                 = "Name"
    value               = "preflop-worker"
    propagate_at_launch = true
  }
}

output "asg_name" {
  value = aws_autoscaling_group.spot_asg.name
}