resource "aws_launch_template" "worker_template" {
  name_prefix = "${var.worker_name}-worker-"
  image_id    = var.ami_id
  key_name    = var.key_name

  network_interfaces {
    device_index                = 0
    associate_public_ip_address = true
  }

  monitoring { enabled = true }

  iam_instance_profile { name = var.instance_profile_name }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  user_data = base64encode(
    templatefile("${path.module}/cloud-init.sh.tpl", {
      github_token          = var.github_token,
      script_to_run         = var.script_to_run,
      aws_sqs_queue_url     = var.aws_sqs_queue_url,
      aws_sqs_dlq_url       = var.aws_sqs_dlq_url,
      aws_access_key_id     = var.aws_access_key_id,
      aws_secret_access_key = var.aws_secret_access_key,
      worker_name           = var.worker_name
    })
  )

  metadata_options { http_tokens = "required" }

  lifecycle { create_before_destroy = true }
}

resource "aws_autoscaling_group" "spot_asg" {
  name                 = "${var.worker_name}-worker-asg"
  max_size             = var.max_size
  min_size             = var.min_size
  desired_capacity     = var.desired_capacity
  vpc_zone_identifier  = var.subnet_ids
  health_check_type    = "EC2"
  health_check_grace_period = 120
  force_delete         = true
  capacity_rebalance   = true

  mixed_instances_policy {
    instances_distribution {
      # Always launch at least one On-Demand so work starts even if Spot is scarce
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "capacity-optimized"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.worker_template.id
        version            = "$Latest"
      }

      # Single "override" using instance requirements so EC2 can pick any matching pool
      override {
        instance_requirements {
          vcpu_count {
            min = 4
            max = 32
          }
          memory_mib {
            min = 32768     # >= 32 GiB
          }
          cpu_manufacturers       = ["intel", "amd"]
          instance_generations    = ["current", "previous"]
          excluded_instance_types = ["*metal*"]
          bare_metal              = "excluded"
          accelerator_count {
            max = 0
          }
        }
      }
    }
  }

  tag {
    key                 = "Name"
    value               = "${var.worker_name}-worker"
    propagate_at_launch = true
  }
}