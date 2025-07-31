resource "aws_iam_role" "ec2_instance_role" {
  name = "preflop-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      },
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_policy" "full_access_policy" {
  name        = "PreflopEC2FullAccessPolicy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "sqs:*",
          "s3:*"
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_policy" {
  role       = aws_iam_role.ec2_instance_role.name
  policy_arn = aws_iam_policy.full_access_policy.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "preflop-ec2-profile"
  role = aws_iam_role.ec2_instance_role.name
}

output "instance_profile_name" {
  value = aws_iam_instance_profile.ec2_profile.name
}