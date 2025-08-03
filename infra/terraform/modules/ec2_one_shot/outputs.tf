output "instance_id" {
  value = aws_instance.one_shot.id
}

output "public_ip" {
  value = aws_instance.one_shot.public_ip
}

output "private_ip" {
  value = aws_instance.one_shot.private_ip
}

output "availability_zone" {
  value = aws_instance.one_shot.availability_zone
}