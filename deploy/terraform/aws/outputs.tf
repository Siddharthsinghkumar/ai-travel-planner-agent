output "public_ip" {
  description = "Auto-assigned public IPv4 of the EC2 instance"
  value       = aws_instance.app.public_ip
}

output "ssh_command" {
  description = "SSH into the instance (paste this)"
  value       = "ssh -i ~/.ssh/${aws_key_pair.app.key_name}.pem ${var.app_user}@${aws_instance.app.public_ip}"
}

output "destroy_command" {
  description = "Tear down all resources (verify clean — releases the IP, no idle-EIP bill)"
  value       = "cd deploy/terraform/aws && terraform destroy"
}
