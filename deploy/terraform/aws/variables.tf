variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "admin_cidr" {
  description = "Your IP /32 for SSH access (NEVER 0.0.0.0/0)"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key content for EC2 key pair"
  type        = string
}

variable "alert_email" {
  description = "Email for AWS Budget alerts"
  type        = string
}

variable "budget_usd" {
  description = "Monthly budget in USD"
  type        = number
  default     = 5
}

variable "instance_name" {
  description = "Name tag for the EC2 instance"
  type        = string
  default     = "llm-travel-agent"
}

variable "swap_size_gb" {
  description = "Swap file size in GB"
  type        = number
  default     = 2
}

variable "app_user" {
  description = "Non-root user for the application"
  type        = string
  default     = "llm-agent"
}

variable "ami_id" {
  description = "Override AMI ID for testing (floci has no Canonical AMIs). Leave empty for real AWS."
  type        = string
  default     = ""
}

variable "enable_budget" {
  description = "Whether to create the AWS Budget resource (floci does not emulate Budgets)"
  type        = bool
  default     = true
}
