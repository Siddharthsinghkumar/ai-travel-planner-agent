provider "aws" {
  region = var.region
}

# Ubuntu 24.04 LTS AMI (Canonical, official)
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-*-24.04-*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# SSH key pair
resource "aws_key_pair" "app" {
  key_name   = "${var.instance_name}-key"
  public_key = var.ssh_public_key
}

# Security group: SSH only from admin CIDR, NO 80/443 ingress, egress all
resource "aws_security_group" "app" {
  name        = "${var.instance_name}-sg"
  description = "SSH from admin IP only; all egress for cloudflared tunnel"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_cidr]
    description = "SSH from operator IP only"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound (cloudflared tunnel, apt, docker pulls)"
  }

  tags = {
    Name = "${var.instance_name}-sg"
  }
}

# EC2 instance — free-tier pinned shape
resource "aws_instance" "app" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = "t3.micro"
  key_name                    = aws_key_pair.app.key_name
  vpc_security_group_ids      = [aws_security_group.app.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    tags = {
      Name = "${var.instance_name}-root"
    }
  }

  # user_data: swap first (1GB RAM box), then Docker + Compose, then non-root user.
  # Mirrors deploy/provision.sh — swap+docker+non-root only; no app deploy here.
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail

    # 1. Swap file (needed on 1GB RAM for compose stack)
    if ! swapon --show | grep -q /swapfile; then
      fallocate -l ${var.swap_size_gb}G /swapfile
      chmod 600 /swapfile
      mkswap /swapfile
      swapon /swapfile
      echo '/swapfile none swap sw 0 0' >> /etc/fstab
    fi

    # 2. Docker
    if ! command -v docker >/dev/null 2>&1; then
      curl -fsSL https://get.docker.com | sh
      systemctl enable docker
      systemctl start docker
    fi

    # 3. Docker Compose plugin
    if ! docker compose version >/dev/null 2>&1; then
      apt-get update -qq
      apt-get install -y -qq docker-compose-plugin
    fi

    # 4. Non-root app user
    id -u "${var.app_user}" &>/dev/null || useradd -m -s /bin/bash "${var.app_user}"
    usermod -aG docker "${var.app_user}"

    # 5. App directories
    mkdir -p /var/lib/llm-travel-agent /var/backups/llm-travel-agent
    chown "${var.app_user}:${var.app_user}" /var/lib/llm-travel-agent /var/backups/llm-travel-agent

    echo "AWS user_data bootstrap complete."
  EOF

  tags = {
    Name = var.instance_name
  }
}

# Budget — cost guard in code (NOTIFY only; does not terminate resources)
resource "aws_budgets_budget" "monthly" {
  name         = "${var.instance_name}-monthly"
  budget_type  = "COST"
  limit_amount = var.budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}
