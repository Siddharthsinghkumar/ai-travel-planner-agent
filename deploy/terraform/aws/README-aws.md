# AWS Bootstrap Runbook — llm-travel-agent

> ⛔ Sid's hands from here. No secrets in this doc — all placeholders.
> Terraform provisions the box; deploy + tunnel are manual next steps.

## Step 0: Budget FIRST (do this BEFORE any resource)

1. Create your AWS account (new = $100–200 credits, 6mo or until spent).
2. Go to **Billing Dashboard → Budgets → Create budget**.
3. **Cost budget**, monthly, amount = $5, add alert email.
4. This doc replicates the budget in Terraform (`aws_budgets_budget`) as a second
   line of defense, but **console-first is safer** — the budget email fires even
   if Terraform isn't running.

> **Credit burn note:** t3.micro + 30GB gp3 + public IPv4 ≈ $14/mo → credits last
> ~7–14 months then bills (~₹1130/mo). If Oracle card clears later, migrate there
> for permanent ₹0/24GB (plan §6c).

## Step 1: IAM user + programmatic access

Create an IAM user with least-privilege and programmatic access:

1. IAM → Users → Create user → `llm-travel-agent-tf`
2. Attach policies directly (minimum set):
   - `AmazonEC2FullAccess`
   - `AmazonVPCFullAccess`
   - `AWSBudgetsReadOnlyAccess` (or `AWSBudgetsActionsWithAWSResourcePolicyAccess`)
3. Under Security credentials → Create access key → **Application running outside AWS**.
4. Save the **Access Key ID** and **Secret Access Key**. You'll paste them in the next step.

## Step 2: AWS CLI

```bash
# Install (Ubuntu):
sudo apt-get install -y awscli

# Or via pip:
pip install awscli

# Configure with the IAM keys from Step 1:
aws configure
# AWS Access Key ID: <paste>
# AWS Secret Access Key: <paste>
# Default region: ap-south-1
# Default output format: json

# Verify:
aws sts get-caller-identity
# Should show the IAM user ARN.
```

## Step 3: Terraform variables

```bash
cd deploy/terraform/aws

# Copy the example and fill YOUR values:
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
region         = "ap-south-1"
admin_cidr     = "YOUR-IP/32"          # NOT 0.0.0.0/0
ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... you@host"
alert_email    = "you@example.com"
budget_usd     = 5
instance_name  = "llm-travel-agent"
swap_size_gb   = 2
app_user       = "llm-agent"
```

> `ssh_public_key` is the FULL public key content. Generate one if needed:
> `ssh-keygen -t ed25519 -C "you@example" -f ~/.ssh/llm-travel-agent-aws`

## Step 4: Terraform apply

```bash
cd deploy/terraform/aws

# Install Terraform if needed (>= 1.5.0):
# terraform version || (see https://developer.hashicorp.com/terraform/install)

terraform init
terraform plan       # Review EVERY resource. Confirm there is NO aws_eip.
terraform apply      # Type "yes" when ready.
```

Terraform outputs the `public_ip` and a pre-written `ssh` command:

```
ssh -i ~/.ssh/llm-travel-agent-key.pem llm-agent@<PUBLIC_IP>
```

## Step 5: SCP your `.env` and bring up the stack

```bash
# From your local machine:
# 1. Copy .env (secrets — NEVER committed):
scp -i ~/.ssh/llm-travel-agent-key.pem .env llm-agent@<PUBLIC_IP>:~/

# 2. Copy the project (or git clone):
scp -i ~/.ssh/llm-travel-agent-key.pem -r . llm-agent@<PUBLIC_IP>:~/llm-travel-agent

# 3. SSH in:
ssh -i ~/.ssh/llm-travel-agent-key.pem llm-agent@<PUBLIC_IP>

# 4. Inside the box:
cd ~/llm-travel-agent
cp ~/.env .

# 5. Bring up the stack:
docker compose -f docker-compose.yml \
               -f deploy/compose.prod.yml \
               -f deploy/compose.tunnel.yml \
               up -d

# 6. Start the tunnel (named — see deploy/tunnel/README.md):
export TUNNEL_TOKEN="<Sid's real token>"
docker compose -f docker-compose.yml \
               -f deploy/compose.prod.yml \
               -f deploy/compose.tunnel.yml \
               up -d
```

> The `user_data` in `main.tf` already created the swapfile, installed Docker +
> Compose plugin, created the `llm-agent` user, and added it to the `docker` group.
> No manual provisioning needed after `terraform apply`.

## Step 6: Teardown

```bash
cd deploy/terraform/aws
terraform destroy   # Type "yes". Verify clean — releases the IP, no idle-EIP bill.
```

The budget resource is also destroyed. The IAM user + console budget survive
(they were created manually in Step 0/1).

---

## Quick reference: what Terraform creates

| Resource | Purpose | Free-tier? |
|---|---|---|
| `aws_instance` (t3.micro) | Compute | ✅ 750h/mo (legacy) or credit-burn |
| `root_block_device` (30GB gp3) | Boot disk | ✅ 30GB EBS |
| `associate_public_ip_address = true` | Public IPv4 | ✅ first 12mo (legacy) or ~$3.65/mo |
| `aws_security_group` | SSH only from admin_cidr | ✅ free |
| `aws_key_pair` | SSH key | ✅ free |
| `aws_budgets_budget` ($5/mo) | Cost guard | ✅ free |

**No `aws_eip`** — idle/detached EIP bills ~$0.005/hr. The auto-assigned public
IP is free while attached and released on `terraform destroy`.

> If you ever retry Oracle and the card clears: migrate to Oracle Always Free
> (4 ARM OCPU / 24GB RAM / 200GB block, permanent ₹0) and run `terraform destroy`
> on the AWS resources.
