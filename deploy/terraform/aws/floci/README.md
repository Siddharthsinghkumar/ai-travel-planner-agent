# floci Local Validation Harness

Validates the AWS EC2 Terraform module against floci (Docker-based AWS emulator) before
real AWS deployment. ₹0 local test — no account, no credits, no billing risk.

## Prerequisites

- docker (floci/floci:latest on port 4566)
- terraform >= 1.5.0
- curl

## Run

```bash
cd deploy/terraform/aws/floci
./run-floci-test.sh
```

Output lands in `plans/qa/floci-apply.txt` and `plans/qa/floci-destroy.txt`.

## What it tests

| Resource | floci support | How tested |
|---|---|---|
| `aws_security_group` | Yes | Terraform provider — created + destroyed |
| `aws_key_pair` | Yes | Terraform provider — created + destroyed |
| `aws_instance` | Partial | curl to floci EC2 API (RunInstances / TerminateInstances) |
| `aws_budgets_budget` | No | Toggled off (`enable_budget=false`) |
| `data.aws_ami` (Canonical) | No | Overridden with `ami_id=ami-0123456789abcdef0` |

## Coverage gaps (floci vs real AWS)

- **aws_instance via Terraform provider** — blocked by two floci limitations:
  1. `DescribeInstanceTypes` returns empty (not implemented), causing the v5.100+
     AWS provider to error with "collecting instance settings: empty result". The
     instance IS created in floci but never lands in Terraform state.
  2. Instances immediately terminate (no real VMs), so the provider's "running"
     poll always fails even without the DescribeInstanceTypes issue.
  Workaround: the harness creates/terminates instances directly via curl to the
  floci EC2 API (RunInstances / TerminateInstances), validating floci's EC2
  emulation while keeping Terraform for SG and keypair management.

- **AWS Budgets** — not emulated. `var.enable_budget` sets `count = 0`.

- **AMI data source** — no Canonical AMI catalog. `var.ami_id` override bypasses it.

- **user_data** — floci VMs are Docker containers, not real VMs. Not validated.

- **t3.micro instance type** — unsupported by floci. Harness substitutes `t2.micro`.

- **Provider version** — harness uses latest AWS provider (v6.x) unpinned. The
  real module pins `~> 5.0` via .terraform.lock.hcl. Provider differences are
  floci-specific and do not affect the real-AWS path.

## HONEST success criterion

Instance + security group + keypair created and destroyed cleanly in floci. Full app
routing through floci is out of scope (docker-in-docker).
