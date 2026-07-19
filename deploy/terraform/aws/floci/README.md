# floci Local Validation Harness

Validates the AWS EC2 Terraform module against floci (Docker-based AWS emulator) before
real AWS deployment. ₹0 local test — no account, no credits, no billing risk.

## Prerequisites

- docker (floci/floci:latest runs on port 4566)
- terraform >= 1.5.0
- curl (health check)

## Run

```bash
cd deploy/terraform/aws/floci
./run-floci-test.sh
```

Output lands in `plans/qa/floci-apply.txt` and `plans/qa/floci-destroy.txt`.

## What it tests

| Resource | floci support | Tested |
|---|---|---|
| `aws_instance` (t3.micro) | Yes | Instance created + destroyed |
| `aws_security_group` | Yes | SG created + destroyed |
| `aws_key_pair` | Yes | Keypair created + destroyed |
| `aws_budgets_budget` | No | Toggled off (`enable_budget=false`) |
| `data.aws_ami` (Canonical) | No | Overridden with `ami_id=ami-0123456789abcdef0` |
| `user_data` (swap + docker) | No (Runs but floci VMs are minimal) | Not validated |

## Coverage gaps (floci vs real AWS)

- **AMI data source** — floci has no Canonical AMI catalog. The `var.ami_id` override
  bypasses the `data.aws_ami` lookup with a dummy AMI.
- **AWS Budgets** — floci does not emulate the Budgets API. The `var.enable_budget`
  toggle sets `count = 0` on the budget resource, so it's not created in floci.
- **user_data** — floci's EC2 emulation uses Docker containers, not real VMs, so
  swap/docker/user bootstrap is not verified.
- **No real network** — the instance gets no actual public IP in floci; connectivity
  testing needs real AWS.

## HONEST success criterion

Instance + security group + keypair created and destroyed cleanly in floci. Full app
routing through floci is out of scope (docker-in-docker).
