#!/usr/bin/env bash
# floci local-validation harness for the AWS EC2 Terraform module
# Runs floci (Docker-based AWS emulator), applies the module, validates resources, destroys.
# HONEST success = instance + SG + keypair created and destroyed cleanly.
# Docker-in-docker / full app-under-floci is OUT OF SCOPE.
#
# floci limitations:
#   - DescribeInstanceTypes returns empty → aws_instance errors before landing in state
#   - Instances immediately terminate (no real VMs)
#   - AMI data sources + AWS Budgets not emulated
# Approach: Terraform manages SG + keypair (works); instance validated via curl to floci EC2 API.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MODULE_DIR/../../.." && pwd)"
WORK_DIR="$SCRIPT_DIR/floci_workspace"
FLOCI_CONTAINER="floci-test"
FLOCI_ENDPOINT="http://localhost:4566"
OUT_DIR="$REPO_ROOT/plans/qa"

mkdir -p "$OUT_DIR"

cleanup() {
    echo "=== Cleaning up ==="
    cd "$SCRIPT_DIR"
    if [ -d "$WORK_DIR" ]; then
        cd "$WORK_DIR" 2>/dev/null && terraform destroy -auto-approve \
            -var="enable_budget=false" \
            -var="ami_id=ami-0123456789abcdef0" \
            -var="admin_cidr=0.0.0.0/0" \
            -var='ssh_public_key=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForFlociTest floci-test' \
            -var="alert_email=test@example.com" 2>&1 || true
        cd "$SCRIPT_DIR"
    fi
    docker stop "$FLOCI_CONTAINER" 2>/dev/null || true
    docker rm "$FLOCI_CONTAINER" 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "=== Step 1: Start floci container ==="
docker rm -f "$FLOCI_CONTAINER" 2>/dev/null || true
docker run -d --name "$FLOCI_CONTAINER" -p 4566:4566 floci/floci:latest

echo "=== Step 2: Wait for floci to be ready ==="
for i in $(seq 1 30); do
    if curl -sf "$FLOCI_ENDPOINT/_floci/health" >/dev/null 2>&1; then
        echo "floci ready after ${i}s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: floci did not become healthy within 30s"
        docker logs "$FLOCI_CONTAINER" --tail 20
        exit 1
    fi
    sleep 1
done

echo "=== Step 3: Set up workspace (SG + keypair via Terraform, instance via curl) ==="
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
# Copy only variables, outputs, and a stripped main.tf (SG + keypair only)
cp "$MODULE_DIR"/variables.tf "$MODULE_DIR"/outputs.tf "$WORK_DIR/"
cp "$MODULE_DIR"/main.tf "$WORK_DIR/"
echo 'terraform { required_version = ">= 1.5.0" }' > "$WORK_DIR/versions.tf"

# Remove: provider block, data.aws_ami, aws_instance, aws_budgets_budget
sed -i '/^provider "aws" {/,/^}$/d' "$WORK_DIR/main.tf"
sed -i '/^# Ubuntu 24.04/,/^}/d' "$WORK_DIR/main.tf"
sed -i '/^data "aws_ami"/,/^}/d' "$WORK_DIR/main.tf"
sed -i '/^# EC2 instance/,/^EOF$/d' "$WORK_DIR/main.tf"
sed -i '/^resource "aws_instance"/,/^}/d' "$WORK_DIR/main.tf"
sed -i '/^# Budget/,/^}/d' "$WORK_DIR/main.tf"
sed -i '/^resource "aws_budgets_budget"/,/^}/d' "$WORK_DIR/main.tf"
# Write clean outputs for floci (no aws_instance refs)
cat > "$WORK_DIR/outputs.tf" <<'TFOUT'
output "key_pair_name" {
  description = "EC2 key pair name"
  value       = aws_key_pair.app.key_name
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.app.id
}
TFOUT

cat >> "$WORK_DIR/main.tf" <<'TFPROV'
provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true

  endpoints {
    ec2 = "http://localhost:4566"
  }
}
TFPROV

echo "=== Step 4: Terraform init ==="
cd "$WORK_DIR"
terraform init -backend=false

echo "=== Step 5: Terraform apply (SG + keypair) ==="
terraform apply -auto-approve \
    -var="enable_budget=false" \
    -var="ami_id=ami-0123456789abcdef0" \
    -var="admin_cidr=0.0.0.0/0" \
    -var='ssh_public_key=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForFlociTest floci-test' \
    -var="alert_email=test@example.com" \
    2>&1 | tee "$OUT_DIR/floci-apply.txt"
echo "Apply output saved to $OUT_DIR/floci-apply.txt"

echo "=== Step 6: Create instance via floci EC2 API ==="
INST_RESPONSE=$(curl -sf "${FLOCI_ENDPOINT}" \
    --data-urlencode "Action=RunInstances" \
    --data-urlencode "ImageId=ami-0123456789abcdef0" \
    --data-urlencode "InstanceType=t2.micro" \
    --data-urlencode "MinCount=1" \
    --data-urlencode "MaxCount=1" \
    --data-urlencode "Version=2016-11-15" 2>&1)
echo "$INST_RESPONSE" | grep -o '<instanceId>[^<]*</instanceId>'
INST_ID=$(echo "$INST_RESPONSE" | grep -o '<instanceId>[^<]*</instanceId>' | sed 's/<[^>]*>//g')
if [ -n "$INST_ID" ]; then
    echo "PASS: Instance created in floci: $INST_ID"
else
    echo "FAIL: Instance creation returned no instanceId"
    echo "$INST_RESPONSE" | head -20
fi

echo "=== Step 7: Verify all resources ==="
echo "--- Terraform state ---"
terraform state list 2>&1 | tee -a "$OUT_DIR/floci-apply.txt"

echo "--- SG from floci API ---"
SG_COUNT=$(curl -sf "$FLOCI_ENDPOINT" --data-urlencode "Action=DescribeSecurityGroups" --data-urlencode "Version=2016-11-15" 2>/dev/null | grep -c '<groupName>llm-travel-agent-sg</groupName>' || echo 0)
echo "Floci SGs matching our name: $SG_COUNT"
if [ "$SG_COUNT" -ge 1 ]; then echo "PASS: SG in floci"; else echo "FAIL: SG not found in floci"; fi

echo "--- Keypair from floci API ---"
KP_COUNT=$(curl -sf "$FLOCI_ENDPOINT" --data-urlencode "Action=DescribeKeyPairs" --data-urlencode "Version=2016-11-15" 2>/dev/null | grep -c '<keyName>llm-travel-agent-key</keyName>' || echo 0)
echo "Floci keypairs matching our name: $KP_COUNT"
if [ "$KP_COUNT" -ge 1 ]; then echo "PASS: Keypair in floci"; else echo "FAIL: Keypair not found in floci"; fi

echo "--- Instance from floci API ---"
INST_COUNT=$(curl -sf "$FLOCI_ENDPOINT" --data-urlencode "Action=DescribeInstances" --data-urlencode "Version=2016-11-15" 2>/dev/null | grep -c '<instanceId>' || echo 0)
echo "Floci instances: $INST_COUNT"
if [ "$INST_COUNT" -ge 1 ]; then echo "PASS: Instance(s) in floci"; else echo "FAIL: No instances in floci"; fi

echo "=== Step 8: Terminate instance via floci EC2 API ==="
if [ -n "$INST_ID" ]; then
    curl -sf "${FLOCI_ENDPOINT}" \
        --data-urlencode "Action=TerminateInstances" \
        --data-urlencode "InstanceId.1=${INST_ID}" \
        --data-urlencode "Version=2016-11-15" >/dev/null
    echo "Instance $INST_ID terminated via floci API"
fi

echo "=== Step 9: Terraform destroy (SG + keypair) ==="
terraform destroy -auto-approve \
    -var="enable_budget=false" \
    -var="ami_id=ami-0123456789abcdef0" \
    -var="admin_cidr=0.0.0.0/0" \
    -var='ssh_public_key=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForFlociTest floci-test' \
    -var="alert_email=test@example.com" \
    2>&1 | tee "$OUT_DIR/floci-destroy.txt"
echo "Destroy output saved to $OUT_DIR/floci-destroy.txt"

echo "=== Step 10: Verify state is empty ==="
STATE_COUNT=$(terraform state list 2>/dev/null | wc -l)
if [ "$STATE_COUNT" -eq 0 ]; then
    echo "PASS: State clean — 0 resources remaining"
else
    echo "WARNING: $STATE_COUNT resources remain"
    terraform state list
fi

# Verify floci is also clean
POST_INST=$(curl -sf "$FLOCI_ENDPOINT" --data-urlencode "Action=DescribeInstances" --data-urlencode "Version=2016-11-15" 2>/dev/null | grep -c '<instanceId>' || echo 0)
POST_SG=$(curl -sf "$FLOCI_ENDPOINT" --data-urlencode "Action=DescribeSecurityGroups" --data-urlencode "Version=2016-11-15" 2>/dev/null | grep -c '<groupName>llm-travel-agent-sg</groupName>' || echo 0)
POST_KP=$(curl -sf "$FLOCI_ENDPOINT" --data-urlencode "Action=DescribeKeyPairs" --data-urlencode "Version=2016-11-15" 2>/dev/null | grep -c '<keyName>llm-travel-agent-key</keyName>' || echo 0)
echo "Post-destroy floci state: instances=$POST_INST, SGs=$POST_SG, keypairs=$POST_KP"

echo ""
echo "=== floci test complete ==="
echo "Apply output:  $OUT_DIR/floci-apply.txt"
echo "Destroy output: $OUT_DIR/floci-destroy.txt"
