#!/usr/bin/env bash
# floci local-validation harness for the AWS EC2 Terraform module
# Runs floci (Docker-based AWS emulator), applies the module, validates resources, destroys.
# HONEST success = instance + SG + keypair created and destroyed cleanly.
# Docker-in-docker / full app-under-floci is OUT OF SCOPE.
#
# floci note: Terraform AWS provider polls for instance "running" state after creation,
# but floci immediately terminates instances (no real VMs). The instance IS created in
# floci and recorded in Terraform state; the apply exit is non-zero but resources exist.
# We verify all 3 in state, then destroy cleans up everything.

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

echo "=== Step 3: Set up workspace ==="
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cp "$MODULE_DIR"/*.tf "$WORK_DIR/"
# Do NOT copy .terraform.lock.hcl — let floci init pick latest provider (v5.100 has
# a DescribeInstanceTypes call that floci doesn't support; latest handles it)

# Strip real-AWS provider block, version constraint, replace with floci provider + no version pin
sed -i '/^provider "aws" {/,/^}$/d' "$WORK_DIR/main.tf"
echo 'terraform { required_version = ">= 1.5.0" }' > "$WORK_DIR/versions.tf"
# t3.micro not supported by floci
sed -i 's/instance_type *= *"t3.micro"/instance_type = "t2.micro"/' "$WORK_DIR/main.tf"

cat > "$WORK_DIR/provider_floci.tf" <<'TFPROV'
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

echo "=== Step 5: Terraform apply ==="
# floci terminates instances immediately (no real VMs). The provider polls for
# "running" and errors, but the instance IS created in floci and tracked in state.
# We allow the apply to exit non-zero and verify resources afterward.
set +e
terraform apply -auto-approve \
    -var="enable_budget=false" \
    -var="ami_id=ami-0123456789abcdef0" \
    -var="admin_cidr=0.0.0.0/0" \
    -var='ssh_public_key=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForFlociTest floci-test' \
    -var="alert_email=test@example.com" \
    2>&1 | tee "$OUT_DIR/floci-apply.txt"
APPLY_EXIT=$?
set -e
echo "Apply exit code: $APPLY_EXIT (non-zero expected — floci instances don't reach 'running')"
echo "Apply output saved to $OUT_DIR/floci-apply.txt"

echo "=== Step 6: Verify resources in state ==="
terraform state list 2>&1 | tee -a "$OUT_DIR/floci-apply.txt"
STATE_LIST=$(terraform state list 2>/dev/null)
if echo "$STATE_LIST" | grep -q "aws_key_pair"; then
    echo "PASS: key_pair in state"
else
    echo "FAIL: key_pair NOT in state"
fi
if echo "$STATE_LIST" | grep -q "aws_security_group"; then
    echo "PASS: security_group in state"
else
    echo "FAIL: security_group NOT in state"
fi
if echo "$STATE_LIST" | grep -q "aws_instance"; then
    echo "PASS: instance in state"
else
    echo "FAIL: instance NOT in state"
fi

echo "=== Step 7: Terraform destroy ==="
terraform destroy -auto-approve \
    -var="enable_budget=false" \
    -var="ami_id=ami-0123456789abcdef0" \
    -var="admin_cidr=0.0.0.0/0" \
    -var='ssh_public_key=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForFlociTest floci-test' \
    -var="alert_email=test@example.com" \
    2>&1 | tee "$OUT_DIR/floci-destroy.txt"
echo "Destroy output saved to $OUT_DIR/floci-destroy.txt"

echo "=== Step 8: Verify state is empty ==="
STATE_COUNT=$(terraform state list 2>/dev/null | wc -l)
if [ "$STATE_COUNT" -eq 0 ]; then
    echo "PASS: State clean — 0 resources remaining"
else
    echo "WARNING: $STATE_COUNT resources remain"
    terraform state list
fi

echo ""
echo "=== floci test complete ==="
echo "Apply output:  $OUT_DIR/floci-apply.txt"
echo "Destroy output: $OUT_DIR/floci-destroy.txt"
