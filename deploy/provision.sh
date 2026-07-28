#!/usr/bin/env bash
set -euo pipefail

# Canonical single-node provision script (host-agnostic: Ubuntu ARM or x86).
# See: deploy/README-deploy.md, docs/deployment-topology.md
#
# Executes only host-level setup: Docker, ufw, fail2ban, non-root user.
# NO secrets. NO app deploy. Run once on a fresh Ubuntu VPS.

APP_USER="${APP_USER:-llm-agent}"
APP_HOME="/home/${APP_USER}"
DOCKER_COMPOSE_VERSION="${DOCKER_COMPOSE_VERSION:-v2.29.1}"

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must run as root (host-level provisioning)" >&2
  exit 1
fi

echo "=== Provisioning $(lsb_release -ds 2>/dev/null || uname -m) for llm-travel-agent ==="

apt-get update -qq
apt-get upgrade -y -qq

apt-get install -y -qq \
  curl wget ca-certificates gnupg lsb-release \
  ufw fail2ban sqlite3 \
  unattended-upgrades apt-listchanges

ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

systemctl enable fail2ban
systemctl restart fail2ban

id -u "${APP_USER}" &>/dev/null || useradd -m -s /bin/bash "${APP_USER}"
usermod -aG docker "${APP_USER}" 2>/dev/null || true
mkdir -p "${APP_HOME}/.ssh"
chmod 700 "${APP_HOME}/.ssh"
touch "${APP_HOME}/.ssh/authorized_keys"
chmod 600 "${APP_HOME}/.ssh/authorized_keys"
chown -R "${APP_USER}:${APP_USER}" "${APP_HOME}/.ssh"

mkdir -p /var/lib/llm-travel-agent
mkdir -p /var/backups/llm-travel-agent
chown "${APP_USER}:${APP_USER}" /var/lib/llm-travel-agent /var/backups/llm-travel-agent

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
fi

if ! command -v docker-compose >/dev/null 2>&1 && ! docker compose version >/dev/null 2>&1; then
  curl -fsSL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
  chmod +x /usr/local/bin/docker-compose
fi

echo "=== Provisioning complete ==="
echo "Next: add your SSH public key to ${APP_HOME}/.ssh/authorized_keys"
echo "      then switch to user ${APP_USER} and proceed with deploy/README-deploy.md"
