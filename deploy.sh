#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------
# Terradev MCP SSE — Deploy to AWS (34.207.59.52)
# Host: terradev-mcp.terradev.cloud
# ---------------------------------------------------------------

SERVER="34.207.59.52"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-~/.ssh/terradev-prod-key.pem}"
REMOTE_DIR="/home/${SSH_USER}/terradev-mcp"
DOMAIN="terradev-mcp.terradev.cloud"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying Terradev MCP SSE to ${SERVER}${NC}"
echo -e "   Domain: https://${DOMAIN}/sse"
echo ""

# ---------------------------------------------------------------
# 1. Pre-flight checks
# ---------------------------------------------------------------
if [ -z "${TERRADEV_MCP_BEARER_TOKEN:-}" ]; then
    echo -e "${YELLOW}⚠  TERRADEV_MCP_BEARER_TOKEN not set locally.${NC}"
    echo "   Generate one:  export TERRADEV_MCP_BEARER_TOKEN=\$(openssl rand -hex 32)"
    echo "   Then re-run this script."
    exit 1
fi

if [ -z "${TERRADEV_RUNPOD_KEY:-}" ]; then
    echo -e "${YELLOW}⚠  TERRADEV_RUNPOD_KEY not set. Some tools will be limited.${NC}"
fi

echo -e "${GREEN}[1/6]${NC} Pre-flight checks passed"

# ---------------------------------------------------------------
# 2. Upload files to server
# ---------------------------------------------------------------
echo -e "${GREEN}[2/6]${NC} Uploading files..."

ssh -i "${SSH_KEY}" "${SSH_USER}@${SERVER}" "mkdir -p ${REMOTE_DIR}"

scp -i "${SSH_KEY}" \
    terradev_mcp.py \
    requirements.txt \
    Dockerfile \
    docker-compose.yml \
    nginx-mcp.conf \
    "${SSH_USER}@${SERVER}:${REMOTE_DIR}/"

echo "   Files uploaded to ${REMOTE_DIR}"

# ---------------------------------------------------------------
# 3. Create .env file on server (secrets never leave the wire)
# ---------------------------------------------------------------
echo -e "${GREEN}[3/6]${NC} Writing secrets to server .env..."

ssh -i "${SSH_KEY}" "${SSH_USER}@${SERVER}" bash -s <<ENVEOF
cat > ${REMOTE_DIR}/.env <<'EOF'
TERRADEV_MCP_BEARER_TOKEN=${TERRADEV_MCP_BEARER_TOKEN}
TERRADEV_RUNPOD_KEY=${TERRADEV_RUNPOD_KEY:-}
TERRADEV_VASTAI_KEY=${TERRADEV_VASTAI_KEY:-}
TERRADEV_AWS_ACCESS_KEY_ID=${TERRADEV_AWS_ACCESS_KEY_ID:-}
TERRADEV_AWS_SECRET_ACCESS_KEY=${TERRADEV_AWS_SECRET_ACCESS_KEY:-}
TERRADEV_AWS_DEFAULT_REGION=${TERRADEV_AWS_DEFAULT_REGION:-us-east-1}
TERRADEV_LAMBDA_API_KEY=${TERRADEV_LAMBDA_API_KEY:-}
TERRADEV_COREWEAVE_API_KEY=${TERRADEV_COREWEAVE_API_KEY:-}
TERRADEV_TENSORDOCK_API_KEY=${TERRADEV_TENSORDOCK_API_KEY:-}
TERRADEV_CRUSOE_API_KEY=${TERRADEV_CRUSOE_API_KEY:-}
EOF
chmod 600 ${REMOTE_DIR}/.env
ENVEOF

# ---------------------------------------------------------------
# 4. Build & start the container
# ---------------------------------------------------------------
echo -e "${GREEN}[4/6]${NC} Building and starting container..."

ssh -i "${SSH_KEY}" "${SSH_USER}@${SERVER}" bash -s <<BUILDEOF
cd ${REMOTE_DIR}

# Stop existing container if running
docker-compose down 2>/dev/null || true

# Build and start
docker-compose --env-file .env up -d --build

# Wait for health check
echo "   Waiting for health check..."
for i in {1..15}; do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "   ✅ Container healthy"
        break
    fi
    if [ \$i -eq 15 ]; then
        echo "   ❌ Health check failed after 15 attempts"
        docker-compose logs --tail=30
        exit 1
    fi
    sleep 2
done
BUILDEOF

# ---------------------------------------------------------------
# 5. Set up nginx + SSL
# ---------------------------------------------------------------
echo -e "${GREEN}[5/6]${NC} Configuring nginx + SSL for ${DOMAIN}..."

ssh -i "${SSH_KEY}" "${SSH_USER}@${SERVER}" bash -s <<NGINXEOF
# Copy nginx config
sudo cp ${REMOTE_DIR}/nginx-mcp.conf /etc/nginx/conf.d/terradev-mcp.conf 2>/dev/null || \
sudo cp ${REMOTE_DIR}/nginx-mcp.conf /etc/nginx/sites-enabled/terradev-mcp.conf

# Check if cert already exists
if [ ! -f "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" ]; then
    echo "   Obtaining SSL certificate..."
    # Temporarily add a plain HTTP server for certbot
    sudo tee /tmp/terradev-mcp-http.conf > /dev/null <<'HTTPCONF'
server {
    listen 80;
    server_name ${DOMAIN};
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }
    location / {
        return 301 https://\$host\$request_uri;
    }
}
HTTPCONF
    sudo cp /tmp/terradev-mcp-http.conf /etc/nginx/conf.d/terradev-mcp-http.conf 2>/dev/null || \
    sudo cp /tmp/terradev-mcp-http.conf /etc/nginx/sites-enabled/terradev-mcp-http.conf
    sudo mkdir -p /var/www/certbot
    sudo nginx -t && sudo systemctl reload nginx

    # Get cert
    sudo certbot certonly --webroot -w /var/www/certbot \
        -d ${DOMAIN} \
        --non-interactive --agree-tos \
        --email admin@terradev.cloud \
        || echo "   ⚠  Certbot failed — you may need to set up DNS first"
fi

# Test and reload nginx
sudo nginx -t && sudo systemctl reload nginx
echo "   ✅ Nginx configured"
NGINXEOF

# ---------------------------------------------------------------
# 6. Verify
# ---------------------------------------------------------------
echo -e "${GREEN}[6/6]${NC} Verifying deployment..."

# Give nginx a moment
sleep 3

# Test health endpoint
echo -n "   Health check: "
HEALTH=$(curl -sf "https://${DOMAIN}/health" 2>/dev/null || curl -sf "http://${SERVER}:8080/health" 2>/dev/null || echo "FAIL")
echo "${HEALTH}"

# Test auth rejection
echo -n "   Auth check (should reject): "
AUTH_REJECT=$(curl -sf -o /dev/null -w "%{http_code}" "https://${DOMAIN}/sse" 2>/dev/null || curl -sf -o /dev/null -w "%{http_code}" "http://${SERVER}:8080/sse" 2>/dev/null || echo "N/A")
echo "HTTP ${AUTH_REJECT}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Terradev MCP SSE deployed!${NC}"
echo ""
echo -e "   Connector URL:  ${GREEN}https://${DOMAIN}/sse${NC}"
echo -e "   Health check:   https://${DOMAIN}/health"
echo -e "   Bearer token:   ${TERRADEV_MCP_BEARER_TOKEN:0:8}..."
echo ""
echo -e "   ${YELLOW}Claude.ai → Settings → Connectors → Add:${NC}"
echo -e "   URL:   https://${DOMAIN}/sse"
echo -e "   Token: ${TERRADEV_MCP_BEARER_TOKEN}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
