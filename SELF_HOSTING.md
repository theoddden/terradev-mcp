# Self-Hosting Terradev MCP Server

Complete guide to running your own Terradev MCP server with Claude.ai Connector support.

## 🎯 Why Self-Host?

- **Your own API keys**: Keep all cloud provider credentials on your infrastructure
- **Custom token management**: Generate unique tokens per user for better tracking
- **Local GPU integration**: Connect your home lab GPUs to Claude.ai
- **No rate limits**: Control your own usage policies
- **Privacy**: All requests stay within your infrastructure

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
python3 --version

# Install dependencies
pip install -r requirements.txt

# Optional: Install terradev CLI for actual GPU provisioning
pip install terradev-cli
```

### 2. Set Environment Variables

**Minimum setup** (for testing):
```bash
export TERRADEV_MCP_BEARER_TOKEN=$(openssl rand -hex 32)
echo "Your token: $TERRADEV_MCP_BEARER_TOKEN"
```

**Production setup** (with actual GPU provisioning):
```bash
# Required for MCP server
export TERRADEV_MCP_BEARER_TOKEN=$(openssl rand -hex 32)

# Required for GPU provisioning (at least one)
export TERRADEV_RUNPOD_KEY=your_runpod_api_key

# Optional: Additional cloud providers
export TERRADEV_AWS_ACCESS_KEY_ID=your_aws_key
export TERRADEV_AWS_SECRET_ACCESS_KEY=your_aws_secret
export TERRADEV_GCP_PROJECT_ID=your_gcp_project
export TERRADEV_AZURE_SUBSCRIPTION_ID=your_azure_sub
```

### 3. Run the Server

**Option A: Direct Python**
```bash
python3 terradev_mcp.py --transport sse --port 8090
```

**Option B: Systemd Service** (recommended for production)
```bash
# Create service file
sudo nano /etc/systemd/system/terradev-mcp.service
```

Paste this configuration:
```ini
[Unit]
Description=Terradev MCP SSE Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/terradev-mcp
Environment="TERRADEV_MCP_BEARER_TOKEN=your-token-here"
Environment="TERRADEV_RUNPOD_KEY=your-runpod-key"
ExecStart=/usr/bin/python3 /home/ubuntu/terradev-mcp/terradev_mcp.py --transport sse --port 8090
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable terradev-mcp
sudo systemctl start terradev-mcp
sudo systemctl status terradev-mcp
```

### 4. Set Up Nginx Reverse Proxy with SSL

**Install Nginx and Certbot:**
```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

**Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/terradev-mcp
```

Paste this configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8090;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE-specific settings
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/terradev-mcp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**Get SSL certificate:**
```bash
sudo certbot --nginx -d your-domain.com
```

### 5. Connect from Claude.ai

1. Go to **Claude.ai → Settings → Connectors**
2. Click **"Add Connector"**
3. Enter:
   - **URL**: `https://your-domain.com/sse`
   - **Token**: Your `TERRADEV_MCP_BEARER_TOKEN` value
4. Click **"Connect"**

## 🔧 Error Messages That Actually Help

The server now provides helpful error messages instead of cryptic Python tracebacks:

### Missing API Key
**Before:**
```
Traceback (most recent call last):
  File "terradev_mcp.py", line 123, in execute_terradev_command
    ...
KeyError: 'TERRADEV_RUNPOD_KEY'
```

**After:**
```
❌ Error: TERRADEV_RUNPOD_KEY not set

💡 Looks like TERRADEV_RUNPOD_KEY isn't set.
   Run: terradev setup runpod --quick
   Or set: export TERRADEV_RUNPOD_KEY=your_key_here
```

### Missing Dependencies
**Before:**
```
ModuleNotFoundError: No module named 'torch'
```

**After:**
```
❌ Error: Missing Python package: torch

💡 Missing Python package: torch
   Run: pip install torch
```

### CLI Not Installed
**Before:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'terradev'
```

**After:**
```
❌ terradev CLI not found.

📦 Install it with: pip install terradev-cli
📚 Docs: https://github.com/terradev-io/terradev-cli
```

## 🏠 Local GPU Integration

The server can discover and use local GPUs:

```bash
# In Claude.ai, ask:
"Scan my local GPUs"

# Response:
🔍 Local GPU Scan Results

✅ Found 2 local GPU(s)
📊 Total VRAM Pool: 48 GB

Devices:

• NVIDIA GeForce RTX 4090
  - Type: CUDA
  - VRAM: 24 GB
  - Compute: 8.9

• Apple Metal
  - Type: MPS
  - VRAM: 24 GB
  - Platform: arm64

💡 Usage:
• Use `provision_gpu` with `--local-first` to prefer local GPUs
• Cloud overflow will be used if local pool is insufficient
```

## 🔐 Security Best Practices

### 1. Generate Unique Tokens Per User

```bash
# Generate a new token for each user
openssl rand -hex 32

# Store in a secure database or secrets manager
# Never commit tokens to git
```

### 2. Use Environment Variables

**Never hardcode credentials:**
```bash
# ❌ BAD
TERRADEV_RUNPOD_KEY="rp-abc123"  # in code

# ✅ GOOD
export TERRADEV_RUNPOD_KEY="rp-abc123"  # in environment
```

### 3. Firewall Configuration

```bash
# Only allow HTTPS traffic
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Block direct access to port 8090
sudo ufw deny 8090/tcp
```

### 4. Monitor Logs

```bash
# Watch for suspicious activity
sudo journalctl -u terradev-mcp -f

# Check for failed auth attempts
sudo journalctl -u terradev-mcp | grep "Auth rejected"
```

## 📊 Monitoring & Debugging

### Check Server Status
```bash
sudo systemctl status terradev-mcp
```

### View Logs
```bash
# Last 50 lines
sudo journalctl -u terradev-mcp -n 50

# Follow logs in real-time
sudo journalctl -u terradev-mcp -f

# Filter for errors
sudo journalctl -u terradev-mcp | grep ERROR
```

### Test Endpoints

```bash
# Health check (no auth required)
curl https://your-domain.com/health

# OAuth discovery
curl https://your-domain.com/.well-known/oauth-authorization-server

# SSE endpoint (requires auth)
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -I https://your-domain.com/sse
```

### Common Issues

**Issue: 500 Internal Server Error**
```bash
# Check logs for Python errors
sudo journalctl -u terradev-mcp -n 100 | grep -A10 "ERROR"

# Verify dependencies
pip install -r requirements.txt
```

**Issue: 401 Unauthorized**
```bash
# Verify token is correct
echo $TERRADEV_MCP_BEARER_TOKEN

# Check if token is set in service
sudo systemctl cat terradev-mcp | grep BEARER_TOKEN
```

**Issue: Connection timeout**
```bash
# Check if service is running
sudo systemctl status terradev-mcp

# Check if port is listening
sudo netstat -tlnp | grep 8090

# Check Nginx configuration
sudo nginx -t
```

## 🎨 Customization

### Add Rate Limiting

Edit `terradev_mcp.py` to add rate limiting middleware:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Add to routes
@limiter.limit("100/hour")
async def handle_sse(request: Request):
    # ... existing code
```

### Custom Domain & Branding

1. Update `server_name` in Nginx config
2. Get SSL cert for your domain
3. Customize the `SKILL.md` file with your branding

### Multi-User Token Management

Create a simple token database:

```python
# tokens.json
{
    "user1@example.com": "token_abc123",
    "user2@example.com": "token_def456"
}

# Load in terradev_mcp.py
import json

with open('tokens.json') as f:
    VALID_TOKENS = json.load(f)
```

## 🌐 Sharing with Others

### Public Connector (Single Token)

Share these details:
```
URL: https://your-domain.com/sse
Token: your-shared-token
```

### Private Connector (Per-User Tokens)

1. Generate unique token per user
2. Share URL + their specific token
3. Track usage per token in logs

## 📦 Docker Deployment (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY terradev_mcp.py .
COPY SKILL.md .

ENV TERRADEV_MCP_BEARER_TOKEN=""
ENV TERRADEV_RUNPOD_KEY=""

EXPOSE 8090

CMD ["python3", "terradev_mcp.py", "--transport", "sse", "--port", "8090"]
```

```bash
# Build
docker build -t terradev-mcp .

# Run
docker run -d \
  -p 8090:8090 \
  -e TERRADEV_MCP_BEARER_TOKEN=your-token \
  -e TERRADEV_RUNPOD_KEY=your-key \
  --name terradev-mcp \
  terradev-mcp
```

## 🆘 Support

- **GitHub Issues**: https://github.com/theoddden/terradev-mcp/issues
- **Discord**: Join the Terradev community
- **Email**: support@terradev.io

## 📄 License

MIT License - See LICENSE file for details
