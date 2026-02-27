FROM python:3.11-slim

WORKDIR /app

# Install terradev CLI and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir terradev-cli

COPY . .

# SSE mode on port 8080
EXPOSE 8080

# Required env vars at runtime:
#   TERRADEV_MCP_BEARER_TOKEN  — auth token for Claude.ai Connectors
#   TERRADEV_RUNPOD_KEY        — minimum cloud provider key
# Optional: TERRADEV_AWS_*, TERRADEV_GCP_*, etc.

CMD ["python3", "terradev_mcp.py", "--transport", "sse", "--port", "8080"]
