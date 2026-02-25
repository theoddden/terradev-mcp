# Terradev MCP Server

GPU Cloud Provisioning for Claude Code - Access 11+ cloud providers through natural language.

## Installation

### Prerequisites

1. Install Terradev CLI:
```bash
pip install terradev-cli
# For all providers + HF Spaces:
pip install "terradev-cli[all]"
```

2. Set up minimum credentials (RunPod only):
```bash
export TERRADEV_RUNPOD_KEY=your_runpod_api_key
```

3. Install the MCP server:
```bash
npm install -g terradev-mcp
```

### Claude Code Setup

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "terradev": {
      "command": "terradev-mcp"
    }
  }
}
```

## Features

- **GPU Price Quotes**: Real-time pricing across 11+ cloud providers
- **GPU Provisioning**: Launch instances with automatic cost optimization
- **Kubernetes Clusters**: Multi-cloud K8s with GPU nodes
- **Inference Deployment**: Serverless model serving with InferX
- **HuggingFace Spaces**: Public model deployment
- **Cost Analytics**: Track and optimize cloud spending

## Quick Start

```bash
# Check MCP connection
/mcp

# Find cheapest H100
terradev quote -g H100

# Provision 4 A100s
terradev provision -g A100 -n 4 --parallel 6

# Create K8s cluster
terradev k8s create my-cluster --gpu H100 --count 4 --multi-cloud
```

## Supported Providers

RunPod, Vast.ai, AWS, GCP, Azure, Lambda Labs, CoreWeave, TensorDock, Oracle Cloud, Crusoe Cloud, DigitalOcean, HyperStack

## Environment Variables

Minimum setup:
- `TERRADEV_RUNPOD_KEY`: RunPod API key

Full multi-cloud setup:
- `TERRADEV_AWS_ACCESS_KEY_ID`, `TERRADEV_AWS_SECRET_ACCESS_KEY`, `TERRADEV_AWS_DEFAULT_REGION`
- `TERRADEV_GCP_PROJECT_ID`, `TERRADEV_GCP_CREDENTIALS_PATH`
- `TERRADEV_AZURE_SUBSCRIPTION_ID`, `TERRADEV_AZURE_CLIENT_ID`, `TERRADEV_AZURE_CLIENT_SECRET`, `TERRADEV_AZURE_TENANT_ID`
- Additional provider keys (VastAI, Oracle, Lambda, CoreWeave, Crusoe, TensorDock)
- `HF_TOKEN`: For HuggingFace Spaces deployment

## Security

**BYOAPI**: All API keys stay on your machine. Terradev never proxies credentials through third parties.

## Links

- [GitHub](https://github.com/theoddden/Terradev)
- [PyPI](https://pypi.org/project/terradev-cli/)
- [Docs](https://theodden.github.io/Terradev/)
