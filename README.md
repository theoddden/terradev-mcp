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

## Available MCP Tools

The Terradev MCP server provides 17 tools for complete GPU cloud management:

### GPU Operations
- **`quote_gpu`** - Get real-time GPU prices across all cloud providers
- **`provision_gpu`** - Provision GPU instances with cost optimization

### Kubernetes Management  
- **`k8s_create`** - Create Kubernetes clusters with GPU nodes
- **`k8s_list`** - List all Kubernetes clusters
- **`k8s_info`** - Get detailed cluster information
- **`k8s_destroy`** - Destroy Kubernetes clusters

### Inference & Model Deployment
- **`inferx_deploy`** - Deploy models to InferX serverless platform
- **`inferx_status`** - Check inference endpoint status
- **`inferx_list`** - List deployed inference models
- **`inferx_optimize`** - Get cost analysis for inference endpoints
- **`hf_space_deploy`** - Deploy models to HuggingFace Spaces

### Instance & Cost Management
- **`status`** - View all instances and costs
- **`manage_instance`** - Stop/start/terminate GPU instances
- **`analytics`** - Get cost analytics and spending trends
- **`optimize`** - Find cheaper alternatives for running instances

### Provider Configuration
- **`setup_provider`** - Get setup instructions for any cloud provider
- **`configure_provider`** - Configure provider credentials locally

## Complete Command Reference

### GPU Price Quotes
```bash
# Get prices for specific GPU type
terradev quote -g H100

# Filter by specific providers
terradev quote -g A100 -p runpod,vastai,lambda

# Quick-provision cheapest option
terradev quote -g H100 --quick
```

### GPU Provisioning
```bash
# Provision single GPU
terradev provision -g A100

# Provision multiple GPUs in parallel
terradev provision -g H100 -n 4 --parallel 6

# Dry run to see plan without launching
terradev provision -g A100 -n 8 --dry-run

# Set maximum price ceiling
terradev provision -g A100 --max-price 2.50

# Prefer spot instances for cost savings
terradev provision -g H100 --prefer-spot
```

### Kubernetes Clusters
```bash
# Create multi-cloud K8s cluster
terradev k8s create my-cluster --gpu H100 --count 4 --multi-cloud --prefer-spot

# List all clusters
terradev k8s list

# Get cluster details
terradev k8s info my-cluster

# Destroy cluster
terradev k8s destroy my-cluster
```

### Inference Deployment
```bash
# Deploy model to InferX
terradev inferx deploy --model meta-llama/Llama-2-7b-hf --gpu-type a10g

# Check endpoint status
terradev inferx status

# List deployed models
terradev inferx list

# Get cost analysis
terradev inferx optimize
```

### HuggingFace Spaces
```bash
# Deploy LLM template
terradev hf-space my-llama --model-id meta-llama/Llama-2-7b-hf --template llm

# Deploy with custom hardware
terradev hf-space my-model --model-id microsoft/DialoGPT-medium --hardware a10g-large --sdk gradio

# Deploy embedding model
terradev hf-space my-embeddings --model-id sentence-transformers/all-MiniLM-L6-v2 --template embedding
```

### Instance Management
```bash
# View all running instances and costs
terradev status --live

# Stop instance
terradev manage -i <instance-id> -a stop

# Start instance
terradev manage -i <instance-id> -a start

# Terminate instance
terradev manage -i <instance-id> -a terminate
```

### Analytics & Optimization
```bash
# Get 30-day cost analytics
terradev analytics --days 30

# Find cheaper alternatives
terradev optimize
```

### Provider Setup
```bash
# Get quick setup instructions
terradev setup runpod --quick
terradev setup aws --quick
terradev setup vastai --quick

# Configure credentials (stored locally)
terradev configure --provider runpod
terradev configure --provider aws
terradev configure --provider vastai
```

## Supported GPU Types

- **H100** - NVIDIA H100 80GB (premium training)
- **A100** - NVIDIA A100 80GB (training/inference)  
- **A10G** - NVIDIA A10G 24GB (inference)
- **L40S** - NVIDIA L40S 48GB (rendering/inference)
- **L4** - NVIDIA L4 24GB (inference)
- **T4** - NVIDIA T4 16GB (light inference)
- **RTX4090** - NVIDIA RTX 4090 24GB (consumer)
- **RTX3090** - NVIDIA RTX 3090 24GB (consumer)
- **V100** - NVIDIA V100 32GB (legacy)

## Supported Cloud Providers

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
