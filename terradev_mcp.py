#!/usr/bin/env python3
"""
Terradev MCP Server - GPU Cloud Provisioning for Claude Code

This MCP server provides access to Terradev CLI functionality for GPU provisioning,
price comparison, Kubernetes cluster management, and inference deployment across
20 cloud providers. Includes Terraform parallel provisioning for optimal efficiency.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import subprocess
import sys
import tempfile
import time
import shutil
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, parse_qs, urlparse

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.server.sse import SseServerTransport
    from mcp.server import NotificationOptions
    from mcp.server.sse import TransportSecuritySettings
    from mcp.types import (
        CallToolRequest,
        CallToolResult,
        GetPromptRequest,
        GetPromptResult,
        ListPromptsRequest,
        ListPromptsResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListToolsRequest,
        ListToolsResult,
        ReadResourceRequest,
        ReadResourceResult,
        Resource,
        TextContent,
        TextResourceContents,
        Tool,
    )
except ImportError:
    print("Error: mcp package not found. Please install it with: pip install 'mcp[cli]'", file=sys.stderr)
    sys.exit(1)

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Mount, Route
    import uvicorn
except ImportError:
    # SSE deps are optional — only needed for remote mode
    Starlette = None
    uvicorn = None

logger = logging.getLogger("terradev-mcp")

# Check if terradev CLI is available
def check_terradev_installation():
    try:
        result = subprocess.run(
            ["terradev", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

# Local GPU Discovery
async def discover_local_gpus() -> Dict[str, Any]:
    """Discover local GPU devices on the network and current machine.
    
    Returns a dict with:
    - local_devices: List of GPUs on current machine
    - total_vram: Total VRAM available locally
    - device_details: Detailed info per device
    """
    devices = []
    total_vram = 0
    
    try:
        # Try to import torch for CUDA detection
        import torch
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(i),
                    'vram_gb': round(props.total_memory / (1024**3), 2),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count
                }
                devices.append(device_info)
                total_vram += device_info['vram_gb']
        
        # Check for Apple Metal/MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Estimate unified memory (simplified - actual detection is complex)
            import platform
            if platform.system() == 'Darwin':
                # Try to get system memory as proxy for unified memory
                try:
                    import psutil
                    total_mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
                    device_info = {
                        'id': len(devices),
                        'type': 'mps',
                        'name': 'Apple Metal',
                        'vram_gb': total_mem_gb,  # Unified memory
                        'platform': platform.machine()
                    }
                    devices.append(device_info)
                    total_vram += device_info['vram_gb']
                except ImportError:
                    pass
    
    except ImportError:
        # torch not available, try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            device_info = {
                                'id': int(parts[0]),
                                'type': 'cuda',
                                'name': parts[1],
                                'vram_gb': round(float(parts[2]) / 1024, 2)
                            }
                            devices.append(device_info)
                            total_vram += device_info['vram_gb']
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return {
        'local_devices': devices,
        'total_vram_gb': round(total_vram, 2),
        'device_count': len(devices),
        'has_local_gpu': len(devices) > 0
    }

async def estimate_model_memory(model_name: str) -> float:
    """Estimate memory requirements for a model.
    
    Simple heuristic based on model size in name (e.g., '72B' -> 72 billion params).
    Returns estimated VRAM in GB.
    """
    import re
    
    # Extract parameter count from model name
    match = re.search(r'(\d+)B', model_name, re.IGNORECASE)
    if match:
        params_b = int(match.group(1))
        # Rough estimate: 2 bytes per param (fp16) + 20% overhead
        return params_b * 2 * 1.2
    
    # Default estimates for common models
    model_lower = model_name.lower()
    if '7b' in model_lower:
        return 16
    elif '13b' in model_lower:
        return 28
    elif '70b' in model_lower or '72b' in model_lower:
        return 150
    elif '405b' in model_lower:
        return 850
    
    # Unknown model, return conservative estimate
    return 20

# Execute terradev command safely with bug fixes
async def execute_terradev_command(args: List[str]) -> Dict[str, Any]:
    """Execute terradev CLI command with helpful error messages."""
    try:
        cmd = ["terradev"] + args
        
        # Apply bug fixes for known issues
        env = os.environ.copy()
        
        # Fix 3: Ensure proxy settings are respected
        env['TRUST_ENV'] = 'true'
        
        # Fix 4: Ensure boto3 is available (will be handled by requirements)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        stdout, stderr = await process.communicate()
        stderr_text = stderr.decode().strip()
        
        # Enhance error messages with helpful guidance
        if process.returncode != 0:
            stderr_text = enhance_error_message(stderr_text, args)
        
        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode().strip(),
            "stderr": stderr_text,
            "returncode": process.returncode
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "❌ terradev CLI not found.\n\n" +
                     "📦 Install it with: pip install terradev-cli\n" +
                     "📚 Docs: https://github.com/terradev-io/terradev-cli",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"❌ Unexpected error: {str(e)}",
            "returncode": -1
        }

def enhance_error_message(stderr: str, args: List[str]) -> str:
    """Add helpful guidance to error messages."""
    # Check for common API key errors
    if "TERRADEV_RUNPOD_KEY" in stderr or "RunPod" in stderr:
        return (f"{stderr}\n\n"
                "💡 Looks like TERRADEV_RUNPOD_KEY isn't set.\n"
                "   Run: terradev setup runpod --quick\n"
                "   Or set: export TERRADEV_RUNPOD_KEY=your_key_here")
    
    if "AWS" in stderr and "credentials" in stderr.lower():
        return (f"{stderr}\n\n"
                "💡 AWS credentials not configured.\n"
                "   Run: aws configure\n"
                "   Or: terradev setup aws --quick")
    
    if "GOOGLE" in stderr or "GCP" in stderr:
        return (f"{stderr}\n\n"
                "💡 Google Cloud credentials not found.\n"
                "   Run: gcloud auth application-default login\n"
                "   Or: terradev setup gcp --quick")
    
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        # Extract module name
        import re
        match = re.search(r"No module named '([^']+)'", stderr)
        if match:
            module = match.group(1)
            return (f"{stderr}\n\n"
                   f"💡 Missing Python package: {module}\n"
                   f"   Run: pip install {module}")
    
    if "permission denied" in stderr.lower():
        return (f"{stderr}\n\n"
                "💡 Permission denied. Try:\n"
                "   • Check file permissions\n"
                "   • Run with appropriate access rights\n"
                "   • Verify API key has required permissions")
    
    # Return original error if no enhancement needed
    return stderr

# Execute generic Terraform command
async def execute_terraform_command(cmd: List[str], cwd: str) -> Dict[str, Any]:
    """Execute a Terraform command in the specified directory"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode().strip(),
            "stderr": stderr.decode().strip(),
            "returncode": process.returncode
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }

# Execute Terraform for parallel provisioning
async def execute_terraform_parallel(gpu_type: str, count: int, providers: List[str] = None, max_price: float = None) -> Dict[str, Any]:
    """Execute Terraform-based parallel provisioning for optimal efficiency"""
    
    # Create temporary directory for Terraform files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Generate Terraform configuration for parallel provisioning
            terraform_config = generate_terraform_config(gpu_type, count, providers, max_price)
            
            # Write main.tf
            main_tf_path = os.path.join(temp_dir, "main.tf")
            with open(main_tf_path, 'w') as f:
                f.write(terraform_config)
            
            # Write variables.tf
            vars_tf_path = os.path.join(temp_dir, "variables.tf")
            with open(vars_tf_path, 'w') as f:
                f.write(generate_variables_file())
            
            # Initialize Terraform
            init_result = await asyncio.create_subprocess_exec(
                "terraform", "init",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            init_stdout, init_stderr = await init_result.communicate()
            
            if init_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Terraform init failed: {init_stderr.decode()}",
                    "returncode": init_result.returncode
                }
            
            # Plan Terraform (dry run)
            plan_result = await asyncio.create_subprocess_exec(
                "terraform", "plan", "-out=tfplan",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            plan_stdout, plan_stderr = await plan_result.communicate()
            
            if plan_result.returncode != 0:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Terraform plan failed: {plan_stderr.decode()}",
                    "returncode": plan_result.returncode
                }
            
            # Apply Terraform
            apply_result = await asyncio.create_subprocess_exec(
                "terraform", "apply", "-auto-approve", "tfplan",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            apply_stdout, apply_stderr = await apply_result.communicate()
            
            # Get outputs
            output_result = await asyncio.create_subprocess_exec(
                "terraform", "output", "-json",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            output_stdout, output_stderr = await output_result.communicate()
            
            outputs = {}
            if output_result.returncode == 0:
                try:
                    outputs = json.loads(output_stdout.decode())
                except json.JSONDecodeError:
                    pass
            
            return {
                "success": apply_result.returncode == 0,
                "stdout": apply_stdout.decode(),
                "stderr": apply_stderr.decode(),
                "returncode": apply_result.returncode,
                "terraform_outputs": outputs,
                "plan_output": plan_stdout.decode()
            }
            
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Terraform execution failed: {str(e)}",
                "returncode": -1
            }

def generate_k8s_terraform_config(cluster_name: str, gpu_type: str, node_count: int, multi_cloud: bool = False, prefer_spot: bool = True) -> str:
    """Generate Terraform configuration for Kubernetes clusters"""
    
    config = f"""
terraform {{
  required_providers {{
    terradev = {{
      source  = "theoddden/terradev"
      version = "~> 3.0"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }}
  }}
}}

variable "cluster_name" {{
  description = "Kubernetes cluster name"
  type        = string
  default     = "{cluster_name}"
}}

variable "gpu_type" {{
  description = "GPU type for nodes"
  type        = string
  default     = "{gpu_type}"
}}

variable "node_count" {{
  description = "Number of GPU nodes"
  type        = number
  default     = {node_count}
}}

variable "multi_cloud" {{
  description = "Use multi-cloud node pools"
  type        = bool
  default     = {str(multi_cloud).lower()}
}}

variable "prefer_spot" {{
  description = "Prefer spot instances"
  type        = bool
  default     = {str(prefer_spot).lower()}
}}

# Kubernetes cluster with GPU nodes
resource "terradev_kubernetes_cluster" "main" {{
  name        = var.cluster_name
  gpu_type    = var.gpu_type
  node_count  = var.node_count
  spot        = var.prefer_spot
  
  tags = {{
    Name        = var.cluster_name
    Provisioned = "terraform"
    GPU_Type    = var.gpu_type
    MultiCloud  = var.multi_cloud
  }}
}}
"""

    if multi_cloud:
        # Add multi-cloud node pools for enhanced resilience
        providers = ["runpod", "vastai", "lambda", "aws", "alibaba", "ovhcloud", "fluidstack", "hetzner", "siliconflow"]
        for i, provider in enumerate(providers[:node_count]):
            config += (
                "\n# Multi-cloud node pool - " + provider + "\n"
                'resource "terradev_node_pool" "pool_' + str(i) + '" {\n'
                "  cluster_name = terradev_kubernetes_cluster.main.name\n"
                '  provider     = "' + provider + '"\n'
                "  gpu_type     = var.gpu_type\n"
                "  node_count   = 1\n"
                "  spot         = var.prefer_spot\n"
                "\n"
                "  depends_on = [terradev_kubernetes_cluster.main]\n"
                "\n"
                "  tags = {\n"
                '    Name        = "${var.cluster_name}-pool-' + str(i) + '"\n'
                '    Provider    = "' + provider + '"\n'
                '    Provisioned = "terraform"\n'
                "  }\n"
                "}\n"
            )

    # Add outputs
    config += """
# Cluster outputs
output "cluster_name" {
  description = "Kubernetes cluster name"
  value       = terradev_kubernetes_cluster.main.name
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint"
  value       = terradev_kubernetes_cluster.main.endpoint
}

output "kubeconfig" {
  description = "Kubernetes configuration"
  value       = terradev_kubernetes_cluster.main.kubeconfig
  sensitive   = true
}

output "node_pools" {
  description = "Node pool information"
  value = {
"""
    
    if multi_cloud:
        for i in range(min(node_count, len(providers))):
            config += f'    pool_{i} = terradev_node_pool.pool_{i}[*].id,\n'
    
    config += """  }
}
"""
    
    return config

def generate_inference_terraform_config(model: str, gpu_type: str, endpoint_name: str = None) -> str:
    """Generate Terraform configuration for inference deployments"""
    
    endpoint_name = endpoint_name or f"inferx-{model.replace('/', '-').replace(':', '-')}"
    
    config = (
        "terraform {\n"
        "  required_providers {\n"
        "    terradev = {\n"
        '      source  = "theoddden/terradev"\n'
        '      version = "~> 3.0"\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        'variable "model" {\n'
        '  description = "Model ID for deployment"\n'
        "  type        = string\n"
        '  default     = "' + model + '"\n'
        "}\n\n"
        'variable "gpu_type" {\n'
        '  description = "GPU type for inference"\n'
        "  type        = string\n"
        '  default     = "' + gpu_type + '"\n'
        "}\n\n"
        'variable "endpoint_name" {\n'
        '  description = "Inference endpoint name"\n'
        "  type        = string\n"
        '  default     = "' + endpoint_name + '"\n'
        "}\n\n"
    )

    config += """
# InferX serverless endpoint
resource "terradev_inference_endpoint" "main" {
  name        = var.endpoint_name
  model       = var.model
  gpu_type    = var.gpu_type

  tags = {
    Name        = var.endpoint_name
    Model       = var.model
    GPU_Type    = var.gpu_type
    Provisioned = "terraform"
  }
}

# HuggingFace Spaces deployment (optional)
resource "terradev_hf_space" "main" {
  count       = contains(["A10G", "L4", "T4"], var.gpu_type) ? 1 : 0
  name        = var.endpoint_name
  model_id    = var.model
  hardware    = var.gpu_type
  sdk         = "gradio"

  tags = {
    Name        = var.endpoint_name
    Model       = var.model
    Hardware    = var.gpu_type
    Provisioned = "terraform"
  }
}

# Outputs
output "endpoint_url" {
  description = "Inference endpoint URL"
  value       = terradev_inference_endpoint.main.url
}

output "endpoint_status" {
  description = "Endpoint deployment status"
  value       = terradev_inference_endpoint.main.status
}

output "hf_space_url" {
  description = "HuggingFace Space URL"
  value       = length(terradev_hf_space.main) > 0 ? terradev_hf_space.main[0].url : null
}
"""

    return config

def generate_terraform_config(gpu_type: str, count: int, providers: List[str] = None, max_price: float = None) -> str:
    """Generate Terraform configuration for parallel GPU provisioning"""
    
    providers = providers or ["runpod", "vastai", "lambda", "aws", "alibaba", "ovhcloud", "fluidstack", "hetzner", "siliconflow"]
    
    config = f"""
terraform {{
  required_providers {{
    terradev = {{
      source = "theoddden/terradev"
      version = "~> 3.0"
    }}
  }}
}}

variable "gpu_type" {{
  description = "GPU type to provision"
  type        = string
  default     = "{gpu_type}"
}}

variable "gpu_count" {{
  description = "Number of GPUs to provision"
  type        = number
  default     = {count}
}}

variable "max_price" {{
  description = "Maximum price per hour"
  type        = number
  default     = {max_price if max_price else "null"}
}}

# Parallel provisioning across multiple providers
"""
    
    # Add provider blocks for parallel provisioning
    for i, provider in enumerate(providers[:count]):  # Distribute across providers
        config += (
            '\nresource "terradev_instance" "gpu_' + str(i) + '" {\n'
            "  gpu_type    = var.gpu_type\n"
            '  provider    = ' + provider + '\n'
            "  spot        = true\n"
            "  count       = 1\n"
            "\n"
            "  # Dynamic pricing and availability\n"
            '  dynamic "pricing" {\n'
            "    for_each = var.max_price != null ? [1] : []\n"
            "    content {\n"
            "      max_hourly = var.max_price\n"
            "    }\n"
            "  }\n"
            "\n"
            "  tags = {\n"
            '    Name        = "terradev-mcp-gpu-' + str(i) + '"\n'
            '    Provisioned = "terraform"\n'
            "    GPU_Type    = var.gpu_type\n"
            "  }\n"
            "}\n\n"
        )
    
    # Add outputs for instance information
    config += """
# Outputs for instance management
output "instance_ids" {
  description = "Provisioned instance IDs"
  value = [
"""
    
    for i in range(min(count, len(providers))):
        config += f'    terradev_instance.gpu_{i}[*].id,\n'
    
    config += """  ]
}

output "instance_ips" {
  description = "Instance IP addresses"
  value = [
"""
    
    for i in range(min(count, len(providers))):
        config += f'    terradev_instance.gpu_{i}[*].public_ip,\n'
    
    config += """  ]
}

output "provider_costs" {
  description = "Hourly costs by provider"
  value = {
"""
    
    for i, provider in enumerate(providers[:count]):
        config += f'    {provider} = terradev_instance.gpu_{i}[*].hourly_cost,\n'
    
    config += """  }
}
"""
    
    return config

def generate_variables_file() -> str:
    """Generate Terraform variables file"""
    return """
variable "gpu_type" {
  description = "GPU type to provision"
  type        = string
  
  validation {
    condition = contains([
      "H100", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100"
    ], var.gpu_type)
    error_message = "GPU type must be one of: H100, A100, A10G, L40S, L4, T4, RTX4090, RTX3090, V100."
  }
}

variable "gpu_count" {
  description = "Number of GPUs to provision"
  type        = number
  
  validation {
    condition     = var.gpu_count > 0 && var.gpu_count <= 32
    error_message = "GPU count must be between 1 and 32."
  }
}

variable "max_price" {
  description = "Maximum price per hour"
  type        = number
  default     = null
  
  validation {
    condition     = var.max_price == null || var.max_price > 0
    error_message = "Max price must be null or greater than 0."
  }
}
"""

# Create MCP server
server = Server("terradev-mcp")

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available Terradev tools"""
    tools = [
        Tool(
            name="quote_gpu",
            description="Get real-time GPU prices across 20 cloud providers (incl. Alibaba, OVHcloud, FluidStack, Hetzner, SiliconFlow)",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type (H100, H200, H800, A100, A10G, L40S, L4, T4, RTX4090, RTX3090, V100, V100S, A6000)",
                        "enum": ["H100", "H200", "H800", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100", "V100S", "A6000"]
                    },
                    "providers": {
                        "type": "string",
                        "description": "Optional comma-separated list of providers to filter",
                        "pattern": "^[a-z,-]*$"
                    },
                    "quick": {
                        "type": "boolean",
                        "description": "Quick-provision the cheapest option"
                    }
                },
                "required": ["gpu_type"]
            }
        ),
        Tool(
            name="provision_gpu",
            description="Provision GPU instances using Terraform for optimal parallel efficiency",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type to provision",
                        "enum": ["H100", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100"]
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of GPUs to provision",
                        "minimum": 1,
                        "default": 1
                    },
                    "providers": {
                        "type": "array",
                        "description": "Cloud providers for parallel distribution",
                        "items": {
                            "type": "string",
                            "enum": ["runpod", "vastai", "lambda", "aws", "gcp", "azure", "coreweave", "tensordock", "oracle", "crusoe", "digitalocean", "hyperstack", "alibaba", "ovhcloud", "fluidstack", "hetzner", "siliconflow"]
                        }
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price per hour",
                        "minimum": 0
                    },
                    "plan_only": {
                        "type": "boolean",
                        "description": "Generate Terraform plan without applying",
                        "default": False
                    },
                    "state_file": {
                        "type": "string",
                        "description": "Terraform state file path (optional)",
                        "default": None
                    }
                },
                "required": ["gpu_type"]
            }
        ),
        Tool(
            name="terraform_plan",
            description="Generate Terraform execution plan for GPU provisioning",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration"
                    },
                    "var_file": {
                        "type": "string",
                        "description": "Terraform variables file path (optional)"
                    },
                    "destroy": {
                        "type": "boolean",
                        "description": "Generate destroy plan",
                        "default": False
                    }
                },
                "required": ["config_dir"]
            }
        ),
        Tool(
            name="terraform_apply",
            description="Apply Terraform configuration for GPU provisioning",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration"
                    },
                    "plan_file": {
                        "type": "string",
                        "description": "Terraform plan file to apply (optional)"
                    },
                    "var_file": {
                        "type": "string",
                        "description": "Terraform variables file path (optional)"
                    },
                    "auto_approve": {
                        "type": "boolean",
                        "description": "Auto-approve the apply",
                        "default": True
                    }
                },
                "required": ["config_dir"]
            }
        ),
        Tool(
            name="terraform_destroy",
            description="Destroy Terraform-managed GPU infrastructure",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration"
                    },
                    "var_file": {
                        "type": "string",
                        "description": "Terraform variables file path (optional)"
                    },
                    "auto_approve": {
                        "type": "boolean",
                        "description": "Auto-approve the destroy",
                        "default": True
                    }
                },
                "required": ["config_dir"]
            }
        ),
        Tool(
            name="local_scan",
            description="Scan local machine and network for available GPU devices. Returns total VRAM pool for local-first provisioning.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="k8s_create",
            description="Create Kubernetes cluster with GPU nodes using Terraform for optimal multi-cloud deployment",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the cluster"
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type for nodes",
                        "enum": ["H100", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100"]
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of GPU nodes",
                        "minimum": 1,
                        "default": 1
                    },
                    "multi_cloud": {
                        "type": "boolean",
                        "description": "Use multi-cloud node pools for resilience",
                        "default": False
                    },
                    "prefer_spot": {
                        "type": "boolean",
                        "description": "Prefer spot instances for cost savings",
                        "default": True
                    },
                    "use_terraform": {
                        "type": "boolean",
                        "description": "Use Terraform for infrastructure as code",
                        "default": True
                    }
                },
                "required": ["cluster_name", "gpu_type"]
            }
        ),
        Tool(
            name="k8s_list",
            description="List Kubernetes clusters",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="k8s_info",
            description="Get information about a specific cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the cluster"
                    }
                },
                "required": ["cluster_name"]
            }
        ),
        Tool(
            name="k8s_destroy",
            description="Destroy a Kubernetes cluster",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_name": {
                        "type": "string",
                        "description": "Name of the cluster to destroy"
                    }
                },
                "required": ["cluster_name"]
            }
        ),
        Tool(
            name="inferx_deploy",
            description="Deploy model to InferX serverless platform using Terraform for infrastructure as code",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID (e.g., meta-llama/Llama-2-7b-hf)"
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type for inference",
                        "enum": ["H100", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100"]
                    },
                    "endpoint_name": {
                        "type": "string",
                        "description": "Custom endpoint name (optional)"
                    },
                    "use_terraform": {
                        "type": "boolean",
                        "description": "Use Terraform for infrastructure as code",
                        "default": True
                    }
                },
                "required": ["model", "gpu_type"]
            }
        ),
        Tool(
            name="inferx_status",
            description="Check InferX endpoint status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="inferx_list",
            description="List deployed InferX models",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="inferx_optimize",
            description="Get cost analysis for inference endpoints",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="hf_space_deploy",
            description="Deploy model to HuggingFace Spaces",
            inputSchema={
                "type": "object",
                "properties": {
                    "space_name": {
                        "type": "string",
                        "description": "Name of the Space"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID"
                    },
                    "template": {
                        "type": "string",
                        "description": "Template type",
                        "enum": ["llm", "embedding", "image"]
                    },
                    "hardware": {
                        "type": "string",
                        "description": "Hardware specification"
                    },
                    "sdk": {
                        "type": "string",
                        "description": "SDK to use (e.g., gradio, streamlit)"
                    }
                },
                "required": ["space_name", "model_id", "template"]
            }
        ),
        Tool(
            name="status",
            description="View all instances and costs with Terraform state optimization",
            inputSchema={
                "type": "object",
                "properties": {
                    "live": {
                        "type": "boolean",
                        "description": "Show live status vs cached Terraform state",
                        "default": False
                    },
                    "use_terraform_state": {
                        "type": "boolean",
                        "description": "Use Terraform state for faster queries",
                        "default": True
                    },
                    "state_file": {
                        "type": "string",
                        "description": "Path to Terraform state file (optional)"
                    }
                }
            }
        ),
        Tool(
            name="terraform_status",
            description="Fast status query using Terraform state",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_dir": {
                        "type": "string",
                        "description": "Directory containing Terraform configuration"
                    },
                    "show_outputs": {
                        "type": "boolean",
                        "description": "Show Terraform outputs",
                        "default": True
                    }
                },
                "required": ["config_dir"]
            }
        ),
        Tool(
            name="manage_instance",
            description="Manage GPU instances (stop/start/terminate)",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": "Instance ID"
                    },
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": ["stop", "start", "terminate"]
                    }
                },
                "required": ["instance_id", "action"]
            }
        ),
        Tool(
            name="analytics",
            description="Get cost analytics",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze",
                        "minimum": 1,
                        "default": 30
                    }
                }
            }
        ),
        Tool(
            name="optimize",
            description="Find cheaper alternatives for running instances",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="setup_provider",
            description="Get setup instructions for a provider",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name",
                        "enum": ["runpod", "aws", "vastai", "gcp", "azure", "lambda", "coreweave", "tensordock", "oracle", "crusoe", "digitalocean", "hyperstack", "alibaba", "ovhcloud", "fluidstack", "hetzner", "siliconflow"]
                    },
                    "quick": {
                        "type": "boolean",
                        "description": "Quick setup instructions",
                        "default": True
                    }
                },
                "required": ["provider"]
            }
        ),
        Tool(
            name="configure_provider",
            description="Configure provider credentials",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider to configure",
                        "enum": ["runpod", "aws", "vastai", "gcp", "azure", "lambda", "coreweave", "tensordock", "oracle", "crusoe", "digitalocean", "hyperstack", "alibaba", "ovhcloud", "fluidstack", "hetzner", "siliconflow"]
                    }
                },
                "required": ["provider"]
            }
        ),
        # ── v3.2.0 Tools: Semantic Routing, Disaggregated Inference, GPU Topology, Price Intelligence ──
        Tool(
            name="infer_route",
            description="Semantic-aware inference routing. Analyzes query content across 6 signal dimensions (modality, complexity, domain, language, safety, keywords), applies NUMA-aware endpoint scoring, and selects the optimal inference endpoint. Uses Terraform-style DAG parallel execution for signal extraction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Filter by model name (e.g., llama-3-70b, deepseek-coder-33b)"
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Routing strategy",
                        "enum": ["latency", "cost", "score"],
                        "default": "latency"
                    },
                    "measure": {
                        "type": "boolean",
                        "description": "Run fresh latency measurements (HTTP TTFB / WebPageTest) before routing",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="infer_route_disagg",
            description="Disaggregated Prefill/Decode routing (DistServe architecture). Splits LLM inference into compute-bound prefill phase (routed to FLOPS-optimized GPUs like H100 SXM) and memory-bound decode phase (routed to bandwidth-optimized GPUs like MI300X). Tracks KV cache handoffs between endpoint pairs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name for disaggregated pair selection (e.g., llama-3-70b)"
                    },
                    "check_health": {
                        "type": "boolean",
                        "description": "Run health probes before selecting pairs",
                        "default": True
                    }
                },
                "required": ["model"]
            }
        ),
        Tool(
            name="infer_status",
            description="Show inference endpoint health, latency, failover status, and disaggregated phase assignments (PREFILL/DECODE/MIXED). Includes KV cache warm status per endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "check": {
                        "type": "boolean",
                        "description": "Run live health probes before showing status",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="infer_failover",
            description="Run health checks and auto-failover for inference endpoints. If a primary endpoint is unhealthy and has a backup configured, traffic automatically shifts to the backup provider.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show what would happen without executing failover",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="gpu_topology",
            description="GPU NUMA topology report with intra-GPU XCD (Accelerated Compute Die) awareness. Models MI300X (8 XCDs, 192GB HBM3), MI300A (6 XCDs, 128GB), H200 (unified 141GB HBM3e), H100 (80GB). Reports PCIe locality (PIX/PXB/PHB/SYS), GPU-NIC pairing, SR-IOV VF status, and generates XCD-aware NCCL/AITER environment variables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_arch": {
                        "type": "string",
                        "description": "Filter by GPU architecture",
                        "enum": ["mi300x", "mi300a", "h200", "h100", "a100", "auto"]
                    },
                    "generate_env": {
                        "type": "boolean",
                        "description": "Generate XCD-aware NCCL/AITER/CK env vars for optimal attention kernel performance",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="price_intel",
            description="GPU price intelligence with quantitative analytics. Computes delta (rate of change), gamma (acceleration), and annualized realized volatility on GPU spot/on-demand prices across all 15 providers. Identifies cheapest time windows and provider arbitrage opportunities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type to analyze",
                        "enum": ["H100", "H200", "H800", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100", "V100S", "A6000", "MI300X"]
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history to analyze",
                        "minimum": 1,
                        "default": 7
                    },
                    "provider": {
                        "type": "string",
                        "description": "Filter to specific provider (optional)"
                    }
                },
                "required": ["gpu_type"]
            }
        ),
        Tool(
            name="moe_deploy",
            description="Deploy Mixture-of-Experts models with production-ready cluster templates. Auto-applies vLLM cost optimizations (KV cache offloading for up to 9x throughput, MTP speculative decoding for up to 2.8x speed, sleep mode for 18-200x faster restarts). Supports GLM-5, Qwen 3.5, Mistral Large 3, DeepSeek V4, Llama 5. Configures NVLink topology, tensor parallelism, FP8 quantization, vLLM/SGLang backends, and GPU-aware HPA autoscaling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "HuggingFace model ID (e.g., zai-org/GLM-5-FP8, Qwen/Qwen3.5-397B-A17B)"
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type for MoE serving",
                        "enum": ["H100", "H200", "H800", "A100", "MI300X"]
                    },
                    "tp_size": {
                        "type": "integer",
                        "description": "Tensor parallelism size (must match NVLink domain)",
                        "enum": [1, 2, 4, 8],
                        "default": 8
                    },
                    "backend": {
                        "type": "string",
                        "description": "Serving backend",
                        "enum": ["vllm", "sglang"],
                        "default": "vllm"
                    },
                    "quantization": {
                        "type": "string",
                        "description": "Quantization method",
                        "enum": ["fp8", "bf16", "awq", "gptq"],
                        "default": "fp8"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show deployment plan without executing",
                        "default": False
                    }
                },
                "required": ["model_id", "gpu_type"]
            }
        ),
        Tool(
            name="gitops_init",
            description="Initialize GitOps repository with ArgoCD or Flux CD. Creates cluster manifests, app definitions, policy-as-code templates, and multi-environment structure. Supports GitHub, GitLab, Bitbucket, Azure DevOps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Git repository (e.g., my-org/infra)"
                    },
                    "tool": {
                        "type": "string",
                        "description": "GitOps tool to use",
                        "enum": ["argocd", "flux"],
                        "default": "argocd"
                    },
                    "provider": {
                        "type": "string",
                        "description": "Git provider",
                        "enum": ["github", "gitlab", "bitbucket", "azure-devops"],
                        "default": "github"
                    },
                    "cluster": {
                        "type": "string",
                        "description": "Target cluster name"
                    }
                },
                "required": ["repo"]
            }
        ),
        # ── v3.4.0 Tools: Training Pipeline, Checkpoints, Monitoring ──
        Tool(
            name="train",
            description="Launch distributed training on provisioned GPU nodes. Supports torchrun, deepspeed, accelerate, and megatron. Use from_provision='latest' to auto-resolve node IPs from your last provision command.",
            inputSchema={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "Training script path (e.g., train.py)"},
                    "framework": {"type": "string", "description": "Training framework", "enum": ["torchrun", "deepspeed", "accelerate", "megatron"], "default": "torchrun"},
                    "from_provision": {"type": "string", "description": "Resolve nodes from provision. Use 'latest' or a parallel_group_id."},
                    "nodes": {"type": "array", "description": "Manual node IPs", "items": {"type": "string"}},
                    "gpus_per_node": {"type": "integer", "description": "GPUs per node", "default": 8},
                    "script_args": {"type": "string", "description": "Extra args for training script"}
                },
                "required": ["script"]
            }
        ),
        Tool(
            name="train_status",
            description="List all training jobs and their state (created, running, completed, failed).",
            inputSchema={"type": "object", "properties": {"job_id": {"type": "string", "description": "Filter to a specific job ID"}}}
        ),
        Tool(
            name="train_monitor",
            description="Real-time GPU monitoring for training jobs. Shows utilization, memory, temperature, power, and cost.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to monitor"},
                    "cost_rate": {"type": "number", "description": "$/GPU-hr for cost estimation", "default": 2.0}
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="checkpoint_list",
            description="List all checkpoints for a training job.",
            inputSchema={"type": "object", "properties": {"job_id": {"type": "string", "description": "Job ID"}}, "required": ["job_id"]}
        ),
        Tool(
            name="checkpoint_save",
            description="Manually trigger a checkpoint save for a running training job.",
            inputSchema={"type": "object", "properties": {"job_id": {"type": "string", "description": "Job ID"}, "step": {"type": "integer", "description": "Step number"}}, "required": ["job_id"]}
        ),
        Tool(
            name="preflight",
            description="Pre-training validation: GPU availability, NCCL, RDMA, drivers across all nodes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "description": "Node IPs to validate", "items": {"type": "string"}},
                    "from_provision": {"type": "string", "description": "Resolve nodes from provision. 'latest' or a group ID."}
                }
            }
        ),
        Tool(
            name="price_discovery",
            description="Enhanced price discovery with capacity scoring, confidence intervals, and trend analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {"type": "string", "description": "GPU type", "enum": ["H100", "H200", "H800", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "V100S", "A6000", "MI300X"]},
                    "region": {"type": "string", "description": "Filter by region"},
                    "hours": {"type": "integer", "description": "Hours of history", "default": 24}
                },
                "required": ["gpu_type"]
            }
        ),
        # ── v2.0.0 Tools: Complete Agentic Loop ─────────────────────────────
        # Training pipeline completion
        Tool(
            name="train_stop",
            description="Stop a running training job. Kills training processes on all nodes in parallel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to stop"}
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="train_resume",
            description="Resume a training job from its latest checkpoint. Rebuilds config with topology revalidation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to resume"},
                    "checkpoint_id": {"type": "string", "description": "Specific checkpoint to resume from (default: latest)"}
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="checkpoint_restore",
            description="Restore a specific checkpoint for a training job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "step": {"type": "integer", "description": "Checkpoint step to restore"},
                    "checkpoint_id": {"type": "string", "description": "Checkpoint ID to restore"}
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="checkpoint_promote",
            description="Promote a checkpoint to a final model path for serving.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "checkpoint_id": {"type": "string", "description": "Checkpoint ID to promote"},
                    "dest": {"type": "string", "description": "Destination path (e.g., /models/final)"}
                },
                "required": ["job_id", "checkpoint_id", "dest"]
            }
        ),
        Tool(
            name="checkpoint_delete",
            description="Delete a checkpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID"},
                    "checkpoint_id": {"type": "string", "description": "Checkpoint ID to delete"}
                },
                "required": ["job_id", "checkpoint_id"]
            }
        ),
        # Data staging
        Tool(
            name="stage",
            description="Compress, chunk, checksum, and position datasets near compute. Supports local paths, S3/GCS URIs, HTTP URLs, and HuggingFace dataset names. Returns staging plan with agent recommendations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "description": "Dataset path, S3 URI, GCS URI, HTTP URL, or HuggingFace name"},
                    "target_regions": {"type": "string", "description": "Comma-separated target regions"},
                    "compression": {"type": "string", "description": "Compression algorithm", "enum": ["auto", "zstd", "gzip", "none"], "default": "auto"},
                    "plan_only": {"type": "boolean", "description": "Show staging plan without executing", "default": False}
                },
                "required": ["dataset"]
            }
        ),
        # Inference deployment
        Tool(
            name="infer_deploy",
            description="Deploy an inference endpoint with auto-scaling, idle shutdown, and cost optimization. Returns estimated cost and requires_confirmation for expensive deployments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Model path or HuggingFace model ID"},
                    "name": {"type": "string", "description": "Endpoint name"},
                    "provider": {"type": "string", "description": "Provider for deployment", "enum": ["runpod", "vastai", "lambda", "aws", "gcp", "coreweave", "alibaba", "ovhcloud", "fluidstack", "hetzner", "siliconflow"]},
                    "gpu_type": {"type": "string", "description": "GPU type", "enum": ["H100", "H200", "H800", "A100", "A10G", "L40S", "L4", "T4", "V100S", "A6000", "MI300X"]},
                    "min_workers": {"type": "integer", "description": "Minimum workers", "default": 0},
                    "max_workers": {"type": "integer", "description": "Maximum workers", "default": 3},
                    "idle_timeout": {"type": "integer", "description": "Idle timeout in seconds", "default": 300},
                    "cost_optimize": {"type": "boolean", "description": "Enable cost optimization", "default": True},
                    "dry_run": {"type": "boolean", "description": "Show deployment plan without deploying", "default": False}
                },
                "required": ["model_path", "name"]
            }
        ),
        # Manifest cache & drift detection
        Tool(
            name="up",
            description="CLI-native provisioning with manifest cache and drift detection. Use --fix-drift to detect and auto-fix drifted infrastructure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job": {"type": "string", "description": "Job name for manifest tracking"},
                    "gpu_type": {"type": "string", "description": "GPU type", "default": "A100"},
                    "gpu_count": {"type": "integer", "description": "Number of GPUs", "default": 1},
                    "hours": {"type": "number", "description": "Estimated runtime in hours", "default": 1.0},
                    "budget": {"type": "number", "description": "Budget constraint ($/hr)"},
                    "region": {"type": "string", "description": "Preferred region"},
                    "ttl": {"type": "string", "description": "Time to live for nodes", "default": "1h"},
                    "fix_drift": {"type": "boolean", "description": "Detect and fix drift automatically", "default": False}
                },
                "required": ["job"]
            }
        ),
        Tool(
            name="rollback",
            description="Explicit versioned rollback. Format: job@version (e.g., llama3@v3).",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_version": {"type": "string", "description": "Job@version string (e.g., llama3@v3)"}
                },
                "required": ["job_version"]
            }
        ),
        Tool(
            name="manifests",
            description="List cached manifests and versions for jobs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job": {"type": "string", "description": "Show versions for specific job (optional)"}
                }
            }
        ),
        # Smart deployment
        Tool(
            name="smart_deploy",
            description="AI-ranked deployment options with confidence/risk scoring. Returns recommendations with cost estimates and requires_confirmation for execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Container image"},
                    "workload": {"type": "string", "description": "Workload type", "enum": ["training", "inference", "batch"]},
                    "gpu_type": {"type": "string", "description": "GPU type"},
                    "budget": {"type": "number", "description": "Max $/hr budget"},
                    "option": {"type": "integer", "description": "Execute a specific recommended option by index"}
                },
                "required": ["image", "workload"]
            }
        ),
        Tool(
            name="helm_generate",
            description="Generate Helm charts from workload specifications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workload": {"type": "string", "description": "Workload type", "enum": ["training", "inference", "batch"]},
                    "image": {"type": "string", "description": "Container image"},
                    "gpu_type": {"type": "string", "description": "GPU type"},
                    "replicas": {"type": "integer", "description": "Number of replicas", "default": 1}
                },
                "required": ["workload", "image"]
            }
        ),
        # GitOps complete
        Tool(
            name="gitops_bootstrap",
            description="Bootstrap ArgoCD or Flux on the cluster.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "GitOps tool", "enum": ["argocd", "flux"]},
                    "cluster": {"type": "string", "description": "Cluster name"},
                    "namespace": {"type": "string", "description": "Namespace", "default": "gitops-system"}
                },
                "required": ["tool", "cluster"]
            }
        ),
        Tool(
            name="gitops_sync",
            description="Sync cluster with Git repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {"type": "string", "description": "Cluster name"},
                    "environment": {"type": "string", "description": "Environment to sync", "default": "prod"},
                    "tool": {"type": "string", "description": "GitOps tool", "enum": ["argocd", "flux"], "default": "argocd"}
                },
                "required": ["cluster"]
            }
        ),
        Tool(
            name="gitops_validate",
            description="Validate GitOps configuration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster": {"type": "string", "description": "Cluster name"},
                    "dry_run": {"type": "boolean", "description": "Dry run validation", "default": True}
                }
            }
        ),
        # Orchestrator tools
        Tool(
            name="orchestrator_start",
            description="Start the model orchestrator for multi-model GPU sharing with eviction policies.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_id": {"type": "integer", "description": "GPU ID", "default": 0},
                    "memory_gb": {"type": "number", "description": "Total GPU memory in GB", "default": 80.0},
                    "policy": {"type": "string", "description": "Scaling policy", "enum": ["billing_optimized", "latency_optimized", "hybrid"], "default": "billing_optimized"}
                }
            }
        ),
        Tool(
            name="orchestrator_register",
            description="Register a model with the orchestrator.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model identifier"},
                    "model_path": {"type": "string", "description": "Path to model weights"},
                    "framework": {"type": "string", "description": "Framework", "enum": ["pytorch", "vllm", "sglang"], "default": "pytorch"}
                },
                "required": ["model_id", "model_path"]
            }
        ),
        Tool(
            name="orchestrator_load",
            description="Load a model into GPU memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model to load"},
                    "force": {"type": "boolean", "description": "Force loading even if memory is full", "default": False}
                },
                "required": ["model_id"]
            }
        ),
        Tool(
            name="orchestrator_evict",
            description="Evict a model from GPU memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model to evict"}
                },
                "required": ["model_id"]
            }
        ),
        Tool(
            name="orchestrator_status",
            description="Get orchestrator and model status including GPU memory utilization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Get details for specific model (optional)"}
                }
            }
        ),
        Tool(
            name="orchestrator_infer",
            description="Test inference with a model via the orchestrator.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model to run inference on"}
                },
                "required": ["model_id"]
            }
        ),
        # Warm pool tools
        Tool(
            name="warm_pool_start",
            description="Start the warm pool manager for intelligent model pre-warming. 5 strategies: traffic_based, time_based, priority_based, cost_optimized, latency_optimized.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "description": "Warm pool strategy", "enum": ["traffic_based", "time_based", "priority_based", "cost_optimized", "latency_optimized"], "default": "traffic_based"},
                    "max_warm": {"type": "integer", "description": "Max models to keep warm", "default": 10},
                    "min_warm": {"type": "integer", "description": "Min models to keep warm", "default": 3}
                }
            }
        ),
        Tool(
            name="warm_pool_status",
            description="Get warm pool status: hit rate, cold starts, memory saved, cost saved.",
            inputSchema={"type": "object", "properties": {}}
        ),
        # Cost scaler tools
        Tool(
            name="cost_scaler_start",
            description="Start the budget-aware auto-scaling manager. 4 strategies: minimize_cost, balance_cost_latency, latency_critical, budget_constrained.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "description": "Cost strategy", "enum": ["minimize_cost", "balance_cost_latency", "latency_critical", "budget_constrained"], "default": "balance_cost_latency"},
                    "budget": {"type": "number", "description": "Hourly budget in USD", "default": 15.0},
                    "cost_per_gb": {"type": "number", "description": "Cost per GB per hour", "default": 0.10}
                }
            }
        ),
        Tool(
            name="cost_scaler_status",
            description="Get cost scaler status with budget utilization, predictions, and optimization recommendations.",
            inputSchema={"type": "object", "properties": {}}
        ),
        # InferX complete
        Tool(
            name="inferx_configure",
            description="Configure InferX serverless platform credentials.",
            inputSchema={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "description": "InferX API key"},
                    "endpoint": {"type": "string", "description": "InferX API endpoint", "default": "https://api.inferx.net"},
                    "region": {"type": "string", "description": "Region", "default": "us-west-2"}
                },
                "required": ["api_key"]
            }
        ),
        Tool(
            name="inferx_delete",
            description="Delete an InferX model deployment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Model deployment ID to delete"}
                },
                "required": ["model_id"]
            }
        ),
        Tool(
            name="inferx_usage",
            description="Get InferX account usage statistics: requests, cost, GPU hours, latency.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="inferx_quote",
            description="Get InferX pricing quotes for a GPU type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {"type": "string", "description": "GPU type to quote", "default": "A100"},
                    "region": {"type": "string", "description": "Region for quote"}
                }
            }
        ),
        # HF Space deploy (already exists above)
        Tool(
            name="hf_space_status",
            description="Get HuggingFace Space deployment status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "space_name": {"type": "string", "description": "Space name to check"}
                },
                "required": ["space_name"]
            }
        ),
        # Workflow primitives
        Tool(
            name="run_workflow",
            description="Run a declarative YAML workflow that chains multiple Terradev commands (provision → preflight → train → monitor → checkpoint). Returns step-by-step execution status with cost estimates and confirmation gates for expensive operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow": {"type": "string", "description": "Workflow YAML path or inline YAML string"},
                    "dry_run": {"type": "boolean", "description": "Show execution plan without running", "default": False},
                    "template": {"type": "string", "description": "Use built-in template", "enum": ["finetune-llama", "inference-deploy", "benchmark-gpu", "cost-optimize"]}
                }
            }
        ),
        # Active context — session-start awareness
        Tool(
            name="active_context",
            description="Get current Terradev state: running training jobs, active instances, spend-to-date, alerts. Call this on session start to resume context from previous sessions.",
            inputSchema={"type": "object", "properties": {}}
        ),
        # ── v3.5.0: Multi-LoRA adapter management ────────────────────────
        Tool(
            name="lora_list",
            description="List LoRA adapters loaded on a running vLLM endpoint. Shows base models and hot-loaded fine-tuned adapters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)"},
                    "api_key": {"type": "string", "description": "vLLM API key (if set)"}
                },
                "required": ["endpoint"]
            }
        ),
        Tool(
            name="lora_add",
            description="Hot-load a LoRA adapter onto a running vLLM endpoint. The adapter becomes immediately available as a model name for inference requests. Uses vLLM's fused_moe_lora kernel for 454% higher output tokens/sec on MoE models.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)"},
                    "name": {"type": "string", "description": "Adapter name (becomes the model name in API requests)"},
                    "path": {"type": "string", "description": "Path to adapter weights (local path or HuggingFace ID)"},
                    "api_key": {"type": "string", "description": "vLLM API key (if set)"}
                },
                "required": ["endpoint", "name", "path"]
            }
        ),
        Tool(
            name="lora_remove",
            description="Hot-unload a LoRA adapter from a running vLLM endpoint. Frees GPU memory for other adapters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "vLLM endpoint URL (e.g. http://10.0.0.1:8000)"},
                    "name": {"type": "string", "description": "Adapter name to unload"},
                    "api_key": {"type": "string", "description": "vLLM API key (if set)"}
                },
                "required": ["endpoint", "name"]
            }
        ),
    ]
    
    return ListToolsResult(tools=tools)

@server.call_tool()
async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls"""
    tool_name = request.params.name
    arguments = request.params.arguments or {}
    
    # Map tool names to terradev commands
    command_map = {
        "quote_gpu": ["quote"],
        "provision_gpu": ["terraform"],  # Now uses Terraform by default
        "terraform_plan": ["terraform", "plan"],
        "terraform_apply": ["terraform", "apply"],
        "terraform_destroy": ["terraform", "destroy"],
        "terraform_status": ["terraform", "status"],  # Fast state queries
        "k8s_create": ["k8s", "create"],
        "k8s_list": ["k8s", "list"],
        "k8s_info": ["k8s", "info"],
        "k8s_destroy": ["k8s", "destroy"],
        "inferx_deploy": ["inferx", "deploy"],
        "inferx_status": ["inferx", "status"],
        "inferx_list": ["inferx", "list"],
        "inferx_optimize": ["inferx", "optimize"],
        "hf_space_deploy": ["hf-space"],
        "status": ["status"],
        "manage_instance": ["manage"],
        "analytics": ["analytics"],
        "optimize": ["optimize"],
        "setup_provider": ["setup"],
        "configure_provider": ["configure"],
        # v3.4.0 tools
        "train": ["train"],
        "train_status": ["train-status"],
        "train_monitor": ["monitor"],
        "checkpoint_list": ["checkpoint", "list"],
        "checkpoint_save": ["checkpoint", "save"],
        "preflight": ["preflight"],
        "price_discovery": ["price-discovery"],
        # v3.2.0 tools
        "infer_route": ["inference", "route"],
        "infer_route_disagg": ["inference", "route"],
        "infer_status": ["inference", "status"],
        "infer_failover": ["inference", "failover"],
        "gpu_topology": ["inference", "topology"],
        "price_intel": ["analytics"],
        "moe_deploy": ["provision"],
        "gitops_init": ["gitops", "init"],
        # v2.0.0 tools — complete agentic loop
        "train_stop": ["train-stop"],
        "train_resume": ["train-resume"],
        "checkpoint_restore": ["checkpoint", "restore"],
        "checkpoint_promote": ["checkpoint", "promote"],
        "checkpoint_delete": ["checkpoint", "delete"],
        "stage": ["stage"],
        "infer_deploy": ["infer-deploy"],
        "up": ["up"],
        "rollback": ["rollback"],
        "manifests": ["manifests"],
        "smart_deploy": ["smart-deploy"],
        "helm_generate": ["helm-generate"],
        "gitops_bootstrap": ["gitops", "bootstrap"],
        "gitops_sync": ["gitops", "sync"],
        "gitops_validate": ["gitops", "validate"],
        "orchestrator_start": ["orchestrator-start"],
        "orchestrator_register": ["orchestrator-register"],
        "orchestrator_load": ["orchestrator-load"],
        "orchestrator_evict": ["orchestrator-evict"],
        "orchestrator_status": ["orchestrator-status"],
        "orchestrator_infer": ["orchestrator-infer"],
        "warm_pool_start": ["warm-pool-start"],
        "warm_pool_status": ["warm-pool-status"],
        "cost_scaler_start": ["cost-scaler-start"],
        "cost_scaler_status": ["cost-scaler-status"],
        "inferx_configure": ["inferx", "configure"],
        "inferx_delete": ["inferx", "delete"],
        "inferx_usage": ["inferx", "usage"],
        "inferx_quote": ["inferx", "quote"],
        "hf_space_status": ["hf-space"],
        "run_workflow": ["workflow", "run"],
        "active_context": ["status"],  # Composite — handled specially
        # v3.5.0: Multi-LoRA
        "lora_list": ["lora", "list"],
        "lora_add": ["lora", "add"],
        "lora_remove": ["lora", "remove"],
    }
    
    if tool_name not in command_map:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
            isError=True
        )
    
    # Build command arguments
    cmd_args = command_map[tool_name].copy()
    
    # Handle local_scan tool separately (doesn't use terradev CLI)
    if tool_name == "local_scan":
        local_info = await discover_local_gpus()
        
        output_text = "🔍 **Local GPU Scan Results**\n\n"
        
        if local_info['has_local_gpu']:
            output_text += f"✅ **Found {local_info['device_count']} local GPU(s)**\n"
            output_text += f"📊 **Total VRAM Pool:** {local_info['total_vram_gb']} GB\n\n"
            
            output_text += "**Devices:**\n"
            for device in local_info['local_devices']:
                output_text += f"\n• **{device['name']}**\n"
                output_text += f"  - Type: {device['type'].upper()}\n"
                output_text += f"  - VRAM: {device['vram_gb']} GB\n"
                if 'compute_capability' in device:
                    output_text += f"  - Compute: {device['compute_capability']}\n"
                if 'platform' in device:
                    output_text += f"  - Platform: {device['platform']}\n"
            
            output_text += "\n\n💡 **Usage:**\n"
            output_text += "• Use `provision_gpu` with `--local-first` to prefer local GPUs\n"
            output_text += "• Cloud overflow will be used if local pool is insufficient\n"
        else:
            output_text += "❌ **No local GPUs detected**\n\n"
            output_text += "💡 **Tip:** Install PyTorch or nvidia-smi for GPU detection\n"
            output_text += "   - CUDA: `pip install torch`\n"
            output_text += "   - Apple Silicon: PyTorch with MPS support\n"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    
    if tool_name == "quote_gpu":
        cmd_args.extend(["-g", arguments["gpu_type"]])
        if "providers" in arguments:
            cmd_args.extend(["-p", arguments["providers"]])
        if arguments.get("quick"):
            cmd_args.append("--quick")
    
    if tool_name == "provision_gpu":
        # Use Terraform for all GPU provisioning (core IP)
        gpu_type = arguments["gpu_type"]
        count = arguments.get("count", 1)
        providers = arguments.get("providers", ["runpod", "vastai", "lambda", "aws"])
        max_price = arguments.get("max_price")
        plan_only = arguments.get("plan_only", False)
        
        result = await execute_terraform_parallel(gpu_type, count, providers, max_price)
        
        if result["success"]:
            output_text = f"✅ GPU provisioning via Terraform successful!\n\n"
            output_text += f"**GPU Type:** {gpu_type}\n"
            output_text += f"**Count:** {count}\n"
            output_text += f"**Providers:** {', '.join(providers)}\n"
            
            if result.get("terraform_outputs"):
                outputs = result["terraform_outputs"]
                if "instance_ids" in outputs:
                    output_text += f"\n**Instance IDs:** {outputs['instance_ids']}\n"
                if "instance_ips" in outputs:
                    output_text += f"**Instance IPs:** {outputs['instance_ips']}\n"
                if "provider_costs" in outputs:
                    output_text += f"**Provider Costs:** {outputs['provider_costs']}\n"
            
            output_text += f"\n**Terraform State:** Managed\n"
            output_text += f"**Full Output:**\n{result['stdout']}"
            
            return CallToolResult(
                content=[TextContent(type="text", text=output_text)]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Terraform provisioning failed: {result['stderr']}")],
                isError=True
            )
    
    elif tool_name == "terraform_plan":
        config_dir = arguments["config_dir"]
        var_file = arguments.get("var_file")
        destroy = arguments.get("destroy", False)
        
        cmd = ["terraform", "plan"]
        if destroy:
            cmd.append("-destroy")
        if var_file:
            cmd.extend(["-var-file", var_file])
        cmd.append("-out=tfplan")
        
        result = await execute_terraform_command(cmd, config_dir)
        
        if result["success"]:
            return CallToolResult(
                content=[TextContent(type="text", text=f"✅ Terraform plan generated:\n\n{result['stdout']}")]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Terraform plan failed: {result['stderr']}")],
                isError=True
            )
    
    elif tool_name == "terraform_apply":
        config_dir = arguments["config_dir"]
        plan_file = arguments.get("plan_file", "tfplan")
        var_file = arguments.get("var_file")
        auto_approve = arguments.get("auto_approve", True)
        
        cmd = ["terraform", "apply"]
        if auto_approve:
            cmd.append("-auto-approve")
        if plan_file:
            cmd.append(plan_file)
        if var_file:
            cmd.extend(["-var-file", var_file])
        
        result = await execute_terraform_command(cmd, config_dir)
        
        if result["success"]:
            return CallToolResult(
                content=[TextContent(type="text", text=f"✅ Terraform apply successful:\n\n{result['stdout']}")]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Terraform apply failed: {result['stderr']}")],
                isError=True
            )
    
    elif tool_name == "terraform_destroy":
        config_dir = arguments["config_dir"]
        var_file = arguments.get("var_file")
        auto_approve = arguments.get("auto_approve", True)
        
        cmd = ["terraform", "destroy"]
        if auto_approve:
            cmd.append("-auto-approve")
        if var_file:
            cmd.extend(["-var-file", var_file])
        
        result = await execute_terraform_command(cmd, config_dir)
        
        if result["success"]:
            return CallToolResult(
                content=[TextContent(type="text", text=f"✅ Terraform destroy successful:\n\n{result['stdout']}")]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Terraform destroy failed: {result['stderr']}")],
                isError=True
            )
    
    elif tool_name == "k8s_create":
        cluster_name = arguments["cluster_name"]
        gpu_type = arguments["gpu_type"]
        node_count = arguments.get("count", 1)
        multi_cloud = arguments.get("multi_cloud", False)
        prefer_spot = arguments.get("prefer_spot", True)
        use_terraform = arguments.get("use_terraform", True)
        
        if use_terraform:
            # Use Terraform for optimal multi-cloud K8s deployment
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Generate K8s Terraform configuration
                    k8s_config = generate_k8s_terraform_config(
                        cluster_name, gpu_type, node_count, multi_cloud, prefer_spot
                    )
                    
                    # Write configuration files
                    main_tf_path = os.path.join(temp_dir, "main.tf")
                    with open(main_tf_path, 'w') as f:
                        f.write(k8s_config)
                    
                    # Initialize and apply Terraform
                    init_result = await execute_terraform_command(["terraform", "init"], temp_dir)
                    if not init_result["success"]:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"❌ Terraform init failed: {init_result['stderr']}")],
                            isError=True
                        )
                    
                    apply_result = await execute_terraform_command(["terraform", "apply", "-auto-approve"], temp_dir)
                    
                    if apply_result["success"]:
                        output_text = f"✅ Kubernetes cluster created via Terraform!\n\n"
                        output_text += f"**Cluster Name:** {cluster_name}\n"
                        output_text += f"**GPU Type:** {gpu_type}\n"
                        output_text += f"**Node Count:** {node_count}\n"
                        output_text += f"**Multi-Cloud:** {multi_cloud}\n"
                        output_text += f"**Spot Instances:** {prefer_spot}\n"
                        output_text += f"\n**Terraform State:** Managed\n"
                        output_text += f"**Full Output:**\n{apply_result['stdout']}"
                        
                        return CallToolResult(
                            content=[TextContent(type="text", text=output_text)]
                        )
                    else:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"❌ Terraform apply failed: {apply_result['stderr']}")],
                            isError=True
                        )
                        
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ K8s Terraform deployment failed: {str(e)}")],
                        isError=True
                    )
        else:
            # Fall back to regular terradev command
            cmd_args.extend([cluster_name])
            cmd_args.extend(["--gpu", gpu_type])
            if "count" in arguments:
                cmd_args.extend(["--count", str(arguments["count"])])
            if multi_cloud:
                cmd_args.append("--multi-cloud")
            if prefer_spot:
                cmd_args.append("--prefer-spot")
    
    elif tool_name == "k8s_info":
        cmd_args.append(arguments["cluster_name"])
    
    elif tool_name == "k8s_destroy":
        cmd_args.append(arguments["cluster_name"])
    
    elif tool_name == "inferx_deploy":
        model = arguments["model"]
        gpu_type = arguments["gpu_type"]
        endpoint_name = arguments.get("endpoint_name")
        use_terraform = arguments.get("use_terraform", True)
        
        if use_terraform:
            # Use Terraform for inference deployment
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Generate inference Terraform configuration
                    inference_config = generate_inference_terraform_config(model, gpu_type, endpoint_name)
                    
                    # Write configuration files
                    main_tf_path = os.path.join(temp_dir, "main.tf")
                    with open(main_tf_path, 'w') as f:
                        f.write(inference_config)
                    
                    # Initialize and apply Terraform
                    init_result = await execute_terraform_command(["terraform", "init"], temp_dir)
                    if not init_result["success"]:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"❌ Terraform init failed: {init_result['stderr']}")],
                            isError=True
                        )
                    
                    apply_result = await execute_terraform_command(["terraform", "apply", "-auto-approve"], temp_dir)
                    
                    if apply_result["success"]:
                        output_text = f"✅ Inference endpoint deployed via Terraform!\n\n"
                        output_text += f"**Model:** {model}\n"
                        output_text += f"**GPU Type:** {gpu_type}\n"
                        output_text += f"**Endpoint Name:** {endpoint_name or 'auto-generated'}\n"
                        output_text += f"\n**Terraform State:** Managed\n"
                        output_text += f"**Full Output:**\n{apply_result['stdout']}"
                        
                        return CallToolResult(
                            content=[TextContent(type="text", text=output_text)]
                        )
                    else:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"❌ Terraform apply failed: {apply_result['stderr']}")],
                            isError=True
                        )
                        
                except Exception as e:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"❌ Inference Terraform deployment failed: {str(e)}")],
                        isError=True
                    )
        else:
            # Fall back to regular terradev command
            cmd_args.extend(["--model", model])
            cmd_args.extend(["--gpu-type", gpu_type])
    
    elif tool_name == "hf_space_deploy":
        cmd_args.append(arguments["space_name"])
        cmd_args.extend(["--model-id", arguments["model_id"]])
        cmd_args.extend(["--template", arguments["template"]])
        if "hardware" in arguments:
            cmd_args.extend(["--hardware", arguments["hardware"]])
        if "sdk" in arguments:
            cmd_args.extend(["--sdk", arguments["sdk"]])
    
    elif tool_name == "terraform_status":
        config_dir = arguments["config_dir"]
        show_outputs = arguments.get("show_outputs", True)
        
        # Fast status query using Terraform state
        output_result = await execute_terraform_command(["terraform", "output", "-json"], config_dir)
        
        if output_result["success"] and show_outputs:
            try:
                outputs = json.loads(output_result["stdout"])
                output_text = f"✅ Terraform Status (from state):\n\n"
                
                for key, value in outputs.items():
                    if isinstance(value, dict) and "value" in value:
                        output_text += f"**{key}:** {value['value']}\n"
                
                # Also show state summary
                state_result = await execute_terraform_command(["terraform", "show", "-json"], config_dir)
                if state_result["success"]:
                    state_data = json.loads(state_result["stdout"])
                    resource_count = len(state_data.get("values", {}).get("root_module", {}).get("resources", []))
                    output_text += f"\n**Resources Managed:** {resource_count}\n"
                    output_text += f"**State File:** Terraform managed\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=output_text)]
                )
            except json.JSONDecodeError:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"✅ Terraform Status:\n\n{output_result['stdout']}")]
                )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Terraform status query failed: {output_result['stderr']}")],
                isError=True
            )
    
    elif tool_name == "status":
        if arguments.get("live"):
            cmd_args.append("--live")
    
    elif tool_name == "manage_instance":
        cmd_args.extend(["-i", arguments["instance_id"]])
        cmd_args.extend(["-a", arguments["action"]])
    
    elif tool_name == "analytics":
        if "days" in arguments:
            cmd_args.extend(["--days", str(arguments["days"])])
    
    elif tool_name == "setup_provider":
        cmd_args.append(arguments["provider"])
        if arguments.get("quick"):
            cmd_args.append("--quick")
    
    elif tool_name == "configure_provider":
        cmd_args.extend(["--provider", arguments["provider"]])
    
    # ── v3.2.0 Handlers ──────────────────────────────────────────────────

    elif tool_name == "infer_route":
        # Semantic-aware inference routing via terradev inference route
        cmd_args = ["inference", "route"]
        if "model" in arguments:
            cmd_args.extend(["--model", arguments["model"]])
        strategy = arguments.get("strategy", "latency")
        cmd_args.extend(["--strategy", strategy])
        if arguments.get("measure"):
            cmd_args.append("--measure")
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = "🧠 **Semantic Inference Routing**\n\n"
        if result["success"]:
            output_text += f"**Strategy:** {strategy}\n"
            output_text += f"**Signals:** modality, complexity, domain, language, safety, keywords\n"
            output_text += f"**NUMA scoring:** enabled\n\n"
            output_text += output
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 **Tip:** Register inference endpoints first with:\n"
            output_text += "   `terradev inference deploy --provider runpod --model <model>`\n"
            output_text += "   Then route with: `terradev inference route --strategy latency`"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "infer_route_disagg":
        # Disaggregated prefill/decode routing
        cmd_args = ["inference", "route", "--disagg"]
        cmd_args.extend(["--model", arguments["model"]])
        if arguments.get("check_health", True):
            cmd_args.append("--check")
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = "⚡ **Disaggregated Prefill/Decode Routing (DistServe)**\n\n"
        if result["success"]:
            output_text += f"**Model:** {arguments['model']}\n"
            output_text += "**Architecture:** DistServe — PREFILL (compute-bound) → DECODE (memory-bound)\n"
            output_text += "**KV Cache Handoff:** tracked via PrefillDecodeTracker\n\n"
            output_text += output
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 **Tip:** Disaggregated routing requires endpoints tagged with phase:\n"
            output_text += "   PREFILL endpoints: high-FLOPS GPUs (H100 SXM)\n"
            output_text += "   DECODE endpoints: high-bandwidth GPUs (H200, MI300X)\n"
            output_text += "   Register with: `terradev inference deploy --phase prefill --gpu H100`"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "infer_status":
        cmd_args = ["inference", "status"]
        if arguments.get("check"):
            cmd_args.append("--check")
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = "📊 **Inference Endpoint Status**\n\n"
        if result["success"]:
            output_text += output
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 No inference endpoints registered. Deploy one with:\n"
            output_text += "   `terradev inference deploy --provider runpod --model <model> --gpu H100`"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "infer_failover":
        cmd_args = ["inference", "failover"]
        if arguments.get("dry_run"):
            cmd_args.append("--dry-run")
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = "🔄 **Inference Auto-Failover**\n\n"
        if result["success"]:
            output_text += output
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 Register backup endpoints with:\n"
            output_text += "   `terradev inference deploy --provider <backup> --model <model> --backup`"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "gpu_topology":
        cmd_args = ["inference", "topology"]
        gpu_arch = arguments.get("gpu_arch", "auto")
        if gpu_arch and gpu_arch != "auto":
            cmd_args.extend(["--arch", gpu_arch])
        if arguments.get("generate_env", True):
            cmd_args.append("--env")
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = "🔬 **GPU NUMA Topology Report**\n\n"
        if result["success"]:
            output_text += output
            if arguments.get("generate_env", True):
                output_text += "\n\n**XCD-Aware Environment Variables Generated**\n"
                output_text += "Apply these to your vLLM/SGLang process for optimal attention kernel performance.\n"
        else:
            # Provide useful topology info even without live GPUs
            output_text += f"⚠️ {output}\n\n"
            output_text += "📋 **Reference: Intra-GPU NUMA Topology**\n\n"
            output_text += "| GPU | XCDs | HBM | Architecture |\n"
            output_text += "|-----|------|-----|-------------|\n"
            output_text += "| MI300X | 8 XCDs | 192GB HBM3 | CDNA3 chiplet |\n"
            output_text += "| MI300A | 6 XCDs | 128GB HBM3 | CDNA3 APU |\n"
            output_text += "| H200 | 1 (unified) | 141GB HBM3e | Hopper |\n"
            output_text += "| H100 SXM | 1 (unified) | 80GB HBM3 | Hopper |\n"
            output_text += "| A100 | 1 (unified) | 80GB HBM2e | Ampere |\n\n"
            output_text += "💡 **XCD-aware env vars for MI300X:**\n"
            output_text += "```\n"
            output_text += "AITER_XCD_AWARE_ATTENTION=1\n"
            output_text += "CK_BLOCK_MAPPING_POLICY=xcd_aware\n"
            output_text += "NCCL_INTRA_GPU_NUMA=1\n"
            output_text += "```"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "price_intel":
        gpu_type = arguments["gpu_type"]
        days = arguments.get("days", 7)
        cmd_args = ["analytics", "--price-intel", "--gpu", gpu_type, "--days", str(days)]
        if "provider" in arguments:
            cmd_args.extend(["--provider", arguments["provider"]])
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = f"📈 **GPU Price Intelligence — {gpu_type}**\n\n"
        if result["success"]:
            output_text += f"**Period:** {days} days\n"
            output_text += "**Metrics:** delta (δ), gamma (γ), realized volatility (σ)\n\n"
            output_text += output
        else:
            # Still useful — run a fresh quote to seed the price tick db
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 **Tip:** Price intelligence requires historical data. Seed it with:\n"
            output_text += f"   `terradev quote -g {gpu_type}` (run periodically to build history)\n\n"
            output_text += "**Metrics available after seeding:**\n"
            output_text += "- **Delta (δ):** Rate of price change ($/hr/day)\n"
            output_text += "- **Gamma (γ):** Acceleration of price change\n"
            output_text += "- **Realized Volatility (σ):** Annualized price volatility\n"
            output_text += "- **Cheapest Window:** Best time to provision\n"
            output_text += "- **Arbitrage Spread:** Max price difference across providers"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "moe_deploy":
        model_id = arguments["model_id"]
        gpu_type = arguments["gpu_type"]
        tp_size = arguments.get("tp_size", 8)
        backend = arguments.get("backend", "vllm")
        quantization = arguments.get("quantization", "fp8")
        dry_run = arguments.get("dry_run", False)
        
        cmd_args = ["provision", "--task", "clusters/moe-template/task.yaml",
                     "--set", f"model_id={model_id}",
                     "--set", f"tp_size={tp_size}",
                     "--set", f"gpu_type={gpu_type}",
                     "--set", f"backend={backend}",
                     "--set", f"quantization={quantization}"]
        if dry_run:
            cmd_args.append("--dry-run")
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = f"🧬 **MoE Cluster Deployment**\n\n"
        output_text += f"**Model:** {model_id}\n"
        output_text += f"**GPU:** {gpu_type} × {tp_size} (TP={tp_size})\n"
        output_text += f"**Backend:** {backend}\n"
        output_text += f"**Quantization:** {quantization}\n"
        output_text += f"**Dry Run:** {dry_run}\n\n"
        output_text += "💰 **Auto-Applied Cost Optimizations:**\n"
        output_text += "• KV Cache Offloading → CPU DRAM (up to 9x throughput)\n"
        output_text += "• MTP Speculative Decoding (up to 2.8x generation speed)\n"
        output_text += "• Sleep Mode (18-200x faster than cold restart on idle)\n"
        output_text += "• Expert Load Balancing + DeepEP/DeepGEMM kernels\n\n"
        
        if result["success"]:
            output_text += output
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 **Manual deployment:**\n"
            output_text += f"```bash\n"
            output_text += f"terradev provision --task clusters/moe-template/task.yaml \\\n"
            output_text += f"  --set model_id={model_id} --set tp_size={tp_size}\n"
            output_text += f"```\n\n"
            output_text += "**Or via Kubernetes:**\n"
            output_text += f"```bash\n"
            output_text += f"kubectl apply -f clusters/moe-template/k8s/\n"
            output_text += f"```"
        
        output_text += "\n\n🔗 **Next:** Use `lora_add` to hot-load fine-tuned adapters onto this endpoint."
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    # ── v3.5.0 Handlers: Multi-LoRA ───────────────────────────────────

    elif tool_name == "lora_list":
        endpoint = arguments["endpoint"]
        api_key = arguments.get("api_key", "")
        cmd_args = ["lora", "list", "-e", endpoint]
        if api_key:
            cmd_args.extend(["--api-key", api_key])
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = f"🔍 **LoRA Adapters on {endpoint}**\n\n"
        output_text += output if output.strip() else "No adapters loaded.\n"
        output_text += "\n💡 Use `lora_add` to hot-load a fine-tuned adapter."
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "lora_add":
        endpoint = arguments["endpoint"]
        name = arguments["name"]
        path = arguments["path"]
        api_key = arguments.get("api_key", "")
        cmd_args = ["lora", "add", "-e", endpoint, "-n", name, "-p", path]
        if api_key:
            cmd_args.extend(["--api-key", api_key])
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        if result["success"]:
            output_text = f"✅ **Adapter '{name}' loaded on {endpoint}**\n\n"
            output_text += f"Use in API requests: `\"model\": \"{name}\"`\n\n"
            output_text += f"```bash\n"
            output_text += f"curl {endpoint}/v1/chat/completions \\\n"
            output_text += f"  -d '{{\"model\": \"{name}\", \"messages\": [...]}}' \n"
            output_text += f"```"
        else:
            output_text = f"❌ **Failed to load adapter '{name}'**\n\n{output}"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    elif tool_name == "lora_remove":
        endpoint = arguments["endpoint"]
        name = arguments["name"]
        api_key = arguments.get("api_key", "")
        cmd_args = ["lora", "remove", "-e", endpoint, "-n", name]
        if api_key:
            cmd_args.extend(["--api-key", api_key])
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        if result["success"]:
            output_text = f"✅ **Adapter '{name}' unloaded from {endpoint}**\n"
            output_text += "GPU memory freed for other adapters."
        else:
            output_text = f"❌ **Failed to unload adapter '{name}'**\n\n{output}"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    # ── v3.4.0 Handlers ──────────────────────────────────────────────────

    elif tool_name == "train":
        cmd_args = ["train", "--script", arguments["script"]]
        if "framework" in arguments:
            cmd_args.extend(["--framework", arguments["framework"]])
        if "from_provision" in arguments:
            cmd_args.extend(["--from-provision", arguments["from_provision"]])
        elif "nodes" in arguments:
            for node in arguments["nodes"]:
                cmd_args.extend(["--node", node])
        if "gpus_per_node" in arguments:
            cmd_args.extend(["--gpus-per-node", str(arguments["gpus_per_node"])])
        if "script_args" in arguments:
            cmd_args.extend(["--", arguments["script_args"]])

        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🚀 **Training Launch**\n\n"
        if result["success"]:
            output_text += f"**Script:** {arguments['script']}\n"
            output_text += f"**Framework:** {arguments.get('framework', 'torchrun')}\n\n"
            output_text += output
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 **Tip:** Provision GPU nodes first:\n"
            output_text += "   `terradev provision -g H100 -n 4`\n"
            output_text += "   Then: `terradev train --script train.py --from-provision latest`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "train_status":
        cmd_args = ["train-status"]
        if "job_id" in arguments and arguments["job_id"]:
            cmd_args.extend(["--job", arguments["job_id"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "📋 **Training Jobs**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "train_monitor":
        cmd_args = ["monitor", "--job", arguments["job_id"]]
        if "cost_rate" in arguments:
            cmd_args.extend(["--cost-rate", str(arguments["cost_rate"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "📊 **GPU Monitor**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "checkpoint_list":
        cmd_args = ["checkpoint", "list", "--job", arguments["job_id"]]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "💾 **Checkpoints**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "checkpoint_save":
        cmd_args = ["checkpoint", "save", "--job", arguments["job_id"]]
        if "step" in arguments:
            cmd_args.extend(["--step", str(arguments["step"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "💾 **Checkpoint Save**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "preflight":
        cmd_args = ["preflight"]
        if "from_provision" in arguments:
            cmd_args.extend(["--from-provision", arguments["from_provision"]])
        elif "nodes" in arguments:
            for node in arguments["nodes"]:
                cmd_args.extend(["--node", node])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "✅ **Preflight Validation**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "price_discovery":
        cmd_args = ["price-discovery", "--gpu-type", arguments["gpu_type"]]
        if "region" in arguments:
            cmd_args.extend(["--region", arguments["region"]])
        if "hours" in arguments:
            cmd_args.extend(["--hours", str(arguments["hours"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"💰 **Price Discovery — {arguments['gpu_type']}**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "gitops_init":
        repo = arguments["repo"]
        tool = arguments.get("tool", "argocd")
        provider = arguments.get("provider", "github")
        cluster = arguments.get("cluster", "production")
        
        cmd_args = ["gitops", "init",
                     "--provider", provider,
                     "--repo", repo,
                     "--tool", tool,
                     "--cluster", cluster]
        
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        
        output_text = f"🔧 **GitOps Repository Initialized**\n\n"
        if result["success"]:
            output_text += f"**Repository:** {repo}\n"
            output_text += f"**Tool:** {tool}\n"
            output_text += f"**Provider:** {provider}\n"
            output_text += f"**Cluster:** {cluster}\n\n"
            output_text += output
            output_text += "\n\n**Next steps:**\n"
            output_text += f"1. `terradev gitops bootstrap --tool {tool} --cluster {cluster}`\n"
            output_text += f"2. `terradev gitops sync --cluster {cluster} --environment prod`\n"
            output_text += f"3. `terradev gitops validate --dry-run --cluster {cluster}`"
        else:
            output_text += f"⚠️ {output}"
        
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)],
            isError=not result["success"]
        )
    
    # ── v2.0.0 Handlers — Complete Agentic Loop ────────────────────────

    elif tool_name == "train_stop":
        cmd_args = ["train-stop", "--job-id", arguments["job_id"], "-f", "json"]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "⏹️ **Training Stop**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Check final status with `train_status`, then optionally `train_resume` later."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "train_resume":
        cmd_args = ["train-resume", "--job-id", arguments["job_id"], "-f", "json"]
        if arguments.get("checkpoint_id"):
            cmd_args.extend(["--checkpoint-id", arguments["checkpoint_id"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "▶️ **Training Resume**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Monitor progress with `train_monitor`. Check `train_status` for ETA."
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 **Tip:** Ensure the job has checkpoints: `checkpoint_list`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "checkpoint_restore":
        cmd_args = ["checkpoint", "restore", "--job-id", arguments["job_id"], "-f", "json"]
        if arguments.get("step"):
            cmd_args.extend(["--step", str(arguments["step"])])
        if arguments.get("checkpoint_id"):
            cmd_args.extend(["--checkpoint-id", arguments["checkpoint_id"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "💾 **Checkpoint Restore**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Resume training with `train_resume` or promote with `checkpoint_promote`."
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "💡 List available checkpoints: `checkpoint_list`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "checkpoint_promote":
        cmd_args = ["checkpoint", "promote", "--job-id", arguments["job_id"],
                     "--checkpoint-id", arguments["checkpoint_id"],
                     "--dest", arguments["dest"], "-f", "json"]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🏆 **Checkpoint Promoted**\n\n"
        if result["success"]:
            output_text += f"**Destination:** {arguments['dest']}\n\n"
            output_text += output
            output_text += "\n\n**suggest_action:** Deploy for inference with `infer_deploy` or `inferx_deploy`."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "checkpoint_delete":
        cmd_args = ["checkpoint", "delete", "--job-id", arguments["job_id"],
                     "--checkpoint-id", arguments["checkpoint_id"], "-f", "json"]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🗑️ **Checkpoint Deleted**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "stage":
        cmd_args = ["stage", "--dataset", arguments["dataset"]]
        if arguments.get("target_regions"):
            cmd_args.extend(["--target-regions", arguments["target_regions"]])
        if arguments.get("compression"):
            cmd_args.extend(["--compression", arguments["compression"]])
        if arguments.get("plan_only"):
            cmd_args.append("--plan-only")
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "📦 **Data Staging**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Data is staged. Proceed with `train` to start training or `preflight` to validate nodes."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "infer_deploy":
        model_path = arguments["model_path"]
        name = arguments["name"]
        # Cost guardrail: estimate and warn
        est_cost = 0.0
        gpu_type = arguments.get("gpu_type", "A100")
        gpu_costs = {"H100": 3.50, "A100": 2.20, "A10G": 1.10, "L40S": 1.80, "L4": 0.80, "T4": 0.50}
        est_cost = gpu_costs.get(gpu_type, 2.00) * arguments.get("max_workers", 3)

        if arguments.get("dry_run"):
            output_text = "📋 **Inference Deployment Plan (Dry Run)**\n\n"
            output_text += f"**Model:** {model_path}\n"
            output_text += f"**Endpoint:** {name}\n"
            output_text += f"**GPU:** {gpu_type}\n"
            output_text += f"**Workers:** {arguments.get('min_workers', 0)}-{arguments.get('max_workers', 3)}\n"
            output_text += f"**Idle Timeout:** {arguments.get('idle_timeout', 300)}s\n"
            output_text += f"**Estimated Max Cost:** ${est_cost:.2f}/hr\n\n"
            output_text += "**requires_confirmation:** true\n"
            output_text += f"**estimated_cost:** ${est_cost:.2f}/hr (max {arguments.get('max_workers', 3)} workers × ${gpu_costs.get(gpu_type, 2.00):.2f}/hr)\n\n"
            output_text += "**suggest_action:** Call `infer_deploy` again without `dry_run` to execute."
            return CallToolResult(content=[TextContent(type="text", text=output_text)])

        cmd_args = ["infer-deploy", model_path, "--name", name]
        if arguments.get("provider"):
            cmd_args.extend(["--provider", arguments["provider"]])
        if arguments.get("gpu_type"):
            cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
        if "idle_timeout" in arguments:
            cmd_args.extend(["--idle-timeout", str(arguments["idle_timeout"])])
        if arguments.get("cost_optimize"):
            cmd_args.append("--cost-optimize")
        if "min_workers" in arguments:
            cmd_args.extend(["--min-workers", str(arguments["min_workers"])])
        if "max_workers" in arguments:
            cmd_args.extend(["--max-workers", str(arguments["max_workers"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🚀 **Inference Deployment**\n\n"
        if result["success"]:
            output_text += output
            output_text += f"\n\n**estimated_cost:** ${est_cost:.2f}/hr (max)\n"
            output_text += "**suggest_action:** Check status with `infer_status`. Monitor with `infer_failover --dry-run`."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "up":
        cmd_args = ["up", "--job", arguments["job"]]
        if arguments.get("gpu_type"):
            cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
        if arguments.get("gpu_count"):
            cmd_args.extend(["--count", str(arguments["gpu_count"])])
        if arguments.get("ttl"):
            cmd_args.extend(["--ttl", arguments["ttl"]])
        if arguments.get("budget"):
            cmd_args.extend(["--budget", str(arguments["budget"])])
        if arguments.get("region"):
            cmd_args.extend(["--region", arguments["region"]])
        if arguments.get("fix_drift"):
            cmd_args.append("--fix-drift")
        # Cost guardrail
        gpu_type = arguments.get("gpu_type", "A100")
        gpu_count = arguments.get("gpu_count", 1)
        gpu_costs = {"H100": 3.50, "A100": 2.20, "A10G": 1.10, "L40S": 1.80, "L4": 0.80, "T4": 0.50}
        est_hourly = gpu_costs.get(gpu_type, 2.00) * gpu_count
        hours = arguments.get("hours", 1.0)
        est_total = est_hourly * hours

        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "⬆️ **Manifest-Cached Provision**\n\n"
        if result["success"]:
            output_text += output
            output_text += f"\n\n**estimated_cost:** ${est_hourly:.2f}/hr × {hours}h = ${est_total:.2f}\n"
            if est_total > 50:
                output_text += "⚠️ **Cost Warning:** Estimated spend exceeds $50. Monitor with `status`.\n"
            output_text += "**suggest_action:** Run `preflight` then `train` on provisioned nodes."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "rollback":
        cmd_args = ["rollback", arguments["job_version"]]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "⏪ **Rollback**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Check current state with `manifests` and verify with `status`."
        else:
            output_text += f"⚠️ {output}\n\n💡 List versions: `manifests`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "manifests":
        cmd_args = ["manifests"]
        if arguments.get("job"):
            cmd_args.extend(["--job", arguments["job"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "📋 **Cached Manifests**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "smart_deploy":
        cmd_args = ["smart-deploy", "--image", arguments["image"], "--workload", arguments["workload"]]
        if arguments.get("gpu_type"):
            cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
        if arguments.get("budget"):
            cmd_args.extend(["--budget", str(arguments["budget"])])
        if arguments.get("option") is not None:
            cmd_args.extend(["--option", str(arguments["option"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🧠 **Smart Deployment**\n\n"
        if result["success"]:
            output_text += output
            if arguments.get("option") is None:
                output_text += "\n\n**requires_confirmation:** true\n"
                output_text += "**suggest_action:** Review options above. Execute with `smart_deploy` and `option` parameter set to your chosen index."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "helm_generate":
        cmd_args = ["helm-generate", "--workload", arguments["workload"], "--image", arguments["image"]]
        if arguments.get("gpu_type"):
            cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
        if arguments.get("replicas"):
            cmd_args.extend(["--replicas", str(arguments["replicas"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "⎈ **Helm Chart Generated**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Apply with `kubectl apply -f` or deploy to cluster with `k8s_create`."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "gitops_bootstrap":
        tool = arguments["tool"]
        cluster = arguments["cluster"]
        cmd_args = ["gitops", "bootstrap", "--tool", tool, "--cluster", cluster]
        if arguments.get("namespace"):
            cmd_args.extend(["--namespace", arguments["namespace"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"🔧 **GitOps Bootstrap ({tool})**\n\n"
        if result["success"]:
            output_text += output
            output_text += f"\n\n**suggest_action:** Sync with `gitops_sync --cluster {cluster}`. Validate with `gitops_validate`."
        else:
            output_text += f"⚠️ {output}\n\n💡 Initialize first: `gitops_init`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "gitops_sync":
        cluster = arguments["cluster"]
        cmd_args = ["gitops", "sync", "--cluster", cluster]
        if arguments.get("environment"):
            cmd_args.extend(["--environment", arguments["environment"]])
        if arguments.get("tool"):
            cmd_args.extend(["--tool", arguments["tool"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"🔄 **GitOps Sync — {cluster}**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Validate sync with `gitops_validate`."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "gitops_validate":
        cmd_args = ["gitops", "validate"]
        if arguments.get("cluster"):
            cmd_args.extend(["--cluster", arguments["cluster"]])
        if arguments.get("dry_run", True):
            cmd_args.append("--dry-run")
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "✅ **GitOps Validation**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "orchestrator_start":
        cmd_args = ["orchestrator-start"]
        if arguments.get("gpu_id") is not None:
            cmd_args.extend(["--gpu-id", str(arguments["gpu_id"])])
        if arguments.get("memory_gb"):
            cmd_args.extend(["--memory-gb", str(arguments["memory_gb"])])
        if arguments.get("policy"):
            cmd_args.extend(["--policy", arguments["policy"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🎛️ **Model Orchestrator Started**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Register models with `orchestrator_register`, then `orchestrator_load` to load them."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "orchestrator_register":
        cmd_args = ["orchestrator-register", arguments["model_id"], arguments["model_path"]]
        if arguments.get("framework"):
            cmd_args.extend(["--framework", arguments["framework"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"📝 **Model Registered: {arguments['model_id']}**\n\n"
        if result["success"]:
            output_text += output
            output_text += f"\n\n**suggest_action:** Load into GPU memory with `orchestrator_load`."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "orchestrator_load":
        cmd_args = ["orchestrator-load", arguments["model_id"]]
        if arguments.get("force"):
            cmd_args.append("--force")
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"📥 **Model Loaded: {arguments['model_id']}**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Test with `orchestrator_infer`. Check memory with `orchestrator_status`."
        else:
            output_text += f"⚠️ {output}\n\n💡 Check memory: `orchestrator_status`. Try `--force` to evict least-used."
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "orchestrator_evict":
        cmd_args = ["orchestrator-evict", arguments["model_id"]]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"📤 **Model Evicted: {arguments['model_id']}**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "orchestrator_status":
        cmd_args = ["orchestrator-status"]
        if arguments.get("model_id"):
            cmd_args.append(arguments["model_id"])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🎛️ **Orchestrator Status**\n\n"
        if result["success"]:
            output_text += output
            # Agent recommendations based on output
            if "utilization" in output.lower():
                output_text += "\n\n**recommend:** "
                if "90%" in output or "95%" in output or "100%" in output:
                    output_text += "GPU memory is near capacity. Consider `orchestrator_evict` for idle models or scaling up."
                elif "10%" in output or "15%" in output or "20%" in output:
                    output_text += "GPU memory is underutilized. Consider loading more models with `orchestrator_load`."
                else:
                    output_text += "Memory utilization is balanced."
        else:
            output_text += f"⚠️ {output}\n\n💡 Start orchestrator first: `orchestrator_start`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "orchestrator_infer":
        cmd_args = ["orchestrator-infer", arguments["model_id"]]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"⚡ **Inference Test: {arguments['model_id']}**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "warm_pool_start":
        cmd_args = ["warm-pool-start"]
        if arguments.get("strategy"):
            cmd_args.extend(["--strategy", arguments["strategy"]])
        if arguments.get("max_warm"):
            cmd_args.extend(["--max-warm", str(arguments["max_warm"])])
        if arguments.get("min_warm"):
            cmd_args.extend(["--min-warm", str(arguments["min_warm"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🔥 **Warm Pool Started**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Register models with `warm_pool_register`. Check `warm_pool_status` for hit rates."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "warm_pool_status":
        cmd_args = ["warm-pool-status"]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🔥 **Warm Pool Status**\n\n"
        if result["success"]:
            output_text += output
            # Agent recommendation
            output_text += "\n\n**recommend:** "
            if "hit rate" in output.lower():
                output_text += "Review hit rate. If below 80%, consider adjusting strategy or increasing max_warm."
            else:
                output_text += "Pool is operational."
        else:
            output_text += f"⚠️ {output}\n\n💡 Start warm pool first: `warm_pool_start`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "cost_scaler_start":
        cmd_args = ["cost-scaler-start"]
        if arguments.get("strategy"):
            cmd_args.extend(["--strategy", arguments["strategy"]])
        if arguments.get("budget"):
            cmd_args.extend(["--budget", str(arguments["budget"])])
        if arguments.get("cost_per_gb"):
            cmd_args.extend(["--cost-per-gb", str(arguments["cost_per_gb"])])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "💰 **Cost Scaler Started**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Monitor with `cost_scaler_status` for budget utilization and predictions."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "cost_scaler_status":
        cmd_args = ["cost-scaler-status"]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "💰 **Cost Scaler Status**\n\n"
        if result["success"]:
            output_text += output
            # Agent recommendations
            output_text += "\n\n**recommend:** "
            if "budget" in output.lower() and ("exceed" in output.lower() or "over" in output.lower()):
                output_text += "⚠️ Budget at risk. Consider scaling down or switching strategy to `minimize_cost`."
            elif "under" in output.lower():
                output_text += "Budget is healthy. You have headroom to scale up if needed."
            else:
                output_text += "Cost scaling is within bounds."
        else:
            output_text += f"⚠️ {output}\n\n💡 Start cost scaler first: `cost_scaler_start`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "inferx_configure":
        cmd_args = ["inferx", "configure", "--api-key", arguments["api_key"]]
        if arguments.get("endpoint"):
            cmd_args.extend(["--endpoint", arguments["endpoint"]])
        if arguments.get("region"):
            cmd_args.extend(["--region", arguments["region"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🔑 **InferX Configured**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Deploy a model with `inferx_deploy` or check quotes with `inferx_quote`."
        else:
            output_text += f"⚠️ {output}"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "inferx_delete":
        cmd_args = ["inferx", "delete", "--model-id", arguments["model_id"]]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"🗑️ **InferX Deployment Deleted: {arguments['model_id']}**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "inferx_usage":
        cmd_args = ["inferx", "usage"]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "📊 **InferX Usage**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Optimize costs with `inferx_optimize`."
        else:
            output_text += f"⚠️ {output}\n\n💡 Configure InferX first: `inferx_configure`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "inferx_quote":
        cmd_args = ["inferx", "quote"]
        if arguments.get("gpu_type"):
            cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
        if arguments.get("region"):
            cmd_args.extend(["--region", arguments["region"]])
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "💰 **InferX Pricing Quote**\n\n"
        if result["success"]:
            output_text += output
            output_text += "\n\n**suggest_action:** Deploy with `inferx_deploy` at these rates."
        else:
            output_text += f"⚠️ {output}\n\n💡 Configure InferX first: `inferx_configure`"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "hf_space_status":
        cmd_args = ["hf-space", "--status", arguments["space_name"]]
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = f"🤗 **HF Space Status: {arguments['space_name']}**\n\n" + output
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "run_workflow":
        # Workflow primitives — runs a YAML pipeline or built-in template
        if arguments.get("template"):
            cmd_args = ["workflow", "run", "--template", arguments["template"]]
        elif arguments.get("workflow"):
            cmd_args = ["workflow", "run", arguments["workflow"]]
        else:
            return CallToolResult(
                content=[TextContent(type="text", text="⚠️ Provide either `workflow` (YAML path) or `template` (built-in).")],
                isError=True
            )
        if arguments.get("dry_run"):
            cmd_args.append("--dry-run")
        result = await execute_terradev_command(cmd_args)
        output = result["stdout"] if result["success"] else result["stderr"]
        output_text = "🔄 **Workflow Execution**\n\n"
        if result["success"]:
            output_text += output
            if arguments.get("dry_run"):
                output_text += "\n\n**requires_confirmation:** true\n"
                output_text += "**suggest_action:** Review the plan above. Run again without `dry_run` to execute."
            else:
                output_text += "\n\n**suggest_action:** Monitor progress with `active_context` or `train_status`."
        else:
            output_text += f"⚠️ {output}\n\n"
            output_text += "**Available templates:** finetune-llama, inference-deploy, benchmark-gpu, cost-optimize"
        return CallToolResult(content=[TextContent(type="text", text=output_text)], isError=not result["success"])

    elif tool_name == "active_context":
        # Composite tool: gather state from multiple commands
        context_parts = []

        # 1. Running training jobs
        jobs_result = await execute_terradev_command(["train-status", "-f", "json"])
        if jobs_result["success"]:
            context_parts.append(f"**Training Jobs:**\n{jobs_result['stdout']}")
        else:
            context_parts.append("**Training Jobs:** None running")

        # 2. Active instances
        status_result = await execute_terradev_command(["status", "-f", "json"])
        if status_result["success"]:
            context_parts.append(f"\n**Active Instances:**\n{status_result['stdout']}")
        else:
            context_parts.append("\n**Active Instances:** None")

        # 3. Cost analytics (last 7 days)
        analytics_result = await execute_terradev_command(["analytics", "--days", "7", "-f", "json"])
        if analytics_result["success"]:
            context_parts.append(f"\n**Spend (7 days):**\n{analytics_result['stdout']}")
        else:
            context_parts.append("\n**Spend:** No data")

        output_text = "🏠 **Active Context — Terradev State**\n\n"
        output_text += "\n".join(context_parts)
        output_text += "\n\n**suggest_action:** "
        if jobs_result["success"] and "running" in jobs_result["stdout"].lower():
            output_text += "You have running jobs. Monitor with `train_monitor` or check `train_status`."
        elif status_result["success"] and status_result["stdout"].strip() and status_result["stdout"].strip() != "[]":
            output_text += "You have active instances. Consider `optimize` to find cheaper alternatives."
        else:
            output_text += "No active workloads. Start with `quote_gpu` to compare prices, then `provision_gpu` or `up`."
        return CallToolResult(content=[TextContent(type="text", text=output_text)])

    # Execute the command (generic fallback for simple tools)
    result = await execute_terradev_command(cmd_args)
    
    if result["success"]:
        return CallToolResult(
            content=[TextContent(type="text", text=result["stdout"])]
        )
    else:
        error_msg = result["stderr"] or "Command failed"
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {error_msg}")],
            isError=True
        )

@server.list_resources()
async def handle_list_resources() -> ListResourcesResult:
    """List available MCP resources for session-start context and polling."""
    return ListResourcesResult(resources=[
        Resource(
            uri="terradev://active_context",
            name="Active Context",
            description="Current Terradev state: running jobs, active instances, spend-to-date, alerts. Read on session start.",
            mimeType="application/json",
        ),
        Resource(
            uri="terradev://instances",
            name="Active Instances",
            description="Currently provisioned GPU instances across all providers.",
            mimeType="application/json",
        ),
        Resource(
            uri="terradev://jobs",
            name="Training Jobs",
            description="All training jobs with status, progress, and ETA.",
            mimeType="application/json",
        ),
        Resource(
            uri="terradev://spend",
            name="Spend Summary",
            description="Cost analytics and spend-to-date across all providers.",
            mimeType="application/json",
        ),
        Resource(
            uri="terradev://alerts",
            name="Alerts",
            description="Active alerts: straggler nodes, budget warnings, drift detected, failed health checks.",
            mimeType="application/json",
        ),
    ])

@server.read_resource()
async def handle_read_resource(request: ReadResourceRequest) -> ReadResourceResult:
    """Read a Terradev resource."""
    uri = str(request.params.uri)

    if uri == "terradev://active_context":
        # Composite: jobs + instances + spend
        jobs = await execute_terradev_command(["train-status", "-f", "json"])
        instances = await execute_terradev_command(["status", "-f", "json"])
        spend = await execute_terradev_command(["analytics", "--days", "7", "-f", "json"])
        context = {
            "jobs": jobs["stdout"] if jobs["success"] else None,
            "instances": instances["stdout"] if instances["success"] else None,
            "spend_7d": spend["stdout"] if spend["success"] else None,
            "suggest_action": "Call active_context tool for formatted recommendations.",
        }
        return ReadResourceResult(contents=[
            TextResourceContents(uri=uri, mimeType="application/json", text=json.dumps(context, indent=2))
        ])

    elif uri == "terradev://instances":
        result = await execute_terradev_command(["status", "-f", "json"])
        text = result["stdout"] if result["success"] else json.dumps({"error": result["stderr"]})
        return ReadResourceResult(contents=[
            TextResourceContents(uri=uri, mimeType="application/json", text=text)
        ])

    elif uri == "terradev://jobs":
        result = await execute_terradev_command(["train-status", "-f", "json"])
        text = result["stdout"] if result["success"] else json.dumps({"error": result["stderr"]})
        return ReadResourceResult(contents=[
            TextResourceContents(uri=uri, mimeType="application/json", text=text)
        ])

    elif uri == "terradev://spend":
        result = await execute_terradev_command(["analytics", "--days", "30", "-f", "json"])
        text = result["stdout"] if result["success"] else json.dumps({"error": result["stderr"]})
        return ReadResourceResult(contents=[
            TextResourceContents(uri=uri, mimeType="application/json", text=text)
        ])

    elif uri == "terradev://alerts":
        # Aggregate alerts from multiple sources
        alerts = []
        # Check for straggler nodes via monitor
        monitor = await execute_terradev_command(["monitor", "--check-stragglers", "-f", "json"])
        if monitor["success"] and monitor["stdout"].strip():
            alerts.append({"type": "straggler", "data": monitor["stdout"]})
        # Check cost budget
        cost = await execute_terradev_command(["cost-scaler-status", "-f", "json"])
        if cost["success"] and "exceed" in cost["stdout"].lower():
            alerts.append({"type": "budget_warning", "data": cost["stdout"]})
        # Check drift
        drift = await execute_terradev_command(["manifests", "--check-drift", "-f", "json"])
        if drift["success"] and "drift" in drift["stdout"].lower():
            alerts.append({"type": "drift_detected", "data": drift["stdout"]})
        text = json.dumps({"alerts": alerts, "count": len(alerts)}, indent=2)
        return ReadResourceResult(contents=[
            TextResourceContents(uri=uri, mimeType="application/json", text=text)
        ])

    return ReadResourceResult(contents=[
        TextResourceContents(uri=uri, mimeType="application/json", text=json.dumps({"error": f"Unknown resource: {uri}"}))
    ])

@server.list_prompts()
async def handle_list_prompts() -> ListPromptsResult:
    """List available prompts"""
    return ListPromptsResult(prompts=[])

@server.get_prompt()
async def handle_get_prompt(request: GetPromptRequest) -> GetPromptResult:
    """Get a prompt"""
    return GetPromptResult(description="", messages=[])

# ---------------------------------------------------------------------------
# OAuth 2.0 PKCE auth for Claude.ai Connectors
# ---------------------------------------------------------------------------

TERRADEV_MCP_BEARER_TOKEN = os.getenv("TERRADEV_MCP_BEARER_TOKEN", "")

# In-memory stores (single-instance server)
_auth_codes: Dict[str, Dict[str, Any]] = {}   # code -> {client_id, code_challenge, redirect_uri, expires}
_access_tokens: Dict[str, Dict[str, Any]] = {}  # token -> {client_id, expires}


def _cleanup_expired():
    """Remove expired auth codes and tokens."""
    now = time.time()
    for store in (_auth_codes, _access_tokens):
        expired = [k for k, v in store.items() if v.get("expires", 0) < now]
        for k in expired:
            del store[k]


# ---------------------------------------------------------------------------
# OAuth endpoint handlers (added as Starlette routes)
# ---------------------------------------------------------------------------

async def oauth_authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 — OAuth Authorization Server Metadata."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": base + "/authorize",
        "token_endpoint": base + "/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


async def oauth_protected_resource(request: Request) -> JSONResponse:
    """RFC 9728 — OAuth Protected Resource Metadata."""
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "resource": base,
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
    })


async def oauth_authorize(request: Request) -> Response:
    """OAuth 2.0 Authorization Endpoint — auto-approves if client_id matches our token."""
    from starlette.responses import RedirectResponse

    params = dict(request.query_params)
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "")
    state = params.get("state", "")

    logger.info("OAuth authorize: client_id=%s... redirect=%s", client_id[:16], redirect_uri)

    # Validate client_id matches our configured token
    if TERRADEV_MCP_BEARER_TOKEN and client_id != TERRADEV_MCP_BEARER_TOKEN:
        logger.warning("OAuth authorize rejected: bad client_id")
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if code_challenge_method and code_challenge_method != "S256":
        return JSONResponse({"error": "invalid_request", "error_description": "Only S256 supported"}, status_code=400)

    # Generate authorization code
    _cleanup_expired()
    code = secrets.token_urlsafe(48)
    _auth_codes[code] = {
        "client_id": client_id,
        "code_challenge": code_challenge,
        "redirect_uri": redirect_uri,
        "expires": time.time() + 300,  # 5 min
    }

    # Redirect back to Claude.ai with the code
    sep = "&" if "?" in redirect_uri else "?"
    redirect = redirect_uri + sep + urlencode({"code": code, "state": state})
    logger.info("OAuth authorize: issuing code, redirecting to %s", redirect_uri)
    return RedirectResponse(url=redirect, status_code=302)


async def oauth_token(request: Request) -> JSONResponse:
    """OAuth 2.0 Token Endpoint — exchanges auth code for access token (PKCE)."""
    if request.method == "GET":
        return JSONResponse({"error": "method_not_allowed"}, status_code=405)

    try:
        body = await request.form()
    except Exception:
        body = {}
    body = dict(body)

    grant_type = body.get("grant_type", "")
    code = body.get("code", "")
    code_verifier = body.get("code_verifier", "")
    client_id = body.get("client_id", "")
    redirect_uri = body.get("redirect_uri", "")

    logger.info("OAuth token: grant_type=%s client_id=%s...", grant_type, (client_id or "")[:16])

    if grant_type != "authorization_code":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    _cleanup_expired()
    auth_data = _auth_codes.pop(code, None)
    if not auth_data:
        logger.warning("OAuth token: invalid or expired code")
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    # Verify PKCE code_challenge
    if auth_data.get("code_challenge") and code_verifier:
        digest = hashlib.sha256(code_verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if expected != auth_data["code_challenge"]:
            logger.warning("OAuth token: PKCE verification failed")
            return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)

    # Issue access token
    access_token = secrets.token_urlsafe(48)
    _access_tokens[access_token] = {
        "client_id": auth_data["client_id"],
        "expires": time.time() + 86400,  # 24 hours
    }

    logger.info("OAuth token: issued access token for client_id=%s...", auth_data["client_id"][:16])
    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 86400,
    })


# ---------------------------------------------------------------------------
# ASGI auth middleware (validates Bearer tokens from OAuth flow)
# ---------------------------------------------------------------------------

# Paths that don't require auth
_PUBLIC_PATHS = frozenset([
    "/health",
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
    "/.well-known/oauth-protected-resource/sse",
    "/authorize",
    "/token",
])


class OAuthBearerMiddleware:
    """Pure ASGI middleware — validates OAuth Bearer tokens on protected routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "?")

        # Public routes pass through
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Extract auth header
        headers_raw = scope.get("headers", [])
        header_dict = {k.decode(): v.decode() for k, v in headers_raw}
        auth = header_dict.get("authorization", "")

        logger.info("Request: %s %s host=%s auth=%s",
                     method, path,
                     header_dict.get("host", "-"),
                     auth[:30] + "..." if auth else "-")

        if not auth.startswith("Bearer "):
            logger.warning("Auth rejected for %s %s (no Bearer token)", method, path)
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        token = auth[7:]  # strip "Bearer "

        # Accept the raw configured token OR any valid OAuth-issued token
        _cleanup_expired()
        if token == TERRADEV_MCP_BEARER_TOKEN:
            await self.app(scope, receive, send)
            return

        if token in _access_tokens:
            await self.app(scope, receive, send)
            return

        logger.warning("Auth rejected for %s %s (invalid token)", method, path)
        response = JSONResponse({"error": "unauthorized"}, status_code=401)
        await response(scope, receive, send)


# ---------------------------------------------------------------------------
# SSE app factory
# ---------------------------------------------------------------------------

def create_sse_app() -> "Starlette":
    """Build the Starlette app that exposes the MCP server over SSE."""
    if Starlette is None:
        print(
            "Error: starlette/uvicorn not installed. "
            "Install with: pip install 'mcp[cli]' starlette uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "terradev-mcp.terradev.cloud",
            "localhost:8090",
            "127.0.0.1:8090",
        ],
        allowed_origins=[
            "https://claude.ai",
            "https://www.claude.ai",
            "https://terradev-mcp.terradev.cloud",
        ],
    )
    sse_transport = SseServerTransport("/messages", security_settings=security_settings)

    
    async def handle_messages(request: Request) -> None:
        await sse_transport.handle_post_message(
            request.scope, request.receive, request._send
        )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "server": "terradev-mcp", "version": "2.0.1"})

    # SSE handler wraps the MCP server
    class SseHandler:
        def __init__(self):
            self._server = server
            
        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return
            
            # SSE endpoint only accepts GET requests
            if scope["method"] != "GET":
                from starlette.responses import Response
                response = Response("Method Not Allowed - SSE endpoint only accepts GET", status_code=405)
                await response(scope, receive, send)
                return
            
            async with sse_transport.connect_sse(
                scope, receive, send
            ) as streams:
                await self._server.run(
                    streams[0],
                    streams[1],
                    InitializationOptions(
                        server_name="terradev-mcp",
                        server_version="2.0.1",
                        capabilities=self._server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities=None,
                        ),
                    ),
                )
    
    sse_handler = SseHandler()

    inner_app = Starlette(
        debug=False,
        routes=[
            # OAuth 2.0 endpoints (public — handled before auth middleware)
            Route("/.well-known/oauth-authorization-server", endpoint=oauth_authorization_server_metadata),
            Route("/.well-known/oauth-protected-resource", endpoint=oauth_protected_resource),
            Route("/.well-known/oauth-protected-resource/sse", endpoint=oauth_protected_resource),
            Route("/authorize", endpoint=oauth_authorize),
            Route("/token", endpoint=oauth_token, methods=["POST"]),
            # MCP endpoints
            Route("/health", endpoint=health),
            Route("/sse", endpoint=sse_handler),
            Route("/messages/{path:path}", endpoint=handle_messages, methods=["POST"]),
            Route("/messages", endpoint=handle_messages, methods=["POST"]),
        ],
    )

    app = OAuthBearerMiddleware(inner_app)
    return app


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_stdio():
    """Run in stdio mode (Claude Code / local)."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="terradev-mcp",
                server_version="2.0.1",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )


def main():
    """Main entry point — supports both stdio and SSE transports."""
    parser = argparse.ArgumentParser(description="Terradev MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default, for Claude Code) or sse (remote, for Claude.ai Connectors)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="SSE port (default: 8080)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Check if terradev is installed
    if not check_terradev_installation():
        logger.warning("terradev CLI not found. Tools will fail until installed: pip install terradev-cli")

    if not os.getenv("TERRADEV_RUNPOD_KEY"):
        logger.warning("TERRADEV_RUNPOD_KEY not set. Some functionality may be limited.")

    if args.transport == "stdio":
        asyncio.run(run_stdio())
    else:
        if not TERRADEV_MCP_BEARER_TOKEN:
            logger.warning(
                "TERRADEV_MCP_BEARER_TOKEN is not set — SSE endpoint is UNAUTHENTICATED. "
                "Set this env var in production."
            )
        app = create_sse_app()
        logger.info("Starting Terradev MCP SSE server on %s:%s", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
