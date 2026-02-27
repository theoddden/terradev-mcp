#!/usr/bin/env python3
"""
Terradev MCP Server - GPU Cloud Provisioning for Claude Code

This MCP server provides access to Terradev CLI functionality for GPU provisioning,
price comparison, Kubernetes cluster management, and inference deployment across
11+ cloud providers. Includes Terraform parallel provisioning for optimal efficiency.
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
        TextContent,
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
        providers = ["runpod", "vastai", "lambda", "aws"]
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
    
    providers = providers or ["runpod", "vastai", "lambda", "aws"]
    
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
            description="Get real-time GPU prices across all cloud providers",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type (H100, A100, A10G, L40S, L4, T4, RTX4090, RTX3090, V100)",
                        "enum": ["H100", "A100", "A10G", "L40S", "L4", "T4", "RTX4090", "RTX3090", "V100"]
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
                            "enum": ["runpod", "vastai", "lambda", "aws", "gcp", "azure", "coreweave", "tensordock", "oracle", "crusoe", "digitalocean", "hyperstack"]
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
                        "enum": ["runpod", "aws", "vastai", "gcp", "azure", "lambda", "coreweave", "tensordock", "oracle", "crusoe", "digitalocean", "hyperstack"]
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
                        "enum": ["runpod", "aws", "vastai", "gcp", "azure", "lambda", "coreweave", "tensordock", "oracle", "crusoe", "digitalocean", "hyperstack"]
                    }
                },
                "required": ["provider"]
            }
        )
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
        "configure_provider": ["configure"]
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
    
    # Execute the command
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
    """List available resources"""
    return ListResourcesResult(resources=[])

@server.read_resource()
async def handle_read_resource(request: ReadResourceRequest) -> ReadResourceResult:
    """Read a resource"""
    return ReadResourceResult(contents=[])

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
        return JSONResponse({"status": "ok", "server": "terradev-mcp", "version": "1.2.2"})

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
                        server_version="1.2.2",
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
                server_version="1.2.2",
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
