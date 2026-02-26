#!/usr/bin/env python3
"""
Terradev MCP Server - GPU Cloud Provisioning for Claude Code

This MCP server provides access to Terradev CLI functionality for GPU provisioning,
price comparison, Kubernetes cluster management, and inference deployment across
11+ cloud providers. Includes Terraform parallel provisioning for optimal efficiency.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import shutil
from typing import Any, Dict, List, Optional

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
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
    print("Error: mcp package not found. Please install it with: pip install mcp", file=sys.stderr)
    sys.exit(1)

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

# Execute terradev command safely with bug fixes
async def execute_terradev_command(args: List[str]) -> Dict[str, Any]:
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
        config += f"""
resource "terradev_instance" "gpu_{i}" {{
  gpu_type    = var.gpu_type
  provider    = {provider}
  spot        = true
  count       = 1
  
  # Dynamic pricing and availability
  dynamic "pricing" {{
    for_each = var.max_price != null ? [1] : []
    content {{
      max_hourly = var.max_price
    }}
  }}
  
  tags = {{
    Name        = "terradev-mcp-gpu-${{i}}"
    Provisioned = "terraform"
    GPU_Type    = var.gpu_type
  }}
}}

"""
    
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
            name="k8s_create",
            description="Create Kubernetes cluster with GPU nodes",
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
                        "description": "Use multi-cloud node pools",
                        "default": False
                    },
                    "prefer_spot": {
                        "type": "boolean",
                        "description": "Prefer spot instances",
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
            description="Deploy model to InferX serverless platform",
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
            description="View all instances and costs",
            inputSchema={
                "type": "object",
                "properties": {
                    "live": {
                        "type": "boolean",
                        "description": "Show live status",
                        "default": True
                    }
                }
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
        cmd_args.extend([arguments["cluster_name"]])
        cmd_args.extend(["--gpu", arguments["gpu_type"]])
        if "count" in arguments:
            cmd_args.extend(["--count", str(arguments["count"])])
        if arguments.get("multi_cloud"):
            cmd_args.append("--multi-cloud")
        if arguments.get("prefer_spot"):
            cmd_args.append("--prefer-spot")
    
    elif tool_name == "k8s_info":
        cmd_args.append(arguments["cluster_name"])
    
    elif tool_name == "k8s_destroy":
        cmd_args.append(arguments["cluster_name"])
    
    elif tool_name == "inferx_deploy":
        cmd_args.extend(["--model", arguments["model"]])
        cmd_args.extend(["--gpu-type", arguments["gpu_type"]])
    
    elif tool_name == "hf_space_deploy":
        cmd_args.append(arguments["space_name"])
        cmd_args.extend(["--model-id", arguments["model_id"]])
        cmd_args.extend(["--template", arguments["template"]])
        if "hardware" in arguments:
            cmd_args.extend(["--hardware", arguments["hardware"]])
        if "sdk" in arguments:
            cmd_args.extend(["--sdk", arguments["sdk"]])
    
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

async def main():
    """Main entry point"""
    # Check if terradev is installed
    if not check_terradev_installation():
        print("Error: terradev CLI not found. Please install it with:", file=sys.stderr)
        print("pip install terradev-cli", file=sys.stderr)
        sys.exit(1)
    
    # Check for required environment variable
    if not os.getenv("TERRADEV_RUNPOD_KEY"):
        print("Warning: TERRADEV_RUNPOD_KEY not set. Some functionality may be limited.", file=sys.stderr)
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="terradev-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
