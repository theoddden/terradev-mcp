#!/usr/bin/env python3
"""
Terradev MCP Server - GPU Cloud Provisioning for Claude Code

This MCP server provides access to Terradev CLI functionality for GPU provisioning,
price comparison, Kubernetes cluster management, and inference deployment across
11+ cloud providers.
"""

import asyncio
import json
import os
import subprocess
import sys
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

# Execute terradev command safely
async def execute_terradev_command(args: List[str]) -> Dict[str, Any]:
    try:
        cmd = ["terradev"] + args
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
            description="Provision GPU instances across cloud providers",
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
                    "parallel": {
                        "type": "integer",
                        "description": "Parallel provisioning limit",
                        "minimum": 1,
                        "default": 6
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show plan without launching"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price per hour",
                        "minimum": 0
                    },
                    "prefer_spot": {
                        "type": "boolean",
                        "description": "Prefer spot instances for cost savings",
                        "default": True
                    }
                },
                "required": ["gpu_type"]
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
        "provision_gpu": ["provision"],
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
    
    elif tool_name == "provision_gpu":
        cmd_args.extend(["-g", arguments["gpu_type"]])
        if "count" in arguments:
            cmd_args.extend(["-n", str(arguments["count"])])
        if "parallel" in arguments:
            cmd_args.extend(["--parallel", str(arguments["parallel"])])
        if arguments.get("dry_run"):
            cmd_args.append("--dry-run")
        if "max_price" in arguments:
            cmd_args.extend(["--max-price", str(arguments["max_price"])])
        if arguments.get("prefer_spot"):
            cmd_args.append("--prefer-spot")
    
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
