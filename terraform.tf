terraform {
  required_version = ">= 1.0"
  required_providers {
    terradev = {
      source  = "theoddden/terradev"
      version = "~> 3.0"
    }
  }
}

# Provider configuration for optimal parallel provisioning
provider "terradev" {
  # Configuration will be handled by environment variables
  # TERRADEV_RUNPOD_KEY, TERRADEV_AWS_ACCESS_KEY_ID, etc.
}

# Example GPU instance resource
resource "terradev_instance" "gpu" {
  gpu_type = var.gpu_type
  spot     = true
  count    = var.gpu_count
  
  dynamic "pricing" {
    for_each = var.max_price != null ? [1] : []
    content {
      max_hourly = var.max_price
    }
  }
  
  tags = {
    Name        = "terradev-mcp-gpu"
    Provisioned = "terraform"
    GPU_Type    = var.gpu_type
  }
}

# Variables
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
  default     = 1
  
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

# Outputs
output "instance_ids" {
  description = "Provisioned instance IDs"
  value       = terradev_instance.gpu[*].id
}

output "instance_ips" {
  description = "Instance IP addresses"
  value       = terradev_instance.gpu[*].public_ip
}

output "hourly_costs" {
  description = "Hourly costs per instance"
  value       = terradev_instance.gpu[*].hourly_cost
}
