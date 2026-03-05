#!/usr/bin/env python3
"""
Terradev MCP Optimizer — Tool Schema Compression + Parallel Dispatch

Reduces token usage by 60-80% when Claude Code enumerates tools, and speeds
up multi-tool turns via asyncio.gather parallel execution.

Two independent layers that can be enabled separately:

1. ToolCompressor
   - Groups 175+ flat tools into ~30 namespace tools with `action` enum
   - Strips optional params (only required params in schema; handler fills defaults)
   - Compresses descriptions to 1-line summaries
   - Example: 8 separate hf_* tools → 1 `hf` tool with action enum

2. ParallelDispatcher
   - When Claude issues N tool calls in one turn, executes all concurrently
   - Falls back to sequential for mutating tools (destroy, delete, apply)
   - TTL + LRU result caching for deterministic read-only tools

Usage:
    from terradev_mcp_optimizer import ToolCompressor, ParallelDispatcher, ResultCache

    compressor = ToolCompressor()
    compressed_tools = compressor.compress(original_tools)  # For list_tools
    original_name, original_args = compressor.expand(compressed_name, compressed_args)  # For call_tool

    dispatcher = ParallelDispatcher(handle_single_tool)
    results = await dispatcher.dispatch_batch(tool_calls)

    cache = ResultCache(ttl=60, maxsize=256)
    cached = cache.get(tool_name, args)
    if cached is None:
        result = await handle(tool_name, args)
        cache.put(tool_name, args, result)
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("terradev-mcp-optimizer")


# ── Namespace Definitions ────────────────────────────────────────────────────
# Maps namespace prefix → (compressed description, list of action suffixes)
# The compressor auto-detects these from tool names using prefix matching.

NAMESPACE_GROUPS: Dict[str, Dict[str, Any]] = {
    "hf": {
        "description": "HuggingFace Hub: models, datasets, endpoints, inference, smart templates, hardware recommendations.",
        "actions": [
            "list_models", "list_datasets", "model_info",
            "create_endpoint", "list_endpoints", "endpoint_info",
            "delete_endpoint", "endpoint_infer",
            "smart_template", "hardware_recommend", "hardware_compare",
            "space_deploy", "space_status",
        ],
    },
    "wandb": {
        "description": "Weights & Biases: projects, runs, dashboards, reports, alerts.",
        "actions": [
            "list_projects", "list_runs", "run_details",
            "create_dashboard", "create_terradev_dashboard",
            "create_report", "create_terradev_report",
            "setup_alerts", "create_terradev_alerts", "dashboard_status",
        ],
    },
    "langsmith": {
        "description": "LangSmith: projects, runs, traces, workspaces, GPU correlation.",
        "actions": [
            "list_projects", "list_runs", "gpu_correlate",
            "create_project", "get_workspaces", "create_trace",
        ],
    },
    "langgraph": {
        "description": "LangGraph: workflow creation, orchestrator-worker, evaluation, status.",
        "actions": [
            "create_workflow", "orchestrator_worker",
            "evaluation_workflow", "workflow_status",
        ],
    },
    "langchain": {
        "description": "LangChain: workflow creation, SGLang pipeline integration.",
        "actions": ["create_workflow", "create_sglang_pipeline"],
    },
    "governance": {
        "description": "Data governance: consent, OPA policy evaluation, data movement, compliance reports.",
        "actions": [
            "request_consent", "record_consent", "evaluate_opa",
            "move_data", "movement_history", "compliance_report",
        ],
    },
    "datadog": {
        "description": "Datadog: status, metrics push, events, monitors, dashboards, queries, Terraform export.",
        "actions": [
            "status", "push_metrics", "send_event",
            "create_monitors", "list_monitors",
            "create_dashboard", "list_dashboards",
            "query", "terraform_export", "metric_catalog",
        ],
    },
    "ray": {
        "description": "Ray cluster: start/stop, job submission, Wide-EP deploy, disaggregated P/D, parallelism strategy.",
        "actions": [
            "status", "start", "stop", "submit_job", "list_jobs",
            "wide_ep_deploy", "disagg_pd_deploy", "parallelism_strategy",
        ],
    },
    "vllm": {
        "description": "vLLM: start/stop server, inference, info, sleep/wake power management.",
        "actions": ["start", "stop", "inference", "info", "sleep", "wake"],
    },
    "sglang": {
        "description": "SGLang: start/stop server, inference, metrics.",
        "actions": ["start", "stop", "inference", "metrics"],
    },
    "ollama": {
        "description": "Ollama: list/pull models, generate, chat, model info.",
        "actions": ["list", "pull", "generate", "chat", "model_info"],
    },
    "k8s": {
        "description": "Kubernetes: create/list/info/destroy clusters, GPU operator, device plugin, MIG, time-slicing, monitoring.",
        "actions": [
            "create", "list", "info", "destroy",
            "gpu_operator_install", "device_plugin",
            "mig_configure", "time_slicing", "monitoring_stack",
        ],
    },
    "inferx": {
        "description": "InferX serverless: deploy/status/list/optimize/configure/delete/usage/quote.",
        "actions": [
            "deploy", "status", "list", "optimize",
            "configure", "delete", "usage", "quote",
        ],
    },
    "terraform": {
        "description": "Terraform: plan/apply/destroy/status for GPU infrastructure.",
        "actions": ["plan", "apply", "destroy", "status"],
    },
    "train": {
        "description": "Training: launch, stop, resume, monitor, status, snapshot, straggler detection, distributed launch, config generation.",
        "actions": [
            "launch", "stop", "resume", "monitor", "status",
            "snapshot", "detect_stragglers",
            "config_generate", "launch_distributed",
        ],
    },
    "checkpoint": {
        "description": "Checkpoints: list, save, restore, promote, delete.",
        "actions": ["list", "save", "restore", "promote", "delete"],
    },
    "gitops": {
        "description": "GitOps: init, bootstrap, sync, validate.",
        "actions": ["init", "bootstrap", "sync", "validate"],
    },
    "orchestrator": {
        "description": "Model orchestrator: start, register, load, evict, status, infer.",
        "actions": ["start", "register", "load", "evict", "status", "infer"],
    },
    "cost": {
        "description": "Cost optimization: analyze, recommend, simulate, budget optimize.",
        "actions": ["analyze", "optimize_recommend", "simulate", "budget_optimize"],
    },
    "price": {
        "description": "Price intelligence: discovery, trends, budget optimize, spot risk, intel.",
        "actions": ["discovery", "trends", "budget_optimize", "spot_risk", "intel"],
    },
    "preflight": {
        "description": "Preflight validation: full check, report, GPU check, network check.",
        "actions": ["check", "report", "gpu_check", "network_check"],
    },
    "egress": {
        "description": "Egress optimization: cheapest route, optimized staging plan.",
        "actions": ["cheapest_route", "optimize_staging"],
    },
    "lora": {
        "description": "LoRA adapters: list, add, remove hot-loaded adapters on vLLM endpoints.",
        "actions": ["list", "add", "remove"],
    },
    "mlflow": {
        "description": "MLflow: list experiments, log runs, register models.",
        "actions": ["list_experiments", "log_run", "register_model"],
    },
    "dvc": {
        "description": "DVC: status, diff, stage checkpoint, push.",
        "actions": ["status", "diff", "stage_checkpoint", "push"],
    },
    "kserve": {
        "description": "KServe: generate YAML, list, status of InferenceServices.",
        "actions": ["generate_yaml", "list", "status"],
    },
    "phoenix": {
        "description": "Arize Phoenix: LLM trace observability — projects, spans, traces, OTEL, K8s.",
        "actions": ["test", "projects", "spans", "trace", "otel_env", "snippet", "k8s"],
    },
    "guardrails": {
        "description": "NeMo Guardrails: LLM output safety — chat, config generation, K8s.",
        "actions": ["test", "chat", "generate_config", "k8s"],
    },
    "qdrant": {
        "description": "Qdrant: vector DB for RAG — collections, create, info, count, K8s.",
        "actions": ["test", "collections", "create_collection", "info", "count", "k8s"],
    },
}

# ── Advanced Meta-Tool Categories ─────────────────────────────────────────────
# Maps category → (description, list of original tool names)
# These tools are consolidated into a single `terradev_advanced` meta-tool.

ADVANCED_TOOL_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "inference": {
        "description": "Routing, failover, disaggregated P/D, deployment.",
        "tools": ["infer_route", "infer_route_disagg", "infer_status", "infer_failover", "infer_deploy"],
    },
    "topology": {
        "description": "GPU NUMA topology, MoE cluster deployment.",
        "tools": ["gpu_topology", "moe_deploy"],
    },
    "deploy": {
        "description": "Provisioning, rollback, manifests, smart deploy, Helm generation.",
        "tools": ["up", "rollback", "manifests", "smart_deploy", "helm_generate"],
    },
    "scaling": {
        "description": "Warm pool and budget-aware cost scaler management.",
        "tools": ["warm_pool_start", "warm_pool_status", "cost_scaler_start", "cost_scaler_status"],
    },
    "data": {
        "description": "Dataset staging, workflow execution.",
        "tools": ["stage", "run_workflow"],
    },
    "discovery": {
        "description": "Local GPU scan, analytics, optimization, provider setup.",
        "tools": ["local_scan", "analytics", "optimize", "setup_provider", "configure_provider"],
    },
}

# Build reverse map: original_tool_name -> (category, tool_name)
_ADVANCED_TOOL_MAP: Dict[str, Tuple[str, str]] = {}
for _cat, _group in ADVANCED_TOOL_CATEGORIES.items():
    for _tool in _group["tools"]:
        _ADVANCED_TOOL_MAP[_tool] = (_cat, _tool)

# Tools that stay as true top-level (never grouped)
UNGROUPED_TOOLS: Set[str] = {
    "quote_gpu", "provision_gpu", "status",
    "manage_instance", "active_context",
}

# Mutating tools that must NOT be parallelized or cached
MUTATING_TOOLS: Set[str] = {
    "provision_gpu", "terraform_apply", "terraform_destroy",
    "k8s_create", "k8s_destroy", "inferx_deploy", "inferx_delete",
    "hf_create_endpoint", "hf_delete_endpoint",
    "train", "train_stop", "train_resume", "training_launch_distributed",
    "checkpoint_save", "checkpoint_restore", "checkpoint_promote", "checkpoint_delete",
    "manage_instance", "up", "rollback", "smart_deploy",
    "orchestrator_start", "orchestrator_register", "orchestrator_load", "orchestrator_evict",
    "warm_pool_start", "cost_scaler_start",
    "vllm_start", "vllm_stop", "vllm_sleep", "vllm_wake",
    "sglang_start", "sglang_stop",
    "ollama_pull",
    "ray_start", "ray_stop",
    "lora_add", "lora_remove",
    "gitops_init", "gitops_bootstrap", "gitops_sync",
    "governance_record_consent", "governance_move_data",
    "datadog_create_monitors", "datadog_create_dashboard", "datadog_send_event",
    "datadog_push_metrics", "datadog_terraform_export",
    "dvc_stage_checkpoint", "dvc_push",
    "stage", "infer_deploy",
    "langsmith_create_project", "langsmith_create_trace",
    "wandb_create_dashboard", "wandb_create_terradev_dashboard",
    "wandb_create_report", "wandb_create_terradev_report",
    "wandb_setup_alerts", "wandb_create_terradev_alerts",
    "k8s_gpu_operator_install", "k8s_device_plugin",
    "k8s_mig_configure", "k8s_time_slicing", "k8s_monitoring_stack",
    "mlflow_log_run", "mlflow_register_model",
    "guardrails_chat", "guardrails_generate_config",
    "qdrant_create_collection",
    "phoenix_k8s", "guardrails_k8s", "qdrant_k8s",
}


# ── Tool Name Mapping ────────────────────────────────────────────────────────

def _build_name_maps() -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    """Build bidirectional maps: original_name -> (namespace, action) and (namespace, action) -> original_name.

    Returns:
        flat_to_ns: original tool name -> "namespace.action" (for compression)
        ns_to_flat: (namespace, action) -> original tool name (for expansion)
    """
    flat_to_ns: Dict[str, str] = {}
    ns_to_flat: Dict[Tuple[str, str], str] = {}

    for ns, group in NAMESPACE_GROUPS.items():
        for action in group["actions"]:
            # Try common naming patterns to find the original flat tool name
            candidates = [
                f"{ns}_{action}",                    # hf_list_models
                f"{ns}_{action}".replace("_", ""),   # unlikely but safe
            ]
            # Special cases where the original name differs from ns_action
            special_map = {
                ("hf", "space_deploy"): "hf_space_deploy",
                ("hf", "space_status"): "hf_space_status",
                ("train", "launch"): "train",
                ("train", "status"): "train_status",
                ("train", "monitor"): "train_monitor",
                ("train", "config_generate"): "training_config_generate",
                ("train", "launch_distributed"): "training_launch_distributed",
                ("checkpoint", "list"): "checkpoint_list",
                ("checkpoint", "save"): "checkpoint_save",
                ("checkpoint", "restore"): "checkpoint_restore",
                ("checkpoint", "promote"): "checkpoint_promote",
                ("checkpoint", "delete"): "checkpoint_delete",
                ("preflight", "check"): "preflight",
                ("price", "discovery"): "price_discovery",
                ("price", "intel"): "price_intel",
                ("cost", "optimize_recommend"): "cost_optimize_recommend",
                ("egress", "cheapest_route"): "egress_cheapest_route",
                ("egress", "optimize_staging"): "egress_optimize_staging",
            }

            original = special_map.get((ns, action))
            if original is None:
                original = f"{ns}_{action}"

            flat_to_ns[original] = f"{ns}.{action}"
            ns_to_flat[(ns, action)] = original

    return flat_to_ns, ns_to_flat


_FLAT_TO_NS, _NS_TO_FLAT = _build_name_maps()


# ── ToolCompressor ───────────────────────────────────────────────────────────


class ToolCompressor:
    """
    Compresses 175+ flat MCP tools into ~30 namespace tools + ungrouped tools.

    Compression strategies:
    1. Namespace grouping: hf_list_models, hf_model_info, ... → hf(action=...)
    2. Description shortening: 1-line per tool
    3. Optional param stripping: only required params in schema

    Token savings: ~60-80% of tool schema tokens per list_tools call.
    """

    def __init__(self, enable_compression: bool = True, strip_optional: bool = True):
        self.enabled = enable_compression
        self.strip_optional = strip_optional
        # Original Tool objects keyed by name (populated on first compress() call)
        self._original_tools: Dict[str, Any] = {}
        # Merged schemas for namespace tools
        self._ns_schemas: Dict[str, Any] = {}

    def compress(self, tools: List[Any]) -> List[Any]:
        """Compress a list of Tool objects into namespace-grouped tools.

        Args:
            tools: Original list of mcp.types.Tool objects from handle_list_tools

        Returns:
            Compressed list of Tool objects (typically ~30 instead of 175+)
        """
        if not self.enabled:
            return tools

        from mcp.types import Tool

        # Index originals
        self._original_tools = {t.name: t for t in tools}

        compressed = []
        seen_ns = set()
        ungrouped_seen = set()
        advanced_emitted = False

        for tool in tools:
            name = tool.name

            # Check if this tool belongs to a namespace group
            if name in _FLAT_TO_NS and name not in UNGROUPED_TOOLS and name not in _ADVANCED_TOOL_MAP:
                ns = _FLAT_TO_NS[name].split(".")[0]
                if ns in seen_ns:
                    continue  # Already emitted this namespace tool
                seen_ns.add(ns)

                # Build the namespace tool
                ns_tool = self._build_namespace_tool(ns, Tool)
                if ns_tool:
                    compressed.append(ns_tool)
            elif name in _ADVANCED_TOOL_MAP:
                # Absorb into the terradev_advanced meta-tool
                if not advanced_emitted:
                    advanced_emitted = True
                    meta_tool = self._build_advanced_meta_tool(Tool)
                    if meta_tool:
                        compressed.append(meta_tool)
            elif name in UNGROUPED_TOOLS or name not in _FLAT_TO_NS:
                if name not in ungrouped_seen:
                    ungrouped_seen.add(name)
                    if self.strip_optional:
                        compressed.append(self._strip_optional_params(tool, Tool))
                    else:
                        compressed.append(tool)

        logger.info(
            f"ToolCompressor: {len(tools)} tools → {len(compressed)} "
            f"({len(seen_ns)} namespaces + {len(ungrouped_seen)} ungrouped"
            f"{' + 1 meta-tool' if advanced_emitted else ''})"
        )
        return compressed

    def expand(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Expand a compressed namespace tool call back to the original flat tool name.

        Args:
            tool_name: Compressed tool name (e.g., "hf" or original "quote_gpu")
            arguments: Arguments dict, may contain "action" key for namespace tools

        Returns:
            (original_tool_name, original_arguments) with "action"/"category" keys removed
        """
        if not self.enabled:
            return tool_name, arguments

        # Check if it's the advanced meta-tool
        if tool_name == "terradev_advanced":
            category = arguments.get("category")
            tool = arguments.get("tool")
            if not category or not tool:
                raise ValueError(
                    "terradev_advanced requires 'category' and 'tool' parameters. "
                    f"Categories: {list(ADVANCED_TOOL_CATEGORIES.keys())}"
                )
            cat_group = ADVANCED_TOOL_CATEGORIES.get(category)
            if not cat_group:
                raise ValueError(
                    f"Unknown category '{category}'. "
                    f"Valid: {list(ADVANCED_TOOL_CATEGORIES.keys())}"
                )
            if tool not in cat_group["tools"]:
                raise ValueError(
                    f"Unknown tool '{tool}' in category '{category}'. "
                    f"Valid: {cat_group['tools']}"
                )
            # Remove routing keys, pass remaining args to original handler
            expanded_args = {k: v for k, v in arguments.items() if k not in ("category", "tool")}
            return tool, expanded_args

        # Check if it's a namespace tool
        if tool_name in NAMESPACE_GROUPS:
            action = arguments.get("action")
            if not action:
                raise ValueError(f"Namespace tool '{tool_name}' requires 'action' parameter")
            original_name = _NS_TO_FLAT.get((tool_name, action))
            if not original_name:
                raise ValueError(
                    f"Unknown action '{action}' for namespace '{tool_name}'. "
                    f"Valid: {NAMESPACE_GROUPS[tool_name]['actions']}"
                )
            # Remove 'action' from arguments before passing to original handler
            expanded_args = {k: v for k, v in arguments.items() if k != "action"}
            return original_name, expanded_args

        # Not a namespace tool — pass through unchanged
        return tool_name, arguments

    def _build_advanced_meta_tool(self, ToolClass: type) -> Any:
        """Build the terradev_advanced meta-tool from ADVANCED_TOOL_CATEGORIES.

        Emits a single tool with category + tool enum routing, and a merged
        superset of all child tool properties.
        """
        # Build category descriptions for the enum
        category_enum = list(ADVANCED_TOOL_CATEGORIES.keys())
        category_desc_parts = [f"{c}: {g['description']}" for c, g in ADVANCED_TOOL_CATEGORIES.items()]

        # Build tool enum per category
        all_tool_names: List[str] = []
        for cat_group in ADVANCED_TOOL_CATEGORIES.values():
            all_tool_names.extend(cat_group["tools"])

        # Merge all child tool properties into a union schema
        all_properties: Dict[str, Any] = {
            "category": {
                "type": "string",
                "description": "Category: " + "; ".join(category_desc_parts),
                "enum": category_enum,
            },
            "tool": {
                "type": "string",
                "description": "Tool name within category.",
                "enum": sorted(set(all_tool_names)),
            },
        }

        for tool_name in all_tool_names:
            if tool_name not in self._original_tools:
                continue
            orig_schema = self._original_tools[tool_name].inputSchema
            if not isinstance(orig_schema, dict):
                continue
            for prop_name, prop_def in orig_schema.get("properties", {}).items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = prop_def

        # Build the description from child tool descriptions
        child_descs = []
        for cat, cat_group in ADVANCED_TOOL_CATEGORIES.items():
            tools_in_cat = cat_group["tools"]
            tool_summaries = []
            for tn in tools_in_cat:
                orig = self._original_tools.get(tn)
                if orig:
                    tool_summaries.append(f"{tn}: {orig.description}")
            child_descs.append(f"[{cat}] {', '.join(t for t in tools_in_cat)}")

        description = (
            "Advanced Terradev operations. Categories: "
            + "; ".join(child_descs)
            + "."
        )

        return ToolClass(
            name="terradev_advanced",
            description=description,
            inputSchema={
                "type": "object",
                "properties": all_properties,
                "required": ["category", "tool"],
            },
        )

    def _build_namespace_tool(self, ns: str, ToolClass: type) -> Optional[Any]:
        """Build a single namespace Tool from all tools in the group."""
        group = NAMESPACE_GROUPS.get(ns)
        if not group:
            return None

        actions = group["actions"]
        description = group["description"]

        # Merge all required properties from child tools into a union schema
        all_properties: Dict[str, Any] = {
            "action": {
                "type": "string",
                "description": f"Action to perform",
                "enum": actions,
            }
        }
        all_required: Set[str] = {"action"}

        for action in actions:
            original_name = _NS_TO_FLAT.get((ns, action))
            if not original_name or original_name not in self._original_tools:
                continue

            orig_schema = self._original_tools[original_name].inputSchema
            if not isinstance(orig_schema, dict):
                continue

            for prop_name, prop_def in orig_schema.get("properties", {}).items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = prop_def
            # Only "action" is required at the namespace level
            # Sub-tool validation happens in the original handler

        schema = {
            "type": "object",
            "properties": all_properties,
            "required": list(all_required),
        }

        self._ns_schemas[ns] = schema

        return ToolClass(
            name=ns,
            description=description,
            inputSchema=schema,
        )

    def _strip_optional_params(self, tool: Any, ToolClass: type) -> Any:
        """Return a copy of the tool with optional params removed from schema."""
        schema = tool.inputSchema
        if not isinstance(schema, dict):
            return tool

        required = set(schema.get("required", []))
        properties = schema.get("properties", {})

        if not required or len(properties) <= len(required):
            return tool  # Nothing to strip

        stripped_props = {k: v for k, v in properties.items() if k in required}

        return ToolClass(
            name=tool.name,
            description=tool.description,
            inputSchema={
                "type": "object",
                "properties": stripped_props,
                "required": list(required),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        return {
            "original_tool_count": len(self._original_tools),
            "namespace_count": len(NAMESPACE_GROUPS),
            "ungrouped_count": len(UNGROUPED_TOOLS),
            "flat_to_ns_mappings": len(_FLAT_TO_NS),
        }


# ── Result Cache ─────────────────────────────────────────────────────────────


class ResultCache:
    """
    TTL + LRU cache for deterministic, read-only tool results.

    Automatically skips caching for mutating tools.
    Cache key = sha256(tool_name + sorted(args)).
    """

    def __init__(self, ttl: int = 60, maxsize: int = 256):
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (expires_at, result)
        self._access_order: List[str] = []  # LRU tracking

    def _make_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        raw = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        """Get cached result, or None if miss/expired/mutating."""
        if tool_name in MUTATING_TOOLS:
            return None

        key = self._make_key(tool_name, args)
        entry = self._cache.get(key)
        if entry is None:
            return None

        expires_at, result = entry
        if time.time() > expires_at:
            # Expired - clean up
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return None

        # Update LRU order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        return result

    def put(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Cache a result for a read-only tool."""
        if tool_name in MUTATING_TOOLS:
            return

        key = self._make_key(tool_name, args)
        expires_at = time.time() + self.ttl

        # Evict if at capacity
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest = self._access_order[0]
            del self._cache[oldest]
            self._access_order.remove(oldest)

        self._cache[key] = (expires_at, result)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "mutating_tools_excluded": len(MUTATING_TOOLS),
        }


# ── Parallel Dispatcher ───────────────────────────────────────────────────────


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None  # For correlation


@dataclass
class BatchResult:
    tool_id: Optional[str]
    name: str
    result: Any
    error: Optional[Exception] = None
    duration_ms: Optional[float] = None


class ParallelDispatcher:
    """
    Executes multiple tool calls concurrently when safe.

    - Read-only tools run in parallel via asyncio.gather
    - Mutating tools run sequentially to preserve order
    - Optional result caching for deterministic read-only tools
    """

    def __init__(
        self,
        handler: Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]],
        cache: Optional[ResultCache] = None,
        enable_parallel: bool = True,
    ):
        self.handler = handler
        self.cache = cache or ResultCache()
        self.enable_parallel = enable_parallel

    async def dispatch_batch(self, calls: List[ToolCall]) -> List[BatchResult]:
        """Dispatch a batch of tool calls with optimal parallelism."""
        if not self.enable_parallel or len(calls) <= 1:
            # Sequential fallback
            results = []
            for call in calls:
                start = time.time()
                try:
                    result = await self._handle_single(call)
                    duration_ms = (time.time() - start) * 1000
                    results.append(BatchResult(call.id, call.name, result, duration_ms=duration_ms))
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    results.append(BatchResult(call.id, call.name, None, error=e, duration_ms=duration_ms))
            return results

        # Separate read-only and mutating calls
        readonly_calls = []
        mutating_calls = []
        for call in calls:
            if call.name in MUTATING_TOOLS:
                mutating_calls.append(call)
            else:
                readonly_calls.append(call)

        # Execute read-only calls in parallel
        readonly_results = []
        if readonly_calls:
            readonly_coros = [self._handle_single(call) for call in readonly_calls]
            readonly_results_raw = await asyncio.gather(*readonly_coros, return_exceptions=True)
            readonly_results = []
            for call, result in zip(readonly_calls, readonly_results_raw):
                if isinstance(result, Exception):
                    readonly_results.append(BatchResult(call.id, call.name, None, error=result))
                else:
                    readonly_results.append(BatchResult(call.id, call.name, result))

        # Execute mutating calls sequentially (preserve order)
        mutating_results = []
        for call in mutating_calls:
            start = time.time()
            try:
                result = await self.handler(call.name, call.arguments)
                duration_ms = (time.time() - start) * 1000
                mutating_results.append(BatchResult(call.id, call.name, result, duration_ms=duration_ms))
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                mutating_results.append(BatchResult(call.id, call.name, None, error=e, duration_ms=duration_ms))

        # Combine results preserving original order
        readonly_idx = 0
        mutating_idx = 0
        combined = []
        for call in calls:
            if call.name in MUTATING_TOOLS:
                combined.append(mutating_results[mutating_idx])
                mutating_idx += 1
            else:
                combined.append(readonly_results[readonly_idx])
                readonly_idx += 1

        logger.info(
            f"ParallelDispatcher: {len(calls)} calls → "
            f"{len(readonly_calls)} parallel + {len(mutating_calls)} sequential"
        )
        return combined

    async def _handle_single(self, call: ToolCall) -> Any:
        """Handle a single tool call with optional caching."""
        # Check cache first for read-only tools
        if call.name not in MUTATING_TOOLS:
            cached = self.cache.get(call.name, call.arguments)
            if cached is not None:
                logger.debug(f"Cache hit for {call.name}")
                return cached

        # Execute handler
        result = await self.handler(call.name, call.arguments)

        # Cache result for read-only tools
        if call.name not in MUTATING_TOOLS:
            self.cache.put(call.name, call.arguments, result)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return dispatcher statistics."""
        return {
            "parallel_enabled": self.enable_parallel,
            "mutating_tool_count": len(MUTATING_TOOLS),
            "cache_stats": self.cache.get_stats(),
        }


# ── Convenience Wrapper ─────────────────────────────────────────────────────


class MCPOptimizer:
    """
    High-level wrapper that combines compression and parallel dispatch.
    """

    def __init__(
        self,
        enable_compression: bool = True,
        strip_optional: bool = True,
        enable_parallel: bool = True,
        cache_ttl: int = 60,
        cache_maxsize: int = 256,
    ):
        self.compressor = ToolCompressor(enable_compression, strip_optional)
        self.cache = ResultCache(cache_ttl, cache_maxsize)
        self.parallel_enabled = enable_parallel

    def compress_tools(self, tools: List[Any]) -> List[Any]:
        """Compress tool list for list_tools response."""
        return self.compressor.compress(tools)

    def expand_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Expand compressed tool call for handler."""
        return self.compressor.expand(tool_name, arguments)

    def make_dispatcher(self, handler: Callable) -> ParallelDispatcher:
        """Create a parallel dispatcher with the given handler."""
        return ParallelDispatcher(handler, self.cache, self.parallel_enabled)

    def get_stats(self) -> Dict[str, Any]:
        """Return combined statistics."""
        return {
            "compressor": self.compressor.get_stats(),
            "cache": self.cache.get_stats(),
            "parallel_enabled": self.parallel_enabled,
        }


# ── CLI Debug Helper ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Terradev MCP Optimizer Debug")
    parser.add_argument("--stats", action="store_true", help="Show compression and cache stats")
    parser.add_argument("--list-namespaces", action="store_true", help="List all namespace groups")
    parser.add_argument("--list-mutating", action="store_true", help="List all mutating tools")
    args = parser.parse_args()

    if args.stats:
        opt = MCPOptimizer()
        stats = opt.get_stats()
        print(json.dumps(stats, indent=2))
    elif args.list_namespaces:
        for ns, group in NAMESPACE_GROUPS.items():
            print(f"{ns}: {len(group['actions'])} actions")
            for action in group["actions"]:
                original = _NS_TO_FLAT.get((ns, action), "MISSING")
                print(f"  - {action} → {original}")
    elif args.list_mutating:
        print(f"Mutating tools ({len(MUTATING_TOOLS)}):")
        for tool in sorted(MUTATING_TOOLS):
            print(f"  - {tool}")
    else:
        parser.print_help()
        sys.exit(0)
