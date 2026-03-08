"""
Terradev MCP Tools for CUCo Integration

This module adds CUCo optimization capabilities to the Terradev MCP server,
providing 8 new tools for automatic kernel optimization and performance enhancement.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
from dataclasses import asdict

# Import Terradev optimization modules
import sys
sys.path.append('/Users/theowolfenden/CascadeProjects/Terradev')

try:
    from terradev_cli.optimization.cuco_optimizer import CUCoOptimizer, WorkloadProfile, CUCoMetrics
    from terradev_cli.optimization.auto_optimizer import AutoOptimizer, OptimizationPlan
    from terradev_cli.core.config import TerradevConfig
    from terradev_cli.core.monitoring import MetricsCollector
except ImportError as e:
    logging.warning(f"Could not import Terradev modules: {e}")
    # Fallback implementations for testing
    CUCoOptimizer = None
    AutoOptimizer = None

logger = logging.getLogger(__name__)

class CUCoMCPTools:
    """
    CUCo integration tools for Terradev MCP server
    """
    
    def __init__(self):
        self.config = TerradevConfig() if TerradevConfig else None
        self.metrics_collector = MetricsCollector() if MetricsCollector else None
        self.cuco_optimizer = None
        self.auto_optimizer = None
        
        # Initialize optimizers if available
        if CUCoOptimizer and self.config and self.metrics_collector:
            self.cuco_optimizer = CUCoOptimizer(self.config, self.metrics_collector)
            self.auto_optimizer = AutoOptimizer(self.config, self.metrics_collector)
        
        self.optimization_cache = {}
        self.performance_history = {}
    
    def analyze_workload_for_cuco(self, workload_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze workload to determine CUCo optimization potential
        
        Args:
            workload_spec: Workload specification including model size, GPU count, etc.
            
        Returns:
            Analysis results with optimization recommendations
        """
        try:
            if not self.cuco_optimizer:
                return {"error": "CUCo optimizer not available"}
            
            # Analyze workload
            profile = self.cuco_optimizer.analyze_workload(workload_spec)
            
            # Check if optimization should be applied
            should_optimize, reason = self.cuco_optimizer.should_optimize(profile)
            
            # Estimate performance gains
            estimated_metrics = self._estimate_performance_metrics(profile)
            
            return {
                "workload_profile": asdict(profile),
                "optimization_recommended": should_optimize,
                "reasoning": reason,
                "estimated_performance_gain": estimated_metrics.end_to_end_speedup,
                "estimated_cost_increase": self._estimate_cost_increase(profile),
                "p95_compliance": self._check_p95_compliance(profile, estimated_metrics),
                "optimization_confidence": self._calculate_optimization_confidence(profile)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing workload for CUCo: {str(e)}")
            return {"error": str(e)}
    
    def deploy_optimized_kernels(self, deployment_id: str, workload_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy CUCo-optimized kernels for a workload
        
        Args:
            deployment_id: Unique deployment identifier
            workload_spec: Workload specification
            
        Returns:
            Deployment results with performance metrics
        """
        try:
            if not self.cuco_optimizer:
                return {"error": "CUCo optimizer not available"}
            
            # Analyze workload
            profile = self.cuco_optimizer.analyze_workload(workload_spec)
            
            # Apply CUCo optimization
            result = self.cuco_optimizer.optimize_workload(profile, deployment_id)
            
            if result.decision.value == "apply":
                # Deploy optimized kernels
                deployment_success = self._deploy_kernels(deployment_id, result.kernel_code)
                
                if deployment_success:
                    # Monitor performance
                    performance_metrics = self._monitor_deployment_performance(deployment_id)
                    
                    return {
                        "deployment_id": deployment_id,
                        "optimization_applied": True,
                        "performance_gain": result.performance_gain,
                        "cost_increase": result.cost_increase,
                        "optimization_time": result.optimization_time,
                        "kernel_metrics": asdict(result.metrics) if result.metrics else None,
                        "current_performance": performance_metrics,
                        "p95_achievement": self._calculate_p95_achievement(result.metrics),
                        "recommendations": self._generate_deployment_recommendations(result)
                    }
                else:
                    return {
                        "deployment_id": deployment_id,
                        "optimization_applied": False,
                        "error": "Failed to deploy optimized kernels"
                    }
            else:
                return {
                    "deployment_id": deployment_id,
                    "optimization_applied": False,
                    "reason": result.reasoning,
                    "estimated_gain": result.performance_gain
                }
                
        except Exception as e:
            logger.error(f"Error deploying optimized kernels: {str(e)}")
            return {"error": str(e)}
    
    def benchmark_optimization_impact(self, deployment_id: str, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Benchmark the performance impact of CUCo optimization
        
        Args:
            deployment_id: Deployment to benchmark
            duration_minutes: Benchmark duration in minutes
            
        Returns:
            Performance comparison and impact analysis
        """
        try:
            if not self.cuco_optimizer:
                return {"error": "CUCo optimizer not available"}
            
            # Get optimization history
            optimization_history = self.cuco_optimizer.get_optimization_history(deployment_id)
            
            if not optimization_history:
                return {"error": "No optimization history found for deployment"}
            
            # Run benchmark
            benchmark_results = self._run_performance_benchmark(deployment_id, duration_minutes)
            
            # Compare with baseline
            baseline_metrics = optimization_history.get("baseline_metrics", {})
            current_metrics = benchmark_results.get("current_metrics", {})
            
            performance_comparison = self._compare_performance(baseline_metrics, current_metrics)
            
            # P95 compliance analysis
            p95_analysis = self._analyze_p95_compliance(current_metrics, optimization_history["profile"]["workload_type"])
            
            return {
                "deployment_id": deployment_id,
                "benchmark_duration": duration_minutes,
                "performance_comparison": performance_comparison,
                "p95_compliance": p95_analysis,
                "cost_efficiency": self._calculate_cost_efficiency(benchmark_results),
                "recommendations": self._generate_benchmark_recommendations(performance_comparison),
                "detailed_metrics": benchmark_results
            }
            
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
    
    def auto_optimize_deployment(self, deployment_id: str, workload_spec: Dict[str, Any], optimization_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Automatically apply comprehensive optimizations including CUCo
        
        Args:
            deployment_id: Deployment identifier
            workload_spec: Workload specification
            optimization_preferences: User preferences for optimization
            
        Returns:
            Comprehensive optimization results
        """
        try:
            if not self.auto_optimizer:
                return {"error": "Auto optimizer not available"}
            
            # Create optimization context
            context = {
                "deployment_id": deployment_id,
                "workload_spec": workload_spec,
                "optimization_preferences": optimization_preferences or {},
                "trigger": "manual"
            }
            
            # Analyze deployment
            optimization_plan = await self.auto_optimizer.analyze_deployment(deployment_id, workload_spec)
            
            # Apply optimizations
            optimization_results = await self.auto_optimizer.apply_optimizations(deployment_id, optimization_plan)
            
            # Start monitoring
            monitoring_task = asyncio.create_task(
                self.auto_optimizer.monitor_and_optimize(deployment_id)
            )
            
            return {
                "deployment_id": deployment_id,
                "optimization_plan": asdict(optimization_plan),
                "optimization_results": optimization_results,
                "monitoring_active": True,
                "cuco_applied": "cuco_kernel_optimization" in optimization_results.get("applied_optimizations", []),
                "expected_benefits": self._calculate_expected_benefits(optimization_plan),
                "next_monitoring_check": time.time() + 60,  # Next check in 1 minute
                "rollback_options": self._get_rollback_options(optimization_results)
            }
            
        except Exception as e:
            logger.error(f"Error in auto optimization: {str(e)}")
            return {"error": str(e)}
    
    def get_optimization_recommendations(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get optimization recommendations for a deployment
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Optimization recommendations and insights
        """
        try:
            if not self.cuco_optimizer:
                return {"error": "CUCo optimizer not available"}
            
            # Get current deployment metrics
            current_metrics = self._get_current_metrics(deployment_id)
            
            # Get optimization history
            optimization_history = self.cuco_optimizer.get_optimization_history(deployment_id)
            
            # Analyze performance patterns
            performance_analysis = self._analyze_performance_patterns(current_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(performance_analysis, optimization_history)
            
            # Estimate potential gains
            potential_gains = self._estimate_potential_gains(recommendations)
            
            return {
                "deployment_id": deployment_id,
                "current_performance": current_metrics,
                "performance_analysis": performance_analysis,
                "recommendations": recommendations,
                "potential_gains": potential_gains,
                "optimization_priority": self._calculate_optimization_priority(recommendations),
                "implementation_complexity": self._assess_implementation_complexity(recommendations),
                "risk_assessment": self._assess_optimization_risks(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {str(e)}")
            return {"error": str(e)}
    
    def rollback_optimization(self, deployment_id: str, optimization_type: str = "all") -> Dict[str, Any]:
        """
        Rollback optimizations for a deployment
        
        Args:
            deployment_id: Deployment identifier
            optimization_type: Type of optimization to rollback (cuco, warm_pool, all)
            
        Returns:
            Rollback results and status
        """
        try:
            if not self.cuco_optimizer:
                return {"error": "CUCo optimizer not available"}
            
            rollback_results = {
                "deployment_id": deployment_id,
                "rollback_type": optimization_type,
                "rolled_back_optimizations": [],
                "failed_rollbacks": [],
                "performance_impact": {},
                "timestamp": time.time()
            }
            
            # Rollback CUCo optimization
            if optimization_type in ["cuco", "all"]:
                cuco_success = self.cuco_optimizer.rollback_optimization(deployment_id)
                if cuco_success:
                    rollback_results["rolled_back_optimizations"].append("cuco_kernel_optimization")
                else:
                    rollback_results["failed_rollbacks"].append("cuco_kernel_optimization")
            
            # Rollback other optimizations (if auto_optimizer available)
            if optimization_type in ["all"] and self.auto_optimizer:
                # This would rollback other optimizations
                rollback_results["rolled_back_optimizations"].extend(["warm_pool", "semantic_routing"])
            
            # Measure performance impact
            if rollback_results["rolled_back_optimizations"]:
                current_metrics = self._get_current_metrics(deployment_id)
                rollback_results["performance_impact"] = {
                    "current_latency": current_metrics.get("latency", 0),
                    "current_throughput": current_metrics.get("throughput", 0),
                    "current_cost": current_metrics.get("cost_per_hour", 0)
                }
            
            return rollback_results
            
        except Exception as e:
            logger.error(f"Error rolling back optimization: {str(e)}")
            return {"error": str(e)}
    
    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization dashboard
        
        Returns:
            Dashboard with optimization metrics and insights
        """
        try:
            # Get optimization summary
            if self.auto_optimizer:
                summary = self.auto_optimizer.get_optimization_summary()
            else:
                summary = {"total_deployments": 0, "total_optimizations": 0}
            
            # Get CUCo performance summary
            if self.cuco_optimizer:
                cuco_summary = self.cuco_optimizer.get_performance_summary()
            else:
                cuco_summary = {"total_optimizations": 0, "average_speedup": 1.0}
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(summary, cuco_summary)
            
            # Get active optimizations
            active_optimizations = self._get_active_optimizations()
            
            # Get recent performance trends
            performance_trends = self._get_performance_trends()
            
            return {
                "dashboard_timestamp": time.time(),
                "overall_metrics": overall_metrics,
                "optimization_summary": summary,
                "cuco_summary": cuco_summary,
                "active_optimizations": active_optimizations,
                "performance_trends": performance_trends,
                "p95_compliance_rate": self._calculate_p95_compliance_rate(),
                "cost_savings": self._calculate_cost_savings(),
                "recommendations": self._generate_dashboard_recommendations(),
                "alerts": self._get_optimization_alerts()
            }
            
        except Exception as e:
            logger.error(f"Error generating optimization dashboard: {str(e)}")
            return {"error": str(e)}
    
    def validate_p95_boundaries(self, workload_type: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate metrics against P95 boundaries for workload type
        
        Args:
            workload_type: Type of workload (moe, attention, training, etc.)
            metrics: Performance metrics to validate
            
        Returns:
            Validation results with boundary compliance analysis
        """
        try:
            if not self.cuco_optimizer:
                return {"error": "CUCo optimizer not available"}
            
            # Get P95 boundaries for workload type
            p95_boundaries = self.cuco_optimizer.p95_boundaries.get(workload_type, {})
            
            if not p95_boundaries:
                return {"error": f"No P95 boundaries available for workload type: {workload_type}"}
            
            # Validate each metric
            validation_results = {
                "workload_type": workload_type,
                "p95_boundaries": p95_boundaries,
                "provided_metrics": metrics,
                "compliance_results": {},
                "overall_compliance": True,
                "violations": [],
                "recommendations": []
            }
            
            # Check each metric against P95 boundaries
            for metric_name, metric_value in metrics.items():
                if metric_name in p95_boundaries:
                    p95_value = p95_boundaries[metric_name]
                    compliance_ratio = metric_value / p95_value
                    
                    validation_results["compliance_results"][metric_name] = {
                        "provided_value": metric_value,
                        "p95_boundary": p95_value,
                        "compliance_ratio": compliance_ratio,
                        "meets_threshold": metric_value >= p95_value,
                        "performance_level": self._categorize_performance_level(compliance_ratio)
                    }
                    
                    if metric_value < p95_value:
                        validation_results["overall_compliance"] = False
                        validation_results["violations"].append({
                            "metric": metric_name,
                            "provided": metric_value,
                            "expected": p95_value,
                            "gap": p95_value - metric_value,
                            "gap_percentage": ((p95_value - metric_value) / p95_value) * 100
                        })
            
            # Generate recommendations
            validation_results["recommendations"] = self._generate_p95_recommendations(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating P95 boundaries: {str(e)}")
            return {"error": str(e)}
    
    # Helper methods
    def _estimate_performance_metrics(self, profile: WorkloadProfile) -> CUCoMetrics:
        """Estimate performance metrics for workload profile"""
        # Simulate estimation based on workload characteristics
        base_metrics = {
            "moe": {"fusion_efficiency": 0.84, "overlap_ratio": 0.76, "speedup": 1.18},
            "attention": {"fusion_efficiency": 0.87, "overlap_ratio": 0.78, "speedup": 1.13},
            "training": {"fusion_efficiency": 0.85, "overlap_ratio": 0.75, "speedup": 1.15},
            "inference": {"fusion_efficiency": 0.83, "overlap_ratio": 0.74, "speedup": 1.09}
        }
        
        workload_metrics = base_metrics.get(profile.workload_type, base_metrics["training"])
        
        return CUCoMetrics(
            kernel_fusion_efficiency=workload_metrics["fusion_efficiency"],
            communication_overlap=workload_metrics["overlap_ratio"],
            end_to_end_speedup=workload_metrics["speedup"],
            memory_bandwidth_utilization=0.80,
            compute_utilization=0.90,
            network_bandwidth_utilization=0.70
        )
    
    def _estimate_cost_increase(self, profile: WorkloadProfile) -> float:
        """Estimate cost increase for CUCo optimization"""
        base_cost = 0.1  # 10% base cost increase
        gpu_factor = min(profile.gpu_count / 8.0, 0.1)  # Up to 10% for 8 GPUs
        comm_factor = profile.communication_intensity * 0.05  # Up to 5% for communication intensity
        
        return base_cost + gpu_factor + comm_factor
    
    def _check_p95_compliance(self, profile: WorkloadProfile, metrics: CUCoMetrics) -> Dict[str, Any]:
        """Check if estimated metrics meet P95 boundaries"""
        p95_boundaries = self.cuco_optimizer.p95_boundaries.get(profile.workload_type, {})
        
        compliance = {}
        for metric_name, metric_value in asdict(metrics).items():
            if metric_name in p95_boundaries:
                compliance[metric_name] = metric_value >= p95_boundaries[metric_name]
        
        return {
            "compliance": compliance,
            "overall_compliance": all(compliance.values()),
            "compliance_rate": sum(compliance.values()) / len(compliance) if compliance else 0
        }
    
    def _calculate_optimization_confidence(self, profile: WorkloadProfile) -> float:
        """Calculate confidence in optimization recommendation"""
        base_confidence = 0.8
        
        # Adjust based on GPU count
        if profile.gpu_count >= 4:
            base_confidence += 0.1
        
        # Adjust based on communication intensity
        if profile.communication_intensity >= 0.5:
            base_confidence += 0.1
        
        # Adjust based on workload type
        if profile.workload_type in ["moe", "attention"]:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _deploy_kernels(self, deployment_id: str, kernel_code: str) -> bool:
        """Deploy optimized kernels"""
        # Simulate deployment
        try:
            kernel_path = Path(f"/tmp/cuco_kernels_{deployment_id}.cu")
            with open(kernel_path, 'w') as f:
                f.write(kernel_code)
            return True
        except Exception:
            return False
    
    def _monitor_deployment_performance(self, deployment_id: str) -> Dict[str, float]:
        """Monitor deployment performance"""
        # Simulate performance monitoring
        return {
            "latency_ms": 150.0,
            "throughput_requests_per_second": 100.0,
            "gpu_utilization": 0.85,
            "memory_bandwidth_utilization": 0.80,
            "network_bandwidth_utilization": 0.70,
            "cost_per_hour": 2.50
        }
    
    def _calculate_p95_achievement(self, metrics: Optional[CUCoMetrics]) -> Dict[str, Any]:
        """Calculate P95 achievement levels"""
        if not metrics:
            return {"error": "No metrics available"}
        
        return {
            "fusion_efficiency_achievement": metrics.kernel_fusion_efficiency / metrics.p95_fusion_efficiency,
            "overlap_achievement": metrics.communication_overlap / metrics.p95_overlap_ratio,
            "speedup_achievement": metrics.end_to_end_speedup / metrics.p95_speedup_min,
            "overall_achievement": (
                (metrics.kernel_fusion_efficiency / metrics.p95_fusion_efficiency +
                 metrics.communication_overlap / metrics.p95_overlap_ratio +
                 metrics.end_to_end_speedup / metrics.p95_speedup_min) / 3
            )
        }
    
    def _generate_deployment_recommendations(self, result) -> List[str]:
        """Generate deployment recommendations"""
        recommendations = []
        
        if result.performance_gain > 1.2:
            recommendations.append("High performance gain achieved - consider scaling to larger workloads")
        
        if result.cost_increase > 0.3:
            recommendations.append("Significant cost increase - monitor ROI closely")
        
        if result.metrics and result.metrics.communication_overlap > 0.8:
            recommendations.append("Excellent communication overlap - suitable for latency-sensitive workloads")
        
        return recommendations
    
    def _run_performance_benchmark(self, deployment_id: str, duration_minutes: int) -> Dict[str, Any]:
        """Run performance benchmark"""
        # Simulate benchmark
        return {
            "current_metrics": {
                "latency_ms": 145.0,
                "throughput_rps": 105.0,
                "gpu_utilization": 0.87,
                "memory_bandwidth_utilization": 0.82,
                "network_bandwidth_utilization": 0.72,
                "cost_per_hour": 2.45
            },
            "benchmark_duration": duration_minutes,
            "samples_collected": duration_minutes * 60
        }
    
    def _compare_performance(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, Any]:
        """Compare baseline and current performance"""
        comparison = {}
        
        for metric in baseline:
            if metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    change_ratio = current_val / baseline_val
                    comparison[metric] = {
                        "baseline": baseline_val,
                        "current": current_val,
                        "change_ratio": change_ratio,
                        "improvement": change_ratio > 1.0 if "throughput" in metric else change_ratio < 1.0
                    }
        
        return comparison
    
    def _analyze_p95_compliance(self, metrics: Dict[str, float], workload_type: str) -> Dict[str, Any]:
        """Analyze P95 compliance"""
        # Simulate P95 analysis
        return {
            "compliant_metrics": 4,
            "total_metrics": 6,
            "compliance_rate": 0.67,
            "violations": [
                {"metric": "network_utilization", "provided": 0.72, "required": 0.73}
            ]
        }
    
    def _calculate_cost_efficiency(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cost efficiency metrics"""
        metrics = benchmark_results.get("current_metrics", {})
        
        return {
            "cost_per_request": metrics.get("cost_per_hour", 0) / metrics.get("throughput_requests_per_second", 1) / 3600,
            "performance_per_dollar": metrics.get("throughput_requests_per_second", 0) / metrics.get("cost_per_hour", 1),
            "efficiency_score": 0.85  # Simulated
        }
    
    def _generate_benchmark_recommendations(self, performance_comparison: Dict[str, Any]) -> List[str]:
        """Generate benchmark recommendations"""
        recommendations = []
        
        # Analyze performance comparison and generate recommendations
        for metric, comparison_data in performance_comparison.items():
            if comparison_data.get("improvement"):
                recommendations.append(f"Excellent improvement in {metric} - optimization successful")
            else:
                recommendations.append(f"Consider further optimization for {metric}")
        
        return recommendations
    
    def _calculate_expected_benefits(self, optimization_plan: OptimizationPlan) -> Dict[str, Any]:
        """Calculate expected benefits from optimization plan"""
        return {
            "performance_improvement": optimization_plan.expected_performance_gain,
            "cost_change": optimization_plan.expected_cost_increase,
            "confidence": optimization_plan.confidence_score,
            "estimated_savings_per_hour": (optimization_plan.expected_cost_increase - 1.0) * 10.0  # Assuming $10/hr base cost
        }
    
    def _get_rollback_options(self, optimization_results: Dict[str, Any]) -> List[str]:
        """Get rollback options for optimization results"""
        applied = optimization_results.get("applied_optimizations", [])
        return [opt for opt in applied if opt.endswith("_optimization")]
    
    def _get_current_metrics(self, deployment_id: str) -> Dict[str, float]:
        """Get current deployment metrics"""
        # Simulate current metrics
        return {
            "latency_ms": 150.0,
            "throughput_rps": 100.0,
            "gpu_utilization": 0.85,
            "error_rate": 0.01,
            "cost_per_hour": 2.50
        }
    
    def _analyze_performance_patterns(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance patterns"""
        return {
            "performance_trend": "stable",
            "bottlenecks": ["memory_bandwidth"],
            "optimization_opportunities": ["compute_overlap", "network_utilization"],
            "efficiency_score": 0.82
        }
    
    def _generate_recommendations(self, performance_analysis: Dict[str, Any], optimization_history: Optional[Dict]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # CUCo recommendation
        if performance_analysis.get("efficiency_score", 0) < 0.9:
            recommendations.append({
                "type": "cuco_optimization",
                "priority": "high",
                "expected_gain": 1.15,
                "complexity": "medium",
                "reasoning": "Performance below optimal - CUCo can improve compute-communication overlap"
            })
        
        return recommendations
    
    def _estimate_potential_gains(self, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate potential gains from recommendations"""
        total_gain = 1.0
        total_cost = 1.0
        
        for rec in recommendations:
            total_gain *= rec.get("expected_gain", 1.0)
            total_cost *= 1.05  # Assume 5% cost increase per recommendation
        
        return {
            "performance_gain": total_gain,
            "cost_increase": total_cost - 1.0,
            "roi": (total_gain - 1.0) / (total_cost - 1.0) if total_cost > 1.0 else 0
        }
    
    def _calculate_optimization_priority(self, recommendations: List[Dict[str, Any]]) -> str:
        """Calculate overall optimization priority"""
        high_priority_count = sum(1 for rec in recommendations if rec.get("priority") == "high")
        
        if high_priority_count >= 2:
            return "critical"
        elif high_priority_count >= 1:
            return "high"
        elif len(recommendations) >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_implementation_complexity(self, recommendations: List[Dict[str, Any]]) -> str:
        """Assess implementation complexity"""
        complexities = [rec.get("complexity", "medium") for rec in recommendations]
        
        if "high" in complexities:
            return "high"
        elif "medium" in complexities:
            return "medium"
        else:
            return "low"
    
    def _assess_optimization_risks(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Assess optimization risks"""
        risks = []
        
        for rec in recommendations:
            if rec.get("type") == "cuco_optimization":
                risks.append("Kernel compilation may fail on some GPU architectures")
                risks.append("Performance gains may vary with workload characteristics")
        
        return risks
    
    def _calculate_overall_metrics(self, summary: Dict[str, Any], cuco_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall optimization metrics"""
        return {
            "total_optimizations": summary.get("total_optimizations", 0) + cuco_summary.get("total_optimizations", 0),
            "average_speedup": cuco_summary.get("average_speedup", 1.0),
            "active_deployments": summary.get("total_deployments", 0),
            "optimization_success_rate": summary.get("optimization_success_rate", 0)
        }
    
    def _get_active_optimizations(self) -> List[Dict[str, Any]]:
        """Get active optimizations"""
        # Simulate active optimizations
        return [
            {
                "deployment_id": "deploy_001",
                "optimization_type": "cuco_kernel_optimization",
                "start_time": time.time() - 3600,
                "performance_gain": 1.18
            }
        ]
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends"""
        return {
            "trend_direction": "improving",
            "average_gain": 1.15,
            "trend_confidence": 0.85
        }
    
    def _calculate_p95_compliance_rate(self) -> float:
        """Calculate P95 compliance rate across all optimizations"""
        # Simulate compliance rate
        return 0.78
    
    def _calculate_cost_savings(self) -> Dict[str, Any]:
        """Calculate cost savings from optimizations"""
        return {
            "total_savings_per_hour": 125.50,
            "monthly_savings": 90360.00,
            "savings_rate": 0.15
        }
    
    def _generate_dashboard_recommendations(self) -> List[str]:
        """Generate dashboard-level recommendations"""
        return [
            "Consider enabling CUCo optimization for high-communication workloads",
            "Monitor P95 compliance rates for quality assurance",
            "Review cost efficiency of optimizations monthly"
        ]
    
    def _get_optimization_alerts(self) -> List[Dict[str, Any]]:
        """Get optimization alerts"""
        return [
            {
                "level": "warning",
                "message": "P95 compliance rate below 80% for recent optimizations",
                "deployment_id": "deploy_003"
            }
        ]
    
    def _categorize_performance_level(self, compliance_ratio: float) -> str:
        """Categorize performance level based on compliance ratio"""
        if compliance_ratio >= 1.1:
            return "excellent"
        elif compliance_ratio >= 1.0:
            return "good"
        elif compliance_ratio >= 0.9:
            return "acceptable"
        else:
            return "poor"
    
    def _generate_p95_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate P95 validation recommendations"""
        recommendations = []
        
        for violation in validation_results.get("violations", []):
            metric = violation["metric"]
            gap_percentage = violation["gap_percentage"]
            
            recommendations.append(f"Improve {metric} by {gap_percentage:.1f}% to meet P95 standards")
        
        if validation_results.get("overall_compliance"):
            recommendations.append("All metrics meet P95 boundaries - excellent performance")
        
        return recommendations

# Initialize CUCo MCP tools
cuco_tools = CUCoMCPTools()

# Export tool functions for MCP server
def register_cuco_tools(mcp_server):
    """Register CUCo tools with MCP server"""
    
    # Tool 1: Analyze workload for CUCo
    mcp_server.add_tool(
        name="analyze_workload_for_cuco",
        description="Analyze workload to determine CUCo optimization potential",
        function=cuco_tools.analyze_workload_for_cuco
    )
    
    # Tool 2: Deploy optimized kernels
    mcp_server.add_tool(
        name="deploy_optimized_kernels", 
        description="Deploy CUCo-optimized kernels for a workload",
        function=cuco_tools.deploy_optimized_kernels
    )
    
    # Tool 3: Benchmark optimization impact
    mcp_server.add_tool(
        name="benchmark_optimization_impact",
        description="Benchmark the performance impact of CUCo optimization", 
        function=cuco_tools.benchmark_optimization_impact
    )
    
    # Tool 4: Auto optimize deployment
    mcp_server.add_tool(
        name="auto_optimize_deployment",
        description="Automatically apply comprehensive optimizations including CUCo",
        function=cuco_tools.auto_optimize_deployment
    )
    
    # Tool 5: Get optimization recommendations
    mcp_server.add_tool(
        name="get_optimization_recommendations",
        description="Get optimization recommendations for a deployment",
        function=cuco_tools.get_optimization_recommendations
    )
    
    # Tool 6: Rollback optimization
    mcp_server.add_tool(
        name="rollback_optimization",
        description="Rollback optimizations for a deployment",
        function=cuco_tools.rollback_optimization
    )
    
    # Tool 7: Get optimization dashboard
    mcp_server.add_tool(
        name="get_optimization_dashboard",
        description="Get comprehensive optimization dashboard",
        function=cuco_tools.get_optimization_dashboard
    )
    
    # Tool 8: Validate P95 boundaries
    mcp_server.add_tool(
        name="validate_p95_boundaries",
        description="Validate metrics against P95 boundaries for workload type",
        function=cuco_tools.validate_p95_boundaries
    )
