"""
Real-time monitoring for agent evaluation.

Provides metrics collection, alerting, and visualization support
for ongoing benchmark runs.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class MetricPoint:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """An alert triggered by metric thresholds."""
    id: str
    metric_name: str
    condition: str
    threshold: float
    actual_value: float
    severity: str  # "info", "warning", "critical"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """
    Collects and aggregates metrics from benchmark runs.

    Provides:
    - Real-time metric tracking
    - Rolling window statistics
    - Threshold alerting
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize collector.

        Args:
            window_size: Size of rolling window for statistics
            alert_callback: Callback when alert is triggered
        """
        self.window_size = window_size
        self.alert_callback = alert_callback

        # Metrics storage
        self.metrics: Dict[str, deque] = {}
        self.alerts: List[Alert] = []

        # Alert thresholds
        self.thresholds: Dict[str, Dict[str, float]] = {}

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)

        point = MetricPoint(name=name, value=value, tags=tags or {})
        self.metrics[name].append(point)

        # Check thresholds
        self._check_thresholds(name, value)

    def set_threshold(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ):
        """
        Set alert thresholds for a metric.

        Args:
            metric_name: Metric to monitor
            min_value: Alert if below this value
            max_value: Alert if above this value
        """
        self.thresholds[metric_name] = {}
        if min_value is not None:
            self.thresholds[metric_name]["min"] = min_value
        if max_value is not None:
            self.thresholds[metric_name]["max"] = max_value

    def _check_thresholds(self, name: str, value: float):
        """Check if metric value triggers any alerts."""
        if name not in self.thresholds:
            return

        thresholds = self.thresholds[name]

        if "min" in thresholds and value < thresholds["min"]:
            self._trigger_alert(
                name,
                f"below minimum",
                thresholds["min"],
                value,
                "warning"
            )

        if "max" in thresholds and value > thresholds["max"]:
            self._trigger_alert(
                name,
                f"above maximum",
                thresholds["max"],
                value,
                "warning"
            )

    def _trigger_alert(
        self,
        metric_name: str,
        condition: str,
        threshold: float,
        actual: float,
        severity: str
    ):
        """Create and trigger an alert."""
        alert = Alert(
            id=f"{metric_name}_{len(self.alerts)}",
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            actual_value=actual,
            severity=severity,
            message=f"{metric_name} is {condition}: {actual:.4f} (threshold: {threshold:.4f})"
        )

        self.alerts.append(alert)

        if self.alert_callback:
            self.alert_callback(alert)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = [p.value for p in self.metrics[metric_name]]
        import numpy as np

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics}

    def get_recent(
        self,
        metric_name: str,
        count: int = 10
    ) -> List[MetricPoint]:
        """Get recent metric values."""
        if metric_name not in self.metrics:
            return []
        return list(self.metrics[metric_name])[-count:]

    def get_active_alerts(self) -> List[Alert]:
        """Get unresolved alerts."""
        return [a for a in self.alerts if not a.resolved]

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                break

    def clear(self):
        """Clear all metrics and alerts."""
        self.metrics = {}
        self.alerts = []


class Monitor:
    """
    Real-time monitoring for benchmark runs.

    Integrates with AgentRunner and Scheduler to provide
    comprehensive monitoring capabilities.
    """

    def __init__(self):
        """Initialize monitor."""
        self.collector = MetricsCollector()
        self.start_time = datetime.now()
        self.events: List[Dict[str, Any]] = []

        # Set default thresholds
        self.collector.set_threshold("latency_ms", max_value=1000)
        self.collector.set_threshold("overall_score", min_value=0.5)

    def on_step(self, step_data: Dict[str, Any]):
        """
        Record metrics from a step.

        Args:
            step_data: Step data including metrics
        """
        # Record latency
        if "latency_ms" in step_data:
            self.collector.record("latency_ms", step_data["latency_ms"])

        # Record score if available
        if "score" in step_data:
            self.collector.record("step_score", step_data["score"])

        # Record event
        self.events.append({
            "type": "step",
            "timestamp": datetime.now().isoformat(),
            "data": step_data,
        })

    def on_benchmark_complete(self, result: Any):
        """
        Record metrics from completed benchmark.

        Args:
            result: BenchmarkResult
        """
        self.collector.record("overall_score", result.overall_score)
        self.collector.record("benchmark_duration", result.duration_seconds)

        # Record per-rubric scores
        for rubric_name, score in result.rubric_scores.items():
            self.collector.record(f"rubric_{rubric_name}", score)

        # Record latency stats
        if result.latency_stats:
            self.collector.record("mean_latency", result.latency_stats.get("mean_ms", 0))
            self.collector.record("p95_latency", result.latency_stats.get("p95_ms", 0))

        # Record event
        self.events.append({
            "type": "benchmark_complete",
            "timestamp": datetime.now().isoformat(),
            "agent": result.agent_name,
            "environment": result.environment_name,
            "score": result.overall_score,
            "passed": result.passed,
        })

    def on_error(self, error: str, context: Optional[Dict] = None):
        """Record an error event."""
        self.events.append({
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {},
        })

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for dashboard display.

        Returns structured data suitable for UI rendering.
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "status": {
                "uptime_seconds": uptime,
                "start_time": self.start_time.isoformat(),
                "active_alerts": len(self.collector.get_active_alerts()),
            },
            "metrics": self.collector.get_all_stats(),
            "alerts": [
                {
                    "id": a.id,
                    "message": a.message,
                    "severity": a.severity,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self.collector.get_active_alerts()
            ],
            "recent_events": self.events[-20:],
            "summary": {
                "total_benchmarks": sum(
                    1 for e in self.events if e["type"] == "benchmark_complete"
                ),
                "total_errors": sum(
                    1 for e in self.events if e["type"] == "error"
                ),
            }
        }

    def get_metrics_timeseries(
        self,
        metric_name: str,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeseries data for a metric.

        Args:
            metric_name: Metric to retrieve
            since: Only include points after this time

        Returns:
            List of {timestamp, value} points
        """
        points = self.collector.get_recent(metric_name, count=1000)

        if since:
            points = [p for p in points if p.timestamp >= since]

        return [
            {"timestamp": p.timestamp.isoformat(), "value": p.value}
            for p in points
        ]

    def export_report(self, filepath: str):
        """Export monitoring report to JSON."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "metrics_summary": self.collector.get_all_stats(),
            "alerts": [
                {
                    "id": a.id,
                    "metric": a.metric_name,
                    "message": a.message,
                    "severity": a.severity,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                }
                for a in self.collector.alerts
            ],
            "events_summary": {
                "total_events": len(self.events),
                "by_type": {},
            }
        }

        # Count events by type
        for event in self.events:
            etype = event["type"]
            report["events_summary"]["by_type"][etype] = \
                report["events_summary"]["by_type"].get(etype, 0) + 1

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    def reset(self):
        """Reset monitor state."""
        self.collector.clear()
        self.events = []
        self.start_time = datetime.now()


async def create_monitored_runner(monitor: Optional[Monitor] = None):
    """
    Create an AgentRunner with monitoring enabled.

    Args:
        monitor: Monitor instance (creates new if None)

    Returns:
        Tuple of (AgentRunner, Monitor)
    """
    from verifiers.orchestrator.runner import AgentRunner

    monitor = monitor or Monitor()
    runner = AgentRunner()

    return runner, monitor
