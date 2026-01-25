"""
Performance and latency rubrics for benchmarking agent efficiency.

Evaluates:
- Response latency
- Throughput
- Resource utilization
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Rubric, RubricResult


class LatencyRubric(Rubric):
    """
    Evaluate response latency against threshold.

    Scores based on how close latency is to target.
    """

    def __init__(
        self,
        name: str = "latency",
        weight: float = 0.5,  # Lower weight - performance is secondary
        max_latency_ms: float = 500.0,  # Maximum acceptable latency
        target_latency_ms: float = 200.0  # Target for full score
    ):
        super().__init__(name, weight)
        self.max_latency_ms = max_latency_ms
        self.target_latency_ms = target_latency_ms

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate latency."""
        # Get latency from context (set by environment)
        latency_ms = 0.0
        if context and "latency_ms" in context:
            latency_ms = context["latency_ms"]
        elif hasattr(prediction, "inference_time_ms"):
            latency_ms = prediction.inference_time_ms
        elif hasattr(prediction, "total_inference_ms"):
            latency_ms = prediction.total_inference_ms

        # Calculate score
        if latency_ms <= self.target_latency_ms:
            score = 1.0
        elif latency_ms <= self.max_latency_ms:
            # Linear interpolation between target and max
            score = 1.0 - (latency_ms - self.target_latency_ms) / (
                self.max_latency_ms - self.target_latency_ms
            )
        else:
            # Over max - exponential decay
            overage = latency_ms - self.max_latency_ms
            score = max(0.0, 0.5 * (0.9 ** (overage / 100)))

        return RubricResult(
            score=score,
            passed=latency_ms <= self.max_latency_ms,
            metrics={
                "latency_ms": latency_ms,
                "target_ms": self.target_latency_ms,
                "max_ms": self.max_latency_ms,
                "within_target": float(latency_ms <= self.target_latency_ms),
                "within_max": float(latency_ms <= self.max_latency_ms),
            },
            feedback=f"Latency: {latency_ms:.1f}ms (target: {self.target_latency_ms}ms, max: {self.max_latency_ms}ms)"
        )


class ThroughputRubric(Rubric):
    """
    Evaluate processing throughput.

    Measures frames/samples per second against target.
    """

    def __init__(
        self,
        name: str = "throughput",
        weight: float = 0.5,
        target_fps: float = 5.0,  # Target frames per second
        min_fps: float = 1.0  # Minimum acceptable FPS
    ):
        super().__init__(name, weight)
        self.target_fps = target_fps
        self.min_fps = min_fps

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate throughput."""
        # Calculate FPS from latency
        latency_ms = 0.0
        if context and "latency_ms" in context:
            latency_ms = context["latency_ms"]
        elif hasattr(prediction, "inference_time_ms"):
            latency_ms = prediction.inference_time_ms

        if latency_ms > 0:
            fps = 1000.0 / latency_ms
        else:
            fps = 0.0

        # Calculate score
        if fps >= self.target_fps:
            score = 1.0
        elif fps >= self.min_fps:
            score = (fps - self.min_fps) / (self.target_fps - self.min_fps)
        else:
            score = 0.0

        return RubricResult(
            score=score,
            passed=fps >= self.min_fps,
            metrics={
                "fps": fps,
                "target_fps": self.target_fps,
                "min_fps": self.min_fps,
                "latency_ms": latency_ms,
            },
            feedback=f"Throughput: {fps:.1f} FPS (target: {self.target_fps} FPS)"
        )


class MemoryRubric(Rubric):
    """
    Evaluate memory usage.

    Measures memory consumption against limits.
    """

    def __init__(
        self,
        name: str = "memory",
        weight: float = 0.3,
        max_memory_mb: float = 1024.0,
        target_memory_mb: float = 512.0
    ):
        super().__init__(name, weight)
        self.max_memory_mb = max_memory_mb
        self.target_memory_mb = target_memory_mb

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate memory usage."""
        # Get memory from context or prediction metadata
        memory_mb = 0.0
        if context and "memory_mb" in context:
            memory_mb = context["memory_mb"]
        elif hasattr(prediction, "memory_used_mb"):
            memory_mb = prediction.memory_used_mb

        if memory_mb == 0:
            # No memory tracking available
            return RubricResult(
                score=1.0,
                passed=True,
                feedback="Memory tracking not available"
            )

        # Calculate score
        if memory_mb <= self.target_memory_mb:
            score = 1.0
        elif memory_mb <= self.max_memory_mb:
            score = 1.0 - (memory_mb - self.target_memory_mb) / (
                self.max_memory_mb - self.target_memory_mb
            )
        else:
            score = 0.0

        return RubricResult(
            score=score,
            passed=memory_mb <= self.max_memory_mb,
            metrics={
                "memory_mb": memory_mb,
                "target_mb": self.target_memory_mb,
                "max_mb": self.max_memory_mb,
            },
            feedback=f"Memory: {memory_mb:.1f}MB (target: {self.target_memory_mb}MB)"
        )


class ConsistencyRubric(Rubric):
    """
    Evaluate prediction consistency across similar inputs.

    Measures variance in predictions for near-identical test cases.
    Used in multi-turn environments to ensure stable predictions.
    """

    def __init__(
        self,
        name: str = "consistency",
        weight: float = 0.5,
        variance_threshold: float = 0.1  # Max allowed variance
    ):
        super().__init__(name, weight)
        self.variance_threshold = variance_threshold
        self._predictions: List[Dict] = []

    def reset(self):
        """Reset prediction history."""
        self._predictions = []

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate consistency."""
        # Store prediction for variance calculation
        pred_dict = self._extract_key_values(prediction)
        self._predictions.append(pred_dict)

        if len(self._predictions) < 2:
            return RubricResult(
                score=1.0,
                passed=True,
                feedback="Insufficient data for consistency check"
            )

        # Calculate variance across predictions
        variance = self._calculate_variance()
        score = max(0.0, 1.0 - variance / self.variance_threshold)

        return RubricResult(
            score=score,
            passed=variance <= self.variance_threshold,
            metrics={
                "variance": variance,
                "threshold": self.variance_threshold,
                "predictions_compared": len(self._predictions),
            },
            feedback=f"Prediction variance: {variance:.3f} (threshold: {self.variance_threshold})"
        )

    def _extract_key_values(self, prediction: Any) -> Dict[str, float]:
        """Extract numeric values from prediction for variance calculation."""
        values = {}
        if prediction is None:
            return values

        # Extract common numeric attributes
        for attr in ["confidence", "urgency_score", "severity", "chaos_level"]:
            if hasattr(prediction, attr):
                val = getattr(prediction, attr)
                if isinstance(val, (int, float)):
                    values[attr] = float(val)
                elif hasattr(val, "value"):
                    values[attr] = float(val.value)

        return values

    def _calculate_variance(self) -> float:
        """Calculate average variance across all numeric predictions."""
        if len(self._predictions) < 2:
            return 0.0

        # Get all keys that appear in predictions
        all_keys = set()
        for pred in self._predictions:
            all_keys.update(pred.keys())

        if not all_keys:
            return 0.0

        # Calculate variance for each key
        variances = []
        for key in all_keys:
            values = [p.get(key, 0) for p in self._predictions if key in p]
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                variances.append(variance)

        return sum(variances) / len(variances) if variances else 0.0
