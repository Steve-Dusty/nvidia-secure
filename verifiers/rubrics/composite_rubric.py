"""
Composite rubrics for multi-modal and combined evaluation.

These rubrics combine multiple evaluation criteria into unified scores
for integrated emergency detection systems.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Rubric, RubricResult


class CompositeRubric(Rubric):
    """
    Combine multiple rubrics into a single weighted score.

    Useful for creating domain-specific evaluation bundles.
    """

    def __init__(
        self,
        name: str,
        rubrics: List[Rubric],
        weight: float = 1.0,
        require_all_pass: bool = False
    ):
        super().__init__(name, weight)
        self.rubrics = rubrics
        self.require_all_pass = require_all_pass

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate using all sub-rubrics."""
        results = {}
        total_score = 0.0
        total_weight = 0.0
        all_passed = True

        for rubric in self.rubrics:
            result = await rubric.evaluate(prediction, ground_truth, context)
            results[rubric.name] = result
            total_score += result.score * rubric.weight
            total_weight += rubric.weight
            if not result.passed:
                all_passed = False

        weighted_score = total_score / total_weight if total_weight > 0 else 0.0
        passed = all_passed if self.require_all_pass else weighted_score >= 0.5

        # Collect all metrics
        combined_metrics = {
            "composite_score": weighted_score,
            "sub_rubric_count": len(self.rubrics),
            "all_passed": float(all_passed),
        }
        for rubric_name, result in results.items():
            combined_metrics[f"{rubric_name}_score"] = result.score
            combined_metrics[f"{rubric_name}_passed"] = float(result.passed)

        return RubricResult(
            score=weighted_score,
            passed=passed,
            metrics=combined_metrics,
            feedback=f"Composite score: {weighted_score:.2%} ({sum(1 for r in results.values() if r.passed)}/{len(results)} passed)"
        )


class EmergencyClassificationRubric(Rubric):
    """
    Evaluate emergency type classification.

    Compares predicted emergency type against ground truth with
    support for severity-aware scoring.
    """

    # Emergency type severity mapping
    SEVERITY_MAP = {
        "none": 0,
        "medical": 3,
        "violence": 4,
        "fall": 3,
        "distress": 2,
        "crowd_incident": 3,
        "unknown": 1,
    }

    def __init__(
        self,
        name: str = "emergency_classification",
        weight: float = 2.0,
        severity_aware: bool = True
    ):
        super().__init__(name, weight)
        self.severity_aware = severity_aware

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate emergency classification."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected = ground_truth.get("emergency_type", "none").lower()

        # Extract predicted type
        if hasattr(prediction, "emergency_type"):
            predicted = prediction.emergency_type.value if hasattr(
                prediction.emergency_type, "value"
            ) else str(prediction.emergency_type).lower()
        else:
            predicted = "none"

        # Exact match
        if expected == predicted:
            score = 1.0
            correct = True
        elif self.severity_aware:
            # Partial credit based on severity difference
            exp_severity = self.SEVERITY_MAP.get(expected, 1)
            pred_severity = self.SEVERITY_MAP.get(predicted, 1)
            severity_diff = abs(exp_severity - pred_severity)

            if severity_diff == 0:
                score = 0.8  # Same severity level
            elif severity_diff == 1:
                score = 0.5  # Adjacent severity
            else:
                score = max(0.0, 0.3 - severity_diff * 0.1)

            correct = score >= 0.5
        else:
            score = 0.0
            correct = False

        return RubricResult(
            score=score,
            passed=correct,
            metrics={
                "expected_type": expected,
                "predicted_type": predicted,
                "exact_match": float(expected == predicted),
                "expected_severity": self.SEVERITY_MAP.get(expected, 1),
                "predicted_severity": self.SEVERITY_MAP.get(predicted, 1),
            },
            feedback=f"Emergency: '{predicted}' (expected: '{expected}')"
        )


class MultiModalFusionRubric(Rubric):
    """
    Evaluate multi-modal fusion quality.

    Assesses how well the agent combines visual and audio signals
    into a coherent emergency assessment.
    """

    def __init__(
        self,
        name: str = "multi_modal_fusion",
        weight: float = 1.5
    ):
        super().__init__(name, weight)

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate multi-modal fusion."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        scores = []
        metrics = {}

        # Check visual component integration
        has_visual = hasattr(prediction, "visual_result") and prediction.visual_result is not None
        metrics["has_visual"] = float(has_visual)
        if has_visual:
            scores.append(1.0)

        # Check audio component integration
        has_audio = hasattr(prediction, "audio_result") and prediction.audio_result is not None
        metrics["has_audio"] = float(has_audio)
        if has_audio:
            scores.append(1.0)

        # Check combined severity assessment
        has_combined_severity = hasattr(prediction, "overall_severity")
        metrics["has_combined_severity"] = float(has_combined_severity)
        if has_combined_severity:
            scores.append(1.0)

        # Check for consistency between modalities
        if has_visual and has_audio:
            visual_severity = 0
            audio_severity = 0

            if hasattr(prediction, "visual_result") and prediction.visual_result:
                if hasattr(prediction.visual_result, "scene_severity"):
                    visual_severity = prediction.visual_result.scene_severity.value if hasattr(
                        prediction.visual_result.scene_severity, "value"
                    ) else int(prediction.visual_result.scene_severity)

            if hasattr(prediction, "audio_result") and prediction.audio_result:
                if hasattr(prediction.audio_result, "severity"):
                    audio_severity = prediction.audio_result.severity.value if hasattr(
                        prediction.audio_result.severity, "value"
                    ) else int(prediction.audio_result.severity)

            # Combined should be at least the max of individual severities
            combined_severity = 0
            if hasattr(prediction, "overall_severity"):
                combined_severity = prediction.overall_severity.value if hasattr(
                    prediction.overall_severity, "value"
                ) else int(prediction.overall_severity)

            fusion_correct = combined_severity >= max(visual_severity, audio_severity)
            scores.append(1.0 if fusion_correct else 0.5)
            metrics["fusion_correct"] = float(fusion_correct)
            metrics["visual_severity"] = visual_severity
            metrics["audio_severity"] = audio_severity
            metrics["combined_severity"] = combined_severity

        # Check dispatch recommendation presence
        has_dispatch = hasattr(prediction, "dispatch_recommended")
        metrics["has_dispatch"] = float(has_dispatch)
        if has_dispatch:
            scores.append(1.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RubricResult(
            score=avg_score,
            passed=avg_score >= 0.6,
            metrics=metrics,
            feedback=f"Multi-modal fusion score: {avg_score:.2%}"
        )


class DispatchDecisionRubric(Rubric):
    """
    Evaluate the overall dispatch decision quality.

    Combines multiple factors: urgency, routing, timing, and appropriateness.
    """

    def __init__(
        self,
        name: str = "dispatch_decision",
        weight: float = 2.0
    ):
        super().__init__(name, weight)

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate dispatch decision."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        scores = []
        metrics = {}

        # Check dispatch_recommended vs ground truth
        expected_dispatch = ground_truth.get("dispatch_recommended", False)
        predicted_dispatch = getattr(prediction, "dispatch_recommended", False)

        if expected_dispatch == predicted_dispatch:
            scores.append(1.0)
            metrics["dispatch_correct"] = 1.0
        elif expected_dispatch and not predicted_dispatch:
            # Missed dispatch - more serious
            scores.append(0.0)
            metrics["dispatch_correct"] = 0.0
            metrics["missed_dispatch"] = 1.0
        else:
            # False alarm - less serious
            scores.append(0.5)
            metrics["dispatch_correct"] = 0.5
            metrics["false_alarm"] = 1.0

        # Check response priority
        expected_priority = ground_truth.get("response_priority", 3)
        predicted_priority = getattr(prediction, "response_priority", 3)

        priority_diff = abs(expected_priority - predicted_priority)
        priority_score = max(0.0, 1.0 - priority_diff * 0.25)
        scores.append(priority_score)
        metrics["priority_score"] = priority_score
        metrics["priority_diff"] = priority_diff

        # Check confidence
        confidence = getattr(prediction, "confidence", 0.0)
        if confidence > 0:
            scores.append(confidence)
            metrics["confidence"] = confidence

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RubricResult(
            score=avg_score,
            passed=avg_score >= 0.6 and metrics.get("dispatch_correct", 0) >= 0.5,
            metrics=metrics,
            feedback=f"Dispatch decision score: {avg_score:.2%}"
        )


class AlertTimingRubric(Rubric):
    """
    Evaluate the timing of alert generation in multi-turn scenarios.

    Checks if alerts are generated at the appropriate turn/frame.
    """

    def __init__(
        self,
        name: str = "alert_timing",
        weight: float = 1.0,
        early_tolerance: int = 2,  # Allowed frames early
        late_tolerance: int = 1    # Allowed frames late
    ):
        super().__init__(name, weight)
        self.early_tolerance = early_tolerance
        self.late_tolerance = late_tolerance

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate alert timing."""
        if context is None:
            return RubricResult(
                score=0.5,
                passed=True,
                feedback="No context for timing evaluation"
            )

        expected_alert_turn = ground_truth.get("alert_turn", -1)
        if expected_alert_turn < 0:
            # No specific timing requirement
            return RubricResult(
                score=1.0,
                passed=True,
                feedback="No timing requirement"
            )

        current_turn = context.get("turn", 0)
        alert_triggered = getattr(prediction, "dispatch_recommended", False)

        metrics = {
            "expected_turn": expected_alert_turn,
            "current_turn": current_turn,
            "alert_triggered": float(alert_triggered),
        }

        if not alert_triggered:
            if current_turn < expected_alert_turn:
                # Not yet time for alert
                return RubricResult(score=1.0, passed=True, metrics=metrics, feedback="Waiting for alert")
            elif current_turn <= expected_alert_turn + self.late_tolerance:
                # Within late tolerance - partial credit
                return RubricResult(score=0.7, passed=True, metrics=metrics, feedback="Slightly late alert window")
            else:
                # Too late
                return RubricResult(score=0.0, passed=False, metrics=metrics, feedback="Alert too late or missed")

        # Alert was triggered
        if current_turn < expected_alert_turn - self.early_tolerance:
            # Too early
            score = 0.5
            feedback = f"Alert triggered too early (turn {current_turn}, expected {expected_alert_turn})"
        elif current_turn <= expected_alert_turn + self.late_tolerance:
            # On time or within tolerance
            score = 1.0
            feedback = "Alert triggered on time"
        else:
            # Late
            late_by = current_turn - expected_alert_turn
            score = max(0.0, 1.0 - late_by * 0.2)
            feedback = f"Alert triggered late by {late_by} turns"

        return RubricResult(
            score=score,
            passed=score >= 0.5,
            metrics=metrics,
            feedback=feedback
        )
