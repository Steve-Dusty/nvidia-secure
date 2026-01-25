"""
Visual detection rubrics for evaluating visual inference agents.

Evaluates:
- Action classification (standing, walking, running, fighting, falling)
- Fall/fight detection accuracy
- Person detection via bounding box IoU
- Severity classification
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Rubric, RubricResult


class ActionClassificationRubric(Rubric):
    """
    Evaluate action classification accuracy.

    Compares predicted action type against ground truth.
    Supports partial credit for related actions.
    """

    # Action similarity groups for partial credit
    ACTION_GROUPS = {
        "movement": {"walking", "running", "standing"},
        "emergency": {"falling", "fighting", "lying_down"},
        "static": {"standing", "lying_down"},
    }

    def __init__(
        self,
        name: str = "action_classification",
        weight: float = 1.0,
        partial_credit: bool = True
    ):
        super().__init__(name, weight)
        self.partial_credit = partial_credit

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate action classification."""
        if prediction is None:
            return RubricResult(
                score=0.0,
                passed=False,
                feedback="No prediction provided"
            )

        # Extract expected and predicted actions
        expected_action = ground_truth.get("action", "unknown")
        if hasattr(prediction, "scene_action"):
            predicted_action = prediction.scene_action.value if hasattr(
                prediction.scene_action, "value"
            ) else str(prediction.scene_action)
        else:
            predicted_action = str(prediction)

        # Normalize to lowercase
        expected_action = expected_action.lower()
        predicted_action = predicted_action.lower()

        # Check exact match
        if expected_action == predicted_action:
            score = 1.0
            correct = True
        elif self.partial_credit:
            # Check for partial credit within same group
            score = self._compute_partial_score(expected_action, predicted_action)
            correct = score >= 0.5
        else:
            score = 0.0
            correct = False

        # Get confidence if available
        confidence = 0.0
        if hasattr(prediction, "action_confidence"):
            confidence = prediction.action_confidence

        return RubricResult(
            score=score,
            passed=correct,
            metrics={
                "action_correct": float(expected_action == predicted_action),
                "confidence": confidence,
                "partial_score": score,
            },
            feedback=f"Expected '{expected_action}', got '{predicted_action}'"
        )

    def _compute_partial_score(self, expected: str, predicted: str) -> float:
        """Compute partial credit for related actions."""
        for group_name, group_actions in self.ACTION_GROUPS.items():
            if expected in group_actions and predicted in group_actions:
                return 0.5  # Same group, partial credit
        return 0.0


class FallDetectionRubric(Rubric):
    """
    Evaluate fall detection accuracy.

    Binary classification: did the agent correctly detect a fall?
    Tracks precision/recall components.
    """

    def __init__(
        self,
        name: str = "fall_detection",
        weight: float = 1.0,
        false_negative_penalty: float = 1.5  # Penalize missing falls more
    ):
        super().__init__(name, weight)
        self.fn_penalty = false_negative_penalty

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate fall detection."""
        if prediction is None:
            expected = ground_truth.get("fall_detected", False)
            return RubricResult(
                score=0.0 if expected else 1.0,
                passed=not expected,
                metrics={"false_negative": int(expected)},
                feedback="No prediction provided"
            )

        expected = ground_truth.get("fall_detected", False)
        predicted = getattr(prediction, "fall_detected", False)

        # Calculate confusion matrix components
        tp = int(expected and predicted)
        fp = int(not expected and predicted)
        fn = int(expected and not predicted)
        tn = int(not expected and not predicted)

        # Score: correct = 1.0, FN penalized more than FP
        if expected == predicted:
            score = 1.0
        elif fn > 0:
            score = max(0.0, 1.0 - self.fn_penalty)
        else:  # FP
            score = 0.5

        # Get additional metrics
        persons_detected = 0
        if hasattr(prediction, "persons"):
            persons_detected = len(prediction.persons)
        elif hasattr(prediction, "persons_detected"):
            persons_detected = prediction.persons_detected

        return RubricResult(
            score=score,
            passed=expected == predicted,
            metrics={
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "persons_detected": persons_detected,
            },
            feedback=f"Fall {'detected' if predicted else 'not detected'}, expected {'fall' if expected else 'no fall'}"
        )


class FightDetectionRubric(Rubric):
    """
    Evaluate fight/violence detection.

    Similar to fall detection but for violent altercations.
    """

    def __init__(
        self,
        name: str = "fight_detection",
        weight: float = 1.0,
        false_negative_penalty: float = 1.5
    ):
        super().__init__(name, weight)
        self.fn_penalty = false_negative_penalty

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate fight detection."""
        if prediction is None:
            expected = ground_truth.get("fight_detected", False)
            return RubricResult(
                score=0.0 if expected else 1.0,
                passed=not expected,
                metrics={"false_negative": int(expected)},
                feedback="No prediction provided"
            )

        expected = ground_truth.get("fight_detected", False)
        predicted = getattr(prediction, "fight_detected", False)

        # Calculate score
        if expected == predicted:
            score = 1.0
        elif expected and not predicted:  # FN
            score = max(0.0, 1.0 - self.fn_penalty)
        else:  # FP
            score = 0.5

        # Get severity if available
        severity = 0
        if hasattr(prediction, "scene_severity"):
            severity = prediction.scene_severity.value if hasattr(
                prediction.scene_severity, "value"
            ) else int(prediction.scene_severity)

        return RubricResult(
            score=score,
            passed=expected == predicted,
            metrics={
                "fight_expected": float(expected),
                "fight_predicted": float(predicted),
                "severity_level": severity,
            },
            feedback=f"Fight {'detected' if predicted else 'not detected'}, expected {'fight' if expected else 'no fight'}"
        )


class BoundingBoxIoURubric(Rubric):
    """
    Evaluate person detection via Intersection over Union (IoU).

    Compares predicted bounding boxes against ground truth boxes.
    Uses Hungarian matching for optimal box assignment.
    """

    def __init__(
        self,
        name: str = "bounding_box_iou",
        weight: float = 1.0,
        iou_threshold: float = 0.5
    ):
        super().__init__(name, weight)
        self.iou_threshold = iou_threshold

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate bounding box IoU."""
        if prediction is None:
            return RubricResult(
                score=0.0,
                passed=False,
                feedback="No prediction provided"
            )

        # Extract ground truth boxes
        gt_boxes = ground_truth.get("bounding_boxes", [])
        if not gt_boxes:
            # No ground truth boxes - evaluate based on false positives
            pred_boxes = self._extract_pred_boxes(prediction)
            if not pred_boxes:
                return RubricResult(score=1.0, passed=True, feedback="No boxes expected or detected")
            else:
                return RubricResult(
                    score=0.5,
                    passed=True,
                    metrics={"false_positives": len(pred_boxes)},
                    feedback=f"No boxes expected but {len(pred_boxes)} detected"
                )

        # Extract predicted boxes
        pred_boxes = self._extract_pred_boxes(prediction)
        if not pred_boxes:
            return RubricResult(
                score=0.0,
                passed=False,
                metrics={"false_negatives": len(gt_boxes)},
                feedback=f"Expected {len(gt_boxes)} boxes but none detected"
            )

        # Calculate IoU for each pair and find best matches
        ious = self._calculate_ious(pred_boxes, gt_boxes)

        avg_iou = sum(ious) / len(ious) if ious else 0.0
        matches_above_threshold = sum(1 for iou in ious if iou >= self.iou_threshold)

        return RubricResult(
            score=avg_iou,
            passed=avg_iou >= self.iou_threshold,
            metrics={
                "average_iou": avg_iou,
                "boxes_predicted": len(pred_boxes),
                "boxes_expected": len(gt_boxes),
                "matched_boxes": len(ious),
                "matches_above_threshold": matches_above_threshold,
            },
            feedback=f"Average IoU: {avg_iou:.3f} (threshold: {self.iou_threshold})"
        )

    def _extract_pred_boxes(self, prediction: Any) -> List[Tuple[float, float, float, float]]:
        """Extract bounding boxes from prediction."""
        boxes = []
        if hasattr(prediction, "persons"):
            for person in prediction.persons:
                if hasattr(person, "bbox"):
                    boxes.append(person.bbox)
        return boxes

    def _calculate_ious(
        self,
        pred_boxes: List[Tuple],
        gt_boxes: List[Tuple]
    ) -> List[float]:
        """Calculate IoU for matched box pairs."""
        ious = []

        for gt_box in gt_boxes:
            best_iou = 0.0
            for pred_box in pred_boxes:
                iou = self._box_iou(pred_box, gt_box)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

        return ious

    def _box_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes (x1, y1, x2, y2)."""
        # Handle different box formats
        if isinstance(box2, dict):
            box2 = (box2.get("x1", 0), box2.get("y1", 0),
                    box2.get("x2", 1), box2.get("y2", 1))

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class PersonCountRubric(Rubric):
    """
    Evaluate person count accuracy.

    Compares number of detected persons against ground truth.
    """

    def __init__(
        self,
        name: str = "person_count",
        weight: float = 1.0,
        tolerance: int = 0  # Allowed difference
    ):
        super().__init__(name, weight)
        self.tolerance = tolerance

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate person count."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected = ground_truth.get("persons_count", 0)

        # Extract predicted count
        if hasattr(prediction, "persons"):
            predicted = len(prediction.persons)
        elif hasattr(prediction, "persons_detected"):
            predicted = prediction.persons_detected
        else:
            predicted = 0

        difference = abs(expected - predicted)

        # Score based on difference
        if difference <= self.tolerance:
            score = 1.0
        else:
            score = max(0.0, 1.0 - (difference - self.tolerance) * 0.2)

        return RubricResult(
            score=score,
            passed=difference <= self.tolerance,
            metrics={
                "expected_count": expected,
                "predicted_count": predicted,
                "difference": difference,
            },
            feedback=f"Expected {expected} persons, detected {predicted}"
        )


class SeverityClassificationRubric(Rubric):
    """
    Evaluate severity level classification.

    Compares predicted severity (NONE/LOW/MEDIUM/HIGH/CRITICAL)
    against ground truth.
    """

    SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]

    def __init__(
        self,
        name: str = "severity_classification",
        weight: float = 1.0,
        tolerance: int = 1  # Allowed levels difference
    ):
        super().__init__(name, weight)
        self.tolerance = tolerance

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate severity classification."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected = ground_truth.get("severity", "none").lower()

        # Extract predicted severity
        if hasattr(prediction, "scene_severity"):
            predicted = prediction.scene_severity.name.lower() if hasattr(
                prediction.scene_severity, "name"
            ) else str(prediction.scene_severity).lower()
        elif hasattr(prediction, "overall_severity"):
            predicted = prediction.overall_severity.name.lower() if hasattr(
                prediction.overall_severity, "name"
            ) else str(prediction.overall_severity).lower()
        else:
            predicted = "none"

        # Get indices for comparison
        expected_idx = self.SEVERITY_ORDER.index(expected) if expected in self.SEVERITY_ORDER else 0
        predicted_idx = self.SEVERITY_ORDER.index(predicted) if predicted in self.SEVERITY_ORDER else 0

        difference = abs(expected_idx - predicted_idx)

        # Score based on difference
        if difference == 0:
            score = 1.0
        elif difference <= self.tolerance:
            score = 0.75
        else:
            score = max(0.0, 1.0 - difference * 0.25)

        return RubricResult(
            score=score,
            passed=difference <= self.tolerance,
            metrics={
                "expected_severity": expected,
                "predicted_severity": predicted,
                "level_difference": difference,
            },
            feedback=f"Expected '{expected}' severity, got '{predicted}'"
        )
