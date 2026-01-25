"""
Emergency response rubrics for evaluating response routing agents.

Evaluates:
- Urgency score accuracy
- 911/311 dispatch routing
- Nearest facility matching
- Response quality and instructions
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Rubric, RubricResult


class UrgencyScoreRubric(Rubric):
    """
    Evaluate urgency classification accuracy.

    Compares predicted urgency score (0-10) against ground truth.
    """

    def __init__(
        self,
        name: str = "urgency_score",
        weight: float = 1.0,
        tolerance: float = 1.0  # Allowed score difference
    ):
        super().__init__(name, weight)
        self.tolerance = tolerance

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate urgency score accuracy."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected_score = ground_truth.get("urgency_score", 5.0)
        predicted_score = getattr(prediction, "urgency_score", 5.0)

        error = abs(expected_score - predicted_score)
        score = max(0.0, 1.0 - (error / 10.0))  # Normalize to 0-1

        # Also check urgency level if available
        expected_level = ground_truth.get("urgency_level", "").upper()
        predicted_level = getattr(prediction, "urgency_level", "").upper()
        level_match = expected_level == predicted_level if expected_level else True

        return RubricResult(
            score=score,
            passed=error <= self.tolerance,
            metrics={
                "urgency_error": error,
                "expected_score": expected_score,
                "predicted_score": predicted_score,
                "level_match": float(level_match),
            },
            feedback=f"Urgency: {predicted_score:.1f} (expected: {expected_score:.1f}, error: {error:.1f})"
        )


class DispatchRoutingRubric(Rubric):
    """
    Evaluate 911 vs 311 routing decision.

    Critical rubric: incorrect routing can have severe consequences.
    """

    def __init__(
        self,
        name: str = "dispatch_routing",
        weight: float = 2.0,  # Higher weight - critical
        false_911_penalty: float = 0.5,  # False 911 is wasteful
        false_311_penalty: float = 1.0   # Missing 911 is dangerous
    ):
        super().__init__(name, weight)
        self.false_911_penalty = false_911_penalty
        self.false_311_penalty = false_311_penalty

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate dispatch routing."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected = ground_truth.get("call_number", "911")
        predicted = getattr(prediction, "call_number", "911")

        # Normalize
        expected = expected.replace("-", "").strip()
        predicted = predicted.replace("-", "").strip()

        if expected == predicted:
            score = 1.0
            correct = True
        elif expected == "911" and predicted == "311":
            # Dangerous: should be 911 but routed to 311
            score = max(0.0, 1.0 - self.false_311_penalty)
            correct = False
        else:
            # Less dangerous: should be 311 but routed to 911
            score = max(0.0, 1.0 - self.false_911_penalty)
            correct = False

        return RubricResult(
            score=score,
            passed=correct,
            metrics={
                "correct_routing": float(expected == predicted),
                "expected_number": expected,
                "predicted_number": predicted,
            },
            feedback=f"Routed to {predicted} (expected: {expected})"
        )


class FacilityMatchingRubric(Rubric):
    """
    Evaluate nearest facility recommendations.

    Checks if recommended facilities are correct and within distance tolerance.
    """

    FACILITY_TYPES = ["nearest_hospital", "nearest_pharmacy", "nearest_clinic"]

    def __init__(
        self,
        name: str = "facility_matching",
        weight: float = 1.0,
        distance_tolerance_miles: float = 0.5
    ):
        super().__init__(name, weight)
        self.tolerance = distance_tolerance_miles

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate facility matching."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        scores = []
        metrics = {}

        for facility_type in self.FACILITY_TYPES:
            expected = ground_truth.get(facility_type)
            predicted = getattr(prediction, facility_type, None)

            if expected is None:
                continue  # Skip if not in ground truth

            if predicted is None:
                scores.append(0.0)
                metrics[f"{facility_type}_found"] = 0.0
                continue

            # Compare distances
            expected_distance = expected.get("distance", 0) if isinstance(expected, dict) else 0
            predicted_distance = predicted.distance_miles if hasattr(predicted, "distance_miles") else 0

            distance_error = abs(expected_distance - predicted_distance)

            if distance_error <= self.tolerance:
                scores.append(1.0)
            else:
                scores.append(max(0.0, 1.0 - distance_error / 5.0))

            metrics[f"{facility_type}_distance_error"] = distance_error
            metrics[f"{facility_type}_found"] = 1.0

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RubricResult(
            score=avg_score,
            passed=avg_score >= 0.7,
            metrics={
                "facility_accuracy": avg_score,
                "facilities_checked": len(scores),
                **metrics
            },
            feedback=f"Facility matching accuracy: {avg_score:.2%}"
        )


class ResponseQualityRubric(Rubric):
    """
    Evaluate overall response quality.

    Checks for presence and quality of:
    - Immediate instructions
    - Rationale
    - Recommended actions
    """

    def __init__(
        self,
        name: str = "response_quality",
        weight: float = 1.0,
        min_instructions: int = 2
    ):
        super().__init__(name, weight)
        self.min_instructions = min_instructions

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate response quality."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        scores = []
        metrics = {}

        # Check immediate instructions
        instructions = getattr(prediction, "immediate_instructions", [])
        if isinstance(instructions, list):
            instructions_score = min(1.0, len(instructions) / self.min_instructions)
        else:
            instructions_score = 0.5 if instructions else 0.0
        scores.append(instructions_score)
        metrics["instructions_count"] = len(instructions) if isinstance(instructions, list) else 0
        metrics["instructions_score"] = instructions_score

        # Check rationale
        rationale = getattr(prediction, "rationale", "")
        rationale_score = 1.0 if rationale and len(rationale) > 20 else 0.0
        scores.append(rationale_score)
        metrics["has_rationale"] = float(bool(rationale))
        metrics["rationale_length"] = len(rationale) if rationale else 0

        # Check recommended action
        action = getattr(prediction, "recommended_action", "")
        action_score = 1.0 if action and len(action) > 10 else 0.0
        scores.append(action_score)
        metrics["has_action"] = float(bool(action))

        # Check ETA if available
        eta = getattr(prediction, "estimated_ems_arrival_min", None)
        if eta is not None and eta > 0:
            scores.append(1.0)
            metrics["has_eta"] = 1.0
            metrics["eta_minutes"] = eta

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RubricResult(
            score=avg_score,
            passed=avg_score >= 0.6,
            metrics=metrics,
            feedback=f"Response quality: {avg_score:.2%}"
        )


class InstructionRelevanceRubric(Rubric):
    """
    Evaluate relevance of immediate instructions to the emergency type.

    Checks if instructions are appropriate for the detected emergency.
    """

    # Keyword mappings for instruction validation
    INSTRUCTION_KEYWORDS = {
        "medical": ["breathing", "pulse", "recovery", "ambulance", "cpr"],
        "overdose": ["naloxone", "narcan", "breathing", "recovery", "911"],
        "fall": ["move", "spine", "neck", "immobilize", "conscious"],
        "violence": ["safe", "distance", "police", "leave", "hide"],
        "distress": ["calm", "listen", "help", "emergency", "911"],
    }

    def __init__(
        self,
        name: str = "instruction_relevance",
        weight: float = 1.0
    ):
        super().__init__(name, weight)

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate instruction relevance."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        emergency_type = ground_truth.get("emergency_type", "").lower()
        instructions = getattr(prediction, "immediate_instructions", [])

        if not instructions:
            return RubricResult(
                score=0.0,
                passed=False,
                feedback="No instructions provided"
            )

        # Get relevant keywords for this emergency type
        relevant_keywords = set()
        for etype, keywords in self.INSTRUCTION_KEYWORDS.items():
            if etype in emergency_type or emergency_type in etype:
                relevant_keywords.update(keywords)

        if not relevant_keywords:
            # Unknown emergency type - just check instructions exist
            return RubricResult(
                score=0.5,
                passed=True,
                feedback="Emergency type not recognized for relevance check"
            )

        # Check how many instructions contain relevant keywords
        instructions_text = " ".join(instructions).lower()
        found_keywords = sum(1 for kw in relevant_keywords if kw in instructions_text)

        relevance_score = min(1.0, found_keywords / (len(relevant_keywords) / 2))

        return RubricResult(
            score=relevance_score,
            passed=relevance_score >= 0.5,
            metrics={
                "relevant_keywords_found": found_keywords,
                "total_relevant_keywords": len(relevant_keywords),
                "instructions_count": len(instructions),
            },
            feedback=f"Found {found_keywords}/{len(relevant_keywords)} relevant instruction keywords"
        )
