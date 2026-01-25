"""
Audio detection rubrics for evaluating audio inference agents.

Evaluates:
- Speech recognition accuracy (WER)
- Audio event classification
- Distress call detection
- Keyword detection
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Rubric, RubricResult


class SpeechRecognitionRubric(Rubric):
    """
    Evaluate speech recognition accuracy via Word Error Rate (WER).

    Lower WER = better accuracy. Score is inverted for 0-1 scale.
    """

    def __init__(
        self,
        name: str = "speech_recognition",
        weight: float = 1.0,
        wer_threshold: float = 0.3  # Max acceptable WER
    ):
        super().__init__(name, weight)
        self.wer_threshold = wer_threshold

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate speech recognition accuracy."""
        expected_transcript = ground_truth.get("transcript", "")
        if not expected_transcript:
            # No transcript expected - check if prediction is also empty
            predicted = self._extract_transcript(prediction)
            if not predicted:
                return RubricResult(score=1.0, passed=True, feedback="No transcript expected or detected")
            else:
                return RubricResult(
                    score=0.5,
                    passed=True,
                    feedback=f"No transcript expected but got: '{predicted[:50]}...'"
                )

        predicted = self._extract_transcript(prediction)
        if not predicted:
            return RubricResult(
                score=0.0,
                passed=False,
                feedback="Expected transcript but none detected"
            )

        # Calculate Word Error Rate
        wer = self._calculate_wer(expected_transcript, predicted)
        score = max(0.0, 1.0 - wer)

        return RubricResult(
            score=score,
            passed=wer <= self.wer_threshold,
            metrics={
                "word_error_rate": wer,
                "expected_words": len(expected_transcript.split()),
                "predicted_words": len(predicted.split()),
            },
            feedback=f"WER: {wer:.2%}"
        )

    def _extract_transcript(self, prediction: Any) -> str:
        """Extract transcript from prediction."""
        if prediction is None:
            return ""
        if hasattr(prediction, "transcript"):
            return prediction.transcript or ""
        if isinstance(prediction, str):
            return prediction
        return ""

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate using Levenshtein distance."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Dynamic programming for edit distance
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,      # Deletion
                        d[i][j - 1] + 1,      # Insertion
                        d[i - 1][j - 1] + 1   # Substitution
                    )

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)


class AudioEventClassificationRubric(Rubric):
    """
    Evaluate audio event classification accuracy.

    Compares predicted audio event type against ground truth.
    Supports partial credit for related event types.
    """

    # Event similarity groups
    EVENT_GROUPS = {
        "speech": {"speech_normal", "speech_distress", "help_call"},
        "distress": {"speech_distress", "help_call", "scream"},
        "violence": {"fighting_sounds", "scream", "glass_breaking"},
        "medical": {"coughing", "speech_distress"},
    }

    def __init__(
        self,
        name: str = "audio_event_classification",
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
        """Evaluate audio event classification."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected = ground_truth.get("primary_event", "unknown").lower()

        # Extract predicted event
        if hasattr(prediction, "primary_event"):
            predicted = prediction.primary_event.value if hasattr(
                prediction.primary_event, "value"
            ) else str(prediction.primary_event).lower()
        else:
            predicted = "unknown"

        # Check exact match
        if expected == predicted:
            score = 1.0
            correct = True
        elif self.partial_credit:
            score = self._compute_partial_score(expected, predicted)
            correct = score >= 0.5
        else:
            score = 0.0
            correct = False

        return RubricResult(
            score=score,
            passed=correct,
            metrics={
                "event_correct": float(expected == predicted),
                "partial_score": score,
            },
            feedback=f"Expected '{expected}', got '{predicted}'"
        )

    def _compute_partial_score(self, expected: str, predicted: str) -> float:
        """Compute partial credit for related events."""
        for group_name, group_events in self.EVENT_GROUPS.items():
            if expected in group_events and predicted in group_events:
                return 0.5
        return 0.0


class DistressDetectionRubric(Rubric):
    """
    Evaluate distress call detection.

    Binary classification: did the agent correctly detect distress?
    Combines help_detected, severity, and keyword detection.
    """

    def __init__(
        self,
        name: str = "distress_detection",
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
        """Evaluate distress detection."""
        if prediction is None:
            expected = ground_truth.get("distress_detected", False)
            return RubricResult(
                score=0.0 if expected else 1.0,
                passed=not expected,
                feedback="No prediction provided"
            )

        expected = ground_truth.get("distress_detected", False) or ground_truth.get("help_detected", False)

        # Check multiple indicators
        help_detected = getattr(prediction, "help_detected", False)
        high_severity = False
        if hasattr(prediction, "severity"):
            severity_val = prediction.severity.value if hasattr(
                prediction.severity, "value"
            ) else int(prediction.severity)
            high_severity = severity_val >= 3

        predicted = help_detected or high_severity

        # Calculate score
        if expected == predicted:
            score = 1.0
        elif expected and not predicted:  # FN
            score = max(0.0, 1.0 - self.fn_penalty)
        else:  # FP
            score = 0.5

        return RubricResult(
            score=score,
            passed=expected == predicted,
            metrics={
                "distress_expected": float(expected),
                "distress_predicted": float(predicted),
                "help_detected": float(help_detected),
                "high_severity": float(high_severity),
            },
            feedback=f"Distress {'detected' if predicted else 'not detected'}"
        )


class KeywordDetectionRubric(Rubric):
    """
    Evaluate detection of specific keywords in audio.

    Measures recall of expected keywords and precision of detected keywords.
    """

    # Default emergency keywords
    DEFAULT_KEYWORDS = {
        "help", "911", "emergency", "police", "ambulance",
        "fire", "stop", "danger", "hurt", "pain"
    }

    def __init__(
        self,
        name: str = "keyword_detection",
        weight: float = 1.0,
        keywords: Optional[Set[str]] = None
    ):
        super().__init__(name, weight)
        self.keywords = keywords or self.DEFAULT_KEYWORDS

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate keyword detection."""
        expected_keywords = set(ground_truth.get("keywords_detected", []))
        expected_keywords = {k.lower() for k in expected_keywords}

        # Extract predicted keywords
        predicted_keywords: Set[str] = set()
        if hasattr(prediction, "keywords_detected"):
            predicted_keywords = {k.lower() for k in prediction.keywords_detected}
        elif hasattr(prediction, "transcript") and prediction.transcript:
            # Extract from transcript
            transcript_words = set(prediction.transcript.lower().split())
            predicted_keywords = transcript_words & self.keywords

        if not expected_keywords:
            # No keywords expected
            if not predicted_keywords:
                return RubricResult(score=1.0, passed=True, feedback="No keywords expected or detected")
            else:
                return RubricResult(
                    score=0.75,
                    passed=True,
                    metrics={"false_positives": len(predicted_keywords)},
                    feedback=f"Detected unexpected keywords: {predicted_keywords}"
                )

        # Calculate precision and recall
        true_positives = expected_keywords & predicted_keywords
        false_positives = predicted_keywords - expected_keywords
        false_negatives = expected_keywords - predicted_keywords

        precision = len(true_positives) / len(predicted_keywords) if predicted_keywords else 0.0
        recall = len(true_positives) / len(expected_keywords)

        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return RubricResult(
            score=f1,
            passed=recall >= 0.5,  # At least half of keywords detected
            metrics={
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": len(true_positives),
                "false_positives": len(false_positives),
                "false_negatives": len(false_negatives),
            },
            feedback=f"Detected {len(true_positives)}/{len(expected_keywords)} keywords (F1: {f1:.2f})"
        )


class ChaosLevelRubric(Rubric):
    """
    Evaluate chaos/noise level estimation.

    Compares predicted chaos level against ground truth.
    """

    def __init__(
        self,
        name: str = "chaos_level",
        weight: float = 1.0,
        tolerance: float = 0.2  # Allowed difference
    ):
        super().__init__(name, weight)
        self.tolerance = tolerance

    async def evaluate(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RubricResult:
        """Evaluate chaos level estimation."""
        if prediction is None:
            return RubricResult(score=0.0, passed=False, feedback="No prediction")

        expected = ground_truth.get("chaos_level", 0.0)
        predicted = getattr(prediction, "chaos_level", 0.0)

        difference = abs(expected - predicted)
        score = max(0.0, 1.0 - difference)

        return RubricResult(
            score=score,
            passed=difference <= self.tolerance,
            metrics={
                "expected_chaos": expected,
                "predicted_chaos": predicted,
                "difference": difference,
            },
            feedback=f"Chaos level: {predicted:.2f} (expected: {expected:.2f})"
        )
