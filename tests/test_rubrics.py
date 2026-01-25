"""
Tests for Verifiers rubrics.

Tests all rubric types for correct evaluation behavior.
"""

import sys
from pathlib import Path
import asyncio
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers.base import RubricResult
from verifiers.rubrics.visual_rubric import (
    ActionClassificationRubric,
    FallDetectionRubric,
    FightDetectionRubric,
    BoundingBoxIoURubric,
    PersonCountRubric,
    SeverityClassificationRubric,
)
from verifiers.rubrics.audio_rubric import (
    SpeechRecognitionRubric,
    AudioEventClassificationRubric,
    DistressDetectionRubric,
    KeywordDetectionRubric,
)
from verifiers.rubrics.response_rubric import (
    UrgencyScoreRubric,
    DispatchRoutingRubric,
    FacilityMatchingRubric,
)
from verifiers.rubrics.latency_rubric import (
    LatencyRubric,
    ThroughputRubric,
)
from verifiers.rubrics.composite_rubric import (
    CompositeRubric,
    EmergencyClassificationRubric,
)


class MockVisualPrediction:
    """Mock visual analysis result."""

    def __init__(
        self,
        action: str = "standing",
        fall_detected: bool = False,
        fight_detected: bool = False,
        severity: int = 0,
        persons: list = None
    ):
        self.scene_action = type('Action', (), {'value': action})()
        self.action_confidence = 0.85
        self.fall_detected = fall_detected
        self.fight_detected = fight_detected
        self.scene_severity = type('Severity', (), {'value': severity, 'name': 'MEDIUM'})()
        self.persons = persons or []
        self.persons_detected = len(self.persons)


class MockAudioPrediction:
    """Mock audio analysis result."""

    def __init__(
        self,
        primary_event: str = "speech_normal",
        help_detected: bool = False,
        transcript: str = "",
        keywords: list = None,
        severity: int = 0
    ):
        self.primary_event = type('Event', (), {'value': primary_event})()
        self.help_detected = help_detected
        self.transcript = transcript
        self.keywords_detected = keywords or []
        self.severity = type('Severity', (), {'value': severity})()


class MockResponsePrediction:
    """Mock response recommendation."""

    def __init__(
        self,
        urgency_score: float = 5.0,
        call_number: str = "911",
        dispatch: bool = False
    ):
        self.urgency_score = urgency_score
        self.urgency_level = "HIGH" if urgency_score > 7 else "MODERATE"
        self.call_number = call_number
        self.dispatch_recommended = dispatch


class TestVisualRubrics:
    """Tests for visual detection rubrics."""

    @pytest.mark.asyncio
    async def test_action_classification_correct(self):
        """Test correct action classification."""
        rubric = ActionClassificationRubric()
        prediction = MockVisualPrediction(action="falling")
        ground_truth = {"action": "falling"}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0
        assert result.passed == True

    @pytest.mark.asyncio
    async def test_action_classification_wrong(self):
        """Test incorrect action classification."""
        rubric = ActionClassificationRubric(partial_credit=False)
        prediction = MockVisualPrediction(action="standing")
        ground_truth = {"action": "falling"}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 0.0
        assert result.passed == False

    @pytest.mark.asyncio
    async def test_action_classification_partial(self):
        """Test partial credit for similar actions."""
        rubric = ActionClassificationRubric(partial_credit=True)
        prediction = MockVisualPrediction(action="walking")
        ground_truth = {"action": "running"}  # Both in "movement" group

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 0.5
        assert result.passed == True

    @pytest.mark.asyncio
    async def test_fall_detection_true_positive(self):
        """Test fall detection - true positive."""
        rubric = FallDetectionRubric()
        prediction = MockVisualPrediction(fall_detected=True)
        ground_truth = {"fall_detected": True}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0
        assert result.metrics["true_positive"] == 1

    @pytest.mark.asyncio
    async def test_fall_detection_false_negative(self):
        """Test fall detection - false negative (penalized)."""
        rubric = FallDetectionRubric(false_negative_penalty=1.5)
        prediction = MockVisualPrediction(fall_detected=False)
        ground_truth = {"fall_detected": True}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score < 0.5  # Penalized
        assert result.metrics["false_negative"] == 1

    @pytest.mark.asyncio
    async def test_fight_detection(self):
        """Test fight detection."""
        rubric = FightDetectionRubric()
        prediction = MockVisualPrediction(fight_detected=True)
        ground_truth = {"fight_detected": True}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0
        assert result.passed == True


class TestAudioRubrics:
    """Tests for audio detection rubrics."""

    @pytest.mark.asyncio
    async def test_distress_detection(self):
        """Test distress detection."""
        rubric = DistressDetectionRubric()
        prediction = MockAudioPrediction(help_detected=True)
        ground_truth = {"distress_detected": True}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0
        assert result.passed == True

    @pytest.mark.asyncio
    async def test_keyword_detection(self):
        """Test keyword detection."""
        rubric = KeywordDetectionRubric()
        prediction = MockAudioPrediction(
            transcript="help please call 911",
            keywords=["help", "911"]
        )
        ground_truth = {"keywords_detected": ["help", "911"]}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0  # All keywords found
        assert result.metrics["recall"] == 1.0

    @pytest.mark.asyncio
    async def test_audio_event_classification(self):
        """Test audio event classification."""
        rubric = AudioEventClassificationRubric()
        prediction = MockAudioPrediction(primary_event="speech_distress")
        ground_truth = {"primary_event": "speech_distress"}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0
        assert result.passed == True


class TestResponseRubrics:
    """Tests for response quality rubrics."""

    @pytest.mark.asyncio
    async def test_urgency_score_correct(self):
        """Test urgency score accuracy."""
        rubric = UrgencyScoreRubric(tolerance=1.0)
        prediction = MockResponsePrediction(urgency_score=8.0)
        ground_truth = {"urgency_score": 8.5}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score > 0.9
        assert result.passed == True  # Within tolerance

    @pytest.mark.asyncio
    async def test_dispatch_routing_correct(self):
        """Test correct dispatch routing."""
        rubric = DispatchRoutingRubric()
        prediction = MockResponsePrediction(call_number="911")
        ground_truth = {"call_number": "911"}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 1.0
        assert result.passed == True

    @pytest.mark.asyncio
    async def test_dispatch_routing_missed_911(self):
        """Test missed 911 (penalized more)."""
        rubric = DispatchRoutingRubric(false_311_penalty=1.0)
        prediction = MockResponsePrediction(call_number="311")
        ground_truth = {"call_number": "911"}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score == 0.0  # Full penalty
        assert result.passed == False


class TestLatencyRubrics:
    """Tests for performance rubrics."""

    @pytest.mark.asyncio
    async def test_latency_within_target(self):
        """Test latency within target."""
        rubric = LatencyRubric(target_latency_ms=200, max_latency_ms=500)

        result = await rubric.evaluate(None, {}, {"latency_ms": 150})

        assert result.score == 1.0
        assert result.passed == True

    @pytest.mark.asyncio
    async def test_latency_over_max(self):
        """Test latency over maximum."""
        rubric = LatencyRubric(target_latency_ms=200, max_latency_ms=500)

        result = await rubric.evaluate(None, {}, {"latency_ms": 600})

        assert result.score < 0.5
        assert result.passed == False


class TestCompositeRubrics:
    """Tests for composite rubrics."""

    @pytest.mark.asyncio
    async def test_emergency_classification(self):
        """Test emergency type classification."""
        rubric = EmergencyClassificationRubric()

        class MockPred:
            emergency_type = type('ET', (), {'value': 'fall'})()

        result = await rubric.evaluate(MockPred(), {"emergency_type": "fall"})

        assert result.score == 1.0
        assert result.passed == True

    @pytest.mark.asyncio
    async def test_composite_rubric(self):
        """Test composite of multiple rubrics."""
        sub_rubrics = [
            FallDetectionRubric(name="fall", weight=2.0),
            FightDetectionRubric(name="fight", weight=1.0),
        ]
        rubric = CompositeRubric(name="composite", rubrics=sub_rubrics)

        prediction = MockVisualPrediction(fall_detected=True, fight_detected=False)
        ground_truth = {"fall_detected": True, "fight_detected": False}

        result = await rubric.evaluate(prediction, ground_truth)

        assert result.score > 0.5
        assert "fall_score" in result.metrics
        assert "fight_score" in result.metrics


class TestRubricResult:
    """Tests for RubricResult dataclass."""

    def test_score_clamping(self):
        """Test that scores are clamped to 0-1."""
        result = RubricResult(score=1.5, passed=True)
        assert result.score == 1.0

        result = RubricResult(score=-0.5, passed=False)
        assert result.score == 0.0

    def test_result_with_metrics(self):
        """Test result with metrics."""
        result = RubricResult(
            score=0.8,
            passed=True,
            metrics={"accuracy": 0.9, "latency": 100},
            feedback="Good performance"
        )

        assert result.metrics["accuracy"] == 0.9
        assert result.feedback == "Good performance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
