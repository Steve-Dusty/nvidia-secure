"""
Evaluation rubrics for agent testing.

Rubrics define how agent outputs are scored against ground truth.
Available rubrics:
- Visual: Action classification, fall/fight detection, bounding box IoU
- Audio: Speech recognition, event classification, distress detection
- Response: Urgency scoring, dispatch routing, facility matching
- Latency: Performance benchmarking
- Composite: Multi-modal combined evaluation
"""

from .visual_rubric import (
    ActionClassificationRubric,
    FallDetectionRubric,
    FightDetectionRubric,
    BoundingBoxIoURubric,
    PersonCountRubric,
    SeverityClassificationRubric,
)

from .audio_rubric import (
    SpeechRecognitionRubric,
    AudioEventClassificationRubric,
    DistressDetectionRubric,
    KeywordDetectionRubric,
)

from .response_rubric import (
    UrgencyScoreRubric,
    DispatchRoutingRubric,
    FacilityMatchingRubric,
    ResponseQualityRubric,
)

from .latency_rubric import (
    LatencyRubric,
    ThroughputRubric,
)

from .composite_rubric import (
    CompositeRubric,
    EmergencyClassificationRubric,
    MultiModalFusionRubric,
)

__all__ = [
    # Visual
    "ActionClassificationRubric",
    "FallDetectionRubric",
    "FightDetectionRubric",
    "BoundingBoxIoURubric",
    "PersonCountRubric",
    "SeverityClassificationRubric",

    # Audio
    "SpeechRecognitionRubric",
    "AudioEventClassificationRubric",
    "DistressDetectionRubric",
    "KeywordDetectionRubric",

    # Response
    "UrgencyScoreRubric",
    "DispatchRoutingRubric",
    "FacilityMatchingRubric",
    "ResponseQualityRubric",

    # Latency
    "LatencyRubric",
    "ThroughputRubric",

    # Composite
    "CompositeRubric",
    "EmergencyClassificationRubric",
    "MultiModalFusionRubric",
]
