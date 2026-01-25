"""
Test environments for agent evaluation.

Environments combine datasets with rubrics to create complete
testing scenarios. Available environments:

- IntegratedEmergencyEnv: Multi-modal (visual+audio) emergency detection
- FallDetectionEnv: Visual fall detection
- FightDetectionEnv: Violence detection
- DistressCallEnv: Audio distress detection
- MedicalEmergencyEnv: Emergency response routing
"""

from .integrated_env import (
    IntegratedEmergencyEnv,
    MultiModalTestCase,
    load_environment as load_integrated_environment,
)

__all__ = [
    # Primary environment
    "IntegratedEmergencyEnv",
    "MultiModalTestCase",
    "load_integrated_environment",
]
