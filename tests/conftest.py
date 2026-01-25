"""
Pytest configuration and fixtures for verifiers tests.
"""

import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth data for testing."""
    return {
        "emergency_type": "fall",
        "severity": "high",
        "dispatch_recommended": True,
        "fall_detected": True,
        "fight_detected": False,
        "distress_detected": False,
        "keywords_detected": ["help"],
        "alert_turn": 3,
    }
