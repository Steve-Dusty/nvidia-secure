"""
Synthetic test data generator for agent evaluation.

Generates realistic test cases for:
- Visual detection (falls, fights, normal activity)
- Audio detection (distress calls, keywords, ambient)
- Integrated multi-modal scenarios

Uses numpy for frame/audio generation without requiring
actual video/audio files.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Dataset, TestCase


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    emergency_type: str
    severity: str
    dispatch_required: bool
    alert_turn: int  # Turn when alert should trigger (-1 for none)
    keywords: List[str]
    visual_features: Dict[str, Any]
    audio_features: Dict[str, Any]


# Predefined scenario configurations
SCENARIOS = {
    "fall": ScenarioConfig(
        name="fall_detection",
        emergency_type="fall",
        severity="high",
        dispatch_required=True,
        alert_turn=3,
        keywords=["help", "fell"],
        visual_features={
            "person_count": 1,
            "fall_detected": True,
            "fight_detected": False,
            "action_sequence": ["standing", "walking", "falling", "lying_down"],
        },
        audio_features={
            "distress_detected": True,
            "chaos_level": 0.3,
        }
    ),
    "fight": ScenarioConfig(
        name="fight_detection",
        emergency_type="violence",
        severity="critical",
        dispatch_required=True,
        alert_turn=2,
        keywords=["stop", "help", "police"],
        visual_features={
            "person_count": 2,
            "fall_detected": False,
            "fight_detected": True,
            "action_sequence": ["standing", "fighting", "fighting", "fighting"],
        },
        audio_features={
            "distress_detected": True,
            "chaos_level": 0.8,
        }
    ),
    "distress": ScenarioConfig(
        name="distress_call",
        emergency_type="distress",
        severity="high",
        dispatch_required=True,
        alert_turn=2,
        keywords=["help", "911", "emergency"],
        visual_features={
            "person_count": 1,
            "fall_detected": False,
            "fight_detected": False,
            "action_sequence": ["standing", "standing", "standing"],
        },
        audio_features={
            "distress_detected": True,
            "chaos_level": 0.4,
        }
    ),
    "medical": ScenarioConfig(
        name="medical_emergency",
        emergency_type="medical",
        severity="critical",
        dispatch_required=True,
        alert_turn=2,
        keywords=["ambulance", "hurt", "pain", "help"],
        visual_features={
            "person_count": 2,
            "fall_detected": True,
            "fight_detected": False,
            "action_sequence": ["standing", "falling", "lying_down", "lying_down"],
        },
        audio_features={
            "distress_detected": True,
            "chaos_level": 0.5,
        }
    ),
    "normal": ScenarioConfig(
        name="normal_activity",
        emergency_type="none",
        severity="none",
        dispatch_required=False,
        alert_turn=-1,
        keywords=[],
        visual_features={
            "person_count": 3,
            "fall_detected": False,
            "fight_detected": False,
            "action_sequence": ["standing", "walking", "standing", "walking"],
        },
        audio_features={
            "distress_detected": False,
            "chaos_level": 0.1,
        }
    ),
}


class SyntheticFrameGenerator:
    """
    Generate synthetic video frames for testing.

    Creates numpy arrays that simulate video frames with
    embedded features for detection testing.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        channels: int = 3,
        seed: Optional[int] = None
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self.rng = np.random.default_rng(seed)

    def generate_frame(
        self,
        person_count: int = 1,
        action: str = "standing",
        include_fall: bool = False,
        include_fight: bool = False
    ) -> np.ndarray:
        """
        Generate a synthetic frame.

        The frame encodes scenario information in a way that
        can be used for testing without actual visual content.
        """
        # Create base frame with random noise
        frame = self.rng.integers(50, 200, size=(self.height, self.width, self.channels), dtype=np.uint8)

        # Encode scenario information in specific regions
        # This allows mock agents to "detect" the encoded features

        # Person count encoded in top-left corner intensity
        frame[0:10, 0:10] = person_count * 25

        # Action encoded in top-right corner
        action_codes = {
            "standing": 50,
            "walking": 100,
            "running": 150,
            "fighting": 200,
            "falling": 225,
            "lying_down": 250,
        }
        frame[0:10, -10:] = action_codes.get(action, 50)

        # Fall indicator in bottom-left
        if include_fall:
            frame[-20:, 0:20] = 255

        # Fight indicator in bottom-right
        if include_fight:
            frame[-20:, -20:] = 255

        return frame

    def generate_sequence(
        self,
        config: ScenarioConfig,
        num_frames: int = 10
    ) -> List[np.ndarray]:
        """Generate a sequence of frames for a scenario."""
        frames = []
        actions = config.visual_features.get("action_sequence", ["standing"])

        for i in range(num_frames):
            action_idx = min(i, len(actions) - 1)
            action = actions[action_idx]

            frame = self.generate_frame(
                person_count=config.visual_features.get("person_count", 1),
                action=action,
                include_fall=config.visual_features.get("fall_detected", False) and i >= config.alert_turn,
                include_fight=config.visual_features.get("fight_detected", False) and i >= config.alert_turn - 1,
            )
            frames.append(frame)

        return frames


class SyntheticAudioGenerator:
    """
    Generate synthetic audio data for testing.

    Creates numpy arrays that simulate audio with
    embedded features for detection testing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 1000,
        seed: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.rng = np.random.default_rng(seed)

    def generate_chunk(
        self,
        include_speech: bool = False,
        include_distress: bool = False,
        chaos_level: float = 0.0,
        keywords: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate a synthetic audio chunk.

        Encodes audio features that can be detected by mock agents.
        """
        # Base audio: low-level noise
        audio = self.rng.normal(0, 0.01, self.chunk_samples).astype(np.float32)

        # Add speech-like patterns
        if include_speech:
            t = np.linspace(0, self.chunk_duration_ms / 1000, self.chunk_samples)
            speech = 0.1 * np.sin(2 * np.pi * 200 * t)  # 200Hz base
            speech += 0.05 * np.sin(2 * np.pi * 400 * t)  # Harmonic
            audio += speech.astype(np.float32)

        # Add distress indicators (higher frequency, louder)
        if include_distress:
            t = np.linspace(0, self.chunk_duration_ms / 1000, self.chunk_samples)
            distress = 0.3 * np.sin(2 * np.pi * 800 * t)  # High pitch
            audio += distress.astype(np.float32)

        # Add chaos (random noise bursts)
        if chaos_level > 0:
            chaos_mask = self.rng.random(self.chunk_samples) < chaos_level * 0.1
            audio[chaos_mask] += self.rng.normal(0, chaos_level * 0.5, chaos_mask.sum()).astype(np.float32)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9

        return audio

    def generate_sequence(
        self,
        config: ScenarioConfig,
        num_chunks: int = 10
    ) -> List[np.ndarray]:
        """Generate a sequence of audio chunks for a scenario."""
        chunks = []

        for i in range(num_chunks):
            include_distress = (
                config.audio_features.get("distress_detected", False) and
                i >= config.alert_turn
            )

            chunk = self.generate_chunk(
                include_speech=True,
                include_distress=include_distress,
                chaos_level=config.audio_features.get("chaos_level", 0.0),
                keywords=config.keywords if i >= config.alert_turn else None,
            )
            chunks.append(chunk)

        return chunks


def generate_integrated_dataset(
    scenario: str = "all",
    num_cases: int = 10,
    max_frames: int = 10,
    seed: Optional[int] = None
) -> Dataset:
    """
    Generate an integrated multi-modal dataset.

    Args:
        scenario: Specific scenario or "all" for mixed
        num_cases: Number of test cases to generate
        max_frames: Maximum frames per test case
        seed: Random seed for reproducibility

    Returns:
        Dataset with MultiModalTestCase instances
    """
    from verifiers.environments.integrated_env import MultiModalTestCase

    frame_gen = SyntheticFrameGenerator(seed=seed)
    audio_gen = SyntheticAudioGenerator(seed=seed)

    rng = np.random.default_rng(seed)
    test_cases = []

    # Select scenarios
    if scenario == "all":
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [scenario] if scenario in SCENARIOS else ["normal"]

    for i in range(num_cases):
        # Select scenario (cycle through if "all")
        scenario_name = scenario_names[i % len(scenario_names)]
        config = SCENARIOS[scenario_name]

        # Generate frames and audio
        frames = frame_gen.generate_sequence(config, max_frames)
        audio_chunks = audio_gen.generate_sequence(config, max_frames)

        # Build ground truth
        ground_truth = {
            "emergency_type": config.emergency_type,
            "severity": config.severity,
            "dispatch_recommended": config.dispatch_required,
            "alert_turn": config.alert_turn,
            "fall_detected": config.visual_features.get("fall_detected", False),
            "fight_detected": config.visual_features.get("fight_detected", False),
            "distress_detected": config.audio_features.get("distress_detected", False),
            "keywords_detected": config.keywords,
            "persons_count": config.visual_features.get("person_count", 1),
            # Per-turn ground truth
            "per_turn": {
                turn: {
                    "action": config.visual_features["action_sequence"][
                        min(turn, len(config.visual_features["action_sequence"]) - 1)
                    ]
                }
                for turn in range(max_frames)
            }
        }

        tc = MultiModalTestCase(
            id=f"{scenario_name}_{i:03d}",
            frames=frames,
            audio_chunks=audio_chunks,
            ground_truth=ground_truth,
            scenario_type=scenario_name,
            metadata={
                "generated": True,
                "seed": seed,
                "config_name": config.name,
            }
        )
        test_cases.append(tc)

    return Dataset(
        name=f"integrated_{scenario}_{num_cases}",
        inputs=test_cases,
        metadata={
            "scenario": scenario,
            "num_cases": num_cases,
            "max_frames": max_frames,
            "seed": seed,
        }
    )


def generate_visual_dataset(
    scenario: str = "fall",
    num_cases: int = 10,
    num_frames: int = 5,
    seed: Optional[int] = None
) -> Dataset:
    """Generate visual-only dataset for single-turn environments."""
    frame_gen = SyntheticFrameGenerator(seed=seed)
    test_cases = []

    config = SCENARIOS.get(scenario, SCENARIOS["normal"])

    for i in range(num_cases):
        frames = frame_gen.generate_sequence(config, num_frames)

        # Use last frame as the test input (most likely to show event)
        ground_truth = {
            "fall_detected": config.visual_features.get("fall_detected", False),
            "fight_detected": config.visual_features.get("fight_detected", False),
            "action": config.visual_features["action_sequence"][-1],
            "persons_count": config.visual_features.get("person_count", 1),
            "severity": config.severity,
        }

        tc = TestCase(
            id=f"visual_{scenario}_{i:03d}",
            data=frames[-1],  # Single frame
            ground_truth=ground_truth,
            metadata={"scenario": scenario, "frame_index": num_frames - 1}
        )
        test_cases.append(tc)

    return Dataset(
        name=f"visual_{scenario}_{num_cases}",
        inputs=test_cases,
        metadata={"scenario": scenario, "type": "visual"}
    )


def generate_audio_dataset(
    scenario: str = "distress",
    num_cases: int = 10,
    num_chunks: int = 3,
    seed: Optional[int] = None
) -> Dataset:
    """Generate audio-only dataset for audio testing."""
    audio_gen = SyntheticAudioGenerator(seed=seed)
    test_cases = []

    config = SCENARIOS.get(scenario, SCENARIOS["normal"])

    for i in range(num_cases):
        chunks = audio_gen.generate_sequence(config, num_chunks)

        ground_truth = {
            "distress_detected": config.audio_features.get("distress_detected", False),
            "help_detected": "help" in config.keywords,
            "keywords_detected": config.keywords,
            "chaos_level": config.audio_features.get("chaos_level", 0.0),
            "primary_event": "speech_distress" if config.audio_features.get("distress_detected") else "speech_normal",
        }

        # Concatenate chunks for single input
        audio_data = np.concatenate(chunks)

        tc = TestCase(
            id=f"audio_{scenario}_{i:03d}",
            data=audio_data,
            ground_truth=ground_truth,
            metadata={"scenario": scenario, "num_chunks": num_chunks}
        )
        test_cases.append(tc)

    return Dataset(
        name=f"audio_{scenario}_{num_cases}",
        inputs=test_cases,
        metadata={"scenario": scenario, "type": "audio"}
    )
