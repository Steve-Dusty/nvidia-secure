"""
Tests for Verifiers environments.

Tests:
- Environment initialization
- Dataset loading
- Step execution
- Result aggregation
"""

import sys
from pathlib import Path
import asyncio
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers.base import Dataset, TestCase, Environment, SingleTurnEnv, MultiTurnEnv, EnvironmentConfig
from verifiers.environments.integrated_env import IntegratedEmergencyEnv, MultiModalTestCase, load_environment
from verifiers.datasets.synthetic_generator import generate_integrated_dataset, generate_visual_dataset


class TestDataset:
    """Tests for Dataset class."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        test_cases = [
            TestCase(id="tc1", data={"input": 1}, ground_truth={"output": 1}),
            TestCase(id="tc2", data={"input": 2}, ground_truth={"output": 2}),
        ]
        dataset = Dataset(name="test", inputs=test_cases)

        assert len(dataset) == 2
        assert dataset.name == "test"
        assert dataset[0].id == "tc1"

    def test_dataset_iteration(self):
        """Test dataset iteration."""
        test_cases = [
            TestCase(id=f"tc{i}", data=i, ground_truth={"value": i})
            for i in range(5)
        ]
        dataset = Dataset(name="test", inputs=test_cases)

        ids = [tc.id for tc in dataset]
        assert ids == ["tc0", "tc1", "tc2", "tc3", "tc4"]

    def test_dataset_filter(self):
        """Test dataset filtering."""
        test_cases = [
            TestCase(id=f"tc{i}", data=i, ground_truth={"value": i})
            for i in range(10)
        ]
        dataset = Dataset(name="test", inputs=test_cases)

        filtered = dataset.filter(lambda tc: tc.data > 5)
        assert len(filtered) == 4  # 6, 7, 8, 9

    def test_dataset_sample(self):
        """Test dataset sampling."""
        test_cases = [
            TestCase(id=f"tc{i}", data=i, ground_truth={"value": i})
            for i in range(100)
        ]
        dataset = Dataset(name="test", inputs=test_cases)

        sampled = dataset.sample(10, seed=42)
        assert len(sampled) == 10


class TestSyntheticGenerator:
    """Tests for synthetic data generation."""

    def test_generate_integrated_dataset(self):
        """Test integrated dataset generation."""
        dataset = generate_integrated_dataset(
            scenario="all",
            num_cases=5,
            max_frames=5,
            seed=42
        )

        assert len(dataset) == 5
        assert "integrated" in dataset.name

        # Check test case structure
        tc = dataset[0]
        assert hasattr(tc, 'id')
        assert hasattr(tc, 'data')

    def test_generate_visual_dataset(self):
        """Test visual dataset generation."""
        dataset = generate_visual_dataset(
            scenario="fall",
            num_cases=5,
            num_frames=3,
            seed=42
        )

        assert len(dataset) == 5
        assert "visual" in dataset.name

    def test_multimodal_test_case(self):
        """Test MultiModalTestCase structure."""
        dataset = generate_integrated_dataset(
            scenario="fall",
            num_cases=1,
            max_frames=5,
            seed=42
        )

        tc = dataset[0]
        mm_tc = tc.data if hasattr(tc, 'data') else tc

        assert hasattr(mm_tc, 'frames')
        assert hasattr(mm_tc, 'audio_chunks')
        assert hasattr(mm_tc, 'ground_truth')
        assert hasattr(mm_tc, 'scenario_type')
        assert len(mm_tc.frames) == 5


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, dispatch_at_turn: int = 3):
        self.dispatch_at_turn = dispatch_at_turn
        self.call_count = 0

    def analyze_frame(self, frame, audio=None):
        """Mock analysis returning a result object."""
        self.call_count += 1

        class MockResult:
            def __init__(self, call_count, dispatch_at):
                self.emergency_type = type('EmergencyType', (), {'value': 'fall'})()
                self.overall_severity = type('Severity', (), {'value': 2, 'name': 'MEDIUM'})()
                self.dispatch_recommended = call_count >= dispatch_at
                self.confidence = 0.8
                self.fight_detected = False
                self.fall_detected = call_count >= dispatch_at - 1
                self.visual_result = None
                self.audio_result = None

        return MockResult(self.call_count, self.dispatch_at_turn)


class TestIntegratedEnvironment:
    """Tests for IntegratedEmergencyEnv."""

    def test_environment_creation(self):
        """Test environment initialization."""
        env = load_environment(
            scenario="fall",
            num_test_cases=3,
            max_turns=5
        )

        assert env is not None
        assert len(env.dataset) == 3
        assert len(env.rubrics) > 0

    def test_environment_reset(self):
        """Test environment reset."""
        env = load_environment(scenario="fall", num_test_cases=2, max_turns=5)

        test_case = env.dataset[0]
        state = env.reset(test_case)

        assert state is not None
        assert env.current_turn == 0
        assert env.alert_triggered == False

    @pytest.mark.asyncio
    async def test_environment_step(self):
        """Test single step execution."""
        env = load_environment(scenario="fall", num_test_cases=2, max_turns=5)
        agent = MockAgent(dispatch_at_turn=3)

        test_case = env.dataset[0]
        env.reset(test_case)

        result = await env.step(agent, test_case)

        assert result is not None
        assert agent.call_count == 1
        assert env.current_turn == 1

    @pytest.mark.asyncio
    async def test_environment_run(self):
        """Test full environment run."""
        env = load_environment(scenario="fall", num_test_cases=2, max_turns=5)
        agent = MockAgent(dispatch_at_turn=3)

        results = await env.run(agent)

        assert "overall_score" in results
        assert "rubric_scores" in results
        assert "total_episodes" in results
        assert results["total_episodes"] == 2


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnvironmentConfig(name="test")

        assert config.name == "test"
        assert config.timeout_ms == 30000
        assert config.max_retries == 3
        assert "memory_mb" in config.resource_limits

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnvironmentConfig(
            name="custom",
            timeout_ms=5000,
            max_retries=5,
            resource_limits={"memory_mb": 512}
        )

        assert config.timeout_ms == 5000
        assert config.max_retries == 5
        assert config.resource_limits["memory_mb"] == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
