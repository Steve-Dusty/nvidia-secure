"""
Tests for Verifiers RL training components.

Tests:
- Reward shaping
- Trajectory collection
- Training loop
- Export adapters
"""

import sys
from pathlib import Path
import asyncio
import pytest
import json
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from verifiers.base import Trajectory, TrajectoryStep, Rubric, RubricResult
from verifiers.training.reward_shaper import RewardShaper, RewardConfig
from verifiers.training.rl_trainer import RLTrainer, TrainingConfig
from verifiers.training.prime_rl_adapter import PrimeRLExporter, ExportConfig


class SimpleRubric(Rubric):
    """Simple rubric for testing."""

    async def evaluate(self, prediction, ground_truth, context=None):
        score = 1.0 if prediction == ground_truth.get("expected") else 0.0
        return RubricResult(score=score, passed=score > 0.5)


class TestRewardShaper:
    """Tests for RewardShaper."""

    @pytest.fixture
    def rubrics(self):
        return [
            SimpleRubric(name="test1", weight=1.0),
            SimpleRubric(name="test2", weight=2.0),
        ]

    @pytest.fixture
    def shaper(self, rubrics):
        config = RewardConfig(normalize=False)
        return RewardShaper(rubrics, config)

    @pytest.mark.asyncio
    async def test_compute_reward_correct(self, shaper):
        """Test reward for correct prediction."""
        prediction = "correct"
        ground_truth = {"expected": "correct"}

        reward = await shaper.compute_reward(prediction, ground_truth)

        # Both rubrics score 1.0, weighted average should be positive
        assert reward > 0

    @pytest.mark.asyncio
    async def test_compute_reward_wrong(self, shaper):
        """Test reward for wrong prediction."""
        prediction = "wrong"
        ground_truth = {"expected": "correct"}

        reward = await shaper.compute_reward(prediction, ground_truth)

        # Both rubrics score 0.0, reward should be negative
        assert reward < 0

    def test_compute_returns(self, shaper):
        """Test discounted returns computation."""
        rewards = [1.0, 0.5, 0.25]
        returns = shaper.compute_returns(rewards, None)

        assert len(returns) == 3
        # Returns should be discounted sums
        assert returns[0] > returns[1] > returns[2]

    def test_compute_advantages(self, shaper):
        """Test advantage computation."""
        rewards = [1.0, 0.5, 0.25]
        advantages = shaper.compute_advantages(rewards)

        assert len(advantages) == 3

    def test_reward_stats(self, rubrics):
        """Test reward statistics tracking."""
        config = RewardConfig(normalize=True, normalize_window=10)
        shaper = RewardShaper(rubrics, config)

        # Record some rewards
        for r in [0.5, 0.6, 0.7, 0.8]:
            shaper.reward_history.append(r)

        stats = shaper.get_stats()

        assert "mean" in stats
        assert "std" in stats
        assert stats["count"] == 4


class TestTrajectory:
    """Tests for Trajectory class."""

    @pytest.fixture
    def trajectory(self):
        steps = [
            TrajectoryStep(
                state={"turn": 0},
                action={"output": "a"},
                reward=1.0,
                done=False
            ),
            TrajectoryStep(
                state={"turn": 1},
                action={"output": "b"},
                reward=0.5,
                done=False
            ),
            TrajectoryStep(
                state={"turn": 2},
                action={"output": "c"},
                reward=0.25,
                done=True
            ),
        ]
        return Trajectory(
            environment_name="test_env",
            agent_name="test_agent",
            steps=steps,
            total_reward=1.75
        )

    def test_trajectory_length(self, trajectory):
        """Test trajectory length."""
        assert len(trajectory) == 3

    def test_to_training_format(self, trajectory):
        """Test conversion to training format."""
        data = trajectory.to_training_format()

        assert data["env"] == "test_env"
        assert data["agent"] == "test_agent"
        assert len(data["states"]) == 3
        assert len(data["actions"]) == 3
        assert len(data["rewards"]) == 3
        assert data["total_reward"] == 1.75

    def test_compute_returns(self, trajectory):
        """Test return computation."""
        returns = trajectory.compute_returns(gamma=0.99)

        assert len(returns) == 3
        assert returns[0] > returns[2]  # First return should be highest

    def test_compute_advantages(self, trajectory):
        """Test advantage computation."""
        advantages = trajectory.compute_advantages(gamma=0.99)

        assert len(advantages) == 3


class TestPrimeRLExporter:
    """Tests for export adapters."""

    @pytest.fixture
    def trajectories(self):
        good_traj = Trajectory(
            environment_name="test",
            agent_name="good_agent",
            steps=[
                TrajectoryStep(
                    state={"turn": 0},
                    action={"emergency_type": "fall", "dispatch_recommended": True},
                    reward=0.8,
                    done=True
                )
            ],
            total_reward=0.8
        )
        bad_traj = Trajectory(
            environment_name="test",
            agent_name="bad_agent",
            steps=[
                TrajectoryStep(
                    state={"turn": 0},
                    action={"emergency_type": "none", "dispatch_recommended": False},
                    reward=0.2,
                    done=True
                )
            ],
            total_reward=0.2
        )
        return [good_traj, bad_traj]

    def test_export_for_sft(self, trajectories):
        """Test SFT export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(output_dir=tmpdir, sft_min_reward=0.5)
            exporter = PrimeRLExporter(config)

            path = exporter.export_for_sft(trajectories, "test_sft.jsonl")

            assert Path(path).exists()

            with open(path) as f:
                lines = f.readlines()

            # Only good trajectory should be exported (reward >= 0.5)
            assert len(lines) == 1

    def test_export_for_rl(self, trajectories):
        """Test RL export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(output_dir=tmpdir)
            exporter = PrimeRLExporter(config)

            path = exporter.export_for_rl(trajectories, "test_rl.jsonl")

            assert Path(path).exists()

            with open(path) as f:
                lines = f.readlines()

            # Both trajectories should be exported
            assert len(lines) == 2

            # Check structure
            data = json.loads(lines[0])
            assert "trajectory_id" in data
            assert "states" in data
            assert "actions" in data
            assert "rewards" in data
            assert "returns" in data
            assert "advantages" in data

    def test_export_for_preference(self, trajectories):
        """Test preference learning export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportConfig(output_dir=tmpdir, preference_pairs=True)
            exporter = PrimeRLExporter(config)

            path = exporter.export_for_preference(trajectories, "test_pref.jsonl")

            assert Path(path).exists()

            with open(path) as f:
                lines = f.readlines()

            # Should have 1 pair (good vs bad)
            assert len(lines) == 1

            pair = json.loads(lines[0])
            assert "chosen" in pair
            assert "rejected" in pair
            assert pair["chosen_reward"] > pair["rejected_reward"]


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()

        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.gamma == 0.99
        assert config.normalize_rewards == True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            num_epochs=50,
            batch_size=16,
            gamma=0.95
        )

        assert config.num_epochs == 50
        assert config.batch_size == 16
        assert config.gamma == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
