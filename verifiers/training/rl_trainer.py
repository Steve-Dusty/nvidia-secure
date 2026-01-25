"""
Reinforcement Learning trainer for agent fine-tuning.

Implements trajectory-based training compatible with prime-rl patterns.
Collects experience from environments and exports for training.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import asyncio
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Environment, Trajectory, TrajectoryStep
from verifiers.training.reward_shaper import RewardShaper, RewardConfig


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    # Training loop
    num_epochs: int = 100
    batch_size: int = 32
    max_steps_per_episode: int = 50

    # Checkpointing
    checkpoint_every: int = 10
    checkpoint_dir: str = "./checkpoints"

    # Reward shaping
    gamma: float = 0.99
    normalize_rewards: bool = True

    # Export
    export_format: str = "jsonl"
    export_dir: str = "./training_data"

    # Logging
    log_every: int = 1
    verbose: bool = True


class RLTrainer:
    """
    Reinforcement Learning trainer for emergency detection agents.

    Implements trajectory collection and training data export
    compatible with prime-rl training framework.

    Features:
    - Async trajectory collection
    - Multi-environment support
    - Reward shaping from rubrics
    - Checkpointing and resume
    - JSONL export for prime-rl
    """

    def __init__(
        self,
        environment: Environment,
        agent: Any,
        reward_shaper: RewardShaper,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            environment: Test environment
            agent: Agent to train
            reward_shaper: Reward computation
            config: Training configuration
        """
        self.environment = environment
        self.agent = agent
        self.reward_shaper = reward_shaper
        self.config = config or TrainingConfig()

        # Training state
        self.trajectories: List[Trajectory] = []
        self.episode_count = 0
        self.total_steps = 0
        self.epoch_results: List[Dict[str, Any]] = []

        # Callbacks
        self.on_trajectory_complete: Optional[Callable[[Trajectory], None]] = None
        self.on_step: Optional[Callable[[TrajectoryStep], None]] = None
        self.on_epoch_complete: Optional[Callable[[Dict[str, Any]], None]] = None

        # Ensure directories exist
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.export_dir).mkdir(parents=True, exist_ok=True)

    async def collect_trajectory(self, test_case: Any) -> Trajectory:
        """
        Collect a single trajectory from the environment.

        Args:
            test_case: Test case to run

        Returns:
            Complete trajectory with states, actions, rewards
        """
        steps: List[TrajectoryStep] = []

        # Reset environment
        if hasattr(self.environment, 'reset'):
            state = self.environment.reset(test_case)
        else:
            state = {"input": test_case}

        done = False
        turn = 0
        max_turns = getattr(self.environment, 'max_turns', self.config.max_steps_per_episode)

        while not done and turn < max_turns:
            # Get current state
            current_state = {
                "turn": turn,
                **state
            }

            # Agent produces action (prediction)
            try:
                action = await self.environment.step(self.agent, test_case)
            except Exception as e:
                print(f"Agent error at turn {turn}: {e}")
                action = None

            # Get ground truth from test case
            ground_truth = test_case.ground_truth if hasattr(test_case, 'ground_truth') else {}

            # Compute reward from rubrics
            reward = await self.reward_shaper.compute_reward(
                action,
                ground_truth,
                {"turn": turn, "state": current_state}
            )

            # Check if done
            done = self._check_done(action, test_case, turn, max_turns)

            # Create trajectory step
            step = TrajectoryStep(
                state=current_state,
                action=self._action_to_dict(action),
                reward=reward,
                next_state={"turn": turn + 1},
                done=done,
                info={
                    "turn": turn,
                    "test_case_id": getattr(test_case, 'id', str(turn))
                }
            )
            steps.append(step)

            # Callback
            if self.on_step:
                self.on_step(step)

            turn += 1
            self.total_steps += 1

            # Update state
            state = step.next_state

        # Create trajectory
        trajectory = Trajectory(
            environment_name=self.environment.config.name if hasattr(self.environment, 'config') else "unknown",
            agent_name=type(self.agent).__name__,
            steps=steps,
            total_reward=sum(s.reward for s in steps),
            metadata={
                "turns": turn,
                "test_case_id": getattr(test_case, 'id', None),
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.episode_count += 1

        # Callback
        if self.on_trajectory_complete:
            self.on_trajectory_complete(trajectory)

        return trajectory

    async def train_epoch(self) -> Dict[str, Any]:
        """
        Run one training epoch over all dataset items.

        Returns:
            Epoch statistics
        """
        epoch_trajectories: List[Trajectory] = []

        for test_case in self.environment.dataset:
            trajectory = await self.collect_trajectory(test_case)
            epoch_trajectories.append(trajectory)
            self.trajectories.append(trajectory)

        # Compute epoch statistics
        rewards = [t.total_reward for t in epoch_trajectories]
        lengths = [len(t) for t in epoch_trajectories]

        epoch_result = {
            "epoch_trajectories": len(epoch_trajectories),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "total_steps": self.total_steps,
            "total_episodes": self.episode_count,
            "reward_stats": self.reward_shaper.get_stats(),
        }

        self.epoch_results.append(epoch_result)
        return epoch_result

    async def train(
        self,
        num_epochs: Optional[int] = None,
        checkpoint_every: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            num_epochs: Override config num_epochs
            checkpoint_every: Override config checkpoint_every

        Returns:
            Training summary
        """
        num_epochs = num_epochs or self.config.num_epochs
        checkpoint_every = checkpoint_every or self.config.checkpoint_every

        if self.config.verbose:
            print(f"Starting training: {num_epochs} epochs")
            print(f"Environment: {self.environment.config.name if hasattr(self.environment, 'config') else 'unknown'}")
            print(f"Dataset size: {len(self.environment.dataset)}")
            print()

        for epoch in range(num_epochs):
            epoch_start = datetime.now()

            # Run epoch
            epoch_result = await self.train_epoch()
            epoch_result["epoch"] = epoch
            epoch_result["duration_seconds"] = (datetime.now() - epoch_start).total_seconds()

            # Logging
            if self.config.verbose and (epoch + 1) % self.config.log_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: "
                      f"reward={epoch_result['mean_reward']:.4f} "
                      f"(std={epoch_result['std_reward']:.4f}), "
                      f"steps={epoch_result['total_steps']}")

            # Checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                self._save_checkpoint(epoch)
                if self.config.verbose:
                    print(f"  Checkpoint saved at epoch {epoch + 1}")

            # Callback
            if self.on_epoch_complete:
                self.on_epoch_complete(epoch_result)

        # Final export
        self.export_trajectories()

        return {
            "epochs": self.epoch_results,
            "total_trajectories": len(self.trajectories),
            "total_steps": self.total_steps,
            "final_reward_stats": self.reward_shaper.get_stats(),
        }

    def export_trajectories(
        self,
        format: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Export trajectories for external training.

        Args:
            format: Export format ("jsonl" or "json")
            output_dir: Output directory

        Returns:
            Path to exported file
        """
        format = format or self.config.export_format
        output_dir = output_dir or self.config.export_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectories_{timestamp}.{format}"
        filepath = Path(output_dir) / filename

        if format == "jsonl":
            with open(filepath, "w") as f:
                for traj in self.trajectories:
                    f.write(json.dumps(traj.to_training_format()) + "\n")
        else:  # json
            data = {
                "trajectories": [t.to_training_format() for t in self.trajectories],
                "metadata": {
                    "total_trajectories": len(self.trajectories),
                    "total_steps": self.total_steps,
                    "exported_at": datetime.now().isoformat(),
                }
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        if self.config.verbose:
            print(f"Exported {len(self.trajectories)} trajectories to {filepath}")

        return str(filepath)

    def _check_done(
        self,
        action: Any,
        test_case: Any,
        turn: int,
        max_turns: int
    ) -> bool:
        """Check if episode is done."""
        # Done if dispatch recommended for emergency detection
        if hasattr(action, 'dispatch_recommended') and action.dispatch_recommended:
            return True

        # Done if max turns reached
        if turn >= max_turns - 1:
            return True

        return False

    def _action_to_dict(self, action: Any) -> Dict[str, Any]:
        """Convert agent action to dictionary."""
        if action is None:
            return {"error": "no_action"}

        if isinstance(action, dict):
            return action

        # Extract attributes from dataclass or object
        result = {}
        for attr in ["emergency_type", "overall_severity", "dispatch_recommended",
                     "confidence", "recommended_response", "response_priority"]:
            if hasattr(action, attr):
                val = getattr(action, attr)
                if hasattr(val, "value"):
                    result[attr] = val.value
                elif hasattr(val, "name"):
                    result[attr] = val.name
                else:
                    result[attr] = val

        return result if result else {"raw": str(action)}

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "config": {
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "gamma": self.config.gamma,
            },
            "epoch_results": self.epoch_results,
            "reward_stats": self.reward_shaper.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        filepath = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.json"
        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch to resume from
        """
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        self.total_steps = checkpoint["total_steps"]
        self.episode_count = checkpoint["episode_count"]
        self.epoch_results = checkpoint.get("epoch_results", [])

        if self.config.verbose:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint["epoch"] + 1

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.epoch_results:
            return {"status": "not_started"}

        recent_rewards = [e["mean_reward"] for e in self.epoch_results[-10:]]

        return {
            "status": "in_progress" if len(self.epoch_results) < self.config.num_epochs else "completed",
            "epochs_completed": len(self.epoch_results),
            "total_trajectories": len(self.trajectories),
            "total_steps": self.total_steps,
            "current_mean_reward": recent_rewards[-1] if recent_rewards else 0.0,
            "reward_trend": "improving" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "stable",
            "recent_rewards": recent_rewards,
        }


class ParallelTrainer(RLTrainer):
    """
    Trainer that collects trajectories in parallel.

    Useful for faster data collection with multiple agents/environments.
    """

    def __init__(
        self,
        environments: List[Environment],
        agents: List[Any],
        reward_shaper: RewardShaper,
        config: Optional[TrainingConfig] = None
    ):
        # Use first environment/agent as primary
        super().__init__(environments[0], agents[0], reward_shaper, config)
        self.environments = environments
        self.agents = agents

    async def train_epoch(self) -> Dict[str, Any]:
        """Run epoch with parallel trajectory collection."""
        tasks = []

        # Create tasks for all environment-agent pairs
        for env, agent in zip(self.environments, self.agents):
            for test_case in env.dataset:
                # Create temporary trainer for this pair
                temp_trainer = RLTrainer(env, agent, self.reward_shaper, self.config)
                tasks.append(temp_trainer.collect_trajectory(test_case))

        # Run all trajectories in parallel
        epoch_trajectories = await asyncio.gather(*tasks)

        # Collect results
        self.trajectories.extend(epoch_trajectories)

        rewards = [t.total_reward for t in epoch_trajectories]
        return {
            "epoch_trajectories": len(epoch_trajectories),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "total_steps": self.total_steps,
        }
