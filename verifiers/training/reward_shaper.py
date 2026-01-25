"""
Reward shaping for converting rubric evaluations to RL rewards.

Converts multi-rubric evaluation results into scalar rewards
suitable for reinforcement learning training.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Rubric, RubricResult


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Discount factor for future rewards
    gamma: float = 0.99

    # Whether to normalize rewards
    normalize: bool = True
    normalize_window: int = 100  # Window size for running normalization

    # Reward clipping
    clip_min: float = -10.0
    clip_max: float = 10.0

    # Bonus/penalty multipliers
    correct_dispatch_bonus: float = 1.0
    missed_dispatch_penalty: float = -2.0
    false_alarm_penalty: float = -0.5
    early_detection_bonus: float = 0.5  # Per turn early

    # Rubric weight overrides (if different from rubric.weight)
    rubric_weights: Dict[str, float] = field(default_factory=dict)


class RewardShaper:
    """
    Converts rubric scores to RL rewards.

    Supports:
    - Weighted combination of rubric scores
    - Reward normalization (running mean/std)
    - Reward shaping with bonuses/penalties
    - Clipping for stability
    """

    def __init__(
        self,
        rubrics: List[Rubric],
        config: Optional[RewardConfig] = None
    ):
        """
        Initialize reward shaper.

        Args:
            rubrics: List of rubrics to use for reward computation
            config: Reward configuration
        """
        self.rubrics = rubrics
        self.config = config or RewardConfig()

        # State for normalization
        self.reward_history: List[float] = []
        self._running_mean = 0.0
        self._running_std = 1.0

    async def compute_reward(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute reward from rubric evaluations.

        Args:
            prediction: Agent's prediction/action
            ground_truth: Expected output
            context: Additional context (turn number, etc.)

        Returns:
            Scalar reward value
        """
        # Evaluate all rubrics
        rubric_results = await self._evaluate_rubrics(prediction, ground_truth, context)

        # Compute base reward from rubric scores
        base_reward = self._compute_base_reward(rubric_results)

        # Apply reward shaping
        shaped_reward = self._apply_shaping(base_reward, prediction, ground_truth, context)

        # Normalize if enabled
        if self.config.normalize:
            shaped_reward = self._normalize_reward(shaped_reward)

        # Clip reward
        clipped_reward = np.clip(shaped_reward, self.config.clip_min, self.config.clip_max)

        return float(clipped_reward)

    async def _evaluate_rubrics(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, RubricResult]:
        """Evaluate all rubrics."""
        results = {}
        for rubric in self.rubrics:
            try:
                result = await rubric.evaluate(prediction, ground_truth, context)
                results[rubric.name] = result
            except Exception as e:
                # On error, return zero score
                results[rubric.name] = RubricResult(
                    score=0.0,
                    passed=False,
                    feedback=f"Error: {str(e)}"
                )
        return results

    def _compute_base_reward(self, rubric_results: Dict[str, RubricResult]) -> float:
        """Compute weighted average reward from rubric scores."""
        total_reward = 0.0
        total_weight = 0.0

        for rubric in self.rubrics:
            result = rubric_results.get(rubric.name)
            if result is None:
                continue

            # Use override weight if specified
            weight = self.config.rubric_weights.get(rubric.name, rubric.weight)

            # Convert score (0-1) to reward
            # Score of 0.5 -> reward of 0
            # Score of 1.0 -> reward of +weight
            # Score of 0.0 -> reward of -weight
            reward = (result.score - 0.5) * 2 * weight

            total_reward += reward
            total_weight += weight

        # Normalize by weight
        if total_weight > 0:
            return total_reward / total_weight
        return 0.0

    def _apply_shaping(
        self,
        base_reward: float,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Apply reward shaping bonuses and penalties."""
        reward = base_reward

        # Dispatch decision shaping
        expected_dispatch = ground_truth.get("dispatch_recommended", False)
        predicted_dispatch = getattr(prediction, "dispatch_recommended", False) if prediction else False

        if expected_dispatch and predicted_dispatch:
            # Correct dispatch
            reward += self.config.correct_dispatch_bonus
        elif expected_dispatch and not predicted_dispatch:
            # Missed dispatch (dangerous)
            reward += self.config.missed_dispatch_penalty
        elif not expected_dispatch and predicted_dispatch:
            # False alarm
            reward += self.config.false_alarm_penalty

        # Early detection bonus
        if context and expected_dispatch and predicted_dispatch:
            current_turn = context.get("turn", 0)
            expected_turn = ground_truth.get("alert_turn", current_turn)
            if current_turn < expected_turn:
                turns_early = expected_turn - current_turn
                reward += self.config.early_detection_bonus * turns_early

        return reward

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        # Add to history
        self.reward_history.append(reward)

        # Keep only recent rewards
        if len(self.reward_history) > self.config.normalize_window:
            self.reward_history = self.reward_history[-self.config.normalize_window:]

        # Need minimum samples for normalization
        if len(self.reward_history) < 10:
            return reward

        # Update running stats
        self._running_mean = np.mean(self.reward_history)
        self._running_std = np.std(self.reward_history) + 1e-8

        # Normalize
        return (reward - self._running_mean) / self._running_std

    def compute_returns(
        self,
        rewards: List[float],
        dones: Optional[List[bool]] = None
    ) -> List[float]:
        """
        Compute discounted returns.

        Args:
            rewards: List of step rewards
            dones: Optional list of episode termination flags

        Returns:
            List of discounted returns
        """
        returns = []
        G = 0.0

        for i in reversed(range(len(rewards))):
            # Reset return at episode boundaries
            if dones and dones[i]:
                G = 0.0

            G = rewards[i] + self.config.gamma * G
            returns.insert(0, G)

        return returns

    def compute_advantages(
        self,
        rewards: List[float],
        values: Optional[List[float]] = None,
        dones: Optional[List[bool]] = None
    ) -> List[float]:
        """
        Compute advantage estimates using GAE.

        Args:
            rewards: List of step rewards
            values: Optional value estimates (uses rewards if not provided)
            dones: Optional episode termination flags

        Returns:
            List of advantage estimates
        """
        returns = self.compute_returns(rewards, dones)

        if values is None:
            # Simple advantage: return - immediate reward
            values = rewards

        advantages = [r - v for r, v in zip(returns, values)]
        return advantages

    def reset(self):
        """Reset normalization state."""
        self.reward_history = []
        self._running_mean = 0.0
        self._running_std = 1.0

    def get_stats(self) -> Dict[str, float]:
        """Get current reward statistics."""
        if not self.reward_history:
            return {
                "mean": 0.0,
                "std": 1.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        return {
            "mean": float(np.mean(self.reward_history)),
            "std": float(np.std(self.reward_history)),
            "min": float(np.min(self.reward_history)),
            "max": float(np.max(self.reward_history)),
            "count": len(self.reward_history),
        }


class MultiObjectiveRewardShaper(RewardShaper):
    """
    Reward shaper with multiple objective tracking.

    Tracks separate rewards for different objectives (e.g., accuracy vs latency)
    for multi-objective optimization.
    """

    def __init__(
        self,
        rubrics: List[Rubric],
        objectives: Dict[str, List[str]],  # objective -> rubric names
        config: Optional[RewardConfig] = None
    ):
        super().__init__(rubrics, config)
        self.objectives = objectives
        self.objective_history: Dict[str, List[float]] = {
            obj: [] for obj in objectives
        }

    async def compute_reward(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute scalar reward (primary objective)."""
        return await super().compute_reward(prediction, ground_truth, context)

    async def compute_objective_rewards(
        self,
        prediction: Any,
        ground_truth: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Compute rewards for each objective."""
        rubric_results = await self._evaluate_rubrics(prediction, ground_truth, context)

        objective_rewards = {}
        for objective, rubric_names in self.objectives.items():
            total = 0.0
            count = 0
            for name in rubric_names:
                if name in rubric_results:
                    total += rubric_results[name].score
                    count += 1
            obj_reward = total / count if count > 0 else 0.0
            objective_rewards[objective] = obj_reward
            self.objective_history[objective].append(obj_reward)

        return objective_rewards

    def get_objective_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each objective."""
        stats = {}
        for objective, history in self.objective_history.items():
            if history:
                stats[objective] = {
                    "mean": float(np.mean(history)),
                    "std": float(np.std(history)),
                    "min": float(np.min(history)),
                    "max": float(np.max(history)),
                }
            else:
                stats[objective] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return stats
