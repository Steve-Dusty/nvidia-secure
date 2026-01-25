"""
Adapter for exporting trajectories to prime-rl format.

Prime-RL expects:
- States: Environmental observations
- Actions: Model outputs/predictions
- Rewards: Scalar rewards
- Done flags: Episode termination

Supports export for:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning (RL)
- Preference Learning (DPO/RLHF)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Trajectory, TrajectoryStep


@dataclass
class ExportConfig:
    """Configuration for data export."""
    output_dir: str = "./training_data"

    # SFT settings
    sft_min_reward: float = 0.5  # Only use trajectories with reward >= this
    sft_format: str = "input_output"  # "input_output" or "chat"

    # RL settings
    include_returns: bool = True
    include_advantages: bool = True
    gamma: float = 0.99

    # Preference learning
    preference_pairs: bool = False  # Generate preference pairs


class PrimeRLExporter:
    """
    Export trajectories for prime-rl training.

    Supports multiple export formats:
    - SFT: For supervised fine-tuning
    - RL: For reinforcement learning
    - Preference: For preference learning (DPO/RLHF)
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def export_for_sft(
        self,
        trajectories: List[Trajectory],
        output_file: str = "sft_data.jsonl"
    ) -> str:
        """
        Export for supervised fine-tuning.

        Only includes successful trajectories (reward >= threshold).
        Formats as input-output pairs.

        Args:
            trajectories: Trajectories to export
            output_file: Output filename

        Returns:
            Path to exported file
        """
        lines = []

        for traj in trajectories:
            # Filter by reward threshold
            if traj.total_reward < self.config.sft_min_reward:
                continue

            # Export each successful step
            for step in traj.steps:
                if step.reward < 0:
                    continue  # Skip negative reward steps

                if self.config.sft_format == "chat":
                    example = self._format_chat_example(step, traj)
                else:
                    example = self._format_input_output_example(step, traj)

                lines.append(json.dumps(example))

        path = Path(self.config.output_dir) / output_file
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return str(path)

    def export_for_rl(
        self,
        trajectories: List[Trajectory],
        output_file: str = "rl_data.jsonl"
    ) -> str:
        """
        Export for reinforcement learning.

        Includes full trajectory data with returns and advantages.

        Args:
            trajectories: Trajectories to export
            output_file: Output filename

        Returns:
            Path to exported file
        """
        lines = []

        for traj in trajectories:
            rl_example = {
                "trajectory_id": str(id(traj)),
                "environment": traj.environment_name,
                "agent": traj.agent_name,
                "states": [self._format_state(s.state) for s in traj.steps],
                "actions": [self._format_action(s.action) for s in traj.steps],
                "rewards": [s.reward for s in traj.steps],
                "dones": [s.done for s in traj.steps],
                "total_reward": traj.total_reward,
            }

            if self.config.include_returns:
                rl_example["returns"] = traj.compute_returns(self.config.gamma)

            if self.config.include_advantages:
                rl_example["advantages"] = traj.compute_advantages(self.config.gamma)

            rl_example["metadata"] = traj.metadata

            lines.append(json.dumps(rl_example))

        path = Path(self.config.output_dir) / output_file
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return str(path)

    def export_for_preference(
        self,
        trajectories: List[Trajectory],
        output_file: str = "preference_data.jsonl"
    ) -> str:
        """
        Export for preference learning (DPO/RLHF).

        Creates pairs of (chosen, rejected) based on reward.

        Args:
            trajectories: Trajectories to export
            output_file: Output filename

        Returns:
            Path to exported file
        """
        # Sort trajectories by reward
        sorted_trajs = sorted(trajectories, key=lambda t: t.total_reward, reverse=True)

        lines = []
        n = len(sorted_trajs)

        # Create pairs: top half vs bottom half
        for i in range(n // 2):
            chosen = sorted_trajs[i]
            rejected = sorted_trajs[n - 1 - i]

            if chosen.total_reward <= rejected.total_reward:
                continue  # Skip if not clearly better

            # Format as preference pair
            pair = {
                "prompt": self._get_trajectory_prompt(chosen),
                "chosen": self._get_trajectory_response(chosen),
                "rejected": self._get_trajectory_response(rejected),
                "chosen_reward": chosen.total_reward,
                "rejected_reward": rejected.total_reward,
                "environment": chosen.environment_name,
            }
            lines.append(json.dumps(pair))

        path = Path(self.config.output_dir) / output_file
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return str(path)

    def export_all(
        self,
        trajectories: List[Trajectory],
        prefix: str = ""
    ) -> Dict[str, str]:
        """
        Export in all formats.

        Args:
            trajectories: Trajectories to export
            prefix: Filename prefix

        Returns:
            Dictionary mapping format to filepath
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_" if prefix else ""

        paths = {
            "sft": self.export_for_sft(
                trajectories,
                f"{prefix}sft_{timestamp}.jsonl"
            ),
            "rl": self.export_for_rl(
                trajectories,
                f"{prefix}rl_{timestamp}.jsonl"
            ),
        }

        if self.config.preference_pairs:
            paths["preference"] = self.export_for_preference(
                trajectories,
                f"{prefix}preference_{timestamp}.jsonl"
            )

        return paths

    def _format_state(self, state: Dict) -> str:
        """Format state for model input."""
        parts = []

        if "visual" in state:
            parts.append(f"Visual observation: {state['visual']}")
        if "audio" in state:
            parts.append(f"Audio observation: {state['audio']}")
        if "turn" in state:
            parts.append(f"Turn: {state['turn']}")
        if "context" in state:
            parts.append(f"Context: {state['context']}")

        if not parts:
            return json.dumps(state)

        return " | ".join(parts)

    def _format_action(self, action: Dict) -> str:
        """Format action for model output."""
        if "emergency_type" in action:
            return json.dumps({
                "emergency_type": action.get("emergency_type"),
                "severity": action.get("overall_severity") or action.get("severity"),
                "dispatch": action.get("dispatch_recommended", False),
                "response": action.get("recommended_response", ""),
                "priority": action.get("response_priority"),
            })
        return json.dumps(action)

    def _format_input_output_example(
        self,
        step: TrajectoryStep,
        trajectory: Trajectory
    ) -> Dict[str, Any]:
        """Format as input-output pair for SFT."""
        return {
            "input": self._format_state(step.state),
            "output": self._format_action(step.action),
            "reward": step.reward,
            "environment": trajectory.environment_name,
        }

    def _format_chat_example(
        self,
        step: TrajectoryStep,
        trajectory: Trajectory
    ) -> Dict[str, Any]:
        """Format as chat messages for SFT."""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an emergency detection agent. Environment: {trajectory.environment_name}"
                },
                {
                    "role": "user",
                    "content": self._format_state(step.state)
                },
                {
                    "role": "assistant",
                    "content": self._format_action(step.action)
                }
            ],
            "reward": step.reward,
        }

    def _get_trajectory_prompt(self, trajectory: Trajectory) -> str:
        """Get prompt from trajectory (first state)."""
        if trajectory.steps:
            return self._format_state(trajectory.steps[0].state)
        return ""

    def _get_trajectory_response(self, trajectory: Trajectory) -> str:
        """Get response from trajectory (final action)."""
        if trajectory.steps:
            return self._format_action(trajectory.steps[-1].action)
        return ""


def export_for_sft(
    trajectories: List[Trajectory],
    output_dir: str = "./training_data",
    min_reward: float = 0.5
) -> str:
    """
    Convenience function for SFT export.

    Args:
        trajectories: Trajectories to export
        output_dir: Output directory
        min_reward: Minimum reward threshold

    Returns:
        Path to exported file
    """
    config = ExportConfig(output_dir=output_dir, sft_min_reward=min_reward)
    exporter = PrimeRLExporter(config)
    return exporter.export_for_sft(trajectories)


def export_for_rl(
    trajectories: List[Trajectory],
    output_dir: str = "./training_data",
    gamma: float = 0.99
) -> str:
    """
    Convenience function for RL export.

    Args:
        trajectories: Trajectories to export
        output_dir: Output directory
        gamma: Discount factor

    Returns:
        Path to exported file
    """
    config = ExportConfig(output_dir=output_dir, gamma=gamma)
    exporter = PrimeRLExporter(config)
    return exporter.export_for_rl(trajectories)


def load_trajectories(filepath: str) -> List[Dict[str, Any]]:
    """
    Load trajectories from JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of trajectory dictionaries
    """
    trajectories = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories
