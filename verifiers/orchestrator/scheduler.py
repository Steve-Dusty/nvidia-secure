"""
Benchmark scheduling and management.

Schedules benchmark runs, manages queues, and handles
recurring evaluations.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verifiers.base import Environment
from verifiers.orchestrator.runner import AgentRunner, AgentConfig, BenchmarkResult


class RunStatus(Enum):
    """Status of a scheduled run."""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledRun:
    """A scheduled benchmark run."""
    id: str
    agent_config: AgentConfig
    environment: Environment
    run_at: datetime
    status: RunStatus = RunStatus.SCHEDULED
    priority: int = 0  # Higher = more important
    result: Optional[BenchmarkResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent": self.agent_config.name,
            "environment": self.environment.config.name if hasattr(self.environment, 'config') else "unknown",
            "run_at": self.run_at.isoformat(),
            "status": self.status.value,
            "priority": self.priority,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class BenchmarkScheduler:
    """
    Schedules and manages benchmark runs.

    Features:
    - Time-based scheduling
    - Priority queuing
    - Recurring runs
    - Status tracking
    """

    def __init__(self, runner: Optional[AgentRunner] = None):
        """
        Initialize scheduler.

        Args:
            runner: AgentRunner to use for execution
        """
        self.runner = runner or AgentRunner()
        self.scheduled_runs: Dict[str, ScheduledRun] = {}
        self.run_history: List[ScheduledRun] = []

        # Callbacks
        self.on_run_complete: Optional[Callable[[ScheduledRun], None]] = None
        self.on_run_failed: Optional[Callable[[ScheduledRun], None]] = None

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def schedule(
        self,
        agent_config: AgentConfig,
        environment: Environment,
        run_at: Optional[datetime] = None,
        priority: int = 0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Schedule a benchmark run.

        Args:
            agent_config: Agent to evaluate
            environment: Environment to run in
            run_at: When to run (default: now)
            priority: Run priority (higher = more important)
            metadata: Additional metadata

        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())[:8]
        run = ScheduledRun(
            id=run_id,
            agent_config=agent_config,
            environment=environment,
            run_at=run_at or datetime.now(),
            priority=priority,
            metadata=metadata or {},
        )

        self.scheduled_runs[run_id] = run
        return run_id

    def schedule_recurring(
        self,
        agent_config: AgentConfig,
        environment: Environment,
        interval: timedelta,
        count: int = 10,
        start_at: Optional[datetime] = None
    ) -> List[str]:
        """
        Schedule recurring benchmark runs.

        Args:
            agent_config: Agent to evaluate
            environment: Environment to run in
            interval: Time between runs
            count: Number of runs to schedule
            start_at: First run time (default: now)

        Returns:
            List of run IDs
        """
        run_ids = []
        current_time = start_at or datetime.now()

        for i in range(count):
            run_id = self.schedule(
                agent_config,
                environment,
                run_at=current_time,
                metadata={"recurring": True, "sequence": i}
            )
            run_ids.append(run_id)
            current_time += interval

        return run_ids

    def cancel(self, run_id: str) -> bool:
        """
        Cancel a scheduled run.

        Args:
            run_id: ID of run to cancel

        Returns:
            True if cancelled, False if not found or already running
        """
        if run_id not in self.scheduled_runs:
            return False

        run = self.scheduled_runs[run_id]
        if run.status == RunStatus.SCHEDULED:
            run.status = RunStatus.CANCELLED
            return True
        return False

    async def execute_run(self, run: ScheduledRun) -> ScheduledRun:
        """
        Execute a single scheduled run.

        Args:
            run: Run to execute

        Returns:
            Updated run with result
        """
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now()

        try:
            result = await self.runner.run_benchmark(
                run.agent_config,
                run.environment
            )
            run.result = result
            run.status = RunStatus.COMPLETED

            if self.on_run_complete:
                self.on_run_complete(run)

        except Exception as e:
            run.error = str(e)
            run.status = RunStatus.FAILED

            if self.on_run_failed:
                self.on_run_failed(run)

        run.completed_at = datetime.now()
        self.run_history.append(run)

        return run

    async def execute_scheduled(self) -> List[ScheduledRun]:
        """
        Execute all scheduled runs that are due.

        Returns:
            List of completed runs
        """
        now = datetime.now()
        completed = []

        # Get due runs, sorted by priority
        due_runs = [
            run for run in self.scheduled_runs.values()
            if run.status == RunStatus.SCHEDULED and run.run_at <= now
        ]
        due_runs.sort(key=lambda r: -r.priority)  # Higher priority first

        for run in due_runs:
            result = await self.execute_run(run)
            completed.append(result)

            # Remove from scheduled
            del self.scheduled_runs[run.id]

        return completed

    async def run_loop(self, check_interval: float = 1.0):
        """
        Background loop that executes scheduled runs.

        Args:
            check_interval: Seconds between checks
        """
        self._running = True

        while self._running:
            await self.execute_scheduled()
            await asyncio.sleep(check_interval)

    def start_background(self, check_interval: float = 1.0):
        """Start background execution loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_loop(check_interval))

    def stop_background(self):
        """Stop background execution loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    def get_run(self, run_id: str) -> Optional[ScheduledRun]:
        """Get a scheduled run by ID."""
        return self.scheduled_runs.get(run_id)

    def get_pending(self) -> List[ScheduledRun]:
        """Get all pending scheduled runs."""
        return [
            run for run in self.scheduled_runs.values()
            if run.status == RunStatus.SCHEDULED
        ]

    def get_history(self, limit: int = 100) -> List[ScheduledRun]:
        """Get run history."""
        return self.run_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        status_counts = {}
        for run in self.scheduled_runs.values():
            status = run.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "running": self._running,
            "scheduled_count": len(self.scheduled_runs),
            "history_count": len(self.run_history),
            "status_breakdown": status_counts,
            "next_run": min(
                (r.run_at for r in self.scheduled_runs.values()
                 if r.status == RunStatus.SCHEDULED),
                default=None
            ),
        }

    def export_history(self, filepath: str):
        """Export run history to JSON."""
        data = {
            "history": [run.to_dict() for run in self.run_history],
            "exported_at": datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def clear_history(self):
        """Clear run history."""
        self.run_history = []
