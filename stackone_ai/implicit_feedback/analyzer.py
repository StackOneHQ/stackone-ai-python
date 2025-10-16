from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data import ToolCallQualitySignals, ToolCallRecord


@dataclass(frozen=True)
class BehaviorAnalyzerConfig:
    quick_refinement_window_seconds: float = 12.0
    task_switch_window_seconds: float = 180.0
    failure_penalty: float = 0.3
    quick_refinement_penalty: float = 0.25
    task_switch_penalty: float = 0.2


class BehaviorAnalyzer:
    """Derive behavioural quality signals from a stream of tool calls."""

    def __init__(self, config: BehaviorAnalyzerConfig | None = None) -> None:
        self._config = config or BehaviorAnalyzerConfig()

    def analyze(self, history: Sequence[ToolCallRecord], current: ToolCallRecord) -> ToolCallQualitySignals:
        """Compute quality signals for a tool call."""

        session_history = [
            call for call in history
            if call.session_id == current.session_id and call.call_id != current.call_id
        ]

        quick_refinement, refinement_window = self._detect_quick_refinement(session_history, current)
        task_switch = self._detect_task_switch(session_history, current)
        suitability_score = self._compute_suitability_score(current.status, quick_refinement, task_switch)

        return ToolCallQualitySignals(
            quick_refinement=quick_refinement,
            task_switch=task_switch,
            suitability_score=suitability_score,
            refinement_window_seconds=refinement_window,
        )

    def _detect_quick_refinement(
        self, history: Sequence[ToolCallRecord], current: ToolCallRecord
    ) -> tuple[bool, float | None]:
        if not current.session_id or not history:
            return False, None

        last_event = history[-1]
        elapsed = (current.start_time - last_event.end_time).total_seconds()
        if elapsed < 0:
            # Ignore out-of-order events
            return False, None

        if (
            last_event.tool_name == current.tool_name
            and elapsed <= self._config.quick_refinement_window_seconds
        ):
            return True, elapsed

        return False, None

    def _detect_task_switch(self, history: Sequence[ToolCallRecord], current: ToolCallRecord) -> bool:
        if not current.session_id or not history:
            return False

        for previous in reversed(history):
            elapsed = (current.start_time - previous.end_time).total_seconds()
            if elapsed < 0:
                continue
            if elapsed > self._config.task_switch_window_seconds:
                break
            if previous.tool_name != current.tool_name:
                return True

        return False

    def _compute_suitability_score(self, status: str, quick_refinement: bool, task_switch: bool) -> float:
        score = 1.0
        if status != "success":
            score -= self._config.failure_penalty
        if quick_refinement:
            score -= self._config.quick_refinement_penalty
        if task_switch:
            score -= self._config.task_switch_penalty
        return max(0.0, min(1.0, score))
