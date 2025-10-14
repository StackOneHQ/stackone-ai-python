from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable

from .analyzer import BehaviorAnalyzer
from .data import ImplicitFeedbackEvent, ToolCallRecord

logger = logging.getLogger("stackone.implicit_feedback")


class SessionTracker:
    """Maintain per-session tool call history and emit implicit feedback events."""

    def __init__(
        self,
        analyzer: BehaviorAnalyzer,
        max_history: int = 32,
        suitability_alert_threshold: float = 0.4,
    ) -> None:
        self._analyzer = analyzer
        self._max_history = max_history
        self._suitability_alert_threshold = suitability_alert_threshold
        self._history: Dict[str, Deque[ToolCallRecord]] = defaultdict(deque)

    def record_tool_call(self, record: ToolCallRecord) -> tuple[ToolCallRecord, list[ImplicitFeedbackEvent]]:
        """Record a tool execution and return derived events."""

        session_key = self._resolve_session_key(record)
        session_history = list(self._history.get(session_key, deque()))
        quality = self._analyzer.analyze(session_history, record)
        enriched_record = record.with_quality(quality)

        history = self._history[session_key]
        history.append(enriched_record)
        if len(history) > self._max_history:
            history.popleft()

        events = self._build_events(enriched_record)
        return enriched_record, events

    def _resolve_session_key(self, record: ToolCallRecord) -> str:
        if record.session_id:
            return record.session_id
        if record.user_id:
            return f"user:{record.user_id}"
        return "global"

    def _build_events(self, record: ToolCallRecord) -> list[ImplicitFeedbackEvent]:
        quality = record.quality
        if quality is None:
            return []

        events: list[ImplicitFeedbackEvent] = []
        if quality.quick_refinement:
            events.append(
                ImplicitFeedbackEvent(
                    name="refinement_needed",
                    payload={
                        "tool_name": record.tool_name,
                        "duration_ms": record.duration_ms,
                        "refinement_window_seconds": quality.refinement_window_seconds,
                    },
                )
            )
        if quality.task_switch:
            events.append(
                ImplicitFeedbackEvent(
                    name="task_switch_detected",
                    payload={
                        "tool_name": record.tool_name,
                        "suitability_score": quality.suitability_score,
                    },
                )
            )
        if quality.suitability_score <= self._suitability_alert_threshold:
            events.append(
                ImplicitFeedbackEvent(
                    name="low_suitability",
                    score=quality.suitability_score,
                    payload={
                        "tool_name": record.tool_name,
                        "status": record.status,
                    },
                )
            )
        if events:
            logger.debug(
                "Implicit feedback events generated",
                extra={
                    "session_id": record.session_id,
                    "user_id": record.user_id,
                    "events": [event.name for event in events],
                },
            )
        return events

    def iter_session_history(self, session_id: str) -> Iterable[ToolCallRecord]:
        return tuple(self._history.get(session_id, ()))
