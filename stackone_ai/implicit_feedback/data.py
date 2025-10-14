from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, MutableMapping

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ToolCallQualitySignals:
    """Derived behavioural signals describing a tool execution."""

    quick_refinement: bool = False
    task_switch: bool = False
    suitability_score: float = 1.0
    refinement_window_seconds: float | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    """Mutable-free record of a tool call for behaviour analysis."""

    call_id: str
    tool_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: int
    session_id: str | None
    user_id: str | None
    status: str
    params: Any | None = None
    result: Any | None = None
    error: str | None = None
    quality: ToolCallQualitySignals | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_quality(self, quality: ToolCallQualitySignals) -> "ToolCallRecord":
        """Return a new instance with computed quality signals."""

        data: MutableMapping[str, Any] = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }
        return ToolCallRecord(quality=quality, **data)


@dataclass(frozen=True)
class ImplicitFeedbackEvent:
    """Implicit user feedback inferred from behaviour."""

    name: str
    score: float | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
