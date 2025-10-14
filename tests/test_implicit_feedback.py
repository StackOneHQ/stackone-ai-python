from __future__ import annotations

from datetime import datetime, timedelta, timezone

from stackone_ai.implicit_feedback import (
    BehaviorAnalyzer,
    BehaviorAnalyzerConfig,
    ImplicitFeedbackManager,
    SessionTracker,
)
from stackone_ai.implicit_feedback.data import ToolCallRecord


def test_behavior_analyzer_detects_quick_refinement() -> None:
    analyzer = BehaviorAnalyzer(BehaviorAnalyzerConfig(quick_refinement_window_seconds=15.0))

    start = datetime.now(timezone.utc)
    first_record = ToolCallRecord(
        call_id="first",
        tool_name="crm.search",
        start_time=start,
        end_time=start + timedelta(seconds=2),
        duration_ms=2000,
        session_id="session-1",
        user_id="user-1",
        status="success",
        params={"query": "alpha"},
        result={"count": 1},
    )

    second_start = first_record.end_time + timedelta(seconds=3)
    second_record = ToolCallRecord(
        call_id="second",
        tool_name="crm.search",
        start_time=second_start,
        end_time=second_start + timedelta(seconds=1),
        duration_ms=1000,
        session_id="session-1",
        user_id="user-1",
        status="success",
        params={"query": "alpha"},
        result={"count": 0},
    )

    signals = analyzer.analyze([first_record], second_record)

    assert signals.quick_refinement is True
    assert signals.task_switch is False
    assert signals.suitability_score < 1.0
    assert signals.refinement_window_seconds is not None
    assert signals.refinement_window_seconds <= 3.1


class StubLangsmithClient:
    def __init__(self) -> None:
        self.is_ready = True
        self.runs: list[dict[str, object]] = []
        self.feedback: list[dict[str, object]] = []

    def create_run(self, **kwargs: object) -> dict[str, object]:
        self.runs.append(kwargs)
        return {"id": f"run-{len(self.runs)}"}

    def create_feedback(
        self,
        *,
        run_id: str,
        key: str,
        score: float | None = None,
        comment: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.feedback.append(
            {
                "run_id": run_id,
                "key": key,
                "score": score,
                "comment": comment,
                "metadata": metadata,
            }
        )


def test_feedback_manager_uses_stub_client() -> None:
    analyzer = BehaviorAnalyzer()
    tracker = SessionTracker(analyzer)
    client = StubLangsmithClient()

    manager = ImplicitFeedbackManager(
        enabled=True,
        session_tracker=tracker,
        langsmith_client=client,  # type: ignore[arg-type]
    )

    start = datetime.now(timezone.utc)
    first_end = start + timedelta(seconds=2)
    manager.record_tool_call(
        tool_name="crm.search",
        start_time=start,
        end_time=first_end,
        status="success",
        params={"query": "alpha"},
        result={"count": 1},
        error=None,
        session_id="session-1",
        user_id="user-1",
        metadata={"source": "test"},
        fire_and_forget=False,
    )

    second_start = first_end + timedelta(seconds=3)
    manager.record_tool_call(
        tool_name="crm.search",
        start_time=second_start,
        end_time=second_start + timedelta(seconds=1),
        status="success",
        params={"query": "alpha"},
        result={"count": 0},
        error=None,
        session_id="session-1",
        user_id="user-1",
        metadata={"source": "test"},
        fire_and_forget=False,
    )

    assert len(client.runs) == 2
    assert client.feedback, "Expected implicit feedback events"  # Quick refinement should trigger feedback
    feedback_entry = client.feedback[0]
    assert feedback_entry["key"] == "refinement_needed"
    assert feedback_entry["run_id"] == "run-2"
    assert isinstance(feedback_entry["metadata"], dict)
    assert feedback_entry["metadata"].get("tool_name") == "crm.search"

