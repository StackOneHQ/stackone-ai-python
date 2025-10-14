from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict
from datetime import datetime
from threading import Lock, Thread
from typing import Any, Callable, ClassVar, Optional, Sequence
from uuid import uuid4

from .analyzer import BehaviorAnalyzer
from .data import ImplicitFeedbackEvent, ToolCallRecord
from .langsmith_client import LangsmithFeedbackClient
from .session import SessionTracker
from .utils import sanitize_payload

logger = logging.getLogger("stackone.implicit_feedback")

SessionResolver = Callable[[Mapping[str, Any]], Optional[str]]
UserResolver = Callable[[], Optional[str]]


class ImplicitFeedbackManager:
    """Singleton coordinator for implicit feedback processing."""

    _instance: ClassVar[ImplicitFeedbackManager | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        *,
        enabled: bool,
        session_tracker: SessionTracker,
        langsmith_client: LangsmithFeedbackClient,
        session_resolver: SessionResolver | None = None,
        user_resolver: UserResolver | None = None,
    ) -> None:
        self._enabled = enabled
        self._session_tracker = session_tracker
        self._langsmith_client = langsmith_client
        self._session_resolver = session_resolver
        self._user_resolver = user_resolver

    @classmethod
    def configure(
        cls,
        *,
        enabled: bool | None = None,
        api_key: str | None = None,
        project_name: str | None = None,
        default_tags: Sequence[str] | None = None,
        session_resolver: SessionResolver | None = None,
        user_resolver: UserResolver | None = None,
        behavior_analyzer: BehaviorAnalyzer | None = None,
        session_tracker: SessionTracker | None = None,
        langsmith_client: LangsmithFeedbackClient | None = None,
    ) -> None:
        api_key = api_key if api_key is not None else os.getenv("LANGSMITH_API_KEY")
        if langsmith_client is None:
            langsmith_client = LangsmithFeedbackClient(
                api_key=api_key,
                project_name=project_name or os.getenv("STACKONE_IMPLICIT_FEEDBACK_PROJECT"),
                default_tags=default_tags or cls._parse_tags(os.getenv("STACKONE_IMPLICIT_FEEDBACK_TAGS")),
            )
        if behavior_analyzer is None:
            behavior_analyzer = BehaviorAnalyzer()
        if session_tracker is None:
            session_tracker = SessionTracker(behavior_analyzer)
        if enabled is None:
            enabled_env = os.getenv("STACKONE_IMPLICIT_FEEDBACK_ENABLED")
            enabled = cls._parse_enabled_flag(enabled_env, default=bool(api_key))

        manager = cls(
            enabled=enabled,
            session_tracker=session_tracker,
            langsmith_client=langsmith_client,
            session_resolver=session_resolver,
            user_resolver=user_resolver,
        )
        with cls._lock:
            cls._instance = manager

    @classmethod
    def get(cls) -> ImplicitFeedbackManager:
        with cls._lock:
            if cls._instance is None:
                cls.configure()
            assert cls._instance is not None
            return cls._instance

    @staticmethod
    def _parse_enabled_flag(flag: str | None, *, default: bool) -> bool:
        if flag is None:
            return default
        return flag.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _parse_tags(raw: str | None) -> Sequence[str]:
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def record_tool_call(
        self,
        *,
        tool_name: str,
        start_time: datetime,
        end_time: datetime,
        status: str,
        params: Mapping[str, Any] | None,
        result: Mapping[str, Any] | None,
        error: str | None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        if not self._enabled:
            return

        sanitized_params = sanitize_payload(params) if params is not None else None
        sanitized_result = sanitize_payload(result) if result is not None else None
        sanitized_error = sanitize_payload(error) if error else None
        safe_metadata = sanitize_payload(metadata) if metadata is not None else None

        context: dict[str, Any] = {
            "params": sanitized_params,
            "result": sanitized_result,
            "metadata": safe_metadata,
        }

        resolved_session = session_id or (self._session_resolver(context) if self._session_resolver else None)
        resolved_user = user_id or (self._user_resolver() if self._user_resolver else None)

        duration_ms = int(max(0.0, (end_time - start_time).total_seconds()) * 1000)

        record = ToolCallRecord(
            call_id=uuid4().hex,
            tool_name=tool_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            session_id=resolved_session,
            user_id=resolved_user,
            status=status,
            params=sanitized_params,
            result=sanitized_result,
            error=sanitized_error if isinstance(sanitized_error, str) else str(sanitized_error) if sanitized_error else None,
            metadata=safe_metadata if isinstance(safe_metadata, Mapping) else {},
        )

        if fire_and_forget:
            Thread(target=self._process_record, args=(record,), daemon=True).start()
        else:
            self._process_record(record)

    def _process_record(self, record: ToolCallRecord) -> None:
        try:
            enriched_record, events = self._session_tracker.record_tool_call(record)
            self._publish_to_langsmith(enriched_record, events)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to record implicit feedback: %s", exc, exc_info=True)

    def _publish_to_langsmith(
        self, record: ToolCallRecord, events: Sequence[ImplicitFeedbackEvent]
    ) -> None:
        if not self._langsmith_client.is_ready:
            return

        quality_dict = asdict(record.quality) if record.quality else None
        outputs: Any
        if record.result is None:
            outputs = {}
        else:
            outputs = record.result
        if record.error:
            if isinstance(outputs, MutableMapping):
                outputs = dict(outputs)
                outputs["error"] = record.error
            else:
                outputs = {"value": outputs, "error": record.error}

        run_payload: MutableMapping[str, Any] = {
            "name": record.tool_name,
            "run_type": "tool",
            "inputs": record.params or {},
            "outputs": outputs,
            "start_time": record.start_time.isoformat(),
            "end_time": record.end_time.isoformat(),
            "status": record.status,
            "metadata": {
                "duration_ms": record.duration_ms,
                "session_id": record.session_id,
                "user_id": record.user_id,
                "quality": quality_dict,
            },
        }
        if record.error:
            run_payload.setdefault("metadata", {})["error"] = record.error
        if record.metadata:
            run_payload.setdefault("metadata", {})["context"] = dict(record.metadata)

        run = self._langsmith_client.create_run(**run_payload)
        run_id = getattr(run, "id", None)
        if isinstance(run, Mapping):
            run_id = run.get("id") or run.get("run_id") or run_id

        if not run_id:
            return

        for event in events:
            self._langsmith_client.create_feedback(
                run_id=run_id,
                key=event.name,
                score=event.score,
                metadata=event.payload,
            )


def configure_implicit_feedback(**kwargs: Any) -> None:
    ImplicitFeedbackManager.configure(**kwargs)


def get_implicit_feedback_manager() -> ImplicitFeedbackManager:
    return ImplicitFeedbackManager.get()
