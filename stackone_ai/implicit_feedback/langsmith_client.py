from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - exercised via integration tests
    from langsmith import Client
except Exception:  # pragma: no cover - optional dependency
    Client = None  # type: ignore[assignment]


logger = logging.getLogger("stackone.implicit_feedback")


class LangsmithFeedbackClient:
    """Thin wrapper around LangSmith's SDK so the dependency stays optional."""

    def __init__(
        self,
        api_key: str | None,
        project_name: str | None = None,
        default_tags: Sequence[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._project_name = project_name
        self._default_tags = list(default_tags or [])
        self._client = self._build_client()

    def _build_client(self) -> Client | None:
        if not self._api_key:
            return None
        if Client is None:
            logger.warning("LangSmith client unavailable; implicit feedback disabled.")
            return None
        try:
            return Client(api_key=self._api_key)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to initialise LangSmith client: %s", exc, exc_info=True)
            return None

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    def create_run(self, **kwargs: Any) -> Any:
        client = self._client
        if client is None:
            return None
        if self._project_name and "project_name" not in kwargs:
            kwargs["project_name"] = self._project_name
        if self._default_tags:
            tags = list(kwargs.get("tags", []))
            kwargs["tags"] = sorted(set(tags + self._default_tags))
        try:
            return client.create_run(**kwargs)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.warning("Failed to create LangSmith run: %s", exc, exc_info=True)
            return None

    def create_feedback(
        self,
        run_id: str,
        key: str,
        *,
        score: float | None = None,
        comment: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        client = self._client
        if client is None:
            return
        payload: dict[str, Any] = {
            "run_id": run_id,
            "key": key,
        }
        if score is not None:
            payload["score"] = score
        if comment is not None:
            payload["comment"] = comment
        if metadata:
            payload["metadata"] = dict(metadata)
        try:
            client.create_feedback(**payload)
        except Exception as exc:  # pragma: no cover - log and continue
            logger.warning("Failed to submit LangSmith feedback: %s", exc, exc_info=True)
