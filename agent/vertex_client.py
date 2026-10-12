"""Google Cloud Vertex AI adapter using terminal-based gcloud authentication.

Connects to Vertex AI's official OpenAI-compatible endpoint:
https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions

Uses OAuth2 access tokens obtained via ``gcloud auth print-access-token``
and managed by :class:`agent.gcloud_auth.GCloudAuthManager`.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.gcloud_auth import gcloud_auth
from agent.llm_base import ChatCompletionsClient, post_with_cancel
from agent.settings import Settings

_LOG = logging.getLogger(__name__)

DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"


class VertexClient(ChatCompletionsClient):
    """Google Cloud Vertex AI client using gcloud terminal authentication."""

    requires_api_key = False
    preserve_reasoning = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, provider_label="Google Cloud Vertex AI (gcloud)")

    def _resolved_project(self) -> str:
        if self.settings.llm_provider in ("gemini", "google"):
            configured = self.settings.gemini_project_id or self.settings.vertex_project_id
        else:
            configured = self.settings.vertex_project_id or self.settings.gemini_project_id
        return gcloud_auth.get_project(configured)

    def _resolved_location(self) -> str:
        if self.settings.llm_provider in ("gemini", "google"):
            configured = self.settings.gemini_location or self.settings.vertex_location
        else:
            configured = self.settings.vertex_location or self.settings.gemini_location
        return gcloud_auth.get_location(configured or DEFAULT_LOCATION)

    def _api_key(self) -> str:
        return gcloud_auth.get_access_token()

    def _endpoint(self) -> str:
        project = self._resolved_project()
        location = self._resolved_location()
        return (
            f"https://{location}-aiplatform.googleapis.com/v1beta1/"
            f"projects/{project}/locations/{location}/endpoints/openapi/chat/completions"
        )

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _normalize_model_name(self, model: str) -> str:
        name = model.strip() if model else DEFAULT_MODEL
        if name.startswith("models/"):
            name = name[7:]
        # Vertex AI OpenAPI endpoint expects <publisher>/<model>, e.g. 'google/gemini-2.5-flash'
        if "/" not in name:
            return f"google/{name}"
        return name

    @property
    def model(self) -> str:
        raw_model = self.settings.llm_model or DEFAULT_MODEL
        return self._normalize_model_name(raw_model)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        wire_messages = self.sanitize_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools

        max_tokens = self.settings.llm_max_completion_tokens
        if max_tokens:
            # Vertex AI OpenAPI enforces maxOutputTokens < 65536 (exclusive upper bound).
            # Clamp to 65535 to prevent 400 INVALID_ARGUMENT when 65536 is configured.
            payload["max_tokens"] = min(max_tokens, 65535)

        effort = (
            self.settings.gemini_reasoning_effort
            if self.settings.llm_provider in ("gemini", "google")
            else self.settings.vertex_reasoning_effort
        )
        if effort:
            payload["reasoning_effort"] = effort

        return payload

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        timeout = (
            self.settings.gemini_timeout_seconds
            if self.settings.llm_provider in ("gemini", "google")
            else self.settings.vertex_timeout_seconds
        )
        # Always inject current access token dynamically for each attempt
        current_token = self._api_key()
        active_headers = {**headers, "Authorization": f"Bearer {current_token}"}
        try:
            response = post_with_cancel(
                url=self._endpoint(),
                payload=payload,
                headers=active_headers,
                timeout_seconds=timeout,
                stop_event=self.stop_event,
            )
        except Exception as err:
            if "401" in str(err):
                gcloud_auth.invalidate_token()
            raise

        # Check for expired/invalid token response (401 Unauthorized)
        if getattr(response, "status_code", None) == 401:
            try:
                response.close()
            except Exception:
                pass
            gcloud_auth.invalidate_token()
            if not (self.stop_event and self.stop_event.is_set()):
                try:
                    fresh_token = gcloud_auth.get_access_token(force_refresh=True)
                    retry_headers = {**headers, "Authorization": f"Bearer {fresh_token}"}
                    response = post_with_cancel(
                        url=self._endpoint(),
                        payload=payload,
                        headers=retry_headers,
                        timeout_seconds=timeout,
                        stop_event=self.stop_event,
                    )
                    if getattr(response, "status_code", None) == 401:
                        gcloud_auth.invalidate_token()
                except Exception:
                    gcloud_auth.invalidate_token()
        return response

