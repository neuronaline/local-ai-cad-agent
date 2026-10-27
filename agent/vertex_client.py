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
DEFAULT_MODEL = "gemini-3.8-flash"


class VertexClient(ChatCompletionsClient):
    """Google Cloud Vertex AI client using gcloud terminal authentication."""

    requires_api_key = False
    # Vertex AI's OpenAPI ChatCompletions schema rejects top-level
    # ``reasoning*`` fields on assistant turns; reasoning state travels via
    # ``tool_calls[].extra_content.google.thought_signature`` instead, which
    # ``sanitize_assistant_message`` always populates.
    preserve_reasoning = False

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
        model_lower = self.model.lower()
        if "gemini-3" in model_lower or "gemini-4" in model_lower:
            # Vertex AI provisions Gemini 3+ preview/frontier models exclusively on the 'global' endpoint.
            if configured and configured.lower() != "us-central1":
                return gcloud_auth.get_location(configured)
            return "global"
        return gcloud_auth.get_location(configured or DEFAULT_LOCATION)

    def _api_key(self) -> str:
        return gcloud_auth.get_access_token()

    def _endpoint(self, location_override: str | None = None) -> str:
        project = self._resolved_project()
        location = (location_override or self._resolved_location()).strip()
        host = (
            "aiplatform.googleapis.com"
            if location.lower() == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/v1beta1/"
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
        # Vertex AI OpenAPI endpoint expects <publisher>/<model>, e.g. 'google/gemini-3.8-flash'
        if "/" not in name:
            return f"google/{name}"
        return name

    @property
    def model(self) -> str:
        # Not ``settings.llm_model``: as the fallback client, ``llm_provider``
        # names the primary provider, not Vertex.
        if self.settings.llm_provider in ("gemini", "google"):
            raw_model = self.settings.gemini_model
        else:
            raw_model = self.settings.vertex_model
        return self._normalize_model_name(raw_model or DEFAULT_MODEL)

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
            # Vertex AI OpenAPI enforces maxOutputTokens < 65536. Clamp to 65535 to
            # prevent 400 INVALID_ARGUMENT when 65536 is configured.
            payload["max_tokens"] = min(max_tokens, 65535)

        effort = (
            self.settings.gemini_reasoning_effort
            if self.settings.llm_provider in ("gemini", "google")
            else self.settings.vertex_reasoning_effort
        )
        model_lower = self.model.lower()
        is_gemini_2 = "2.5" in model_lower or "gemini-2" in model_lower
        eff = (effort or "").lower().strip()
        thinking_config: dict[str, Any] = {"include_thoughts": eff != "none"}
        if eff and eff != "none":
            if not is_gemini_2:
                level_map = {
                    "minimal": "MINIMAL" if "flash" in model_lower else "LOW",
                    "low": "LOW",
                    "medium": "MEDIUM",
                    "high": "HIGH",
                    "xhigh": "HIGH",
                    "max": "HIGH",
                }
                level = level_map.get(eff, "HIGH")
                if "gemini-4" in model_lower and level == "MINIMAL":
                    level = "LOW"
                thinking_config["thinking_level"] = level
            else:
                budget_map = {
                    "minimal": 1024,
                    "low": 2048,
                    "medium": 4096,
                    "high": 8192,
                    "xhigh": 16384,
                    "max": 24576,
                }
                thinking_config["thinking_budget"] = budget_map.get(eff, 8192)
        elif eff == "none":
            if not is_gemini_2:
                thinking_config["thinking_level"] = (
                    "MINIMAL"
                    if ("flash" in model_lower and "gemini-4" not in model_lower)
                    else "LOW"
                )
            elif "pro" not in model_lower:
                thinking_config["thinking_budget"] = 0
        payload["extra_body"] = {"google": {"thinking_config": thinking_config}}

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
        endpoint_url = self._endpoint()
        try:
            response = post_with_cancel(
                url=endpoint_url,
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
                    active_headers = {**headers, "Authorization": f"Bearer {fresh_token}"}
                    response = post_with_cancel(
                        url=endpoint_url,
                        payload=payload,
                        headers=active_headers,
                        timeout_seconds=timeout,
                        stop_event=self.stop_event,
                    )
                    if getattr(response, "status_code", None) == 401:
                        gcloud_auth.invalidate_token()
                except Exception:
                    gcloud_auth.invalidate_token()

        # Many Gemini 3+ preview/frontier models on Vertex AI are provisioned exclusively
        # on the 'global' endpoint. If a regional endpoint returns 404 NOT_FOUND,
        # transparently retry against locations/global.
        if (
            getattr(response, "status_code", None) == 404
            and self._resolved_location().lower() != "global"
            and not (self.stop_event and self.stop_event.is_set())
        ):
            try:
                response.close()
            except Exception:
                pass
            response = post_with_cancel(
                url=self._endpoint(location_override="global"),
                payload=payload,
                headers=active_headers,
                timeout_seconds=timeout,
                stop_event=self.stop_event,
            )
        return response

