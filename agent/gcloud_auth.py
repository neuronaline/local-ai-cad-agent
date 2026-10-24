"""Google Cloud CLI (`gcloud`) authentication and configuration manager.

Provides token caching, project resolution, and environment discovery
for Google Cloud Vertex AI without external vendor SDK dependencies.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time

_LOG = logging.getLogger(__name__)

# Token validity: GCP access tokens are typically valid for 3600 seconds (1 hour).
# Cache for 50 minutes (3000 seconds) to ensure safety against edge-of-expiry requests.
_TOKEN_CACHE_TTL_SECONDS = 3000.0


class GCloudAuthManager:
    """Thread-safe access token cache and configuration discovery for gcloud CLI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_token: str | None = None
        self._token_expiry_monotonic: float = 0.0
        self._cached_project: str | None = None
        self._cached_location: str | None = None

    def invalidate_token(self) -> None:
        """Clear cached access token, forcing subsequent requests to fetch a fresh token."""
        with self._lock:
            self._cached_token = None
            self._token_expiry_monotonic = 0.0

    def get_access_token(self, *, force_refresh: bool = False) -> str:
        """Retrieve a valid Google Cloud OAuth2 access token.

        Uses cached token if still within the TTL window, or runs
        `gcloud auth print-access-token` to obtain a fresh one.
        """
        with self._lock:
            now = time.monotonic()
            if (
                not force_refresh
                and self._cached_token
                and now < self._token_expiry_monotonic
            ):
                return self._cached_token

            gcloud_bin = shutil.which("gcloud")
            if not gcloud_bin:
                raise RuntimeError(
                    "Google Cloud CLI ('gcloud') was not found in PATH. "
                    "Please install Google Cloud SDK or configure an API key in .env."
                )

            try:
                result = subprocess.run(
                    [gcloud_bin, "auth", "print-access-token"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "Timed out while running 'gcloud auth print-access-token'. "
                    "Check your terminal network connection or gcloud status."
                ) from exc
            except OSError as exc:
                raise RuntimeError(f"Failed to execute gcloud: {exc}") from exc

            if result.returncode != 0:
                stderr = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(
                    f"Google Cloud authentication failed: {stderr}. "
                    "Please run 'gcloud auth login' in your terminal."
                )

            raw_output = result.stdout.strip()
            if not raw_output:
                raise RuntimeError(
                    "gcloud auth print-access-token returned an empty token. "
                    "Please run 'gcloud auth login' in your terminal."
                )

            # Extract the last non-empty line to filter out any banner/warning messages
            candidate_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
            token = candidate_lines[-1] if candidate_lines else ""
            if not token or any(ch.isspace() for ch in token):
                raise RuntimeError(
                    f"Unexpected token format from gcloud CLI: {raw_output[:80]!r}. "
                    "Please run 'gcloud auth login' in your terminal."
                )

            self._cached_token = token
            self._token_expiry_monotonic = time.monotonic() + _TOKEN_CACHE_TTL_SECONDS
            return token

    def get_project(self, configured_project: str | None = None) -> str:
        """Resolve the Google Cloud project ID.

        Priority order:
        1. Explicit configuration from config.yaml.
        2. Cached project ID from previous resolution.
        3. Environment variables (CLOUDSDK_CORE_PROJECT, GOOGLE_CLOUD_PROJECT, GCP_PROJECT).
        4. Local gcloud config ('gcloud config get-value project').
        """
        if configured_project and configured_project.strip():
            return configured_project.strip()

        with self._lock:
            if self._cached_project:
                return self._cached_project

        for env_var in (
            "CLOUDSDK_CORE_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
            "GCP_PROJECT",
        ):
            val = os.getenv(env_var, "").strip()
            if val:
                with self._lock:
                    self._cached_project = val
                return val

        gcloud_bin = shutil.which("gcloud")
        if gcloud_bin:
            try:
                res = subprocess.run(
                    [gcloud_bin, "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if res.returncode == 0:
                    candidate_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                    val = candidate_lines[-1] if candidate_lines else ""
                    if val and val != "(unset)":
                        with self._lock:
                            self._cached_project = val
                        return val
            except Exception as err:
                _LOG.debug("Failed to read project from gcloud config: %s", err)

        raise RuntimeError(
            "Google Cloud project ID not configured. Please specify 'vertex.project_id' "
            "in config.yaml or set your default project via 'gcloud config set project <PROJECT_ID>'."
        )

    # Alias for get_project
    get_project_id = get_project

    def get_location(self, configured_location: str | None = None) -> str:
        """Resolve the Google Cloud location / region (defaults to 'us-central1')."""
        if configured_location and configured_location.strip():
            return configured_location.strip()

        with self._lock:
            if self._cached_location:
                return self._cached_location

        for env_var in (
            "GOOGLE_CLOUD_LOCATION",
            "CLOUD_ML_REGION",
            "CLOUDSDK_AI_REGION",
            "CLOUDSDK_COMPUTE_REGION",
        ):
            env_val = os.getenv(env_var, "").strip()
            if env_val:
                with self._lock:
                    self._cached_location = env_val
                return env_val

        gcloud_bin = shutil.which("gcloud")
        if gcloud_bin:
            try:
                res = subprocess.run(
                    [gcloud_bin, "config", "get-value", "compute/region"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if res.returncode == 0:
                    candidate_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                    val = candidate_lines[-1] if candidate_lines else ""
                    if val and val != "(unset)":
                        with self._lock:
                            self._cached_location = val
                        return val
            except Exception as err:
                _LOG.debug("Failed to read region from gcloud config: %s", err)

        default_loc = "us-central1"
        with self._lock:
            self._cached_location = default_loc
        return default_loc

    def check_gcloud_status(self) -> tuple[bool, str]:
        """Check whether gcloud is installed and authenticated."""
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            return False, "gcloud CLI not installed"
        try:
            self.get_access_token()
            return True, "authenticated"
        except Exception as err:
            return False, str(err)


# Global singleton instance for token caching across LLM requests
gcloud_auth = GCloudAuthManager()


def check_gcloud_status() -> tuple[bool, str]:
    """Convenience helper to check gcloud CLI installation and authentication status."""
    return gcloud_auth.check_gcloud_status()

