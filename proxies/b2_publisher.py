"""Backblaze B2 publisher for research reports.

Uploads HTML reports to a public B2 bucket and returns the public URL.
The bucket is created idempotently on first use.

Configuration via environment variables:
  B2_APPLICATION_KEY_ID  — Backblaze application key ID
  B2_APPLICATION_KEY     — Backblaze application key
  B2_BUCKET_NAME         — Bucket name (default: "deep-search-reports")
  B2_REPORT_PREFIX       — Key prefix for reports (default: "reports/")
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

log = logging.getLogger("b2_publisher")

B2_KEY_ID = os.getenv("B2_APPLICATION_KEY_ID", "")
B2_APP_KEY = os.getenv("B2_APPLICATION_KEY", "")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "deep-search-reports")
B2_REPORT_PREFIX = os.getenv("B2_REPORT_PREFIX", "reports/")

# Cached B2 API + bucket reference (initialised lazily, thread-safe)
_b2_lock = threading.Lock()
_b2_api = None
_b2_bucket = None


def _get_b2_bucket():
    """Lazily initialise the B2 API and get-or-create the public bucket.

    Thread-safe via a lock.  The bucket is created with ``allPublic`` type
    so that uploaded files are directly accessible via the friendly URL.
    """
    global _b2_api, _b2_bucket

    if _b2_bucket is not None:
        return _b2_bucket

    with _b2_lock:
        # Double-check after acquiring lock
        if _b2_bucket is not None:
            return _b2_bucket

        if not B2_KEY_ID or not B2_APP_KEY:
            raise RuntimeError(
                "B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must be set "
                "to publish reports to Backblaze B2"
            )

        from b2sdk.v2 import B2Api, InMemoryAccountInfo

        info = InMemoryAccountInfo()
        api = B2Api(info)
        api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)

        # Idempotent bucket creation: try to get existing, create if not found
        try:
            bucket = api.get_bucket_by_name(B2_BUCKET_NAME)
            log.info(f"Using existing B2 bucket: {B2_BUCKET_NAME}")
        except Exception:
            # Bucket doesn't exist — create it as allPublic
            bucket = api.create_bucket(
                B2_BUCKET_NAME,
                bucket_type="allPublic",
                lifecycle_rules=[],
                cors_rules=[{
                    "corsRuleName": "allowAll",
                    "allowedOrigins": ["*"],
                    "allowedHeaders": ["*"],
                    "allowedOperations": [
                        "b2_download_file_by_name",
                        "b2_download_file_by_id",
                    ],
                    "maxAgeSeconds": 86400,
                }],
            )
            log.info(f"Created new public B2 bucket: {B2_BUCKET_NAME}")

        _b2_api = api
        _b2_bucket = bucket
        return bucket


def publish_report(
    session_id: str,
    html_content: str,
    content_type: str = "text/html",
) -> str:
    """Upload an HTML report to B2 and return its public URL.

    Args:
        session_id: Research session identifier (used in the object key).
        html_content: The full HTML string to upload.
        content_type: MIME type for the uploaded file.

    Returns:
        The public URL where the report can be accessed.

    Raises:
        RuntimeError: If B2 credentials are not configured.
        Exception: On upload failure.
    """
    bucket = _get_b2_bucket()

    file_name = f"{B2_REPORT_PREFIX}{session_id}.html"
    content_bytes = html_content.encode("utf-8")

    from b2sdk.v2 import UploadSourceBytes

    source = UploadSourceBytes(content_bytes)
    file_version = bucket.upload(
        source,
        file_name=file_name,
        content_type=content_type,
    )

    # Build the friendly public URL
    # Format: https://f{cluster}.backblazeb2.com/file/{bucket_name}/{file_name}
    download_url = _b2_api.get_download_url_for_fileid(file_version.id_)
    log.info(f"Published report to B2: {download_url}")
    return download_url


def publish_metrics(
    session_id: str,
    metrics_json: str,
) -> str:
    """Upload metrics JSON to B2 and return its public URL.

    Args:
        session_id: Research session identifier.
        metrics_json: The JSON string to upload.

    Returns:
        The public URL where the metrics can be accessed.
    """
    bucket = _get_b2_bucket()

    file_name = f"{B2_REPORT_PREFIX}{session_id}_metrics.json"
    content_bytes = metrics_json.encode("utf-8")

    from b2sdk.v2 import UploadSourceBytes

    source = UploadSourceBytes(content_bytes)
    file_version = bucket.upload(
        source,
        file_name=file_name,
        content_type="application/json",
    )

    download_url = _b2_api.get_download_url_for_fileid(file_version.id_)
    log.info(f"Published metrics to B2: {download_url}")
    return download_url


def is_configured() -> bool:
    """Return True if B2 credentials are set."""
    return bool(B2_KEY_ID and B2_APP_KEY)
