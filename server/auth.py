"""Shared GEE authentication module.

Implements a 4-level fallback chain:
  1. gee-key.json service account file
  2. Environment variables (GEE_SERVICE_ACCOUNT + key path)
  3. Default user credentials
  4. Interactive gcloud authentication
"""

import json
import logging
import os

import ee
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def setup_gee(key_path: str | None = None) -> None:
    """Authenticate and initialise the Earth Engine API.

    Tries four methods in order, stopping at the first success:

    1. **Service-account key file** — uses *key_path* if provided, otherwise
       looks for ``<project_root>/.config/geo-stars-2fc6f2e84e4e.json``.
    2. **Environment variables** — uses ``GEE_SERVICE_ACCOUNT`` (or
       ``EE_SERVICE_ACCOUNT``) together with ``GEE_PRIVATE_KEY_PATH`` (or
       ``GOOGLE_APPLICATION_CREDENTIALS``).
    3. **Default user credentials** — calls ``ee.Initialize()`` with no
       explicit credentials (works when already authenticated via gcloud).
    4. **Interactive gcloud auth** — runs ``ee.Authenticate()`` as a last
       resort (controlled by ``GEE_AUTH_MODE``, default ``"gcloud"``).

    All methods use ``GEE_PROJECT`` for the GEE project id. If the key file
    (level 1) is found, the project id is read from it as a fallback.

    :param key_path: optional path to a GEE service-account JSON key file.
        When provided this is tried first, skipping the default key location.
    :raises RuntimeError: if all four methods fail.
    """
    gee_project = os.getenv("GEE_PROJECT")
    logger.debug("GEE auth: GEE_PROJECT=%s", gee_project or "(not set)")

    # --- Level 1: service-account key file ------------------------------
    # Walk upward from the current file's directory looking for .config/
    _search = os.path.dirname(os.path.abspath(__file__))
    key_path = None
    subdirs = [".config", "env"]
    for _ in range(4):
        for subdir in subdirs:
            candidate = os.path.join(
                _search, subdir, "geo-stars-2fc6f2e84e4e.json"
            )
            if os.path.exists(candidate):
                key_path = os.path.abspath(candidate)
                break
        _search = os.path.dirname(_search)
    logger.debug("GEE auth [1/4]: looking for key file %s", key_path)
    if key_path and os.path.exists(key_path):
        try:
            with open(key_path) as f:
                key_data = json.load(f)
            project = gee_project or key_data.get("project_id")
            sa_email = key_data.get("client_email", "?")
            logger.debug(
                "GEE auth [1/4]: found key file, service account=%s, project=%s",
                sa_email, project,
            )
            if not project:
                raise ValueError(
                    "GEE project ID not found in key file or GEE_PROJECT env var."
                )
            credentials = ee.ServiceAccountCredentials(sa_email, key_path)
            ee.Initialize(credentials, project=project)
            logger.debug("GEE auth [1/4]: success (service-account key file)")
            return
        except Exception as e:
            logger.warning("GEE auth [1/4]: failed: %s", e)
    else:
        logger.debug("GEE auth [1/4]: key file not found, skipping")

    # --- Level 2: environment variables ---------------------------------
    service_account = os.getenv("GEE_SERVICE_ACCOUNT") or os.getenv(
        "GEE_SERVICE_ACCOUNT"
    )
    env_key_path = os.getenv("GEE_PRIVATE_KEY_PATH") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    logger.debug(
        "GEE auth [2/4]: env vars: service_account=%s, key_path=%s",
        service_account or "(not set)", env_key_path or "(not set)",
    )
    if not gee_project:
        raise RuntimeError(
            "GEE_PROJECT is not set and key file failed or was not found."
        )
    if service_account and env_key_path and os.path.exists(env_key_path):
        try:
            logger.debug(
                "GEE auth [2/4]: trying service account %s with key %s",
                service_account, env_key_path,
            )
            credentials = ee.ServiceAccountCredentials(
                service_account, env_key_path
            )
            ee.Initialize(credentials, project=gee_project)
            logger.debug("GEE auth [2/4]: success (env-var service account)")
            return
        except Exception as e:
            logger.warning("GEE auth [2/4]: failed: %s", e)
    else:
        logger.debug("GEE auth [2/4]: incomplete env vars, skipping")

    # --- Level 3: default user credentials ------------------------------
    logger.debug("GEE auth [3/4]: trying default credentials, project=%s", gee_project)
    try:
        ee.Initialize(project=gee_project)
        logger.debug("GEE auth [3/4]: success (default user credentials)")
        return
    except Exception as e:
        logger.warning("GEE auth [3/4]: failed: %s", e)

    # --- Level 4: interactive gcloud auth -------------------------------
    auth_mode = os.getenv("GEE_AUTH_MODE", "gcloud")
    logger.debug("GEE auth [4/4]: trying interactive auth (mode=%s)", auth_mode)
    try:
        ee.Authenticate(auth_mode=auth_mode)
        ee.Initialize(project=gee_project)
        logger.debug("GEE auth [4/4]: success (interactive %s)", auth_mode)
    except Exception as e:
        raise RuntimeError(
            "All GEE authentication methods failed. "
            f"Last error: {e}"
        ) from e
