"""Shared GEE authentication module.

Implements a 4-level fallback chain:
  1. Explicit service-account key file (param or ``GEE_KEY_PATH``)
  2. Environment variables (``GEE_SERVICE_ACCOUNT`` /
     ``EE_SERVICE_ACCOUNT`` together with ``GEE_PRIVATE_KEY_PATH`` /
     ``GOOGLE_APPLICATION_CREDENTIALS``)
  3. Default user credentials (``ee.Initialize(project=...)``)
  4. Interactive ``ee.Authenticate`` as a last resort
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

    1. **Service-account key file** — uses *key_path* if provided,
       otherwise falls back to the ``GEE_KEY_PATH`` environment
       variable.
    2. **Environment variables** — uses ``GEE_SERVICE_ACCOUNT`` (or
       ``EE_SERVICE_ACCOUNT``) together with ``GEE_PRIVATE_KEY_PATH``
       (or ``GOOGLE_APPLICATION_CREDENTIALS``).
    3. **Default user credentials** — calls ``ee.Initialize`` with no
       explicit credentials (works when already authenticated via
       gcloud).
    4. **Interactive auth** — runs ``ee.Authenticate`` as a last
       resort (controlled by ``GEE_AUTH_MODE``, default ``"gcloud"``).

    The GEE project id is taken from ``GEE_PROJECT``. For level 1 it
    falls back to ``project_id`` from the key file when that env var
    is unset.

    :param key_path: optional path to a GEE service-account JSON key
        file. When provided this is tried first.
    :raises RuntimeError: if all four methods fail.
    """
    if os.getenv("GEE_SKIP_AUTH") == "1":
        logger.debug("GEE auth: skipped (GEE_SKIP_AUTH=1)")
        return

    gee_project = os.getenv("GEE_PROJECT")
    logger.debug("GEE auth: GEE_PROJECT=%s", gee_project or "(not set)")

    # --- Level 1: service-account key file ------------------------------
    key_path = key_path or os.getenv("GEE_KEY_PATH")
    logger.debug("GEE auth [1/4]: key_path=%s", key_path or "(not set)")
    if key_path and os.path.exists(key_path):
        try:
            with open(key_path, encoding="utf-8") as f:
                key_data = json.load(f)
            project = gee_project or key_data.get("project_id")
            sa_email = key_data.get("client_email", "?")
            logger.debug(
                "GEE auth [1/4]: service account=%s, project=%s",
                sa_email,
                project,
            )
            if not project:
                raise ValueError(
                    "GEE project ID not found in key file or "
                    "GEE_PROJECT env var."
                )
            credentials = ee.ServiceAccountCredentials(sa_email, key_path)
            ee.Initialize(credentials, project=project)
            logger.debug("GEE auth [1/4]: success (key file)")
            return
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("GEE auth [1/4]: failed: %s", e)
    else:
        logger.debug("GEE auth [1/4]: key file not found, skipping")

    # --- Level 2: environment variables ---------------------------------
    service_account = os.getenv("GEE_SERVICE_ACCOUNT") or os.getenv(
        "EE_SERVICE_ACCOUNT"
    )
    env_key_path = os.getenv("GEE_PRIVATE_KEY_PATH") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    logger.debug(
        "GEE auth [2/4]: service_account=%s, key_path=%s",
        service_account or "(not set)",
        env_key_path or "(not set)",
    )
    if (
        service_account
        and env_key_path
        and os.path.exists(env_key_path)
        and gee_project
    ):
        try:
            credentials = ee.ServiceAccountCredentials(
                service_account, env_key_path
            )
            ee.Initialize(credentials, project=gee_project)
            logger.debug("GEE auth [2/4]: success (env-var service account)")
            return
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("GEE auth [2/4]: failed: %s", e)
    else:
        logger.debug("GEE auth [2/4]: incomplete env vars, skipping")

    # --- Level 3: default user credentials ------------------------------
    if gee_project:
        logger.debug(
            "GEE auth [3/4]: trying default credentials, project=%s",
            gee_project,
        )
        try:
            ee.Initialize(project=gee_project)
            logger.debug("GEE auth [3/4]: success (default user credentials)")
            return
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("GEE auth [3/4]: failed: %s", e)
    else:
        logger.debug("GEE auth [3/4]: GEE_PROJECT not set, skipping")

    # --- Level 4: interactive auth --------------------------------------
    auth_mode = os.getenv("GEE_AUTH_MODE", "gcloud")
    logger.debug(
        "GEE auth [4/4]: trying interactive auth (mode=%s)", auth_mode
    )
    try:
        ee.Authenticate(auth_mode=auth_mode)
        ee.Initialize(project=gee_project)
        logger.debug("GEE auth [4/4]: success (interactive %s)", auth_mode)
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(
            "All GEE authentication methods failed. " f"Last error: {e}"
        ) from e
