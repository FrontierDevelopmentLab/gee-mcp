"""Google Earth Engine initialisation.

Auth resolution is delegated to ``ee.Initialize``, which walks the
standard Google Cloud credential chain:

  - ``GOOGLE_APPLICATION_CREDENTIALS`` (service-account key JSON)
  - gcloud Application Default Credentials cache
  - The persistent EE credentials from ``earthengine authenticate``
  - GCE / Cloud Run instance metadata
"""

import logging
import os

import ee
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def setup_gee() -> None:
    """Initialise the Earth Engine API for ``GEE_PROJECT``.

    Honours ``GEE_SKIP_AUTH=1`` as a no-op escape hatch (used by the
    test suite). Otherwise requires ``GEE_PROJECT`` and lets
    ``ee.Initialize`` resolve credentials from the standard GCP chain.
    """
    if os.getenv("GEE_SKIP_AUTH") == "1":
        logger.debug("GEE auth: skipped (GEE_SKIP_AUTH=1)")
        return

    project = os.getenv("GEE_PROJECT")
    if not project:
        raise RuntimeError("GEE_PROJECT environment variable is required.")
    ee.Initialize(project=project)
    logger.debug("Earth Engine initialised for project %s", project)
