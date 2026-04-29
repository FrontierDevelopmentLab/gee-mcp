"""Top-level pytest fixtures for gee_mcp.

Sets ``GEE_SKIP_AUTH=1`` before any gee_mcp module is imported, so
``setup_gee`` is a no-op during tests (otherwise the import-time auth
chain would attempt interactive login on CI).
"""

import os

os.environ.setdefault("GEE_SKIP_AUTH", "1")
