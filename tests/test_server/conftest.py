"""Pytest fixtures for the gee_mcp server test suite."""

import os
import sys

# Ensure GEE auth is skipped before any gee_mcp import in subordinate
# fixtures.
os.environ.setdefault("GEE_SKIP_AUTH", "1")

import pytest  # noqa: E402
from fastmcp.client import Client  # noqa: E402

from gee_mcp.config import SERVER_MODULE as PKG  # noqa: E402


@pytest.fixture
async def main_mcp_client():
    """A FastMCP Client context manager for testing the server."""
    from gee_mcp.server import mcp  # noqa: E402

    async with Client(transport=mcp) as mcp_client:
        yield mcp_client


@pytest.fixture()
def server_pkg():
    """Return the imported gee_mcp.server package."""
    import importlib

    return importlib.import_module(PKG)
