"""Tests the FastMCP application instance."""

import importlib

from fastmcp import FastMCP

from gee_mcp.config import SERVER_MODULE as PKG


class TestApp:
    """Test the app."""

    @staticmethod
    def test_mcp_instance_name():
        """The mcp instance has the expected name."""
        app = importlib.import_module(f"{PKG}.app")
        assert app.mcp.name == "gee-mcp"

    @staticmethod
    def test_mcp_is_fastmcp_instance():
        """The mcp object is a FastMCP instance."""
        app = importlib.import_module(f"{PKG}.app")
        assert isinstance(app.mcp, FastMCP)
