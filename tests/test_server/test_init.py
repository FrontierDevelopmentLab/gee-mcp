"""Tests the package ``__init__`` module."""

import importlib

import pytest
from fastmcp import FastMCP
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from gee_mcp.config import SERVER_MODULE as PKG

EXPECTED_TOOLS = {
    "download_satellite_image",
    "compute_index",
    "zonal_statistics",
    "temporal_composite",
    "mask_by_raster",
    "threshold_area",
    "multi_period_analysis",
    "list_datasets",
    "get_dataset_info",
    "get_dataset_metadata",
    "check_imagery_availability",
    "extract_metadata",
    "analyze_metadata",
    "generate_python_from_question",
    "generate_python_from_reasoning_steps",
    "generate_python_from_abstract_graph",
    "generate_abstract_graph_from_question",
    "execute_gee_python",
    "extract_factuality_issues",
    "assess_factuality_issue",
    "get_datasets_locations_and_periods",
    "identify_sensible_variables",
    "sensitivity_analysis",
}


class TestInit:
    """Verify package wiring and tool registration."""

    @pytest.mark.asyncio
    @staticmethod
    async def test_all_tools_registered(
        main_mcp_client: Client[FastMCPTransport],
    ):
        """All expected tools are registered with the FastMCP app."""
        importlib.import_module(PKG)
        tools = await main_mcp_client.list_tools()
        registered = {tool.name for tool in tools}
        missing = EXPECTED_TOOLS - registered
        assert not missing, f"Missing tools: {missing}"

    @staticmethod
    def test_mcp_exported():
        """``mcp`` is re-exported from the package."""
        pkg = importlib.import_module(PKG)
        assert isinstance(pkg.mcp, FastMCP)
