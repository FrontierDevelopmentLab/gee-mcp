"""MCP client for the GEE server (server.py via stdio)."""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import os
import sys

PYTHON_BIN = sys.executable
SERVER_MODULE = "gee_mcp.server"

class GEEToolError(RuntimeError):
    """Raised when the MCP server returns an error for a tool call."""

class GEEMCPClient:
    """Async context manager wrapping an MCP stdio session.

    Usage::

        async with GEEMCPClient() as gee:
            meta = await gee.get_metadata("MODIS_061_MOD10A1")
    """

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._cm_stdio: object = None
        self._cm_session: object = None

    async def __aenter__(self) -> GEEMCPClient:
        # Prepend the local ``src/`` directory to PYTHONPATH so the
        # example client can launch the server straight from a
        # checkout (without installing the package first).
        repo_src = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "src"
        )
        existing = os.environ.get("PYTHONPATH", "")
        pythonpath = (
            f"{repo_src}{os.pathsep}{existing}" if existing else repo_src
        )
        params = StdioServerParameters(
            command=PYTHON_BIN,
            args=["-m", SERVER_MODULE],
            env={**os.environ, "PYTHONPATH": pythonpath},
        )
        self._cm_stdio = stdio_client(params)
        read, write = await self._cm_stdio.__aenter__()  # type: ignore[union-attr]
        self._cm_session = ClientSession(read, write)
        self._session = await self._cm_session.__aenter__()  # type: ignore[union-attr]
        await self._session.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._cm_session is not None:
            await self._cm_session.__aexit__(  # type: ignore[union-attr]
                exc_type, exc_val, exc_tb
            )
        if self._cm_stdio is not None:
            await self._cm_stdio.__aexit__(  # type: ignore[union-attr]
                exc_type, exc_val, exc_tb
            )
        self._session = None


    @staticmethod
    def _parse_result(result: Any) -> Any:
        """Extract JSON from a tool result, raising on server errors."""
        text = result.content[0].text
        if result.isError:
            raise GEEToolError(text)
        
        try:
            return json.loads(text)
        except:
            return text


    async def test(self) -> dict:
        return {"test": "uno"}


    async def get_metadata(self, catalog_id: str) -> dict:
        """Call ``get_dataset_metadata`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "get_dataset_metadata",
            {"catalog_id": catalog_id},
        )
        return json.loads(result.content[0].text)  # type: ignore[union-attr]

    async def check_availability(
        self,
        dataset_id: str,
        start_date: str,
        end_date: str,
        **kwargs: object,
    ) -> dict:
        """Call ``check_imagery_availability`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        args: dict[str, object] = {
            "dataset_id": dataset_id,
            "start_date": start_date,
            "end_date": end_date,
            **kwargs,
        }
        result = await self._session.call_tool(
            "check_imagery_availability",
            args,
        )
        return json.loads(result.content[0].text)  # type: ignore[union-attr]

    async def list_datasets(self) -> dict:
        """Call ``get_dataset_metadata`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool("list_datasets",{})
        return self._parse_result(result)

    async def generate_python_from_question(self, 
                                            question: str, 
                                            gee_datasets: list[dict] = None, 
                                            fix_code: bool = True) -> dict:
        """Call ``generate_python_from_question`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "generate_python_from_question", 
            {"question": question, "gee_datasets": gee_datasets, 'fix_code': fix_code}
        )
        return self._parse_result(result)

    async def generate_abstract_graph_from_question(self, 
                                                    question: str, 
                                                    gee_datasets: list[dict] = None) -> dict:
        """Call ``generate_abstract_graph_from_question`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "generate_abstract_graph_from_question", 
            {"question": question, "gee_datasets": gee_datasets}
        )
        return self._parse_result(result)

    async def generate_python_from_reasoning_steps(self,
                                                    question: str, 
                                                    reasoning_steps: str,
                                                    gee_datasets: list[dict] = None, 
                                                    fix_code: bool = True) -> dict:

        """Call ``generate_python_from_reasoning_steps`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "generate_python_from_reasoning_steps", 
            {"question": question, "gee_datasets": gee_datasets, 'reasoning_steps': reasoning_steps, 'fix_code': fix_code}
        )
        return self._parse_result(result)

    async def generate_python_from_abstract_graph(self,
                                                    question: str, 
                                                    abstract_graph: str,
                                                    gee_datasets: list = None, 
                                                    fix_code: bool = True) -> dict:

        """Call ``generate_python_from_abstract_graph`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "generate_python_from_abstract_graph", 
            {"question": question, "gee_datasets": gee_datasets, 'abstract_graph': abstract_graph, 'fix_code': fix_code}
        )
        return self._parse_result(result)

    async def execute_gee_python(self, code: str) -> dict:
        """Call ``execute_gee_python`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "execute_gee_python", 
            {"code": code}
        )
        return self._parse_result(result)

    async def extract_factuality_issues(self, question: str, python_code: str) -> dict:
        """Call ``extract_factuality_issues`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "extract_factuality_issues",
            {"question": question, "python_code": python_code}
        )
        return self._parse_result(result)

    async def get_datasets_locations_and_periods(self, question: str, gee_datasets: list[dict]) -> dict:
        """Call ``get_datasets_locations_and_periods`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "get_datasets_locations_and_periods",
            {"question": question, "gee_datasets": gee_datasets}
        )
        return self._parse_result(result)

    async def identify_sensible_variables(self, question: str, python_code: str, baseline_answer: str) -> list[dict]:
        """Call ``identify_sensible_variables`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "identify_sensible_variables",
            {"question": question, "python_code": python_code, 'baseline_answer':baseline_answer}
        )
        return self._parse_result(result)


    async def sensitivity_analysis(self, question: str, python_code: str, baseline_answer: str) -> list[dict]:
        """Call ``sensitivity_analysis`` on the MCP server."""
        assert self._session is not None, "Use as async context manager"
        result = await self._session.call_tool(
            "sensitivity_analysis",
            {"question": question, "python_code": python_code, 'baseline_answer':baseline_answer}
        )
        return self._parse_result(result)

