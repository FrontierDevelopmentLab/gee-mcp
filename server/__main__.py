"""Entry point for ``python -m server``."""

from .app import mcp

mcp.run(transport="stdio")
