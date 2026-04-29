"""Entry point for ``python -m gee_mcp.server``."""

from . import mcp


def main():
    """Run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
