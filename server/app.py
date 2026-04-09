"""FastMCP application instance and logging configuration."""

import logging
import os

from fastmcp import FastMCP

log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SERVER - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)

mcp = FastMCP("mcp-gee-server")
