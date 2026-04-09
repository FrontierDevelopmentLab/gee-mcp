# GEE MCP Server

This is a standalone Model Context Protocol (MCP) server for Google Earth Engine (GEE). It provides tools for dataset discovery, metadata extraction, and executing GEE Python code.

## Project Structure

- `server/`: The MCP server package.
- `client.py`: An example MCP client to test the server.
- `requirements.txt`: List of dependencies.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have authenticated with Google Earth Engine and your environment is set up.

## Environment Variables

The server requires several environment variables to be set for proper authentication and feature access.

- `GEE_PROJECT`: The GEE project id (required by `auth.py`).
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: The Gemini API key (required by `tools_catalogue.py` and `tools_analysis.py` for AI features).
- `VERTEXAI_PROJECT` (optional): The GCP project ID for Vertex AI (used if no API key is provided).
- `VERTEXAI_LOCATION` (optional): The GCP region for Vertex AI (defaults to `"global"`).
- `GEE_SERVICE_ACCOUNT` (optional): The GEE service account email.
- `GEE_PRIVATE_KEY_PATH` or `GOOGLE_APPLICATION_CREDENTIALS` (optional): Path to the JSON key file for the service account.
- `GEE_AUTH_MODE` (optional): The GEE authentication mode (defaults to `"gcloud"`).

## Running the Server

You can run the server via stdio (as a module):

```bash
python -m server
```

## Testing with Client

You can run the example client to test connection and tools:

```bash
python client.py
```
