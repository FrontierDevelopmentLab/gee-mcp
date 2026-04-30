# Contributing to GEE MCP

Thanks for your interest in contributing! GEE MCP is an MCP server that
exposes Google Earth Engine as a set of tools. See the
[README](README.md) for what the project does and how to use it.

## Development setup

Follow the [Installation](README.md#installation) and
[Configuration](README.md#configuration) sections of the README, then
install the pre-commit hooks:

```bash
poetry run pre-commit install
```

## Code quality

Pre-commit runs black, isort, autoflake, mypy, pylint, detect-secrets,
and pytest with coverage. Run the full suite locally before pushing:

```bash
poetry run pre-commit run --all-files
```

If a hook fails, fix the underlying issue rather than bypassing the
hook. If `detect-secrets` flags a false positive, update the baseline:

```bash
poetry run detect-secrets scan --baseline .secrets.baseline
```

## Testing

```bash
poetry run pytest
```

The test suite sets `GEE_SKIP_AUTH=1` so you do not need GEE
credentials to run unit tests. New tools and behaviour changes should
ship with tests.

## Adding a new MCP tool

Tools live in `src/gee_mcp/`. Follow the structure of an existing tool
in the same category, add a test, and add the tool to the list in the
README.

## Pull requests

- Fork the repo and branch off `main`.
- Use a short branch prefix: `feat/`, `fix/`, `docs/`, or `chore/`.
- Keep each PR focused on one logical change.
- In the PR description, explain *what* changed and *why*, and link
  any related issue.
- Commit messages: imperative mood, focused on the *why*. 
- Be ready to iterate on review feedback.

## Reporting issues

Open a [GitHub issue](https://github.com/FrontierDevelopmentLab/gee-mcp/issues)
with:

- steps to reproduce,
- expected vs actual behaviour,
- your environment (Python version, OS, relevant package versions).

## For maintainers

Maintainers can push branches directly to the repo; the same branch
naming, PR, and review expectations apply.
