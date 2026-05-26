"""Shared exception types for programmatic (MCP / library) callers.

The CLI path keeps its existing sys.exit() behavior; the MCP server and the
job_runner use raise_on_error=True paths that raise these instead, so a single
bad request never kills a long-lived server process.
"""


class PipelineError(Exception):
    """Raised for recoverable pipeline failures (bad input, missing config,
    validation failure) that an MCP tool should report as a structured error
    rather than crash on."""


class SchemaGenerationError(PipelineError):
    """Raised when GPT-based job/question schema generation or the subsequent
    CSV append fails."""
