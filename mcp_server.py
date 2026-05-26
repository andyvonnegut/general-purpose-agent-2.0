"""stdio MCP server exposing the General Purpose Agent pipeline as tools.

A calling project spawns this as a subprocess (`python mcp_server.py`) and
invokes the tools below over stdio. The engine reads/writes files relative to
the project root, so this module chdir's to its own directory at import — the
calling project's working directory does not matter.

Tools:
  - list_jobs         : enumerate existing reusable jobs
  - generate_job      : GPT-generate + persist a job schema from a prompt + files
  - run_job           : run an existing named job over dropped file paths
  - generate_and_run  : drop files + prompt -> generate -> run -> results path

Contract: processes one job-run at a time (the underlying logger is a global
singleton). Call serially.
"""

import os
import traceback
from typing import List, Optional, Union

# Root all relative paths used by the engine at the project directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP

import schema_generator
import job_runner
import observability
from errors import PipelineError, SchemaGenerationError

mcp = FastMCP("general-purpose-agent")

FilePaths = Union[str, List[str]]


def _error(exc: Exception) -> dict:
    """Convert an exception into a structured error result (never raised out
    of a tool, so the server stays alive across bad requests)."""
    if isinstance(exc, (PipelineError, SchemaGenerationError)):
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
    return {
        "status": "error",
        "error_type": "Unexpected",
        "message": str(exc),
        "trace": traceback.format_exc()[:2000],
    }


@mcp.tool()
def list_jobs() -> dict:
    """List the jobs already defined in the configuration CSVs.

    Returns each job's name, model, description, and number of output questions.
    Works even when the Context input directories are empty.
    """
    try:
        job_cfg = schema_generator._read_table(schema_generator.JOB_CONFIG_PATH)
        questions = schema_generator._read_table(schema_generator.QUESTIONS_PATH)
        counts = questions.groupby('Job_Name').size().to_dict() if 'Job_Name' in questions else {}

        jobs = []
        for _, r in job_cfg.iterrows():
            name = str(r.get('Job_Name', ''))
            jobs.append({
                "job_name": name,
                "model": str(r.get('Model', '')),
                "tool_description": str(r.get('Tool_Descriptions', '')),
                "num_questions": int(counts.get(name, 0)),
            })
        return {"status": "ok", "jobs": jobs}
    except Exception as e:
        return _error(e)


@mcp.tool()
def generate_job(
    prompt: str,
    file_paths: FilePaths,
    model: Optional[str] = None,
    question_context_paths: Optional[List[str]] = None,
) -> dict:
    """Generate a job + output-field schema from a plain-text prompt and the
    dropped data file(s), and persist it to the configuration CSVs as a reusable
    named job. Does NOT run the pipeline.

    Args:
        prompt: Plain-text description of the extraction/transformation task.
        file_paths: Local path(s) to the data file(s) (.csv/.xlsx) to sample.
        model: Optional OpenAI model id for the resulting job (validated against
            API_Pricing.csv; a default is used if unknown).
        question_context_paths: Optional local path(s) to controlled-vocabulary
            reference list files. To get an enum/fixed-choice output column, pass
            the list file here; its first column is treated as the allowed values.
            Pass the same file(s) to run_job when you run the job.

    Returns the created job_name, the job row, the question rows, any enum
    columns that were downgraded to string for lack of a reference file, and the
    config files updated.
    """
    try:
        result = job_runner.create_job_from_prompt(
            prompt=prompt,
            record_paths=file_paths,
            model=model,
            question_context_paths=question_context_paths,
        )
        result["status"] = "created"
        return result
    except Exception as e:
        return _error(e)


@mcp.tool()
def run_job(
    job_name: str,
    file_paths: FilePaths,
    max_parallel_requests: int = 50,
    question_context_paths: Optional[List[str]] = None,
) -> dict:
    """Run an existing named job over the dropped data file(s), record by record.

    The data file(s) and any question-context file(s) are staged into the
    project's Context dirs (previous run's inputs are cleared first), then the
    parallel pipeline runs. Results are written to Results/{job_name}_results.csv.

    Args:
        job_name: Name of a job that exists in GPA_Job_Configuration.csv.
        file_paths: Local path(s) to the data file(s) (.csv/.xlsx) to process.
        max_parallel_requests: Max concurrent OpenAI requests (default 50).
        question_context_paths: Optional path(s) to example/context or enum
            reference files attached to every record.

    Returns the absolute results_path plus a summary: total_records, succeeded,
    failed, input_tokens, output_tokens, total_cost, duration_seconds.
    """
    try:
        job_runner.stage_input_files(
            record_paths=file_paths,
            question_context_paths=question_context_paths,
            clean=True,
        )
        summary = job_runner.run_job_sync(job_name, max_parallel_requests=max_parallel_requests)
        summary["status"] = "completed"
        return summary
    except Exception as e:
        return _error(e)


@mcp.tool()
def generate_and_run(
    prompt: str,
    file_paths: FilePaths,
    model: Optional[str] = None,
    max_parallel_requests: int = 50,
    question_context_paths: Optional[List[str]] = None,
) -> dict:
    """End-to-end: from dropped file(s) + a prompt, generate and persist a job
    schema, then run it over the same file(s) and return the results path.

    Combines generate_job and run_job. The files are staged once. Returns the
    created job definition fields merged with the run summary (status=completed).
    """
    try:
        # Stage first so the same files back both generation sampling and the run.
        job_runner.stage_input_files(
            record_paths=file_paths,
            question_context_paths=question_context_paths,
            clean=True,
        )
        created = job_runner.create_job_from_prompt(
            prompt=prompt,
            record_paths=file_paths,
            model=model,
            question_context_paths=question_context_paths,
        )
        summary = job_runner.run_job_sync(
            created["job_name"], max_parallel_requests=max_parallel_requests
        )
        result = {**created, **summary, "status": "completed"}
        return result
    except Exception as e:
        return _error(e)


# --------------------------------------------------------------------------- #
# Observability tools: job descriptions, run history, transcripts, results, logs
# --------------------------------------------------------------------------- #

@mcp.tool()
def get_job(job_name: str) -> dict:
    """Get the full definition of a job: its plain-language description,
    assistant role, model/temperature, and the list of output fields (questions)
    with their types and any enum reference file.
    """
    try:
        definition = observability.get_job_definition(job_name)
        if definition is None:
            return {"status": "error", "error_type": "PipelineError",
                    "message": f"Job '{job_name}' not found."}
        return {"status": "ok", **definition}
    except Exception as e:
        return _error(e)


@mcp.tool()
def list_runs(job_name: Optional[str] = None, limit: int = 50) -> dict:
    """List past pipeline runs (most recent first), optionally filtered by job.

    Each run record has run_id, job_name, status, started_at/finished_at, the
    input files used, a summary (records/succeeded/failed/tokens/cost/duration),
    and paths to the results CSV and session log. Use run_id with get_transcripts
    or get_logs to drill into a specific run.
    """
    try:
        return {"status": "ok", "runs": observability.list_runs(job_name=job_name, limit=limit)}
    except Exception as e:
        return _error(e)


@mcp.tool()
def get_transcripts(
    job_name: Optional[str] = None,
    run_id: Optional[str] = None,
    batch_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    include_request: bool = True,
    include_response: bool = True,
) -> dict:
    """Retrieve per-record API transcripts — the exact JSON prompt sent and the
    answer received for each processed record.

    Filter by job_name, run_id (a run's session id), batch_id, or status
    ('success', 'empty_results', 'parse_error', 'error'). Set include_request /
    include_response to False to omit the (potentially large) prompt/answer
    bodies and return just metadata + tokens + cost.
    """
    try:
        transcripts = observability.get_transcripts(
            job_name=job_name, session_id=run_id, batch_id=batch_id, status=status,
            limit=limit, include_request=include_request, include_response=include_response,
        )
        return {"status": "ok", "count": len(transcripts), "transcripts": transcripts}
    except Exception as e:
        return _error(e)


@mcp.tool()
def get_results(job_name: str, limit: int = 100, offset: int = 0) -> dict:
    """Read rows from a job's results CSV (Results/{job_name}_results.csv).

    Returns the absolute results_path, total_rows, and a paged window of rows.
    """
    try:
        return {"status": "ok", **observability.get_results(job_name, limit=limit, offset=offset)}
    except Exception as e:
        return _error(e)


@mcp.tool()
def get_logs(
    job_name: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 100,
    log_lines: int = 200,
) -> dict:
    """Get logs for troubleshooting: recorded errors plus the tail of the
    relevant session log.

    Filter errors by job_name; select the session log by run_id (else the most
    recent for the job). Returns {errors: [...], session_log: {log_file, lines}}.
    """
    try:
        return {
            "status": "ok",
            "errors": observability.get_errors(job_name=job_name, limit=limit),
            "session_log": observability.get_session_log(
                job_name=job_name, session_id=run_id, max_lines=log_lines),
        }
    except Exception as e:
        return _error(e)


if __name__ == "__main__":
    mcp.run()
