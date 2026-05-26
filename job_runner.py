"""Non-interactive orchestration for programmatic / MCP callers.

Wraps the existing pipeline (load_data -> allocate_context -> build_batches ->
process_batches) without the CLI prompts or sys.exit() calls, plus helpers to
stage dropped files into the Context dirs and to generate+persist a job schema
from a plain-text prompt.

Contract: one job-run at a time per process (the unified logger is a global
singleton). A calling project should invoke these serially.
"""

import os
import shutil
import asyncio
from datetime import datetime

from errors import PipelineError
from unified_logger import get_logger
from data_loader import load_data
from context_allocator import allocate_context
from batch_builder import build_batches
from batch_processor import process_batches, load_api_key
import schema_generator
import observability

RECORD_CONTEXT_DIR = 'Context/Record_Context'
QUESTION_CONTEXT_DIR = 'Context/Question_Context'
_ALLOWED_EXT = ('.csv', '.xlsx')


def _as_list(paths):
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _clean_dir(directory):
    """Remove non-hidden files from a directory (leaves the dir + dotfiles)."""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.startswith('.'):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            os.remove(path)


def _copy_files(paths, dest_dir):
    for p in paths:
        if not os.path.isfile(p):
            raise PipelineError(f"Input file does not exist: {p}")
        if not p.lower().endswith(_ALLOWED_EXT):
            raise PipelineError(f"Unsupported file type (need .csv/.xlsx): {p}")
        shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))


def stage_input_files(record_paths, question_context_paths=None, clean=True):
    """Copy dropped local files into the Context dirs so the existing loader
    can pick them up. Originals are left untouched (copy, not move).

    When clean=True both Context dirs are emptied first so a previous run's data
    does not leak into this one. Raises PipelineError for missing/unsupported paths.
    """
    record_paths = _as_list(record_paths)
    question_context_paths = _as_list(question_context_paths)

    if not record_paths:
        raise PipelineError("At least one record file path must be provided.")

    os.makedirs(RECORD_CONTEXT_DIR, exist_ok=True)
    os.makedirs(QUESTION_CONTEXT_DIR, exist_ok=True)

    if clean:
        _clean_dir(RECORD_CONTEXT_DIR)
        _clean_dir(QUESTION_CONTEXT_DIR)

    _copy_files(record_paths, RECORD_CONTEXT_DIR)
    _copy_files(question_context_paths, QUESTION_CONTEXT_DIR)


async def run_job(job_name, max_parallel_requests=50, max_records=None):
    """Run the pipeline for an already-staged job. Returns a summary dict.

    Raises PipelineError on job-not-found, context-validation failure, or when
    no batches are produced. If max_records is set, only the first N records are
    processed (one batch == one record), used for previews.
    """
    logger = get_logger(job_name)
    run_id = logger.session_id
    started_at = datetime.now().isoformat(timespec='seconds')
    record_files = sorted(
        f for f in os.listdir(RECORD_CONTEXT_DIR)
        if not f.startswith('.')
    ) if os.path.isdir(RECORD_CONTEXT_DIR) else []

    def _record_history(status, summary=None, error=None):
        entry = {
            'run_id': run_id,
            'job_name': job_name,
            'status': status,
            'started_at': started_at,
            'finished_at': datetime.now().isoformat(timespec='seconds'),
            'max_parallel_requests': max_parallel_requests,
            'record_files': record_files,
            'results_path': os.path.abspath(f"Results/{job_name}_results.csv"),
            'session_log': os.path.abspath(
                os.path.join('Logs', 'sessions', f"{job_name}_{run_id}.log")),
            'summary': summary or {},
        }
        if error:
            entry['error'] = error
        observability.record_run(entry)

    try:
        dataframes_dict = load_data(raise_on_error=True)

        job_cfg = dataframes_dict.get('GPA_Job_Configuration')
        if job_cfg is None or job_cfg[job_cfg['Job_Name'] == job_name].empty:
            raise PipelineError(
                f"Job '{job_name}' not found in GPA_Job_Configuration.csv"
            )

        allocation = allocate_context(dataframes_dict, job_name)
        if not allocation:
            raise PipelineError(
                "Context validation failed: a record + question context exceeds the "
                "configured token limit, or the job config is invalid."
            )

        batches_df = build_batches(dataframes_dict, job_name)
        if batches_df is None or batches_df.empty:
            raise PipelineError("No batches were created (no records or build error).")

        if max_records:
            batches_df = batches_df.head(max_records)

        summary = await process_batches(
            batches_df, dataframes_dict, job_name, logger,
            max_parallel_requests=max_parallel_requests,
        )
    except Exception as e:
        _record_history('error', error=str(e))
        raise

    _record_history('completed', summary=summary)

    result = {
        'run_id': run_id,
        'job_name': job_name,
        'results_path': os.path.abspath(f"Results/{job_name}_results.csv"),
    }
    result.update(summary or {})
    return result


def run_job_sync(job_name, max_parallel_requests=50, max_records=None):
    """Synchronous wrapper around run_job for non-async (MCP tool) callers.
    Runs in a dedicated thread with its own event loop, so this works whether
    the caller is in an existing event loop (e.g. an async MCP framework) or
    plain synchronous code."""
    import threading
    box: dict = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(
                run_job(job_name, max_parallel_requests=max_parallel_requests,
                        max_records=max_records)
            )
        except BaseException as exc:
            box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_runner, name=f"gpa-run-{job_name}")
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def create_job_from_prompt(prompt, record_paths, model=None,
                           question_context_paths=None):
    """Generate a job + question schema from a plain-text prompt and the dropped
    data file(s), then persist it to the configuration CSVs.

    Reads the record files in place to sample columns/rows (no staging needed).
    enum-typed questions are resolved against the basenames of any supplied
    question_context_paths; unbacked enums are downgraded to string.

    Returns a dict describing the created job.
    """
    record_paths = _as_list(record_paths)
    question_context_paths = _as_list(question_context_paths)
    if not record_paths:
        raise PipelineError("At least one record file path must be provided.")

    api_key = load_api_key()
    if not api_key:
        raise PipelineError("Could not load OpenAI API key from Configuration_Files/API_Keys.csv")

    file_samples = [schema_generator.sample_input_file(p) for p in record_paths]
    pricing_df = schema_generator._load_pricing()
    existing_names = schema_generator.load_existing_job_names()

    definition = schema_generator.generate_job_definition(
        prompt=prompt,
        file_samples=file_samples,
        existing_job_names=existing_names,
        api_key=api_key,
        model=model or schema_generator.DEFAULT_MODEL,
    )

    job_params = schema_generator.default_job_params(
        definition.get('suggested_model'), pricing_df
    )

    qctx_basenames = [os.path.basename(p) for p in question_context_paths]
    question_rows, enum_downgraded = schema_generator.finalize_questions(
        definition['questions'], qctx_basenames
    )

    job_name = definition['job_name']
    job_row = {
        'Job_Name': job_name,
        'Model': job_params['Model'],
        'Input_Context_Limit': job_params['Input_Context_Limit'],
        'Input_Context_Overhead': job_params['Input_Context_Overhead'],
        'Output_Context_Limit': job_params['Output_Context_Limit'],
        'Temperature': job_params['Temperature'],
        'Tool_Descriptions': definition.get('tool_description', ''),
        'Assistant_Role': definition.get('assistant_role', ''),
        'Apply_Relevance_Filter': job_params['Apply_Relevance_Filter'],
    }
    # Each question row needs its Job_Name to join back to the config.
    for row in question_rows:
        row['Job_Name'] = job_name

    schema_generator.append_job_to_config(job_row, question_rows)

    return {
        'job_name': job_name,
        'job': job_row,
        'questions': question_rows,
        'enum_downgraded': enum_downgraded,
        'config_files_updated': [
            os.path.abspath(schema_generator.JOB_CONFIG_PATH),
            os.path.abspath(schema_generator.QUESTIONS_PATH),
        ],
    }
