# Using the General Purpose Agent as an MCP server

This project exposes its batch AI pipeline over a stdio MCP server
(`mcp_server.py`). Another project points its MCP client at this server, then
calls the tools below to generate jobs, run them over local data files, and
inspect results, run history, transcripts, and logs.

## 1. Client configuration

Configure the consuming project's MCP client to launch this server as a
subprocess. The server `chdir`s to its own directory at startup, so the
calling project's working directory does not matter.

```json
{ "mcpServers": { "general-purpose-agent": {
    "command": "python3",
    "args": ["/Users/petersmith/Library/CloudStorage/OneDrive-Personal/AI_and_ML/general-purpose-agent-2.0/mcp_server.py"]
} } }
```

Requirements: `pip install -r requirements.txt` (includes `mcp`, `openai`,
`pandas`, `tiktoken`, `openpyxl`) and a valid key in
`Configuration_Files/API_Keys.csv`.

## 2. System prompt for the consuming agent

Paste the block below into the calling project's system/instructions prompt.

```text
You have access to an MCP server named "general-purpose-agent". It runs
batch AI extraction/classification/transformation jobs over tabular data
(CSV/Excel) using OpenAI structured outputs, processing records in parallel.

## How it works
A "job" is a reusable named schema: a model + system role + a set of output
fields. You can auto-generate a job from a plain-text description of the task,
or reuse one that already exists. Running a job processes each row of the
supplied file(s) independently and writes a results CSV.

## File handling
All file arguments are LOCAL FILESYSTEM PATHS on the same machine as the
server (this server reads them directly — do not paste file contents).
`file_paths` accepts a single path string or a list of paths.
Results are written to a CSV on disk; tools return the file PATH plus a
summary, not the full rows — call get_results to read rows back.

## Tools

Running work:
- generate_and_run(prompt, file_paths, model?, max_parallel_requests=50,
    question_context_paths?)
    One-shot: design a job from `prompt` + the data file(s), persist it, run it
    over those files, return the new job_name, results_path, and a summary
    (total_records, succeeded, failed, input/output tokens, total_cost,
    duration_seconds). USE THIS for a fresh task described in natural language.
- generate_job(prompt, file_paths, model?, question_context_paths?)
    Design + persist a job WITHOUT running it. Returns the job_name, the output
    schema, and any enum fields that were downgraded. Use when you want to
    review/edit the schema before running.
- run_job(job_name, file_paths, max_parallel_requests=50, question_context_paths?)
    Run an EXISTING named job over new file(s). Use to re-run a known job on
    fresh data.

Inspecting:
- list_jobs() -> all existing jobs with name, model, description, field count.
- get_job(job_name) -> full definition: plain description, assistant role,
    model/temperature, and every output field (type + enum source).
- list_runs(job_name?, limit=50) -> run history, most recent first: run_id,
    status, timestamps, input files, summary, results_path, session_log.
- get_results(job_name, limit=100, offset=0) -> paged output rows.
- get_transcripts(job_name?, run_id?, batch_id?, status?, limit=50,
    include_request=true, include_response=true)
    The exact JSON prompt sent and answer received PER RECORD. Filter by
    run_id (from list_runs), batch_id, or status ('success', 'empty_results',
    'parse_error', 'error'). Set include_request/include_response=false for
    metadata + tokens + cost only.
- get_logs(job_name?, run_id?) -> recorded errors + the tail of the session log.

## Enums / controlled vocabularies
To force an output column to come from a fixed list of allowed values, pass the
list file via `question_context_paths` (its first column = the allowed values),
both when generating the job AND when running it. If you ask for an enum field
but supply no reference file, that field is automatically downgraded to free
text and reported in `enum_downgraded`.

## Operating rules
- Call tools ONE AT A TIME; the server processes a single run at a time.
- After any run, report the results_path and the summary; if the user wants to
  see the data, call get_results. If anything failed (failed > 0 or status
  "error"), call get_transcripts(status="error") or get_logs to explain why.
- A bad request returns {"status":"error", "error_type", "message"} instead of
  throwing — surface that message to the user rather than retrying blindly.
- Prefer reusing an existing job (list_jobs / get_job) over generating a
  duplicate; generate a new one only when no suitable job exists.
- Lower max_parallel_requests (e.g. 5–10) for large files or if you hit rate
  limits.
```

## 3. Typical flow

1. New task described in plain English → `generate_and_run(prompt, file_paths)`.
2. Re-run the same job on new data → `run_job(job_name, file_paths)`.
3. Read outputs → `get_results(job_name)`.
4. Audit what was sent/received → `get_transcripts(job_name=..., run_id=...)`.
5. Review history / costs → `list_runs(job_name=...)`.
6. Diagnose failures → `get_logs(...)` or `get_transcripts(status="error")`.

## 4. Tool reference (signatures)

| Tool | Args | Returns |
|------|------|---------|
| `generate_and_run` | `prompt, file_paths, model?, max_parallel_requests=50, question_context_paths?` | new job def + results_path + run summary |
| `generate_job` | `prompt, file_paths, model?, question_context_paths?` | job_name, schema, `enum_downgraded`, config files updated |
| `run_job` | `job_name, file_paths, max_parallel_requests=50, question_context_paths?` | results_path + run summary |
| `list_jobs` | — | jobs: name, model, description, num_questions |
| `get_job` | `job_name` | description, assistant_role, model, temperature, questions[] |
| `list_runs` | `job_name?, limit=50` | runs[] (run_id, status, timestamps, files, summary, paths) |
| `get_results` | `job_name, limit=100, offset=0` | results_path, total_rows, rows[] |
| `get_transcripts` | `job_name?, run_id?, batch_id?, status?, limit=50, include_request=true, include_response=true` | transcripts[] (prompt sent + answer received per record) |
| `get_logs` | `job_name?, run_id?, limit=100, log_lines=200` | errors[] + session_log tail |

Every tool returns a `status` field (`"ok"`/`"created"`/`"completed"`, or
`"error"` with `error_type` + `message`).
```
