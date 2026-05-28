"""Read/write helpers for job history, transcripts, logs, and results.

Backs the observability MCP tools. Everything here is read-only against the
Logs/ and Results/ trees except record_run(), which appends one line per
pipeline run to Logs/run_history.jsonl.
"""

import os
import csv
import sys
import json
from datetime import datetime

import pandas as pd

LOGS_DIR = 'Logs'
RUN_HISTORY_PATH = os.path.join(LOGS_DIR, 'run_history.jsonl')
API_CALLS_CSV = os.path.join(LOGS_DIR, 'api_calls', 'api_calls.csv')
COSTS_CSV = os.path.join(LOGS_DIR, 'costs', 'costs.csv')
ERRORS_CSV = os.path.join(LOGS_DIR, 'errors', 'errors.csv')
SESSIONS_DIR = os.path.join(LOGS_DIR, 'sessions')
RESULTS_DIR = 'Results'

CONFIG_DIR = 'Configuration_Files'
JOB_CONFIG_PATH = os.path.join(CONFIG_DIR, 'GPA_Job_Configuration.csv')
QUESTIONS_PATH = os.path.join(CONFIG_DIR, 'GPA_Questions.csv')

# Transcript request_json fields can be large (all question context per record).
try:
    csv.field_size_limit(10_000_000)
except OverflowError:  # pragma: no cover - platform dependent
    csv.field_size_limit(sys.maxsize // 10)

_ENCODINGS = ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']


def _read_table(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    last = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last = e
    raise ValueError(f"Could not read '{path}': {last}")


def _maybe_json(value):
    if value in (None, ''):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


# --------------------------------------------------------------------------- #
# Run history
# --------------------------------------------------------------------------- #

def record_run(entry):
    """Append one run-history record (a dict) to Logs/run_history.jsonl."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(RUN_HISTORY_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def list_runs(job_name=None, limit=50):
    """Return run-history records (most recent first), optionally filtered."""
    if not os.path.isfile(RUN_HISTORY_PATH):
        return []
    runs = []
    with open(RUN_HISTORY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if job_name:
        runs = [r for r in runs if r.get('job_name') == job_name]
    runs.reverse()
    return runs[:limit]


# --------------------------------------------------------------------------- #
# Job definitions (plain description + output schema)
# --------------------------------------------------------------------------- #

def get_job_definition(job_name):
    """Full definition for one job: config fields + its output questions."""
    job_df = _read_table(JOB_CONFIG_PATH)
    match = job_df[job_df['Job_Name'] == job_name]
    if match.empty:
        return None
    cfg = match.iloc[0]

    questions = []
    if os.path.isfile(QUESTIONS_PATH):
        q_df = _read_table(QUESTIONS_PATH)
        for _, q in q_df[q_df['Job_Name'] == job_name].iterrows():
            enum_file = q.get('enum_file_name')
            questions.append({
                'key': q.get('Key'),
                'type': q.get('Type'),
                'description': q.get('Description'),
                'enum_file_name': None if pd.isna(enum_file) else enum_file,
            })

    def _clean(v):
        return None if pd.isna(v) else v

    return {
        'job_name': job_name,
        'description': _clean(cfg.get('Tool_Descriptions')),
        'assistant_role': _clean(cfg.get('Assistant_Role')),
        'model': _clean(cfg.get('Model')),
        'temperature': _clean(cfg.get('Temperature')),
        'input_context_limit': _clean(cfg.get('Input_Context_Limit')),
        'output_context_limit': _clean(cfg.get('Output_Context_Limit')),
        'questions': questions,
    }


# --------------------------------------------------------------------------- #
# Transcripts (prompt sent + answer received, per record)
# --------------------------------------------------------------------------- #

def get_transcripts(job_name=None, session_id=None, batch_id=None, status=None,
                    limit=50, include_request=True, include_response=True):
    """Return per-record API transcripts from Logs/api_calls/api_calls.csv.

    Each entry includes timestamp, session_id (run), batch_id, model, tokens,
    cost, status, and (optionally) the parsed request messages and response.
    """
    if not os.path.isfile(API_CALLS_CSV):
        return []

    rows = []
    with open(API_CALLS_CSV, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if job_name and r.get('job_name') != job_name:
                continue
            if session_id and r.get('session_id') != session_id:
                continue
            if batch_id is not None and str(r.get('batch_id')) != str(batch_id):
                continue
            if status and r.get('status') != status:
                continue
            entry = {
                'timestamp': r.get('timestamp'),
                'session_id': r.get('session_id'),
                'job_name': r.get('job_name'),
                'batch_id': r.get('batch_id'),
                'model': r.get('model'),
                'temperature': r.get('temperature'),
                'input_tokens': r.get('input_tokens'),
                # Present on rows written after the cached-pricing change; missing
                # (None) on older rows from before — readers must handle null.
                'cached_input_tokens': r.get('cached_input_tokens'),
                'output_tokens': r.get('output_tokens'),
                'total_cost': r.get('total_cost'),
                'status': r.get('status'),
            }
            if include_request:
                entry['request'] = _maybe_json(r.get('request_json'))
            if include_response:
                entry['response'] = _maybe_json(r.get('response_json'))
            rows.append(entry)

    rows.reverse()
    return rows[:limit]


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

def get_results(job_name, limit=100, offset=0):
    """Return rows from Results/{job_name}_results.csv with paging metadata."""
    path = os.path.join(RESULTS_DIR, f"{job_name}_results.csv")
    if not os.path.isfile(path):
        return {'results_path': os.path.abspath(path), 'exists': False,
                'total_rows': 0, 'rows': []}
    df = _read_table(path)
    total = len(df)
    window = df.iloc[offset:offset + limit].where(pd.notna(df), None)
    return {
        'results_path': os.path.abspath(path),
        'exists': True,
        'total_rows': int(total),
        'offset': offset,
        'limit': limit,
        'rows': window.to_dict('records'),
    }


# --------------------------------------------------------------------------- #
# Logs (errors + session log text)
# --------------------------------------------------------------------------- #

def get_errors(job_name=None, limit=100):
    """Return ERROR/CRITICAL rows from Logs/errors/errors.csv (most recent first)."""
    if not os.path.isfile(ERRORS_CSV):
        return []
    df = _read_table(ERRORS_CSV)
    if job_name and 'job_name' in df.columns:
        df = df[df['job_name'] == job_name]
    df = df.where(pd.notna(df), None)
    rows = df.to_dict('records')
    rows.reverse()
    return rows[:limit]


def get_session_log(job_name=None, session_id=None, max_lines=300):
    """Return the tail of a session log. If session_id is given, match that file;
    else the most recent session log (optionally for job_name)."""
    if not os.path.isdir(SESSIONS_DIR):
        return {'log_file': None, 'lines': []}

    candidates = [f for f in os.listdir(SESSIONS_DIR) if f.endswith('.log')]
    if session_id:
        candidates = [f for f in candidates if session_id in f]
    if job_name:
        candidates = [f for f in candidates if f.startswith(job_name + '_')]
    if not candidates:
        return {'log_file': None, 'lines': []}

    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(SESSIONS_DIR, f)))
    chosen = candidates[-1]
    path = os.path.join(SESSIONS_DIR, chosen)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    return {'log_file': os.path.abspath(path), 'total_lines': len(lines),
            'lines': lines[-max_lines:]}
