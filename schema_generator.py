"""GPT-powered generation of GPA job + question schemas from a plain-text prompt.

Given a task description and the dropped data file(s), this module asks GPT to
design an extraction job: a job-config row plus a list of output "questions"
(the structured-output fields). The result is sanitized, validated, and
appended to the two configuration CSVs so it becomes a reusable named job that
the existing pipeline can run.

The OpenAI call reuses the project's structured-output pattern
(client.beta.chat.completions.parse with a strict json_schema response_format).
"""

import os
import re
import csv
import json

import pandas as pd
from openai import OpenAI

from errors import SchemaGenerationError

CONFIG_DIR = 'Configuration_Files'
JOB_CONFIG_PATH = os.path.join(CONFIG_DIR, 'GPA_Job_Configuration.csv')
QUESTIONS_PATH = os.path.join(CONFIG_DIR, 'GPA_Questions.csv')
PRICING_PATH = os.path.join(CONFIG_DIR, 'API_Pricing.csv')

DEFAULT_MODEL = 'gpt-5-2025-08-07'
DEFAULT_INPUT_CONTEXT_LIMIT = 112000
DEFAULT_INPUT_CONTEXT_OVERHEAD = 8000
DEFAULT_OUTPUT_CONTEXT_LIMIT = 16000

VALID_TYPES = {'string', 'integer', 'number', 'boolean', 'enum'}
_ENCODINGS = ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']


def _read_table(path):
    """Read a CSV/Excel file with the same encoding fallbacks as data_loader."""
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    last_err = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            continue
    raise SchemaGenerationError(f"Could not read '{path}': {last_err}")


def sample_input_file(path, max_rows=5, max_cols=60):
    """Summarize a data file for the schema-generation prompt.

    Returns a dict with columns, dtypes, a few sample rows (cell values
    truncated to bound token usage), the row count, and the basename.
    """
    if not os.path.isfile(path):
        raise SchemaGenerationError(f"Input file does not exist: {path}")
    if not (path.endswith('.csv') or path.endswith('.xlsx')):
        raise SchemaGenerationError(f"Unsupported file type (need .csv/.xlsx): {path}")

    df = _read_table(path)
    cols = list(df.columns)[:max_cols]

    def _truncate(v):
        s = '' if pd.isna(v) else str(v)
        return s[:200]

    sample_rows = [
        {c: _truncate(r[c]) for c in cols}
        for _, r in df.head(max_rows).iterrows()
    ]

    return {
        'source_file': os.path.basename(path),
        'row_count': int(len(df)),
        'columns': cols,
        'dtypes': {c: str(df[c].dtype) for c in cols},
        'sample_rows': sample_rows,
    }


def _load_pricing():
    if not os.path.isfile(PRICING_PATH):
        return pd.DataFrame(columns=['Model', 'Supported_Temperatures'])
    return _read_table(PRICING_PATH)


def _context_window_for(model, pricing_df):
    """Return the model's input context window from API_Pricing.csv's
    Context_Window column, or DEFAULT_INPUT_CONTEXT_LIMIT if the column is
    absent or the value is missing/unparseable. This lets the generated
    Input_Context_Limit track each model's real window (e.g. gpt-5.4's large
    window) instead of a flat default.
    """
    if pricing_df is None or 'Context_Window' not in pricing_df or 'Model' not in pricing_df:
        return DEFAULT_INPUT_CONTEXT_LIMIT
    match = pricing_df[pricing_df['Model'].astype(str) == str(model)]
    if match.empty:
        return DEFAULT_INPUT_CONTEXT_LIMIT
    raw = match['Context_Window'].values[0]
    if pd.isna(raw) or str(raw).strip() == '':
        return DEFAULT_INPUT_CONTEXT_LIMIT
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_INPUT_CONTEXT_LIMIT


def default_job_params(model, pricing_df=None):
    """Resolve job-config defaults, validating the model against API_Pricing.csv
    and choosing a temperature compatible with that model.
    """
    if pricing_df is None:
        pricing_df = _load_pricing()

    known_models = set(pricing_df['Model'].astype(str)) if 'Model' in pricing_df else set()
    chosen_model = model if (model and model in known_models) else DEFAULT_MODEL

    # Temperature 1 is valid for every row in API_Pricing.csv ("0-2" and
    # "1 (default only)" alike), so it is always the safe default.
    temperature = 1

    return {
        'Model': chosen_model,
        'Input_Context_Limit': _context_window_for(chosen_model, pricing_df),
        'Input_Context_Overhead': DEFAULT_INPUT_CONTEXT_OVERHEAD,
        'Output_Context_Limit': DEFAULT_OUTPUT_CONTEXT_LIMIT,
        'Temperature': temperature,
        'Apply_Relevance_Filter': 'No',
    }


def _clean_identifier(raw, fallback):
    """Collapse to [A-Za-z0-9_-], spaces -> underscore. Used for keys/names."""
    s = re.sub(r'\s+', '_', str(raw).strip())
    s = re.sub(r'[^A-Za-z0-9_-]', '', s)
    return s or fallback


def sanitize_job_name(raw, existing):
    """Return an OpenAI-schema-safe (^[A-Za-z0-9_-]+$), unique Job_Name."""
    base = _clean_identifier(raw, 'Generated_Job')[:50]
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def load_existing_job_names():
    """Union of Job_Name values across both config CSVs (empty if absent)."""
    names = set()
    for path in (JOB_CONFIG_PATH, QUESTIONS_PATH):
        if os.path.isfile(path):
            df = _read_table(path)
            if 'Job_Name' in df.columns:
                names.update(df['Job_Name'].dropna().astype(str))
    return names


# JSON schema GPT must return (strict mode: all props required,
# additionalProperties false, nullable fields typed as [..,"null"]).
JOB_DEFINITION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "gpa_job_definition",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["job_name", "tool_description", "assistant_role",
                         "suggested_model", "questions"],
            "properties": {
                "job_name": {
                    "type": "string",
                    "description": "Short job name, letters/digits/underscores only, e.g. Invoice_Extractor.",
                },
                "tool_description": {
                    "type": "string",
                    "description": "One-sentence description of what this job extracts (becomes Tool_Descriptions).",
                },
                "assistant_role": {
                    "type": "string",
                    "description": "System/developer persona instruction for the model (becomes Assistant_Role).",
                },
                "suggested_model": {
                    "type": "string",
                    "description": "An OpenAI model id appropriate for the task. Validated against the allowed list; replaced with a default if unknown.",
                },
                "questions": {
                    "type": "array",
                    "description": "One entry per output field to extract from each record.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["key", "type", "description", "max_length", "enum_source_hint"],
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Output field name, letters/digits/underscores only.",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["string", "integer", "number", "boolean", "enum"],
                                "description": "Value type. Use 'enum' only when answers must come from a fixed reference list supplied separately as a context file.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Instruction to the model for how to populate this field.",
                            },
                            "max_length": {
                                "type": ["integer", "null"],
                                "description": "Optional max character length hint, or null.",
                            },
                            "enum_source_hint": {
                                "type": ["string", "null"],
                                "description": "If type is 'enum', describe what the allowed-value list contains; otherwise null.",
                            },
                        },
                    },
                },
            },
        },
    },
}

_DEVELOPER_INSTRUCTIONS = (
    "You design structured-extraction job schemas. Given a task description and "
    "samples of the input data, produce a concise job definition: a short job_name, "
    "a one-sentence tool_description, an assistant_role persona, a suggested OpenAI "
    "model, and a list of output questions (one per field to extract from each "
    "record). Prefer the most specific scalar type (boolean/integer/number) over "
    "string when appropriate. Use type 'enum' ONLY when the answer must come from a "
    "fixed controlled vocabulary that will be supplied as a separate reference list; "
    "in that case set enum_source_hint to describe the list. Keys must be valid "
    "identifiers (letters, digits, underscores)."
)


def generate_job_definition(prompt, file_samples, existing_job_names,
                            api_key, model=DEFAULT_MODEL):
    """Call GPT (structured output) to produce a validated job definition dict.

    Returns: {'job_name', 'tool_description', 'assistant_role',
              'suggested_model', 'questions': [ {key,type,description,
              max_length,enum_source_hint}, ... ]} with job_name sanitized to
    be unique and keys cleaned/de-duplicated.
    """
    client = OpenAI(api_key=api_key)
    user_payload = {
        "task_prompt": prompt,
        "input_files": file_samples,
    }
    messages = [
        {"role": "developer", "content": _DEVELOPER_INSTRUCTIONS},
        {"role": "user", "content": f"Task description:\n{prompt}"},
        {"role": "user", "content": "Input data samples (JSON):\n" + json.dumps(user_payload, default=str)},
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=JOB_DEFINITION_SCHEMA,
        )
    except Exception as e:
        raise SchemaGenerationError(f"GPT schema-generation call failed: {e}")

    content = completion.choices[0].message.content
    refusal = getattr(completion.choices[0].message, 'refusal', None)
    if refusal:
        raise SchemaGenerationError(f"Model refused schema generation: {refusal}")
    if not content:
        raise SchemaGenerationError("Model returned empty content for schema generation.")

    try:
        definition = json.loads(content)
    except json.JSONDecodeError as e:
        raise SchemaGenerationError(f"Could not parse model output as JSON: {e}")

    questions = definition.get('questions') or []
    if not questions:
        raise SchemaGenerationError("Generated definition contained no questions.")

    # Sanitize / validate questions
    seen_keys = set()
    clean_questions = []
    for i, q in enumerate(questions):
        qtype = (q.get('type') or 'string').strip()
        if qtype not in VALID_TYPES:
            raise SchemaGenerationError(f"Question {i} has invalid type '{qtype}'.")
        key = _clean_identifier(q.get('key', ''), f"field_{i + 1}")
        # de-duplicate keys within the job
        if key in seen_keys:
            j = 2
            while f"{key}_{j}" in seen_keys:
                j += 1
            key = f"{key}_{j}"
        seen_keys.add(key)
        clean_questions.append({
            'key': key,
            'type': qtype,
            'description': (q.get('description') or '').strip(),
            'max_length': q.get('max_length'),
            'enum_source_hint': q.get('enum_source_hint'),
        })

    definition['job_name'] = sanitize_job_name(definition.get('job_name', ''), set(existing_job_names))
    definition['questions'] = clean_questions
    return definition


def finalize_questions(questions, question_context_basenames):
    """Resolve enum questions against available reference files.

    For each type=='enum' question, assign an enum_file_name from the supplied
    context-file basenames (match by name/hint similarity, else the sole file
    if only one is available). If none can back it, downgrade to 'string'.

    Returns (rows, enum_downgraded) where rows have keys
    Key/Type/Description/Max_Length/enum_file_name and enum_downgraded lists the
    keys that were downgraded.
    """
    basenames = list(question_context_basenames or [])
    enum_downgraded = []
    rows = []

    for q in questions:
        qtype = q['type']
        enum_file = ''
        if qtype == 'enum':
            match = _match_enum_file(q, basenames)
            if match:
                enum_file = match
            else:
                qtype = 'string'
                enum_downgraded.append(q['key'])
        rows.append({
            'Key': q['key'],
            'Type': qtype,
            'Description': q['description'],
            'Max_Length': '' if q.get('max_length') in (None, '') else q['max_length'],
            'enum_file_name': enum_file,
        })

    return rows, enum_downgraded


def _match_enum_file(question, basenames):
    if not basenames:
        return None
    if len(basenames) == 1:
        return basenames[0]
    haystack = f"{question.get('key', '')} {question.get('enum_source_hint') or ''}".lower()
    for b in basenames:
        stem = os.path.splitext(b)[0].lower()
        if stem and (stem in haystack or any(tok and tok in haystack for tok in re.split(r'[_\s-]+', stem))):
            return b
    return basenames[0]


def append_job_to_config(job_row, question_rows):
    """Append one job row and N question rows to the config CSVs.

    Preserves each file's existing column order and uses csv.QUOTE_MINIMAL so
    embedded commas/quotes/newlines are written safely. Appends are written as
    plain utf-8 (the files already carry a leading BOM from utf-8-sig).
    """
    if not os.path.isfile(JOB_CONFIG_PATH):
        raise SchemaGenerationError(f"Missing {JOB_CONFIG_PATH}")
    if not os.path.isfile(QUESTIONS_PATH):
        raise SchemaGenerationError(f"Missing {QUESTIONS_PATH}")

    job_cols = list(_read_table(JOB_CONFIG_PATH).columns)
    q_cols = list(_read_table(QUESTIONS_PATH).columns)

    # On-disk files do NOT contain the in-memory 'source_file' column.
    job_cols = [c for c in job_cols if c != 'source_file']
    q_cols = [c for c in q_cols if c != 'source_file']

    def _ensure_trailing_newline(path):
        # The shipped CSVs may not end in a newline; without this the first
        # appended row would merge onto the last existing line.
        if os.path.getsize(path) == 0:
            return
        with open(path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last not in (b'\n', b'\r'):
            with open(path, 'a', newline='', encoding='utf-8') as f:
                f.write('\r\n')

    def _append(path, columns, rows):
        _ensure_trailing_newline(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            for row in rows:
                writer.writerow(['' if row.get(c) is None else row.get(c) for c in columns])

    try:
        _append(JOB_CONFIG_PATH, job_cols, [job_row])
        _append(QUESTIONS_PATH, q_cols, question_rows)
    except Exception as e:
        raise SchemaGenerationError(
            f"Failed to append generated job to config CSVs: {e}. "
            f"Job '{job_row.get('Job_Name')}' may be partially written."
        )
