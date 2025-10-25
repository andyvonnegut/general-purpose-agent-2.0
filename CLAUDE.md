# General Purpose Agent (GPA) - AI-Powered Batch Processing System

## Overview
The General Purpose Agent is a sophisticated batch processing system that leverages OpenAI's API for data transformation and analysis tasks. It processes large datasets through configurable jobs, intelligently chunking data to optimize token usage while maintaining context integrity.

**Version**: 3.0
**Language**: Python 3.11+
**Primary Framework**: OpenAI API with structured outputs

## Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (venv/ directory exists but is gitignored)
- OpenAI API key configured in `Configuration_Files/API_Keys.csv`

### Running the Application
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run the main application
python3 main.py

# Follow the interactive prompts to select a job
```

## System Architecture

### Data Flow Pipeline
```
1. main.py (orchestrator)
   ↓
2. data_loader.py (loads CSV/Excel files)
   ↓
3. context_allocator.py (calculates token budgets, adds JSON/token_count columns)
   ↓
4. record_context_chunker.py (chunks records by token limit)
   ↓
5. question_context_chunker.py (chunks examples by token limit)
   ↓
6. batch_builder.py (creates API-ready batches with response schemas)
   ↓
7. batch_processor.py (makes OpenAI API calls, saves results)
```

### Core Modules

**main.py** - Entry point and orchestrator
- Interactive job selection menu
- Orchestrates the entire processing pipeline
- Progress tracking and reporting
- Uses unified logging throughout

**data_loader.py** - Data ingestion
- Loads from `Configuration_Files/` (required): API keys, job configs, question schemas
- Loads from `Context/Record_Context/` (required): Data to process (CSV/Excel)
- Loads from `Context/Question_Context/` (optional): Example Q&A pairs (CSV/Excel)
- Adds `source_file` column to all loaded DataFrames
- Handles empty files gracefully

**context_allocator.py** - Token budget management
- Uses `tiktoken` for accurate token counting per OpenAI model
- Adds `json` column (serialized row) and `token_count` column to each DataFrame
- Allocates available context between records and examples
- Strategy: Split 50/50, but if one side needs less, give remainder to the other
- Exports debug output to `testing/` directory

**record_context_chunker.py** - Record chunking
- Assigns `chunk_id` to records based on token limits
- Ensures no chunk exceeds `Record_Context_Token_Limit`
- Accounts for JSON array brackets `[]` in token count
- Exits with error if any single record exceeds limit

**question_context_chunker.py** - Example chunking
- Assigns `chunk_id` to question/answer examples
- Works with both `Question_Context_X` and `GPA_Questions` dataframes
- Ensures no chunk exceeds `Question_Context_Token_Limit`

**batch_builder.py** - Batch construction
- Creates cartesian product of record chunks × question chunks
- Builds OpenAI structured output JSON schema from `GPA_Questions.csv`
- Handles enum types by loading reference data
- Only uses `Question_Context_` files for batching (NOT `GPA_Questions`)
- Returns DataFrame with columns: `record_context_chunk_id`, `question_context_chunk_id`, `response_format`, `system_role`

**batch_processor.py** - API interaction
- Currently processes only the first batch (batch 0)
- Constructs messages with developer and user roles
- Filters out `source_file` from question context examples
- Logs complete API prompt to `Logs/api_debug/complete_prompt.txt`
- Logs raw API call to `Logs/api_debug/raw_api_call.json`
- Uses `client.beta.chat.completions.parse()` for structured outputs
- Calculates costs: $2.50/1M input tokens, $10.00/1M output tokens (hardcoded for o3-mini)
- Saves results to `Results/{job_name}_results.csv`

**error_logger.py** - Backward compatibility wrapper
- Thin wrapper around `unified_logger.log_error()`
- Maintains compatibility with older code

### Logging System

**unified_logger.py** - Centralized logging system
According to `LOGGING_UPGRADE_SUMMARY.md`, this provides:
- `get_logger(job_name=None)` - Get logger instance
- `LogLevel` enum: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `logger.log(level, message, source_file=None, function_name=None, to_file=True)`
- `logger.log_data(filename, data, format='csv', subfolder=None)`
- `logger.log_api_request(messages, response_format)`
- `logger.log_api_response(completion, cost_info)`
- `logger.log_api_cost(cost_details)`
- `logger.log_progress(description, current, total)`
- `logger.log_chunk_stats(context_name, num_chunks, chunk_details)`

Logs to:
- `Logs/error_logs.csv` - All errors with timestamps
- `Logs/activity_log.csv` - All activities
- `Logs/api_debug/` - API requests and responses

## Configuration Files

### GPA_Job_Configuration.csv
Defines available jobs with their parameters:

| Column | Description |
|--------|-------------|
| Job_Name | Unique identifier (e.g., "Body_Part_Lookup", "Policy_Analyzer") |
| Model | OpenAI model (e.g., "gpt-4o-2024-08-06") |
| Input_Context_Limit | Max input tokens (e.g., 128000) |
| Input_Context_Overhead | Reserved for system prompts (e.g., 8000) |
| Output_Context_Limit | Max output tokens (e.g., 16000) |
| Tool_Descriptions | Human-readable job description |
| Assistant_Role | System prompt defining AI's role |
| Apply_Relevance_Filter | Boolean flag (currently unused) |

**Available Jobs** (as of latest commit):
1. Body_Part_Lookup - Medical coding classification
2. Claim_Contact_Triage - Claims processing automation
3. Location_Lookup - Duplicate location detection
4. Taxonomy_Finder - Classification and categorization
5. French_Translator - English to French translation
6. Meeting_Transcriber - Transcript correction
7. Policy_Analyzer - Insurance policy layer parsing

### GPA_Questions.csv
Defines the output schema for each job:

| Column | Description |
|--------|-------------|
| Job_Name | Links to job configuration |
| Key | Field name in output JSON |
| Type | Data type: string, integer, boolean, number, enum |
| Description | Field description (sent to AI) |
| Max_Length | Maximum field length (documented but not enforced) |
| enum_file_name | For enum types, references file in Question_Context/ |

**Critical Pattern**: The system builds OpenAI structured output schemas from this file. Each job's questions become the properties in the JSON schema's "results" array.

### API_Keys.csv
Simple CSV with column `API_Key` containing the OpenAI API key.

## Directory Structure

```
.
├── Configuration_Files/       # System configuration (gitignored in production)
│   ├── API_Keys.csv          # OpenAI credentials
│   ├── GPA_Job_Configuration.csv
│   └── GPA_Questions.csv
│
├── Context/
│   ├── Record_Context/       # Input data to process (CSV/Excel)
│   └── Question_Context/     # Example Q&A pairs (optional)
│
├── Logs/                     # Runtime logs (gitignored)
│   ├── error_logs.csv
│   ├── activity_log.csv
│   ├── api_debug/           # API request/response debugging
│   └── {job_name}_batches.csv
│
├── Results/                  # Final output (gitignored)
│   └── {job_name}_results.csv
│
├── testing/                  # Debug output from context_allocator (gitignored)
│
├── Temp/                     # Legacy/deprecated (gitignored)
│
├── venv/                     # Python virtual environment (gitignored)
│
├── main.py                   # Entry point
├── data_loader.py
├── context_allocator.py
├── record_context_chunker.py
├── question_context_chunker.py
├── batch_builder.py
├── batch_processor.py
├── error_logger.py
├── unified_logger.py
│
├── prd.txt                   # Detailed product requirements
├── LOGGING_UPGRADE_SUMMARY.md
├── PRD_UPDATE_SUMMARY.md
└── .gitignore
```

## Key Patterns and Conventions

### DataFrame Dictionary Pattern
The system uses a `dataframes_dict` passed through all modules:
- Keys: `'GPA_Job_Configuration'`, `'GPA_Questions'`, `'API_Keys'`
- Keys: `'Record_Context_0'`, `'Record_Context_1'`, ... (numbered)
- Keys: `'Question_Context_0'`, `'Question_Context_1'`, ... (numbered)

Each DataFrame gets augmented with:
- `source_file` column (added by data_loader)
- `json` column (added by context_allocator)
- `token_count` column (added by context_allocator)
- `chunk_id` column (added by chunkers)

### Token Accounting
- Uses `tiktoken.encoding_for_model(model_name)` for accurate counting
- Accounts for JSON array brackets: `len(tokenizer.encode('[]'))`
- Accounts for commas between array elements: `len(tokenizer.encode(','))`
- Formula per record: `len(tokenizer.encode(json_str)) + len(tokenizer.encode(','))`

### Chunking Strategy
1. Calculate total tokens for all records and questions
2. Allocate budget between them (50/50 or weighted)
3. Iterate through records, assigning chunk_id
4. When adding next record would exceed limit, increment chunk_id
5. Reset token count for new chunk (starting with bracket tokens)

### API Message Structure
```python
messages = [
    {"role": "developer", "content": system_role},  # From Assistant_Role
    {"role": "user", "content": "Here are the records I want reviewed..."},
    {"role": "user", "content": str(record_context_json)},  # List of JSON objects
    {"role": "developer", "content": "Here are some examples..."},
    {"role": "developer", "content": str(question_context_json)}  # List of examples
]
```

### Structured Output Schema
The system uses OpenAI's structured outputs with JSON schema validation:
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": job_name,
        "description": tool_description,
        "schema": {
            "type": "object",
            "properties": {
                "results": {  # Always an array called "results"
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {...},  # Built from GPA_Questions.csv
                        "required": [...],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"]
        }
    }
}
```

### Enum Handling
For enum types in GPA_Questions.csv:
1. `enum_file_name` points to a file in Question_Context/
2. System loads that file and extracts unique values from first column
3. Adds "No Match" as an additional enum value
4. These become the `enum` array in the JSON schema

## Development Workflow

### Adding a New Job Type

1. **Update GPA_Job_Configuration.csv**
   - Add row with Job_Name, Model, token limits, descriptions, role

2. **Update GPA_Questions.csv**
   - Define output schema fields for the new job
   - Use appropriate types: string, integer, boolean, number, enum
   - For enum types, specify enum_file_name

3. **Prepare Input Data**
   - Place CSV or Excel file(s) in `Context/Record_Context/`
   - Optionally add example Q&A in `Context/Question_Context/`

4. **Run and Test**
   ```bash
   python3 main.py
   # Select new job from menu
   # Check Results/{job_name}_results.csv
   # Review Logs/ for errors
   ```

### Debugging Tips

1. **Check token allocation**: Look at console output showing token splits
2. **Review chunk stats**: Console shows number of chunks and token counts
3. **Inspect batches**: `Logs/{job_name}_batches.csv` shows all batch configurations
4. **Read API prompt**: `Logs/api_debug/complete_prompt.txt` shows exactly what was sent
5. **Check raw API call**: `Logs/api_debug/raw_api_call.json` has programmatic access
6. **Review testing output**: `testing/` directory has DataFrame exports with JSON

### Common Issues

**"Record exceeds token limit"**
- Single record too large for allocated context
- Increase Input_Context_Limit or reduce Input_Context_Overhead
- Or simplify/reduce the input data

**"No matching dataframe found"**
- Check that Context/Record_Context/ has data files
- Verify file formats are CSV or Excel (.xlsx)
- Check that files aren't empty

**Empty results**
- Review `Logs/api_debug/api_response_full.txt` for API errors
- Check if model refused to respond (look for 'refusal' in response)
- Verify response format schema matches what model returned

## Important Notes

### Git Status (from session start)
- Modified: `Configuration_Files/GPA_Job_Configuration.csv`
- Modified: `Configuration_Files/GPA_Questions.csv`
- Deleted: `Context/Question_Context/All_Other_GXO_Locations.csv`
- Deleted: `Context/Record_Context/XPO_Locations.csv`
- Untracked: `venv/` (properly gitignored)

### Hardcoded Values to Watch
- **Model**: `batch_processor.py` line 131, 142 hardcodes "o3-mini"
  - Should read from GPA_Job_Configuration instead
- **Cost Rates**: Lines 192-193 hardcode $2.50/1M input, $10.00/1M output
  - These are specific to o3-mini pricing
- **First Batch Only**: Line 36 only processes `batches_df.iloc[0]`
  - Full implementation would loop through all batches

### Security Notes
- API keys stored in plaintext CSV (Configuration_Files/ is gitignored)
- No authentication or access controls
- Logs may contain sensitive data (Logs/ is gitignored)

## Dependencies

From imports and code analysis:
```
openai          # OpenAI API client
pandas          # DataFrame manipulation
tiktoken        # Token counting for OpenAI models
sys             # System operations
os              # File/directory operations
csv             # CSV file reading
json            # JSON serialization
```

To install:
```bash
pip install openai pandas tiktoken openpyxl  # openpyxl for Excel support
```

## Architecture Decisions

### Why Chunking?
OpenAI models have token limits. Large datasets must be split into chunks that fit within these limits while preserving context coherence.

### Why Two-Phase Chunking?
Records and examples serve different purposes and have different priorities. The allocation strategy ensures optimal use of available context.

### Why Structured Outputs?
Ensures consistent, parseable results. The JSON schema validation catches malformed responses before they become processing errors.

### Why DataFrame Dictionary?
Allows flexible number of input files while maintaining clear naming conventions and easy access patterns throughout the pipeline.

## Future Development Notes

From prd.txt "Future Enhancements":
- Parallel batch processing (currently sequential, first batch only)
- Advanced retry strategies (no retry logic currently)
- Web-based interface (currently CLI only)
- Database integration (currently file-based)
- Multi-model support (currently per-job configuration)
- Streaming response handling (currently blocking)
- Real-time monitoring dashboard

## Version History

**Version 3.0** (Latest) - Enhanced Data Processing
- Added Policy_Analyzer job type
- Fixed pandas column assignment issues
- Added Excel file support
- Enhanced empty DataFrame handling
- Complete API prompt logging
- Fixed question context handling for multiple dataframe types

**Version 2.0** - Unified Logging System
- Implemented unified_logger.py (now missing)
- Replaced scattered logging with centralized system
- Added structured log formats (CSV, JSON)
- Automatic cost tracking
- Backward compatibility maintained

## Contact and Documentation

For detailed specifications, see:
- `prd.txt` - Complete product requirements (510 lines)
- `LOGGING_UPGRADE_SUMMARY.md` - Logging system documentation
- `PRD_UPDATE_SUMMARY.md` - Version 2.0 changelog

Last updated: 2025-09-29 (per prd.txt)
