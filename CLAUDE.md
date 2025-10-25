# General Purpose Agent 2.0 - AI-Powered Parallel Processing System

## Overview
General Purpose Agent 2.0 is a high-performance batch processing system that leverages OpenAI's API for data transformation and analysis tasks. Unlike version 1.0's sequential chunked approach, v2.0 processes records in parallel, sending individual records with complete question context to achieve dramatically faster processing speeds.

**Version**: 2.0
**Language**: Python 3.11+
**Primary Framework**: OpenAI AsyncAPI with structured outputs
**Concurrency Model**: Async/await with semaphore-controlled parallelism (50 concurrent requests)

## What's New in Version 2.0

### Architectural Revolution
- **Parallel Processing**: Up to 50 simultaneous API requests
- **No Chunking**: Each record processed individually
- **Complete Context**: All question examples sent with every request
- **Real-time Results**: CSV appending as requests complete
- **Graceful Shutdown**: Ctrl+C handling for clean termination
- **Enhanced Progress**: Live tracking of completed/failed/in-flight requests

### Performance Improvements
- **50x faster** for large datasets (compared to v1.0 sequential processing)
- Real-time cost tracking across all parallel requests
- Immediate feedback on processing status

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key configured in `Configuration_Files/API_Keys.csv`

### Installation
```bash
# Clone the repository
git clone https://github.com/andyvonnegut/general-purpose-agent-2.0.git
cd general-purpose-agent-2.0

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main application
python3 main.py

# Follow the interactive prompts to select a job
# Press Ctrl+C to gracefully stop processing at any time
```

## System Architecture

### Data Flow Pipeline (Version 2.0)
```
1. main.py (async orchestrator)
   ↓
2. data_loader.py (loads CSV/Excel files)
   ↓
3. context_allocator.py (validates single record + all examples fit token limit)
   ↓
4. batch_builder.py (creates one batch per record with all question context)
   ↓
5. batch_processor.py (async parallel processor - 50 concurrent requests)
   ↓
6. Real-time CSV writing (thread-safe append as each request completes)
```

### Core Modules

**main.py** - Async entry point and orchestrator
- Interactive job selection menu
- Orchestrates the entire processing pipeline
- Uses `asyncio.run()` to execute parallel batch processing
- Progress tracking and reporting
- Unified logging throughout

**data_loader.py** - Data ingestion (unchanged from v1.0)
- Loads from `Configuration_Files/` (required): API keys, job configs, question schemas
- Loads from `Context/Record_Context/` (required): Data to process (CSV/Excel)
- Loads from `Context/Question_Context/` (optional): Example Q&A pairs (CSV/Excel)
- Adds `source_file` column to all loaded DataFrames
- Handles empty files gracefully

**context_allocator.py** - Token validation (v2.0 redesign)
- Uses `tiktoken` for accurate token counting per OpenAI model
- Adds `json` column (serialized row) and `token_count` column to each DataFrame
- Calculates total question context tokens (sent with EVERY record)
- Validates that each individual record + all question context fits within token limit
- Exits with detailed error if any record exceeds limit
- No chunking logic (removed in v2.0)

**batch_builder.py** - Batch construction (v2.0 redesign)
- Creates ONE batch per individual record
- Includes ALL question context in every batch
- Builds OpenAI structured output JSON schema from `GPA_Questions.csv`
- Handles enum types by loading complete reference data
- Returns DataFrame with columns: `batch_id`, `record_data`, `record_json`, `source_file`, `question_context`, `response_format`, `system_role`

**batch_processor.py** - Parallel API processing (v2.0 complete rewrite)
- **Async/Await**: Uses `AsyncOpenAI` client for non-blocking requests
- **Semaphore Control**: `asyncio.Semaphore(50)` limits concurrent requests
- **Thread-Safe CSV**: Uses `asyncio.Lock` for safe concurrent writes
- **Real-time Progress**: Shows completed/failed/in-flight counts during processing
- **Graceful Shutdown**: Signal handler for Ctrl+C allows in-flight requests to complete
- **Cost Tracking**: Accumulates costs across all parallel requests
- **Error Handling**: Per-request exception handling with detailed logging
- Writes results to `Results/{job_name}_results.csv` as they complete

**unified_logger.py** - Centralized logging system (unchanged from v1.0)
- `get_logger(job_name=None)` - Get logger instance
- `LogLevel` enum: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `logger.log(level, message, source_file=None, function_name=None, to_file=True)`
- `logger.log_data(filename, data, format='csv', subfolder=None)`
- Compatible with async/concurrent operations

**error_logger.py** - Backward compatibility wrapper (unchanged from v1.0)
- Thin wrapper around `unified_logger.log_error()`

## Configuration Files

### GPA_Job_Configuration.csv
Defines available jobs with their parameters:

| Column | Description |
|--------|-------------|
| Job_Name | Unique identifier (e.g., "Body_Part_Lookup", "Policy_Analyzer") |
| Model | OpenAI model (e.g., "gpt-4o-2024-08-06", "o3-mini") |
| Input_Context_Limit | Max input tokens (e.g., 128000) |
| Input_Context_Overhead | Reserved for system prompts (e.g., 8000) |
| Output_Context_Limit | Max output tokens (e.g., 16000) |
| Tool_Descriptions | Human-readable job description |
| Assistant_Role | System prompt defining AI's role |
| Temperature | Model temperature (0.0-2.0, defaults to 1.0) |

### API_Pricing.csv
Defines pricing for different models:

| Column | Description |
|--------|-------------|
| Model | Model identifier matching GPA_Job_Configuration |
| Input_Cost_Per_Million | Cost per million input tokens |
| Output_Cost_Per_Million | Cost per million output tokens |

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
├── Configuration_Files/       # System configuration
│   ├── API_Keys.csv          # OpenAI credentials (gitignored)
│   ├── API_Pricing.csv       # Model pricing information
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
│   └── batches/             # Batch configuration logs
│
├── Results/                  # Final output (gitignored)
│   └── {job_name}_results.csv
│
├── testing/                  # Debug output from context_allocator (gitignored)
│
├── venv/                     # Python virtual environment (gitignored)
│
├── main.py                   # Entry point (async orchestrator)
├── data_loader.py           # Data ingestion
├── context_allocator.py     # Token validation
├── batch_builder.py         # Batch construction
├── batch_processor.py       # Async parallel processor
├── error_logger.py          # Error logging wrapper
├── unified_logger.py        # Centralized logging
│
├── requirements.txt         # Python dependencies
├── README.md               # Quick start guide
├── CLAUDE.md              # This file - technical documentation
├── PRD.md                 # Product requirements
└── .gitignore            # Git ignore rules
```

## Key Patterns and Conventions

### DataFrame Dictionary Pattern
The system uses a `dataframes_dict` passed through all modules:
- Keys: `'GPA_Job_Configuration'`, `'GPA_Questions'`, `'API_Keys'`, `'API_Pricing'`
- Keys: `'Record_Context_0'`, `'Record_Context_1'`, ... (numbered)
- Keys: `'Question_Context_0'`, `'Question_Context_1'`, ... (numbered)

Each DataFrame gets augmented with:
- `source_file` column (added by data_loader)
- `json` column (added by context_allocator)
- `token_count` column (added by context_allocator)

**Note**: In v2.0, `chunk_id` columns are NOT added (no chunking).

### Token Accounting
- Uses `tiktoken.encoding_for_model(model_name)` for accurate counting
- Accounts for JSON array brackets: `len(tokenizer.encode('[]'))`
- Accounts for commas between array elements: `len(tokenizer.encode(','))`
- Formula per record: `len(tokenizer.encode(json_str)) + len(tokenizer.encode(','))`

### Parallel Processing Strategy (v2.0)
1. Load and validate all data
2. Calculate total question context tokens (sent with EVERY record)
3. Validate each record + all question context fits in available context
4. Build one batch per record
5. Create async tasks for all batches
6. Use semaphore to limit to 50 concurrent requests
7. Append results to CSV as each request completes
8. Track progress, costs, and errors in real-time

### API Message Structure
```python
# With question context
messages = [
    {"role": "developer", "content": system_role},
    {"role": "user", "content": "Here is the record I want reviewed..."},
    {"role": "user", "content": "[{single_record_json}]"},
    {"role": "developer", "content": "Here is information you can use..."},
    {"role": "developer", "content": str(all_question_context)}
]

# Without question context
messages = [
    {"role": "developer", "content": system_role},
    {"role": "user", "content": "Here is the record I want reviewed..."},
    {"role": "user", "content": "[{single_record_json}]"}
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
2. System loads that file and extracts ALL unique values from first column (no chunking)
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
   # Watch real-time progress
   # Press Ctrl+C to stop if needed
   # Check Results/{job_name}_results.csv
   # Review Logs/ for errors
   ```

### Debugging Tips

1. **Check token validation**: Console output shows if records exceed limits
2. **Monitor progress**: Real-time updates show completed/failed/in-flight counts
3. **Inspect batches**: `Logs/batches/{job_name}_batches.csv` shows batch configurations
4. **Review testing output**: `testing/` directory has DataFrame exports with JSON
5. **Check logs**: `Logs/activity_log.csv` and `Logs/error_logs.csv` for detailed information

### Common Issues

**"Record exceeds token limit"**
- Single record + all question context too large for allocated context
- Solution: Increase Input_Context_Limit or reduce Input_Context_Overhead
- Or reduce the size/number of question context examples
- Or simplify/reduce the input data records

**"No matching dataframe found"**
- Check that Context/Record_Context/ has data files
- Verify file formats are CSV or Excel (.xlsx)
- Check that files aren't empty

**Slow processing or hanging**
- Check network connectivity
- Verify API key is valid
- Review error logs for API rate limiting or errors
- Try reducing parallelism by modifying semaphore limit in batch_processor.py

## Parallel Processing Details

### Concurrency Control
```python
# batch_processor.py
semaphore = asyncio.Semaphore(50)  # Max 50 concurrent requests

async def process_single_batch(batch_row):
    async with semaphore:
        # Only 50 tasks can execute this block simultaneously
        # Others wait until a slot becomes available
        ...
```

### Thread-Safe CSV Writing
```python
csv_lock = asyncio.Lock()

async def write_result(result_row):
    async with csv_lock:
        # Only one task can write at a time
        # Ensures CSV file integrity
        results_df = pd.DataFrame([result_row])
        results_df.to_csv(output_file, mode='a', header=False)
```

### Graceful Shutdown
```python
# Press Ctrl+C during processing
shutdown_requested = True  # Signal set
# Currently executing requests complete
# Pending requests are cancelled
# Final summary is displayed
```

### Progress Tracking
```
Processing: 45/100 complete, 2 failed, 50 in-flight
Processing: 95/100 complete, 2 failed, 3 in-flight
Processing: 100/100 complete, 2 failed, 0 in-flight

================================================================================
PROCESSING COMPLETE
================================================================================
Total records: 100
Successfully processed: 98
Failed: 2
Duration: 45.23 seconds
Average time per record: 0.46 seconds

Cost Summary:
  Input tokens: 1,234,567
  Output tokens: 234,567
  Total cost: $15.4321

Results saved to: Results/Body_Part_Lookup_results.csv
================================================================================
```

## Version Comparison

| Feature | Version 1.0 | Version 2.0 |
|---------|-------------|-------------|
| Processing Model | Sequential, chunked | Parallel, per-record |
| Concurrency | 1 request at a time | 50 simultaneous requests |
| Record Grouping | Multiple records per batch | 1 record per request |
| Question Context | Chunked and rotated | All examples every time |
| Speed (100 records) | ~200 seconds | ~4 seconds |
| Progress Tracking | End-of-job only | Real-time updates |
| Shutdown | Immediate termination | Graceful completion |
| Results Writing | Single write at end | Real-time appending |
| Token Management | Complex chunking logic | Simple validation |

## Security Notes
- API keys stored in plaintext CSV (Configuration_Files/ is gitignored)
- No authentication or access controls
- Logs may contain sensitive data (Logs/ is gitignored)
- Results may contain sensitive data (Results/ is gitignored)

## Dependencies

From `requirements.txt`:
```
openai>=1.0.0          # OpenAI API client with async support
pandas>=2.0.0          # DataFrame manipulation
tiktoken>=0.5.0        # Token counting for OpenAI models
openpyxl>=3.1.0        # Excel file support
```

Standard library (included with Python 3.11+):
- asyncio, csv, json, os, sys, signal, datetime

## Future Enhancements

Potential improvements for future versions:
- Configurable concurrency limit (currently hardcoded to 50)
- Retry logic for failed requests
- Resume capability for interrupted jobs
- Batch processing across multiple jobs simultaneously
- Web-based interface for job management
- Database integration for results storage
- Real-time cost monitoring dashboard
- Custom rate limiting per API key
- Support for streaming responses

## Troubleshooting

### ImportError: No module named 'openai'
```bash
pip install -r requirements.txt
```

### API Rate Limiting
If you see rate limit errors, reduce the semaphore limit in `batch_processor.py`:
```python
semaphore = asyncio.Semaphore(10)  # Reduce from 50 to 10
```

### Out of Memory Errors
For very large datasets, consider processing in smaller batches by splitting input files.

## Contributing

Repository: https://github.com/andyvonnegut/general-purpose-agent-2.0

## Version History

**Version 2.0** (2025-10-25) - Parallel Processing Revolution
- Complete architectural rewrite for async/parallel processing
- 50 concurrent requests with semaphore control
- No chunking - individual record processing
- All question context sent with every request
- Real-time CSV writing and progress tracking
- Graceful shutdown handling
- 50x performance improvement

**Version 1.0** (Previous) - Sequential Chunked Processing
- Sequential batch processing
- Chunking logic for records and questions
- Single-threaded execution
- Batch results writing

## Contact and Documentation

For detailed specifications, see:
- `README.md` - Quick start and overview
- `PRD.md` - Complete product requirements
- This file (CLAUDE.md) - Technical documentation

Last updated: 2025-10-25
