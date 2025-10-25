# General Purpose Agent 2.0

AI-Powered Parallel Batch Processing System

## What's New in 2.0

**Major Architectural Change: Parallel Processing**

Version 2.0 represents a fundamental shift from chunked batch processing to parallel, per-record processing:

- **1 Record = 1 API Request**: Each record is processed individually rather than in chunks
- **50 Concurrent Requests**: Process up to 50 records simultaneously using async/await
- **All Examples Included**: Since we're only sending one record at a time, we have plenty of token budget to include all question context examples with every request
- **Real-time Results**: Results are written to CSV as requests complete (thread-safe)
- **Better Scalability**: Dramatically faster processing for large datasets

## Version 1.0 vs 2.0 Comparison

| Feature | Version 1.0 | Version 2.0 |
|---------|-------------|-------------|
| Processing Model | Sequential chunks | Parallel per-record |
| Concurrency | None (1 at a time) | 50 simultaneous requests |
| Record Grouping | Multiple records per batch | 1 record per request |
| Question Context | Chunked and rotated | All examples every time |
| Speed | Slow for large datasets | 50x faster |
| Code Complexity | Chunking logic required | Simplified, async-based |

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
pip install openai pandas tiktoken openpyxl
```

### Running the Application

```bash
python3 main.py
```

Follow the interactive prompts to select and run a job.

## Architecture

### Data Flow (Version 2.0)

```
1. main.py (orchestrator)
   ↓
2. data_loader.py (loads CSV/Excel files)
   ↓
3. context_allocator.py (validates single record + all examples fit token limit)
   ↓
4. batch_builder.py (creates one batch per record with all question context)
   ↓
5. batch_processor.py (async processor with 50 concurrent requests)
   ↓
6. Real-time CSV writing (thread-safe append as each request completes)
```

### Key Components

- **main.py**: Entry point and orchestrator
- **data_loader.py**: Loads configuration and data files
- **context_allocator.py**: Validates token limits for single-record processing
- **batch_builder.py**: Creates individual batches (1 per record)
- **batch_processor.py**: Async parallel processor with semaphore control
- **unified_logger.py**: Centralized logging for concurrent operations
- **error_logger.py**: Backward compatibility wrapper

## Configuration

Place your configuration files in `Configuration_Files/`:

- **API_Keys.csv**: Your OpenAI API key
- **GPA_Job_Configuration.csv**: Job definitions and token limits
- **GPA_Questions.csv**: Output schema definitions

Place your input data in:
- **Context/Record_Context/**: Records to process (CSV/Excel)
- **Context/Question_Context/**: Example Q&A pairs (CSV/Excel, optional)

## Available Jobs

1. **Body_Part_Lookup**: Medical coding classification
2. **Claim_Contact_Triage**: Claims processing automation
3. **Location_Lookup**: Duplicate location detection
4. **Taxonomy_Finder**: Classification and categorization
5. **French_Translator**: English to French translation
6. **Meeting_Transcriber**: Transcript correction
7. **Policy_Analyzer**: Insurance policy layer parsing

## Performance

With 50 concurrent requests:
- **Small jobs** (< 100 records): Completes in seconds
- **Medium jobs** (100-1000 records): Completes in minutes
- **Large jobs** (1000+ records): Scales linearly

## Documentation

- **CLAUDE.md**: Complete technical documentation
- **PRD.md**: Product requirements and specifications

## License

MIT License

## Support

For issues, please visit: https://github.com/andyvonnegut/general-purpose-agent-2.0/issues
