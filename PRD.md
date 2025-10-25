PRODUCT REQUIREMENTS DOCUMENT (PRD)
=====================================

General Purpose Agent (GPA) - AI-Powered Batch Processing System
Version 4.0 - Parametric Configuration & Enhanced Logging Architecture
Date: 2025-10-15
Last Updated: 2025-10-15

====================
EXECUTIVE SUMMARY
====================

The General Purpose Agent (GPA) is a sophisticated batch processing system designed to leverage OpenAI's API for various data transformation and analysis tasks. The system processes large datasets through configurable jobs, intelligently chunking data to optimize token usage while maintaining context integrity. Version 4.0 introduces fully parametric configuration management, eliminating all hardcoded values, and implements a comprehensive logging architecture with structured CSV-based tracking.

====================
PRODUCT OVERVIEW
====================

PURPOSE:
The GPA system enables organizations to process large volumes of unstructured or semi-structured data through AI models, applying consistent transformations based on predefined job configurations and examples.

KEY CAPABILITIES:
- Batch processing of large datasets with automatic chunking
- Token-aware context allocation and management
- Fully parametric job configuration (models, temperature, pricing)
- Configurable job definitions with role-based prompting
- Example-based learning through question/answer contexts
- Comprehensive structured logging with CSV-based tracking
- Dynamic cost calculation based on model pricing database
- Per-session debug artifact organization

====================
SYSTEM ARCHITECTURE
====================

DIRECTORY STRUCTURE AND FILE DESCRIPTIONS:

.
├── __pycache__/                # Python bytecode cache (auto-generated)
├── .git/                       # Git version control repository
├── Configuration_Files/        # System configuration and job definitions
│   ├── API_Keys.csv           # OpenAI API credentials storage
│   ├── API_Pricing.csv        # NEW: Model pricing configuration (32 models)
│   ├── GPA_Job_Configuration.csv  # Job definitions with model, temperature, token limits
│   └── GPA_Questions.csv      # Output schema definitions for each job type
├── Context/                    # Context data storage
│   ├── Record_Context/        # Input data to process (CSV/Excel, required)
│   └── Question_Context/      # Example Q&A pairs (CSV/Excel, optional)
├── Logs/                       # NEW: Hierarchical logging output directory
│   ├── sessions/              # Detailed per-session logs
│   │   └── {job_name}_{timestamp}.log
│   ├── errors/                # Consolidated error tracking
│   │   └── errors.csv         # All ERROR/CRITICAL logs across sessions
│   ├── api_calls/             # Complete API call history
│   │   └── api_calls.csv      # Request+response+cost per call
│   ├── costs/                 # Cost tracking and analysis
│   │   └── costs.csv          # Spending summary over time
│   ├── batches/               # Batch configuration tracking
│   │   └── {job_name}_batches.csv
│   └── debug/                 # Per-session debug artifacts
│       └── {job_name}_{timestamp}/
│           ├── complete_prompt.txt      # Human-readable formatted prompt
│           ├── raw_api_call.json       # Complete API call data
│           ├── complete_api_payload.json  # Full API payload
│           └── api_response_full.txt   # Complete API response
├── Results/                    # Final processed output files
│   └── {job_name}_results.csv # Job-specific results
├── testing/                    # Debug output from context_allocator
├── venv/                       # Python virtual environment
│
├── batch_builder.py            # Constructs API-ready batches from chunks
├── batch_processor.py          # API integration with parametric config
├── context_allocator.py        # Token budget management with tiktoken
├── data_loader.py              # Data ingestion with API_Pricing support
├── error_logger.py             # Backward compatibility wrapper for unified logger
├── main.py                     # Entry point and orchestrator
├── question_context_chunker.py # Chunks examples by token limit
├── record_context_chunker.py   # Chunks record data within token limits
├── unified_logger.py           # NEW: Enhanced logging with CSV tracking
├── PRD.md                      # This Product Requirements Document
├── CLAUDE.md                   # Project instructions and quick reference
├── LOGGING_UPGRADE_SUMMARY.md  # Logging system documentation
├── PRD_UPDATE_SUMMARY.md       # Version 2.0 changelog
└── .gitignore                  # Git ignore configuration

CORE MODULES:

1. main.py (Entry Point & Orchestrator)
   - Interactive job selection menu
   - Workflow orchestration through pipeline
   - Progress tracking and reporting using unified logger
   - Batch CSV saving to Logs/batches/

2. data_loader.py (Data Management - Enhanced v4.0)
   - Loads CSV and Excel files from Configuration_Files/ and Context/ directories
   - NEW: Loads API_Pricing.csv as required configuration
   - Handles empty files and missing directories gracefully
   - Manages dataframe dictionaries with numbered keys
   - Adds source_file column to all DataFrames
   - Support for optional Question_Context directory

3. context_allocator.py (Token Management)
   - Uses tiktoken for accurate token counting per OpenAI model
   - Adds json and token_count columns to DataFrames
   - Allocates context budget between records and examples (50/50 with spillover)
   - Handles empty DataFrames and zero token scenarios
   - Exports debug output to testing/ directory

4. record_context_chunker.py (Data Chunking - Records)
   - Assigns chunk_id to records based on token limits
   - Ensures no chunk exceeds Record_Context_Token_Limit
   - Accounts for JSON array brackets in token count
   - Exits with error if single record exceeds limit

5. question_context_chunker.py (Data Chunking - Examples)
   - Chunks Question_Context_X and GPA_Questions dataframes
   - Assigns chunk_id based on Question_Context_Token_Limit
   - Maintains context coherence across examples

6. batch_builder.py (Batch Construction)
   - Creates cartesian product of record chunks × question chunks
   - Builds OpenAI structured output JSON schema from GPA_Questions.csv
   - Handles enum types by loading reference data
   - Only uses Question_Context_ files for batching (NOT GPA_Questions)
   - Returns DataFrame with: record_context_chunk_id, question_context_chunk_id, response_format, system_role

7. batch_processor.py (API Integration - Parametric v4.0)
   - NEW: Retrieves model name from GPA_Job_Configuration (no hardcoding)
   - NEW: Retrieves temperature from GPA_Job_Configuration (defaults to 1.0)
   - NEW: Dynamic pricing lookup from API_Pricing.csv
   - NEW: Handles numpy int64/float types for JSON serialization
   - NEW: Logs complete API calls to api_calls.csv with request+response
   - Constructs messages with developer and user roles
   - Filters out source_file from question context examples
   - Uses client.beta.chat.completions.parse() for structured outputs
   - Calculates costs dynamically based on model pricing
   - Marks costs as "unknown" when model not in pricing database
   - Saves results to Results/{job_name}_results.csv
   - Currently processes only first batch (batch 0)

8. unified_logger.py (Unified Logging System - Enhanced v4.0)
   - NEW: Hierarchical directory structure with 6 subdirectories
   - NEW: _append_to_csv() for thread-safe CSV writing
   - NEW: _log_to_errors_csv() for consolidated error tracking
   - NEW: log_api_call_complete() for unified API logging with request+response
   - NEW: _log_cost_summary() for cost tracking to costs.csv
   - NEW: Auto-appends ERROR/CRITICAL logs to errors.csv
   - Modified log_data() to route files to appropriate subdirectories
   - Fixed log_api_cost() to handle "unknown" pricing gracefully
   - Centralized logging for entire application
   - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - Automatic file organization per session
   - Progress tracking and chunk statistics
   - Console and file output options
   - Backward compatibility support

9. error_logger.py (Compatibility Wrapper)
   - Thin wrapper around unified_logger.log_error()
   - Maintains backward compatibility with older code

====================
DATA FLOW PIPELINE
====================

1. main.py (orchestrator)
   ↓
2. data_loader.py (loads CSV/Excel + API_Pricing.csv)
   ↓
3. context_allocator.py (calculates token budgets, adds JSON/token_count columns)
   ↓
4. record_context_chunker.py (chunks records by token limit)
   ↓
5. question_context_chunker.py (chunks examples by token limit)
   ↓
6. batch_builder.py (creates API-ready batches with response schemas)
   ↓
7. batch_processor.py (makes OpenAI API calls with parametric config, saves results)

====================
JOB CONFIGURATION
====================

SUPPORTED JOB TYPES:
1. Body_Part_Lookup - Medical coding classification
2. Claim_Contact_Triage - Claims processing automation
3. Location_Lookup - Duplicate detection and matching
4. Taxonomy_Finder - Classification and categorization
5. French_Translator - Language translation
6. Meeting_Transcriber - Transcript correction and enhancement
7. Policy_Analyzer - Insurance policy layer analysis and parsing

JOB PARAMETERS (GPA_Job_Configuration.csv):
- Job_Name (string): Unique identifier
- Model (string): OpenAI model (e.g., gpt-4o-2024-08-06, o3-mini) - PARAMETRIC
- Input_Context_Limit (integer): Maximum input tokens
- Input_Context_Overhead (integer): Reserved tokens for system prompts
- Output_Context_Limit (integer): Maximum output tokens
- Temperature (float): NEW v4.0 - Sampling temperature (0.0-2.0, default 1.0) - PARAMETRIC
- Tool_Descriptions (string): Job-specific instructions
- Assistant_Role (string): Role-based prompting context
- Apply_Relevance_Filter (string): Optional filtering flag

====================
API PRICING CONFIGURATION
====================

NEW IN VERSION 4.0 - API_Pricing.csv:

Centralized pricing database for 32 OpenAI models with per-million-token costs.

SCHEMA:
- Model (string): Model identifier (exact match required)
- Input_Cost_Per_Million (float): Cost per 1M input tokens in USD
- Cached_Input_Cost_Per_Million (float): Cost per 1M cached input tokens (empty if not supported)
- Output_Cost_Per_Million (float): Cost per 1M output tokens in USD

SUPPORTED MODELS (32 total):
- GPT-5 family: gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat-latest, gpt-5-codex, gpt-5-pro
- GPT-4.1 family: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- GPT-4o family: gpt-4o, gpt-4o-2024-05-13, gpt-4o-mini, gpt-4o-search-preview, gpt-4o-mini-search-preview
- Realtime models: gpt-realtime, gpt-realtime-mini, gpt-4o-realtime-preview, gpt-4o-mini-realtime-preview
- Audio models: gpt-audio, gpt-audio-mini, gpt-4o-audio-preview, gpt-4o-mini-audio-preview
- O-series: o1, o1-pro, o1-mini, o3, o3-pro, o3-mini, o3-deep-research, o4-mini, o4-mini-deep-research
- Codex: codex-mini-latest

PRICING EXAMPLES:
- o3-mini: $1.10 input, $0.55 cached, $4.40 output (per 1M tokens)
- gpt-4o-2024-08-06: $2.50 input, $1.25 cached, $10.00 output
- o1: $15.00 input, $7.50 cached, $60.00 output

UNKNOWN PRICING HANDLING:
When a model is not found in API_Pricing.csv:
- System logs WARNING
- Cost fields marked as "unknown" in all logs
- Token counts still tracked accurately
- Processing continues normally (no crash)

====================
KEY FEATURES
====================

1. PARAMETRIC CONFIGURATION (NEW v4.0):
   - All models configurable per job in GPA_Job_Configuration.csv
   - Temperature parameter configurable per job
   - Pricing looked up dynamically from API_Pricing.csv
   - No hardcoded values in code
   - Easy to add new models or update pricing

2. INTELLIGENT CHUNKING:
   - Token-aware data splitting using tiktoken
   - Context preservation across chunks
   - Optimal batch size calculation
   - Accounts for JSON array brackets and commas

3. EXAMPLE-BASED LEARNING:
   - Question/answer pair injection
   - Few-shot learning support
   - Context-aware example selection
   - Separate handling of schema vs examples

4. COMPREHENSIVE STRUCTURED LOGGING (NEW v4.0):
   - Hierarchical directory organization (6 subdirectories)
   - Consolidated error tracking across sessions (errors.csv)
   - Complete API call history with request+response (api_calls.csv)
   - Cost tracking and analysis (costs.csv)
   - Per-session debug artifacts organized by timestamp
   - Thread-safe CSV appending
   - Human-readable and machine-parsable formats

5. FLEXIBLE OUTPUT FORMATS:
   - CSV results with configurable schemas
   - JSON intermediate formats in debug logs
   - Structured CSV logs for analysis
   - Text-based human-readable logs

6. COST MANAGEMENT:
   - Dynamic cost calculation based on model pricing database
   - Token usage tracking (input and output)
   - Per-call cost logging in api_calls.csv
   - Cost summary tracking in costs.csv
   - Support for cached input token pricing
   - Graceful handling of unknown model pricing

====================
DATA SCHEMAS
====================

1. GPA_JOB_CONFIGURATION SCHEMA (Updated v4.0):
   - Job_Name (string): Unique identifier
   - Model (string): OpenAI model identifier
   - Input_Context_Limit (integer): Maximum input tokens
   - Input_Context_Overhead (integer): Reserved tokens for system prompts
   - Output_Context_Limit (integer): Maximum output tokens
   - Temperature (float): NEW - Sampling temperature (0.0-2.0)
   - Tool_Descriptions (string): Job-specific instructions
   - Assistant_Role (string): Role-based context for the AI
   - Apply_Relevance_Filter (string): Optional filtering flag

2. API_PRICING SCHEMA (NEW v4.0):
   - Model (string): Model identifier (e.g., "o3-mini", "gpt-4o-2024-08-06")
   - Input_Cost_Per_Million (float): Input token cost per 1M tokens
   - Cached_Input_Cost_Per_Million (float): Cached input cost (empty if N/A)
   - Output_Cost_Per_Million (float): Output token cost per 1M tokens

3. GPA_QUESTIONS SCHEMA (Output Definition):
   - Job_Name (string): Links to job configuration
   - Key (string): Field identifier in output
   - Type (string): Data type (string, integer, boolean, number, enum)
   - Description (string): Field description for AI guidance
   - Max_Length (integer): Maximum field length (documented, not enforced)
   - enum_file_name (string): Reference file in Question_Context/ for enum values

4. API_KEYS SCHEMA:
   - API_Key (string): OpenAI API key value

5. INPUT DATA SCHEMAS:
   Context/Record_Context/ folder (required):
   - CSV or Excel (.xlsx) files containing records to process
   - At least one non-empty file required
   - Automatically gets source_file, json, token_count, chunk_id columns added

   Context/Question_Context/ folder (optional):
   - CSV or Excel files with example Q&A pairs
   - Can be empty or missing entirely
   - Gets source_file, json, token_count, chunk_id columns added

6. LOGGING SCHEMAS (Enhanced v4.0):

   errors.csv (Consolidated Error Log):
   - timestamp (string): YYYY-MM-DD HH:MM:SS
   - job_name (string): Job name or "general"
   - level (string): ERROR or CRITICAL
   - source_file (string): Source file name (or empty)
   - function_name (string): Function name (or empty)
   - message (string): Error message text

   api_calls.csv (Complete API History):
   - timestamp (string): YYYY-MM-DD HH:MM:SS
   - job_name (string): Job name or "general"
   - model (string): Model identifier
   - temperature (float): Temperature parameter
   - batch_id (string): Batch identifier (e.g., "1-1")
   - request_json (string): Complete API request as JSON string
   - response_json (string): Complete API response as JSON string
   - input_tokens (integer): Actual input tokens used
   - output_tokens (integer): Actual output tokens used
   - input_cost (float|string): Input cost in USD or "unknown"
   - output_cost (float|string): Output cost in USD or "unknown"
   - total_cost (float|string): Total cost in USD or "unknown"
   - status (string): "success" or "failed"

   costs.csv (Cost Summary):
   - timestamp (string): YYYY-MM-DD HH:MM:SS
   - job_name (string): Job name or "general"
   - model (string): Model identifier
   - input_tokens (integer): Input tokens used
   - output_tokens (integer): Output tokens used
   - input_cost (float|string): Input cost or "unknown"
   - output_cost (float|string): Output cost or "unknown"
   - total_cost (float|string): Total cost or "unknown"

   Session Logs (sessions/{job_name}_{timestamp}.log):
   - All log levels mixed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - Timestamped entries with source file and function context
   - Human-readable text format

   Debug Artifacts (debug/{job_name}_{timestamp}/):
   - complete_prompt.txt: Formatted API prompt with all messages
   - raw_api_call.json: Complete API call data including batch info
   - complete_api_payload.json: Full API payload sent to OpenAI
   - api_response_full.txt: Complete API response object as string

====================
OUTPUT SPECIFICATIONS
====================

RESULTS FORMAT:
- CSV files named: {job_name}_results.csv
- Saved to Results/ directory
- Columns defined in GPA_Questions.csv
- Supports string, number, boolean, and enum types
- Configurable field lengths (documented in schema)

LOG OUTPUTS (Enhanced v4.0):

Directory Structure:
```
Logs/
├── sessions/              # Per-session detailed logs
├── errors/                # Consolidated error tracking (across sessions)
├── api_calls/             # Complete API history (across sessions)
├── costs/                 # Cost tracking summary (across sessions)
├── batches/               # Batch configurations (per job)
└── debug/                 # Per-session debug artifacts
    └── {job_name}_{timestamp}/
```

Benefits:
- Intuitive hierarchical organization
- Easy to find specific information by category
- Consolidated tracking across sessions (errors, API calls, costs)
- Per-session isolation for debug artifacts
- Human-readable and machine-parsable CSV formats
- Time-sortable with timestamp as first column

====================
TECHNICAL REQUIREMENTS
====================

DEPENDENCIES:
- Python 3.11+
- openai: OpenAI API client
- pandas: DataFrame manipulation
- tiktoken: Token counting for OpenAI models
- numpy: Numerical operations
- openpyxl: Excel file support
- Standard libraries: os, csv, json, datetime, sys, enum

API REQUIREMENTS:
- Valid OpenAI API key in Configuration_Files/API_Keys.csv
- Supported models as defined in API_Pricing.csv
- Internet connectivity for API calls

CONFIGURATION REQUIREMENTS (NEW v4.0):
- Configuration_Files/API_Keys.csv (required)
- Configuration_Files/API_Pricing.csv (required)
- Configuration_Files/GPA_Job_Configuration.csv (required)
- Configuration_Files/GPA_Questions.csv (required)
- Context/Record_Context/ with at least one CSV or Excel file (required)
- Context/Question_Context/ (optional)

====================
USAGE WORKFLOW
====================

1. Configure job in GPA_Job_Configuration.csv
   - Set Job_Name, Model, Temperature, token limits, Assistant_Role
2. Define output schema in GPA_Questions.csv
   - Specify fields with types and descriptions
   - Reference enum files if needed
3. Ensure model pricing exists in API_Pricing.csv
   - Add new models if necessary with pricing
4. Place source data in Context/Record_Context/
   - CSV or Excel format
5. Add example Q&A pairs in Context/Question_Context/ (optional)
6. Run: python3 main.py
7. Select job from interactive menu
8. Monitor progress through console output
9. Retrieve results from Results/{job_name}_results.csv
10. Review logs in Logs/ directory:
    - errors.csv for any errors
    - api_calls.csv for complete API history
    - costs.csv for spending tracking
    - debug/{job_name}_{timestamp}/ for detailed debugging

====================
CHANGELOG
====================

VERSION 4.0 (2025-10-15) - Parametric Configuration & Logging Architecture:

NEW CONFIGURATION FILES:
- API_Pricing.csv: Centralized pricing database for 32 OpenAI models
  * Input cost per million tokens
  * Cached input cost per million tokens
  * Output cost per million tokens

PARAMETRIC CONFIGURATION:
- Eliminated all hardcoded model names
  * Model now read from GPA_Job_Configuration.csv per job
- Eliminated all hardcoded pricing
  * Pricing dynamically looked up from API_Pricing.csv
- Eliminated hardcoded temperature
  * Temperature configurable per job in GPA_Job_Configuration.csv (defaults to 1.0)
- Unknown pricing handling
  * Gracefully marks costs as "unknown" when model not in pricing database
  * Logs warning but continues processing

ENHANCED LOGGING ARCHITECTURE:
- New hierarchical directory structure:
  * Logs/sessions/ - Per-session detailed logs
  * Logs/errors/ - Consolidated error tracking (across sessions)
  * Logs/api_calls/ - Complete API call history with request+response
  * Logs/costs/ - Cost summary tracking
  * Logs/batches/ - Batch configurations
  * Logs/debug/{job_name}_{timestamp}/ - Per-session debug artifacts

- New unified_logger.py methods:
  * _append_to_csv() - Thread-safe CSV appending
  * _log_to_errors_csv() - Automatic error consolidation
  * log_api_call_complete() - Unified API request+response logging
  * _log_cost_summary() - Cost tracking to costs.csv
  * Modified log() - Auto-appends ERROR/CRITICAL to errors.csv
  * Modified log_data() - Routes files to correct subdirectories
  * Fixed log_api_cost() - Handles "unknown" pricing gracefully

- New CSV logging files:
  * errors.csv - All ERROR/CRITICAL logs across all sessions
  * api_calls.csv - Complete API history with request JSON, response JSON, costs, status
  * costs.csv - Cost summary for easy analysis

BUG FIXES:
- Fixed numpy int64 JSON serialization errors in batch_processor.py
  * Convert numpy types to native Python types using .item()
  * Applies to temperature, chunk IDs, and batch_id
- Fixed log_api_cost() crash when pricing is "unknown"
  * Now logs token counts instead of attempting float formatting

MODIFIED FILES:
- data_loader.py: Added API_Pricing.csv as required configuration file
- batch_processor.py:
  * Retrieves model from job config (no hardcoding)
  * Retrieves temperature from job config with float conversion
  * Dynamic pricing lookup from API_Pricing.csv
  * Fixed numpy type serialization for JSON
  * Uses log_api_call_complete() for unified API logging
- unified_logger.py:
  * Complete rewrite of __init__() with new directory structure
  * Added CSV helper methods and API call logging
  * Enhanced with consolidated tracking across sessions
- main.py: Updated save_batches_to_csv() to use batches/ subfolder
- GPA_Job_Configuration.csv: Added Temperature column for all jobs

VERSION 3.0 (2025-09-29) - Enhanced Data Processing:

FIXES AND IMPROVEMENTS:
- Fixed pandas "Columns must be same length as key" error in context_allocator
- Fixed Excel file loading support in data_loader
- Fixed batch processor to handle GPA_Questions correctly
- Added comprehensive API prompt logging
- Enhanced empty DataFrame and missing file handling
- Fixed question context chunking for all dataframe types
- Added Policy_Analyzer job type support

NEW FEATURES:
- Complete prompt logging in human-readable format (complete_prompt.txt)
- Raw API call data logging with batch tracking (raw_api_call.json)
- Excel (.xlsx) file support throughout the system
- Graceful handling of empty or missing context files
- Enhanced debugging with formatted prompt output
- Support for Policy_Analyzer job: insurance policy layer analysis

VERSION 2.0 (2025-09-28) - Unified Logging System:

NEW FILES:
- unified_logger.py: Complete unified logging framework
- LOGGING_UPGRADE_SUMMARY.md: Implementation documentation

KEY IMPROVEMENTS:
- Centralized all logging in one module
- Removed duplicate logging functions from batch_processor
- Replaced scattered print statements with structured logging
- Added structured log formats (CSV, JSON)
- Automatic cost tracking and progress monitoring
- Maintained full backward compatibility

====================
KNOWN LIMITATIONS
====================

CURRENT LIMITATIONS:
1. Only processes first batch (batch 0) - subsequent batches ignored
2. No retry logic for failed API calls
3. No parallel batch processing (sequential only)
4. No automatic log rotation or archiving
5. Hardcoded user prompts still in batch_processor.py:
   - "Here are the records I want reviewed..."
   - "Here is some information you can use to help create your responses."
6. No caching support for prompt caching feature
7. No streaming response support

WORKAROUNDS:
- Unknown model pricing: System continues with "unknown" cost tracking
- Numpy type serialization: Automatic conversion using .item()
- Empty dataframes: Graceful handling throughout pipeline

====================
FUTURE ENHANCEMENTS
====================

HIGH PRIORITY:
- Process all batches instead of just batch 0
- Implement retry strategies for failed API calls
- Make user prompts parametric (configurable in job config)
- Add parallel batch processing for improved performance
- Implement log rotation and archiving

MEDIUM PRIORITY:
- Add support for cached input token pricing
- Implement streaming response handling
- Add real-time progress dashboard
- Database integration for logs and results
- Web-based interface for job management

LOW PRIORITY:
- Remote logging capabilities (e.g., CloudWatch, Datadog)
- Advanced cost optimization algorithms
- Multi-model comparison support
- Automated testing framework
- Performance profiling and optimization

====================
SECURITY CONSIDERATIONS
====================

CURRENT IMPLEMENTATION:
- API keys stored in plaintext CSV (Configuration_Files/ is gitignored)
- No authentication or access controls
- Logs may contain sensitive data (Logs/ is gitignored)
- No encryption at rest or in transit (beyond OpenAI's HTTPS)

RECOMMENDATIONS:
- Consider environment variable storage for API keys
- Implement access controls for configuration files
- Add data masking for sensitive fields in logs
- Consider encryption for stored results

====================
PERFORMANCE CHARACTERISTICS
====================

TOKEN PROCESSING:
- Chunking overhead: ~2-3% of total tokens for brackets/commas
- Context allocation: 50/50 split with spillover optimization
- Token counting: Uses tiktoken for accurate model-specific counts

API EFFICIENCY:
- Currently processes one batch per execution
- Batch size optimized to maximize context window usage
- No request batching or parallelization

COST OPTIMIZATION:
- Dynamic pricing lookup prevents cost calculation errors
- Token-aware chunking minimizes wasted context
- Example context optimized to fit within budget

====================
TESTING GUIDANCE
====================

MANUAL TESTING CHECKLIST:
1. New model configuration:
   - Add model to API_Pricing.csv
   - Update job in GPA_Job_Configuration.csv
   - Verify pricing logged correctly in costs.csv

2. Temperature configuration:
   - Set different temperatures per job
   - Verify temperature used in API calls (check raw_api_call.json)

3. Unknown model handling:
   - Configure job with model not in API_Pricing.csv
   - Verify "unknown" appears in api_calls.csv and costs.csv
   - Verify warning logged in errors.csv
   - Verify processing completes successfully

4. Logging verification:
   - Run job and verify all log directories created
   - Check errors.csv for consolidated errors
   - Check api_calls.csv for complete API history
   - Check costs.csv for cost tracking
   - Verify debug artifacts in debug/{job_name}_{timestamp}/

5. Error handling:
   - Introduce intentional error (e.g., missing file)
   - Verify error appears in errors.csv with full context
   - Verify error appears in session log

====================
CONCLUSION
====================

Version 4.0 of the General Purpose Agent represents a significant architectural evolution, introducing fully parametric configuration management and a comprehensive logging infrastructure. The elimination of all hardcoded values enables flexible multi-model deployment, while the hierarchical logging system provides unprecedented visibility into system operations, costs, and errors. The system maintains its robust batch processing capabilities while adding enterprise-grade observability and configuration management suitable for production deployments.

The modular architecture, intelligent chunking capabilities, parametric configuration, and structured logging make GPA suitable for enterprise-level data transformation needs while maintaining cost efficiency, processing accuracy, and operational transparency.
