import json
import pandas as pd
from unified_logger import get_logger, LogLevel

def _pack_chunks(question_records, token_counts, budget):
    """Greedily pack question-context rows into chunks whose token sums stay at
    or under `budget`. Returns a list of chunks (each a list of record dicts).

    When `budget` is falsy or token counts are unavailable, the whole context is
    returned as a single chunk (today's behaviour). A row larger than the budget
    is placed in its own chunk rather than dropped (the allocator already
    guarantees a single row fits, this is just defensive).
    """
    if not question_records:
        # One empty chunk -> the single-request, no-context path downstream.
        return [[]]
    if not budget or any(tc is None for tc in token_counts):
        return [list(question_records)]

    chunks = []
    current, current_tokens = [], 0
    for rec, tokens in zip(question_records, token_counts):
        tokens = int(tokens)
        if current and current_tokens + tokens > budget:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(rec)
        current_tokens += tokens
    if current:
        chunks.append(current)
    return chunks


def build_batches(dataframes_dict, selected_job_name, allocation=None):
    """
    Version 2.1: Builds one batch per individual record. Each batch carries the
    question context as a list of chunks. When the full question context fits a
    single request (the common case) there is exactly one chunk and behaviour is
    identical to 2.0; when the allocator reports the context must be split, the
    rows are packed into multiple chunks so the record can be run against each
    chunk and the answers stitched downstream.

    Args:
        dataframes_dict (dict): Contains all preprocessed data including records and question context.
        selected_job_name (str): The name of the job currently being processed.
        allocation (dict): The allocate_context() result; its Context_Budget_Per_Chunk
            drives chunk packing. If omitted, the whole context is one chunk.

    Returns:
        pd.DataFrame: A DataFrame of batches (one per record) ready to be processed.
    """
    logger = get_logger(selected_job_name)

    try:
        # Retrieve necessary DataFrames from the dictionary
        gpa_job_config_df = dataframes_dict.get('GPA_Job_Configuration')
        if gpa_job_config_df is None:
            logger.log(LogLevel.ERROR, "GPA_Job_Configuration file is missing from dataframes_dict.", source_file="batch_builder.py")
            return pd.DataFrame()

        # Retrieve the selected job configuration
        selected_job = gpa_job_config_df[gpa_job_config_df['Job_Name'] == selected_job_name]
        if selected_job.empty:
            logger.log(LogLevel.ERROR, f"No configuration found for job: {selected_job_name}", source_file="batch_builder.py")
            return pd.DataFrame()

        # Get relevant fields from the job configuration
        assistant_role = selected_job['Assistant_Role'].values[0]
        system_role = json.dumps({"role": "system", "content": assistant_role})

        # Get Record Context DataFrames
        record_context_dfs = [df for df_name, df in dataframes_dict.items() if df_name.startswith('Record_Context_')]

        if not record_context_dfs:
            logger.log(LogLevel.ERROR, "No record context dataframes found.", source_file="batch_builder.py")
            return pd.DataFrame()

        # Collect all question context (we'll send ALL of it with each record),
        # keeping each row's pre-computed token count so it can be packed into
        # budget-sized chunks.
        all_question_context = []
        all_question_token_counts = []
        for df_name, df in dataframes_dict.items():
            if df_name.startswith('Question_Context_') and not df.empty:
                # Filter out metadata columns (source_file, json, token_count)
                question_cols = [col for col in df.columns if col not in ['source_file', 'json', 'token_count']]
                question_records = df[question_cols].to_dict('records')
                all_question_context.extend(question_records)
                if 'token_count' in df.columns:
                    all_question_token_counts.extend(df['token_count'].tolist())
                else:
                    all_question_token_counts.extend([None] * len(question_records))

        # Pack the question context into chunks. One chunk == the full context in a
        # single request (unchanged 2.0 behaviour); more than one means it will be
        # split across requests and stitched.
        budget = (allocation or {}).get('Context_Budget_Per_Chunk')
        question_context_chunks = _pack_chunks(all_question_context, all_question_token_counts, budget)

        # Create response format (same for all batches since the schema is fixed)
        response_format = create_response_format(dataframes_dict, selected_job_name)

        # Build batches - one per record
        batch_rows = []
        batch_id = 0

        for rec_df in record_context_dfs:
            for idx, row in rec_df.iterrows():
                # Extract the record data (excluding metadata columns)
                record_cols = [col for col in rec_df.columns if col not in ['source_file', 'json', 'token_count']]
                record_data = row[record_cols].to_dict()

                # Create batch entry
                batch_row = {
                    "batch_id": batch_id,
                    "record_data": record_data,  # The actual record as a dict
                    "record_json": row['json'],  # Pre-computed JSON string
                    "source_file": row.get('source_file', 'unknown'),
                    "question_context_chunks": question_context_chunks,  # list[list[dict]]
                    "response_format": response_format,
                    "system_role": system_role
                }
                batch_rows.append(batch_row)
                batch_id += 1

        # Convert to DataFrame
        batches_df = pd.DataFrame(batch_rows)

        num_chunks = len(question_context_chunks)
        logger.log(LogLevel.INFO,
                   f"Built {len(batches_df)} batches (one per record), "
                   f"{num_chunks} question-context chunk(s) per record")
        print(f"Built {len(batches_df)} batches (one per record), {num_chunks} context chunk(s) each")

        return batches_df

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error in batch builder: {str(e)}", source_file="batch_builder.py")
        return pd.DataFrame()

def create_response_format(dataframes_dict, selected_job_name):
    """
    Creates a structured response format (JSON Schema) for the job.
    Version 2.0: Simplified since all question context is always included.

    Args:
        dataframes_dict (dict): Contains all data including job and question configurations.
        selected_job_name (str): The name of the job currently being processed.

    Returns:
        dict: The structured JSON schema response format.
    """
    logger = get_logger(selected_job_name)

    try:
        # Retrieve the GPA Job Configuration DataFrame
        gpa_job_config_df = dataframes_dict.get('GPA_Job_Configuration')
        if gpa_job_config_df is None:
            logger.log(LogLevel.ERROR, "GPA_Job_Configuration file is missing from dataframes_dict.", source_file="batch_builder.py")
            return {}

        # Get the tool description for the selected job
        selected_job = gpa_job_config_df[gpa_job_config_df['Job_Name'] == selected_job_name]
        if selected_job.empty:
            logger.log(LogLevel.ERROR, f"No configuration found for job: {selected_job_name}", source_file="batch_builder.py")
            return {}

        tool_description = selected_job['Tool_Descriptions'].values[0]

        # Retrieve the GPA Questions DataFrame
        gpa_questions_df = dataframes_dict.get('GPA_Questions')
        if gpa_questions_df is None:
            logger.log(LogLevel.ERROR, "GPA_Questions file is missing from dataframes_dict.", source_file="batch_builder.py")
            return {}

        # Filter questions based on the job name
        questions_for_job = gpa_questions_df[gpa_questions_df['Job_Name'] == selected_job_name]

        # Build the schema for the individual items inside the 'results' array
        properties = {}
        for _, question_row in questions_for_job.iterrows():
            key = question_row['Key']
            description = question_row['Description']
            question_type = question_row['Type']
            enum_file_name = question_row.get('enum_file_name')

            # Handle enum type if applicable
            if question_type == 'enum' and enum_file_name:
                enum_values = get_enum_values(enum_file_name, dataframes_dict)
                properties[key] = {
                    "type": 'string',
                    "description": description,
                    "enum": enum_values
                }
            else:
                properties[key] = {
                    "type": question_type,
                    "description": description
                }

        # Define the 'results' array schema
        results_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False
            }
        }

        # Return the formatted JSON schema. strict=True turns OpenAI's structured
        # outputs from a hint into a compiled constrained grammar, so the model
        # is forced to emit {"results": [...]} at the root instead of echoing the
        # schema shape (observed on gpt-5.5: {"properties": {"results": [...]}}).
        # The current schema already meets strict's requirements
        # (additionalProperties: False everywhere, every property in required).
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": selected_job_name,
                "description": tool_description,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": results_schema
                    },
                    "additionalProperties": False,
                    "required": ["results"]
                }
            }
        }

        return response_format

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error creating response format: {str(e)}", source_file="batch_builder.py")
        return {}

def get_enum_values(enum_file_name, dataframes_dict):
    """
    Version 2.0: Retrieves ALL enumerated values from a file (no chunking).

    Args:
        enum_file_name (str): The file name containing enumerated values.
        dataframes_dict (dict): Contains all data including enumerated values.

    Returns:
        list: A list of enumerated values.
    """
    logger = get_logger()

    try:
        # Find the relevant question context dataframe by matching the source_file to enum_file_name
        question_context_df = None
        for df_name, df in dataframes_dict.items():
            if df_name.startswith('Question_Context_') and enum_file_name in df['source_file'].unique():
                question_context_df = df
                break

        if question_context_df is None:
            logger.log(LogLevel.ERROR, f"No matching dataframe found for enum file: {enum_file_name}", source_file="batch_builder.py")
            return []

        # Get ALL unique values from the first column (no chunk filtering)
        enum_values = question_context_df.iloc[:, 0].unique().tolist()

        # Add "No Match" to the list of enum values
        enum_values.append("No Match")
        return enum_values

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error fetching enum values: {str(e)}", source_file="batch_builder.py")
        return []
