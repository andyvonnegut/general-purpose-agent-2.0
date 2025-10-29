import json
import pandas as pd
from unified_logger import get_logger, LogLevel

def build_batches(dataframes_dict, selected_job_name):
    """
    Version 2.0: Builds one batch per individual record with all question context.
    No chunking - each record is processed independently with complete question context.

    Args:
        dataframes_dict (dict): Contains all preprocessed data including records and question context.
        selected_job_name (str): The name of the job currently being processed.

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

        # Collect all question context (we'll send ALL of it with each record)
        all_question_context = []
        for df_name, df in dataframes_dict.items():
            if df_name.startswith('Question_Context_') and not df.empty:
                # Filter out metadata columns (source_file, json, token_count)
                question_cols = [col for col in df.columns if col not in ['source_file', 'json', 'token_count']]
                question_records = df[question_cols].to_dict('records')
                all_question_context.extend(question_records)

        # Create response format (same for all batches since we send all question context)
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
                    "question_context": all_question_context,  # ALL question context
                    "response_format": response_format,
                    "system_role": system_role
                }
                batch_rows.append(batch_row)
                batch_id += 1

        # Convert to DataFrame
        batches_df = pd.DataFrame(batch_rows)

        logger.log(LogLevel.INFO, f"Built {len(batches_df)} batches (one per record)")
        print(f"Built {len(batches_df)} batches (one per record)")

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

        # Return the formatted JSON schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": selected_job_name,
                "description": tool_description,
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
