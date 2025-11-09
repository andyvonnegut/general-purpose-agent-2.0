import pandas as pd
import tiktoken
from unified_logger import get_logger, LogLevel
import json
import os

def allocate_context(dataframes_dict, selected_job_name):
    """
    Version 2.0: Validates that a single record + all question context fits within token limits.
    No chunking is performed - each record will be sent individually with all examples.

    Args:
        dataframes_dict (dict): Dictionary containing the loaded DataFrames for records, questions, and job config.
        selected_job_name (str): The name of the selected job to fetch model details.

    Returns:
        context_allocation (dict): Contains token counts and validation status.
    """
    logger = get_logger(selected_job_name)

    try:
        # Retrieve job configuration
        gpa_job_configuration = dataframes_dict.get('GPA_Job_Configuration')
        if gpa_job_configuration is None:
            logger.log(LogLevel.ERROR, "gpa_job_configuration file is missing.", source_file="context_allocator.py")
            return {}

        # Get model parameters for the selected job
        selected_job = gpa_job_configuration[gpa_job_configuration['Job_Name'] == selected_job_name]
        if selected_job.empty:
            logger.log(LogLevel.ERROR, f"No configuration found for job: {selected_job_name}", source_file="context_allocator.py")
            return {}

        selected_model = selected_job['Model'].values[0]
        input_context_limit = int(selected_job['Input_Context_Limit'].values[0])
        overhead_tokens = int(selected_job['Input_Context_Overhead'].values[0])
        available_context = input_context_limit - overhead_tokens

        if available_context <= 0:
            logger.log(LogLevel.ERROR, "Available context is less than or equal to 0.", source_file="context_allocator.py")
            return {}

        # Initialize tokenizer for the selected model
        try:
            tokenizer = tiktoken.encoding_for_model(selected_model)
        except KeyError:
            # If the model is not recognized by tiktoken, use o200k_base encoding (GPT-4o compatible)
            logger.log(LogLevel.WARNING, f"Model '{selected_model}' not recognized by tiktoken. Using o200k_base encoding as fallback.", source_file="context_allocator.py")
            tokenizer = tiktoken.get_encoding("o200k_base")

        # Calculate total question context tokens (we'll send ALL of this with each record)
        question_context_tokens = 0
        empty_json_token_count = len(tokenizer.encode('[]'))

        # Add JSON strings & token counts to all dataframes
        for df_name, df in dataframes_dict.items():
            if not (df_name.startswith('Record_Context_') or
                    df_name.startswith('Question_Context_') or
                    df_name.startswith('GPA_Questions')):
                continue  # Skip non-context dataframes

            # Skip empty DataFrames
            if df.empty:
                continue

            # Convert each row to JSON and calculate token count
            json_strings = []
            token_counts = []

            for _, row in df.iterrows():
                json_str = row.to_json()
                token_count = len(tokenizer.encode(json_str)) + len(tokenizer.encode(','))
                json_strings.append(json_str)
                token_counts.append(token_count)

            # Add new columns to the dataframe
            df['json'] = json_strings
            df['token_count'] = token_counts

            # Sum up all question context tokens (we send ALL of these with each record)
            if df_name.startswith('Question_Context_') or df_name.startswith('GPA_Questions'):
                question_context_tokens += df['token_count'].sum()

        # Add token count for JSON array brackets around question context
        num_question_context_dataframes = sum(1 for df_name in dataframes_dict.keys()
                                             if (df_name.startswith('Question_Context_')
                                                 or df_name.startswith('GPA_Questions'))
                                             and not dataframes_dict[df_name].empty)
        if num_question_context_dataframes > 0:
            question_context_tokens += num_question_context_dataframes * empty_json_token_count

        logger.log(LogLevel.INFO, f"Total Question Context Tokens (sent with EVERY record): {question_context_tokens}")
        print(f"Total Question Context Tokens (sent with EVERY record): {question_context_tokens}")

        # Validate that each record + all question context fits within available context
        max_record_tokens = 0
        total_records = 0
        records_exceeding_limit = []

        for df_name, df in dataframes_dict.items():
            if not df_name.startswith('Record_Context_'):
                continue

            if df.empty:
                continue

            total_records += len(df)

            # Check each record
            for idx, row in df.iterrows():
                record_tokens = row['token_count'] + empty_json_token_count  # Account for array brackets
                max_record_tokens = max(max_record_tokens, record_tokens)

                # Calculate total tokens for this record + all question context
                total_tokens_for_request = record_tokens + question_context_tokens

                if total_tokens_for_request > available_context:
                    records_exceeding_limit.append({
                        'source_file': row.get('source_file', df_name),
                        'record_tokens': record_tokens,
                        'question_tokens': question_context_tokens,
                        'total_tokens': total_tokens_for_request,
                        'available_context': available_context
                    })

        if records_exceeding_limit:
            error_msg = f"\n{'='*80}\nERROR: {len(records_exceeding_limit)} record(s) exceed token limit!\n{'='*80}\n"
            error_msg += f"Available context: {available_context} tokens\n"
            error_msg += f"Question context (sent with every record): {question_context_tokens} tokens\n"
            error_msg += f"Maximum tokens per record available: {available_context - question_context_tokens} tokens\n\n"
            error_msg += "Records exceeding limit:\n"
            for i, record in enumerate(records_exceeding_limit[:5], 1):  # Show first 5
                error_msg += f"{i}. Source: {record['source_file']}, Record: {record['record_tokens']} tokens, "
                error_msg += f"Total: {record['total_tokens']} tokens (exceeds by {record['total_tokens'] - available_context})\n"

            if len(records_exceeding_limit) > 5:
                error_msg += f"... and {len(records_exceeding_limit) - 5} more\n"

            error_msg += f"\nSolutions:\n"
            error_msg += f"1. Increase Input_Context_Limit in job configuration\n"
            error_msg += f"2. Reduce Input_Context_Overhead\n"
            error_msg += f"3. Reduce the size of your question context examples\n"
            error_msg += f"4. Simplify/reduce the input data records\n"
            error_msg += "="*80

            logger.log(LogLevel.ERROR, error_msg, source_file="context_allocator.py")
            print(error_msg)
            return {}

        # Success! All records fit
        logger.log(LogLevel.INFO, f"Validation successful: All {total_records} records fit within token limits")
        logger.log(LogLevel.INFO, f"Largest record: {max_record_tokens} tokens")
        logger.log(LogLevel.INFO, f"Max tokens per request: {max_record_tokens + question_context_tokens} tokens")

        print(f"\nValidation successful!")
        print(f"Total records to process: {total_records}")
        print(f"Largest record: {max_record_tokens} tokens")
        print(f"Max tokens per request: {max_record_tokens + question_context_tokens} tokens")
        print(f"Available context: {available_context} tokens")

        context_allocation = {
            'Total_Question_Context_Tokens': question_context_tokens,
            'Max_Record_Tokens': max_record_tokens,
            'Available_Context': available_context,
            'Total_Records': total_records
        }

        # Write debugging output to testing/ folder
        output_folder = 'testing'
        os.makedirs(output_folder, exist_ok=True)

        for df_name, df in dataframes_dict.items():
            if 'json' in df.columns:
                # Write DataFrame to CSV
                file_path_csv = os.path.join(output_folder, f"{df_name}.csv")
                df.to_csv(file_path_csv, index=False)

                # Write concatenated JSON to text file
                json_concat = json.dumps(df['json'].tolist())
                file_path_txt = os.path.join(output_folder, f"{df_name}_json.txt")
                with open(file_path_txt, 'w') as f:
                    f.write(json_concat)

        return context_allocation

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error occurred during context allocation: {str(e)}", source_file="context_allocator.py")
        return {}
