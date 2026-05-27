import pandas as pd
import tiktoken
from unified_logger import get_logger, LogLevel
from errors import PipelineError
import json
import math
import os

# Fixed token allowance for the static prompt scaffolding (developer/role
# framing, the per-request instruction lines, and response slack) that sits
# around the record + question context. Subtracted from the available context
# when sizing chunks so a packed chunk plus its wrapper does not overflow.
SCAFFOLD_MARGIN = 1500

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
        # Reserve room for the model's output. Input_Context_Limit is the model's
        # full window, shared between the prompt and the completion, so the per-
        # request *input* budget must also subtract the output allowance. Without
        # this the budget overstates what fits and OpenAI rejects the call at
        # request time with context_length_exceeded even though local validation
        # "passed".
        try:
            output_reserve = int(selected_job['Output_Context_Limit'].values[0])
        except (KeyError, ValueError, TypeError):
            output_reserve = 0
        available_context = input_context_limit - overhead_tokens - output_reserve

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
        # Largest single question-context row. A chunk must hold at least one row,
        # so this is the smallest amount of context any request can carry and thus
        # governs whether the data can be made to fit by splitting at all.
        max_question_row_tokens = 0
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
                if len(token_counts) > 0:
                    max_question_row_tokens = max(max_question_row_tokens, max(token_counts))

        # Add token count for JSON array brackets around question context
        num_question_context_dataframes = sum(1 for df_name in dataframes_dict.keys()
                                             if (df_name.startswith('Question_Context_')
                                                 or df_name.startswith('GPA_Questions'))
                                             and not dataframes_dict[df_name].empty)
        if num_question_context_dataframes > 0:
            question_context_tokens += num_question_context_dataframes * empty_json_token_count

        logger.log(LogLevel.INFO, f"Total Question Context Tokens (sent with EVERY record): {question_context_tokens}")
        print(f"Total Question Context Tokens (sent with EVERY record): {question_context_tokens}")

        # Find the largest record. We size chunks against the largest record so a
        # single chunking plan is valid for every record (simple and conservative).
        max_record_tokens = 0
        total_records = 0

        for df_name, df in dataframes_dict.items():
            if not df_name.startswith('Record_Context_'):
                continue
            if df.empty:
                continue
            total_records += len(df)
            for idx, row in df.iterrows():
                record_tokens = row['token_count'] + empty_json_token_count  # Account for array brackets
                max_record_tokens = max(max_record_tokens, record_tokens)

        # Per-request budget left for question context once the (largest) record
        # and a fixed allowance for the static prompt scaffolding are accounted for.
        ctx_budget = available_context - max_record_tokens - SCAFFOLD_MARGIN

        # Feasibility: the smallest possible request is one record + one (largest)
        # question-context row. If even that doesn't fit, no amount of splitting
        # helps -> this is the genuinely-invalid case the caller turns into a
        # PipelineError.
        if max_record_tokens > available_context:
            msg = (f"Largest record ({max_record_tokens} tokens) alone exceeds the available "
                   f"input budget ({available_context} tokens) for model '{selected_model}' "
                   f"(window {input_context_limit}, overhead {overhead_tokens}, output reserve "
                   f"{output_reserve}). Choose a larger-context model or reduce the record size.")
            logger.log(LogLevel.ERROR, msg, source_file="context_allocator.py")
            raise PipelineError(msg)
        if question_context_tokens > 0 and ctx_budget < max_question_row_tokens:
            msg = (f"A single question-context row ({max_question_row_tokens} tokens) cannot fit "
                   f"alongside the largest record ({max_record_tokens} tokens) for model "
                   f"'{selected_model}'. Per-chunk budget is {ctx_budget} tokens (available "
                   f"{available_context}, total question context {question_context_tokens}). "
                   f"Choose a larger-context model, reduce Input_Context_Overhead, or shrink the "
                   f"context rows.")
            logger.log(LogLevel.ERROR, msg, source_file="context_allocator.py")
            raise PipelineError(msg)

        # Decide how many context chunks each record must be run against. One chunk
        # == today's behaviour (whole context in a single request); >1 means the
        # record is run per-chunk and the answers are stitched downstream.
        if question_context_tokens <= ctx_budget:
            num_context_chunks = 1
        else:
            num_context_chunks = math.ceil(question_context_tokens / ctx_budget)

        if num_context_chunks > 1:
            logger.log(LogLevel.INFO,
                       f"Question context ({question_context_tokens} tokens) exceeds the per-record "
                       f"budget ({ctx_budget}); it will be split into {num_context_chunks} chunks and "
                       f"the per-chunk answers stitched together.",
                       source_file="context_allocator.py")
            print(f"Splitting question context into {num_context_chunks} chunks "
                  f"(budget {ctx_budget} tokens/chunk).")
        else:
            logger.log(LogLevel.INFO, f"Validation successful: All {total_records} records fit "
                       f"with the full question context in a single request.")

        print(f"\nValidation successful!")
        print(f"Total records to process: {total_records}")
        print(f"Largest record: {max_record_tokens} tokens")
        print(f"Question context: {question_context_tokens} tokens")
        print(f"Available context: {available_context} tokens")
        print(f"Context chunks per record: {num_context_chunks}")

        context_allocation = {
            'Total_Question_Context_Tokens': question_context_tokens,
            'Max_Record_Tokens': max_record_tokens,
            'Available_Context': available_context,
            'Total_Records': total_records,
            'Num_Context_Chunks': num_context_chunks,
            'Context_Budget_Per_Chunk': ctx_budget,
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

    except PipelineError:
        # Already a clear, caller-facing diagnosis — let it propagate so the MCP
        # tool surfaces the specifics rather than a generic "validation failed".
        raise
    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error occurred during context allocation: {str(e)}", source_file="context_allocator.py")
        return {}
