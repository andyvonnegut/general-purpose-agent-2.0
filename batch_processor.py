import os
import csv
import pandas as pd
import json
import asyncio
import signal
from datetime import datetime
from openai import AsyncOpenAI
from unified_logger import LogLevel, get_logger

# Global shutdown flag
shutdown_requested = False

def sanitize_csv_value(value):
    """Normalize string values so CSV imports do not break on embedded control characters."""
    if not isinstance(value, str):
        return value

    sanitized = value.replace('\x00a0', ' ')
    sanitized = sanitized.replace('\x00', ' ')
    sanitized = sanitized.replace('\r\n', ' ')
    sanitized = sanitized.replace('\r', ' ')
    sanitized = sanitized.replace('\n', ' ')
    return sanitized.strip()

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutdown requested. Waiting for in-flight requests to complete...")

def install_signal_handler():
    """Register the SIGINT handler. Called by the CLI (main.py) only.

    Importing this module no longer registers a process-wide handler, so an
    MCP host that imports batch_processor keeps control of its own signals.
    Signal handlers can only be installed on the main thread; guard for that.
    """
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal_handler)

async def process_batches(
    batches_df,
    dataframes_dict,
    selected_job_name,
    logger,
    max_parallel_requests=50
):
    """
    Version 2.0: Processes batches in parallel with a configurable request limit.
    Each batch is one record + all question context.

    Args:
        batches_df (pd.DataFrame): DataFrame containing batch information (one per record).
        dataframes_dict (dict): Dictionary containing data such as record and question contexts.
        selected_job_name (str): The name of the job being processed.
        logger: The unified logger instance.
        max_parallel_requests (int): Maximum number of concurrent requests to allow.

    Returns:
        dict: Summary of the run with keys total_records, succeeded, failed,
            duration_seconds, input_tokens, output_tokens, total_cost. Returns
            an empty dict on early-exit guard conditions.
    """
    try:
        # Check if batches_df is empty
        if batches_df.empty:
            logger.log(LogLevel.WARNING, "No batches to process.")
            return {}

        # Get job configuration
        job_config_df = dataframes_dict.get('GPA_Job_Configuration')
        if job_config_df is None:
            logger.log(LogLevel.ERROR, "GPA_Job_Configuration not found in dataframes_dict",
                      source_file="batch_processor.py", function_name="process_batches")
            return {}

        # Get the job configuration for the selected job
        job_config = job_config_df[job_config_df['Job_Name'] == selected_job_name]
        if job_config.empty:
            logger.log(LogLevel.ERROR, f"Job configuration for {selected_job_name} not found",
                      source_file="batch_processor.py", function_name="process_batches")
            return {}

        # Extract model name and temperature from job config
        model_name = job_config.iloc[0]['Model']
        temperature_raw = job_config.iloc[0].get('Temperature', 1)
        temperature = float(temperature_raw) if temperature_raw is not None else 1.0

        # Load API pricing configuration
        pricing_df = dataframes_dict.get('API_Pricing')
        if pricing_df is None:
            logger.log(LogLevel.ERROR, "API_Pricing configuration not found in dataframes_dict",
                      source_file="batch_processor.py", function_name="process_batches")
            return {}

        # Load the output schema so CSV rows are always written in a stable column order.
        questions_df = dataframes_dict.get('GPA_Questions')
        if questions_df is None:
            logger.log(LogLevel.ERROR, "GPA_Questions configuration not found in dataframes_dict",
                      source_file="batch_processor.py", function_name="process_batches")
            return {}

        job_questions = questions_df[questions_df['Job_Name'] == selected_job_name]
        if job_questions.empty:
            logger.log(LogLevel.ERROR, f"No GPA_Questions configuration found for job {selected_job_name}",
                      source_file="batch_processor.py", function_name="process_batches")
            return {}

        result_columns = job_questions['Key'].tolist() + ['source_file', 'batch_id']

        # Load the API key
        my_api_key = load_api_key()
        if not my_api_key:
            logger.log(LogLevel.ERROR, "Failed to load API key",
                      source_file="batch_processor.py", function_name="process_batches")
            return {}

        # Create async OpenAI client
        client = AsyncOpenAI(api_key=my_api_key)

        # Prepare output file
        output_file = f"Results/{selected_job_name}_results.csv"
        os.makedirs('Results', exist_ok=True)

        # Create CSV writer lock and file
        csv_lock = asyncio.Lock()
        csv_file_initialized = False

        # Progress tracking
        total_batches = len(batches_df)
        completed_count = 0
        failed_count = 0
        start_time = datetime.now()

        print(f"\nStarting parallel processing of {total_batches} records...")
        print(f"Maximum concurrent requests: {max_parallel_requests}")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")
        print("=" * 80)

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_parallel_requests)

        # Shared state for tracking
        state = {
            'completed': 0,
            'failed': 0,
            'in_flight': 0,
            'total_cost': 0.0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'csv_file_initialized': False
        }
        state_lock = asyncio.Lock()

        def build_cost_info(completion):
            """Tokens + cost for a completion (model-priced), or zeros if usage
            is unavailable. Shape matches unified_logger.log_api_call_complete."""
            usage = getattr(completion, 'usage', None) if completion is not None else None
            in_tok = getattr(usage, 'prompt_tokens', 0) or 0
            out_tok = getattr(usage, 'completion_tokens', 0) or 0
            pr = pricing_df[pricing_df['Model'] == model_name]
            if not pr.empty:
                in_cost = (in_tok / 1_000_000) * float(pr.iloc[0]['Input_Cost_Per_Million'])
                out_cost = (out_tok / 1_000_000) * float(pr.iloc[0]['Output_Cost_Per_Million'])
            else:
                in_cost = out_cost = 0.0
            return {
                'model': model_name,
                'input_tokens': in_tok,
                'output_tokens': out_tok,
                'input': in_cost,
                'output': out_cost,
                'total': in_cost + out_cost,
            }

        async def process_single_batch(batch_row):
            """Process a single batch (one record)"""
            global shutdown_requested
            nonlocal state

            if shutdown_requested:
                return

            async with semaphore:
                if shutdown_requested:
                    return

                batch_id = batch_row['batch_id']

                async with state_lock:
                    state['in_flight'] += 1
                    current_completed = state['completed']
                    current_failed = state['failed']
                    current_in_flight = state['in_flight']

                print(f"Processing: {current_completed}/{total_batches} complete, "
                      f"{current_failed} failed, {current_in_flight} in-flight")

                try:
                    # Build messages
                    system_role_data = json.loads(batch_row['system_role'])
                    record_json_str = batch_row['record_json']
                    question_context = batch_row['question_context']
                    response_format = batch_row['response_format']

                    # Build the API messages
                    if question_context and len(question_context) > 0:
                        messages = [
                            {"role": "developer", "content": system_role_data['content']},
                            {"role": "user", "content": "Here is the record I want reviewed. Provide a detailed response."},
                            {"role": "user", "content": f"[{record_json_str}]"},
                            {"role": "developer", "content": "Here is information you can use to help create your response:"},
                            {"role": "developer", "content": json.dumps(question_context)},
                        ]
                    else:
                        messages = [
                            {"role": "developer", "content": system_role_data['content']},
                            {"role": "user", "content": "Here is the record I want reviewed. Provide a detailed response."},
                            {"role": "user", "content": f"[{record_json_str}]"},
                        ]

                    # Request payload captured for the per-record transcript
                    # (Logs/api_calls/api_calls.csv via log_api_call_complete).
                    request_data = {
                        "model": model_name,
                        "temperature": temperature,
                        "source_file": batch_row.get('source_file', ''),
                        "messages": messages,
                    }

                    # Make API call
                    completion = await client.beta.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        response_format=response_format
                    )

                    # Log detailed API response for debugging
                    logger.log(LogLevel.DEBUG,
                              f"Batch {batch_id} - API Response: "
                              f"finish_reason={completion.choices[0].finish_reason}, "
                              f"refusal={getattr(completion.choices[0].message, 'refusal', None)}, "
                              f"content_length={len(completion.choices[0].message.content) if completion.choices[0].message.content else 0}",
                              source_file="batch_processor.py")

                    # Extract response
                    choice = completion.choices[0].message.content

                    # Parse response
                    try:
                        response_data = json.loads(choice)
                        results = response_data.get('results', [])
                    except json.JSONDecodeError:
                        logger.log(LogLevel.ERROR, f"Failed to parse response for batch {batch_id}. Response: {choice[:500]}",
                                  source_file="batch_processor.py")
                        logger.log_api_call_complete(
                            request_data, {"raw_content": (choice or "")[:5000]},
                            build_cost_info(completion), batch_id=batch_id, status='parse_error')
                        async with state_lock:
                            state['failed'] += 1
                            state['in_flight'] -= 1
                        return

                    if not results:
                        # Handle response_format which may be dict or string
                        if isinstance(response_format, str):
                            response_format_dict = json.loads(response_format)
                        else:
                            response_format_dict = response_format

                        # Log full details for debugging
                        debug_info = {
                            'batch_id': batch_id,
                            'finish_reason': completion.choices[0].finish_reason,
                            'refusal': getattr(completion.choices[0].message, 'refusal', None),
                            'response_data': response_data,
                            'record_preview': record_json_str[:200],
                            'response_format_name': response_format_dict.get('json_schema', {}).get('name', 'unknown'),
                            'usage': {
                                'prompt_tokens': completion.usage.prompt_tokens,
                                'completion_tokens': completion.usage.completion_tokens
                            }
                        }
                        logger.log(LogLevel.ERROR,
                                  f"Empty results for batch {batch_id}. Details: {json.dumps(debug_info, indent=2)}",
                                  source_file="batch_processor.py")

                        # Save full request/response to debug file
                        debug_folder = f"Logs/debug/{selected_job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        os.makedirs(debug_folder, exist_ok=True)
                        with open(f"{debug_folder}/batch_{batch_id}_empty_results.json", 'w') as f:
                            json.dump({
                                'messages': messages,
                                'response_format': response_format_dict,
                                'completion': {
                                    'content': completion.choices[0].message.content,
                                    'finish_reason': completion.choices[0].finish_reason,
                                    'refusal': getattr(completion.choices[0].message, 'refusal', None)
                                }
                            }, f, indent=2)

                        logger.log_api_call_complete(
                            request_data, response_data,
                            build_cost_info(completion), batch_id=batch_id, status='empty_results')
                        async with state_lock:
                            state['failed'] += 1
                            state['in_flight'] -= 1
                        return

                    # Calculate cost
                    cost_info = build_cost_info(completion)
                    total_cost = cost_info['total']

                    # Persist the full prompt + parsed answer for this record so it
                    # is retrievable later via the transcript MCP tools.
                    logger.log_api_call_complete(
                        request_data, response_data, cost_info,
                        batch_id=batch_id, status='success')

                    # Append to results (exactly what the model returns - single result item)
                    raw_result_row = results[0] if len(results) > 0 else {}
                    result_row = {
                        column: sanitize_csv_value(raw_result_row.get(column, ''))
                        for column in result_columns
                    }

                    # Add metadata
                    result_row['source_file'] = sanitize_csv_value(batch_row['source_file'])
                    result_row['batch_id'] = batch_id

                    # Write to CSV (thread-safe)
                    async with csv_lock:
                        # Initialize CSV if needed
                        if not state['csv_file_initialized']:
                            results_df = pd.DataFrame([result_row], columns=result_columns)
                            results_df.to_csv(output_file, index=False, mode='w')
                            state['csv_file_initialized'] = True
                        else:
                            # Append to existing CSV
                            results_df = pd.DataFrame([result_row], columns=result_columns)
                            results_df.to_csv(output_file, index=False, mode='a', header=False)

                    # Update state
                    async with state_lock:
                        state['completed'] += 1
                        state['in_flight'] -= 1
                        state['total_cost'] += cost_info['total']
                        state['total_input_tokens'] += cost_info['input_tokens']
                        state['total_output_tokens'] += cost_info['output_tokens']

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.log(LogLevel.ERROR, f"Error processing batch {batch_id}: {str(e)}\n{error_details}",
                              source_file="batch_processor.py")
                    try:
                        logger.log_api_call_complete(
                            locals().get('request_data', {}), {"error": str(e)},
                            build_cost_info(locals().get('completion')),
                            batch_id=batch_id, status='error')
                    except Exception:
                        pass
                    async with state_lock:
                        state['failed'] += 1
                        state['in_flight'] -= 1

        # Create tasks for all batches
        tasks = []
        for idx, row in batches_df.iterrows():
            if shutdown_requested:
                break
            task = asyncio.create_task(process_single_batch(row))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total records: {total_batches}")
        print(f"Successfully processed: {state['completed']}")
        print(f"Failed: {state['failed']}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Average time per record: {duration/max(state['completed'], 1):.2f} seconds")
        print(f"\nCost Summary:")
        print(f"  Input tokens: {state['total_input_tokens']:,}")
        print(f"  Output tokens: {state['total_output_tokens']:,}")
        print(f"  Total cost: ${state['total_cost']:.4f}")
        print(f"\nResults saved to: {output_file}")
        print("=" * 80)

        # Log summary
        logger.log(LogLevel.INFO, f"Processing complete: {state['completed']}/{total_batches} successful, "
                  f"{state['failed']} failed, ${state['total_cost']:.4f} cost")

        return {
            'total_records': total_batches,
            'succeeded': state['completed'],
            'failed': state['failed'],
            'duration_seconds': round(duration, 2),
            'input_tokens': state['total_input_tokens'],
            'output_tokens': state['total_output_tokens'],
            'total_cost': round(state['total_cost'], 4),
        }

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error in parallel processing: {str(e)}",
                  source_file="batch_processor.py", function_name="process_batches")
        return {}

def load_api_key():
    """
    Loads the OpenAI API key from Configuration_Files/API_Keys.csv

    Returns:
        str: The OpenAI API key.
    """
    try:
        file_path = 'Configuration_Files/API_Keys.csv'
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                return row['API_Key']

        logger = get_logger()
        logger.log(LogLevel.ERROR, "No API Key found in API_Keys.csv",
                  source_file="batch_processor.py", function_name="load_api_key")
        return None

    except Exception as e:
        logger = get_logger()
        logger.log(LogLevel.ERROR, f"Error loading API key: {str(e)}",
                  source_file="batch_processor.py", function_name="load_api_key")
        return None
