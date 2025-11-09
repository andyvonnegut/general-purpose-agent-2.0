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

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutdown requested. Waiting for in-flight requests to complete...")

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

async def process_batches(batches_df, dataframes_dict, selected_job_name, logger):
    """
    Version 2.0: Processes batches in parallel with up to 50 concurrent requests.
    Each batch is one record + all question context.

    Args:
        batches_df (pd.DataFrame): DataFrame containing batch information (one per record).
        dataframes_dict (dict): Dictionary containing data such as record and question contexts.
        selected_job_name (str): The name of the job being processed.
        logger: The unified logger instance.

    Returns:
        None
    """
    try:
        # Check if batches_df is empty
        if batches_df.empty:
            logger.log(LogLevel.WARNING, "No batches to process.")
            return

        # Get job configuration
        job_config_df = dataframes_dict.get('GPA_Job_Configuration')
        if job_config_df is None:
            logger.log(LogLevel.ERROR, "GPA_Job_Configuration not found in dataframes_dict",
                      source_file="batch_processor.py", function_name="process_batches")
            return

        # Get the job configuration for the selected job
        job_config = job_config_df[job_config_df['Job_Name'] == selected_job_name]
        if job_config.empty:
            logger.log(LogLevel.ERROR, f"Job configuration for {selected_job_name} not found",
                      source_file="batch_processor.py", function_name="process_batches")
            return

        # Extract model name and temperature from job config
        model_name = job_config.iloc[0]['Model']
        temperature_raw = job_config.iloc[0].get('Temperature', 1)
        temperature = float(temperature_raw) if temperature_raw is not None else 1.0

        # Load API pricing configuration
        pricing_df = dataframes_dict.get('API_Pricing')
        if pricing_df is None:
            logger.log(LogLevel.ERROR, "API_Pricing configuration not found in dataframes_dict",
                      source_file="batch_processor.py", function_name="process_batches")
            return

        # Load the API key
        my_api_key = load_api_key()
        if not my_api_key:
            logger.log(LogLevel.ERROR, "Failed to load API key",
                      source_file="batch_processor.py", function_name="process_batches")
            return

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
        print(f"Maximum concurrent requests: 50")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")
        print("=" * 80)

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(50)

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

                        async with state_lock:
                            state['failed'] += 1
                            state['in_flight'] -= 1
                        return

                    # Calculate cost
                    pricing_row = pricing_df[pricing_df['Model'] == model_name]
                    if not pricing_row.empty:
                        input_cost_rate = float(pricing_row.iloc[0]['Input_Cost_Per_Million'])
                        output_cost_rate = float(pricing_row.iloc[0]['Output_Cost_Per_Million'])
                        input_token_cost = (completion.usage.prompt_tokens / 1_000_000) * input_cost_rate
                        output_token_cost = (completion.usage.completion_tokens / 1_000_000) * output_cost_rate
                        total_cost = input_token_cost + output_token_cost
                    else:
                        total_cost = 0.0

                    # Append to results (exactly what the model returns - single result item)
                    result_row = results[0] if len(results) > 0 else {}

                    # Add metadata
                    result_row['source_file'] = batch_row['source_file']
                    result_row['batch_id'] = batch_id

                    # Write to CSV (thread-safe)
                    async with csv_lock:
                        # Initialize CSV if needed
                        if not state['csv_file_initialized']:
                            results_df = pd.DataFrame([result_row])
                            results_df.to_csv(output_file, index=False, mode='w')
                            state['csv_file_initialized'] = True
                        else:
                            # Append to existing CSV
                            results_df = pd.DataFrame([result_row])
                            results_df.to_csv(output_file, index=False, mode='a', header=False)

                    # Update state
                    async with state_lock:
                        state['completed'] += 1
                        state['in_flight'] -= 1
                        state['total_cost'] += total_cost
                        state['total_input_tokens'] += completion.usage.prompt_tokens
                        state['total_output_tokens'] += completion.usage.completion_tokens

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.log(LogLevel.ERROR, f"Error processing batch {batch_id}: {str(e)}\n{error_details}",
                              source_file="batch_processor.py")
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

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error in parallel processing: {str(e)}",
                  source_file="batch_processor.py", function_name="process_batches")

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
