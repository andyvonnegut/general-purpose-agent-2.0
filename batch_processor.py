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

        question_keys = job_questions['Key'].tolist()
        result_columns = question_keys + ['source_file', 'batch_id']

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
            'total_cached_input_tokens': 0,
            'total_output_tokens': 0,
            'csv_file_initialized': False
        }
        state_lock = asyncio.Lock()

        def build_cost_info(completion):
            """Tokens + cost for a completion (model-priced), or zeros if usage
            is unavailable. Shape matches unified_logger.log_api_call_complete.

            Splits input into fresh vs cached so the cost matches what OpenAI
            actually bills: prompts >= 1024 tokens get automatic prompt caching
            and the cached portion is billed at Cached_Input_Cost_Per_Million
            from API_Pricing.csv (e.g. gpt-5.4 cached $0.25/M vs full $2.50/M).
            Falls back to the full rate for models that don't list a cached rate.
            """
            usage = getattr(completion, 'usage', None) if completion is not None else None
            in_tok = getattr(usage, 'prompt_tokens', 0) or 0
            out_tok = getattr(usage, 'completion_tokens', 0) or 0
            details = getattr(usage, 'prompt_tokens_details', None) if usage else None
            cached_in_tok = getattr(details, 'cached_tokens', 0) or 0
            # Defensive: some response shapes report cached > prompt; clamp so
            # fresh stays non-negative.
            cached_in_tok = min(cached_in_tok, in_tok)
            fresh_in_tok = in_tok - cached_in_tok
            pr = pricing_df[pricing_df['Model'] == model_name]
            if not pr.empty:
                full_rate = float(pr.iloc[0]['Input_Cost_Per_Million'])
                cached_raw = pr.iloc[0].get('Cached_Input_Cost_Per_Million')
                cached_rate = (
                    float(cached_raw)
                    if cached_raw not in (None, '') and pd.notna(cached_raw)
                    else full_rate
                )
                in_cost = (fresh_in_tok / 1_000_000) * full_rate \
                        + (cached_in_tok / 1_000_000) * cached_rate
                out_cost = (out_tok / 1_000_000) * float(pr.iloc[0]['Output_Cost_Per_Million'])
            else:
                in_cost = out_cost = 0.0
            return {
                'model': model_name,
                'input_tokens': in_tok,
                'output_tokens': out_tok,
                'cached_input_tokens': cached_in_tok,
                'input': in_cost,
                'output': out_cost,
                'total': in_cost + out_cost,
            }

        def build_messages(system_content, record_json_str, question_context):
            """Single-request message scaffolding: a record plus (optionally) a
            slice of question context. Identical to the 2.0 layout."""
            messages = [
                {"role": "developer", "content": system_content},
                {"role": "user", "content": "Here is the record I want reviewed. Provide a detailed response."},
                {"role": "user", "content": f"[{record_json_str}]"},
            ]
            if question_context and len(question_context) > 0:
                messages.append({"role": "developer", "content": "Here is information you can use to help create your response:"})
                # default=str so native Excel-typed cells (datetime, Timestamp,
                # Decimal, etc.) serialize as their readable str() form instead
                # of raising — pandas keeps these as native types when the
                # context source is .xlsx, and stdlib json has no encoder for
                # them. Mirrors observability.record_run's pattern.
                messages.append({"role": "developer", "content": json.dumps(question_context, default=str)})
            return messages

        def build_reduce_messages(system_content, record_json_str, partial_answers):
            """Reduce-step scaffolding: consolidate per-chunk answers for the same
            record into one final answer using the same output schema."""
            return [
                {"role": "developer", "content": system_content},
                {"role": "user", "content": "Here is the record I want reviewed. Provide a detailed response."},
                {"role": "user", "content": f"[{record_json_str}]"},
                {"role": "developer", "content": (
                    "The reference data was too large to send at once, so this record was evaluated "
                    "against several slices of it separately. Below are the partial answers, one per "
                    "slice, for THIS SAME record. Consolidate them into a single final answer in the "
                    "required schema: union any matches (e.g. combine multiple matching IDs), drop "
                    "non-matches such as 'No Match' whenever a real match exists, and reconcile any "
                    "conflicts into the single best answer.")},
                # default=str for the same reason as build_messages — defense
                # in depth, in case any partial answer contains a non-JSON type.
                {"role": "developer", "content": json.dumps(partial_answers, default=str)},
            ]

        def deterministic_union(partial_answers):
            """Fallback stitch if the reduce call fails: per answer field, union the
            non-empty, non-'No Match' values across chunks, de-duplicated and joined."""
            merged = {}
            for key in question_keys:
                seen, values = set(), []
                for ans in partial_answers:
                    val = ans.get(key, '')
                    if val is None:
                        continue
                    text = str(val).strip()
                    if text == '' or text.lower() == 'no match':
                        continue
                    if text not in seen:
                        seen.add(text)
                        values.append(text)
                merged[key] = ', '.join(values)
            return merged

        async def invoke_model(messages, batch_id, response_format, source_file, status_label):
            """Make one model call under the shared concurrency limit, parse it, log
            the transcript, and accumulate cost. Returns the parsed `results` list,
            or None when the call was skipped / failed / returned no results.
            The semaphore wraps the individual call (not the whole record) so a
            record split across chunks still respects max_parallel_requests."""
            global shutdown_requested
            nonlocal state

            request_data = {
                "model": model_name,
                "temperature": temperature,
                "source_file": source_file,
                "messages": messages,
            }

            async with semaphore:
                if shutdown_requested:
                    return None
                try:
                    completion = await client.beta.chat.completions.parse(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        response_format=response_format,
                    )
                except Exception as e:
                    # Record an error transcript so this per-record failure is
                    # visible via get_transcripts — otherwise an all-failed run
                    # surfaces no explanation at all. Re-raise so the caller's
                    # failure accounting (failed += 1) is unchanged.
                    logger.log_api_call_complete(
                        request_data, {"error": str(e)},
                        {"input_tokens": 0, "output_tokens": 0,
                         "cached_input_tokens": 0,
                         "input": 0, "output": 0, "total": 0},
                        batch_id=batch_id, status='error')
                    raise

            cost_info = build_cost_info(completion)
            async with state_lock:
                state['total_cost'] += cost_info['total']
                state['total_input_tokens'] += cost_info['input_tokens']
                state['total_cached_input_tokens'] += cost_info.get('cached_input_tokens', 0)
                state['total_output_tokens'] += cost_info['output_tokens']

            choice = completion.choices[0].message.content
            try:
                response_data = json.loads(choice)
                results = response_data.get('results')
                # Defense in depth for non-strict / future models that echo
                # the JSON Schema outer shape and bury the value under the
                # schema's "properties" keyword — observed on gpt-5.5 before
                # strict mode landed: {"properties": {"results": [...]}}.
                if not results and isinstance(response_data.get('properties'), dict):
                    results = response_data['properties'].get('results')
                results = results or []
            except (json.JSONDecodeError, TypeError):
                logger.log(LogLevel.ERROR,
                           f"Failed to parse response for batch {batch_id} ({status_label}). "
                           f"Response: {(choice or '')[:500]}",
                           source_file="batch_processor.py")
                logger.log_api_call_complete(
                    request_data, {"raw_content": (choice or "")[:5000]},
                    cost_info, batch_id=batch_id, status='parse_error')
                return None

            status = status_label if results else 'empty_results'
            logger.log_api_call_complete(
                request_data, response_data, cost_info, batch_id=batch_id, status=status)
            if not results:
                logger.log(LogLevel.WARNING,
                           f"Empty results for batch {batch_id} ({status_label}).",
                           source_file="batch_processor.py")
                return None
            return results

        async def process_single_batch(batch_row):
            """Process one record. One context chunk == a single call (unchanged);
            multiple chunks == one call per chunk plus a reduce call that stitches
            the partial answers into the single output row written for the record."""
            global shutdown_requested
            nonlocal state

            if shutdown_requested:
                return

            batch_id = batch_row['batch_id']
            source_file = batch_row.get('source_file', 'unknown')

            async with state_lock:
                state['in_flight'] += 1
                current_completed = state['completed']
                current_failed = state['failed']
                current_in_flight = state['in_flight']
            print(f"Processing: {current_completed}/{total_batches} complete, "
                  f"{current_failed} failed, {current_in_flight} in-flight")

            failed = False
            try:
                system_content = json.loads(batch_row['system_role'])['content']
                record_json_str = batch_row['record_json']
                response_format = batch_row['response_format']
                chunks = batch_row['question_context_chunks']

                if len(chunks) <= 1:
                    # Single request — full (or empty) context in one call.
                    context = chunks[0] if chunks else []
                    results = await invoke_model(
                        build_messages(system_content, record_json_str, context),
                        batch_id, response_format, source_file, status_label='success')
                    if not results:
                        failed = True
                        return
                    final_answer = results[0]
                else:
                    # Fan out: run the record against each context chunk concurrently.
                    chunk_results = await asyncio.gather(*[
                        invoke_model(
                            build_messages(system_content, record_json_str, chunk),
                            batch_id, response_format, source_file, status_label='partial')
                        for chunk in chunks
                    ])
                    partial_answers = [r[0] for r in chunk_results if r]
                    if not partial_answers:
                        logger.log(LogLevel.ERROR,
                                   f"All {len(chunks)} context chunks failed for batch {batch_id}.",
                                   source_file="batch_processor.py")
                        failed = True
                        return

                    # Reduce: consolidate the partial answers into one final answer.
                    reduce_results = await invoke_model(
                        build_reduce_messages(system_content, record_json_str, partial_answers),
                        batch_id, response_format, source_file, status_label='success')
                    if reduce_results:
                        final_answer = reduce_results[0]
                    else:
                        logger.log(LogLevel.WARNING,
                                   f"Reduce step failed for batch {batch_id}; "
                                   f"falling back to deterministic union of {len(partial_answers)} chunk answers.",
                                   source_file="batch_processor.py")
                        final_answer = deterministic_union(partial_answers)

                # One result row per record (preserves the row-position join downstream).
                result_row = {
                    column: sanitize_csv_value(final_answer.get(column, ''))
                    for column in result_columns
                }
                result_row['source_file'] = sanitize_csv_value(source_file)
                result_row['batch_id'] = batch_id

                async with csv_lock:
                    results_df = pd.DataFrame([result_row], columns=result_columns)
                    if not state['csv_file_initialized']:
                        results_df.to_csv(output_file, index=False, mode='w')
                        state['csv_file_initialized'] = True
                    else:
                        results_df.to_csv(output_file, index=False, mode='a', header=False)

            except Exception as e:
                import traceback
                logger.log(LogLevel.ERROR, f"Error processing batch {batch_id}: {str(e)}\n{traceback.format_exc()}",
                          source_file="batch_processor.py")
                failed = True
            finally:
                async with state_lock:
                    if failed:
                        state['failed'] += 1
                    else:
                        state['completed'] += 1
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
        cached_in = state['total_cached_input_tokens']
        total_in = state['total_input_tokens']
        cache_ratio = (cached_in / total_in) if total_in else 0.0
        print(f"\nCost Summary:")
        print(f"  Input tokens: {total_in:,}  (cached: {cached_in:,}, "
              f"{cache_ratio:.0%} hit rate)")
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
            'cached_input_tokens': state['total_cached_input_tokens'],
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
