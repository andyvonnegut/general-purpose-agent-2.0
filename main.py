import os
import sys
import asyncio
import pandas as pd
from unified_logger import get_logger, LogLevel
from data_loader import load_data
from context_allocator import allocate_context
from batch_builder import build_batches
from batch_processor import process_batches, install_signal_handler

DEFAULT_MAX_PARALLEL_REQUESTS = 50
MAX_PARALLEL_REQUESTS_LIMIT = 1000

def prompt_user_for_job(job_config_df, logger):
    """
    Prompts the user to select a job from the available job configurations.

    Args:
        job_config_df (pd.DataFrame): DataFrame containing the job configuration details.
        logger: The unified logger instance.

    Returns:
        job_name (str): The job name selected by the user.
    """
    job_names = job_config_df['Job_Name'].unique()

    # Display job names to the user
    logger.log(LogLevel.INFO, "Available Jobs:", to_file=False)
    for idx, job in enumerate(job_names, 1):
        logger.log(LogLevel.INFO, f"{idx}. {job}", to_file=False)

    # Prompt user to select a job by index
    while True:
        try:
            job_index = int(input(f"Please select a job by entering the corresponding number (1-{len(job_names)}): "))
            if 1 <= job_index <= len(job_names):
                selected_job = job_names[job_index - 1]
                return selected_job
            else:
                logger.log(LogLevel.WARNING,
                          f"Invalid input. Please select a number between 1 and {len(job_names)}.",
                          to_file=False)
        except ValueError:
            logger.log(LogLevel.WARNING, "Invalid input. Please enter a valid number.", to_file=False)

def prompt_user_for_max_parallel_requests(logger, default=DEFAULT_MAX_PARALLEL_REQUESTS):
    """
    Prompts the user for the maximum number of parallel requests to run.

    Args:
        logger: The unified logger instance.
        default (int): The default concurrency to use when the user presses Enter
            or provides invalid input.

    Returns:
        int: The selected maximum concurrency.
    """
    user_input = input(
        f"Enter max parallel threads (1-{MAX_PARALLEL_REQUESTS_LIMIT}) [{default}]: "
    ).strip()

    if not user_input:
        return default

    try:
        max_parallel_requests = int(user_input)
        if 1 <= max_parallel_requests <= MAX_PARALLEL_REQUESTS_LIMIT:
            return max_parallel_requests
    except ValueError:
        pass

    logger.log(
        LogLevel.WARNING,
        f"Invalid max parallel threads value (must be 1-{MAX_PARALLEL_REQUESTS_LIMIT}). "
        f"Using default: {default}",
        to_file=False
    )
    return default

def save_batches_to_csv(batches_df, selected_job_name, logger):
    """
    Saves the batch DataFrame to a CSV file in the Logs/batches directory.

    Args:
        batches_df (pd.DataFrame): The DataFrame containing batch data.
        selected_job_name (str): The name of the job being processed (used in the file name).
        logger: The unified logger instance.
    """
    try:
        file_name = f"{selected_job_name}_batches.csv"

        # Convert complex columns to strings for CSV compatibility
        batches_to_save = batches_df.copy()
        batches_to_save['record_data'] = batches_to_save['record_data'].apply(str)
        batches_to_save['question_context'] = batches_to_save['question_context'].apply(str)
        batches_to_save['response_format'] = batches_to_save['response_format'].apply(str)

        logger.log_data(file_name, batches_to_save.to_dict('records'), format='csv', subfolder='batches')
        logger.log(LogLevel.INFO, f"Batches saved to Logs/batches/{file_name}")
    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error saving batches to CSV: {str(e)}",
                  source_file="main.py", function_name="save_batches_to_csv")

def main():
    """
    Version 2.0: Main orchestrator for parallel batch processing.
    No chunking - each record is processed individually with all question context.
    """
    # Initialize logger
    logger = get_logger()

    # Register Ctrl+C handler for graceful shutdown (CLI only)
    install_signal_handler()

    try:
        # Step 1: Load data
        logger.log(LogLevel.INFO, "Loading data...")
        dataframes_dict = load_data()

        # Step 2: Extract job configuration from data loader
        gpa_job_config = dataframes_dict.get('GPA_Job_Configuration')
        if gpa_job_config is None:
            logger.log(LogLevel.ERROR, "GPA_Job_Configuration file is missing in data.",
                      source_file="main.py", function_name="main")
            sys.exit(1)

        # Step 3: List jobs and prompt user to pick one
        selected_job_name = prompt_user_for_job(gpa_job_config, logger)
        max_parallel_requests = prompt_user_for_max_parallel_requests(logger)

        # Update logger with job name
        logger = get_logger(selected_job_name)
        logger.log(LogLevel.INFO, f"You selected the job: {selected_job_name}")
        logger.log(LogLevel.INFO, f"Max parallel threads: {max_parallel_requests}")

        # Step 4: Validate context allocation (no chunking in 2.0)
        logger.log(LogLevel.INFO, "Validating context allocation...")
        context_allocation = allocate_context(dataframes_dict, selected_job_name)

        # Step 5: Check validation result
        if context_allocation:
            logger.log(LogLevel.INFO, "Context validation successful")
            logger.log(LogLevel.DEBUG, f"Validation details: {context_allocation}")
        else:
            logger.log(LogLevel.ERROR, "Context validation failed. Some records exceed token limits.",
                      source_file="main.py", function_name="main")
            sys.exit(1)

        # Step 6: Build batches (one per record, all with full question context)
        logger.log(LogLevel.INFO, "Building batches...")
        batches_df = build_batches(dataframes_dict, selected_job_name, allocation=context_allocation)

        if not batches_df.empty:
            logger.log(LogLevel.INFO, f"Batch building complete. {len(batches_df)} batches created (one per record).")
        else:
            logger.log(LogLevel.WARNING, "No batches were created.")
            sys.exit(1)

        # Step 7: Save batches to CSV in the Logs folder
        save_batches_to_csv(batches_df, selected_job_name, logger)

        # Step 8: Process all batches in parallel (async)
        logger.log(LogLevel.INFO, "Starting parallel batch processing...")
        asyncio.run(
            process_batches(
                batches_df,
                dataframes_dict,
                selected_job_name,
                logger,
                max_parallel_requests=max_parallel_requests
            )
        )
        logger.log(LogLevel.INFO, "Parallel batch processing complete.")

    except Exception as e:
        logger.log(LogLevel.ERROR, f"Error in main pipeline: {str(e)}",
                  source_file="main.py", function_name="main")
        sys.exit(1)

if __name__ == "__main__":
    main()
