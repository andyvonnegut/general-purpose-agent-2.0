"""
Unified logging system for the General Purpose Agent application.
Provides centralized logging with different levels and output formats.
"""

import os
import csv
import json
from datetime import datetime
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class UnifiedLogger:
    """Unified logger class for consistent logging across the application."""

    def __init__(self, job_name=None):
        """Initialize the logger with optional job name."""
        self.job_name = job_name
        self.log_dir = "Logs"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory structure
        self.sessions_dir = os.path.join(self.log_dir, "sessions")
        self.errors_dir = os.path.join(self.log_dir, "errors")
        self.api_calls_dir = os.path.join(self.log_dir, "api_calls")
        self.costs_dir = os.path.join(self.log_dir, "costs")
        self.batches_dir = os.path.join(self.log_dir, "batches")
        self.debug_dir = os.path.join(self.log_dir, "debug", f"{job_name}_{self.session_id}" if job_name else f"general_{self.session_id}")

        for dir_path in [self.sessions_dir, self.errors_dir, self.api_calls_dir,
                         self.costs_dir, self.batches_dir, self.debug_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Session log file in sessions/ subdirectory
        filename = f"{job_name}_{self.session_id}.log" if job_name else f"general_{self.session_id}.log"
        self.log_file = os.path.join(self.sessions_dir, filename)

        # CSV file paths
        self.errors_csv = os.path.join(self.errors_dir, "errors.csv")
        self.api_calls_csv = os.path.join(self.api_calls_dir, "api_calls.csv")
        self.costs_csv = os.path.join(self.costs_dir, "costs.csv")

    def log(self, level, message, source_file=None, function_name=None, to_file=True):
        """Log a message with the specified level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the log message
        log_entry = f"{timestamp} - {level.value if isinstance(level, LogLevel) else level}"

        if source_file:
            log_entry += f" - {source_file}"
        if function_name:
            log_entry += f"::{function_name}"

        log_entry += f" - {message}"

        # Print to console
        print(log_entry)

        # Write to file if requested
        if to_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")

        # Auto-append ERROR and CRITICAL logs to errors.csv
        if isinstance(level, LogLevel) and level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._log_to_errors_csv(level, message, source_file, function_name)

    def log_data(self, filename, data, format='txt', subfolder=None):
        """Log data to a file - defaults to debug/{session}/ directory."""
        # Use debug directory by default if no subfolder specified
        if subfolder is None:
            output_dir = self.debug_dir
        elif subfolder == 'batches':
            output_dir = self.batches_dir
        else:
            # Allow custom subfolder within debug
            output_dir = os.path.join(self.debug_dir, subfolder)

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        try:
            if format == 'csv':
                # Write CSV data
                if isinstance(data, list) and len(data) > 0:
                    with open(filepath, 'w', newline='', encoding='utf-8') as f:
                        if isinstance(data[0], dict):
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                        else:
                            writer = csv.writer(f)
                            writer.writerows(data)
                    self.log(LogLevel.INFO, f"Data written to {filepath}")
            elif format == 'json':
                # Write JSON data
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                self.log(LogLevel.INFO, f"Data written to {filepath}")
            else:
                # Write text data
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(str(data))
                self.log(LogLevel.DEBUG, f"Data written to {filepath}")
        except Exception as e:
            self.log(LogLevel.ERROR, f"Error writing data to {filepath}: {e}")

    def log_chunk_stats(self, context_type, num_chunks, chunk_details):
        """Log statistics about chunking."""
        self.log(LogLevel.INFO, f"{context_type} chunking complete. {num_chunks} chunks created.")
        for detail in chunk_details:
            self.log(LogLevel.DEBUG, f"  Chunk {detail['id']}: {detail['tokens']} tokens")

    def log_api_request(self, messages, response_format=None):
        """Log API request for debugging."""
        self.log(LogLevel.INFO, "API request logged for debugging")
        request_data = {
            "messages": messages,
            "response_format": response_format
        }
        self.log_data("api_request.json", request_data, format='json', subfolder='api_debug')

    def log_api_response(self, completion, cost_info):
        """Log API response and cost information."""
        # Log cost information
        self.log_api_cost(cost_info)
        # Note: Full response is already logged separately via log_data

    def log_api_cost(self, cost_details):
        """Log API cost information."""
        # Handle unknown pricing gracefully
        if cost_details.get('total') == 'unknown':
            model = cost_details.get('model', 'unknown')
            input_tokens = cost_details.get('input_tokens', 0)
            output_tokens = cost_details.get('output_tokens', 0)
            cost_str = f"API Cost - Unknown (model: {model}, input_tokens: {input_tokens}, output_tokens: {output_tokens})"
        else:
            cost_str = f"API Cost - Input: ${cost_details['input']:.6f}, Output: ${cost_details['output']:.6f}, Total: ${cost_details['total']:.6f}"
        self.log(LogLevel.INFO, cost_str)

    def _append_to_csv(self, filepath, row_dict, headers):
        """Thread-safe append to CSV file."""
        try:
            # Check if file exists
            file_exists = os.path.exists(filepath)

            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)

                # Write header if file is new or empty
                if not file_exists or os.path.getsize(filepath) == 0:
                    writer.writeheader()

                writer.writerow(row_dict)

        except Exception as e:
            # Don't crash if logging fails, but print warning
            print(f"Warning: Could not write to {filepath}: {e}")

    def _log_to_errors_csv(self, level, message, source_file, function_name):
        """Append error to consolidated errors.csv."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        headers = ['timestamp', 'job_name', 'level', 'source_file', 'function_name', 'message']
        row = {
            'timestamp': timestamp,
            'job_name': self.job_name or 'general',
            'level': level.value,
            'source_file': source_file or '',
            'function_name': function_name or '',
            'message': message
        }

        self._append_to_csv(self.errors_csv, row, headers)

    def log_api_call_complete(self, request_data, response_data, cost_info, batch_id=None, status='success'):
        """Log complete API call with request and response to api_calls.csv."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        headers = ['timestamp', 'job_name', 'model', 'temperature', 'batch_id',
                   'request_json', 'response_json', 'input_tokens', 'output_tokens',
                   'input_cost', 'output_cost', 'total_cost', 'status']

        row = {
            'timestamp': timestamp,
            'job_name': self.job_name or 'general',
            'model': request_data.get('model', ''),
            'temperature': request_data.get('temperature', ''),
            'batch_id': batch_id or '',
            'request_json': json.dumps(request_data),
            'response_json': json.dumps(response_data) if response_data else '',
            'input_tokens': cost_info.get('input_tokens', 0),
            'output_tokens': cost_info.get('output_tokens', 0),
            'input_cost': cost_info.get('input', 'unknown'),
            'output_cost': cost_info.get('output', 'unknown'),
            'total_cost': cost_info.get('total', 'unknown'),
            'status': status
        }

        self._append_to_csv(self.api_calls_csv, row, headers)

        # Also log cost summary
        self._log_cost_summary(cost_info)

    def _log_cost_summary(self, cost_info):
        """Log cost summary to costs.csv."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        headers = ['timestamp', 'job_name', 'model', 'input_tokens', 'output_tokens',
                   'input_cost', 'output_cost', 'total_cost']

        row = {
            'timestamp': timestamp,
            'job_name': self.job_name or 'general',
            'model': cost_info.get('model', ''),
            'input_tokens': cost_info.get('input_tokens', 0),
            'output_tokens': cost_info.get('output_tokens', 0),
            'input_cost': cost_info.get('input', 'unknown'),
            'output_cost': cost_info.get('output', 'unknown'),
            'total_cost': cost_info.get('total', 'unknown')
        }

        self._append_to_csv(self.costs_csv, row, headers)

# Global logger instance
_logger_instance = None

def get_logger(job_name=None):
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None or job_name:
        _logger_instance = UnifiedLogger(job_name)
    return _logger_instance

# Backward compatibility function
def log_error(file_name, error_text):
    """
    Legacy error logging function for backward compatibility.

    Args:
        file_name: Source file name
        error_text: Error message text
    """
    logger = get_logger()
    logger.log(LogLevel.ERROR, error_text, source_file=file_name)