"""
Backward compatibility wrapper for error_logger.
This module now redirects all calls to the unified logger.
"""

from unified_logger import log_error as unified_log_error

def log_error(file_name, error_text):
    """
    Helper function to log errors - now using unified logger.
    Maintained for backward compatibility.

    Args:
        file_name: Source file name
        error_text: Error message text
    """
    # Call the unified logger's backward compatible function
    unified_log_error(file_name, error_text)