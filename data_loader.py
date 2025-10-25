import os
import pandas as pd
from unified_logger import log_error  # Backward compatible import
import sys

# Define the required config files
required_config_files = [
    'GPA_Job_Configuration.csv',
    'GPA_Questions.csv',
    'API_Keys.csv',
    'API_Pricing.csv'
]

def list_files(directory):
    """
    Helper function to list all files in a directory, ignoring hidden files.
    """
    if not os.path.exists(directory):
        error_message = f"Directory {directory} does not exist."
        log_error("data_loader.py", error_message)
        sys.exit(1)  # Exit the script if a folder does not exist
    
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]

def load_data():
    dataframes_dict = {}

    # Check if Configuration_Files directory exists
    config_dir = 'Configuration_Files'
    config_files = list_files(config_dir)
    
    # Check for required config files
    for required_file in required_config_files:
        if required_file not in config_files:
            error_message = f"Required configuration file {required_file} is missing."
            log_error("data_loader.py", error_message)
            sys.exit(1)  # Exit the script if a required file is missing
    
    # Load configuration files
    for file_name in config_files:
        # Try multiple encodings to handle different file formats
        encodings = ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
        loaded = False
        for encoding in encodings:
            try:
                df = pd.read_csv(f'{config_dir}/{file_name}', encoding=encoding)
                df['source_file'] = file_name  # Add the file name to the dataframe
                key = file_name.replace('.csv', '')
                dataframes_dict[key] = df
                loaded = True
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue  # Try next encoding
            except Exception as e:
                error_message = f"Error loading configuration file {file_name}: {e}"
                log_error("data_loader.py", error_message)
                break

        if not loaded:
            error_message = f"Could not load configuration file {file_name} with any supported encoding"
            log_error("data_loader.py", error_message)

    # Check if Record_Context directory exists and is not empty
    record_context_dir = 'Context/Record_Context'
    record_context_files = list_files(record_context_dir)
    
    if not record_context_files:
        error_message = f"No data found in the {record_context_dir} directory."
        log_error("data_loader.py", error_message)
        sys.exit(1)  # Exit the script if there is no data in the record context folder
    
    # Load Record Context files
    for idx, file_name in enumerate(record_context_files):
        try:
            if file_name.endswith('.xlsx'):
                df = pd.read_excel(f'{record_context_dir}/{file_name}')
            elif file_name.endswith('.csv'):
                df = pd.read_csv(f'{record_context_dir}/{file_name}', encoding='utf-8-sig')
            else:
                error_message = f"Unsupported file format for {file_name}. Only CSV and Excel files are supported."
                log_error("data_loader.py", error_message)
                continue
            df['source_file'] = file_name  # Add the file name to the dataframe
            key = f'Record_Context_{idx}'
            dataframes_dict[key] = df
        except Exception as e:
            error_message = f"Error loading record context file {file_name}: {e}"
            log_error("data_loader.py", error_message)

    # Check if Question_Context directory exists
    question_context_dir = 'Context/Question_Context'
    if os.path.exists(question_context_dir):
        question_context_files = [f for f in os.listdir(question_context_dir)
                                 if os.path.isfile(os.path.join(question_context_dir, f))
                                 and not f.startswith('.')]

        # Load Question Context files if they exist
        for idx, file_name in enumerate(question_context_files):
            if file_name.endswith('.xlsx'):
                try:
                    df = pd.read_excel(f'{question_context_dir}/{file_name}')
                    # Skip empty DataFrames
                    if df.empty:
                        error_message = f"Question context file {file_name} is empty, skipping."
                        log_error("data_loader.py", error_message)
                        continue

                    df['source_file'] = file_name  # Add the file name to the dataframe
                    key = f'Question_Context_{idx}'
                    dataframes_dict[key] = df
                except Exception as e:
                    error_message = f"Error loading question context file {file_name}: {e}"
                    log_error("data_loader.py", error_message)
            elif file_name.endswith('.csv'):
                # Try multiple encodings for CSV files
                encodings = ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
                loaded = False
                for encoding in encodings:
                    try:
                        df = pd.read_csv(f'{question_context_dir}/{file_name}', encoding=encoding)
                        # Skip empty DataFrames
                        if df.empty:
                            error_message = f"Question context file {file_name} is empty, skipping."
                            log_error("data_loader.py", error_message)
                            loaded = True  # Mark as loaded even if empty, to avoid trying other encodings
                            break

                        df['source_file'] = file_name  # Add the file name to the dataframe
                        key = f'Question_Context_{idx}'
                        dataframes_dict[key] = df
                        loaded = True
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue  # Try next encoding
                    except Exception as e:
                        error_message = f"Error loading question context file {file_name}: {e}"
                        log_error("data_loader.py", error_message)
                        break

                if not loaded:
                    error_message = f"Could not load question context file {file_name} with any supported encoding"
                    log_error("data_loader.py", error_message)
            else:
                error_message = f"Unsupported file format for {file_name}. Only CSV and Excel files are supported."
                log_error("data_loader.py", error_message)
    else:
        error_message = f"Question_Context directory does not exist. Application will proceed without question context."
        log_error("data_loader.py", error_message)

    return dataframes_dict
