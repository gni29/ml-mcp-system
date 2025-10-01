#!/usr/bin/env python3
"""
Error Handler Module for ML MCP System
Provides robust error handling, recovery, and reporting
"""

import sys
import json
import traceback
import logging
from typing import Any, Dict, Optional, Callable
from functools import wraps
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MLMCPError(Exception):
    """Base exception for ML MCP System"""
    def __init__(self, message: str, error_code: str = "UNKNOWN", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DataLoadError(MLMCPError):
    """Error loading data"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "DATA_LOAD_ERROR", details)


class ValidationError(MLMCPError):
    """Data validation error"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class ProcessingError(MLMCPError):
    """Error during data processing"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "PROCESSING_ERROR", details)


class ModelError(MLMCPError):
    """ML model error"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "MODEL_ERROR", details)


class ConfigurationError(MLMCPError):
    """Configuration error"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "CONFIG_ERROR", details)


class ErrorHandler:
    """Handle and recover from errors"""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize error handler

        Args:
            log_file: Path to error log file
        """
        self.log_file = log_file
        self.error_count = 0
        self.warning_count = 0

        # Setup logging
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logging.basicConfig(
                filename=log_file,
                level=logging.ERROR,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle an error and return formatted error response

        Args:
            error: Exception that occurred
            context: Additional context information

        Returns:
            Error response dictionary
        """
        self.error_count += 1

        error_info = {
            'success': False,
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.now().isoformat()
        }

        # Add error code if available
        if isinstance(error, MLMCPError):
            error_info['error_code'] = error.error_code
            if error.details:
                error_info['details'] = error.details

        # Add context
        if context:
            error_info['context'] = context

        # Add traceback in debug mode
        if '--debug' in sys.argv or '--verbose' in sys.argv:
            error_info['traceback'] = traceback.format_exc()

        # Log error
        if self.log_file:
            logging.error(json.dumps(error_info, default=str))

        # Print to stderr
        print(f"ERROR: {error_info['error']}", file=sys.stderr)

        return error_info

    def log_warning(self, message: str, context: Optional[Dict] = None):
        """Log a warning"""
        self.warning_count += 1

        warning_info = {
            'level': 'warning',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

        if context:
            warning_info['context'] = context

        if self.log_file:
            logging.warning(json.dumps(warning_info, default=str))

        print(f"WARNING: {message}", file=sys.stderr)

    def get_recovery_suggestions(self, error: Exception) -> list:
        """
        Get recovery suggestions based on error type

        Args:
            error: Exception that occurred

        Returns:
            List of recovery suggestions
        """
        suggestions = []

        error_msg = str(error).lower()

        # File not found
        if 'file not found' in error_msg or 'no such file' in error_msg:
            suggestions.extend([
                "Check if the file path is correct",
                "Ensure the file exists at the specified location",
                "Verify you have read permissions for the file"
            ])

        # Memory errors
        elif 'memory' in error_msg or 'memoryerror' in type(error).__name__.lower():
            suggestions.extend([
                "Try processing the data in smaller chunks",
                "Reduce the dataset size or use sampling",
                "Close other applications to free up memory",
                "Use memory-optimized data types"
            ])

        # Data type errors
        elif 'dtype' in error_msg or 'type' in error_msg:
            suggestions.extend([
                "Check if columns have the expected data types",
                "Convert columns to appropriate types before processing",
                "Handle missing values that may cause type issues"
            ])

        # Missing columns
        elif 'column' in error_msg and 'not found' in error_msg:
            suggestions.extend([
                "Check if the column name is spelled correctly",
                "Use df.columns to see available column names",
                "Verify the data was loaded correctly"
            ])

        # Model errors
        elif 'model' in error_msg or isinstance(error, ModelError):
            suggestions.extend([
                "Ensure training data has the correct format",
                "Check if target variable is properly defined",
                "Verify all features are numeric (or properly encoded)",
                "Ensure sufficient training data is provided"
            ])

        # General suggestions
        suggestions.extend([
            "Check the error details for more specific information",
            "Try running with --debug flag for detailed output",
            "Consult documentation for parameter requirements"
        ])

        return suggestions[:5]  # Return top 5 suggestions


def safe_execute(func: Callable, *args, error_handler: Optional[ErrorHandler] = None,
                context: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
    """
    Safely execute a function with error handling

    Args:
        func: Function to execute
        *args: Function arguments
        error_handler: ErrorHandler instance
        context: Additional context
        **kwargs: Function keyword arguments

    Returns:
        Function result or error response
    """
    handler = error_handler or ErrorHandler()

    try:
        result = func(*args, **kwargs)
        return {'success': True, 'result': result}

    except Exception as e:
        error_response = handler.handle_error(e, context)
        error_response['recovery_suggestions'] = handler.get_recovery_suggestions(e)
        return error_response


def with_error_handling(context_name: Optional[str] = None):
    """
    Decorator for error handling

    Args:
        context_name: Name of the operation for context

    Example:
        @with_error_handling(context_name="data_loading")
        def load_data(file_path):
            return pd.read_csv(file_path)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            context = {'operation': context_name or func.__name__}

            try:
                return func(*args, **kwargs)

            except FileNotFoundError as e:
                raise DataLoadError(f"File not found: {str(e)}", {'file_path': args[0] if args else None})

            except pd.errors.EmptyDataError as e:
                raise DataLoadError("Empty data file", {'error': str(e)})

            except (ValueError, TypeError) as e:
                raise ValidationError(f"Validation error: {str(e)}", {'context': context})

            except MemoryError as e:
                raise ProcessingError("Out of memory", {
                    'suggestion': 'Try processing data in chunks or reduce dataset size'
                })

            except Exception as e:
                error_response = handler.handle_error(e, context)
                error_response['recovery_suggestions'] = handler.get_recovery_suggestions(e)

                # Return error response in JSON format
                print(json.dumps(error_response, ensure_ascii=False, indent=2, default=str))
                sys.exit(1)

        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay_seconds: int = 1):
    """
    Decorator to retry function on failure

    Args:
        max_retries: Maximum number of retry attempts
        delay_seconds: Delay between retries

    Example:
        @retry_on_failure(max_retries=3, delay_seconds=2)
        def flaky_operation():
            # Operation that might fail occasionally
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}", file=sys.stderr)

                    if attempt < max_retries - 1:
                        print(f"Retrying in {delay_seconds} seconds...", file=sys.stderr)
                        time.sleep(delay_seconds)
                    else:
                        print(f"All {max_retries} attempts failed", file=sys.stderr)

            # All retries failed, raise the last exception
            raise last_exception

        return wrapper
    return decorator


def graceful_degradation(fallback_value: Any = None):
    """
    Decorator for graceful degradation - return fallback on error

    Args:
        fallback_value: Value to return on error

    Example:
        @graceful_degradation(fallback_value=[])
        def get_advanced_stats(data):
            # Complex computation that might fail
            return compute_stats(data)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Warning: {func.__name__} failed, using fallback: {str(e)}", file=sys.stderr)
                return fallback_value

        return wrapper
    return decorator


class ErrorRecovery:
    """Automatic error recovery strategies"""

    @staticmethod
    def recover_from_missing_values(df, strategy: str = 'drop'):
        """
        Recover from missing value errors

        Args:
            df: DataFrame with missing values
            strategy: 'drop', 'mean', 'median', 'mode', 'forward_fill'

        Returns:
            Cleaned DataFrame
        """
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        elif strategy == 'forward_fill':
            return df.fillna(method='ffill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def recover_from_dtype_error(df, column: str, target_dtype: str = 'numeric'):
        """
        Recover from data type errors

        Args:
            df: DataFrame
            column: Column with type error
            target_dtype: Target type ('numeric', 'string', 'datetime')

        Returns:
            DataFrame with converted column
        """
        import pandas as pd

        df = df.copy()

        if target_dtype == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif target_dtype == 'string':
            df[column] = df[column].astype(str)
        elif target_dtype == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')

        return df

    @staticmethod
    def recover_from_memory_error(file_path: str, chunk_processor: Callable):
        """
        Recover from memory errors by processing in chunks

        Args:
            file_path: Path to large file
            chunk_processor: Function to process each chunk

        Returns:
            Combined results
        """
        results = []
        chunk_size = 10000

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            result = chunk_processor(chunk)
            results.append(result)

        # Combine results
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        return results


def main():
    """CLI interface for error handler testing"""
    print(json.dumps({
        'module': 'error_handler',
        'status': 'operational',
        'features': [
            'Custom exception types',
            'Error handling and recovery',
            'Retry mechanisms',
            'Graceful degradation',
            'Recovery suggestions'
        ]
    }, indent=2))


if __name__ == "__main__":
    main()